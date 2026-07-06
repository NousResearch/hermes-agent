"""Micro-tests for karinai/runtime/session_bridge.py registration semantics.

Importing ``gateway.session_context`` must register the KarinAI request-scoped
vars (product run id + app-tool gateway creds) into its ``_VAR_MAP`` with the
gateway's own ``_UNSET`` sentinel, preserving the exact semantics the vars had
when they lived inline in that file:

- ``get_session_env`` round-trip after ``bind_karinai_run_context``;
- ``reset_session_vars`` restores ``_UNSET`` so the ``os.environ`` fallback
  works again (CLI/cron/dev compat);
- ``clear_session_vars`` pins the vars to ``""`` ("explicitly cleared") with
  NO env fallback — the thread-reuse leak safety that keeps a stale
  product_run_id / gateway token from leaking into the next run executed on a
  reused ThreadPoolExecutor thread.
"""

from __future__ import annotations

import pytest

from gateway.session_context import (
    clear_session_vars,
    get_session_env,
    reset_session_vars,
)
from karinai.runtime.session_bridge import KARINAI_SESSION_VARS, bind_karinai_run_context

VAR_NAMES = (
    "HERMES_PRODUCT_RUN_ID",
    "KARINAI_APP_TOOL_GATEWAY_URL",
    "KARINAI_APP_TOOL_GATEWAY_TOKEN",
    "KARINAI_APP_TOOL_GATEWAY_EXPIRES_AT",
)


@pytest.fixture(autouse=True)
def _restore_unset():
    """Never leak bound/cleared state into later tests in this process."""
    yield
    reset_session_vars()


def test_import_registers_all_karinai_vars() -> None:
    from gateway.session_context import _VAR_MAP

    assert set(VAR_NAMES) == set(KARINAI_SESSION_VARS)
    for name in VAR_NAMES:
        # The SAME ContextVar object in both registries — session_context reads
        # (get_session_env / reset) and bridge writes (bind) must share state.
        assert _VAR_MAP[name] is KARINAI_SESSION_VARS[name]


def test_bind_round_trips_through_get_session_env() -> None:
    tokens = bind_karinai_run_context(
        "run_bridge",
        {"url": "http://gw.internal", "token": "kat_tok", "expires_at": "2999-01-01T00:00:00Z"},
    )
    assert len(tokens) == 4  # reset tokens, concatenated onto set_session_vars()'s list
    assert get_session_env("HERMES_PRODUCT_RUN_ID") == "run_bridge"
    assert get_session_env("KARINAI_APP_TOOL_GATEWAY_URL") == "http://gw.internal"
    assert get_session_env("KARINAI_APP_TOOL_GATEWAY_TOKEN") == "kat_tok"
    assert get_session_env("KARINAI_APP_TOOL_GATEWAY_EXPIRES_AT") == "2999-01-01T00:00:00Z"


def test_bind_defaults_and_non_dict_gateway_bind_empty() -> None:
    bind_karinai_run_context("", app_tool_gateway="not-a-dict")  # type: ignore[arg-type]
    for name in VAR_NAMES:
        assert get_session_env(name) == ""


def test_reset_restores_unset_so_environ_fallback_works(monkeypatch: pytest.MonkeyPatch) -> None:
    bind_karinai_run_context("run_stale")
    reset_session_vars()
    # _UNSET restored => get_session_env falls back to os.environ again.
    monkeypatch.setenv("HERMES_PRODUCT_RUN_ID", "run_from_env")
    assert get_session_env("HERMES_PRODUCT_RUN_ID") == "run_from_env"


def test_clear_pins_empty_with_no_environ_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in VAR_NAMES:
        monkeypatch.setenv(name, "stale_env_value")
    tokens = bind_karinai_run_context("run_live", {"url": "u", "token": "t"})
    clear_session_vars(tokens)
    for name in VAR_NAMES:
        # "explicitly cleared": empty string wins over the stale os.environ value.
        assert get_session_env(name) == ""


def test_set_session_vars_masks_karinai_vars_for_every_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression (bridge-refactor review finding): ANY host's set_session_vars
    must pin the KarinAI vars to "" — a Telegram/cron/TUI turn in a process
    whose environment contains e.g. KARINAI_APP_TOOL_GATEWAY_TOKEN must NOT
    fall back to those env credentials (app-tool advertising gates on this)."""
    from gateway.session_context import set_session_vars

    for name in VAR_NAMES:
        monkeypatch.setenv(name, "leaked_env_value")
    tokens = set_session_vars(platform="telegram", chat_id="c1", async_delivery=True)
    try:
        for name in VAR_NAMES:
            assert get_session_env(name) == ""
    finally:
        clear_session_vars(tokens)
        reset_session_vars()


def test_api_server_rebind_overrides_the_mask() -> None:
    from gateway.session_context import set_session_vars

    tokens = set_session_vars(platform="api_server", async_delivery=False)
    tokens += bind_karinai_run_context("run_real", {"url": "u", "token": "t"})
    try:
        assert get_session_env("HERMES_PRODUCT_RUN_ID") == "run_real"
        assert get_session_env("KARINAI_APP_TOOL_GATEWAY_TOKEN") == "t"
    finally:
        clear_session_vars(tokens)
        reset_session_vars()

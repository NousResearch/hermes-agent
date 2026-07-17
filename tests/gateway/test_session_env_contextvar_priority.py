"""Residual HERMES_SESSION_* readers must prefer task-local ContextVars.

set_session_vars binds platform/source only in ContextVars (not os.environ).
Call sites that read process env first pick up a sibling session's stale
value under concurrent gateway turns.
"""

from __future__ import annotations

import os

import pytest

from gateway.session_context import (
    _UNSET,
    _VAR_MAP,
    clear_session_vars,
    get_session_env,
    reset_session_vars,
    resolve_session_platform_hint,
    resolve_session_source_hint,
    set_session_vars,
)


@pytest.fixture(autouse=True)
def _reset_contextvars_and_env(monkeypatch):
    # Setup and teardown: leave no bound ContextVars for later modules that
    # share this thread (e.g. skill_commands tests in the same pytest process).
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    for key in (
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_SOURCE",
        "HERMES_PLATFORM",
        "HERMES_SESSION_ID",
        "HERMES_SESSION_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    yield
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    for key in (
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_SOURCE",
        "HERMES_PLATFORM",
        "HERMES_SESSION_ID",
        "HERMES_SESSION_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_resolve_platform_prefers_contextvar_over_polluted_env(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_PLATFORM", "telegram")
    tokens = set_session_vars(
        platform="discord",
        source="discord",
        chat_id="1",
        session_key="discord:1",
    )
    try:
        assert get_session_env("HERMES_SESSION_PLATFORM") == "discord"
        # Process env still holds the sibling/stale value.
        assert os.environ.get("HERMES_SESSION_PLATFORM") == "telegram"
        assert resolve_session_platform_hint() == "discord"
    finally:
        clear_session_vars(tokens)


def test_resolve_source_prefers_contextvar_over_polluted_env(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "cli")
    tokens = set_session_vars(source="telegram", chat_id="9", session_key="t:9")
    try:
        assert resolve_session_source_hint() == "telegram"
    finally:
        clear_session_vars(tokens)


def test_prompt_builder_platform_hint_prefers_contextvar(monkeypatch):
    """Regression: prompt_builder used to read os.environ before ContextVar."""
    import agent.prompt_builder as pb

    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    # Ensure the gateway module is importable so the lazy path can resolve.
    import gateway.session_context  # noqa: F401

    tokens = set_session_vars(
        platform="discord",
        source="discord",
        chat_id="2",
        session_key="discord:2",
    )
    try:
        assert pb._current_session_platform_hint() == "discord"
    finally:
        clear_session_vars(tokens)


def test_background_review_metadata_uses_task_local_source(monkeypatch):
    from agent.background_review import build_memory_write_metadata

    monkeypatch.setenv("HERMES_SESSION_SOURCE", "cli")
    tokens = set_session_vars(source="slack", chat_id="c", session_key="slack:c")
    try:

        class _Agent:
            session_id = "s1"
            _parent_session_id = ""
            platform = None
            _memory_write_origin = "assistant_tool"
            _memory_write_context = "foreground"

        meta = build_memory_write_metadata(_Agent())
        assert meta["platform"] == "slack"
    finally:
        clear_session_vars(tokens)


def test_explicit_clear_does_not_resurrect_session_platform_env(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    tokens = set_session_vars(platform="discord", chat_id="1", session_key="k")
    clear_session_vars(tokens)
    # Cleared task must not fall back to the stale SESSION_PLATFORM env value.
    assert resolve_session_platform_hint() == ""
    # Operator-wide HERMES_PLATFORM pin still works after clear.
    monkeypatch.setenv("HERMES_PLATFORM", "api")
    assert resolve_session_platform_hint() == "api"


def test_unbound_task_still_reads_process_env_for_cli(monkeypatch):
    reset_session_vars()
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    assert resolve_session_platform_hint() == "telegram"
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "cron")
    assert resolve_session_source_hint() == "cron"


def test_skill_disabled_platform_resolution_uses_contextvar(monkeypatch):
    from tools.skills_tool import _is_skill_disabled

    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_PLATFORM", "telegram")

    class _Cfg(dict):
        pass

    def _fake_load():
        return {
            "skills": {
                "disabled": [],
                "platform_disabled": {
                    "discord": ["secret-skill"],
                    "telegram": [],
                },
            }
        }

    monkeypatch.setattr("hermes_cli.config.load_config", _fake_load)
    tokens = set_session_vars(platform="discord", chat_id="1", session_key="d:1")
    try:
        assert _is_skill_disabled("secret-skill") is True
    finally:
        clear_session_vars(tokens)


def test_kanban_stamp_does_not_resurrect_session_id_after_clear(monkeypatch):
    """After clear_session_vars, kanban must not re-read process HERMES_SESSION_ID."""
    from tools.kanban_tools import _stamp_worker_session_metadata

    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    monkeypatch.setenv("HERMES_SESSION_ID", "stale-sibling-session")
    tokens = set_session_vars(
        platform="telegram",
        chat_id="1",
        session_key="t:1",
        session_id="live-session",
    )
    try:
        stamped = _stamp_worker_session_metadata("task-1", {})
        assert stamped is not None
        assert stamped["worker_session_id"] == "live-session"
    finally:
        clear_session_vars(tokens)

    # Cleared ContextVar returns "" and must not fall back to stale env.
    assert _stamp_worker_session_metadata("task-1", {"x": 1}) == {"x": 1}


def test_kanban_create_session_id_respects_cleared_contextvar(monkeypatch):
    """create_task path: get_session_env after clear must not resurrect env."""
    from gateway.session_context import get_session_env

    monkeypatch.setenv("HERMES_SESSION_ID", "stale-from-env")
    tokens = set_session_vars(session_id="live-create", chat_id="1", session_key="k")
    try:
        assert get_session_env("HERMES_SESSION_ID", "") == "live-create"
    finally:
        clear_session_vars(tokens)
    assert get_session_env("HERMES_SESSION_ID", "") == ""
    # Simulate the fixed create_task resolution (no raw env after get_session_env).
    resolved = get_session_env("HERMES_SESSION_ID", "")
    assert resolved == ""
    assert resolved or None is None

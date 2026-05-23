"""Regression test for #31026 — cron profile switching must not leak Feishu
pairing state across profiles.

The cron ticker runs as a background thread in the gateway process. When a
job has ``profile: <other_profile>`` set, ``_job_profile_context`` mutates
``os.environ`` (via ``load_dotenv(..., override=True)``) so the cron job's
agent sees that profile's API keys and config.

Before the fix, the gateway's auth path read those env vars directly via
``os.getenv`` per-message, so users authorized in the source profile got
rejected mid-cron and the unauthorized-DM-pair flow wrote pending pairing
codes to ``feishu-pending.json`` for them.

After the fix, ``_is_user_authorized`` and ``_get_unauthorized_dm_behavior``
read through ``gateway_getenv`` which consults the pre-context baseline
snapshot published by ``_job_profile_context``, so the gateway's verdict
stays pinned to profile A's config regardless of cron-thread mutations.
"""

from __future__ import annotations

import os
import types
from pathlib import Path

import pytest

from gateway.pairing import PairingStore
from gateway.session import Platform, SessionSource
from hermes_constants import (
    clear_gateway_baseline_env,
    gateway_getenv,
    set_gateway_baseline_env,
)


@pytest.fixture(autouse=True)
def _isolate_baseline():
    clear_gateway_baseline_env()
    yield
    clear_gateway_baseline_env()


def _build_test_runner(pairing_dir: Path) -> types.SimpleNamespace:
    """Construct the minimum runner-shaped object the auth path needs.

    GatewayRunner has heavy dependencies for full construction; the auth
    methods only touch ``self.pairing_store`` and (via getattr) ``self.config``.
    We bind the unbound methods onto a SimpleNamespace for direct invocation.
    """
    from gateway.run import GatewayRunner

    # Module-level constant captured at import time — repoint at the test dir.
    import gateway.pairing as pairing_mod
    pairing_mod.PAIRING_DIR = pairing_dir
    pairing_dir.mkdir(parents=True, exist_ok=True)

    runner = types.SimpleNamespace()
    runner.config = None
    runner.pairing_store = PairingStore()
    runner._is_user_authorized = GatewayRunner._is_user_authorized.__get__(runner)
    runner._get_unauthorized_dm_behavior = GatewayRunner._get_unauthorized_dm_behavior.__get__(runner)
    return runner


def test_user_stays_authorized_during_cron_profile_context(tmp_path, monkeypatch):
    """Profile A's user remains authorized even while os.environ is mutated."""
    runner = _build_test_runner(tmp_path / "pairing")

    # Profile A's gateway baseline: alice is authorized.
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "alice")
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)

    # Snapshot the pre-context env (mirrors what _job_profile_context does).
    set_gateway_baseline_env(dict(os.environ))

    # Now simulate the cron job's load_dotenv(override=True) for profile B:
    # B has NO feishu allowlist — alice is not in profile B's allowed users.
    os.environ.pop("FEISHU_ALLOWED_USERS", None)

    # Sanity: live env now lacks alice's entry.
    assert os.environ.get("FEISHU_ALLOWED_USERS") is None
    # The gateway baseline still authorizes alice.
    assert gateway_getenv("FEISHU_ALLOWED_USERS") == "alice"

    alice = SessionSource(
        platform=Platform.FEISHU,
        chat_id="dm-alice",
        chat_type="dm",
        user_id="alice",
        user_name="Alice",
    )

    assert runner._is_user_authorized(alice) is True, (
        "Profile A's user must remain authorized while cron has mutated env "
        "for profile B (#31026)"
    )


def test_unauthorized_dm_behavior_stays_ignore_during_cron_profile_context(
    tmp_path, monkeypatch
):
    """`_get_unauthorized_dm_behavior` must keep returning 'ignore' for profile
    A (which has an allowlist) even while cron has cleared the env for profile
    B. Before the fix this returned 'pair' mid-cron and triggered the
    feishu-pending.json write."""
    runner = _build_test_runner(tmp_path / "pairing")

    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "alice")
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)

    set_gateway_baseline_env(dict(os.environ))

    # Cron mutates env to profile B's (empty allowlist).
    os.environ.pop("FEISHU_ALLOWED_USERS", None)

    assert runner._get_unauthorized_dm_behavior(Platform.FEISHU) == "ignore", (
        "Gateway must stay in 'ignore' mode for profile A; otherwise the "
        "unauthorized-DM-pair flow runs and contaminates feishu-pending.json "
        "(#31026)"
    )


def test_after_baseline_cleared_gateway_reads_live_env(tmp_path, monkeypatch):
    """Sanity: when the cron context exits and the baseline is cleared, the
    gateway resumes reading the live env."""
    runner = _build_test_runner(tmp_path / "pairing")

    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "alice")
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
    set_gateway_baseline_env(dict(os.environ))

    # Cron mutates...
    os.environ.pop("FEISHU_ALLOWED_USERS", None)
    # ...and cron exits: baseline clears, env restored (we simulate restore here).
    clear_gateway_baseline_env()
    os.environ["FEISHU_ALLOWED_USERS"] = "alice"

    alice = SessionSource(
        platform=Platform.FEISHU,
        chat_id="dm-alice",
        chat_type="dm",
        user_id="alice",
        user_name="Alice",
    )
    assert runner._is_user_authorized(alice) is True


def test_no_pending_pairing_entry_written_during_profile_cron(tmp_path, monkeypatch):
    """End-to-end regression: simulate the exact scenario from the issue.

    Profile A authorizes alice. While a cron profile context for B is active
    (env mutated), a message from alice is processed. With the fix the
    authorization succeeds and the unauthorized-DM-pair flow does NOT run, so
    ``feishu-pending.json`` is not created.
    """
    pairing_dir = tmp_path / "pairing"
    runner = _build_test_runner(pairing_dir)

    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "alice")
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)

    # Enter the cron profile context.
    set_gateway_baseline_env(dict(os.environ))
    os.environ.pop("FEISHU_ALLOWED_USERS", None)

    alice = SessionSource(
        platform=Platform.FEISHU,
        chat_id="dm-alice",
        chat_type="dm",
        user_id="alice",
        user_name="Alice",
    )

    # Authorization succeeds → caller never enters the pair-code path that
    # would write feishu-pending.json (see gateway/run.py:6586).
    assert runner._is_user_authorized(alice) is True

    pending_file = pairing_dir / "feishu-pending.json"
    assert not pending_file.exists(), (
        f"feishu-pending.json was created at {pending_file} — the cron "
        "profile-context leak still leaks (#31026)"
    )

    clear_gateway_baseline_env()

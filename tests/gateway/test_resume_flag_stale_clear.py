"""Regression tests for stale ``resume_pending`` maintenance."""

import ast
import inspect
import textwrap
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.run import GatewayRunner, _clear_stale_resume_pending_flags
from gateway.session import AsyncSessionStore, SessionEntry, SessionStore
from hermes_cli.config import DEFAULT_CONFIG


def _store(tmp_path) -> SessionStore:
    config = GatewayConfig(default_reset_policy=SessionResetPolicy(mode="none"))
    with patch("hermes_state.SessionDB", side_effect=RuntimeError("JSON-only test")):
        store = SessionStore(sessions_dir=tmp_path, config=config)
    store._loaded = True
    return store


def _resume_entry(
    key: str,
    *,
    updated_hours_ago: float,
    marked_hours_ago: float,
    suspended: bool = False,
) -> SessionEntry:
    now = datetime.now()
    return SessionEntry(
        session_key=key,
        session_id=f"sid_{key}",
        created_at=now - timedelta(days=30),
        updated_at=now - timedelta(hours=updated_hours_ago),
        platform=Platform.DISCORD,
        chat_type="thread",
        suspended=suspended,
        resume_pending=True,
        resume_reason="shutdown_timeout",
        resume_kind="self",
        resume_handoff="continue",
        resume_request_id=f"request_{key}",
        last_resume_marked_at=now - timedelta(hours=marked_hours_ago),
    )


def test_resume_flag_stale_clear_defaults_on() -> None:
    assert DEFAULT_CONFIG["agent"]["resume_flag_stale_clear"] is True


def test_hourly_watcher_invokes_stale_flag_maintenance() -> None:
    source = textwrap.dedent(inspect.getsource(GatewayRunner._session_expiry_watcher))
    tree = ast.parse(source)
    call_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_clear_stale_resume_pending_flags" in call_names


@pytest.mark.asyncio
async def test_maintenance_clears_only_stale_unsuspended_flags_and_persists(
    tmp_path, monkeypatch, caplog
) -> None:
    monkeypatch.setenv("HERMES_RESUME_FLAG_STALE_CLEAR", "true")
    monkeypatch.setenv("HERMES_AUTO_CONTINUE_FRESHNESS", "3600")
    store = _store(tmp_path)
    store._entries = {
        "stale": _resume_entry(
            "stale", updated_hours_ago=30, marked_hours_ago=25
        ),
        # Recent activity wins over an old mark.
        "fresh": _resume_entry(
            "fresh", updated_hours_ago=0.5, marked_hours_ago=25
        ),
        "suspended": _resume_entry(
            "suspended",
            updated_hours_ago=30,
            marked_hours_ago=30,
            suspended=True,
        ),
    }
    store._save()

    with caplog.at_level("INFO", logger="gateway.session"):
        cleared = await _clear_stale_resume_pending_flags(AsyncSessionStore(store))

    assert cleared == 1
    stale = store._entries["stale"]
    assert stale.resume_pending is False
    assert stale.resume_reason is None
    assert stale.resume_kind is None
    assert stale.resume_handoff is None
    assert stale.resume_request_id is None
    assert stale.last_resume_marked_at is None
    assert store._entries["fresh"].resume_pending is True
    assert store._entries["suspended"].resume_pending is True
    assert "stale" in caplog.text
    assert "age_seconds=" in caplog.text

    # A fresh SessionStore must observe the clear. This catches the prior
    # in-memory/disk clobber class where shutdown resurrected the flag.
    reloaded = _store(tmp_path)
    reloaded._loaded = False
    reloaded._ensure_loaded()
    assert reloaded._entries["stale"].resume_pending is False
    assert reloaded._entries["fresh"].resume_pending is True
    assert reloaded._entries["suspended"].resume_pending is True


@pytest.mark.asyncio
async def test_ttl_is_six_freshness_windows_when_that_exceeds_one_day(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HERMES_RESUME_FLAG_STALE_CLEAR", "true")
    monkeypatch.setenv("HERMES_AUTO_CONTINUE_FRESHNESS", str(5 * 60 * 60))
    store = _store(tmp_path)
    store._entries = {
        "under-30h": _resume_entry(
            "under-30h", updated_hours_ago=29, marked_hours_ago=29
        ),
        "over-30h": _resume_entry(
            "over-30h", updated_hours_ago=31, marked_hours_ago=31
        ),
    }

    assert await _clear_stale_resume_pending_flags(AsyncSessionStore(store)) == 1
    assert store._entries["under-30h"].resume_pending is True
    assert store._entries["over-30h"].resume_pending is False


@pytest.mark.asyncio
async def test_zero_freshness_still_uses_one_day_floor(tmp_path, monkeypatch) -> None:
    """The cleanup has its own off-switch; freshness=0 still uses the TTL floor."""
    monkeypatch.setenv("HERMES_RESUME_FLAG_STALE_CLEAR", "true")
    monkeypatch.setenv("HERMES_AUTO_CONTINUE_FRESHNESS", "0")
    store = _store(tmp_path)
    store._entries = {
        "under-24h": _resume_entry(
            "under-24h", updated_hours_ago=23, marked_hours_ago=23
        ),
        "over-24h": _resume_entry(
            "over-24h", updated_hours_ago=25, marked_hours_ago=25
        ),
    }

    assert await _clear_stale_resume_pending_flags(AsyncSessionStore(store)) == 1
    assert store._entries["under-24h"].resume_pending is True
    assert store._entries["over-24h"].resume_pending is False


@pytest.mark.asyncio
async def test_config_off_switch_leaves_stale_flag_untouched(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_RESUME_FLAG_STALE_CLEAR", "false")
    monkeypatch.setenv("HERMES_AUTO_CONTINUE_FRESHNESS", "3600")
    store = _store(tmp_path)
    store._entries["stale"] = _resume_entry(
        "stale", updated_hours_ago=100, marked_hours_ago=100
    )

    assert await _clear_stale_resume_pending_flags(AsyncSessionStore(store)) == 0
    assert store._entries["stale"].resume_pending is True

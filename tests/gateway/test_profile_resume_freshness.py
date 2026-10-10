"""A served profile owns its restart recovery freshness policy."""

import asyncio
from datetime import datetime, timedelta

import pytest

from agent.secret_scope import reset_multiplex_context, set_multiplex_context
from gateway.run import (
    _auto_continue_freshness_window,
    _is_fresh_gateway_interruption,
    _profile_runtime_scope,
)
from gateway.session import SessionStore
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def _profile_homes(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_AUTO_CONTINUE_FRESHNESS", "3600")
    homes = {}
    for name, window in (("short", 60), ("long", 7200)):
        home = root / "profiles" / name
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            f"agent:\n  gateway_auto_continue_freshness: {window}\n", encoding="utf-8"
        )
        homes[name] = home
    return root, homes


def test_scoped_resume_policy_returns_to_owning_profile(tmp_path, monkeypatch):
    _, homes = _profile_homes(tmp_path, monkeypatch)
    token = set_multiplex_context(True)
    try:
        marker = datetime.now() - timedelta(minutes=30)
        decisions = []
        for name in ("short", "long", "short"):
            with _profile_runtime_scope(homes[name], {}):
                decisions.append(_is_fresh_gateway_interruption(
                    marker, window_secs=_auto_continue_freshness_window()
                ))
        assert decisions == [False, True, False]
        # Scoped omissions/invalid values use the default, not a longer launch-profile bridge;
        # an explicit zero keeps the documented opt-out of the freshness gate.
        monkeypatch.setenv("HERMES_AUTO_CONTINUE_FRESHNESS", "7200")
        marker = datetime.now() - timedelta(minutes=90)
        for name, config, expected in (
            ("missing", "agent: {}\n", False),
            ("invalid", "agent:\n  gateway_auto_continue_freshness: invalid\n", False),
            ("disabled", "agent:\n  gateway_auto_continue_freshness: 0\n", True),
        ):
            home = homes["short"].parent / name
            home.mkdir()
            (home / "config.yaml").write_text(config, encoding="utf-8")
            with _profile_runtime_scope(home, {}):
                assert _is_fresh_gateway_interruption(
                    marker, window_secs=_auto_continue_freshness_window()
                ) is expected
    finally:
        reset_multiplex_context(token)
    assert _auto_continue_freshness_window() == 7200


@pytest.mark.asyncio
async def test_startup_resume_uses_each_persisted_profile_policy(tmp_path, monkeypatch):
    root, homes = _profile_homes(tmp_path, monkeypatch)
    runner, adapter = make_restart_runner()
    runner.config.multiplex_profiles = True
    runner._primary_profile_name = "default"
    runner._profile_adapters = {name: runner.adapters for name in homes}
    runner.session_store = SessionStore(root / "sessions", runner.config)
    token = set_multiplex_context(True)
    try:
        marker = datetime.now() - timedelta(minutes=30)
        entries = {}
        for name, home in homes.items():
            source = make_restart_source(chat_id=f"{name}-chat")
            source.profile = name
            with _profile_runtime_scope(home, {}):
                entry = runner.session_store.get_or_create_session(source)
                runner.session_store.mark_resume_pending(entry.session_key)
                runner.session_store._update_entry(
                    entry.session_key, lambda current: setattr(current, "last_resume_marked_at", marker)
                )
            entries[name] = entry
        scheduled = runner._schedule_resume_pending_sessions()
        await asyncio.gather(*runner._background_tasks)
        assert scheduled == 1
        adapter._message_handler.assert_awaited_once()
        event = adapter._message_handler.await_args.args[0]
        assert event.source.profile == "long"
        assert entries["short"].resume_pending
        assert entries["short"].session_key not in runner._running_agents
        assert runner.session_store.load_transcript(entries["short"].session_id) == []
    finally:
        reset_multiplex_context(token)

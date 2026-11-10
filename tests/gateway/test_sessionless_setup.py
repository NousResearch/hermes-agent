"""Sessionless setup creates no sessions or turns and retires legacy metadata without losing history."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.onboarding_flow import Facts
from tests.waiting import async_wait_until

FACTS = Facts(os="macos", arch="arm64", machine_kind="Mac", plugins=())


class _Transport:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame)
        return True

    def close(self):
        pass


async def _desktop(tmp_path, monkeypatch, turns, *, configured=True):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from tui_gateway import server

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_hermes_home", str(home))
    monkeypatch.setattr(server, "_db", None)
    if configured:
        (home / ".env").write_text("OPENROUTER_API_KEY=sk-or-test\n", encoding="utf-8")
        (home / "config.yaml").write_text("model:\n  provider: openrouter\n  default: anthropic/claude-sonnet-4\n",
                                          encoding="utf-8")

    async def answer(event):
        turns.append(event)
        return "ok"

    store = SessionStore(home / "sessions", GatewayConfig())
    runner = SimpleNamespace(_session_db=store._db, session_store=store, _draining=False, adapters={},
                             _handle_message=answer, _evict_cached_agent=lambda route: None,
                             _cached_agent_for=lambda route: None, _peek_session_state=lambda key: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id="owner")
    transport = _Transport()
    identity = {"user_id": "uid:1", "provider": "local", "profile_id": str(home), "instance_id": "owner",
                "native_bootstrap": True}
    transport.desktop_local = True
    connection = AuthorityConnection(authority, transport, identity, operator=True)
    from gateway.session_local_migration import bind_native_transport
    bind_native_transport(authority, connection.actor, identity)
    return SimpleNamespace(authority=authority, connection=connection, transport=transport, db=store._db)


async def _request(desktop, rid, method, params):
    from tui_gateway.ws import _dispatch_request
    params = {"sessionless": True, **params}
    reply = await _dispatch_request(desktop.connection, {"jsonrpc": "2.0", "id": rid, "method": method,
                                                         "params": params}, method, desktop.transport)
    if reply is None:
        def written():
            return next((f for f in desktop.transport.frames if isinstance(f, dict) and f.get("id") == rid), None)
        await async_wait_until(written, timeout=20)
        reply = written()
    return reply


@pytest.mark.asyncio
async def test_sessionless_progress_survives_completion_and_stale_window_without_inference(tmp_path, monkeypatch):
    from hermes_cli import onboarding_run
    turns = []
    desktop = await _desktop(tmp_path, monkeypatch, turns, configured=False)
    try:
        def no_inference(**kwargs):
            raise AssertionError("Setup must never require inference or provision access")
        monkeypatch.setattr("hermes_cli.welcome_trial.inference_available", no_inference)
        monkeypatch.setattr("hermes_cli.welcome_trial.claim_desktop_free_tier", no_inference)
        fresh = await _request(desktop, 0, "onboarding.state", {"assume_fresh": True})
        assert fresh["result"] == {"run": True, "eligible": True, "step": "preferences", "started": False}
        started = await _request(desktop, 1, "onboarding.set_run", {"run": True, "step": "preferences", "restart": True})
        assert started["result"]["eligible"] is True
        await _request(desktop, 2, "onboarding.set_run", {"run": True, "step": "model"})
        assert onboarding_run.guide_progress() == {"step": "model", "started": True}
        assert (await _request(desktop, 3, "onboarding.state", {}))["result"]["step"] == "model"
        await _request(desktop, 4, "onboarding.set_run", {"run": False, "mark_profile_offered": True})
        late = await _request(desktop, 5, "onboarding.set_run", {"run": True, "step": "preferences"})
        assert late["result"]["run"] is False
        assert desktop.db.search_sessions(limit=100) == []
        assert turns == []
    finally:
        await desktop.connection.close()


@pytest.mark.asyncio
async def test_retirement_preserves_history_and_late_old_tool_cannot_finish_new_guide(tmp_path, monkeypatch):
    from agent import onboarding_record
    from agent.onboarding_flow import End
    from hermes_cli import onboarding_run
    from tools.setup_beat_tool import setup_beat_tool
    turns = []
    desktop = await _desktop(tmp_path, monkeypatch, turns)
    try:
        desktop.db.create_session("legacy-setup", "desktop")
        desktop.db.append_message("legacy-setup", "user", "Keep this conversation")
        onboarding_record.start(desktop.db, "legacy-setup", FACTS)
        messages = desktop.db.get_messages("legacy-setup")
        # A classic read leaves its live conversation alone; only redesigned state opts into retirement.
        await _request(desktop, 0, "onboarding.state", {"sessionless": False})
        assert onboarding_record.open_session_ids(desktop.db) == ["legacy-setup"]
        await _request(desktop, -1, "onboarding.state", {})
        await _request(desktop, 1, "onboarding.set_run", {"run": True, "step": "model"})
        assert onboarding_record.open_session_ids(desktop.db) == []
        assert desktop.db.get_messages("legacy-setup") == messages
        onboarding_record.advance(desktop.db, "legacy-setup", End("skipped"))
        end_turn = Mock()
        result = setup_beat_tool({"beat": "next"}, db=desktop.db, session_id="legacy-setup",
                                 callback=lambda card: pytest.fail("Retired setup must not show a card"),
                                 end_turn=end_turn)
        assert json.loads(result)["error"]
        end_turn.assert_called_once_with()
        assert onboarding_run.should_run() is True
        assert onboarding_run.guide_progress()["step"] == "model"
        assert len(desktop.db.search_sessions(limit=100)) == 1
        assert turns == []
    finally:
        await desktop.connection.close()

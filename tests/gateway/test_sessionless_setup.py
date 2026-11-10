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


async def _request(desktop, rid, method, params, *, sessionless=True):
    from tui_gateway.ws import _dispatch_request
    if sessionless is not None:
        params = {"sessionless": sessionless, **params}
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
        started = await _request(desktop, 1, "onboarding.set_run", {"run": True})
        assert started["result"] == {"run": True, "eligible": True, "step": "preferences", "started": True}
        await _request(desktop, 2, "onboarding.set_run", {"run": True, "step": "model"})
        assert onboarding_run.guide_progress() == {"step": "model", "started": True}
        assert (await _request(desktop, 3, "onboarding.state", {}))["result"]["step"] == "model"
        # A window that read the original fresh snapshot starts after a peer reached Model.
        resumed = await _request(desktop, 7, "onboarding.set_run", {"run": True})
        assert resumed["result"]["step"] == "model"
        await _request(desktop, 4, "onboarding.set_run", {"run": False, "mark_profile_offered": True})
        late = await _request(desktop, 5, "onboarding.set_run", {"run": True, "step": "preferences"})
        assert late["result"]["run"] is False
        late_start = await _request(desktop, 8, "onboarding.set_run", {"run": True})
        assert late_start["result"]["run"] is False
        restarted = await _request(
            desktop, 9, "onboarding.set_run", {"run": True, "step": "preferences", "restart": True},
        )
        assert restarted["result"]["run"] is True
        assert restarted["result"]["step"] == "preferences"
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


@pytest.mark.asyncio
@pytest.mark.parametrize("sessionless", [None, False, True], ids=["classic-omitted", "classic", "sessionless"])
async def test_setup_dispatch_requires_operator_and_preserves_root_scope(tmp_path, monkeypatch, sessionless):
    from agent import onboarding_record
    from gateway.runtime_bootstrap import TicketStore
    from gateway.session_authorities import SessionAuthorities
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from hermes_cli.config import read_user_config_raw
    from hermes_state import SessionDB
    from tui_gateway.ws_legacy_fallback import legacy_fallback_allowed

    desktop = await _desktop(tmp_path, monkeypatch, [])
    home = tmp_path / ".hermes"
    root_config = home / "config.yaml"
    root_config.write_text(
        root_config.read_text(encoding="utf-8")
        + "onboarding:\n  run: true\n  step: preferences\n",
        encoding="utf-8",
    )
    work = home / "profiles" / "work"
    work.mkdir(parents=True)
    work_config = work / "config.yaml"
    work_config.write_text("onboarding:\n  run: false\n  step: model\n", encoding="utf-8")
    runner = desktop.authority.runner
    runner.config = SimpleNamespace(multiplex_profiles=True)
    work_db = SessionDB(db_path=work / "state.db")
    secondary = await initialize_session_authority(
        runner, profile_id=str(work), instance_id="owner", db=work_db, register=False,
    )
    runner.session_authorities = SessionAuthorities(home, multiplexed=True)
    runner.session_authorities.add(home, desktop.authority, name="default")
    runner.session_authorities.add(work, secondary, name="work")
    for db, sid in ((desktop.db, "root-setup"), (work_db, "work-setup")):
        db.create_session(sid, "desktop")
        onboarding_record.start(db, sid, FACTS)
    root_before, work_before = root_config.read_bytes(), work_config.read_bytes()

    tickets = TicketStore("owner", frozenset({str(work)}))
    ticket = tickets.mint(profile_id=str(work), subject="uid:1", purpose="worker-adoption")
    grant = tickets.redeem(ticket, profile_id=str(work), purpose="worker-adoption")
    identity = {"user_id": grant["subject"], "provider": "local",
                "profile_id": grant["profile_id"], "instance_id": grant["instance_id"]}
    worker = AuthorityConnection(
        secondary, _Transport(), {**identity, "capabilities": grant["capabilities"]}, operator=False,
    )
    operator = AuthorityConnection(
        secondary, desktop.transport, {**identity, "profile_scope": "host"}, operator=True,
    )
    restricted = SimpleNamespace(connection=worker, transport=worker.transport)
    authorized = SimpleNamespace(connection=operator, transport=operator.transport)
    try:
        assert worker.actor.capabilities == frozenset({"worker:adopt"})
        for rid, method, params in (
            (1, "onboarding.state", {}),
            (2, "onboarding.set_run", {"run": False, "mark_profile_offered": True}),
        ):
            assert legacy_fallback_allowed(worker.actor, method) is False
            denied = await _request(restricted, rid, method, params, sessionless=sessionless)
            assert denied["error"]["data"]["reason"] == "permission_denied", denied
            assert "result" not in denied
        assert root_config.read_bytes() == root_before
        assert work_config.read_bytes() == work_before
        assert onboarding_record.open_session_ids(desktop.db) == ["root-setup"]
        assert onboarding_record.open_session_ids(work_db) == ["work-setup"]

        foreign = await _request(
            restricted, 3, "onboarding.set_run", {"profile": "default", "run": False},
            sessionless=sessionless,
        )
        assert foreign["error"]["data"]["reason"] == "profile_mismatch", foreign
        assert root_config.read_bytes() == root_before

        state = await _request(
            authorized, 4, "onboarding.state", {"profile": "work"}, sessionless=sessionless,
        )
        assert state["result"]["run"] is True, state
        changed = await _request(
            authorized, 5, "onboarding.set_run",
            {"profile": "work", "run": False, "mark_profile_offered": True, "step": "model"},
            sessionless=sessionless,
        )
        assert changed["result"]["run"] is False, changed
        persisted = read_user_config_raw(root_config)
        assert persisted["onboarding"] == {
            "run": False,
            "step": "model" if sessionless else "preferences",
            "seen": {"profile_build_offered": True},
        }
        assert persisted["model"]["provider"] == "openrouter"
        assert work_config.read_bytes() == work_before
        state = await _request(
            authorized, 6, "onboarding.state", {"profile": "work"}, sessionless=sessionless,
        )
        assert state["result"]["run"] is False, state
        assert onboarding_record.open_session_ids(desktop.db) == ([] if sessionless else ["root-setup"])
        assert onboarding_record.open_session_ids(work_db) == ["work-setup"]
    finally:
        await worker.close()
        await operator.close()
        await desktop.connection.close()
        work_db.close()

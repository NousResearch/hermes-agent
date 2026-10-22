"""The desktop's onboarding chat on the unified gateway: the window resumes the chat that
``onboarding.start`` opened.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

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


async def _desktop(tmp_path, monkeypatch, turns):
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
    monkeypatch.setattr("agent.onboarding_facts.read_facts", lambda **kw: FACTS)
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
    identity = {"user_id": "uid:1", "provider": "local", "profile_id": str(home), "instance_id": "owner"}
    connection = AuthorityConnection(authority, transport, identity)
    from gateway.session_local_migration import bind_native_transport
    bind_native_transport(authority, connection.actor, identity)
    return SimpleNamespace(authority=authority, connection=connection, transport=transport, db=store._db)


async def _request(desktop, rid, method, params):
    from tui_gateway.ws import _dispatch_request
    reply = await _dispatch_request(desktop.connection, {"jsonrpc": "2.0", "id": rid, "method": method,
                                                         "params": params}, method, desktop.transport)
    if reply is None:
        def written():
            return next((f for f in desktop.transport.frames if isinstance(f, dict) and f.get("id") == rid), None)
        await async_wait_until(written, timeout=20)
        reply = written()
    return reply


@pytest.mark.asyncio
@pytest.mark.parametrize("start", [{"rerun": True}, {"fresh_launch": True}], ids=["rerun", "launch"])
async def test_the_window_resumes_the_chat_onboarding_start_opened(tmp_path, monkeypatch, start):
    turns = []
    desktop = await _desktop(tmp_path, monkeypatch, turns)
    try:
        started = await _request(desktop, 1, "onboarding.start", {"facts_source": "local", "locale": "en", **start})
        assert "result" in started, started
        stored = started["result"]["stored_session_id"]

        resumed = await _request(desktop, 2, "session.resume", {"session_id": stored})

        assert "error" not in resumed, resumed
        assert resumed["result"]["session_id"] == stored
        assert stored.startswith("local-") and stored in desktop.authority.sessions
        await async_wait_until(lambda: turns, timeout=10)
        assert "first-run onboarding" in turns[0].text and "setup_beat" in turns[0].text
    finally:
        await desktop.connection.close()


@pytest.mark.asyncio
async def test_a_rerun_replaces_the_open_run_and_skip_ends_the_new_one(tmp_path, monkeypatch):
    from agent import onboarding_record
    from gateway.session_server_requests import ask_viewer_result, waiting
    turns = []
    desktop = await _desktop(tmp_path, monkeypatch, turns)
    try:
        first = (await _request(desktop, 1, "onboarding.start", {"rerun": True}))["result"]["stored_session_id"]
        second = (await _request(desktop, 2, "onboarding.start", {"rerun": True}))["result"]["stored_session_id"]
        assert second != first
        assert onboarding_record.load(desktop.db, first)["phase"] != "onboarding"
        assert onboarding_record.load(desktop.db, second)["phase"] == "onboarding"
        again = (await _request(desktop, 3, "onboarding.start", {}))["result"]
        assert again["stored_session_id"] == second and again["resumed"] is True

        live = desktop.authority.sessions[second]
        answer = asyncio.get_running_loop().run_in_executor(None, lambda: ask_viewer_result(
            live.event_stream, second, "setup_beat", {"beat": "name"}, timeout=10))
        await async_wait_until(lambda: waiting(live.event_stream, second, "setup_beat"), timeout=5)
        skipped = await _request(desktop, 4, "onboarding.skip", {"session_id": second})
        assert skipped["result"] == {"skipped": True}, skipped
        assert await answer is None
        assert onboarding_record.load(desktop.db, second)["phase"] == "skipped"
    finally:
        await desktop.connection.close()


@pytest.mark.asyncio
async def test_a_rerun_with_no_model_opens_no_chat(tmp_path, monkeypatch):
    """Nothing can run the chat: error 5080 and no row, never a session whose launch policy cannot be
    restored (every later resume of it would answer storage_unavailable)."""
    desktop = await _desktop(tmp_path, monkeypatch, [])
    (tmp_path / ".hermes" / "config.yaml").write_text("{}\n", encoding="utf-8")
    (tmp_path / ".hermes" / ".env").write_text("", encoding="utf-8")
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda config=None: None)
    try:
        before = {row["id"] for row in desktop.db.search_sessions(limit=100)}
        refused = await _request(desktop, 1, "onboarding.start", {"rerun": True})
        assert refused["error"]["code"] == 5080, refused
        assert {row["id"] for row in desktop.db.search_sessions(limit=100)} == before
    finally:
        await desktop.connection.close()


def test_an_onboarding_chat_folds_in_setup_beat_for_its_whole_life(tmp_path):
    """The authority resolves toolsets every turn; the fold-in keys on the record existing, so a run
    that ends never changes the tool array mid-conversation (prompt cache)."""
    from agent import onboarding_record
    from agent.onboarding_flow import End
    from agent.onboarding_session import onboarding_chat_toolsets
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("onb", source="local")
        db.create_session("plain", source="local")
        onboarding_record.start(db, "onb", FACTS)
        assert onboarding_chat_toolsets(["web"], db, "onb") == ["web", "onboarding"]
        assert onboarding_chat_toolsets(["web"], db, "plain") == ["web"]
        assert onboarding_chat_toolsets([], db, "onb") == []
        onboarding_record.advance(db, "onb", End("skipped"))
        assert onboarding_chat_toolsets(["web"], db, "onb") == ["web", "onboarding"]
    finally:
        db.close()


def test_the_authority_turn_plays_the_opening_and_skips_the_note_while_a_card_waits(tmp_path):
    from agent import onboarding_record
    from agent.onboarding_text import GREETING
    from gateway.session_onboarding import arm_onboarding_turn
    from gateway.session_server_requests import _Pending, _lock, _open
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("onb", source="local")
        onboarding_record.start(db, "onb", FACTS)
        stream = object()
        owner = (SimpleNamespace(sessions={"onb": SimpleNamespace(event_stream=stream)}), "onb", 1)
        agent = SimpleNamespace(_session_db=db, session_id="onb", valid_tool_names={"setup_beat"},
                                _gateway_turn_context_notes="")
        assert list(arm_onboarding_turn(agent, owner, [])) == [(GREETING, "setup_beat", {"beat": "name"})]
        assert arm_onboarding_turn(SimpleNamespace(valid_tool_names={"terminal"}), owner, []) is None
        assert arm_onboarding_turn(agent, None, []) is None
        with _lock:
            _open["srq-test"] = _Pending("onb", events=stream, method="setup_beat")
        try:
            arm_onboarding_turn(agent, owner, [{"role": "user", "content": "hi"}])
            assert agent._gateway_turn_context_notes == ""
        finally:
            with _lock:
                _open.pop("srq-test", None)
    finally:
        db.close()


async def _settled(desktop, session_id):
    """The opening turn's lease released: a delete is not refused as a write in flight."""
    def idle():
        with desktop.db._read_ctx() as conn:
            return not desktop.db._guarded_ids(conn, [session_id])
    await async_wait_until(idle, timeout=10)


@pytest.mark.asyncio
async def test_start_from_a_secondary_profile_claims_in_the_default_home(tmp_path, monkeypatch):
    """Multiplexed host socket on a secondary profile: onboarding runs in ``default``, so the trial claim,
    the provider check and the chat's policy read the default home, never the asking connection's."""
    from gateway.session_authorities import SessionAuthorities
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    desktop = await _desktop(tmp_path, monkeypatch, [])
    home = tmp_path / ".hermes"
    work = home / "profiles" / "work"
    work.mkdir(parents=True)
    (work / "config.yaml").write_text("{}\n", encoding="utf-8")  # nothing to run on in the asking profile
    claims = []

    def claim(*, assume_fresh):
        claims.append(Path(get_hermes_home()))
    monkeypatch.setattr("hermes_cli.welcome_trial.claim_desktop_free_tier", claim)
    runner = desktop.authority.runner
    runner.config = SimpleNamespace(multiplex_profiles=True)
    work_db = SessionDB(db_path=work / "state.db")
    secondary = await initialize_session_authority(runner, profile_id=str(work), instance_id="owner", db=work_db,
                                                   register=False)
    runner.session_authorities = SessionAuthorities(home, multiplexed=True)
    runner.session_authorities.add(home, desktop.authority, name="default")
    runner.session_authorities.add(work, secondary, name="work")
    connection = AuthorityConnection(secondary, desktop.transport, {
        "user_id": "uid:1", "provider": "local", "profile_id": str(work), "instance_id": "owner",
        "profile_scope": "host"})
    try:
        started = await connection.dispatch({"jsonrpc": "2.0", "id": 1, "method": "onboarding.start", "params": {}})
        assert "result" in started, started
        assert claims == [home]
        assert started["result"]["stored_session_id"] in desktop.authority.sessions
    finally:
        await connection.close()
        await desktop.connection.close()
        work_db.close()


@pytest.mark.asyncio
async def test_skip_accepts_the_profile_the_desktop_names(tmp_path, monkeypatch):
    desktop = await _desktop(tmp_path, monkeypatch, [])
    try:
        stored = (await _request(desktop, 1, "onboarding.start", {"rerun": True}))["result"]["stored_session_id"]
        skipped = await _request(desktop, 2, "onboarding.skip", {"session_id": stored, "profile": "default"})
        assert skipped.get("result") == {"skipped": True}, skipped
    finally:
        await desktop.connection.close()


@pytest.mark.asyncio
async def test_deleting_the_onboarding_chat_from_the_sidebar_ends_the_run(tmp_path, monkeypatch):
    from agent import onboarding_record
    from hermes_cli.onboarding_run import _root_config
    from hermes_cli.config import read_user_config_raw
    turns, ended = [], []
    desktop = await _desktop(tmp_path, monkeypatch, turns)
    onboarding_record.on_ended(ended.append)
    try:
        stored = (await _request(desktop, 1, "onboarding.start", {"rerun": True}))["result"]["stored_session_id"]
        await async_wait_until(lambda: turns, timeout=10)
        await _settled(desktop, stored)
        row = desktop.db.get_session(stored)
        deleted = await _request(desktop, 2, "session.mutate", {
            "session_id": stored, "request_id": "delete-onboarding", "expected_revision": row["runtime_revision"],
            "expected_generation": row["runtime_generation"], "operation": "delete", "payload": {}})
        assert "result" in deleted, deleted
        assert [e["outcome"] for e in ended if e["stored_session_id"] == stored] == ["deleted"]
        assert read_user_config_raw(_root_config())["onboarding"]["run"] is False
    finally:
        onboarding_record._ended_listeners.remove(ended.append)
        await desktop.connection.close()


@pytest.mark.asyncio
async def test_a_fresh_launch_deletes_the_chat_the_last_launch_left_with_no_viewer(tmp_path, monkeypatch):
    """The previous launch's chat is still mounted (a backend that outlived the app, or local recovery),
    but no window watches it and no turn runs in it: the new launch's start deletes it and drops it
    from the authority."""
    from gateway.session_controls import AuthorityConnection
    turns = []
    desktop = await _desktop(tmp_path, monkeypatch, turns)
    first = (await _request(desktop, 1, "onboarding.start", {"rerun": True}))["result"]["stored_session_id"]
    await async_wait_until(lambda: turns, timeout=10)
    await _settled(desktop, first)
    await desktop.connection.close()  # the app quits
    assert first in desktop.authority.sessions
    identity = {"user_id": "uid:1", "provider": "local", "profile_id": desktop.authority.profile_id,
                "instance_id": "owner"}
    desktop.connection = AuthorityConnection(desktop.authority, desktop.transport, identity)
    try:
        second = (await _request(desktop, 2, "onboarding.start", {"fresh_launch": True}))["result"]
        assert second["stored_session_id"] != first
        assert desktop.db.get_session(first) is None
        assert first not in desktop.authority.sessions
    finally:
        await desktop.connection.close()

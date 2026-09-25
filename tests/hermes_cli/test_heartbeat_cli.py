"""``hermes heartbeat`` CLI ↔ gateway restore contract.

A heartbeat written from the shell must be the SAME persisted row the gateway's
``restore_heartbeat_watches`` scan (startup + every poller tick) registers — and a CLI ``clear``
must make the poller drop the watch. Real imports, real SessionDB/SessionStore in a temp
HERMES_HOME, no mocks on the persistence path.
"""
import asyncio
import io
from argparse import Namespace
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run import GatewayRunner
from gateway.run_heartbeat_restore import restore_heartbeat_watches
from gateway.session import SessionStore, SessionSource
from hermes_cli import goals
from hermes_cli.heartbeat import HeartbeatManager
from hermes_cli.heartbeat_cmd import heartbeat_command
from hermes_state import SessionDB


def _run(**kw):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = heartbeat_command(Namespace(**kw))
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = SessionDB(db_path=home / "state.db")
    monkeypatch.setattr(goals, "_DB_CACHE", {str(home): db})
    yield home
    db.close()


class _FakeAdapter(BasePlatformAdapter):
    """Real adapter lifecycle with an in-memory transport: no bot, no model."""

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="wire-1")


def _runner(home, config, adapter=None):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = config
    runner.session_store = SessionStore(home / "sessions", config)
    runner._heartbeat_watch = {}
    runner._running_agents = {}
    runner._start_heartbeat_poller = lambda: None
    runner._profile_name_for_source = lambda source, adapter_profile=None: source.profile
    runner._delivery_adapter_for = lambda source: adapter if adapter is not None else object()
    runner._run_in_executor_with_context = asyncio.to_thread
    return runner


async def _tick(runner):
    """One body of ``_start_heartbeat_poller``'s loop, minus the sleep (gateway/run_goals.py)."""
    await restore_heartbeat_watches(runner)
    if runner._heartbeat_watch:
        await runner._heartbeat_poll_once(runner._heartbeat_watch)


def test_set_refuses_unknown_session_and_bad_interval(home):
    code, _out, err = _run(heartbeat_command="set", session_id="nope", every="30m", prompt="x", prompt_file=None)
    assert code == 1 and "no session 'nope'" in err
    db = goals._get_session_db()
    db.create_session("s1", "cli")
    code, _out, err = _run(heartbeat_command="set", session_id="s1", every="5s", prompt="x", prompt_file=None)
    assert code == 1 and "at least 60s" in err
    code, _out, err = _run(heartbeat_command="set", session_id="s1", every="soon", prompt="x", prompt_file=None)
    assert code == 1 and "not an interval" in err
    assert HeartbeatManager("s1").state is None, "a refused set must not persist anything"


def test_set_uses_existing_persistence_and_reports_cli_scope(home, tmp_path):
    db = goals._get_session_db()
    db.create_session("s1", "cli")
    prompt_file = tmp_path / "p.txt"
    prompt_file.write_text("from file\n")
    code, out, _err = _run(heartbeat_command="set", session_id="s1", every="every 90 minutes",
                           prompt=None, prompt_file=str(prompt_file))
    assert code == 0
    state = HeartbeatManager("s1").state
    assert (state.prompt, state.interval_seconds, state.status) == ("from file", 5400, "active")
    assert db.get_meta("heartbeat:s1") is not None, "must land in the same state_meta row the gateway reads"
    assert "no gateway routing key" in out  # honest: the gateway will not fire a CLI-only session

    code, out, _err = _run(heartbeat_command="list", all=False, json=False)
    assert code == 0 and "s1" in out and "90m" in out and "from file" in out
    code, _out, err = _run(heartbeat_command="status", session_id="missing", json=False)
    assert code == 1 and "No heartbeat" in err


@pytest.mark.asyncio
async def test_cli_set_is_registered_by_gateway_restore_and_clear_drops_it(home):
    config = GatewayConfig()
    store = SessionStore(home / "sessions", config)
    runner = None
    try:
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="777", user_id="42", chat_type="dm")
        entry = store.get_or_create_session(source)
        store.close_all_db_handles()
        session_id = entry.session_id
        # The routing index does not create the ``sessions`` row; the first agent turn does. Mirror that.
        db = goals._get_session_db()
        db.create_session(session_id, "telegram", user_id="42", chat_id="777", chat_type="dm")
        db.record_gateway_session_peer(session_id, source="telegram", user_id="42", session_key=entry.session_key,
                                       chat_id="777", chat_type="dm")

        # Gateway "process": nothing registered yet (no /heartbeat ever typed in chat).
        runner = _runner(home, config)
        await restore_heartbeat_watches(runner)
        assert runner._heartbeat_watch == {}

        # Operator sets it from the shell while the gateway is up.
        code, out, _err = _run(heartbeat_command="set", session_id=session_id, every="30m",
                               prompt="Check the board", prompt_file=None)
        assert code == 0 and "no restart needed" in out
        assert "chat=777" in out and "telegram" in out

        # Next poller tick == restore scan: the watch is registered under the routing key
        # with the persisted origin (platform/chat/thread), same as a chat /heartbeat would.
        await restore_heartbeat_watches(runner)
        assert list(runner._heartbeat_watch) == [entry.session_key]
        src, sid = runner._heartbeat_watch[entry.session_key]
        assert (sid, src.platform, src.chat_id) == (session_id, Platform.TELEGRAM, "777")

        # pause -> restore no longer counts it active (existing watch is left alone: the poller
        # drops it itself via has_heartbeat/due_prompt); clear -> has_heartbeat False.
        assert _run(heartbeat_command="pause", session_id=session_id)[0] == 0
        assert not HeartbeatManager(session_id).is_active()
        assert _run(heartbeat_command="resume", session_id=session_id)[0] == 0
        assert HeartbeatManager(session_id).is_active()
        assert _run(heartbeat_command="clear", session_id=session_id)[0] == 0
        assert not HeartbeatManager(session_id).has_heartbeat()
        runner._heartbeat_watch.clear()
        await restore_heartbeat_watches(runner)
        assert runner._heartbeat_watch == {}, "a cleared row must not be resurrected by the scan"

        code, out, _err = _run(heartbeat_command="list", all=True, json=True)
        assert code == 0 and '"status": "cleared"' in out
    finally:
        store.close_all_db_handles()
        if runner is not None:
            runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_cli_set_then_poller_tick_enqueues_message_event_when_due(home, monkeypatch):
    """CLI ``set`` -> poller tick -> watch registered -> a MessageEvent reaches the adapter when due."""
    from hermes_cli import heartbeat as hb_mod

    clock = SimpleNamespace(now=1_000_000.0)
    monkeypatch.setattr(hb_mod, "time", SimpleNamespace(time=lambda: clock.now))
    config = GatewayConfig()
    store = SessionStore(home / "sessions", config)
    adapter = _FakeAdapter(PlatformConfig(enabled=True, typing_indicator=False), Platform.TELEGRAM)
    received = []

    async def handler(event):
        event._heartbeat_execution_started = True
        received.append(event)
        return None

    adapter.set_message_handler(handler)
    runner = _runner(home, config, adapter=adapter)
    try:
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="777", user_id="42", chat_type="dm")
        entry = store.get_or_create_session(source)
        store.close_all_db_handles()
        session_id = entry.session_id
        db = goals._get_session_db()
        db.create_session(session_id, "telegram", user_id="42", chat_id="777", chat_type="dm")
        db.record_gateway_session_peer(session_id, source="telegram", user_id="42", session_key=entry.session_key,
                                       chat_id="777", chat_type="dm")

        code, out, _err = _run(heartbeat_command="set", session_id=session_id, every="30m",
                               prompt="Check the board", prompt_file=None)
        assert code == 0 and "no restart needed" in out

        await _tick(runner)  # tick 1: registered, not yet due
        assert entry.session_key in runner._heartbeat_watch
        assert received == []

        clock.now += 30 * 60
        await _tick(runner)  # tick 2: due -> synthetic MessageEvent handed to the adapter
        await asyncio.gather(*adapter._background_tasks)
        assert len(received) == 1
        ev = received[0]
        assert "Check the board" in ev.text  # rendered via HeartbeatState.render_prompt (header + instruction)
        assert ev.source.platform == Platform.TELEGRAM and ev.source.chat_id == "777"
        assert ev.metadata["gateway_session_key"] == entry.session_key
        assert HeartbeatManager(session_id).state.fire_count == 1
    finally:
        store.close_all_db_handles()
        runner.session_store.close_all_db_handles()

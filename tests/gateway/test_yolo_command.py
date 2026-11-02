"""Tests for gateway /yolo session scoping."""

import os

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from tools.approval import disable_session_yolo, is_session_yolo_enabled


@pytest.fixture(autouse=True)
def _clean_yolo_state(monkeypatch):
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    disable_session_yolo("agent:main:telegram:dm:chat-a")
    disable_session_yolo("agent:main:telegram:dm:chat-b")
    yield
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    disable_session_yolo("agent:main:telegram:dm:chat-a")
    disable_session_yolo("agent:main:telegram:dm:chat-b")


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.session_store = None
    runner.config = None
    return runner


def _make_event(chat_id: str) -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id=f"user-{chat_id}",
        chat_id=chat_id,
        user_name="tester",
        chat_type="dm",
    )
    return MessageEvent(text="/yolo", source=source)


@pytest.mark.asyncio
async def test_yolo_command_toggles_only_current_session(monkeypatch):
    runner = _make_runner()

    event_a = _make_event("chat-a")
    session_a = runner._session_key_for_source(event_a.source)
    session_b = runner._session_key_for_source(_make_event("chat-b").source)

    await runner._handle_yolo_command(event_a)

    assert is_session_yolo_enabled(session_a) is True
    assert is_session_yolo_enabled(session_b) is False
    assert os.environ.get("HERMES_YOLO_MODE") is None

    await runner._handle_yolo_command(event_a)

    assert is_session_yolo_enabled(session_a) is False
    assert os.environ.get("HERMES_YOLO_MODE") is None


def test_launch_yolo_revocation_survives_the_next_turn(monkeypatch):
    """`hermes chat --yolo` seeds the bypass once; a later `/yolo` off is not re-enabled per turn."""
    from types import SimpleNamespace
    from gateway import session_policy
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    runner = _make_runner()
    source = SessionSource(platform=Platform.LOCAL, chat_id="launch-yolo", user_id="u", chat_type="dm")
    key = "agent:main:local:dm:launch-yolo"
    monkeypatch.setattr(session_policy, "policy_for_source",
                        lambda _runner, _source: SimpleNamespace(yolo=True, platform="cli", max_turns=5,
                                                                        skills_prompt=None))

    def _stop(**_kw):
        raise RuntimeError("stop after the yolo seam")
    runner._resolve_session_agent_runtime = _stop
    runner._pre_agent_fallback_notice = None
    runner._get_system_prompt_for_channel = lambda *a, **k: ""

    def turn():
        TurnRunner(runner, TurnContext(source=source, session_key=key))._run_sync_scoped()

    try:
        turn()
        assert is_session_yolo_enabled(key) is True
        disable_session_yolo(key)
        turn()
        assert is_session_yolo_enabled(key) is False
    finally:
        from tools.approval import clear_session
        clear_session(key)


@pytest.mark.asyncio
async def test_yolo_survives_gateway_restart_and_dies_at_session_boundary(tmp_path):
    """/yolo is persisted on the routing entry: a fresh process re-arms it on the next turn, and a
    conversation boundary (/new, /resume) clears both copies so a restart cannot revive it."""
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from tools.approval_yolo import restore_session_yolo

    def _runner():
        runner = _make_runner()
        runner.session_store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
        runner.session_store._db = None  # JSON routing index: the restart reads it back from disk
        return runner

    event = _make_event("chat-a")
    first = _runner()
    key = first._session_key_for_source(event.source)
    await first._handle_yolo_command(event)
    assert is_session_yolo_enabled(key) is True

    disable_session_yolo(key)  # a new process starts with an empty in-memory approval set
    second = _runner()
    restore_session_yolo(key, second.session_store.get_or_create_session(event.source).yolo)
    assert is_session_yolo_enabled(key) is True

    second._clear_session_boundary_security_state(key)
    assert is_session_yolo_enabled(key) is False
    third = _runner()
    restore_session_yolo(key, third.session_store.get_or_create_session(event.source).yolo)
    assert is_session_yolo_enabled(key) is False


@pytest.mark.asyncio
async def test_owner_yolo_off_is_not_revived_by_the_persisted_restore(tmp_path):
    """The owner's `config.set key=yolo` (TUI/Desktop/`hermes chat` /yolo) writes the route's persisted
    copy too: switching off a persisted /yolo must not be re-armed by the next turn's restore."""
    from types import SimpleNamespace
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_busy_controls import busy_config
    from tools.approval_yolo import restore_session_yolo

    runner = _make_runner()
    runner.session_store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    runner.session_store._db = None
    event = _make_event("chat-a")
    key = runner._session_key_for_source(event.source)
    await runner._handle_yolo_command(event)
    authority = SimpleNamespace(authorize=lambda *_a: None, profile_id=str(tmp_path), runner=runner,
                                sessions={"sid": SimpleNamespace(route=key, source=event.source)})
    connection, ref = SimpleNamespace(actor="a", authority=authority), SimpleNamespace(session_id="sid")

    off = await busy_config(connection, ref, {"session_id": "sid", "key": "yolo", "value": "0"}, write=True)
    assert off["value"] == "0" and is_session_yolo_enabled(key) is False
    restore_session_yolo(key, runner.session_store.get_or_create_session(event.source).yolo)
    assert is_session_yolo_enabled(key) is False

"""``/heartbeat`` in the gateway: profile scope and promote must not fall through to a usage error.

The poller owns firing; this covers the command surface that arms the instruction.
"""

import asyncio

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.heartbeat import PROFILE_SCOPE_KEY, HeartbeatManager, load_heartbeat


@pytest.fixture
def runner():
    """A GatewayRunner with just enough session plumbing for the heartbeat command handler."""
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    goals._get_session_db()
    instance = object.__new__(GatewayRunner)
    instance._heartbeat_watch = {}
    instance._register_heartbeat_watch = lambda *a, **k: None
    instance._unregister_heartbeat_watch = lambda *a, **k: None
    return instance


@pytest.fixture
def source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="42", chat_type="dm")


def _run(runner, source, args: str, session_id: str):
    """Drive the real command handler with a stubbed session lookup."""
    event = MessageEvent(
        text=f"/heartbeat {args}".strip(), message_type=MessageType.TEXT, source=source
    )

    async def _manager_for_event(_event):
        return HeartbeatManager(session_id=session_id), None

    runner._get_heartbeat_manager_for_event = _manager_for_event
    runner._session_key_for_source = lambda src: "telegram:42:42"
    return asyncio.run(runner._handle_heartbeat_command(event))


def test_profile_set_and_status_round_trip(runner, source):
    out = _run(runner, source, "profile every 10m Check the deploy", "hb-gw-sid")
    assert "10m" in out and "Check the deploy" in out

    stored = load_heartbeat(PROFILE_SCOPE_KEY)
    assert stored is not None and stored.interval_seconds == 600

    out = _run(runner, source, "profile status", "hb-gw-sid")
    assert "10m" in out and "Check the deploy" in out


def test_profile_status_without_one_is_not_an_error(runner, source):
    out = _run(runner, source, "profile status", "hb-gw-sid")
    assert "profile" in out.lower()


def test_profile_clear_disarms_future_sessions(runner, source):
    _run(runner, source, "profile every 10m Check the deploy", "hb-gw-sid")
    _run(runner, source, "profile clear", "hb-gw-sid")
    assert load_heartbeat(PROFILE_SCOPE_KEY) is None
    assert HeartbeatManager(session_id="hb-gw-fresh").has_heartbeat() is False


def test_promote_lifts_the_session_heartbeat(runner, source):
    HeartbeatManager(session_id="hb-gw-promote").set("watch CI", 900)
    out = _run(runner, source, "promote", "hb-gw-promote")
    assert "watch CI" in out
    assert load_heartbeat(PROFILE_SCOPE_KEY).prompt == "watch CI"


def test_promote_without_a_heartbeat_reports_plainly(runner, source):
    out = _run(runner, source, "promote", "hb-gw-empty")
    assert "promote" in out.lower()


def test_session_status_flags_an_inherited_profile_heartbeat(runner, source):
    _run(runner, source, "profile every 10m Check the deploy", "hb-gw-inherits")
    out = _run(runner, source, "status", "hb-gw-inherits")
    assert "profile" in out.lower(), "an inherited heartbeat must say so rather than claim to be the session's"

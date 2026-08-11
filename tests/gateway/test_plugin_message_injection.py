"""RED-first tests for exact-session gateway injection (H-105, H-106, H-113).

Pins the host-owned seam ``GatewayRunner.inject_plugin_message``:

- the gateway reuses the existing authorised route (live adapter) and never
  fabricates a synthetic platform route;
- gateway injection is disabled per plugin unless
  ``plugins.entries.<id>.allow_gateway_injection: true`` is set;
- idle target -> dispatched as a synthetic internal turn;
- busy target -> queued behind active work in the session FIFO;
- closed, rotated, unknown or unauthorised targets fail closed (False);
- internal events never reach command dispatch (H-107 inert-control).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from gateway.platforms.base import MessageEvent, SessionSource, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionEntry


class FakeAdapter:
    """Stand-in for a live platform adapter with a session route."""

    def __init__(self) -> None:
        self._pending_messages: dict = {}
        self._active_sessions: set[str] = set()


class FakeSessionStore:
    """Duck-typed SessionStore stand-in (host seam uses only these attrs)."""

    def __init__(self, entries: dict[str, SessionEntry]) -> None:
        self._entries = entries

    def _is_session_ended_in_db(self, session_id: str) -> bool:
        return False


class FakeConfig:
    def __init__(self) -> None:
        self.platforms = {}


def _entry(session_key: str, platform: Platform) -> SessionEntry:
    return SessionEntry(
        session_key=session_key,
        session_id=f"sess-{session_key}",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        origin=SessionSource(
            platform=platform,
            chat_id="chat-1",
            chat_name="Test",
            chat_type="dm",
            user_id="user-1",
            user_name="User",
        ),
        platform=platform,
    )


KEY = "telegram:dm:chat-1:user-1"


@pytest.fixture
def runner(monkeypatch):
    """GatewayRunner stand-in with the injection seam wired to fakes."""

    def _build(*, gate: bool, with_route: bool = True) -> GatewayRunner:
        adapter = FakeAdapter()
        r = GatewayRunner.__new__(GatewayRunner)
        r.session_store = FakeSessionStore({KEY: _entry(KEY, Platform.TELEGRAM)})
        r.adapters = {"telegram": adapter} if with_route else {}
        r.config = FakeConfig()
        r._sessions = {}
        r._dispatched: list[MessageEvent] = []

        def fake_resolve(platform, config, adapters):
            if platform == Platform.TELEGRAM and with_route:
                return type("T", (), {"adapter": adapter})()
            return None

        monkeypatch.setattr("gateway.run.resolve_delivery_transport", fake_resolve)
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"plugins": {"entries": {"hermes-peer": {"allow_gateway_injection": gate}}}},
        )

        async def _fake_handle_message(event):
            r._dispatched.append(event)

        r._handle_message = _fake_handle_message  # type: ignore[method-assign]
        return r

    return _build


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# H-105 — exact gateway target via the existing authorised route
# ---------------------------------------------------------------------------


class TestGatewayExactTarget:
    def test_idle_session_dispatches_internal_event(self, runner):
        r = runner(gate=True)
        ok = run(r.inject_plugin_message("hello peer", target_session=KEY, plugin_id="hermes-peer"))
        assert ok is True
        assert len(r._dispatched) == 1
        event = r._dispatched[0]
        assert event.text == "hello peer"
        assert event.internal is True
        assert event.source.platform == Platform.TELEGRAM

    def test_busy_session_queues_in_pending_slot(self, runner):
        r = runner(gate=True)
        adapter = r.adapters["telegram"]
        adapter._active_sessions.add(KEY)
        r._session_state(KEY).turn.agent = object()  # busy at runner level

        ok = run(r.inject_plugin_message("queued work", target_session=KEY, plugin_id="hermes-peer"))
        assert ok is True
        assert r._dispatched == []  # never dispatched while busy
        assert adapter._pending_messages[KEY].text == "queued work"

    def test_busy_session_overflow_uses_fifo(self, runner):
        r = runner(gate=True)
        adapter = r.adapters["telegram"]
        adapter._active_sessions.add(KEY)
        r._session_state(KEY).turn.agent = object()
        adapter._pending_messages[KEY] = MessageEvent(text="first", source=_entry(KEY, Platform.TELEGRAM).origin, internal=True)

        ok = run(r.inject_plugin_message("second", target_session=KEY, plugin_id="hermes-peer"))
        assert ok is True
        queued = r._session_state(KEY).conversation.queued_events
        assert [e.text for e in queued] == ["second"]

    def test_missing_route_fails_closed(self, runner):
        r = runner(gate=True, with_route=False)
        ok = run(r.inject_plugin_message("no route", target_session=KEY, plugin_id="hermes-peer"))
        assert ok is False
        assert r._dispatched == []


# ---------------------------------------------------------------------------
# H-113 — per-plugin gateway authorisation gate
# ---------------------------------------------------------------------------


class TestGatewayAuthorisation:
    def test_injection_disabled_without_explicit_gate(self, runner):
        r = runner(gate=False)
        ok = run(r.inject_plugin_message("blocked", target_session=KEY, plugin_id="hermes-peer"))
        assert ok is False
        assert r._dispatched == []


# ---------------------------------------------------------------------------
# H-106 — closed/unknown targets fail closed
# ---------------------------------------------------------------------------


class TestGatewayClosedTarget:
    def test_unknown_session_fails_closed(self, runner):
        r = runner(gate=True)
        ok = run(r.inject_plugin_message("who?", target_session="telegram:dm:chat-9:user-9", plugin_id="hermes-peer"))
        assert ok is False
        assert r._dispatched == []

    def test_missing_target_rejected(self, runner):
        r = runner(gate=True)
        assert run(r.inject_plugin_message("who?", plugin_id="hermes-peer")) is False

    def test_invalid_mode_rejected(self, runner):
        r = runner(gate=True)
        ok = run(r.inject_plugin_message("hi", target_session=KEY, plugin_id="hermes-peer", mode="bogus"))
        assert ok is False
        assert r._dispatched == []

    def test_gateway_steer_and_interrupt_unsupported_in_v1(self, runner):
        r = runner(gate=True)
        assert run(r.inject_plugin_message("hi", target_session=KEY, plugin_id="hermes-peer", mode="steer")) is False
        assert run(r.inject_plugin_message("hi", target_session=KEY, plugin_id="hermes-peer", mode="interrupt")) is False

    def test_ended_session_fails_closed(self, runner):
        r = runner(gate=True)
        r.session_store._is_session_ended_in_db = lambda sid: True
        ok = run(r.inject_plugin_message("too late", target_session=KEY, plugin_id="hermes-peer"))
        assert ok is False
        assert r._dispatched == []


# ---------------------------------------------------------------------------
# H-107 — internal injected events never reach command dispatch
# ---------------------------------------------------------------------------


class TestGatewayInertControl:
    def test_internal_event_skips_command_routing(self, runner):
        """Injected events are marked non_control; the host command gate
        treats them as conversational only."""
        r = runner(gate=True)
        ok = run(r.inject_plugin_message("/approve", target_session=KEY, plugin_id="hermes-peer"))
        assert ok is True
        assert len(r._dispatched) == 1
        event = r._dispatched[0]
        assert event.internal is True
        assert event.non_control is True  # the host command gate key

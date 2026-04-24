# -*- coding: utf-8 -*-
"""Unit tests for gateway.platforms.qqbot.core.websocket.

Uses WSCallbacks injection — no adapter mock needed.
All imports reference core/ directly.
"""

from __future__ import annotations

import asyncio
import time
from unittest import mock

import pytest

from gateway.platforms.qqbot.core.dto import (
    CloseAction,
    EventType,
    OPCode,
    classify_close_code,
)
from gateway.platforms.qqbot.core.websocket import QQCloseError, QQWebSocket, WSCallbacks


# ── Helpers ───────────────────────────────────────────────────────────

def _make_callbacks(**overrides):
    """Build a WSCallbacks with sensible defaults for testing."""
    session = {"id": None, "seq": None}
    state = {"heartbeat_interval": 30.0, "connected": False, "fatal": None}

    async def _get_token():
        return "test-token"

    async def _get_gateway_url():
        return "wss://test/"

    async def _on_message(event_type, data):
        pass

    defaults = WSCallbacks(
        on_message_event=_on_message,
        on_connected=lambda: state.update({"connected": True}),
        on_disconnected=lambda: state.update({"connected": False}),
        on_fatal_error=lambda code, msg, retry: state.update({"fatal": (code, msg)}),
        get_token=_get_token,
        get_gateway_url=_get_gateway_url,
        get_session=lambda: (session["id"], session["seq"]),
        set_session=lambda sid, seq: session.update(
            {k: v for k, v in (("id", sid), ("seq", seq)) if v is not None or k == "id"}
        ),
        set_heartbeat_interval=lambda v: state.update({"heartbeat_interval": v}),
        clear_token=lambda: None,
        fail_pending=lambda reason: None,
    )
    for k, v in overrides.items():
        object.__setattr__(defaults, k, v)
    return defaults, session, state


def _make_ws(**overrides):
    callbacks, session, state = _make_callbacks(**overrides)
    ws = QQWebSocket(callbacks=callbacks, log_tag="QQBot:test")
    return ws, callbacks, session, state


# ── classify_close_code ───────────────────────────────────────────────

class TestClassifyCloseCode:
    def test_fatal(self):
        assert classify_close_code(4914) == CloseAction.STOP
        assert classify_close_code(4915) == CloseAction.STOP

    def test_rate_limit(self):
        assert classify_close_code(4008) == CloseAction.RATE_LIMIT

    def test_clear_token_is_now_reconnect(self):
        assert classify_close_code(4004) == CloseAction.RECONNECT

    def test_identify_only(self):
        assert classify_close_code(4006) == CloseAction.IDENTIFY_ONLY
        assert classify_close_code(4900) == CloseAction.IDENTIFY_ONLY

    def test_reconnect(self):
        assert classify_close_code(1000) == CloseAction.RECONNECT
        assert classify_close_code(None) == CloseAction.RECONNECT


# ── QQCloseError ──────────────────────────────────────────────────────

class TestQQCloseError:
    def test_int_code(self):
        err = QQCloseError(4004, "bad token")
        assert err.code == 4004
        assert err.reason == "bad token"

    def test_string_code(self):
        assert QQCloseError("4914", "").code == 4914

    def test_none_code(self):
        assert QQCloseError(None, "").code is None


# ── _parse_json ───────────────────────────────────────────────────────

class TestParseJson:
    def test_valid(self):
        assert QQWebSocket._parse_json('{"op": 10}') == {"op": 10}

    def test_invalid(self):
        assert QQWebSocket._parse_json("not json") is None

    def test_non_dict(self):
        assert QQWebSocket._parse_json("[1,2,3]") is None

    def test_empty_dict(self):
        assert QQWebSocket._parse_json("{}") == {}

    def test_none(self):
        assert QQWebSocket._parse_json(None) is None


# ── _create_task ──────────────────────────────────────────────────────

class TestCreateTask:
    def test_no_event_loop_returns_none(self):
        async def dummy():
            pass

        result = QQWebSocket._create_task(dummy())
        assert result is None


# ── _dispatch_payload ─────────────────────────────────────────────────

class TestDispatchPayload:
    def test_hello_sets_heartbeat_interval(self):
        ws, cb, session, state = _make_ws()
        ws._dispatch_payload({"op": OPCode.HELLO, "d": {"heartbeat_interval": 41250}})
        # 41250ms * 0.8 = 33.0s
        assert state["heartbeat_interval"] == pytest.approx(33.0)

    def test_hello_triggers_identify_when_no_session(self):
        ws, cb, session, state = _make_ws()
        with mock.patch.object(ws, "_create_task") as ct:
            ct.side_effect = lambda coro: coro.close() or None
            ws._dispatch_payload({"op": OPCode.HELLO, "d": {"heartbeat_interval": 30000}})
            assert ct.called

    def test_hello_triggers_resume_when_session_exists(self):
        ws, cb, session, state = _make_ws()
        session["id"] = "s-1"
        session["seq"] = 42
        with mock.patch.object(ws, "_create_task") as ct:
            ct.side_effect = lambda coro: coro.close() or None
            ws._dispatch_payload({"op": OPCode.HELLO, "d": {"heartbeat_interval": 30000}})
            assert ct.called

    def test_ready_stores_session_id(self):
        ws, cb, session, state = _make_ws()
        ws._dispatch_payload({
            "op": OPCode.DISPATCH, "t": EventType.READY,
            "s": 1, "d": {"session_id": "s-new"},
        })
        assert session["id"] == "s-new"

    def test_seq_tracking(self):
        ws, cb, session, state = _make_ws()
        ws._dispatch_payload({"op": OPCode.DISPATCH, "t": "RESUMED", "s": 5})
        assert session["seq"] == 5

    def test_seq_only_increments(self):
        ws, cb, session, state = _make_ws()
        session["seq"] = 10
        ws._dispatch_payload({"op": OPCode.DISPATCH, "t": "RESUMED", "s": 5})
        assert session["seq"] == 10

    def test_heartbeat_ack_no_error(self):
        ws, cb, session, state = _make_ws()
        ws._dispatch_payload({"op": OPCode.HEARTBEAT_ACK})

    def test_unknown_op_no_error(self):
        ws, cb, session, state = _make_ws()
        ws._dispatch_payload({"op": 99, "d": {}})


# ── _update_quick_count ───────────────────────────────────────────────

class TestQuickDisconnect:
    def test_increments_count(self):
        ws, *_ = _make_ws()
        connect_time = time.monotonic() - 1.0
        assert ws._update_quick_count(connect_time, 0) == 1

    def test_resets_on_long_connection(self):
        ws, *_ = _make_ws()
        connect_time = time.monotonic() - 60.0
        assert ws._update_quick_count(connect_time, 5) == 0

    def test_fatal_on_too_many(self):
        ws, cb, session, state = _make_ws()
        connect_time = time.monotonic() - 1.0
        result = ws._update_quick_count(connect_time, 2)
        assert result == -1
        assert state["fatal"] is not None

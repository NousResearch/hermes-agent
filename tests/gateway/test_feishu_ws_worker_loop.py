"""Regression tests for the Feishu/Lark WS worker-loop exception handler.

The Lark SDK schedules its ``_receive_message_loop`` task on the adapter's
own thread-local worker loop, not on the gateway loop. A normal close (1000)
or transient disconnect surfacing from that task must be swallowed-with-logging
on the worker loop so it never escalates to a gateway TEMPFAIL (issue #67358,
sweeper review of PR #67366).
"""

import asyncio

from plugins.platforms.feishu.adapter import _feishu_ws_loop_exception_handler


class ConnectionClosedOK(Exception):
    """Stand-in for websockets ConnectionClosedOK (normal close 1000)."""


class _RealBug(Exception):
    """A non-transient error that must fall through to default handling."""


def _make_context(exc):
    return {"message": "Task exception was never retrieved", "exception": exc}


def test_worker_loop_swallows_normal_close(monkeypatch):
    loop = asyncio.new_event_loop()
    try:
        forwarded = {"called": False}
        monkeypatch.setattr(
            loop,
            "default_exception_handler",
            lambda ctx: forwarded.__setitem__("called", True),
        )
        _feishu_ws_loop_exception_handler(
            loop, _make_context(ConnectionClosedOK("1000 (OK)"))
        )
        # Transient close is swallowed; default handler NOT invoked.
        assert forwarded["called"] is False
    finally:
        loop.close()


def test_worker_loop_forwards_real_bug(monkeypatch):
    loop = asyncio.new_event_loop()
    try:
        forwarded = {"ctx": None}
        monkeypatch.setattr(
            loop,
            "default_exception_handler",
            lambda ctx: forwarded.__setitem__("ctx", ctx),
        )
        _feishu_ws_loop_exception_handler(loop, _make_context(_RealBug("boom")))
        # Non-transient error is forwarded to the default handler.
        assert forwarded["ctx"] is not None
        assert isinstance(forwarded["ctx"]["exception"], _RealBug)
    finally:
        loop.close()


def test_worker_loop_no_exception_forwards(monkeypatch):
    """A context without an exception must not be swallowed."""
    loop = asyncio.new_event_loop()
    try:
        forwarded = {"called": False}
        monkeypatch.setattr(
            loop,
            "default_exception_handler",
            lambda ctx: forwarded.__setitem__("called", True),
        )
        _feishu_ws_loop_exception_handler(loop, {"message": "no exc"})
        assert forwarded["called"] is True
    finally:
        loop.close()

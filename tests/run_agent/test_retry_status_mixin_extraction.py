"""Regression tests for the retry-status mixin extraction (shard s1, c18).

The 5 buffered retry/fallback status helpers moved VERBATIM from
``run_agent.py`` into ``plugins.agent.mixins.retry_status_mixin``.
Tests pin the buffering semantics (accumulate, flush on terminal
failure, drop on success, one-shot fallback notice) and the MRO wiring.
"""

from __future__ import annotations

import run_agent
from plugins.agent.mixins.retry_status_mixin import RetryStatusMixin

MOVED = (
    "_buffer_status",
    "_buffer_vprint",
    "_clear_status_buffer",
    "_emit_pending_fallback_notice",
    "_flush_status_buffer",
)


def _bare_agent(**attrs):
    """Object.__new__-based bare adapter (no __init__ side effects)."""
    agent = object.__new__(run_agent.AIAgent)
    agent.log_prefix = ""
    agent._emit_status = lambda msg: None
    agent._emit_warning = lambda msg: None
    agent._vprint = lambda msg, force=False: None
    for key, value in attrs.items():
        setattr(agent, key, value)
    return agent


def test_methods_are_wired_through_mro():
    for name in MOVED:
        assert getattr(run_agent.AIAgent, name) is getattr(RetryStatusMixin, name), name


def test_buffer_status_accumulates_then_flushes():
    agent = _bare_agent()
    emitted = []
    agent._emit_status = lambda msg: emitted.append(("status", msg))

    agent._buffer_status("\u23f3 Retrying in 5s...")
    agent._buffer_status("\u26a0\ufe0f Fallback attempt 2...")
    assert emitted == []  # deferred until flush

    agent._flush_status_buffer()
    assert emitted == [("status", "\u23f3 Retrying in 5s..."), ("status", "\u26a0\ufe0f Fallback attempt 2...")]

    # Buffer drained: a second flush emits nothing.
    agent._flush_status_buffer()
    assert emitted == [("status", "\u23f3 Retrying in 5s..."), ("status", "\u26a0\ufe0f Fallback attempt 2...")]


def test_buffer_vprint_flush_uses_log_prefix():
    agent = _bare_agent(log_prefix="[agent] ")
    vprinted = []
    agent._vprint = lambda msg, force=False: vprinted.append((msg, force))

    agent._buffer_vprint("retry detail line")
    agent._flush_status_buffer()
    assert vprinted == [("[agent] retry detail line", True)]


def test_flush_emits_warning_kind():
    agent = _bare_agent()
    warned = []
    agent._emit_warning = lambda msg: warned.append(msg)

    agent._buffer_status("status line")
    # Simulate a warn-kind entry by poking the buffer directly (private shape
    # is (kind, text) tuples appended by _buffer_status/_buffer_vprint).
    buf = getattr(agent, "_retry_status_buffer")
    buf.append(("warn", "warning line"))
    agent._flush_status_buffer()
    assert warned == ["warning line"]


def test_clear_status_buffer_drops_silently():
    agent = _bare_agent()
    emitted = []
    agent._emit_status = lambda msg: emitted.append(msg)

    agent._buffer_status("doomed retry")
    agent._clear_status_buffer()
    agent._flush_status_buffer()
    assert emitted == []


def test_emit_pending_fallback_notice_emits_once_then_clears():
    agent = _bare_agent()
    emitted = []
    agent._emit_status = lambda msg: emitted.append(msg)
    agent._pending_fallback_notice = "Switched provider X -> Y"

    agent._emit_pending_fallback_notice()
    assert emitted == ["Switched provider X -> Y"]
    # Cleared after emission: a second call emits nothing.
    agent._emit_pending_fallback_notice()
    assert emitted == ["Switched provider X -> Y"]


def test_flush_discards_pending_fallback_notice():
    agent = _bare_agent()
    emitted = []
    agent._emit_status = lambda msg: emitted.append(msg)
    agent._pending_fallback_notice = "Switched provider X -> Y"

    agent._buffer_status("gave up after retries")
    agent._flush_status_buffer()
    # The buffered trace carries the switch line already; the one-shot notice
    # must be dropped so it cannot leak into a later successful turn.
    assert agent._pending_fallback_notice is None
    assert emitted == ["gave up after retries"]

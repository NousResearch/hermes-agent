"""Tests for the retry/fallback status buffer helpers on AIAgent.

These helpers defer noisy retry chatter (rate-limit retries, fallback
switches, compression attempts) so users only see the trace when
everything ultimately fails.  On successful recovery the buffer is
silently dropped.
"""

from __future__ import annotations


from run_agent import AIAgent


def _make_bare_agent():
    """Construct an AIAgent without running __init__ — we only need the
    buffered-status helpers, which are pure-Python and depend only on a
    handful of attributes."""
    agent = object.__new__(AIAgent)
    agent.log_prefix = ""
    agent.status_callback = None
    agent.suppress_status_output = False
    agent._mute_post_response = False
    agent._executing_tools = False
    agent._print_fn = None
    return agent


def test_buffer_status_accumulates_then_flushes(capsys):
    agent = _make_bare_agent()
    emitted = []
    agent._emit_status = lambda msg: emitted.append(("status", msg))

    agent._buffer_status("⏳ Retrying...")
    agent._buffer_status("⚠️ Fallback...")

    # Nothing emitted yet — they are buffered.
    assert emitted == []
    assert agent._retry_status_buffer == [
        ("status", "⏳ Retrying..."),
        ("status", "⚠️ Fallback..."),
    ]

    # Flush surfaces them in order through _emit_status.
    agent._flush_status_buffer()
    assert emitted == [
        ("status", "⏳ Retrying..."),
        ("status", "⚠️ Fallback..."),
    ]
    # Buffer is drained.
    assert agent._retry_status_buffer == []


def test_clear_drops_buffered_messages_silently():
    agent = _make_bare_agent()
    emitted = []
    agent._emit_status = lambda msg: emitted.append(msg)

    agent._buffer_status("⏳ Retrying...")
    agent._buffer_status("⚠️ Fallback...")
    agent._clear_status_buffer()

    # Nothing was emitted — clear is the success path.
    assert emitted == []
    assert agent._retry_status_buffer == []

    # Subsequent flush is a no-op.
    agent._flush_status_buffer()
    assert emitted == []


def test_buffer_vprint_replays_via_vprint_with_log_prefix():
    agent = _make_bare_agent()
    agent.log_prefix = "[abc] "
    seen = []
    agent._vprint = lambda msg, force=False, **kw: seen.append((msg, force))

    agent._buffer_vprint("⚠️  API call failed")
    agent._flush_status_buffer()

    # Replays through _vprint with force=True and the agent's log_prefix
    # prepended (matching the original direct-emit format).
    assert seen == [("[abc] ⚠️  API call failed", True)]


def test_flush_empty_buffer_is_noop():
    agent = _make_bare_agent()
    emitted = []
    agent._emit_status = lambda msg: emitted.append(msg)
    agent._vprint = lambda msg, force=False, **kw: emitted.append(msg)

    # No buffer attribute yet — flush should be a quiet no-op.
    agent._flush_status_buffer()
    assert emitted == []

    # Even after touching the buffer (via clear on an empty/missing buffer).
    agent._clear_status_buffer()
    agent._flush_status_buffer()
    assert emitted == []


def test_re_buffer_after_flush_works():
    agent = _make_bare_agent()
    emitted = []
    agent._emit_status = lambda msg: emitted.append(msg)

    agent._buffer_status("first")
    agent._flush_status_buffer()
    agent._buffer_status("second")
    agent._flush_status_buffer()

    assert emitted == ["first", "second"]


def test_mixed_kinds_replay_through_correct_channels():
    agent = _make_bare_agent()
    agent.log_prefix = ""
    statuses = []
    vprints = []
    warns = []
    agent._emit_status = lambda msg: statuses.append(msg)
    agent._vprint = lambda msg, force=False, **kw: vprints.append((msg, force))
    agent._emit_warning = lambda msg: warns.append(msg)

    agent._buffer_status("status-1")
    agent._buffer_vprint("vprint-1")
    # Manually mix in a "warn" record to verify the dispatch still works.
    agent._retry_status_buffer.append(("warn", "warn-1"))
    agent._buffer_status("status-2")

    agent._flush_status_buffer()

    assert statuses == ["status-1", "status-2"]
    assert vprints == [("vprint-1", True)]
    assert warns == ["warn-1"]


def test_pending_fallback_notice_emitted_once_on_success():
    """On successful recovery the one-shot fallback notice is surfaced even
    though the noisy retry buffer is dropped."""
    agent = _make_bare_agent()
    emitted = []
    agent._emit_status = lambda msg: emitted.append(msg)

    # Simulate try_activate_fallback: buffer the noisy switch line AND record
    # the durable one-shot notice.
    agent._buffer_status("🔄 Primary model failed — switching to fallback: m2 via p2")
    agent._pending_fallback_notice = "🔄 Switched to fallback model: m1 via p1 → m2 via p2"

    # Success path order: emit pending notice, then drop the buffer.
    agent._emit_pending_fallback_notice()
    agent._clear_status_buffer()

    # The durable notice was shown exactly once; the buffered retry noise was
    # silently dropped.
    assert emitted == ["🔄 Switched to fallback model: m1 via p1 → m2 via p2"]
    assert agent._retry_status_buffer == []
    # Notice is cleared so it cannot re-emit on a later turn.
    assert agent._pending_fallback_notice is None

    # A second success path with no new fallback emits nothing.
    agent._emit_pending_fallback_notice()
    assert emitted == ["🔄 Switched to fallback model: m1 via p1 → m2 via p2"]


def test_pending_fallback_notice_noop_when_unset():
    """No fallback this turn → no notice emitted on the success path."""
    agent = _make_bare_agent()
    emitted = []
    agent._emit_status = lambda msg: emitted.append(msg)

    # No _pending_fallback_notice attribute set at all.
    agent._emit_pending_fallback_notice()
    assert emitted == []


def test_flush_discards_pending_fallback_notice():
    """On terminal failure the flushed buffer already carries the switch line,
    so the one-shot notice is discarded to avoid a stale duplicate later."""
    agent = _make_bare_agent()
    emitted = []
    agent._emit_status = lambda msg: emitted.append(msg)

    agent._buffer_status("🔄 Primary model failed — switching to fallback: m2 via p2")
    agent._pending_fallback_notice = "🔄 Switched to fallback model: m1 via p1 → m2 via p2"

    # Terminal failure flushes the buffered trace...
    agent._flush_status_buffer()
    assert emitted == ["🔄 Primary model failed — switching to fallback: m2 via p2"]
    # ...and discards the pending notice so it won't re-emit on a later turn.
    assert agent._pending_fallback_notice is None

    emitted.clear()
    agent._emit_pending_fallback_notice()
    assert emitted == []


def test_pending_fallback_notice_survives_emit_callback_error():
    """A failing status callback must not leave the notice set for a stale
    re-emit, and must not raise."""
    agent = _make_bare_agent()
    seen = []

    def boom(msg):
        seen.append(msg)
        raise RuntimeError("simulated callback failure")

    agent._emit_status = boom
    agent._pending_fallback_notice = "🔄 Switched to fallback model: m1 via p1 → m2 via p2"

    # Should not raise.
    agent._emit_pending_fallback_notice()
    # Attempt was made and the notice is cleared regardless.
    assert seen == ["🔄 Switched to fallback model: m1 via p1 → m2 via p2"]
    assert agent._pending_fallback_notice is None


def test_flush_swallows_callback_exceptions():
    agent = _make_bare_agent()
    seen = []

    def boom(msg):
        seen.append(msg)
        raise RuntimeError("simulated callback failure")

    agent._emit_status = boom

    agent._buffer_status("first")
    agent._buffer_status("second")
    # Should not raise even though _emit_status raises for every message.
    agent._flush_status_buffer()

    # Both messages were attempted.
    assert seen == ["first", "second"]
    # Buffer drained regardless of failures.
    assert agent._retry_status_buffer == []

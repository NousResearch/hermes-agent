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

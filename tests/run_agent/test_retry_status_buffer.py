"""Tests for the retry/fallback status buffer helpers on AIAgent.

These helpers defer noisy retry chatter (rate-limit retries, compression
attempts) so users only see the trace when everything ultimately fails.
On successful recovery the buffer is silently dropped.  A provider/model
switch is not chatter: it is emitted at the moment it happens, so it is
never buffered and never deferred.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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


def _make_fallback_agent(fallback_model):
    """Real AIAgent with a fallback chain, so the switch path is exercised
    rather than simulated."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _mock_client():
    mock = MagicMock()
    mock.base_url = "https://openrouter.ai/api/v1"
    mock.api_key = "fb-key"
    return mock


def _switch_lines(emitted):
    return [m for m in emitted if "Switched to fallback model:" in m]


def test_switch_notice_emitted_at_the_switch_not_after_the_response():
    """The switch is announced while it happens, before any content exists.

    A notice that can only arrive after the fallback's answer cannot warn
    anyone against acting on that answer.
    """
    agent = _make_fallback_agent([{"provider": "openai", "model": "gpt-4o"}])
    emitted = []
    agent._emit_status = lambda msg: emitted.append(msg)

    with patch("agent.auxiliary_client.resolve_provider_client",
               return_value=(_mock_client(), "gpt-4o")):
        assert agent._try_activate_fallback() is True

    # Emitted by the time the switch call returns — no content has been
    # requested from the fallback model yet, let alone produced.
    assert len(_switch_lines(emitted)) == 1
    assert "gpt-4o" in _switch_lines(emitted)[0]
    # And nothing about the switch is left sitting in the retry buffer, which
    # is what used to delay it.
    assert _switch_lines(
        [msg for _kind, msg in getattr(agent, "_retry_status_buffer", [])]
    ) == []


def test_switch_seen_exactly_once_on_success():
    """Success path: the retry noise is dropped, the switch stays seen once."""
    agent = _make_fallback_agent([{"provider": "openai", "model": "gpt-4o"}])
    emitted = []
    agent._emit_status = lambda msg: emitted.append(msg)

    with patch("agent.auxiliary_client.resolve_provider_client",
               return_value=(_mock_client(), "gpt-4o")):
        assert agent._try_activate_fallback() is True
    agent._clear_status_buffer()          # success path

    assert len(_switch_lines(emitted)) == 1


def test_switch_seen_exactly_once_on_terminal_failure():
    """Terminal failure flushes the retry trace — it must not repeat the
    switch that was already announced at the moment it happened."""
    agent = _make_fallback_agent([{"provider": "openai", "model": "gpt-4o"}])
    emitted = []
    agent._emit_status = lambda msg: emitted.append(msg)

    with patch("agent.auxiliary_client.resolve_provider_client",
               return_value=(_mock_client(), "gpt-4o")):
        assert agent._try_activate_fallback() is True
    agent._flush_status_buffer()          # terminal-failure path

    assert len(_switch_lines(emitted)) == 1


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


def test_terminal_trace_ends_on_the_model_that_failed():
    """The switch lines are live now rather than buffered, so the flushed
    trace — all a client reconnecting after the failure gets — would otherwise
    carry no model identity at all."""
    agent = _make_bare_agent()
    agent.model = "gpt-4o"
    agent.provider = "openai"
    emitted = []
    agent._emit_status = lambda msg: emitted.append(msg)

    agent._buffer_status("⏳ Retrying...")
    agent._flush_status_buffer()

    assert emitted[-1] == "⏹ Ended on gpt-4o via openai"


def test_no_identity_line_when_there_is_no_trace_to_end():
    """An empty buffer means the turn never degraded — nothing to close."""
    agent = _make_bare_agent()
    agent.model = "gpt-4o"
    agent.provider = "openai"
    emitted = []
    agent._emit_status = lambda msg: emitted.append(msg)

    agent._flush_status_buffer()

    assert emitted == []


def test_emit_status_swallows_its_own_exceptions():
    """Load-bearing for the switch notice: ``try_activate_fallback`` emits it
    with no guard of its own, on the argument that ``_emit_status`` cannot
    raise. If it could, a status hiccup would land in that function's ``except``
    and cascade down the rest of the fallback chain.
    """
    agent = _make_bare_agent()

    def boom(*args, **kwargs):
        raise RuntimeError("simulated status failure")

    agent._vprint = boom          # CLI channel fails
    agent.status_callback = boom  # gateway channel fails too

    # Both channels raising, and still nothing escapes.
    agent._emit_status("🔄 Switched to fallback model: a via p → b via q")


"""Tests for the retry/fallback status buffer helpers on AIAgent.

These helpers defer noisy retry chatter (rate-limit retries, compression
attempts) so users only see the trace when everything ultimately fails.
On successful recovery the buffer is silently dropped.  An in-loop
provider/model switch is not chatter: it is emitted at the moment it
happens, never buffered and never deferred.  The one-shot pending notice
remains for switches made before the agent exists (gateway / TUI
credential resolution), which have no agent to emit through.
"""

from __future__ import annotations

from types import SimpleNamespace
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
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="http://127.0.0.1:1234/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


_FALLBACK = {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}


def _mock_client():
    mock = MagicMock()
    mock.base_url = "https://openrouter.ai/api/v1"
    mock.api_key = "fb-key"
    return mock


def _resolves_to(client):
    return patch("agent.auxiliary_client.resolve_provider_client",
                 return_value=(client, _FALLBACK["model"]))


def _switch_lines(emitted):
    return [m for m in emitted if "Model fallback:" in m]


def test_switch_notice_emitted_at_the_switch_not_after_the_response():
    """The switch is announced while it happens, before any content exists.

    A notice that can only arrive after the fallback's answer cannot warn
    anyone against acting on that answer.
    """
    agent = _make_fallback_agent([_FALLBACK])
    emitted = []
    agent._emit_status = lambda msg: emitted.append(msg)

    with _resolves_to(_mock_client()):
        assert agent._try_activate_fallback() is True

    # Emitted by the time the switch call returns — no content has been
    # requested from the fallback model yet, let alone produced.
    assert len(_switch_lines(emitted)) == 1
    assert _FALLBACK["model"] in _switch_lines(emitted)[0]
    # Nothing about the switch is left waiting: not in the retry buffer, not
    # in the pending one-shot (reserved for pre-agent switches).
    assert _switch_lines(
        [msg for _kind, msg in getattr(agent, "_retry_status_buffer", None) or []]
    ) == []
    assert not getattr(agent, "_pending_fallback_notice", None)


def test_switch_seen_exactly_once_on_success():
    """Success path: the retry noise is dropped, the switch stays seen once."""
    agent = _make_fallback_agent([_FALLBACK])
    emitted = []
    agent._emit_status = lambda msg: emitted.append(msg)

    with _resolves_to(_mock_client()):
        assert agent._try_activate_fallback() is True
    agent._emit_pending_fallback_notice()  # success path, as the loop runs it
    agent._clear_status_buffer()

    assert len(_switch_lines(emitted)) == 1


def test_switch_seen_exactly_once_on_terminal_failure():
    """Terminal failure flushes the retry trace — it must not repeat the
    switch that was already announced at the moment it happened."""
    agent = _make_fallback_agent([_FALLBACK])
    emitted = []
    agent._emit_status = lambda msg: emitted.append(msg)

    with _resolves_to(_mock_client()):
        assert agent._try_activate_fallback() is True
    agent._flush_status_buffer()          # terminal-failure path

    assert len(_switch_lines(emitted)) == 1


def test_switch_notice_precedes_the_fallback_answer_in_a_real_turn():
    """End to end: the primary returns empty content, the loop switches, the
    fallback answers.  The operator must see the switch before the fallback
    model is even called, not underneath its answer."""
    agent = _make_fallback_agent([_FALLBACK])
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False

    def _response(content):
        msg = SimpleNamespace(content=content, tool_calls=None, reasoning=None,
                              reasoning_content=None, reasoning_details=None)
        choice = SimpleNamespace(message=msg, finish_reason="stop")
        return SimpleNamespace(choices=[choice], model="m", usage=None)

    events = []
    agent.status_callback = lambda _kind, msg: (
        events.append("switch") if "Model fallback:" in msg else None)
    agent.client.chat.completions.create.return_value = _response(None)

    fallback_client = _mock_client()

    def _answer(*_a, **_k):
        events.append("answer")
        return _response("Fallback answer.")

    fallback_client.chat.completions.create.side_effect = _answer

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        _resolves_to(fallback_client),
    ):
        result = agent.run_conversation("answer me")

    assert result["final_response"] == "Fallback answer."
    assert events == ["switch", "answer"]


def test_pending_fallback_notice_emitted_once_on_success():
    """The one-shot pre-agent notice (set by the gateway / TUI before the
    turn) is surfaced on success even though the retry buffer is dropped."""
    agent = _make_bare_agent()
    emitted = []
    agent._emit_status = lambda msg: emitted.append(msg)

    # Retry chatter buffered during the turn, plus the notice attached to the
    # agent before it ran.
    agent._buffer_status("⏳ Retrying...")
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


def test_pending_fallback_notice_emits_all_switches_in_order():
    agent = _make_bare_agent()
    emitted = []
    agent._emit_status = emitted.append
    agent._pending_fallback_notice = ["primary → fallback-1", "fallback-1 → fallback-2"]

    agent._emit_pending_fallback_notice()

    assert emitted == ["primary → fallback-1", "fallback-1 → fallback-2"]
    assert agent._pending_fallback_notice is None


def test_pending_fallback_notice_continues_after_callback_error():
    agent = _make_bare_agent()
    attempted = []
    agent._pending_fallback_notice = ["first", "second"]

    def emit(message):
        attempted.append(message)
        if message == "first":
            raise RuntimeError("surface unavailable")

    agent._emit_status = emit
    agent._emit_pending_fallback_notice()

    assert attempted == ["first", "second"]
    assert agent._pending_fallback_notice is None


def test_pending_fallback_notice_noop_when_unset():
    """No fallback this turn → no notice emitted on the success path."""
    agent = _make_bare_agent()
    emitted = []
    agent._emit_status = lambda msg: emitted.append(msg)

    # No _pending_fallback_notice attribute set at all.
    agent._emit_pending_fallback_notice()
    assert emitted == []


def test_flush_discards_pending_fallback_notice():
    """On terminal failure the one-shot pre-agent notice is discarded so it
    cannot leak into a later successful turn."""
    agent = _make_bare_agent()
    emitted = []
    agent._emit_status = lambda msg: emitted.append(msg)

    agent._buffer_status("⏳ Retrying...")
    agent._pending_fallback_notice = "🔄 Switched to fallback model: m1 via p1 → m2 via p2"

    # Terminal failure flushes the buffered trace...
    agent._flush_status_buffer()
    assert emitted == ["⏳ Retrying..."]
    # ...and discards the pending notice so it won't re-emit on a later turn.
    assert agent._pending_fallback_notice is None

    emitted.clear()
    agent._emit_pending_fallback_notice()
    assert emitted == []




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
    agent._emit_diagnostic_status("⚠️ Model fallback: a via p unavailable; using b via q.")

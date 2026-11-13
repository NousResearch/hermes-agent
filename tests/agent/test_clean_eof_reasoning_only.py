"""#144116 — a transport-clean EOF with no visible text must not be reported as
output-token exhaustion, and must not describe itself as a network error.

Pins four classes apart, because they share one terminal copy today:

* clean EOF (no exception, no ``finish_reason``, no usage) — reasoning-only
* transport drop (an exception mid-stream; the stub is NOT tagged ``_clean_eof``)
* genuine output-token exhaustion (``finish_reason="length"`` from the provider)
* a healthy turn (control)

Also pins that a clean-EOF turn leaves no synthetic continuation nudge in the
transcript: the nudge used to be appended as a ``user`` row with no assistant row
before it, and the alternation repair then welded it into the user's own message.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_constants import FINISH_REASON_LENGTH, PARTIAL_STREAM_STUB_ID

OUTPUT_LIMIT_MARK = "hit its output-token limit on every continuation attempt"
STREAM_CLOSED_MARK = "before any answer text arrived"
NETWORK_ERROR_MARK = "network error"
NUDGE_MARK = "[System: The previous response"
# The copy reports what Hermes observed; it may not assert what the upstream did.
ABSOLUTE_CAUSE_MARKS = ("not an output-length", "not a token", "never hit", "was not a network fault")


def _agent():
    from run_agent import AIAgent

    a = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    a.api_mode = "chat_completions"
    a._interrupt_requested = False
    a.client = MagicMock()
    return a


def _clean_eof_stub(content=None, reasoning="reasoning that consumed the budget"):
    """The shape ``_finish_chat_stream`` builds for a cleanly closed stream."""
    msg = SimpleNamespace(role="assistant", content=content, tool_calls=None,
                          reasoning_content=reasoning)
    return SimpleNamespace(
        id=PARTIAL_STREAM_STUB_ID, model="test/model",
        choices=[SimpleNamespace(index=0, message=msg, finish_reason=FINISH_REASON_LENGTH)],
        usage=None, _clean_eof=True,
    )


def _transport_drop_stub(content=None, reasoning="reasoning that consumed the budget"):
    """Same shape after a real transport exception: NOT tagged clean EOF."""
    stub = _clean_eof_stub(content=content, reasoning=reasoning)
    stub._clean_eof = False
    return stub


def _genuine_length(content=None, reasoning="reasoning that consumed the budget"):
    """The provider itself reported the output-token stop."""
    msg = SimpleNamespace(role="assistant", content=content, tool_calls=None,
                          reasoning_content=reasoning)
    return SimpleNamespace(
        id="chatcmpl-real", model="test/model",
        choices=[SimpleNamespace(index=0, message=msg, finish_reason=FINISH_REASON_LENGTH)],
        usage=None,
    )


def _healthy(text="the answer"):
    msg = SimpleNamespace(role="assistant", content=text, tool_calls=None,
                          reasoning_content=None)
    return SimpleNamespace(
        id="chatcmpl-ok", model="test/model",
        choices=[SimpleNamespace(index=0, message=msg, finish_reason="stop")],
        usage=None,
    )


def _run(agent, response, user_message="explain the migration plan"):
    """One turn; returns (result, api_call_count, persisted rows, request payloads)."""
    requests = []

    def _create(**kwargs):
        requests.append(kwargs)
        return response

    agent.client.chat.completions.create.side_effect = _create
    captured = {}
    real_persist = agent._persist_session

    def _capture(messages, history, *a, **k):
        captured["rows"] = [dict(m) for m in messages if isinstance(m, dict)]
        return real_persist(messages, history, *a, **k)

    with (patch.object(agent, "_persist_session", side_effect=_capture),
          patch.object(agent, "_save_trajectory"),
          patch.object(agent, "_cleanup_task_resources")):
        result = agent.run_conversation(user_message)
    return result, len(requests), captured.get("rows") or [], requests


# ── 1. clean EOF ──────────────────────────────────────────────────────────
def test_clean_eof_reasoning_only_is_not_reported_as_output_token_exhaustion():
    result, calls, _rows, _reqs = _run(_agent(), _clean_eof_stub())
    final = result["final_response"]

    assert OUTPUT_LIMIT_MARK not in final, (
        "a stream the server closed cleanly is still explained to the user as the "
        "model's reasoning eating the output budget"
    )
    assert STREAM_CLOSED_MARK in final
    assert "/reasoning" not in final, (
        "lowering reasoning effort cannot fix a stream the provider closed"
    )
    assert "max_tokens" not in final
    assert not any(mark in final for mark in ABSOLUTE_CAUSE_MARKS), (
        "the upstream cause of a clean close is not established: the copy may report what "
        "Hermes observed, not assert what the provider did"
    )
    # The failure class is unchanged for downstream consumers (tui/gateway).
    assert result["failure_reason"] == "truncated"
    assert result["completed"] is False
    # The recovery window is NOT removed: four attempts still go out.
    assert calls == 4, "clean-EOF recovery attempts must not be reduced"


# ── 2. transport drop keeps its existing wording ──────────────────────────
def test_transport_drop_stub_keeps_output_token_copy():
    result, calls, _rows, _reqs = _run(_agent(), _transport_drop_stub())
    final = result["final_response"]

    assert OUTPUT_LIMIT_MARK in final, (
        "the sibling transport-drop class is out of scope for #144116 and must keep "
        "its existing terminal copy"
    )
    assert STREAM_CLOSED_MARK not in final
    assert calls == 4


# ── 3. genuine output-token exhaustion keeps its copy ─────────────────────
def test_genuine_output_token_exhaustion_keeps_its_copy():
    result, calls, _rows, _reqs = _run(_agent(), _genuine_length())
    final = result["final_response"]

    assert OUTPUT_LIMIT_MARK in final
    assert STREAM_CLOSED_MARK not in final
    assert calls == 4


# ── 4. control: a healthy turn is untouched ───────────────────────────────
def test_healthy_response_is_untouched():
    result, calls, _rows, _reqs = _run(_agent(), _healthy("here is the answer"))
    final = result["final_response"]

    assert final == "here is the answer"
    assert result["completed"] is True
    assert calls == 1
    assert OUTPUT_LIMIT_MARK not in final and STREAM_CLOSED_MARK not in final


# ── 5. the model must not be told a network error happened ───────────────
def test_clean_eof_continuation_prompt_does_not_claim_a_network_error():
    _result, calls, _rows, requests = _run(_agent(), _clean_eof_stub(content="partial text so far"))
    sent_nudges = []
    for request in requests[1:]:
        for message in request.get("messages", []):
            content = message.get("content")
            if isinstance(content, str) and message.get("role") == "user" and NUDGE_MARK in content:
                sent_nudges.append(content)

    assert sent_nudges, "a delivered partial still needs a continuation nudge"
    for nudge in sent_nudges:
        assert NETWORK_ERROR_MARK not in nudge, (
            "the stream closed without any transport exception; telling the model it was a "
            "network error contradicts the user-facing copy in the same turn"
        )
        assert "output length limit" not in nudge
        assert not any(mark in nudge for mark in ABSOLUTE_CAUSE_MARKS), (
            "the nudge must describe the observed close, not assert the upstream's cause"
        )
    assert calls == 4


# ── 6. no synthetic nudge welded into the user's own message ─────────────
def test_clean_eof_turn_leaves_no_continuation_nudge_in_transcript():
    prompt = "explain the migration plan"
    result, _calls, rows, _reqs = _run(_agent(), _clean_eof_stub(), user_message=prompt)

    user_rows = [r for r in rows if r.get("role") == "user"]
    assert len(user_rows) == 1
    assert user_rows[0]["content"] == prompt, (
        "the synthetic continuation nudge was merged into the user's own message and "
        f"would be replayed on every later turn: {user_rows[0]['content']!r}"
    )
    assert not any(
        isinstance(r.get("content"), str) and NUDGE_MARK in r["content"] for r in rows
    )
    assert result["completed"] is False


# ── 7. a later turn replays the cleaned history and answers ──────────────
def test_later_turn_after_clean_eof_replays_clean_history():
    first = _agent()
    _r1, _calls, rows, _reqs = _run(first, _clean_eof_stub(), user_message="first question")

    second = _agent()
    requests = []

    def _create(**kwargs):
        requests.append(kwargs)
        return _healthy("second answer")

    second.client.chat.completions.create.side_effect = _create
    with (patch.object(second, "_persist_session"),
          patch.object(second, "_save_trajectory"),
          patch.object(second, "_cleanup_task_resources")):
        result = second.run_conversation("second question",
                                         conversation_history=[dict(r) for r in rows])

    sent = requests[0].get("messages", []) if requests else []
    user_contents = [m.get("content") for m in sent if m.get("role") == "user"]
    assert not any(isinstance(c, str) and NUDGE_MARK in c for c in user_contents), (
        f"the failed turn's history still carries a synthetic nudge: {user_contents}"
    )
    assert result["completed"] is True
    assert len(requests) == 1

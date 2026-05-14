"""Continuous high-quality compaction tests.

These tests lock the UX contract for long sessions: Hermes may prepare
compaction work before the hard threshold, but final compression may only adopt
an up-to-date GPT-quality candidate. It must never silently use a stale or
low-quality fallback summary.
"""

from types import SimpleNamespace
from unittest.mock import patch

from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX


def _response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _good_summary(label: str = "prepared") -> str:
    return f"""
## Active Task
Implement continuous high-quality compaction ({label}).

## Goal
Preserve equal-or-better GPT-quality session continuity.

## Completed Actions
1. Added candidate preparation.

## Active State
Runtime uses a validated compaction candidate.

## Blocked
None.

## Pending User Asks
None.

## Critical Context
/path/to/file.py SHA abc123 command pytest tests/agent/test_continuous_compaction.py
""".strip()


def _messages(n: int = 18):
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "initial task"},
        {"role": "assistant", "content": "ack"},
    ] + [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"middle message {i} with file /tmp/work-{i}.py and decision D-{i}",
        }
        for i in range(n)
    ] + [
        {"role": "user", "content": "latest user request must stay in tail"},
        {"role": "assistant", "content": "latest assistant response"},
    ]


def _compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        return ContextCompressor(
            model="gpt-5.5",
            provider="openai-codex",
            threshold_percent=0.70,
            protect_first_n=3,
            protect_last_n=4,
            quiet_mode=True,
        )


def test_prepare_continuous_summary_accepts_quality_gated_candidate():
    c = _compressor()
    messages = _messages()

    with patch("agent.context_compressor.call_llm", return_value=_response(_good_summary("candidate"))):
        candidate = c.prepare_continuous_summary(messages, current_tokens=55_000, synchronous=True)

    assert candidate is not None
    status = c.get_continuous_compaction_status()
    assert status["accepted"] is True
    assert status["quality_gate_status"] == "pass"
    assert status["accepted_message_count"] == len(messages)
    assert status["accepted_summary"].startswith(SUMMARY_PREFIX)


def test_prepare_continuous_summary_rejects_candidate_missing_active_task():
    c = _compressor()
    messages = _messages()
    bad = "## Goal\nA summary without the required active task section."

    with patch("agent.context_compressor.call_llm", return_value=_response(bad)):
        candidate = c.prepare_continuous_summary(messages, current_tokens=55_000, synchronous=True)

    assert candidate is None
    status = c.get_continuous_compaction_status()
    assert status["accepted"] is False
    assert status["quality_gate_status"] == "fail"
    assert "Active Task" in status["quality_gate_errors"][0]


def test_final_compression_uses_valid_prepared_candidate_without_new_llm_call():
    c = _compressor()
    messages = _messages()

    with patch("agent.context_compressor.call_llm", return_value=_response(_good_summary("prepared"))):
        assert c.prepare_continuous_summary(messages, current_tokens=55_000, synchronous=True)

    with patch("agent.context_compressor.call_llm", side_effect=AssertionError("final compaction should reuse prepared candidate")):
        compressed = c.compress(messages, current_tokens=80_000)

    joined = "\n".join(str(m.get("content", "")) for m in compressed)
    assert "continuous high-quality compaction (prepared)" in joined


def test_final_compression_rejects_stale_candidate_and_regenerates():
    c = _compressor()
    messages = _messages()

    with patch("agent.context_compressor.call_llm", return_value=_response(_good_summary("old"))):
        assert c.prepare_continuous_summary(messages, current_tokens=55_000, synchronous=True)

    changed = [m.copy() for m in messages]
    changed[6]["content"] += " -- later changed before final compaction"

    with patch("agent.context_compressor.call_llm", return_value=_response(_good_summary("fresh"))) as mock_call:
        compressed = c.compress(changed, current_tokens=80_000)

    assert mock_call.call_count == 1
    joined = "\n".join(str(m.get("content", "")) for m in compressed)
    assert "continuous high-quality compaction (fresh)" in joined
    assert "continuous high-quality compaction (old)" not in joined


def test_long_context_threshold_simulation_reuses_prepared_candidate():
    c = _compressor()
    messages = _messages(n=240)
    for msg in messages:
        msg["content"] = str(msg.get("content", "")) + (" x" * 750)

    with patch("agent.context_compressor.call_llm", return_value=_response(_good_summary("long-sim"))):
        assert c.prepare_continuous_summary(messages, current_tokens=65_000, synchronous=True)

    with patch("agent.context_compressor.call_llm", side_effect=AssertionError("final should reuse long prepared candidate")):
        compressed = c.compress(messages, current_tokens=80_000)

    joined = "\n".join(str(m.get("content", "")) for m in compressed)
    assert "continuous high-quality compaction (long-sim)" in joined
    assert len(compressed) < len(messages)

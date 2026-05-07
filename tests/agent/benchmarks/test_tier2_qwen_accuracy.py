# tests/agent/benchmarks/test_tier2_qwen_accuracy.py
"""Plant N facts in early turns, force compaction, ask the model
N follow-up questions, score retention.

Marked ``integration`` (skipped by default; opt in with ``-m integration``)
AND auto-skipped when the moe profile isn't reachable on :8085.
"""
import pytest
import socket

from agent.context_compressor import ContextCompressor
from agent.auxiliary_client import call_llm
from tests.agent.benchmarks._report import record

# Module-level marker — every test in this file is integration.
pytestmark = pytest.mark.integration


def _qwen_up() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 8085), timeout=0.5):
            return True
    except OSError:
        return False


qwen_required = pytest.mark.skipif(
    not _qwen_up(), reason="local-qwen moe profile not running on :8085",
)


# 10 facts paired with verification questions and expected substring
# matches in the model's answer (case-insensitive).
FACT_PROBES = [
    ("My favorite programming language is Rust.",
     "What is my favorite programming language?", ("rust",)),
    ("The deployment deadline is March 15, 2026.",
     "When is the deployment deadline?", ("march 15", "march 15, 2026")),
    ("The database password is stored in /etc/secrets/db.env.",
     "Where is the database password stored?", ("/etc/secrets/db.env",)),
    ("My API rate limit is 1000 requests per minute.",
     "What's my API rate limit?", ("1000", "1,000")),
    ("The project uses Python 3.13.",
     "Which Python version does the project use?", ("3.13", "python 3.13")),
    ("Production deploys go to us-east-2.",
     "Which AWS region for production?", ("us-east-2",)),
    ("Tests live under tests/ — never under src/.",
     "Where do tests live?", ("tests/", "under tests")),
    ("The CI provider is Buildkite.",
     "Which CI provider?", ("buildkite",)),
    ("Errors above 5xx page the on-call.",
     "Which errors page on-call?", ("5xx", "above 5xx", "500")),
    ("The lead reviewer is Akhil.",
     "Who is the lead reviewer?", ("akhil",)),
]


def _build_session_with_facts() -> list[dict]:
    msgs = [{"role": "system", "content": "You are a helpful assistant."}]
    # Plant facts in early turns, padded with chat that pushes toward compaction
    for fact, _, _ in FACT_PROBES:
        msgs.append({"role": "user", "content": fact})
        msgs.append({"role": "assistant", "content": "Got it."})
    # Filler turns to force compaction
    for i in range(60):
        msgs.append({"role": "user", "content": f"Context filler turn {i}: " + "x" * 1500})
        msgs.append({"role": "assistant", "content": f"Acknowledged turn {i}."})
    return msgs


def _ask_questions(messages: list[dict]) -> int:
    """For each probe, append the question, call qwen-instruct, score.

    Note: ``call_llm`` signature (verified in agent/auxiliary_client.py)
    accepts ``provider``, ``model``, ``base_url``, ``api_key``,
    ``messages``, ``temperature``, ``max_tokens``, ``tools``, ``timeout``,
    ``extra_body`` — NO ``api_mode`` kwarg. Don't add one.
    """
    score = 0
    for _, question, expected in FACT_PROBES:
        probe_msgs = messages + [{"role": "user", "content": question}]
        resp = call_llm(
            messages=probe_msgs,
            model="qwen-instruct",
            base_url="http://127.0.0.1:8085/v1",
            api_key="not-needed",
            # provider="custom" — the auxiliary_client routing keyword
            # for "use the explicit base_url + api_key passed here."
            provider="custom",
            max_tokens=400,
        )
        answer = (resp.choices[0].message.content or "").lower()
        if any(e.lower() in answer for e in expected):
            score += 1
    return score


@qwen_required
def test_2_3_fact_retention_after_compaction():
    """A/B fact retention. Both compressors compact the same fixture,
    then we ask the model 10 fact-recall questions. Acceptance: with-flags
    score >= 0.80 AND >= (baseline - 0.05)."""
    fixture = _build_session_with_facts()

    def _make(**flags):
        return ContextCompressor(
            model="qwen-instruct", threshold_percent=0.50,
            protect_first_n=3, protect_last_n=20,
            summary_target_ratio=0.20, quiet_mode=True,
            base_url="http://127.0.0.1:8085/v1", api_key="not-needed",
            # provider="custom" — auxiliary_client routing keyword;
            # see test_tier2_qwen_walltime.py for the explanation.
            provider="custom", api_mode="chat_completions",
            config_context_length=262_144, **flags,
        )

    baseline_compacted = _make().compress(fixture.copy(), current_tokens=200_000)
    candidate_compacted = _make(
        qwen_aware_enabled=True, dedup_operations=True,
        anchor_first_assistant=True, threshold_absolute_max=80_000,
        message_threshold=200, turn_threshold=30,
    ).compress(fixture.copy(), current_tokens=200_000)

    base_score = _ask_questions(baseline_compacted)
    cand_score = _ask_questions(candidate_compacted)
    record("2.3", "fact_retention_pct",
           base_score / len(FACT_PROBES) * 100,
           cand_score / len(FACT_PROBES) * 100, "%")

    assert cand_score >= 8, (
        f"Fact retention {cand_score}/10 below 0.80 floor"
    )
    assert cand_score >= base_score - 1, (
        f"Fact retention regressed: {base_score} -> {cand_score} "
        f"(allowed drop: 1 fact)"
    )

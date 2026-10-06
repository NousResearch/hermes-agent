"""Regression tests for the semantic-cascade stop-path guard (#131098 defect 2).

A model in free-association drift emits a long final reply whose tail walks an
association chain (chemistry -> thermodynamics -> ... -> socks) with every n-gram
unique, so the runaway-repetition gate (#100716) cannot see it. ``is_semantic_cascade``
catches that shape -- near-zero content-word return from head to tail -- and the
stop path in ``turn_final_response`` reuses the same ``_REPETITION_STOPPED`` verdict.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.repetition_guard import (
    CASCADE_MIN_CHARS,
    STOP_PATH_MIN_CHARS,
    is_runaway_repetition,
    is_semantic_cascade,
)
from hermes_constants import FINISH_REASON_LENGTH, PARTIAL_STREAM_STUB_ID

# Coherent prefix: a correct, evidence-backed finding (~368 chars, the incident shape).
_INCIDENT_PREFIX = (
    "Our postgres query planner keeps choosing a sequential scan over the index "
    "because the table statistics undercount distinct column values, so vacuum analyze "
    "refreshes the planner cache and the checkpoint buffer absorbs the transaction load. "
    "Each database index entry maps query predicates to table rows; when the buffer cache "
    "holds the hot index pages the checkpoint cost stays low and every transaction benefits. "
)[:368]

_CORE = [
    "database",
    "index",
    "query",
    "planner",
    "cache",
    "vacuum",
    "postgres",
    "table",
    "column",
    "transaction",
    "buffer",
    "checkpoint",
]
_CHAIN = [
    "chemistry",
    "thermodynamics",
    "folding",
    "medicine",
    "pharmaceuticals",
    "biologics",
    "generics",
    "trademark",
    "copyright",
    "litigation",
    "courtroom",
    "theater",
    "costumes",
    "wardrobe",
    "hosiery",
]
_FRAMES = [
    "brings to mind",
    "gives way to thoughts of",
    "slides toward",
    "drifts into",
    "opens onto",
    "shades into",
]


def _unique_word(i: int, j: int) -> str:
    """Deterministic unique alphabetic word per (paragraph, slot); no n-gram may repeat."""
    a, letters = i * 8 + j, ""
    for _ in range(3):
        letters += chr(97 + a % 26)
        a //= 26
    return "zx" + letters + chr(97 + j)


def _topic_name(i: int) -> str:
    return "matter" + chr(97 + (i // 26) % 26) + chr(97 + i % 26)


def build_drift_chain(n_extra: int = 46) -> str:
    """SYNTHESIZED incident shape (~12.7k chars, cf. the reported 12,657): a 368-char
    coherent prefix, then a free-association walk across unrelated topics that stops
    mid-thought with a ~12.3k novel tail. Every 8-gram is unique."""
    parts, bridge = [_INCIDENT_PREFIX], "index cardinality"
    for i, topic in enumerate(list(_CHAIN) + [_topic_name(k) for k in range(n_extra)]):
        words = [_unique_word(i, j) for j in range(8)]
        frame = _FRAMES[i % len(_FRAMES)]
        parts.append(
            "Regarding %s, %s %s %s. %s %s %s %s %s %s %s %s. "
            % (
                topic,
                bridge,
                frame,
                words[0],
                words[1],
                words[2],
                words[3],
                words[4],
                words[5],
                words[6],
                words[7],
                words[0],
            )
        )
        parts.append(
            "This %s %s %s with %s beside %s, while %s settles near %s. "
            % (words[1], frame, words[2], words[3], words[4], words[5], words[6])
        )
        bridge = words[0]
    parts.append("and finally the drawer of folded socks")
    return "".join(parts)


def build_coherent_reply(n_paras: int = 90) -> str:
    """Long reply that stays on one topic: every paragraph reuses the same core
    vocabulary, so head words keep resurfacing through the middle and the end."""
    frames = [
        "Section {i}: the {a} {b} guides the {c} {d} so the {e} {f} keeps every "
        "{g} {h} fast; meanwhile {i0} and {j} tune the {k} {l} behind the {a} {c}. ",
        "Part {i}: engineers watch the {c} {d} feed the {e} {f} while the {g} {h} "
        "steadies the {k} {l}; thus {i0}, {j} and {b} keep the {a} {c} healthy. ",
        "Note {i}: when the {e} {f} fills, the {g} {h} spills into the {k} {l}; "
        "the {a} {b} then helps the {c} {d} and {i0} {j} recover. ",
    ]
    parts = [_INCIDENT_PREFIX]
    for i in range(n_paras):
        parts.append(
            frames[i % 3].format(
                i=i,
                a=_CORE[i % 12],
                b=_CORE[(i + 1) % 12],
                c=_CORE[(i + 2) % 12],
                d=_CORE[(i + 3) % 12],
                e=_CORE[(i + 4) % 12],
                f=_CORE[(i + 5) % 12],
                g=_CORE[(i + 6) % 12],
                h=_CORE[(i + 7) % 12],
                i0=_CORE[(i + 8) % 12],
                j=_CORE[(i + 9) % 12],
                k=_CORE[(i + 10) % 12],
                l=_CORE[(i + 11) % 12],
            )
        )
    return "".join(parts)


def _ngram_uniqueness(text: str, n: int) -> float:
    toks = text.split()
    grams = {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}
    return len(grams) / max(1, len(toks) - n + 1)


@pytest.fixture()
def loop_agent():
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.compression_enabled = False
        a.save_trajectories = False
        return a


def _response(
    content,
    *,
    finish_reason=FINISH_REASON_LENGTH,
    response_id=PARTIAL_STREAM_STUB_ID,
):
    from tests.agent.test_run_agent import _mock_assistant_msg

    return SimpleNamespace(
        id=response_id,
        model="test/model",
        choices=[
            SimpleNamespace(
                index=0,
                message=_mock_assistant_msg(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


def _run(agent, message):
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(message)


class TestSemanticCascadePredicate:
    def test_drift_chain_trips_guard(self):
        drift = build_drift_chain()
        # Incident shape: 12.6k-scale, well below the 16k repetition floor the old
        # shared floor would have required -- this must still trip.
        assert CASCADE_MIN_CHARS <= len(drift) < STOP_PATH_MIN_CHARS
        assert len(_INCIDENT_PREFIX) == 368
        # The incident signature: no repeated substrings for the sibling gate to see.
        assert _ngram_uniqueness(drift, 8) == 1.0
        assert is_runaway_repetition(drift) is False
        assert is_semantic_cascade(drift) is True

    def test_long_coherent_reply_not_flagged(self):
        coherent = build_coherent_reply()
        assert len(coherent) >= STOP_PATH_MIN_CHARS
        assert is_runaway_repetition(coherent) is False
        assert is_semantic_cascade(coherent) is False

    def test_batch_output_not_flagged(self):
        # Small-vocabulary templated output must fail open (signal floors).
        head = (
            "INSERT INTO warehouse_inventory (sku, bin, quantity, price) VALUES " * 40
        )[:1500]
        rows = "".join(
            "INSERT INTO warehouse_inventory (sku, bin, quantity, price) VALUES "
            "('SKU-%05d', 'BIN-%03d', %d, %d.%02d);\n"
            % (i, i % 97, i * 3 % 500, i % 90, i % 99)
            for i in range(420)
        )
        batch = head + rows
        assert len(batch) >= STOP_PATH_MIN_CHARS
        assert is_semantic_cascade(batch) is False

    def test_reply_returning_to_topic_not_flagged(self):
        coherent = build_coherent_reply()
        drift = build_drift_chain()
        text = coherent[:8000] + drift[2000:9000] + coherent[:8000]
        assert len(text) >= STOP_PATH_MIN_CHARS
        assert is_semantic_cascade(text) is False

    def test_short_drift_not_flagged(self):
        assert is_semantic_cascade(build_drift_chain()[:7000]) is False

    def test_non_string_inputs_fail_open(self):
        assert is_semantic_cascade(None) is False
        assert is_semantic_cascade(12345) is False
        assert is_semantic_cascade("") is False


class TestSemanticCascadeStopPath:
    def test_cascade_stop_response_aborts(self, loop_agent):
        echo = build_drift_chain()
        # Sub-16k incident scale: the stop path must catch this via the cascade floor.
        assert len(echo) < STOP_PATH_MIN_CHARS
        loop_agent.client.chat.completions.create.side_effect = [
            _response(echo, finish_reason="stop", response_id="completed-response")
        ]

        result = _run(loop_agent, "write me a long report")

        assert loop_agent.client.chat.completions.create.call_count == 1
        assert result["completed"] is False
        assert result["partial"] is True
        assert (result["failure_reason"], result["failure_retryable"]) == (
            "truncated",
            True,
        )
        assert "Repetition" in (result["final_response"] or "")
        assert not any(
            isinstance(m, dict) and m.get("content") == echo for m in result["messages"]
        )

    def test_coherent_stop_response_delivered(self, loop_agent):
        echo = build_coherent_reply()
        loop_agent.client.chat.completions.create.side_effect = [
            _response(echo, finish_reason="stop", response_id="completed-response")
        ]

        result = _run(loop_agent, "write me a long report")

        assert loop_agent.client.chat.completions.create.call_count == 1
        assert result["completed"] is True
        assert result["final_response"] == echo.strip()

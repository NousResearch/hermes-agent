"""Tests for epistemic-stance preservation in compaction summary prompts.

The summarizer prompt must instruct the aux model to keep uncertain claims
labelled as UNVERIFIED instead of hardening them into facts or dropping them
(arXiv:2608.06953; validated on real transcripts — see PR body). These tests
pin the behavior contract: the rule and its section reach the LLM prompt on
both the first-compaction and iterative-update paths.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _user_turns() -> list[dict]:
    return [
        {"role": "user", "content": "Investigate the flaky gateway timeout."},
        {
            "role": "assistant",
            "content": (
                "I suspect the timeout is probably caused by DNS resolution "
                "lag, though I haven't verified this yet. Digging further."
            ),
        },
    ]


@pytest.fixture()
def compressor() -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        instance = ContextCompressor(
            model="test/model",
            threshold_percent=0.50,
            protect_first_n=0,
            protect_last_n=2,
            quiet_mode=True,
        )
    instance.tail_token_budget = 80
    return instance


def _captured_prompt(compressor: ContextCompressor) -> str:
    captured: dict = {}

    def _capture(*args, **kwargs):
        messages = kwargs.get("messages")
        if messages is None:
            for a in args:
                if isinstance(a, list):
                    messages = a
                    break
        captured["prompt"] = "\n".join(
            str(m.get("content", "")) for m in (messages or []) if isinstance(m, dict)
        )
        return _response("## Goal\nStub summary.")

    with patch("agent.context_compressor.call_llm", side_effect=_capture):
        compressor._generate_summary(_user_turns())
    return captured.get("prompt", "")


def test_first_compaction_prompt_carries_stance_rule(compressor):
    prompt = _captured_prompt(compressor)
    assert "EPISTEMIC STATUS PRESERVATION" in prompt
    assert "## Unverified / Working Hypotheses" in prompt
    assert "UNVERIFIED:" in prompt


def test_iterative_update_prompt_carries_stance_rule(compressor):
    compressor._previous_summary = "## Goal\nEarlier summary."
    prompt = _captured_prompt(compressor)
    assert "EPISTEMIC STATUS PRESERVATION" in prompt
    assert "## Unverified / Working Hypotheses" in prompt


def test_stance_rule_does_not_touch_handoff_prefix(compressor):
    """The stance rule lives in the summarizer prompt only — the handoff
    prefix prepended to the persisted summary message must be unchanged, so
    resume-time prefix stripping keeps matching."""
    from agent.context_compressor import (
        _EPISTEMIC_STANCE_RULE,
        _HISTORICAL_SUMMARY_PREFIXES,
        SUMMARY_PREFIX,
    )

    assert _EPISTEMIC_STANCE_RULE not in SUMMARY_PREFIX
    for prefix in _HISTORICAL_SUMMARY_PREFIXES:
        assert _EPISTEMIC_STANCE_RULE not in prefix

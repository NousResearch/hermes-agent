"""Epistemic-stance preservation in compaction summary prompts.

The summarizer prompt must ask the aux model to keep uncertain claims labelled UNVERIFIED instead
of hardening them into facts or dropping them, on both the first-compaction and the iterative-update
paths.
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


@pytest.mark.parametrize("previous_summary", [None, "## Goal\nEarlier summary."], ids=["first", "iterative"])
def test_summary_prompt_asks_for_labelled_uncertain_claims(compressor, previous_summary):
    compressor._previous_summary = previous_summary
    prompt = _captured_prompt(compressor)
    assert "Never restate an uncertain claim" in prompt
    assert "\n## Unverified / Working Hypotheses\n" in prompt
    assert "UNVERIFIED: <claim>" in prompt

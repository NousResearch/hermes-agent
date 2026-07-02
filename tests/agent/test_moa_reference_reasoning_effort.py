"""Per-slot ``reasoning_effort`` on MoA reference advisors.

References are advisors: on reasoning models, most of an advisor's turn is
private thinking the aggregator never sees. A preset can now set
``reasoning_effort`` on a reference slot and have it forwarded to that slot's
backend via ``extra_body``, without touching the acting aggregator's own
reasoning configuration.

Covers both halves: config normalization preserves the key on slots
(``moa_config._clean_slot`` whitelists slot keys), and ``_run_reference``
forwards it to ``call_llm``.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest

from agent import moa_loop
from hermes_cli.moa_config import normalize_moa_config


# ------------------------------------------------------------ moa_config --

def _preset_with_slot(slot: Dict[str, Any]) -> Dict[str, Any]:
    return normalize_moa_config(
        {"presets": {"default": {
            "reference_models": [slot],
            "aggregator": {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
        }}}
    )["presets"]["default"]


def test_normalize_preserves_slot_reasoning_effort():
    preset = _preset_with_slot(
        {"provider": "openrouter", "model": "openai/gpt-5.5", "reasoning_effort": "Low "}
    )
    assert preset["reference_models"] == [
        {"provider": "openrouter", "model": "openai/gpt-5.5", "reasoning_effort": "low"}
    ]


def test_normalize_drops_empty_reasoning_effort():
    preset = _preset_with_slot(
        {"provider": "openrouter", "model": "openai/gpt-5.5", "reasoning_effort": "  "}
    )
    assert preset["reference_models"] == [
        {"provider": "openrouter", "model": "openai/gpt-5.5"}
    ]


def test_normalize_without_key_is_unchanged():
    preset = _preset_with_slot({"provider": "openrouter", "model": "openai/gpt-5.5"})
    assert preset["reference_models"] == [
        {"provider": "openrouter", "model": "openai/gpt-5.5"}
    ]


# -------------------------------------------------------------- moa_loop --

class _FakeResponse:
    class _Choice:
        class _Msg:
            content = "advice"
        message = _Msg()
    choices = [_Choice()]
    usage = None


def _run_reference_capturing(monkeypatch, slot):
    captured: Dict[str, Any] = {}

    def fake_call_llm(**kwargs):
        captured.update(kwargs)
        return _FakeResponse()

    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)
    # Bypass provider resolution — the slot's endpoint doesn't exist in tests.
    monkeypatch.setattr(moa_loop, "_slot_runtime",
                        lambda s: {"provider": s["provider"], "model": s["model"]})
    label, text, _usage = moa_loop._run_reference(
        slot, [{"role": "user", "content": "q"}], temperature=0.3, max_tokens=None,
    )
    assert text == "advice"
    return captured


def test_run_reference_forwards_reasoning_effort(monkeypatch):
    captured = _run_reference_capturing(
        monkeypatch,
        {"provider": "ref", "model": "gpt-oss-20b", "reasoning_effort": "low"},
    )
    assert captured["extra_body"] == {"reasoning_effort": "low"}
    assert captured["task"] == "moa_reference"


def test_run_reference_omits_extra_body_without_effort(monkeypatch):
    captured = _run_reference_capturing(
        monkeypatch, {"provider": "ref", "model": "gpt-oss-20b"},
    )
    assert captured.get("extra_body") is None

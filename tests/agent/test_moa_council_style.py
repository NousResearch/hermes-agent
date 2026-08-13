"""Tests for the MoA 'council' synthesis style (Model Council).

Inspired by Perplexity Computer's Model Council (Aug 2026): reference models
answer independently, and the aggregator acts as a council chair producing a
user-facing consensus/disagreement report instead of private guidance.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli.moa_config import (
    build_moa_turn_prompt,
    coerce_synthesis_style,
    decode_moa_turn,
    normalize_moa_config,
)


# ── coerce_synthesis_style normalization ────────────────────────────────────


def test_coerce_synthesis_style_defaults_and_tolerance():
    assert coerce_synthesis_style(None) == "guidance"
    assert coerce_synthesis_style("") == "guidance"
    assert coerce_synthesis_style("guidance") == "guidance"
    assert coerce_synthesis_style("council") == "council"
    assert coerce_synthesis_style("COUNCIL ") == "council"
    # Unknown / bad types degrade to the default (tolerant-read contract).
    assert coerce_synthesis_style("debate") == "guidance"
    assert coerce_synthesis_style(42) == "guidance"


def test_normalize_preset_carries_synthesis_style():
    cfg = normalize_moa_config(
        {"presets": {"p": {
            "reference_models": [{"provider": "openrouter", "model": "openai/gpt-5.5"}],
            "aggregator": {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
            "synthesis_style": "council",
        }}}
    )
    assert cfg["presets"]["p"]["synthesis_style"] == "council"
    # Flattened compatibility view mirrors the default preset.
    assert cfg["synthesis_style"] == "council"


def test_normalize_preset_synthesis_style_defaults_to_guidance():
    cfg = normalize_moa_config({})
    assert cfg["presets"]["default"]["synthesis_style"] == "guidance"
    assert cfg["synthesis_style"] == "guidance"


# ── one-shot marker round-trip (/council) ───────────────────────────────────


def test_council_one_shot_marker_round_trip():
    encoded = build_moa_turn_prompt(
        "should we launch in Q4 or Q1?", {}, synthesis_style="council"
    )
    prompt, config = decode_moa_turn(encoded)
    assert prompt == "should we launch in Q4 or Q1?"
    assert config is not None
    assert config["synthesis_style"] == "council"


def test_moa_one_shot_marker_keeps_guidance_default():
    encoded = build_moa_turn_prompt("hello", {})
    _prompt, config = decode_moa_turn(encoded)
    assert config is not None
    assert config["synthesis_style"] == "guidance"


# ── aggregate_moa_context council path ──────────────────────────────────────


def _response(content: str = "ok"):
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake")


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _run_aggregate(monkeypatch, synthesis_style):
    from agent.moa_loop import aggregate_moa_context

    calls: list[dict] = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _response(
            "advice" if kwargs.get("task") == "moa_reference" else "synthesis"
        )

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)

    result = aggregate_moa_context(
        user_prompt="buy or lease?",
        api_messages=[{"role": "user", "content": "buy or lease?"}],
        reference_models=[{"provider": "openrouter", "model": "openai/gpt-5.5"}],
        aggregator={"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
        synthesis_style=synthesis_style,
    )
    return result, calls


def _flatten(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") for part in content if isinstance(part, dict)
    )


def test_council_style_uses_chair_prompt_and_report_framing(hermes_home, monkeypatch):
    result, calls = _run_aggregate(monkeypatch, "council")

    agg_calls = [c for c in calls if c.get("task") == "moa_aggregator"]
    assert len(agg_calls) == 1
    synth_prompt = _flatten(agg_calls[0]["messages"][-1]["content"])
    assert "CHAIR of a model council" in synth_prompt
    assert "Council member responses" in synth_prompt

    # The returned context is a user-facing council report, not private
    # guidance.
    assert result.startswith("[Model Council report")
    assert "Chair: openrouter:anthropic/claude-opus-4.8" in result
    assert "Council members: openrouter:openai/gpt-5.5" in result


def test_guidance_style_unchanged(hermes_home, monkeypatch):
    """Default style must be byte-compatible with the classic MoA framing."""
    result, calls = _run_aggregate(monkeypatch, "guidance")

    agg_calls = [c for c in calls if c.get("task") == "moa_aggregator"]
    synth_prompt = _flatten(agg_calls[0]["messages"][-1]["content"])
    assert "aggregator in a Mixture of Agents process" in synth_prompt
    assert result.startswith("[Mixture of Agents context")
    assert "CHAIR" not in synth_prompt

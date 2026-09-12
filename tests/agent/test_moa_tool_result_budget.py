"""Regression coverage for the config-driven MoA tool-result budget."""

from types import SimpleNamespace

import pytest

from agent import moa_loop
from agent.turn_request_assembly import _append_moa_context
from hermes_cli.config import _validate_config_key
from hermes_cli.config_defaults import DEFAULT_CONFIG
from hermes_cli.moa_config import (
    _coerce_reference_tool_result_budget,
    normalize_moa_config,
    resolve_moa_preset,
)


def _tool_transcript(size: int = 50_000) -> list[dict]:
    return [
        {"role": "user", "content": "inspect"},
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "tool_call_id": "c1", "content": "X" * size},
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, 4000),
        (True, 4000),
        (0, 4000),
        (-1, 4000),
        (500, 1000),
        (12_000, 12_000),
        (99_999, 32_000),
        ("12000", 12_000),
        ("bad", 4000),
    ],
)
def test_reference_tool_result_budget_normalization(raw, expected):
    cfg = normalize_moa_config({"reference_tool_result_budget": raw})
    assert cfg["reference_tool_result_budget"] == expected
    assert cfg["presets"]["default"]["reference_tool_result_budget"] == expected


def test_default_config_declares_budget_as_known_key():
    assert DEFAULT_CONFIG["moa"]["presets"]["default"]["reference_tool_result_budget"] == 4000
    assert _validate_config_key("moa.presets.default.reference_tool_result_budget") == (True, None)


def test_config_validation_accepts_custom_preset_budget_key():
    assert _validate_config_key("moa.presets.review.reference_tool_result_budget") == (True, None)


def test_config_validation_rejects_unknown_custom_preset_field():
    is_known, suggestion = _validate_config_key(
        "moa.presets.review.reference_tool_result_budge"
    )

    assert not is_known
    assert suggestion == "moa.presets.review.reference_tool_result_budget"


@pytest.mark.parametrize(
    "raw",
    [float("inf"), float("-inf"), float("nan"), "inf", "-inf", "nan"],
    ids=["positive-infinity", "negative-infinity", "nan", "string-inf", "string-neg-inf", "string-nan"],
)
def test_non_finite_reference_tool_result_budgets_fall_back(raw):
    assert _coerce_reference_tool_result_budget(raw) == 4000


@pytest.mark.parametrize("raw", ["1e1000000", "9" * 5000], ids=["huge-exponent", "huge-integer"])
def test_huge_finite_reference_tool_result_budgets_clamp(raw):
    assert _coerce_reference_tool_result_budget(raw) == 32_000


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(12_000.9, 12_000), ("12000.9", 12_000), (999.9, 1000), (0.9, 4000)],
)
def test_fractional_reference_tool_result_budgets_truncate_toward_zero(raw, expected):
    assert _coerce_reference_tool_result_budget(raw) == expected


def test_reference_messages_honors_custom_budget():
    view = moa_loop._reference_messages(_tool_transcript(), tool_result_budget=12_000)
    rendered = "\n".join(str(message.get("content") or "") for message in view)
    assert rendered.count("X") == 12_000
    assert "chars omitted" in rendered


def test_persistent_moa_create_uses_preset_budget(monkeypatch):
    preset = resolve_moa_preset(
        {
            "presets": {
                "review": {
                    "reference_models": [],
                    "aggregator": {"provider": "openrouter", "model": "aggregator"},
                    "reference_tool_result_budget": 12_000,
                }
            },
            "default_preset": "review",
        },
        "review",
    )
    monkeypatch.setattr(moa_loop, "_resolve_preset_cached", lambda _name: (preset, {}))
    captured = {}

    facade = moa_loop.MoAChatCompletions("review")

    def fake_run_fanout(_preset, ref_messages, *_args):
        captured["messages"] = ref_messages
        return []

    monkeypatch.setattr(facade, "_run_fanout", fake_run_fanout)
    facade.prepare(_tool_transcript())

    rendered = "\n".join(str(message.get("content") or "") for message in captured["messages"])
    assert rendered.count("X") == 12_000


def test_one_shot_path_forwards_preset_budget(monkeypatch):
    captured = {}

    def fake_aggregate_moa_context(**kwargs):
        captured.update(kwargs)
        return ""

    monkeypatch.setattr(moa_loop, "aggregate_moa_context", fake_aggregate_moa_context)
    _append_moa_context(
        SimpleNamespace(),
        [{"role": "user", "content": "inspect"}],
        {
            "reference_models": [],
            "aggregator": {},
            "reference_tool_result_budget": 12_000,
        },
        "inspect",
    )

    assert captured["reference_tool_result_budget"] == 12_000

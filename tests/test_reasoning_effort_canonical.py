"""Canonical reasoning effort enum and provider wire-shape tests."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_core_reasoning_efforts_are_official_gpt55_modes_only():
    from hermes_constants import VALID_REASONING_EFFORTS

    assert VALID_REASONING_EFFORTS == ("low", "medium", "high", "extra_high", "max")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("low", "low"),
        ("LOW", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("extra_high", "extra_high"),
        ("Extra High", "extra_high"),
        ("extra-high", "extra_high"),
        ("xhigh", "extra_high"),
        ("max", "max"),
        ("minimal", "low"),
    ],
)
def test_parse_reasoning_effort_normalizes_aliases_to_canonical_values(raw, expected):
    from hermes_constants import parse_reasoning_effort

    assert parse_reasoning_effort(raw) == {"enabled": True, "effort": expected}


def test_parse_reasoning_effort_none_remains_disable_alias_not_effort_level():
    from hermes_constants import parse_reasoning_effort

    assert parse_reasoning_effort("none") == {"enabled": False}


@pytest.mark.parametrize(
    ("raw", "label"),
    [
        ("low", "Low"),
        ("medium", "Medium"),
        ("high", "High"),
        ("extra_high", "Extra High"),
        ("xhigh", "Extra High"),
        ("extra high", "Extra High"),
        ("minimal", "Low"),
    ],
)
def test_reasoning_effort_display_labels_canonical_modes(raw, label):
    from hermes_constants import reasoning_effort_display_label

    assert reasoning_effort_display_label(raw) == label


def test_codex_responses_normalizes_legacy_xhigh_to_extra_high_payload():
    from agent.transports.codex import ResponsesApiTransport

    kwargs = ResponsesApiTransport().build_kwargs(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        reasoning_config={"enabled": True, "effort": "xhigh"},
    )

    assert kwargs["reasoning"]["effort"] == "extra_high"
    assert kwargs["reasoning"]["summary"] == "auto"


def test_codex_responses_sends_extra_high_payload_for_canonical_value():
    from agent.transports.codex import ResponsesApiTransport

    kwargs = ResponsesApiTransport().build_kwargs(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        reasoning_config={"enabled": True, "effort": "extra_high"},
    )

    assert kwargs["reasoning"]["effort"] == "extra_high"


def test_codex_responses_clamps_extra_high_to_high_for_xai_responses(monkeypatch):
    from agent.transports.codex import ResponsesApiTransport

    monkeypatch.setattr(
        "agent.model_metadata.grok_supports_reasoning_effort",
        lambda _model: True,
    )

    kwargs = ResponsesApiTransport().build_kwargs(
        model="grok-5-test",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        reasoning_config={"enabled": True, "effort": "extra_high"},
        is_xai_responses=True,
    )

    assert kwargs["reasoning"] == {"effort": "high"}


def test_copilot_github_models_clamp_extra_high_to_high():
    from agent.transports import get_transport
    from providers import get_provider_profile

    kwargs = get_transport("chat_completions").build_kwargs(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        supports_reasoning=True,
        reasoning_config={"enabled": True, "effort": "extra_high"},
        provider_profile=get_provider_profile("copilot"),
    )

    assert kwargs["extra_body"]["reasoning"] == {"effort": "high"}


def test_reasoning_command_suggestions_surface_only_official_efforts():
    from hermes_cli.commands import COMMAND_REGISTRY, SUBCOMMANDS

    reasoning = next(command for command in COMMAND_REGISTRY if command.name == "reasoning")
    assert reasoning.args_hint == "[low|medium|high|extra_high|max|show|hide|full|clamp]"
    assert {"low", "medium", "high", "extra_high", "max"}.issubset(reasoning.subcommands)
    assert "minimal" not in reasoning.subcommands
    assert "xhigh" not in reasoning.subcommands
    assert "none" not in reasoning.subcommands
    assert SUBCOMMANDS["/reasoning"] == list(reasoning.subcommands)


def test_gateway_locale_reasoning_usage_does_not_advertise_legacy_efforts():
    locales_dir = Path(__file__).resolve().parents[1] / "locales"
    assert locales_dir.is_dir()
    for locale_file in locales_dir.glob("*.yaml"):
        text = locale_file.read_text(encoding="utf-8")
        assert "none, minimal, low, medium, high, xhigh" not in text, locale_file.name
        assert "<none|minimal|low|medium|high|xhigh|reset|show|hide>" not in text, locale_file.name

    en = (locales_dir / "en.yaml").read_text(encoding="utf-8")
    assert "Valid for current model: {valid_levels}" in en
    assert "/reasoning <level|reset|show|hide> [--global]" in en

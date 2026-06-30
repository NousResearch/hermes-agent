"""Canonical reasoning effort enum and provider wire-shape tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def test_core_reasoning_efforts_are_official_provider_style_modes_only():
    from hermes_constants import VALID_REASONING_EFFORTS

    assert VALID_REASONING_EFFORTS == ("minimal", "low", "medium", "high", "xhigh", "max")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("low", "low"),
        ("LOW", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("extra_high", "xhigh"),
        ("Extra High", "xhigh"),
        ("extra-high", "xhigh"),
        ("xhigh", "xhigh"),
        ("max", "max"),
        ("minimal", "minimal"),
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
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("extra_high", "xhigh"),
        ("xhigh", "xhigh"),
        ("extra high", "xhigh"),
        ("minimal", "minimal"),
        ("max", "max"),
    ],
)
def test_reasoning_effort_display_labels_are_provider_style_tags(raw, label):
    from hermes_constants import reasoning_effort_display_label

    assert reasoning_effort_display_label(raw) == label


def test_codex_responses_preserves_xhigh_wire_payload():
    from agent.transports.codex import ResponsesApiTransport

    kwargs = ResponsesApiTransport().build_kwargs(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        reasoning_config={"enabled": True, "effort": "xhigh"},
    )

    assert kwargs["reasoning"]["effort"] == "xhigh"
    assert kwargs["reasoning"]["summary"] == "auto"


def test_codex_responses_maps_legacy_extra_high_to_xhigh_wire_payload():
    from agent.transports.codex import ResponsesApiTransport

    kwargs = ResponsesApiTransport().build_kwargs(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        reasoning_config={"enabled": True, "effort": "extra_high"},
    )

    assert kwargs["reasoning"]["effort"] == "xhigh"


def test_codex_responses_maps_max_alias_to_xhigh_wire_payload():
    from agent.transports.codex import ResponsesApiTransport

    kwargs = ResponsesApiTransport().build_kwargs(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        reasoning_config={"enabled": True, "effort": "max"},
    )

    assert kwargs["reasoning"]["effort"] == "xhigh"


def test_codex_auxiliary_adapter_maps_canonical_and_max_to_xhigh(monkeypatch):
    from agent.auxiliary_client import _CodexCompletionsAdapter

    captured: list[dict] = []

    class _FakeStream:
        def close(self):
            pass

    class _FakeResponses:
        def create(self, **kwargs):
            captured.append(kwargs)
            return _FakeStream()

    class _FakeClient:
        responses = _FakeResponses()

        def close(self):
            pass

    def _fake_consume(*_args, **_kwargs):
        return SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="ok")],
                )
            ]
        )

    monkeypatch.setattr("agent.codex_runtime._consume_codex_event_stream", _fake_consume)

    adapter = _CodexCompletionsAdapter(_FakeClient(), model="gpt-5.5")
    for effort in ("xhigh", "extra_high", "Extra High", "max"):
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            timeout=30,
            extra_body={"reasoning": {"enabled": True, "effort": effort}},
        )

    assert [call["reasoning"]["effort"] for call in captured] == [
        "xhigh",
        "xhigh",
        "xhigh",
        "xhigh",
    ]
    assert all(call["reasoning"].get("summary") == "auto" for call in captured)


def test_auxiliary_gemini_reasoning_builds_model_specific_thinking_config():
    from agent.auxiliary_client import _build_call_kwargs

    flash_kwargs = _build_call_kwargs(
        "gemini",
        "gemini-3.5-flash",
        [{"role": "user", "content": "hi"}],
        extra_body={"reasoning": {"enabled": True, "effort": "minimal"}},
    )
    pro_kwargs = _build_call_kwargs(
        "gemini",
        "gemini-3.1-pro",
        [{"role": "user", "content": "hi"}],
        extra_body={"reasoning": {"enabled": True, "effort": "max"}},
    )

    assert flash_kwargs["extra_body"]["thinking_config"] == {
        "includeThoughts": True,
        "thinkingLevel": "minimal",
    }
    assert pro_kwargs["extra_body"]["thinking_config"] == {
        "includeThoughts": True,
        "thinkingLevel": "high",
    }
    assert "reasoning" not in flash_kwargs["extra_body"]
    assert "reasoning" not in pro_kwargs["extra_body"]


def test_anthropic_extra_high_maps_to_max_not_xhigh_for_modern_claude():
    from agent.anthropic_adapter import build_anthropic_kwargs

    kwargs = build_anthropic_kwargs(
        model="claude-opus-4.8",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        max_tokens=4096,
        reasoning_config={"enabled": True, "effort": "extra_high"},
    )

    assert kwargs["output_config"]["effort"] == "max"


def test_openrouter_anthropic_extra_high_uses_max_verbosity():
    from agent.transports import get_transport
    from providers import get_provider_profile

    kwargs = get_transport("chat_completions").build_kwargs(
        model="anthropic/claude-opus-4.8",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        supports_reasoning=True,
        reasoning_config={"enabled": True, "effort": "extra_high"},
        provider_profile=get_provider_profile("openrouter"),
    )

    assert kwargs["verbosity"] == "max"
    assert "reasoning" not in kwargs.get("extra_body", {})


def test_supported_anthropic_reasoning_efforts_surface_max_not_xhigh():
    from hermes_constants import reasoning_efforts_for_model

    assert reasoning_efforts_for_model("anthropic", "claude-opus-4.8") == (
        "low",
        "medium",
        "high",
        "max",
    )


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
    assert reasoning.args_hint == "[minimal|low|medium|high|xhigh|max|show|hide|full|clamp]"
    assert {"minimal", "low", "medium", "high", "xhigh", "max"}.issubset(reasoning.subcommands)
    assert "extra_high" not in reasoning.subcommands
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
    assert "Valid for current model:" in en
    assert "/reasoning <level|reset|show|hide> [--global]" in en

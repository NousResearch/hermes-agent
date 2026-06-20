import re

import pytest

from hermes_cli.clio_profile import (
    ANTHROPIC_MODEL_ENV_VAR,
    PROFILE_NAME,
    append_clio_execution_profile,
    apply_clio_anthropic_model_override,
    classify_clio_report,
    clio_profile_path,
    configured_clio_profile,
    load_clio_execution_profile,
)


PROFILE_CONFIG = {"agent": {"execution_profile": PROFILE_NAME}}


def _profile_text() -> str:
    text = load_clio_execution_profile(PROFILE_NAME)
    assert text
    return text


def test_profile_includes_hard_approval_gates():
    text = _profile_text()
    required = [
        "Real provider calls",
        "Real prompt execution",
        "Image generation",
        "Production deploy",
        "DNS changes",
        "DB migrations",
        "Billing changes",
        "Credits changes",
        "Payments changes",
        "Secrets",
        "Provider credentials",
        "Worker enablement",
        "Merge to main",
    ]
    for item in required:
        assert item in text


def test_profile_forbids_making_niko_the_operator():
    text = _profile_text()
    for item in [
        "Terminal work",
        "Sudo",
        "Docker",
        "GHCR",
        "GitHub token work",
        "Server debugging",
        "Env file editing",
        "Credentials",
        "Provider keys",
    ]:
        assert item in text
    assert "Niko is not the operator" in text


def test_profile_includes_blind_prompt_protocol_without_live_prompt_fixture():
    text = _profile_text()
    assert "Blind prompt protocol" in text
    assert "live blind prompt is chosen privately" in text
    assert "Known prompts must not become fixtures" in text
    forbidden_fixture_markers = [
        "LIVE_BLIND_PROMPT=",
        "blind_prompt =",
        "known_prompt =",
        "fixture_prompt =",
    ]
    for marker in forbidden_fixture_markers:
        assert marker not in text


def test_profile_does_not_include_provider_key_values():
    text = _profile_text()
    secret_patterns = [
        r"sk-ant-[A-Za-z0-9_-]{12,}",
        r"sk-[A-Za-z0-9_-]{20,}",
        r"ghp_[A-Za-z0-9_]{20,}",
        r"github_pat_[A-Za-z0-9_]{20,}",
        r"ANTHROPIC_API_KEY\s*=\s*\S+",
        r"OPENAI_API_KEY\s*=\s*\S+",
    ]
    for pattern in secret_patterns:
        assert not re.search(pattern, text)


def test_profile_keeps_sensitive_operations_gated():
    text = _profile_text()
    gated_terms = [
        "production deploy",
        "DNS changes",
        "DB migrations",
        "billing changes",
        "credits changes",
        "payments changes",
    ]
    lowered = text.lower()
    for term in gated_terms:
        assert term.lower() in lowered
        assert f"no {term.lower()} without explicit approval" in lowered


def test_model_config_is_env_driven_for_native_anthropic(monkeypatch):
    monkeypatch.setenv(ANTHROPIC_MODEL_ENV_VAR, "claude-fable-5")
    model, notice = apply_clio_anthropic_model_override(
        "claude-opus-4.6",
        "anthropic",
        PROFILE_CONFIG,
    )
    assert model == "claude-fable-5"
    assert notice == "Clio MVP execution profile active with Anthropic model: claude-fable-5"
    assert "API" not in notice.upper()
    assert "KEY" not in notice.upper()


def test_model_config_keeps_fallback_model_when_env_missing(monkeypatch):
    monkeypatch.delenv(ANTHROPIC_MODEL_ENV_VAR, raising=False)
    model, notice = apply_clio_anthropic_model_override(
        "claude-opus-4.6",
        "anthropic",
        PROFILE_CONFIG,
    )
    assert model == "claude-opus-4.6"
    assert notice == "Clio MVP execution profile active with Anthropic model: claude-opus-4.6"


def test_model_config_does_not_override_non_anthropic(monkeypatch):
    monkeypatch.setenv(ANTHROPIC_MODEL_ENV_VAR, "claude-fable-5")
    model, notice = apply_clio_anthropic_model_override(
        "gpt-5.5",
        "openai-codex",
        PROFILE_CONFIG,
    )
    assert model == "gpt-5.5"
    assert notice is None


def test_claude_fable_public_prompt_is_not_copied_verbatim():
    text = _profile_text()
    # The public prompt is long-form model identity lore. This scoped profile is
    # intentionally short, operational and does not adopt those identity claims.
    assert len(text) < 6000
    assert "I am Claude Fable 5" not in text
    assert "You are Claude Fable 5" not in text
    assert "elder-plinius" not in text
    assert "CL4R1T4S" not in text


@pytest.mark.parametrize("label", ["GREEN", "green", "RED", "noise"])
def test_clio_report_classifier_supports_green_red_noise(label):
    assert classify_clio_report(label) in {"GREEN", "RED", "NOISE"}


def test_clio_report_classifier_rejects_unknown_label():
    with pytest.raises(ValueError):
        classify_clio_report("YELLOW")


def test_profile_can_be_appended_without_replacing_existing_prompt(monkeypatch):
    monkeypatch.delenv("CLIO_EXECUTION_PROFILE", raising=False)
    prompt = append_clio_execution_profile("base prompt", PROFILE_CONFIG)
    assert prompt.startswith("base prompt\n\n")
    assert "Mission" in prompt
    assert "GREEN: proceed" in prompt


def test_profile_can_be_selected_by_env(monkeypatch):
    monkeypatch.setenv("CLIO_EXECUTION_PROFILE", PROFILE_NAME)
    assert configured_clio_profile({"agent": {}}) == PROFILE_NAME
    assert clio_profile_path().name == f"{PROFILE_NAME}.md"

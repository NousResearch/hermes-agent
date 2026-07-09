"""Tests for config-driven fixed model policy guards."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import yaml

from hermes_cli.fallback_config import get_fallback_chain
from hermes_cli.model_policy import (
    check_fixed_model_policy,
    normalize_model_id_for_policy,
)


_FIXED_POLICY = {
    "model_policy": {
        "fixed_model": "gpt-5.5",
        "forbid_lower_fallback": True,
    }
}


def test_normalizes_provider_prefixed_model_ids_value_free():
    assert normalize_model_id_for_policy("gpt-5.5") == "gpt-5.5"
    assert normalize_model_id_for_policy("openai/gpt-5.5") == "gpt-5.5"
    assert normalize_model_id_for_policy("openrouter/openai/gpt-5.5") == "gpt-5.5"
    assert normalize_model_id_for_policy("  OpenAI/GPT-5.5  ") == "gpt-5.5"


@pytest.mark.parametrize(
    "candidate",
    ["gpt-5.5", "openai/gpt-5.5", "openrouter/openai/gpt-5.5"],
)
def test_fixed_policy_accepts_gpt55_and_provider_prefixed_equivalents(candidate):
    check = check_fixed_model_policy(_FIXED_POLICY, candidate)
    assert check.allowed is True
    assert check.message == ""


@pytest.mark.parametrize(
    "candidate",
    ["", "gpt-5.4", "openai/gpt-5.4", "gpt-5.4-mini", "gpt-5.5-mini"],
)
def test_fixed_policy_rejects_lower_or_empty_models(candidate):
    check = check_fixed_model_policy(_FIXED_POLICY, candidate)
    assert check.allowed is False
    assert "model policy" in check.message.lower()
    assert "gpt-5.5" in check.message


def test_fallback_chain_ignores_disallowed_lower_models_when_policy_forbids_fallback():
    cfg = {
        **_FIXED_POLICY,
        "fallback_providers": [
            {"provider": "openrouter", "model": "openai/gpt-5.4-mini"},
            {"provider": "openrouter", "model": "openai/gpt-5.5"},
        ],
        "fallback_model": {"provider": "nous", "model": "gpt-5.5-mini"},
    }

    assert get_fallback_chain(cfg) == [
        {"provider": "openrouter", "model": "openai/gpt-5.5"},
    ]


def test_fallback_chain_preserves_existing_behavior_without_fixed_policy():
    cfg = {
        "fallback_providers": [
            {"provider": "openrouter", "model": "openai/gpt-5.4-mini"},
        ],
        "fallback_model": {"provider": "nous", "model": "gpt-5.5-mini"},
    }

    assert get_fallback_chain(cfg) == [
        {"provider": "openrouter", "model": "openai/gpt-5.4-mini"},
        {"provider": "nous", "model": "gpt-5.5-mini"},
    ]


def _run_switch(raw_input: str, *, policy: dict):
    from hermes_cli.model_switch import switch_model

    accepted = {
        "accepted": True,
        "persist": True,
        "recognized": True,
        "message": None,
    }
    with (
        patch("hermes_cli.config.load_config", return_value=policy),
        patch("hermes_cli.model_switch.resolve_alias", return_value=None),
        patch("hermes_cli.model_switch.list_provider_models", return_value=[]),
        patch(
            "hermes_cli.model_switch.normalize_model_for_provider",
            side_effect=lambda model, provider: model,
        ),
        patch("hermes_cli.models.validate_requested_model", return_value=accepted),
        patch("hermes_cli.models.detect_provider_for_model", return_value=None),
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "***",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_mode": "codex_responses",
            },
        ),
    ):
        return switch_model(
            raw_input=raw_input,
            current_provider="openai-codex",
            current_model="gpt-5.5",
        )


def test_model_switch_rejects_lower_model_when_fixed_policy_active():
    result = _run_switch("gpt-5.4-mini", policy=_FIXED_POLICY)

    assert result.success is False
    assert "model policy" in result.error_message.lower()
    assert "gpt-5.5" in result.error_message
    assert "gpt-5.4-mini" in result.error_message


def test_model_switch_preserves_explicit_switch_when_no_fixed_policy():
    result = _run_switch("gpt-5.4-mini", policy={})

    assert result.success is True, result.error_message
    assert result.new_model == "gpt-5.4-mini"
    assert result.target_provider == "openai-codex"


def test_config_set_model_rejects_lower_value_under_fixed_policy(tmp_path, monkeypatch, capsys):
    from hermes_cli.config import set_config_value

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({**_FIXED_POLICY, "model": {"default": "gpt-5.5"}}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as raised:
        set_config_value("model.default", "gpt-5.4")

    assert raised.value.code == 1
    captured = capsys.readouterr()
    assert "model policy" in captured.err.lower()
    assert "gpt-5.5" in captured.err
    assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["model"]["default"] == "gpt-5.5"


def test_config_set_model_preserves_existing_behavior_without_fixed_policy(tmp_path, monkeypatch):
    from hermes_cli.config import set_config_value

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"model": {"default": "gpt-5.5"}}),
        encoding="utf-8",
    )

    set_config_value("model.default", "gpt-5.4-mini")

    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert saved["model"]["default"] == "gpt-5.4-mini"

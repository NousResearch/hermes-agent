"""Repro for NousResearch/hermes-agent#102425: model_overrides never match
when provider id carries the 'custom:' prefix."""
from unittest.mock import patch

import agent.models_dev as md
from agent.models_dev import _provider_override_section, _explicit_model_override

OVERRIDES = {
    "jerrita": {
        "_default": {"supports_reasoning": True},
        "glm-5.3-flash": {"supports_reasoning": True, "context_window": 123456},
    },
}


def test_bare_name_matches():
    with patch.object(md, "_load_model_overrides", return_value=OVERRIDES):
        assert _provider_override_section("jerrita") == OVERRIDES["jerrita"]


def test_custom_prefixed_name_misses():
    with patch.object(md, "_load_model_overrides", return_value=OVERRIDES):
        assert _provider_override_section("custom:jerrita") == OVERRIDES["jerrita"], (
            "override lookup must strip the custom: prefix"
        )


def test_explicit_override_custom_prefix():
    with patch.object(md, "_load_model_overrides", return_value=OVERRIDES):
        result = _explicit_model_override("custom:jerrita", "glm-5.3-flash")
        assert result == OVERRIDES["jerrita"]["glm-5.3-flash"]

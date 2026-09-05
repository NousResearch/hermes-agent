"""T2: delegation model-profile config defaults + `hermes config check` validation.

Contracts: DEFAULT_CONFIG ships the new keys with safe (off/legacy) defaults, and
validate_config_structure surfaces malformed delegation.profiles / default_profile as issues
while staying silent for healthy and legacy configs.
"""

import pytest

from hermes_cli.config_defaults import DEFAULT_CONFIG
from hermes_cli.config import validate_config_structure


class TestDelegationProfileDefaults:
    def test_default_profile_defaults_to_legacy_behavior(self):
        assert DEFAULT_CONFIG["delegation"]["default_profile"] == ""

    def test_agent_routing_gate_defaults_off(self):
        assert DEFAULT_CONFIG["delegation"]["agent_routing"] is False

    def test_profiles_default_empty_mapping(self):
        assert DEFAULT_CONFIG["delegation"]["profiles"] == {}


def _issues(delegation):
    return validate_config_structure({"delegation": delegation})


class TestDelegationProfileValidation:
    def test_healthy_profiles_produce_no_issues(self):
        assert _issues({
            "default_profile": "small",
            "profiles": {"small": {"provider": "anthropic", "model": "claude-haiku-current"}},
        }) == []

    def test_legacy_delegation_config_produces_no_issues(self):
        assert _issues({"provider": "openrouter", "model": "x/y"}) == []

    def test_missing_delegation_section_produces_no_issues(self):
        assert validate_config_structure({}) == []

    def test_profiles_as_list_is_an_error(self):
        issues = _issues({"profiles": ["small"]})
        assert any(i.severity == "error" and "profiles" in i.message for i in issues)

    def test_profile_missing_model_is_an_error(self):
        issues = _issues({"profiles": {"small": {"provider": "anthropic"}}})
        assert any(i.severity == "error" and "small" in i.message and "model" in i.message
                   for i in issues)

    def test_profile_unknown_key_is_an_error(self):
        issues = _issues({"profiles": {"small": {"model": "m", "toolsets": ["web"]}}})
        assert any(i.severity == "error" and "toolsets" in i.message for i in issues)

    def test_default_profile_must_reference_configured_profile(self):
        issues = _issues({
            "default_profile": "huge",
            "profiles": {"small": {"provider": "anthropic", "model": "m"}},
        })
        matching = [i for i in issues if i.severity == "error" and "huge" in i.message]
        assert matching, "default_profile pointing at a missing profile must be an error"
        assert any("small" in i.message for i in matching), (
            "the error must list the configured profile names")

    @pytest.mark.parametrize("falsy", [None, "", False, 0])
    def test_falsy_default_profile_is_fine(self, falsy):
        assert _issues({
            "default_profile": falsy,
            "profiles": {"small": {"provider": "anthropic", "model": "m"}},
        }) == []

    def test_malformed_fallback_is_an_error(self):
        issues = _issues({"profiles": {"small": {"model": "m", "fallback": "openrouter/big"}}})
        assert any(i.severity == "error" and "fallback" in i.message for i in issues)


class TestAgentRoutingWithoutProfiles:
    WARNING_TEXT = "agent_routing is enabled but no delegation.profiles are configured"

    def test_agent_routing_on_with_no_profiles_warns(self):
        issues = _issues({"agent_routing": True})
        assert any(i.severity == "warning" and self.WARNING_TEXT in i.message for i in issues)

    def test_agent_routing_on_with_empty_profiles_mapping_warns(self):
        issues = _issues({"agent_routing": True, "profiles": {}, "default_profile": "x"})
        assert any(i.severity == "warning" and self.WARNING_TEXT in i.message for i in issues)

    def test_agent_routing_off_with_no_profiles_stays_silent(self):
        assert _issues({"agent_routing": False}) == []

    def test_agent_routing_on_with_profiles_stays_silent(self):
        issues = _issues({
            "agent_routing": True,
            "profiles": {"small": {"provider": "anthropic", "model": "m"}},
        })
        assert not any(self.WARNING_TEXT in i.message for i in issues)

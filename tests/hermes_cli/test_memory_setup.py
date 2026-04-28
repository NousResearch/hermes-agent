"""Tests for _get_configured_providers() and _set_configured_providers()."""
import pytest
from hermes_cli.memory_setup import _get_configured_providers, _set_configured_providers


# ── _get_configured_providers ────────────────────────────────────────────

class TestGetConfiguredProviders:
    def test_providers_list(self):
        """New list format is returned directly."""
        config = {"memory": {"providers": ["a", "b"]}}
        assert _get_configured_providers(config) == ["a", "b"]

    def test_legacy_single_provider(self):
        """Old single-provider string is wrapped in a list."""
        config = {"memory": {"provider": "a"}}
        assert _get_configured_providers(config) == ["a"]

    def test_list_wins_over_legacy(self):
        """When both providers and provider exist, the list takes precedence."""
        config = {"memory": {"providers": ["a", "b"], "provider": "c"}}
        assert _get_configured_providers(config) == ["a", "b"]

    def test_empty_config(self):
        """Empty top-level config returns empty list."""
        assert _get_configured_providers({}) == []

    def test_empty_memory_section(self):
        """Empty memory section returns empty list."""
        assert _get_configured_providers({"memory": {}}) == []

    def test_legacy_empty_string(self):
        """Legacy provider='' (empty string) should return empty list."""
        config = {"memory": {"provider": ""}}
        assert _get_configured_providers(config) == []

    def test_providers_list_with_blanks_filtered(self):
        """Blank strings in the providers list are filtered out."""
        config = {"memory": {"providers": ["a", "", "b"]}}
        assert _get_configured_providers(config) == ["a", "b"]


# ── _set_configured_providers ────────────────────────────────────────────

class TestSetConfiguredProviders:
    def test_set_multiple(self):
        """Multiple providers written to both new and legacy keys."""
        config = {}
        _set_configured_providers(config, ["a", "b"])
        assert config["memory"]["providers"] == ["a", "b"]
        assert config["memory"]["provider"] == "a"

    def test_set_empty(self):
        """Empty list clears both keys."""
        config = {}
        _set_configured_providers(config, [])
        assert config["memory"]["providers"] == []
        assert config["memory"]["provider"] == ""

    def test_set_single(self):
        """Single provider appears in both keys identically."""
        config = {}
        _set_configured_providers(config, ["x"])
        assert config["memory"]["providers"] == ["x"]
        assert config["memory"]["provider"] == "x"

    def test_creates_memory_key_if_missing(self):
        """If config has no 'memory' key, it is created."""
        config = {}
        _set_configured_providers(config, ["a"])
        assert "memory" in config
        assert config["memory"]["providers"] == ["a"]
        assert config["memory"]["provider"] == "a"

    def test_overwrites_existing_memory(self):
        """Existing memory values are overwritten, not merged."""
        config = {"memory": {"providers": ["old"], "provider": "old"}}
        _set_configured_providers(config, ["new1", "new2"])
        assert config["memory"]["providers"] == ["new1", "new2"]
        assert config["memory"]["provider"] == "new1"

    def test_does_not_mutate_input_list(self):
        """The caller's list is copied, not stored by reference."""
        names = ["a", "b"]
        config = {}
        _set_configured_providers(config, names)
        names.append("c")
        assert config["memory"]["providers"] == ["a", "b"]

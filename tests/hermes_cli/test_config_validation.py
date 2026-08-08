"""Tests for config.yaml structure validation (validate_config_structure)."""


from hermes_cli.config import (
    DEFAULT_CONFIG,
    _EXTRA_KNOWN_ROOT_KEYS,
    _KNOWN_ROOT_KEYS,
    validate_config_structure,
    ConfigIssue,
)


class TestCustomProvidersValidation:
    """custom_providers must be a YAML list, not a dict."""

    def test_dict_instead_of_list(self):
        """The exact Discord user scenario — custom_providers as flat dict."""
        issues = validate_config_structure({
            "custom_providers": {
                "name": "Generativelanguage.googleapis.com",
                "base_url": "https://generativelanguage.googleapis.com/v1beta",
                "api_key": "xxx",
                "model": "models/gemini-2.5-flash",
                "rate_limit_delay": 2.0,
                "fallback_model": {
                    "provider": "openrouter",
                    "model": "qwen/qwen3.6-plus:free",
                },
            },
            "fallback_providers": [],
        })
        errors = [i for i in issues if i.severity == "error"]
        assert any("dict" in i.message and "list" in i.message for i in errors), (
            "Should detect custom_providers as dict instead of list"
        )

    def test_dict_detects_misplaced_fields(self):
        """When custom_providers is a dict, detect fields that look misplaced."""
        issues = validate_config_structure({
            "custom_providers": {
                "name": "test",
                "base_url": "https://example.com",
                "api_key": "xxx",
            },
        })
        warnings = [i for i in issues if i.severity == "warning"]
        # Should flag base_url, api_key as looking like custom_providers entry fields
        misplaced = [i for i in warnings if "custom_providers entry fields" in i.message]
        assert len(misplaced) == 1


    def test_list_entry_not_dict(self):
        """Non-dict list entries should warn."""
        issues = validate_config_structure({
            "custom_providers": ["not-a-dict"],
            "model": {"provider": "custom"},
        })
        assert any("not a dict" in i.message for i in issues)




class TestMissingModelSection:
    """Warn when custom_providers exists but model section is missing."""


    def test_custom_providers_with_model(self):
        issues = validate_config_structure({
            "custom_providers": [
                {"name": "test", "base_url": "https://example.com/v1"},
            ],
            "model": {"provider": "custom", "default": "test-model"},
        })
        # Should not warn about missing model section
        assert not any("no 'model' section" in i.message for i in issues)


class TestConfigIssueDataclass:
    """ConfigIssue should be a proper dataclass."""

    def test_fields(self):
        issue = ConfigIssue(severity="error", message="test msg", hint="test hint")
        assert issue.severity == "error"
        assert issue.message == "test msg"
        assert issue.hint == "test hint"

    def test_equality(self):
        a = ConfigIssue("error", "msg", "hint")
        b = ConfigIssue("error", "msg", "hint")
        assert a == b


class TestUnknownTopLevelKeys:
    """Arbitrary top-level keys must NOT warn — they are bridged to os.environ.

    Top-level scalars in config.yaml are forwarded into the environment
    (gateway/run.py, hermes send) so users can feed skills and external apps
    env-style keys like DISCORD_HOME_CHANNEL or MY_APP_TOKEN. A closed-world
    allowlist can never enumerate those, so no "Unknown top-level config key"
    warning may exist.
    """


    def test_known_root_keys_derived_from_default_config(self):
        """_KNOWN_ROOT_KEYS must be DEFAULT_CONFIG.keys() plus extras — single source of truth."""
        assert set(DEFAULT_CONFIG.keys()).issubset(_KNOWN_ROOT_KEYS)
        assert _EXTRA_KNOWN_ROOT_KEYS.issubset(_KNOWN_ROOT_KEYS)
        assert _KNOWN_ROOT_KEYS == frozenset(DEFAULT_CONFIG.keys()) | _EXTRA_KNOWN_ROOT_KEYS

    def test_provider_like_unknown_root_keeps_misplaced_message(self):
        """Preserve existing base_url/api_key root-level guidance."""
        issues = validate_config_structure({
            "base_url": "https://example.com/v1",
            "api_key": "secret",
        })
        misplaced = [
            i for i in issues
            if i.severity == "warning" and "looks misplaced" in i.message
        ]
        assert any("base_url" in i.message for i in misplaced)
        assert any("api_key" in i.message for i in misplaced)


class TestFallbackProvidersValidation:
    """fallback_providers entries must be checked like fallback_model entries.

    Both keys are read through ``_iter_fallback_entries``, which drops any
    entry that is not a dict or is missing provider/model without logging.
    The legacy ``fallback_model`` key warned about that; the primary
    ``fallback_providers`` key did not, so a half-filled chain silently
    became an empty chain and failover never ran.
    """

    def _messages(self, config, severity=None):
        issues = validate_config_structure(config)
        return [i.message for i in issues if severity is None or i.severity == severity]

    def test_entry_missing_provider_warns(self):
        msgs = self._messages({"fallback_providers": [{"model": "some/model"}]}, "warning")
        assert any("fallback_providers[0]" in m and "provider" in m for m in msgs)

    def test_entry_missing_model_warns(self):
        msgs = self._messages({"fallback_providers": [{"provider": "openrouter"}]}, "warning")
        assert any("fallback_providers[0]" in m and "model" in m for m in msgs)

    def test_second_entry_is_indexed(self):
        """The index must point at the offending entry, not the first one."""
        msgs = self._messages({
            "fallback_providers": [
                {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
                {"provider": "kimi-coding"},
            ],
        }, "warning")
        assert any("fallback_providers[1]" in m and "model" in m for m in msgs)
        assert not any("fallback_providers[0]" in m for m in msgs)

    def test_non_dict_entry_errors(self):
        msgs = self._messages({"fallback_providers": ["openrouter/some-model"]}, "error")
        assert any("fallback_providers[0]" in m and "dict" in m for m in msgs)

    def test_bare_dict_is_a_valid_single_entry_chain(self):
        """_iter_fallback_entries wraps a dict, so a complete dict must not warn."""
        assert self._messages(
            {"fallback_providers": {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}}
        ) == []

    def test_bare_dict_missing_model_warns_without_an_index(self):
        msgs = self._messages(
            {"fallback_providers": {"provider": "openrouter"}}, "warning")
        assert any("fallback_providers is missing 'model'" in m for m in msgs)

    def test_complete_chain_is_silent(self):
        assert self._messages({
            "fallback_providers": [
                {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
                {"provider": "deepseek", "model": "deepseek-chat"},
            ],
        }) == []

    def test_empty_and_absent_chains_are_silent(self):
        """DEFAULT_CONFIG ships fallback_providers: [] — it must never warn."""
        assert self._messages({"fallback_providers": []}) == []
        assert self._messages({}) == []

    def test_string_chain_is_left_to_the_json_string_fix(self):
        """A JSON-encoded chain is a separate bug (#51560); do not double-report it."""
        assert self._messages(
            {"fallback_providers": '[{"provider": "openrouter", "model": "x"}]'}
        ) == []

    def test_wrong_scalar_type_errors(self):
        msgs = self._messages({"fallback_providers": 42}, "error")
        assert any("fallback_providers should be a list" in m for m in msgs)


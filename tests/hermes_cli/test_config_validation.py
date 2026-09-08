"""Tests for config.yaml structure validation (validate_config_structure)."""


from hermes_cli.config import (
    DEFAULT_CONFIG,
    _EXTRA_KNOWN_ROOT_KEYS,
    _KNOWN_ROOT_KEYS,
    _set_nested,
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


class TestVoiceSubmitModeValidation:
    def test_default_is_direct(self):
        assert DEFAULT_CONFIG["voice"]["submit_mode"] == "direct"

    def test_direct_and_draft_are_valid(self):
        for mode in ("direct", "draft"):
            issues = validate_config_structure({"voice": {"submit_mode": mode}})
            assert not any("voice.submit_mode" in issue.message for issue in issues)

    def test_invalid_mode_is_reported(self):
        issues = validate_config_structure({"voice": {"submit_mode": "refine"}})

        assert any(
            issue.severity == "error"
            and "voice.submit_mode" in issue.message
            and "direct" in issue.hint
            and "draft" in issue.hint
            for issue in issues
        )


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


class TestStringifiedContainers:
    @staticmethod
    def _quoted_string_issues(config):
        return [
            issue for issue in validate_config_structure(config)
            if "quoted string" in issue.message
        ]

    def test_container_typed_values_stored_as_strings_are_reported(self):
        cases = (
            ({"model_catalog": {"excluded_providers": '["openai-api", "copilot"]'}},
             "model_catalog.excluded_providers", "model_catalog.excluded_providers", "list"),
            ({"plugins": {"enabled": "['state', 'status']"}},
             "plugins.enabled", "plugins.enabled", "list"),
            ({"model_overrides": '{"custom": {"model": {"supports_tools": false}}}'},
             "model_overrides", "model_overrides", "mapping"),
            ({"custom_providers": [{
                "name": "local",
                "base_url": "http://localhost:8000/v1",
                "extra_body": "{temperature: 0}",
             }]}, "custom_providers[0].extra_body", "custom_providers.0.extra_body", "mapping"),
            ({"custom_providers": [{
                "name": "local",
                "base_url": "http://localhost:8000/v1",
                "extra_headers": "{X-Service-Token: secret}",
             }]}, "custom_providers[0].extra_headers", "custom_providers.0.extra_headers", "mapping"),
        )

        for config, display_path, set_path, kind in cases:
            issues = self._quoted_string_issues(config)
            assert len(issues) == 1
            assert issues[0].severity == "warning"
            assert display_path in issues[0].message
            assert kind in issues[0].message
            assert f"hermes config set {set_path}" in issues[0].hint

    def test_indexed_repair_hint_targets_the_list_member(self):
        config = {"custom_providers": [{"extra_body": "{temperature: 0}"}]}
        issue = self._quoted_string_issues(config)[0]
        set_path = issue.hint.split("hermes config set ", 1)[1].split(" ", 1)[0]

        _set_nested(config, set_path, {"temperature": 0})

        assert config["custom_providers"][0]["extra_body"] == {"temperature": 0}
        assert "custom_providers[0]" not in config

    def test_dynamic_model_override_containers_are_reported(self):
        cases = (
            ({"model_overrides": {"openai": '{gpt-5: {supports_tools: false}}'}},
             "model_overrides.openai"),
            ({"model_overrides": {"openai": {"gpt-5": "{supports_tools: false}"}}},
             "model_overrides.openai.gpt-5"),
            ({"model_overrides": {"_default": "{supports_tools: false}"}},
             "model_overrides._default"),
        )

        for config, path in cases:
            issues = self._quoted_string_issues(config)
            assert len(issues) == 1
            assert path in issues[0].message

    def test_bracket_like_model_family_string_is_not_reported(self):
        config = {
            "model_overrides": {
                "openai": {"gpt-5": {"model_family": "[reasoning]"}},
                "_default": {"model_family": "{unknown}"},
            },
        }

        assert self._quoted_string_issues(config) == []

    def test_legitimate_strings_and_real_containers_are_not_reported(self):
        config = {
            "approvals": {"mode": "[off]"},
            "model_catalog": {"excluded_providers": ["openai-api"]},
            "plugins": {"enabled": ["state", "[literal-plugin-name]"]},
            "custom_note": "[INST] not a list",
            "MY_APP_SETTING": '["foo"]',
        }

        assert self._quoted_string_issues(config) == []


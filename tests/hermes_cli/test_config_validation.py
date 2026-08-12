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


class TestShadowedBuiltinProviderEntries:
    """providers./custom_providers entries named after canonical built-in
    providers are ignored by the runtime (their base_url/api_key silently do
    nothing) — validate_config_structure must flag them (GitHub #43026)."""

    def test_providers_routing_entry_shadowing_builtin_warns(self):
        """The exact #43026 scenario — providers.gemini pointing at the
        OpenAI-compatible endpoint still hits the native Gemini API."""
        issues = validate_config_structure({
            "providers": {
                "gemini": {
                    "api_key": "AIzaSy-test",
                    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                }
            },
            "model": {"provider": "gemini", "default": "gemini-2.5-flash"},
        })
        warnings = [i for i in issues if i.severity == "warning" and "shadows" in i.message]
        assert len(warnings) == 1
        assert "gemini" in warnings[0].message
        assert "Rename" in warnings[0].hint

    def test_providers_timeout_only_entry_is_not_flagged(self):
        """Per-provider timeout tuning under a built-in id is the documented
        providers: schema (cli-config.yaml.example), not endpoint shadowing."""
        issues = validate_config_structure({
            "providers": {
                "anthropic": {
                    "request_timeout_seconds": 30,
                    "models": {"claude-opus-4.6": {"timeout_seconds": 600}},
                }
            },
            "model": {"provider": "anthropic", "default": "claude-opus-4.6"},
        })
        assert not [i for i in issues if "shadows" in i.message]

    def test_non_builtin_provider_name_is_not_flagged(self):
        issues = validate_config_structure({
            "providers": {
                "gemini-oai": {
                    "key_env": "GEMINI_API_KEY",
                    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                }
            },
            "model": {"provider": "gemini-oai", "default": "gemini-2.5-flash"},
        })
        assert not [i for i in issues if "shadows" in i.message]

    def test_custom_providers_entry_shadowing_builtin_warns(self):
        issues = validate_config_structure({
            "custom_providers": [
                {"name": "nous", "base_url": "http://localhost:1234/v1", "api_key": "k"}
            ],
            "model": {"provider": "nous", "default": "test"},
        })
        assert [i for i in issues if "shadows" in i.message and "nous" in i.message]

    def test_unreferenced_shadow_named_entry_is_not_flagged(self):
        """An entry named after a built-in but selected via the explicit
        ``custom:<name>`` menu key (model.provider stays ``custom``) is still
        honored by the runtime — don't warn about it."""
        issues = validate_config_structure({
            "custom_providers": [
                {"name": "gemini", "base_url": "https://example.com/v1"},
            ],
            "model": {"provider": "custom", "default": "test"},
        })
        assert not [i for i in issues if "shadows" in i.message]

    def test_alias_name_is_not_flagged(self):
        """Alias names (kimi -> kimi-coding) are honored as custom-provider
        names by the runtime (#15743) — they are not shadowing."""
        issues = validate_config_structure({
            "custom_providers": [
                {"name": "kimi", "base_url": "https://my-kimi.example.com/v1", "api_key": "k"}
            ],
            "model": {"provider": "kimi", "default": "test"},
        })
        assert not [i for i in issues if "shadows" in i.message]

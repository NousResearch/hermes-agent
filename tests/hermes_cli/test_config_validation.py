"""Tests for config.yaml structure validation (validate_config_structure)."""


from hermes_cli.config import validate_config_structure, ConfigIssue


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

    def test_dict_detects_nested_fallback(self):
        """When fallback_model gets swallowed into custom_providers dict."""
        issues = validate_config_structure({
            "custom_providers": {
                "name": "test",
                "fallback_model": {"provider": "openrouter", "model": "test"},
            },
        })
        errors = [i for i in issues if i.severity == "error"]
        assert any("fallback_model" in i.message and "inside" in i.message for i in errors)

    def test_valid_list_no_issues(self):
        """Properly formatted custom_providers should produce no issues."""
        issues = validate_config_structure({
            "custom_providers": [
                {"name": "gemini", "base_url": "https://example.com/v1"},
            ],
            "model": {"provider": "custom", "default": "test"},
        })
        assert len(issues) == 0

    def test_list_entry_missing_name(self):
        """List entry without name should warn."""
        issues = validate_config_structure({
            "custom_providers": [{"base_url": "https://example.com/v1"}],
            "model": {"provider": "custom"},
        })
        assert any("missing 'name'" in i.message for i in issues)

    def test_list_entry_missing_base_url(self):
        """List entry without base_url should warn."""
        issues = validate_config_structure({
            "custom_providers": [{"name": "test"}],
            "model": {"provider": "custom"},
        })
        assert any("missing 'base_url'" in i.message for i in issues)

    def test_list_entry_not_dict(self):
        """Non-dict list entries should warn."""
        issues = validate_config_structure({
            "custom_providers": ["not-a-dict"],
            "model": {"provider": "custom"},
        })
        assert any("not a dict" in i.message for i in issues)

    def test_none_custom_providers_no_issues(self):
        """No custom_providers at all should be fine."""
        issues = validate_config_structure({
            "model": {"provider": "openrouter"},
        })
        assert len(issues) == 0


class TestFallbackModelValidation:
    """fallback_model should be a top-level dict with provider + model."""

    def test_missing_provider(self):
        issues = validate_config_structure({
            "fallback_model": {"model": "anthropic/claude-sonnet-4"},
        })
        assert any("missing 'provider'" in i.message for i in issues)

    def test_missing_model(self):
        issues = validate_config_structure({
            "fallback_model": {"provider": "openrouter"},
        })
        assert any("missing 'model'" in i.message for i in issues)

    def test_valid_fallback(self):
        issues = validate_config_structure({
            "fallback_model": {
                "provider": "openrouter",
                "model": "anthropic/claude-sonnet-4",
            },
        })
        # Only fallback-related issues should be absent
        fb_issues = [i for i in issues if "fallback" in i.message.lower()]
        assert len(fb_issues) == 0

    def test_non_dict_fallback(self):
        issues = validate_config_structure({
            "fallback_model": "openrouter:anthropic/claude-sonnet-4",
        })
        assert any("should be a dict" in i.message for i in issues)

    def test_empty_fallback_dict_no_issues(self):
        """Empty fallback_model dict means disabled — no warnings needed."""
        issues = validate_config_structure({
            "fallback_model": {},
        })
        fb_issues = [i for i in issues if "fallback" in i.message.lower()]
        assert len(fb_issues) == 0

    def test_valid_fallback_list(self):
        """List-form fallback_model (chain) should validate when every entry has provider+model."""
        issues = validate_config_structure({
            "fallback_model": [
                {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
                {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            ],
        })
        fb_issues = [i for i in issues if "fallback" in i.message.lower()]
        assert len(fb_issues) == 0

    def test_fallback_list_entry_missing_provider(self):
        issues = validate_config_structure({
            "fallback_model": [
                {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
                {"model": "claude-sonnet-4-6"},
            ],
        })
        assert any("fallback_model[1]" in i.message and "provider" in i.message for i in issues)

    def test_fallback_list_entry_missing_model(self):
        issues = validate_config_structure({
            "fallback_model": [
                {"provider": "openrouter"},
            ],
        })
        assert any("fallback_model[0]" in i.message and "model" in i.message for i in issues)

    def test_fallback_list_entry_not_a_dict(self):
        issues = validate_config_structure({
            "fallback_model": ["openrouter:anthropic/claude-sonnet-4"],
        })
        assert any("fallback_model[0]" in i.message and "should be a dict" in i.message for i in issues)


class TestMissingModelSection:
    """Warn when custom_providers exists but model section is missing."""

    def test_custom_providers_without_model(self):
        issues = validate_config_structure({
            "custom_providers": [
                {"name": "test", "base_url": "https://example.com/v1"},
            ],
        })
        assert any("no 'model' section" in i.message for i in issues)

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

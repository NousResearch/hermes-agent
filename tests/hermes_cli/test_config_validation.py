"""Tests for config.yaml structure validation (validate_config_structure)."""

import pytest

from hermes_cli.config import (
    ConfigIssue,
    DEFAULT_CONFIG,
    describe_named_agent,
    get_named_agent_registry,
    render_named_agents_text,
    validate_config_structure,
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


class TestNamedAgentValidationAndStatusOutput:
    def test_named_agent_schema_fields_normalize_provider_options_permissions_and_mode_aliases(self):
        registry = get_named_agent_registry({
            "agents": {
                "oracle": {
                    "role": "researcher",
                    "category": "deep",
                    "mode": "subagent-only",
                    "providerOptions": {"reasoningEffort": "high", "apiKey": "***"},
                    "permission": {
                        "edit": "deny",
                        "bash": {"git status": "allow"},
                        "webfetch": "ask",
                        "doom_loop": "deny",
                        "external_directory": "ask",
                    },
                    "safe_claim_text": "Oracle is bounded to read-only evidence work.",
                    "ultrawork": {"model": "gpt-5.4-mini", "variant": "burst"},
                }
            }
        })

        oracle = registry["oracle"]
        assert oracle["role"] == "researcher"
        assert oracle["category"] == "deep"
        assert oracle["mode"] == "subagent-only"
        assert oracle["provider_options"] == {"reasoningEffort": "high", "apiKey": "***"}
        assert oracle["permission"] == {
            "edit": "deny",
            "bash": {"git status": "allow"},
            "webfetch": "ask",
            "doom_loop": "deny",
            "external_directory": "ask",
        }
        assert oracle["safe_claim_text"] == "Oracle is bounded to read-only evidence work."
        assert oracle["ultrawork"] == {"model": "gpt-5.4-mini", "variant": "burst"}

    @pytest.mark.parametrize(
        ("agent_entry", "expected"),
        [
            ({"mode": "parent"}, "invalid mode"),
            ({"specialist": "not_a_specialist"}, "unknown specialist"),
            ({"permission": {"edit": "maybe"}}, "permission.edit"),
            ({"providerOptions": "fast"}, "providerOptions must be a mapping"),
        ],
    )
    def test_invalid_named_agent_contract_values_produce_readable_errors(self, agent_entry, expected):
        issues = validate_config_structure({"agents": {"oracle": agent_entry}})

        assert any(
            issue.severity == "error"
            and "agents.oracle" in issue.message
            and expected in issue.message
            for issue in issues
        )

    def test_config_validation_accepts_canonical_named_agent_fields(self):
        issues = validate_config_structure({
            "agents": {
                "oracle": {
                    "role": "researcher",
                    "category": "deep",
                    "safe_claim_text": "Oracle only claims bounded research help.",
                }
            }
        })

        assert not [issue for issue in issues if issue.severity == "error"]

    def test_named_agent_registry_returns_canonical_defaults(self):
        registry = get_named_agent_registry({"agents": {}})
        oracle = registry["oracle"]

        assert oracle["name"] == "oracle"
        assert oracle["role"] == "researcher"
        assert oracle["archetype"] == "researcher"
        assert oracle["category"] == oracle["route_category"]
        assert oracle["description"]
        assert oracle["safe_claim_text"]
        assert "terminal" in oracle["blocked_tools"]

    def test_non_mapping_named_agent_config_value_produces_readable_error(self):
        issues = validate_config_structure({"agents": {"oracle": 123}})

        assert any(
            issue.severity == "error"
            and "agents.oracle" in issue.message
            and "mapping or model string" in issue.message
            for issue in issues
        )

    def test_invalid_named_agent_field_produces_readable_error(self):
        issues = validate_config_structure({
            "agents": {
                "oracle": {
                    "mode": "subagent-only",
                    "tab_cycle": True,
                }
            }
        })

        assert any(
            i.severity == "error"
            and "agents.oracle.tab_cycle" in i.message
            and "unknown field" in i.message
            and "Hermes named-agent field" in i.hint
            for i in issues
        )

    def test_render_named_agents_text_uses_honest_hermes_native_wording_and_no_secret_leak(self):
        rendered = render_named_agents_text(config={
            "agents": {
                "oracle": {
                    "mode": "subagent-only",
                    "api_key": "super-secret-api-key",
                    "providerOptions": {
                        "apiKey": "top-secret-provider-option",
                        "reasoningEffort": "high",
                    },
                    "permission": {"edit": "deny", "webfetch": "ask"},
                    "ultrawork": {"model": "gpt-5.4-mini", "variant": "burst"},
                }
            }
        })

        assert "Hermes named agents" in rendered
        assert "Hermes-native named-agent modes" in rendered
        assert "Tab cycle" not in rendered
        assert "super-secret-api-key" not in rendered
        assert "top-secret-provider-option" not in rendered

    def test_disabled_and_subagent_only_restrictions_are_listed_honestly(self):
        rendered = render_named_agents_text(config={
            "disabled_agents": ["oracle"],
            "agents": {
                "oracle": {"mode": "subagent-only"},
            },
        })
        oracle = describe_named_agent("oracle", config={
            "disabled_agents": ["oracle"],
            "agents": {
                "oracle": {"mode": "subagent-only"},
            },
        })

        assert oracle["mode"] == "subagent-only"
        assert oracle["parent_invocation"] == "blocked"
        assert oracle["is_disabled"] is True
        assert "oracle" in rendered
        assert "disabled via disabled_agents" in rendered
        assert "subagent-only; leading @ invocation allowed, default parent-agent selection blocked" in rendered

    def test_unknown_named_agent_reports_suggestion_without_name_error(self):
        try:
            describe_named_agent("oracl", config={"agents": {}})
        except KeyError as exc:
            message = str(exc)
        else:
            raise AssertionError("unknown named agent should raise KeyError")

        assert "Unknown named agent" in message
        assert "oracle" in message


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


class TestOmoParityConfigValidation:
    """Wave 8 locks Hermes-native OMO compatibility config decisions."""

    def test_default_config_exposes_safe_omo_compatibility_knobs(self):
        compat = DEFAULT_CONFIG["omo_compat"]

        assert compat["jsonc_config"] == "import-only"
        assert compat["disable_omo_env"] is False
        assert compat["hashline_edit"] is False
        assert compat["stale_edit_mode"] == "warn"
        assert compat["dynamic_context_pruning"]["protected_tools"] == []
        assert compat["named_agents"] == {
            "mode": "primary",
            "color": "auto",
            "providerOptions": {},
            "permissions": {},
        }
        assert compat["task_scheduler"]["enabled"] is False
        assert compat["runtime_modes"]["ralph"]["max_iterations"] == 100
        assert compat["mcp"]["builtins"] == "configured"

    def test_valid_omo_compat_config_has_no_issues(self):
        issues = validate_config_structure({
            "omo_compat": {
                "jsonc_config": "import-only",
                "disable_omo_env": False,
                "hashline_edit": True,
                "stale_edit_mode": "error",
                "dynamic_context_pruning": {"protected_tools": ["delegate_task", "todo"]},
                "named_agents": {
                    "mode": "subagent-only",
                    "color": "cyan",
                    "providerOptions": {"temperature": 0.2},
                    "permissions": {"edit": False, "bash": False, "webfetch": True},
                },
                "task_scheduler": {"enabled": False},
                "runtime_modes": {"ralph": {"max_iterations": 80}},
                "mcp": {"builtins": "configured"},
            }
        })
        assert issues == []

    def test_omo_compat_rejects_core_jsonc_and_bad_enums(self):
        issues = validate_config_structure({
            "omo_compat": {
                "jsonc_config": "core",
                "stale_edit_mode": "YOLO",
                "hashline_edit": "yes",
                "dynamic_context_pruning": {"protected_tools": "delegate_task"},
                "named_agents": {"mode": "superuser", "permissions": {"edit": "maybe"}},
                "task_scheduler": {"enabled": "later"},
                "runtime_modes": {"ralph": {"max_iterations": 0}},
                "mcp": {"builtins": "always-on"},
            }
        })
        errors = [i.message for i in issues if i.severity == "error"]
        assert any("jsonc_config" in msg and "import-only" in msg for msg in errors)
        assert any("stale_edit_mode" in msg for msg in errors)
        assert any("hashline_edit" in msg for msg in errors)
        assert any("protected_tools" in msg for msg in errors)
        assert any("named_agents.mode" in msg for msg in errors)
        assert any("permissions.edit" in msg for msg in errors)
        assert any("task_scheduler.enabled" in msg for msg in errors)
        assert any("ralph.max_iterations" in msg for msg in errors)
        assert any("mcp.builtins" in msg for msg in errors)

    def test_jsonc_source_format_gets_readable_error(self):
        issues = validate_config_structure({"_source_format": "jsonc"})
        errors = [i for i in issues if i.severity == "error"]
        assert any("JSONC" in i.message and "YAML" in i.hint for i in errors)

    def test_real_jsonc_config_file_gets_readable_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            '{\n  // OpenCode-style comment\n  "model": "test",\n}\n',
            encoding="utf-8",
        )
        issues = validate_config_structure()
        errors = [i for i in issues if i.severity == "error"]
        assert any("JSONC" in i.message and "YAML" in i.hint for i in errors)

    def test_ralph_max_iterations_rejects_bool(self):
        issues = validate_config_structure({
            "omo_compat": {"runtime_modes": {"ralph": {"max_iterations": True}}}
        })
        errors = [i.message for i in issues if i.severity == "error"]
        assert any("ralph.max_iterations" in msg for msg in errors)

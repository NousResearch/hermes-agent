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


class TestShadowCheckDoesNotReenterProviderResolution:
    """The shadow check must decide "is this a built-in id?" statically.

    ``auth.resolve_provider()`` already calls back into
    ``validate_config_structure()`` to build its unknown-provider hint
    (``_get_config_hint_for_unknown_provider``, upstream since dce5f51c7c,
    2026-04-05). So a shadow check that asks ``resolve_provider`` closes an
    unbounded mutual recursion:

        resolve_provider -> _get_config_hint_for_unknown_provider
          -> validate_config_structure -> find_shadowed_builtin_provider_entries
            -> resolve_provider -> ...

    Each lap re-reads and deep-copies the config, so it does not fail fast —
    it burns CPU until the CI per-file timeout SIGKILLs the process
    (tests/hermes_cli/test_aux_picker_inventory.py, 300s exceeded). These
    tests pin the seam rather than the symptom: they fail immediately instead
    of hanging the suite.
    """

    def test_shadow_scan_never_calls_resolve_provider(self, monkeypatch):
        """The edge that closes the cycle must not exist."""
        import hermes_cli.auth as auth
        from hermes_cli.config import find_shadowed_builtin_provider_entries

        class _Tripwire(BaseException):
            """Not an ``Exception`` on purpose.

            ``_canonical_shadow`` wraps its provider lookup in a broad
            ``except Exception``, which would swallow a plain assertion and
            make this test pass against the very code it is meant to catch.
            """

        def _boom(*args, **kwargs):
            raise _Tripwire

        monkeypatch.setattr(auth, "resolve_provider", _boom)

        try:
            find_shadowed_builtin_provider_entries({
                "providers": {
                    "gemini": {"base_url": "https://example.com/v1", "api_key": "k"},
                    "my-llm": {"base_url": "https://myllm.example.com/v1"},
                },
                "custom_providers": [{"name": "Legacy Box", "base_url": "https://x/v1"}],
                "model": {"provider": "gemini", "default": "m"},
            })
        except _Tripwire:
            raise AssertionError(
                "find_shadowed_builtin_provider_entries() called "
                "auth.resolve_provider(); resolve_provider() calls "
                "validate_config_structure() back, so this edge closes an "
                "unbounded recursion"
            )

    def test_static_scan_keeps_the_original_verdicts(self):
        """Same answers as the resolve_provider-based scan it replaces."""
        from hermes_cli.config import find_shadowed_builtin_provider_entries

        found = find_shadowed_builtin_provider_entries({
            "providers": {
                "gemini": {"base_url": "https://example.com/v1"},
                "openrouter": {"base_url": "https://example.com/v1"},
                "my-llm": {"base_url": "https://myllm.example.com/v1"},
                "kimi": {"base_url": "https://my-kimi.example.com/v1"},
            },
            "model": {"provider": "gemini", "default": "m"},
        })

        assert "gemini" in found, "canonical built-in id is shadowing"
        assert "openrouter" in found, (
            "openrouter is a real built-in that is deliberately kept OUT of "
            "PROVIDER_REGISTRY (auth.py: 'openrouter not in PROVIDER_REGISTRY'), "
            "so a registry-membership test alone would silently stop flagging it"
        )
        assert "my-llm" not in found, "a user's own provider id is not shadowing"
        assert "kimi" not in found, "an alias (kimi -> kimi-coding) is not shadowing"

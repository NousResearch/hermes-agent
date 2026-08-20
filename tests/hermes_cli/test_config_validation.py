"""Tests for config.yaml structure validation (validate_config_structure)."""


import pytest

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

    def test_list_entry_with_valid_base_url_no_warning(self):
        """A well-formed custom_providers entry must not trigger base_url warnings."""
        issues = validate_config_structure({
            "custom_providers": [
                {"name": "test", "base_url": "https://example.com/v1"},
            ],
            "model": {"provider": "custom", "default": "test-model"},
        })
        assert not any(
            i.severity == "warning" and "is not a valid http(s) URL" in i.message
            for i in issues
        )


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


class TestBaseUrlValidation:
    """base_url-style values across all config sections must be valid http(s) URLs.

    Config-time checks only ever emit *warnings* (never errors and never raise)
    so a malformed URL never bricks startup — the runtime validator still raises
    at client init for the same values.
    """

    def _base_url_warnings(self, config):
        """Return only the new base_url-format warnings from a config scan."""
        issues = validate_config_structure(config)
        return [
            i for i in issues
            if i.severity == "warning" and "is not a valid http(s) URL" in i.message
        ]

    # ── invalid URLs produce warnings with the right path ──────────────────

    def test_custom_providers_invalid_base_url_warns(self):
        issues = self._base_url_warnings({
            "custom_providers": [
                {"name": "bad", "base_url": "http://127.0.0.1:6153export"},
            ],
            "model": {"provider": "custom", "default": "x"},
        })
        assert any("custom_providers[0].base_url" in i.message for i in issues)

    def test_providers_dict_invalid_base_url_warns(self):
        issues = self._base_url_warnings({
            "providers": {"myprov": {"base_url": "http://bad:99999"}},
        })
        assert any("providers.myprov.base_url" in i.message for i in issues)

    def test_providers_dict_invalid_url_warns(self):
        issues = self._base_url_warnings({
            "providers": {"myprov": {"url": "http://bad:99999"}},
        })
        assert any("providers.myprov.url" in i.message for i in issues)

    def test_tts_base_url_invalid_warns(self):
        issues = self._base_url_warnings({
            "tts": {"openai": {"base_url": "http://bad:99999"}},
        })
        assert any("tts.openai.base_url" in i.message for i in issues)

    def test_stt_base_url_invalid_warns(self):
        issues = self._base_url_warnings({
            "stt": {"openai": {"base_url": "http://bad:99999"}},
        })
        assert any("stt.openai.base_url" in i.message for i in issues)

    def test_honcho_base_url_invalid_warns(self):
        issues = self._base_url_warnings({
            "honcho": {"base_url": "http://bad:99999"},
        })
        assert any("honcho.base_url" in i.message for i in issues)

    def test_auxiliary_nested_base_url_invalid_warns(self):
        issues = self._base_url_warnings({
            "auxiliary": {"vision": {"base_url": "http://bad:99999"}},
        })
        assert any("auxiliary.vision.base_url" in i.message for i in issues)

    def test_mcp_servers_invalid_url_warns(self):
        issues = self._base_url_warnings({
            "mcp_servers": {"srv": {"url": "http://x:6153export"}},
        })
        assert any("mcp_servers.srv.url" in i.message for i in issues)

    # ── valid URLs, empty strings, and lists must NOT warn ────────────────

    def test_valid_urls_produce_no_base_url_warnings(self):
        config = {
            "model": {"base_url": "https://api.openai.com/v1"},
            "custom_providers": [{"name": "x", "base_url": "https://api.openai.com/v1"}],
            "providers": {"p": {"base_url": "https://api.openai.com/v1",
                                "url": "https://api.openai.com/v1"}},
            "tts": {"openai": {"base_url": "https://api.openai.com/v1"}},
            "stt": {"openai": {"base_url": "https://api.openai.com/v1"}},
            "honcho": {"base_url": "https://api.openai.com/v1"},
            "mcp_servers": {"s": {"url": "https://api.openai.com/v1"}},
            "auxiliary": {"vision": {"base_url": "https://api.openai.com/v1"}},
        }
        assert self._base_url_warnings(config) == []

    def test_empty_base_url_no_warning(self):
        issues = self._base_url_warnings({
            "model": {"base_url": ""},
            "custom_providers": [{"name": "x", "base_url": ""}],
            "tts": {"openai": {"base_url": ""}},
        })
        assert issues == []

    def test_model_stream_only_base_urls_list_not_validated(self):
        # stream_only_base_urls is a substring *allowlist* (a LIST), not URLs.
        issues = self._base_url_warnings({
            "model": {"stream_only_base_urls": ["http://broken:6153export"]},
        })
        assert issues == []

    def test_nested_list_of_dicts_warns_with_index_path(self):
        # Lists of dicts under covered sections are descended (path gets [i]).
        issues = self._base_url_warnings({
            "tts": {"openai": [{"base_url": "http://bad:99999"}]},
        })
        assert any("tts.openai[0].base_url" in i.message for i in issues)
        # Non-dict list items still never warn.
        issues = self._base_url_warnings({
            "tts": {"openai": ["http://bad:99999"]},
        })
        assert issues == []

    def test_schemeless_and_non_http_schemes_do_not_warn(self):
        # Parity with the runtime validator: only malformed http(s) ports flag.
        issues = self._base_url_warnings({
            "providers": {
                "a": {"base_url": "file:///etc/secrets"},   # scheme not http(s)
                "b": {"url": "openai"},                      # schemeless
                "c": {"api": "sse://host"},                  # non-http scheme
            },
        })
        assert issues == []

    # ── direct API tests ─────────────────────────────────────────────────

    def test_import_validate_base_url(self):
        from hermes_cli.config import validate_base_url
        # malformed http port raises RuntimeError
        with pytest.raises(RuntimeError, match="Malformed custom endpoint URL"):
            validate_base_url("http://x:6153export")
        # valid values do not raise
        validate_base_url("")
        validate_base_url("acp://whatever")
        validate_base_url("https://api.openai.com/v1")
        validate_base_url("http://127.0.0.1:6153/v1")

    def test_iter_base_url_values_paths(self):
        from hermes_cli.config import iter_base_url_values
        config = {
            "model": {"base_url": "https://api.openai.com/v1"},
            "custom_providers": [{"name": "x", "base_url": "https://api.openai.com/v1"}],
            "providers": {"p": {"base_url": "https://api.openai.com/v1"}},
            "tts": {"openai": {"base_url": "https://api.openai.com/v1"}},
            "honcho": {"base_url": "https://api.openai.com/v1"},
            "mcp_servers": {"s": {"url": "https://api.openai.com/v1"}},
        }
        result = list(iter_base_url_values(config))
        expected = [
            ("model.base_url", "https://api.openai.com/v1"),
            ("custom_providers[0].base_url", "https://api.openai.com/v1"),
            ("providers.p.base_url", "https://api.openai.com/v1"),
            ("tts.openai.base_url", "https://api.openai.com/v1"),
            ("honcho.base_url", "https://api.openai.com/v1"),
            ("mcp_servers.s.url", "https://api.openai.com/v1"),
        ]
        # exact set of (path, value) tuples, order-independent
        assert sorted(result) == sorted(expected)
        # values are stripped
        assert all(v == v.strip() for _, v in result)

    def test_iter_base_url_values_skips_empty_and_non_string(self):
        from hermes_cli.config import iter_base_url_values
        config = {
            "model": {"base_url": ""},                          # empty — skipped
            "custom_providers": [{"name": "x", "base_url": None}],  # non-string
            "providers": {"p": {"base_url": 8080}},            # non-string — skipped
            "auxiliary": {
                "vision": {"base_url": "   "},                  # whitespace-only — skipped
                "stream_only_base_urls": ["http://broken:6153export"],  # list — skipped
            },
        }
        assert list(iter_base_url_values(config)) == []

    # ── warning hints are section-aware ───────────────────────────────────

    def _base_url_warning_hints(self, config):
        """Return (path, hint) pairs for the base_url-format warnings."""
        issues = validate_config_structure(config)
        return [
            (i.message.split(" is not a valid http(s) URL", 1)[0], i.hint)
            for i in issues
            if i.severity == "warning" and "is not a valid http(s) URL" in i.message
        ]

    def test_mcp_server_warning_hint_names_hermes_mcp(self):
        hints = self._base_url_warning_hints({
            "mcp_servers": {"srv": {"url": "http://x:6153export"}},
        })
        assert hints == [("mcp_servers.srv.url", (
            "Re-add the server with `hermes mcp add <name> --url <valid URL>`, "
            "or fix the URL in config.yaml under `mcp_servers`."
        ))]

    def test_stt_tts_honcho_warning_hints_point_to_config_yaml(self):
        hints = dict(self._base_url_warning_hints({
            "stt": {"openai": {"base_url": "http://bad:99999"}},
            "tts": {"openai": {"base_url": "http://bad:99999"}},
            "honcho": {"base_url": "http://bad:99999"},
        }))
        assert hints["stt.openai.base_url"] == "Fix the URL in config.yaml under `stt`."
        assert hints["tts.openai.base_url"] == "Fix the URL in config.yaml under `tts`."
        assert hints["honcho.base_url"] == "Fix the URL in config.yaml under `honcho`."

    def test_model_path_warning_hints_keep_setup_guidance(self):
        hints = dict(self._base_url_warning_hints({
            "custom_providers": [{"name": "bad", "base_url": "http://x:6153export"}],
            "providers": {"myprov": {"base_url": "http://bad:99999"}},
            "model": {"base_url": "http://bad:99999"},
        }))
        expected = "Run `hermes setup` or `hermes model` and enter a valid http(s) base URL."
        assert hints["custom_providers[0].base_url"] == expected
        assert hints["providers.myprov.base_url"] == expected
        assert hints["model.base_url"] == expected

"""Tests for config.yaml structure validation (validate_config_structure)."""


import pytest

from hermes_cli.config import (
    validate_config_structure,
)
from hermes_cli.config_providers import get_compatible_custom_providers
from hermes_cli.fallback_config import _iter_fallback_entries


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

    def test_scalar_is_an_error_naming_key_and_type(self):
        """A non-list scalar (a bad `config set`) makes every endpoint vanish — name the key and the type."""
        issues = validate_config_structure({"custom_providers": "oops", "model": {"provider": "openrouter"}})
        errors = [i.message for i in issues if i.severity == "error"]
        assert any(m.startswith("custom_providers is a str") and "list" in m for m in errors), errors
        assert not [i for i in validate_config_structure(
            {"custom_providers": [{"name": "x", "base_url": "http://h/v1"}], "model": {"provider": "custom"}})
            if i.severity == "error"]

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

    @pytest.mark.parametrize("field,value", [
        (field, value) for field, valid in (("name", "y"), ("base_url", "http://h2/v1"))
        for value in (None, "", "   ", "\t", valid)])
    def test_blank_name_or_base_url_warns_iff_runtime_drops_it(self, field, value):
        """The runtime strips name/base_url and drops a blank entry; the validator must agree."""
        entry = {"name": "x", "base_url": "http://h/v1", field: value}
        issues = validate_config_structure({"custom_providers": [entry], "model": {"provider": "custom"}})
        warned = any(i.message.startswith(f"custom_providers[0] is missing '{field}'") for i in issues)
        assert warned == (not get_compatible_custom_providers({"custom_providers": [entry]}))




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




class TestVoiceSubmitModeValidation:

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


def _has_tz_database() -> bool:
    try:
        import zoneinfo
        zoneinfo.ZoneInfo("UTC")
        return True
    except Exception:
        return False


def _tz_issues(config):
    return [i for i in validate_config_structure(config) if "timezone" in i.message]


class TestTimezoneValidation:
    """An invalid ``timezone`` silently puts the agent clock and every cron
    schedule on server-local time (hermes_time._get_zoneinfo falls back with
    one log warning). validate_config_structure must report it (#111725)."""

    @pytest.mark.skipif(not _has_tz_database(), reason="no tz database in this interpreter")
    def test_invalid_or_non_string_zone_is_an_error(self):
        [issue] = _tz_issues({"timezone": "Asia/Tokio", "model": {"provider": "nous"}})
        assert issue.severity == "error"
        assert "Asia/Tokio" in issue.message
        assert "IANA" in issue.hint and "HERMES_TIMEZONE" in issue.hint
        [issue] = _tz_issues({"timezone": 9, "model": {"provider": "nous"}})
        assert issue.severity == "error" and "string" in issue.message

    def test_valid_blank_missing_or_unverifiable_zone_is_silent(self, monkeypatch):
        for cfg in ({"timezone": "Asia/Tokyo"}, {}, {"timezone": ""}, {"timezone": "   "}, {"timezone": None}):
            assert _tz_issues({**cfg, "model": {"provider": "nous"}}) == []
        # Bare Windows without tzdata: ZoneInfo cannot load anything, including UTC.
        # A name that cannot be checked must not be flagged.
        import zoneinfo

        def no_db(_key):
            raise zoneinfo.ZoneInfoNotFoundError("no tz database")

        monkeypatch.setattr(zoneinfo, "ZoneInfo", no_db)
        assert _tz_issues({"timezone": "Asia/Tokio", "model": {"provider": "nous"}}) == []


class TestUnknownTopLevelKeys:
    """Arbitrary top-level keys must NOT warn — they are bridged to os.environ.

    Top-level scalars in config.yaml are forwarded into the environment
    (gateway/run.py, hermes send) so users can feed skills and external apps
    env-style keys like DISCORD_HOME_CHANNEL or MY_APP_TOKEN. A closed-world
    allowlist can never enumerate those, so no "Unknown top-level config key"
    warning may exist.
    """



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



class TestQuotedContainerValues:
    """A list/mapping slot holding one quoted string is ignored by every reader (#83308, #105706)."""

    def test_quoted_list_in_container_slot_is_flagged_with_remedy(self):
        issues = validate_config_structure({
            "plugins": {"enabled": '["a","b"]'},
            "model_catalog": {"excluded_providers": '["openai-api"]'},
        })
        flagged = {i.message.split(" ", 1)[0]: i for i in issues if "quoted string" in i.message}
        assert set(flagged) == {"plugins.enabled", "model_catalog.excluded_providers"}
        assert "hermes config set plugins.enabled '[\"a\",\"b\"]'" in flagged["plugins.enabled"].hint

    def test_string_typed_and_tolerant_slots_are_not_flagged(self):
        """`approvals.mode` is a string in the schema; `model: name` is the documented shorthand;
        `agent.disabled_toolsets` readers parse the quoted form themselves."""
        issues = validate_config_structure({
            "approvals": {"mode": "[off]"},
            "model": "gpt-4o",
            "agent": {"disabled_toolsets": '["web"]'},
            "plugins": {"enabled": ["a"]},
        })
        assert not [i for i in issues if "quoted string" in i.message]


_FIELD_VALUES = [None, "", "   ", "\t", 0, "x"]
_GOOD = {"provider": "ok", "model": "ok"}
_FALLBACK_ENTRIES = (
    [{"provider": v, "model": "m"} for v in _FIELD_VALUES]
    + [{"provider": "p", "model": v} for v in _FIELD_VALUES]
    + ["not-a-dict", 42, {}, {"provider": "p"}, {"model": "m"},
       {"provider": "p", "model": "m", "base_url": "http://h/v1", "extra": 1}]
)
_FALLBACK_CASES = (
    [(key, [_GOOD, entry]) for key in ("fallback_providers", "fallback_model") for entry in _FALLBACK_ENTRIES]
    + [(key, entry) for key in ("fallback_providers", "fallback_model")
       for entry in _FALLBACK_ENTRIES if isinstance(entry, dict) and entry]
    # Whole values of the modern key: an explicit "none" is silent, anything else unreadable is one finding.
    + [("fallback_providers", v) for v in ([], {}, "", 42, "openrouter",
                                           '[{"provider": "openrouter", "model": "x"}]')]
)


class TestFallbackChainValidation:
    """``_iter_fallback_entries`` silently drops malformed entries, so a half-filled chain
    becomes empty and failover never engages. Both keys must be flagged exactly where it drops."""

    @pytest.mark.parametrize("key,value", _FALLBACK_CASES, ids=repr)
    def test_validator_flags_exactly_what_the_parser_drops(self, key, value):
        messages = [i.message for i in validate_config_structure({key: value})]
        if isinstance(value, list):
            for i, entry in enumerate(value):
                dropped = not _iter_fallback_entries([entry])
                assert any(m.startswith(f"{key}[{i}] ") for m in messages) == dropped, messages
        elif isinstance(value, dict) and value:
            dropped = not _iter_fallback_entries([value])
            assert any(m.startswith(f"{key} is missing") for m in messages) == dropped, messages
            assert not any(f"{key}[" in m for m in messages), messages
        else:
            findings = [m for m in messages if m.startswith(key)]
            assert len(findings) == (0 if value in ("", {}) else 1), messages

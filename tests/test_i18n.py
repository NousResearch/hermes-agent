"""Tests for agent/i18n.py — language resolution, catalog loading, and translation."""

import os
import pytest
from unittest import mock

from agent.i18n import (
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
    _normalize_lang,
    _flatten_into,
    _load_catalog,
    _config_language_cached,
    reset_language_cache,
    get_language,
    t,
    _LANGUAGE_ALIASES,
)


# ── _normalize_lang ────────────────────────────────────────────────────────────

class TestNormalizeLang:
    """Language code normalization — aliases, case, regions, fallbacks."""

    def test_supported_codes_pass_through(self):
        """Every supported language code should return itself."""
        for code in SUPPORTED_LANGUAGES:
            assert _normalize_lang(code) == code

    def test_case_insensitive(self):
        """Uppercase and mixed-case codes normalize correctly."""
        assert _normalize_lang("EN") == "en"
        assert _normalize_lang("Zh") == "zh"
        assert _normalize_lang("JA") == "ja"
        assert _normalize_lang("De") == "de"

    def test_strips_whitespace(self):
        """Leading/trailing whitespace is stripped."""
        assert _normalize_lang("  en  ") == "en"
        assert _normalize_lang("\tzh\n") == "zh"

    def test_empty_string_falls_back(self):
        """Empty string returns the default language."""
        assert _normalize_lang("") == DEFAULT_LANGUAGE
        assert _normalize_lang("   ") == DEFAULT_LANGUAGE

    def test_non_string_falls_back(self):
        """Non-string inputs (None, int, bool) return the default language."""
        assert _normalize_lang(None) == DEFAULT_LANGUAGE
        assert _normalize_lang(42) == DEFAULT_LANGUAGE
        assert _normalize_lang(True) == DEFAULT_LANGUAGE

    def test_unknown_code_falls_back(self):
        """Totally unknown language codes return the default."""
        assert _normalize_lang("xx") == DEFAULT_LANGUAGE
        assert _normalize_lang("klingon") == DEFAULT_LANGUAGE
        assert _normalize_lang("nope-not-real") == DEFAULT_LANGUAGE

    def test_region_stripping(self):
        """Codes like zh-CN should strip the region and fall through to the base."""
        # zh-CN -> strips to "zh" which IS in SUPPORTED_LANGUAGES
        assert _normalize_lang("zh-CN") == "zh"
        assert _normalize_lang("zh-cn") == "zh"  # lowercase variant
        assert _normalize_lang("ja-JP") == "ja"
        assert _normalize_lang("de-DE") == "de"
        # pt-BR -> strips to "pt" which IS in SUPPORTED_LANGUAGES
        assert _normalize_lang("pt-BR") == "pt"
        # A code whose base is not supported should fall back
        assert _normalize_lang("xx-YY") == DEFAULT_LANGUAGE

    # ── Alias tests ─────────────────────────────────────────────────────────

    def test_chinese_aliases(self):
        """Common Chinese language aliases resolve to zh."""
        for alias in ("chinese", "mandarin", "zh-cn", "zh-hans", "zh-sg",
                       "Chinese", "CHINESE"):
            assert _normalize_lang(alias) == "zh"

    def test_traditional_chinese_aliases(self):
        """Traditional Chinese aliases resolve to zh-hant."""
        for alias in ("traditional-chinese", "traditional_chinese",
                       "zh-tw", "zh-hk", "zh-mo"):
            assert _normalize_lang(alias) == "zh-hant"

    def test_japanese_aliases(self):
        """Japanese aliases resolve to ja."""
        for alias in ("japanese", "jp", "ja-jp"):
            assert _normalize_lang(alias) == "ja"

    def test_german_aliases(self):
        """German aliases resolve to de."""
        for alias in ("german", "deutsch", "de-de", "de-at", "de-ch"):
            assert _normalize_lang(alias) == "de"

    def test_spanish_aliases(self):
        """Spanish aliases resolve to es."""
        for alias in ("spanish", "español", "espanol", "es-es", "es-mx", "es-ar"):
            assert _normalize_lang(alias) == "es"

    def test_french_aliases(self):
        """French aliases resolve to fr."""
        for alias in ("french", "français", "france", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
            assert _normalize_lang(alias) == "fr"

    def test_ukrainian_aliases(self):
        """Ukrainian aliases resolve to uk."""
        for alias in ("ukrainian", "ukrainisch", "українська", "uk-ua", "ua"):
            assert _normalize_lang(alias) == "uk"

    def test_turkish_aliases(self):
        """Turkish aliases resolve to tr."""
        for alias in ("turkish", "türkçe", "tr-tr"):
            assert _normalize_lang(alias) == "tr"

    def test_korean_aliases(self):
        """Korean aliases resolve to ko."""
        for alias in ("korean", "한국어", "ko-kr"):
            assert _normalize_lang(alias) == "ko"

    def test_italian_aliases(self):
        """Italian aliases resolve to it."""
        for alias in ("italian", "italiano", "it-it", "it-ch"):
            assert _normalize_lang(alias) == "it"

    def test_irish_aliases(self):
        """Irish aliases resolve to ga."""
        for alias in ("irish", "gaeilge", "ga-ie"):
            assert _normalize_lang(alias) == "ga"

    def test_portuguese_aliases(self):
        """Portuguese aliases resolve to pt."""
        for alias in ("portuguese", "português", "portugues",
                       "pt-pt", "pt-br", "brazilian", "brasileiro"):
            assert _normalize_lang(alias) == "pt"

    def test_russian_aliases(self):
        """Russian aliases resolve to ru."""
        for alias in ("russian", "русский", "ru-ru"):
            assert _normalize_lang(alias) == "ru"

    def test_hungarian_aliases(self):
        """Hungarian aliases resolve to hu."""
        for alias in ("hungarian", "magyar", "hu-hu"):
            assert _normalize_lang(alias) == "hu"

    def test_afrikaans_aliases(self):
        """Afrikaans aliases resolve to af."""
        for alias in ("afrikaans", "af-za"):
            assert _normalize_lang(alias) == "af"

    def test_alias_beats_region_strip(self):
        """Alias lookup takes priority over region-stripping."""
        # "zh-tw" is an explicit alias for zh-hant (Traditional Chinese)
        # Without the alias, stripping would give "zh" (Simplified)
        assert _normalize_lang("zh-tw") == "zh-hant"
        assert _normalize_lang("zh-tw") != "zh"


# ── _flatten_into ──────────────────────────────────────────────────────────────

class TestFlattenInto:
    """YAML catalog flattening — nested dicts to dotted keys."""

    def test_flat_dict(self):
        """A flat dict with string values produces dotted keys."""
        out: dict[str, str] = {}
        _flatten_into({"hello": "world", "foo": "bar"}, "", out)
        assert out == {"hello": "world", "foo": "bar"}

    def test_nested_dict(self):
        """Nested dicts produce dotted-path keys."""
        out: dict[str, str] = {}
        node = {
            "approval": {
                "choose": "Choose:",
                "yes": "Yes",
            },
            "gateway": {
                "draining": "Draining {count}...",
            },
        }
        _flatten_into(node, "", out)
        assert out == {
            "approval.choose": "Choose:",
            "approval.yes": "Yes",
            "gateway.draining": "Draining {count}...",
        }

    def test_deeply_nested(self):
        """Three levels of nesting produce three-part dotted keys."""
        out: dict[str, str] = {}
        _flatten_into({"a": {"b": {"c": "deep"}}}, "", out)
        assert out == {"a.b.c": "deep"}

    def test_prefix_is_respected(self):
        """When prefix is provided, keys are built on top of it."""
        out: dict[str, str] = {}
        _flatten_into({"name": "Hermes"}, "app", out)
        assert out == {"app.name": "Hermes"}

    def test_non_string_leaves_skipped(self):
        """Integer, float, list, and None leaves are silently ignored."""
        out: dict[str, str] = {}
        _flatten_into({
            "num": 42,
            "flag": True,
            "items": [1, 2, 3],
            "nothing": None,
            "text": "kept",
        }, "", out)
        assert out == {"text": "kept"}

    def test_empty_dict(self):
        """Empty dict produces empty output."""
        out: dict[str, str] = {}
        _flatten_into({}, "", out)
        assert out == {}

    def test_mixed_nesting_with_skips(self):
        """Nested structure where some leaves are strings and others are skipped."""
        out: dict[str, str] = {}
        _flatten_into({
            "section": {
                "title": "Settings",
                "count": 5,
                "subsection": {
                    "label": "Advanced",
                },
            },
        }, "", out)
        assert out == {
            "section.title": "Settings",
            "section.subsection.label": "Advanced",
        }


# ── t() translation function ───────────────────────────────────────────────────

class TestTranslation:
    """The main t() function — catalog lookup, fallback, and formatting."""

    @mock.patch("agent.i18n._load_catalog")
    @mock.patch("agent.i18n.get_language", return_value="en")
    def test_simple_lookup(self, mock_get_lang, mock_load):
        """t() looks up a key in the active language catalog."""
        mock_load.return_value = {"greeting.hello": "Hello!"}
        assert t("greeting.hello") == "Hello!"

    @mock.patch("agent.i18n._load_catalog")
    @mock.patch("agent.i18n.get_language", return_value="en")
    def test_missing_key_returns_key(self, mock_get_lang, mock_load):
        """When a key is missing from both target and English, the key is returned."""
        mock_load.return_value = {}
        assert t("missing.key") == "missing.key"

    @mock.patch("agent.i18n._load_catalog")
    @mock.patch("agent.i18n.get_language", return_value="zh")
    def test_falls_back_to_english(self, mock_get_lang, mock_load):
        """When a key is missing in the target lang, English is tried."""
        def catalog_side_effect(lang):
            if lang == "zh":
                return {}  # zh catalog is empty
            if lang == "en":
                return {"hello": "Hello English"}
            return {}
        mock_load.side_effect = catalog_side_effect
        assert t("hello") == "Hello English"

    @mock.patch("agent.i18n._load_catalog")
    @mock.patch("agent.i18n.get_language", return_value="de")
    def test_target_language_wins_over_english(self, mock_get_lang, mock_load):
        """When both target and English have the key, target wins."""
        def catalog_side_effect(lang):
            if lang == "de":
                return {"hello": "Hallo"}
            if lang == "en":
                return {"hello": "Hello"}
            return {}
        mock_load.side_effect = catalog_side_effect
        assert t("hello") == "Hallo"

    @mock.patch("agent.i18n._load_catalog")
    @mock.patch("agent.i18n.get_language", return_value="en")
    def test_format_kwargs(self, mock_get_lang, mock_load):
        """str.format() substitutions are applied."""
        mock_load.return_value = {"count": "You have {n} items"}
        assert t("count", n=5) == "You have 5 items"

    @mock.patch("agent.i18n._load_catalog")
    @mock.patch("agent.i18n.get_language", return_value="en")
    def test_format_kwargs_failure_returns_raw(self, mock_get_lang, mock_load):
        """If format() raises, the raw template is returned (no crash)."""
        mock_load.return_value = {"broken": "Hello {missing}"}
        # KeyError on format — should return the raw value
        result = t("broken")
        assert result == "Hello {missing}"

    @mock.patch("agent.i18n._load_catalog")
    @mock.patch("agent.i18n.get_language", return_value="en")
    def test_explicit_lang_override(self, mock_get_lang, mock_load):
        """Explicit lang= parameter overrides the active language."""
        def catalog_side_effect(lang):
            if lang == "ja":
                return {"hello": "こんにちは"}
            return {}
        mock_load.side_effect = catalog_side_effect
        assert t("hello", lang="ja") == "こんにちは"

    @mock.patch("agent.i18n._load_catalog")
    @mock.patch("agent.i18n.get_language", return_value="en")
    def test_explicit_lang_falls_back_to_english(self, mock_get_lang, mock_load):
        """Even with explicit lang, missing keys fall back to English."""
        def catalog_side_effect(lang):
            if lang == "ja":
                return {}
            if lang == "en":
                return {"hello": "Hello"}
            return {}
        mock_load.side_effect = catalog_side_effect
        assert t("hello", lang="ja") == "Hello"

    @mock.patch("agent.i18n._load_catalog")
    @mock.patch("agent.i18n.get_language", return_value="en")
    def test_format_with_multiple_kwargs(self, mock_get_lang, mock_load):
        """Multiple format kwargs are all substituted."""
        mock_load.return_value = {
            "greeting": "Hello {name}, you have {count} messages"
        }
        assert t("greeting", name="Ned", count=3) == "Hello Ned, you have 3 messages"


# ── get_language ───────────────────────────────────────────────────────────────

class TestGetLanguage:
    """Language resolution: env var > config > default."""

    def setup_method(self):
        """Clear caches before each test so config reads don't leak between tests."""
        reset_language_cache()

    def test_default_is_english(self):
        """With no env var and no config, returns 'en'."""
        with mock.patch.dict(os.environ, {}, clear=True):
            # Ensure HERMES_LANGUAGE is not set
            os.environ.pop("HERMES_LANGUAGE", None)
            with mock.patch("agent.i18n._config_language_cached", return_value=None):
                assert get_language() == "en"

    def test_env_var_takes_priority(self):
        """HERMES_LANGUAGE env var wins over everything."""
        with mock.patch.dict(os.environ, {"HERMES_LANGUAGE": "ja"}):
            assert get_language() == "ja"

    @mock.patch("agent.i18n._config_language_cached")
    def test_config_when_no_env(self, mock_config):
        """Without env var, config value is used."""
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("HERMES_LANGUAGE", None)
            mock_config.return_value = "de"
            assert get_language() == "de"

    @mock.patch("agent.i18n._config_language_cached")
    def test_env_beats_config(self, mock_config):
        """When both env and config are set, env wins."""
        with mock.patch.dict(os.environ, {"HERMES_LANGUAGE": "fr"}):
            mock_config.return_value = "de"
            assert get_language() == "fr"

    def test_env_var_is_normalized(self):
        """HERMES_LANGUAGE values go through _normalize_lang."""
        with mock.patch.dict(os.environ, {"HERMES_LANGUAGE": "chinese"}):
            assert get_language() == "zh"


# ── reset_language_cache ───────────────────────────────────────────────────────

class TestResetLanguageCache:
    """Cache invalidation clears both the config cache and catalog cache."""

    def test_clears_config_cache(self):
        """reset_language_cache() clears the lru_cache on _config_language_cached."""
        # Fill the cache
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("HERMES_LANGUAGE", None)
            with mock.patch("agent.i18n._config_language_cached",
                            return_value="de"):
                assert get_language() == "de"

        reset_language_cache()

        # After reset, should re-read
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("HERMES_LANGUAGE", None)
            with mock.patch("agent.i18n._config_language_cached",
                            return_value="ja"):
                assert get_language() == "ja"


# ── SUPPORTED_LANGUAGES constant ───────────────────────────────────────────────

class TestSupportedLanguages:
    """The SUPPORTED_LANGUAGES tuple covers expected languages."""

    def test_contains_baseline_languages(self):
        """Core languages are present."""
        assert "en" in SUPPORTED_LANGUAGES
        assert "zh" in SUPPORTED_LANGUAGES
        assert "ja" in SUPPORTED_LANGUAGES
        assert "de" in SUPPORTED_LANGUAGES
        assert "es" in SUPPORTED_LANGUAGES
        assert "fr" in SUPPORTED_LANGUAGES

    def test_default_language_is_in_list(self):
        """DEFAULT_LANGUAGE must be in SUPPORTED_LANGUAGES."""
        assert DEFAULT_LANGUAGE in SUPPORTED_LANGUAGES


# ── Alias integrity ─────────────────────────────────────────────────────────────

class TestAliasIntegrity:
    """Every alias must resolve to a supported language code."""

    def test_all_aliases_point_to_supported(self):
        """No alias should point to an unsupported language."""
        for alias, target in _LANGUAGE_ALIASES.items():
            assert target in SUPPORTED_LANGUAGES, (
                f"Alias '{alias}' -> '{target}' but '{target}' "
                f"is not in SUPPORTED_LANGUAGES"
            )

    def test_no_self_referencing_alias(self):
        """Aliases should not map a supported code to itself redundantly."""
        for code in SUPPORTED_LANGUAGES:
            assert code not in _LANGUAGE_ALIASES, (
                f"'{code}' is already a supported code; "
                f"no need for an alias entry"
            )

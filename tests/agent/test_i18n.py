"""Tests for agent.i18n -- catalog parity, fallback, language resolution."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from agent import i18n


LOCALES_DIR = Path(__file__).resolve().parents[2] / "locales"


def _load_raw(lang: str) -> dict:
    with (LOCALES_DIR / f"{lang}.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flatten(d, prefix="") -> dict:
    flat = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        else:
            flat[key] = v
    return flat


# ---------------------------------------------------------------------------
# Catalog completeness. English is the complete source/fallback and Simplified
# Chinese is the first complete non-English implementation. Other language
# files are independent overlays: contributors may translate them incrementally
# without inheriting Simplified Chinese wording.
# ---------------------------------------------------------------------------



def test_shared_locale_registry_is_valid_and_drives_python_runtime():
    """Language identities and aliases have one machine-readable authority."""
    registry = json.loads((LOCALES_DIR / "registry.json").read_text(encoding="utf-8"))
    locales = registry["locales"]

    assert tuple(locales) == i18n.SUPPORTED_LANGUAGES
    assert registry["default"] == i18n.DEFAULT_LANGUAGE
    assert all(meta.get("name") and meta.get("triggerLabel") for meta in locales.values())
    for section in ("aliases", "compatibilityAliases"):
        assert set(registry[section].values()) <= set(locales)


def test_simplified_chinese_catalog_keys_match_english():
    """The first localization instance must cover the complete source catalog."""
    en_keys = set(_flatten(_load_raw("en")).keys())
    zh_keys = set(_flatten(_load_raw("zh")).keys())
    assert zh_keys == en_keys


@pytest.mark.parametrize("lang", [l for l in i18n.SUPPORTED_LANGUAGES if l != "en"])
def test_locale_overlays_do_not_invent_keys(lang: str):
    """Partial locale overlays may omit English keys but cannot add unknown ones."""
    en_keys = set(_flatten(_load_raw("en")).keys())
    lang_keys = set(_flatten(_load_raw(lang)).keys())
    extra = lang_keys - en_keys
    assert not extra, f"{lang}.yaml has keys not in en.yaml: {sorted(extra)}"


@pytest.mark.parametrize("lang", list(i18n.SUPPORTED_LANGUAGES))
def test_catalog_placeholders_match_english(lang: str):
    """Every translated value must use the same {placeholder} tokens as English.

    A mistranslated placeholder (e.g. ``{description}`` typoed as ``{descricao}``)
    would either raise KeyError at runtime or silently drop the interpolated
    value.  Pin parity at the test layer.
    """
    import re
    placeholder_re = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
    en_flat = _flatten(_load_raw("en"))
    lang_flat = _flatten(_load_raw(lang))
    for key, lang_value in lang_flat.items():
        en_value = en_flat[key]
        en_placeholders = set(placeholder_re.findall(en_value))
        lang_placeholders = set(placeholder_re.findall(lang_value))
        assert en_placeholders == lang_placeholders, (
            f"{lang}.yaml key={key!r}: placeholders {lang_placeholders} "
            f"don't match English {en_placeholders}"
        )


# ---------------------------------------------------------------------------
# Language resolution
# ---------------------------------------------------------------------------



def test_normalize_language_uses_only_registered_aliases():
    cases = {
        "chinese": "zh",
        "traditional-chinese": "zh-hant",
        "日本語": "ja",
        "한국어": "ko",
        "francais": "fr",
        "brazilian": "pt",
        "العربية": "ar",
        "zh-CN": "zh",
        "zh_HK": "zh-hant",
        "pt_BR": "pt",
        "ar-EG": "ar",
    }
    assert {value: i18n.normalize_language(value) for value in cases} == cases
    assert i18n._normalize_lang("zh-extra") == "en"








def test_shared_python_language_changes_only_after_explicit_cache_reset(monkeypatch):
    """Dashboard/TUI live refresh must not leak into other Python surfaces."""
    from hermes_cli import config as config_module

    current = {"display": {"language": "en"}}
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    monkeypatch.setattr(config_module, "load_config_readonly", lambda: current)
    i18n.reset_language_cache()

    assert i18n.get_language() == "en"
    current = {"display": {"language": "zh"}}
    assert i18n.get_language() == "en"

    i18n.reset_language_cache()
    assert i18n.get_language() == "zh"


def test_default_when_nothing_set(monkeypatch):
    """With no env var and no config override, falls back to English."""
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    # Force config lookup to return None.
    i18n.reset_language_cache()
    monkeypatch.setattr(i18n, "_configured_language", lambda: None)
    assert i18n.get_language() == "en"


# ---------------------------------------------------------------------------
# t() semantics
# ---------------------------------------------------------------------------







def test_t_missing_key_in_non_english_falls_back_to_english(tmp_path, monkeypatch):
    """If a key exists in English but not in the target locale, fall back."""
    # Stand up a fake incomplete locale under a temp locales dir.
    fake_locales = tmp_path / "locales"
    fake_locales.mkdir()
    (fake_locales / "en.yaml").write_text("foo: English Foo\n", encoding="utf-8")
    (fake_locales / "zh.yaml").write_text("# intentionally empty\n", encoding="utf-8")
    monkeypatch.setattr(i18n, "_locales_dir", lambda: fake_locales)
    i18n.reset_language_cache()
    try:
        assert i18n.t("foo", lang="zh") == "English Foo"
    finally:
        # Clear the cache on teardown so subsequent tests don't see the
        # fake "foo: English Foo" catalog instead of the real locales/*.yaml.
        i18n.reset_language_cache()




# ---------------------------------------------------------------------------
# _locales_dir resolution ladder -- regression for #23943 / #27632 / #35374.
# Sealed installs (Nix store venv, pip wheel) have no source tree next to
# agent/, so _locales_dir must resolve via env override or the data scheme.
# ---------------------------------------------------------------------------



def test_locales_dir_env_override_ignored_when_missing(tmp_path, monkeypatch):
    """A bogus HERMES_BUNDLED_LOCALES falls through to source/wheel resolution
    instead of returning a path that doesn't exist."""
    monkeypatch.setenv("HERMES_BUNDLED_LOCALES", str(tmp_path / "does-not-exist"))
    result = i18n._locales_dir()
    assert result != tmp_path / "does-not-exist"
    # In a source checkout this is the repo-root locales dir.
    assert result.name == "locales"


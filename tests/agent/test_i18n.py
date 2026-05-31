"""Tests for agent.i18n -- catalog parity, fallback, language resolution."""

from __future__ import annotations

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
# Catalog completeness -- this is the key invariant test.  If someone adds a
# new key to en.yaml they MUST add it to every other locale, else runtime
# falls back to English for those users and defeats the feature.
# ---------------------------------------------------------------------------

def test_all_locales_exist():
    """Every supported language must have a catalog file on disk."""
    for lang in i18n.SUPPORTED_LANGUAGES:
        assert (LOCALES_DIR / f"{lang}.yaml").is_file(), f"missing locales/{lang}.yaml"


@pytest.mark.parametrize("lang", [l for l in i18n.SUPPORTED_LANGUAGES if l != "en"])
def test_catalog_keys_match_english(lang: str):
    """Every non-English catalog must have exactly the same key set as English."""
    en_keys = set(_flatten(_load_raw("en")).keys())
    lang_keys = set(_flatten(_load_raw(lang)).keys())
    missing = en_keys - lang_keys
    extra = lang_keys - en_keys
    assert not missing, f"{lang}.yaml missing keys: {sorted(missing)}"
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
    for key, en_value in en_flat.items():
        en_placeholders = set(placeholder_re.findall(en_value))
        lang_value = lang_flat.get(key, "")
        lang_placeholders = set(placeholder_re.findall(lang_value))
        assert en_placeholders == lang_placeholders, (
            f"{lang}.yaml key={key!r}: placeholders {lang_placeholders} "
            f"don't match English {en_placeholders}"
        )


# ---------------------------------------------------------------------------
# Language resolution
# ---------------------------------------------------------------------------

def test_normalize_lang_accepts_supported():
    assert i18n._normalize_lang("zh") == "zh"
    assert i18n._normalize_lang("EN") == "en"


def test_normalize_lang_accepts_aliases():
    assert i18n._normalize_lang("chinese") == "zh"
    assert i18n._normalize_lang("zh-CN") == "zh"
    assert i18n._normalize_lang("Deutsch") == "de"
    assert i18n._normalize_lang("español") == "es"
    assert i18n._normalize_lang("jp") == "ja"
    assert i18n._normalize_lang("Ukrainian") == "uk"
    assert i18n._normalize_lang("uk-UA") == "uk"
    assert i18n._normalize_lang("ua") == "uk"
    assert i18n._normalize_lang("Turkish") == "tr"
    assert i18n._normalize_lang("tr-TR") == "tr"
    assert i18n._normalize_lang("türkçe") == "tr"


def test_normalize_lang_unknown_falls_back():
    assert i18n._normalize_lang("klingon") == "en"
    assert i18n._normalize_lang("") == "en"
    assert i18n._normalize_lang(None) == "en"


def test_env_var_override(monkeypatch):
    """HERMES_LANGUAGE wins over config."""
    i18n.reset_language_cache()
    monkeypatch.setenv("HERMES_LANGUAGE", "ja")
    assert i18n.get_language() == "ja"


def test_env_var_normalized(monkeypatch):
    i18n.reset_language_cache()
    monkeypatch.setenv("HERMES_LANGUAGE", "Chinese")
    assert i18n.get_language() == "zh"


def test_default_when_nothing_set(monkeypatch):
    """With no env var and no config override, falls back to English."""
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    # Force config lookup to return None -- patch the cached reader.
    i18n.reset_language_cache()
    monkeypatch.setattr(i18n, "_config_language_cached", lambda: None)
    assert i18n.get_language() == "en"


# ---------------------------------------------------------------------------
# t() semantics
# ---------------------------------------------------------------------------

def test_t_explicit_lang():
    assert i18n.t("approval.denied", lang="en").endswith("Denied")
    assert i18n.t("approval.denied", lang="zh").endswith("已拒绝")
    assert i18n.t("approval.denied", lang="uk").endswith("Відхилено")
    assert i18n.t("approval.denied", lang="tr").endswith("Reddedildi")


def test_t_formats_placeholders():
    msg = i18n.t("gateway.draining", lang="en", count=3)
    assert "3" in msg


def test_t_missing_key_returns_key():
    """A missing key returns its own path -- ugly but never crashes."""
    result = i18n.t("nonexistent.key.path", lang="en")
    assert result == "nonexistent.key.path"


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


def test_t_unknown_language_uses_english():
    """Unknown lang codes normalize to English, not to a key-path fallback."""
    assert i18n.t("approval.denied", lang="klingon") == i18n.t("approval.denied", lang="en")


# ---------------------------------------------------------------------------
# Regression tests for #35595 — /model command i18n keys resolve correctly
# ---------------------------------------------------------------------------

def test_gateway_model_keys_have_english_catalog():
    """Every gateway.model.* key used in /model command output MUST exist in
    the English catalog so the t() function never returns a bare dotted key
    path to end users (#35595)."""
    model_keys = [
        "gateway.model.switched",
        "gateway.model.provider_label",
        "gateway.model.context_label",
        "gateway.model.max_output_label",
        "gateway.model.cost_label",
        "gateway.model.capabilities_label",
        "gateway.model.prompt_caching_enabled",
        "gateway.model.warning_prefix",
        "gateway.model.saved_global",
        "gateway.model.session_only_hint",
        "gateway.model.current_label",
        "gateway.model.current_tag",
        "gateway.model.more_models_suffix",
        "gateway.model.usage_switch_model",
        "gateway.model.usage_switch_provider",
        "gateway.model.usage_persist",
        "gateway.model.error_prefix",
    ]
    for key in model_keys:
        result = i18n.t(key, lang="en")
        assert result != key, (
            f"i18n key {key!r} returned itself — missing from en.yaml catalog. "
            f"This causes the /model command to show structured field names "
            f"instead of human-readable messages (#35595)."
        )


def test_model_switch_message_renders_human_readable():
    """End-to-end: a /model switch confirmation should produce a human-readable
    paragraph, not structured i18n key paths."""
    msg = i18n.t("gateway.model.switched", model="gpt-5.4-mini")
    assert "`gpt-5.4-mini`" in msg
    assert msg != "gateway.model.switched"

    provider = i18n.t("gateway.model.provider_label", provider="openai")
    assert "openai" in provider
    assert provider != "gateway.model.provider_label"

    ctx = i18n.t("gateway.model.context_label", tokens="128,000")
    assert "128,000" in ctx
    assert ctx != "gateway.model.context_label"


def test_locales_dir_finds_catalogs(tmp_path, monkeypatch):
    """_locales_dir() should find locales even when the repo-root path
    doesn't exist, using the package-relative fallback."""
    real_locales = i18n._locales_dir()
    assert real_locales.is_dir(), "real locales dir should exist in git checkout"

    # Simulate a pip-installed layout where only the fallback path exists
    fake_agent = tmp_path / "agent"
    fake_agent.mkdir()
    fake_locales = fake_agent / "locales"
    fake_locales.mkdir()
    (fake_locales / "en.yaml").write_text(
        "gateway:\n  model:\n    switched: \"Switched to {model}\"\n",
        encoding="utf-8",
    )

    # Mock __file__ to be inside fake_agent
    fake_i18n = fake_agent / "i18n.py"
    fake_i18n.write_text("", encoding="utf-8")

    import agent.i18n as i18n_module

    original_file = i18n_module.__file__
    try:
        # Point __file__ to the fake location; parent.parent won't have locales
        monkeypatch.setattr(i18n_module, "__file__", str(fake_i18n))
        # Clear any cached catalogs from the real locale path
        i18n.reset_language_cache()
        result = i18n_module._locales_dir()
        assert result.is_dir(), "fallback path should find fake_locales"
    finally:
        monkeypatch.setattr(i18n_module, "__file__", original_file)
        i18n.reset_language_cache()

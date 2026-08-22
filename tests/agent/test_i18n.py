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


# ---------------------------------------------------------------------------
# Register-aware i18n
# ---------------------------------------------------------------------------


def test_default_register_is_technical(monkeypatch):
    """With no config override, the register is 'technical' (byte-compatible)."""
    i18n.reset_language_cache()
    monkeypatch.setattr(i18n, "_config_register_cached", lambda: None)
    assert i18n.get_register() == "technical"


def test_normalize_register():
    """Known registers pass through; unknown values fall back to default."""
    assert i18n._normalize_register("technical") == "technical"
    assert i18n._normalize_register("friendly") == "friendly"
    assert i18n._normalize_register("FRIENDLY") == "friendly"
    assert i18n._normalize_register("") == "technical"
    assert i18n._normalize_register(None) == "technical"
    assert i18n._normalize_register("bogus") == "technical"
    assert i18n._normalize_register(123) == "technical"


def test_t_with_technical_register_is_byte_compatible(monkeypatch):
    """t(key, register='technical') must produce the same output as t(key)."""
    i18n.reset_language_cache()
    monkeypatch.setattr(i18n, "_config_register_cached", lambda: None)
    monkeypatch.setattr(i18n, "_config_language_cached", lambda: None)
    for key in (
        "gateway.reset.header_default",
        "gateway.draining",
        "gateway.stop.stopped",
        "approval.choose_long",
    ):
        assert i18n.t(key) == i18n.t(key, register="technical")


def test_t_friendly_register_uses_overlay(monkeypatch):
    """t(key, register='friendly') returns the friendly variant when one exists."""
    i18n.reset_language_cache()
    monkeypatch.setattr(i18n, "_config_register_cached", lambda: None)
    monkeypatch.setattr(i18n, "_config_language_cached", lambda: None)
    friendly = i18n.t("gateway.reset.header_default", register="friendly")
    technical = i18n.t("gateway.reset.header_default")
    assert friendly != technical, "friendly register should differ from technical"
    assert "Fresh start" in friendly


def test_t_friendly_register_falls_back_to_base_for_missing_keys(monkeypatch):
    """Keys not in the friendly overlay fall through to the base catalog."""
    i18n.reset_language_cache()
    monkeypatch.setattr(i18n, "_config_register_cached", lambda: None)
    monkeypatch.setattr(i18n, "_config_language_cached", lambda: None)
    # approval.choose_long is NOT in en-friendly.yaml, so it should return
    # the same value as the technical register.
    friendly = i18n.t("approval.choose_long", register="friendly")
    technical = i18n.t("approval.choose_long")
    assert friendly == technical


def test_t_friendly_register_with_format_kwargs(monkeypatch):
    """Friendly register respects {placeholder} formatting."""
    i18n.reset_language_cache()
    monkeypatch.setattr(i18n, "_config_register_cached", lambda: None)
    monkeypatch.setattr(i18n, "_config_language_cached", lambda: None)
    result = i18n.t("gateway.draining", register="friendly", count=3)
    assert "3" in result
    assert "active task(s)" in result


def test_t_friendly_register_missing_catalog_falls_back(tmp_path, monkeypatch):
    """If no friendly catalog exists for a language, fall back to base."""
    fake_locales = tmp_path / "locales"
    fake_locales.mkdir()
    (fake_locales / "en.yaml").write_text("foo: English Foo\n", encoding="utf-8")
    # No en-friendly.yaml — the friendly register should still resolve.
    monkeypatch.setattr(i18n, "_locales_dir", lambda: fake_locales)
    i18n.reset_language_cache()
    try:
        result = i18n.t("foo", register="friendly")
        assert result == "English Foo"
    finally:
        i18n.reset_language_cache()


def test_t_unknown_register_falls_back_to_technical(monkeypatch):
    """An unknown register name should not crash — it falls back to technical."""
    i18n.reset_language_cache()
    monkeypatch.setattr(i18n, "_config_register_cached", lambda: None)
    monkeypatch.setattr(i18n, "_config_language_cached", lambda: None)
    result = i18n.t("gateway.reset.header_default", register="bogus")
    technical = i18n.t("gateway.reset.header_default")
    assert result == technical


def test_register_overlay_placeholder_parity():
    """Every friendly variant must use the same {placeholder} tokens as the
    base English catalog.  A mistranslated placeholder would raise KeyError
    at runtime.
    """
    import re
    placeholder_re = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
    en_flat = _flatten(_load_raw("en"))
    overlay_path = LOCALES_DIR / "en-friendly.yaml"
    if not overlay_path.is_file():
        pytest.skip("en-friendly.yaml not found")
    overlay_flat = _flatten(yaml.safe_load(overlay_path.read_text(encoding="utf-8")))
    for key, overlay_value in overlay_flat.items():
        assert key in en_flat, f"en-friendly.yaml has key {key!r} not in en.yaml"
        en_placeholders = set(placeholder_re.findall(en_flat[key]))
        overlay_placeholders = set(placeholder_re.findall(overlay_value))
        assert en_placeholders == overlay_placeholders, (
            f"en-friendly.yaml key={key!r}: placeholders {overlay_placeholders} "
            f"don't match English {en_placeholders}"
        )


def test_reset_language_cache_clears_register_cache(monkeypatch):
    """reset_language_cache() must clear register catalog and config caches."""
    i18n.reset_language_cache()
    # Prime the register catalog cache by loading a real overlay.
    i18n._load_register_catalog("en", "friendly")
    assert i18n._register_catalog_cache  # cache populated
    # Now reset.
    i18n.reset_language_cache()
    assert not i18n._register_catalog_cache  # cache cleared


def test_t_friendly_register_via_config(monkeypatch):
    """When display.message_register is 'friendly' in config, t() uses it."""
    i18n.reset_language_cache()
    monkeypatch.setattr(i18n, "_config_register_cached", lambda: "friendly")
    monkeypatch.setattr(i18n, "_config_language_cached", lambda: None)
    result = i18n.t("gateway.reset.header_default")
    assert "Fresh start" in result


def test_t_explicit_register_overrides_config(monkeypatch):
    """Explicit register= argument takes precedence over config."""
    i18n.reset_language_cache()
    monkeypatch.setattr(i18n, "_config_register_cached", lambda: "friendly")
    monkeypatch.setattr(i18n, "_config_language_cached", lambda: None)
    # Config says friendly, explicit override says technical.
    result = i18n.t("gateway.reset.header_default", register="technical")
    assert "Fresh start" not in result
    assert "Session reset" in result or "New session" in result


def test_t_empty_register_string_resolves_to_technical(monkeypatch):
    """t(key, register='') must normalize to 'technical', not defer to config.

    Regression: the truthiness check `if register` treated an explicit
    empty string as absent, falling through to get_register() which could
    return 'friendly' from config.  The fix uses `register is not None`.
    """
    i18n.reset_language_cache()
    monkeypatch.setattr(i18n, "_config_register_cached", lambda: "friendly")
    monkeypatch.setattr(i18n, "_config_language_cached", lambda: None)
    # Config says friendly, explicit register="" should resolve to technical.
    result = i18n.t("gateway.reset.header_default", register="")
    assert "Fresh start" not in result
    assert "Session reset" in result or "New session" in result


def test_config_propagation_with_temp_hermes_home(tmp_path, monkeypatch):
    """E2E: display.message_register in config.yaml propagates to t().

    Uses a real temp HERMES_HOME with a written config.yaml instead of
    monkeypatching the cached reader, exercising the real config-load path.
    """
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    config_file = hermes_home / "config.yaml"
    config_file.write_text(
        "display:\n  message_register: friendly\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    i18n.reset_language_cache()
    try:
        result = i18n.t("gateway.reset.header_default")
        assert "Fresh start" in result
    finally:
        i18n.reset_language_cache()
        monkeypatch.delenv("HERMES_HOME", raising=False)



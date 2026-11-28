"""Tests for agent.i18n -- catalog parity, fallback, language resolution."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import hermes_yaml as yaml

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


@pytest.mark.parametrize("lang", list(i18n.SUPPORTED_LANGUAGES))
def test_catalog_values_have_no_escaped_newlines(lang: str):
    """A double-escaped ``\\\\n`` in YAML loads as a visible backslash-n, so
    pickers render ``\\n`` text instead of line breaks (e.g. /reasoning, /fast)."""
    escaped = [key for key, value in _flatten(_load_raw(lang)).items() if "\\n" in value]
    assert not escaped, f"{lang}.yaml has literal \\n in: {escaped}"


# ---------------------------------------------------------------------------
# Language resolution
# ---------------------------------------------------------------------------











def test_default_when_nothing_set(monkeypatch):
    """With no env var and no config override, falls back to English."""
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    # Force config lookup to return None -- patch the cached reader.
    i18n.reset_language_cache()
    monkeypatch.setattr(i18n, "_config_language", lambda: None)
    assert i18n.get_language() == "en"


def test_language_is_per_profile_under_multiplex(monkeypatch, tmp_path):
    """HERMES_LANGUAGE in the DEFAULT profile's environ must not leak into a secondary profile's
    turn, and the config-language cache must not freeze one profile's ``display.language`` for all."""
    from agent import secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    default_home = tmp_path / "default"; default_home.mkdir()
    prof_b = tmp_path / "b"; prof_b.mkdir()
    (default_home / "config.yaml").write_text("display:\n  language: fr\n")
    (prof_b / "config.yaml").write_text("display:\n  language: de\n")
    monkeypatch.setenv("HERMES_LANGUAGE", "zh")  # default profile's .env, bridged into environ
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    i18n.reset_language_cache()
    secret_scope.set_multiplex_active(True)  # pins the launch home; a served turn binds an override
    token = secret_scope.set_secret_scope({})
    try:
        assert i18n.get_language() == "fr"  # scoped miss: env ignored, this profile's config wins
        home_token = set_hermes_home_override(prof_b)
        try:
            assert i18n.get_language() == "de"  # not the first profile's cached "fr"
        finally:
            reset_hermes_home_override(home_token)
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)
        i18n.reset_language_cache()


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
# A transient bundled-load failure must not poison the catalog cache (#149931).
# `_load_bundled` answers {} for an absent file and None for a present file it
# could not read, so only the first is cacheable: a short-on-memory startup
# used to memoise the empty merge and every `t()` answered a bare key until
# the gateway restarted.
# ---------------------------------------------------------------------------


def _fake_locales(tmp_path: Path, *, write_en: bool = True) -> Path:
    fake = tmp_path / "locales"
    fake.mkdir(exist_ok=True)
    if write_en:
        (fake / "en.yaml").write_text("foo: English Foo\n", encoding="utf-8")
    return fake


def test_transient_bundled_load_failure_heals_on_the_next_lookup(tmp_path, monkeypatch):
    """One failed load must not pin bare keys for the process lifetime."""
    monkeypatch.setattr(i18n, "_locales_dir", lambda: _fake_locales(tmp_path))
    i18n.reset_language_cache()
    real_parse = i18n.i18n_layers.parse_locale_file
    calls: list[Path] = []

    def flaky_parse(path):
        calls.append(path)
        if len(calls) == 1:
            raise MemoryError()  # its str() is empty — the log line must still name it
        return real_parse(path)

    monkeypatch.setattr(i18n.i18n_layers, "parse_locale_file", flaky_parse)
    try:
        assert i18n.t("foo", lang="en") == "foo"  # this lookup failed: bare key, no crash
        assert i18n.t("foo", lang="en") == "English Foo"  # retried, not served from a poisoned cache
    finally:
        i18n.reset_language_cache()


def test_absent_bundled_catalog_is_still_negatively_cached(tmp_path, monkeypatch):
    """An absent file stays cacheable; only a FAILED load skips the cache."""
    monkeypatch.setattr(i18n, "_locales_dir", lambda: _fake_locales(tmp_path, write_en=False))
    i18n.reset_language_cache()
    try:
        assert i18n._load_catalog("en", home=str(tmp_path / "home")) == {}
        assert (str(tmp_path / "home"), "en") in i18n._catalog_cache
    finally:
        i18n.reset_language_cache()


def test_bundled_load_failure_warning_names_the_exception_type(tmp_path, monkeypatch, caplog):
    """`Failed to load ... : %s` printed an empty tail for MemoryError."""
    monkeypatch.setattr(i18n, "_locales_dir", lambda: _fake_locales(tmp_path))

    def boom(path):
        raise MemoryError()

    monkeypatch.setattr(i18n.i18n_layers, "parse_locale_file", boom)
    with caplog.at_level(logging.WARNING, logger="agent.i18n"):
        assert i18n._load_bundled("en") is None
    assert "MemoryError" in caplog.text


def test_repeated_bundled_load_failure_warns_once_but_keeps_retrying(tmp_path, monkeypatch, caplog):
    """Retrying on every lookup must not turn one broken file into a warning per translated string."""
    monkeypatch.setattr(i18n, "_locales_dir", lambda: _fake_locales(tmp_path))
    i18n.reset_language_cache()
    calls: list[Path] = []

    def always_fails(path):
        calls.append(path)
        raise MemoryError()

    monkeypatch.setattr(i18n.i18n_layers, "parse_locale_file", always_fails)
    try:
        with caplog.at_level(logging.WARNING, logger="agent.i18n"):
            i18n.t("foo", lang="en")
            i18n.t("foo", lang="en")
        assert len(calls) == 2  # still retried: the failure is not cached
        assert caplog.text.count("Failed to load i18n catalog") == 1
    finally:
        i18n.reset_language_cache()


def test_surface_catalog_survives_a_failed_bundled_surface_load(tmp_path, monkeypatch):
    """A None from the failed load must not reach `dict.update` (TypeError)."""
    fake = _fake_locales(tmp_path)
    (fake / "zh.tui.yaml").write_text("foo: 你好\n", encoding="utf-8")
    monkeypatch.setattr(i18n, "_locales_dir", lambda: fake)
    real_parse = i18n.i18n_layers.parse_locale_file

    def flaky_parse(path):
        if str(path).endswith("zh.tui.yaml"):
            raise ValueError("transient")
        return real_parse(path)

    monkeypatch.setattr(i18n.i18n_layers, "parse_locale_file", flaky_parse)
    i18n.reset_language_cache()
    try:
        assert isinstance(i18n.surface_catalog("zh", "tui"), dict)
    finally:
        i18n.reset_language_cache()

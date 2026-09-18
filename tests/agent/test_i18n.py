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
    monkeypatch.setattr(i18n, "_config_language", lambda: None)
    assert i18n.get_language() == "en"


def test_language_is_per_profile_under_multiplex(monkeypatch, tmp_path):
    """HERMES_LANGUAGE in the DEFAULT profile's environ must not leak into a secondary profile's
    turn, and the config-language cache must not freeze one profile's ``display.language`` for all."""
    from agent import secret_scope

    default_home = tmp_path / "default"; default_home.mkdir()
    prof_b = tmp_path / "b"; prof_b.mkdir()
    (default_home / "config.yaml").write_text("display:\n  language: fr\n")
    (prof_b / "config.yaml").write_text("display:\n  language: de\n")
    monkeypatch.setenv("HERMES_LANGUAGE", "zh")  # default profile's .env, bridged into environ
    i18n.reset_language_cache()
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({})
    try:
        monkeypatch.setenv("HERMES_HOME", str(default_home))
        assert i18n.get_language() == "fr"  # scoped miss: env ignored, this profile's config wins
        monkeypatch.setenv("HERMES_HOME", str(prof_b))
        assert i18n.get_language() == "de"  # not the first profile's cached "fr"
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


@pytest.mark.parametrize("alias", ["sv", "sv-SE", "sv-FI", "sv_SE", "sv_FI", "svenska", "Swedish", " SV "])
def test_swedish_alias_renders_catalog(alias):
    assert i18n.t("gateway.model.switched", lang=alias, model="model/example") == (
        "Modellen har bytts till `model/example`"
    )


def test_swedish_profile_config_and_environment_precedence(monkeypatch, tmp_path):
    """Exercise the real config reader without reading or modifying a user's profile."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    (tmp_path / "config.yaml").write_text("display:\n  language: sv_FI\n", encoding="utf-8")
    i18n.reset_language_cache()
    try:
        assert i18n.get_language() == "sv"
        assert i18n.t("gateway.status.state_no") == "Nej"
        monkeypatch.setenv("HERMES_LANGUAGE", "en")
        assert i18n.get_language() == "en"
        assert i18n.t("gateway.status.state_no") == "No"
        assert i18n.t("gateway.status.state_no", lang="sv") == "Nej"
    finally:
        i18n.reset_language_cache()


def test_swedish_bundled_catalog_formats_every_message(monkeypatch, tmp_path):
    """Packaged locale resolution must preserve all format fields, including in notices."""
    import shutil
    from string import Formatter

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    shutil.copy(LOCALES_DIR / "sv.yaml", bundle / "sv.yaml")
    monkeypatch.setenv("HERMES_BUNDLED_LOCALES", str(bundle))
    english = _flatten(_load_raw("en"))
    swedish = _flatten(_load_raw("sv"))
    i18n.reset_language_cache()
    try:
        for key, source in english.items():
            fields = {field for _, field, _, _ in Formatter().parse(source) if field}
            values = {field: f"VALUE_{field}_END" for field in fields}
            rendered = i18n.t(key, lang="sv", **values)
            assert rendered == swedish[key].format(**values), key
            assert all(value in rendered for value in values.values()), key
    finally:
        i18n.reset_language_cache()


@pytest.mark.parametrize(
    "choice,allow_permanent,allow_session,expected",
    [
        ("o", True, True, "once"),
        ("s", True, True, "session"),
        ("a", True, True, "always"),
        ("d", True, True, "deny"),
        ("o", False, True, "once"),
        ("s", False, True, "session"),
        ("o", False, False, "once"),
        ("d", False, False, "deny"),
        ("s", False, False, "deny"),
        ("a", False, False, "deny"),
        ("", True, True, "deny"),
        ("unknown", True, True, "deny"),
    ],
)
def test_swedish_approval_menu_preserves_decisions(
    monkeypatch, capsys, choice, allow_permanent, allow_session, expected
):
    from tools.approval_prompt import prompt_dangerous_approval

    monkeypatch.setenv("HERMES_LANGUAGE", "sv")
    monkeypatch.setattr("builtins.input", lambda prompt: choice)
    i18n.reset_language_cache()
    try:
        result = prompt_dangerous_approval(
            "example-command", "Exempel", timeout_seconds=2,
            allow_permanent=allow_permanent, allow_session=allow_session,
        )
        assert result == expected
        output = capsys.readouterr().out
        assert "FARLIGT KOMMANDO: Exempel" in output
        assert "[o] tillåt en gång" in output
        assert "[d] neka" in output
        assert ("[s] tillåt under sessionen" in output) == allow_session
        assert ("[a] tillåt alltid" in output) == (allow_permanent and allow_session)
    finally:
        i18n.reset_language_cache()

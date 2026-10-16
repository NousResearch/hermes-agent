"""Auto-detect for ``display.language``: with nothing configured the control host's own locale picks
the startup language, and an explicit choice (``HERMES_LANGUAGE`` or ``display.language``) always wins.
Real temp homes, a real plugin directory through the real loader — no mocks of the resolution path.
"""

from __future__ import annotations

import re

import pytest

import hermes_yaml as yaml

from agent import i18n, i18n_layers, i18n_locale
from hermes_cli.plugins import PluginManager

_LOCALE_ENV = ("LC_ALL", "LC_MESSAGES", "LANG")
_TAG_RE = re.compile(r"^[a-z]{2,3}(-[a-z0-9]{2,8})*$")


def _reset() -> None:
    i18n_layers._reset_registry_for_tests()
    i18n.reset_language_cache()


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A temp HERMES_HOME with no plugins, an empty bundled dir and no locale in the environment."""
    from hermes_cli import plugins as plugins_mod

    home = tmp_path / "home"
    (home / "locales").mkdir(parents=True)
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    for name in _LOCALE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: empty_bundled)
    _reset()
    yield home
    _reset()


def _write_pt_br_pack(home, name: str = "hermes-lang-pt-br") -> None:
    """A manifest-only pack registering a real ``pt-br`` variant of the bundled ``pt``."""
    plugin = home / "plugins" / name
    (plugin / "locales").mkdir(parents=True)
    manifest = {"name": name, "version": "1.0.0", "description": "pt-BR language pack",
                "provides_locales": [{"id": "pt-br", "endonym": "Português (Brasil)"}]}
    (plugin / "plugin.yaml").write_text(yaml.safe_dump(manifest, allow_unicode=True), encoding="utf-8")
    (plugin / "locales" / "pt-br.yaml").write_text(
        yaml.safe_dump({"approval": {"denied": "Negado"}}, allow_unicode=True), encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": [name]}}), encoding="utf-8")


# ── the host locale is the fallback ───────────────────────────────────────────────────────────


def test_host_locale_selects_the_bundled_language(home, monkeypatch):
    """pt_BR.UTF-8 with nothing configured → the bundled pt catalog."""
    monkeypatch.setenv("LANG", "pt_BR.UTF-8")
    i18n.reset_language_cache()
    assert i18n.get_language() == "pt"
    assert i18n.t("approval.denied") == i18n.t("approval.denied", lang="pt")


def test_region_tag_falls_back_to_its_base_subtag(home, monkeypatch):
    monkeypatch.setenv("LC_ALL", "pt_PT.UTF-8")
    i18n.reset_language_cache()
    assert i18n.get_language() == "pt"


def test_lc_all_outranks_lc_messages_and_lang(home, monkeypatch):
    monkeypatch.setenv("LANG", "de_DE.UTF-8")
    monkeypatch.setenv("LC_MESSAGES", "fr_FR.UTF-8")
    monkeypatch.setenv("LC_ALL", "pt_BR.UTF-8")
    i18n.reset_language_cache()
    assert i18n.get_language() == "pt"


def test_unsupported_host_locale_falls_back_to_english(home, monkeypatch):
    monkeypatch.setenv("LANG", "vi_VN.UTF-8")
    i18n.reset_language_cache()
    assert i18n.get_language() == "en"


@pytest.mark.parametrize("value", ["C", "POSIX", "C.UTF-8", "@euro", ".UTF-8"])
def test_c_and_posix_name_no_language(value):
    """``LANG=C`` means "no localization requested" — a miss, not a language called "c"."""
    assert i18n_locale._clean_tag(value) is None


def test_clean_tag_strips_encoding_and_modifier():
    assert i18n_locale._clean_tag("pt_BR.UTF-8") == "pt-br"
    assert i18n_locale._clean_tag("pt_BR.UTF-8@latin") == "pt-br"
    assert i18n_locale._clean_tag("  zh_Hant_TW  ") == "zh-hant-tw"


def test_host_locale_reports_no_language_without_one(home, monkeypatch):
    """No usable LANG and no platform answer either (forced) → the default, English."""
    for name in _LOCALE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(i18n_locale, "_windows_locale", lambda: None)
    monkeypatch.setattr(i18n_locale, "_python_locale", lambda: None)
    i18n.reset_language_cache()
    assert i18n.get_language() == "en"


# ── an explicit choice always wins ────────────────────────────────────────────────────────────


def test_display_language_beats_the_host_locale(home, monkeypatch):
    (home / "config.yaml").write_text('display:\n  language: "en"\n', encoding="utf-8")
    monkeypatch.setenv("LANG", "pt_BR.UTF-8")
    i18n.reset_language_cache()
    assert i18n.get_language() == "en"


def test_hermes_language_beats_the_host_locale(home, monkeypatch):
    monkeypatch.setenv("LANG", "pt_BR.UTF-8")
    monkeypatch.setenv("HERMES_LANGUAGE", "fr")
    i18n.reset_language_cache()
    assert i18n.get_language() == "fr"


def test_empty_display_language_means_auto_detect(home, monkeypatch):
    """The documented spelling of "auto": accepted by validation, resolved by the host locale."""
    from hermes_cli.config_language import display_language_error

    assert display_language_error("") is None
    (home / "config.yaml").write_text('display:\n  language: ""\n', encoding="utf-8")
    monkeypatch.setenv("LANG", "pt_BR.UTF-8")
    i18n.reset_language_cache()
    assert i18n.get_language() == "pt"


# ── a supplied variant beats its alias ────────────────────────────────────────────────────────


def test_pack_variant_is_what_a_pt_br_host_lands_on(home, monkeypatch):
    """A real pt-br pack (not the pt alias) is the language the host locale resolves to."""
    _write_pt_br_pack(home)
    PluginManager().discover_and_load()
    monkeypatch.setenv("LANG", "pt_BR.UTF-8")
    i18n.reset_language_cache()
    assert i18n.get_language() == "pt-br"
    assert i18n.t("approval.denied") == "Negado"


# ── the platform fact ─────────────────────────────────────────────────────────────────────────


@pytest.mark.platforms("windows")
def test_windows_reports_a_well_formed_locale_tag(monkeypatch):
    for name in _LOCALE_ENV:
        monkeypatch.delenv(name, raising=False)
    tag = i18n_locale.os_locale()
    assert tag and _TAG_RE.match(tag), tag

"""``hermes config set display.language`` accepts every ``supported_languages()`` id — bundled, alias,
and a plugin language pack found through real discovery — and refuses unknown ids naming the options."""

from __future__ import annotations

import pytest
import hermes_yaml as yaml

from agent import i18n, i18n_layers
from hermes_cli.config import set_config_value


@pytest.fixture
def home(tmp_path, monkeypatch):
    from hermes_cli import plugins as plugins_mod

    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").touch()
    (home / "config.yaml").write_text("display:\n  language: en\n", encoding="utf-8")
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: empty_bundled)
    i18n_layers._reset_registry_for_tests()
    i18n.reset_language_cache()
    yield home
    i18n_layers._reset_registry_for_tests()
    i18n.reset_language_cache()


def _saved_language(home) -> str:
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))["display"]["language"]


def test_bundled_language_and_alias_are_accepted(home, capsys):
    set_config_value("display.language", "de")
    assert _saved_language(home) == "de"
    set_config_value("display.language", "zh-TW")  # alias of a bundled catalog; stored as typed
    assert _saved_language(home) == "zh-TW"
    assert i18n.get_language() == "zh-hant"


def test_unknown_language_is_refused_with_the_available_list(home, capsys):
    with pytest.raises(SystemExit):
        set_config_value("display.language", "klingon")
    err = capsys.readouterr().err
    assert "klingon" in err and "Available:" in err and "en, af" in err
    assert _saved_language(home) == "en"


def test_pack_language_is_accepted_once_the_pack_is_installed(home, capsys):
    plugin = home / "plugins" / "hermes-lang-pl"
    (plugin / "locales").mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(yaml.safe_dump({
        "name": "hermes-lang-pl", "version": "1.0.0", "description": "Polish pack", "provides_locales": ["pl"],
    }), encoding="utf-8")
    (plugin / "locales" / "pl.yaml").write_text(yaml.safe_dump({"approval": {"denied": "Odrzucono"}}), encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump({"display": {"language": "en"},
                                                     "plugins": {"enabled": ["hermes-lang-pl"]}}), encoding="utf-8")

    set_config_value("display.language", "pl")
    assert _saved_language(home) == "pl"
    assert i18n.get_language() == "pl"
    assert i18n.t("approval.denied") == "Odrzucono"

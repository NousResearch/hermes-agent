"""Bundled TUI packs: ``locales/<lang>.tui.yaml`` mirror ``locales/_keys.tui.json`` and are served
by ``i18n.catalog {surface: 'tui'}`` underneath overlay and plugin packs."""

from __future__ import annotations

import json
from pathlib import Path

import hermes_yaml as yaml
import pytest

from agent import i18n, i18n_layers

LOCALES_DIR = Path(__file__).resolve().parents[2] / "locales"
TUI_KEYS = set(json.loads((LOCALES_DIR / "_keys.tui.json").read_text(encoding="utf-8")))


@pytest.mark.parametrize("lang", [l for l in i18n.SUPPORTED_LANGUAGES if l != "en"])
def test_bundled_tui_pack_mirrors_tui_key_export(lang: str):
    pack = i18n_layers.parse_locale_file(LOCALES_DIR / f"{lang}.tui.yaml")
    missing = TUI_KEYS - set(pack)
    extra = set(pack) - TUI_KEYS
    if lang == "zh":
        assert not missing, f"{lang}.tui.yaml missing {len(missing)} keys, e.g. {sorted(missing)[:5]}"
    assert not extra, f"{lang}.tui.yaml has {len(extra)} keys absent from _keys.tui.json, e.g. {sorted(extra)[:5]}"


@pytest.fixture
def home(tmp_path, monkeypatch):
    i18n_layers._reset_registry_for_tests()
    home = tmp_path / "home"
    (home / "locales").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    i18n.reset_language_cache()
    yield home
    i18n_layers._reset_registry_for_tests()
    i18n.reset_language_cache()


def test_tui_surface_serves_bundled_pack_under_overlay(home):
    bundled = i18n_layers.parse_locale_file(LOCALES_DIR / "de.tui.yaml")
    key = min(bundled)
    served = i18n.surface_catalog("de", "tui")
    assert served[key] == bundled[key]
    assert set(served) >= set(bundled)

    (home / "locales" / "de.tui.yaml").write_text(yaml.safe_dump({key: "Overlay"}), encoding="utf-8")
    i18n.reset_language_cache()
    served = i18n.surface_catalog("de", "tui")
    assert served[key] == "Overlay"
    assert len(served) == len(bundled)  # overlay is partial; the rest still comes from the bundled pack


def test_core_surface_never_ships_bundled_python_strings(home):
    assert i18n.surface_catalog("de", i18n_layers.CORE_SURFACE) == {}

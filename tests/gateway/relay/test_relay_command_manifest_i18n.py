"""The relay lane's Discord command manifest is resolved for the active language each time it is
built (per ``hello``), and every description stays inside Discord's 100 UTF-16-unit cap after
translation — one over-long localized description would fail the connector's whole bulk overwrite."""

from __future__ import annotations

import hermes_yaml as yaml
import pytest

from agent import i18n, i18n_layers
from gateway.platforms.base import utf16_len
from gateway.relay.command_manifest import build_relay_command_manifest

# 60 BMP chars + 30 astral chars = 120 UTF-16 units.
_LONG = "Ü" * 60 + "😀" * 30


@pytest.fixture
def home(tmp_path, monkeypatch):
    i18n_layers._reset_registry_for_tests()
    home = tmp_path / "home"
    (home / "locales").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    i18n.reset_language_cache()
    yield home
    i18n_layers._reset_registry_for_tests()
    i18n.reset_language_cache()


def _manifest() -> dict:
    return {row["name"]: row for row in build_relay_command_manifest()}


def _descriptions(manifest: dict) -> list:
    out = []
    for row in manifest.values():
        out.append(row["description"])
        out.extend(opt["description"] for opt in row.get("options", ()))
    return out


def test_manifest_follows_active_language_and_stays_within_discord_cap(home, monkeypatch):
    overlay = {"platform": {
        "discord": {"command": {"new": {"description": "Neues Gespräch beginnen"}}},
        "relay": {"command": {"model": {"arg_name": _LONG}}},
    }}
    (home / "locales" / "de.yaml").write_text(yaml.safe_dump(overlay, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv("HERMES_LANGUAGE", "de")
    i18n.reset_language_cache()

    german = _manifest()
    assert german["new"]["description"] == "Neues Gespräch beginnen"
    model_arg = german["model"]["options"][0]["description"]
    assert _LONG.startswith(model_arg) and utf16_len(model_arg) == 100
    assert all(utf16_len(d) <= 100 for d in _descriptions(german))

    # No import-time freeze: the next build (next hello) carries the new language.
    monkeypatch.setenv("HERMES_LANGUAGE", "en")
    i18n.reset_language_cache()
    english = _manifest()
    assert english["new"]["description"] == i18n.t("platform.discord.command.new.description", lang="en")
    assert english["new"]["description"] != german["new"]["description"]

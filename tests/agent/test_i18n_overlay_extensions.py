"""Every locale overlay discovered for the picker must be available to its renderer."""

import hermes_yaml as yaml
import pytest

from agent import i18n


@pytest.fixture
def overlay_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "locales").mkdir()
    i18n.reset_language_cache()
    yield tmp_path / "locales"
    i18n.reset_language_cache()


def _catalog_value(surface):
    if surface == "core":
        return i18n.t("approval.denied", lang="eo")
    return i18n.surface_catalog("eo", surface).get("approval.denied")


@pytest.mark.parametrize("surface", ["core", "tui", "desktop"])
def test_discovered_yml_overlay_is_served(overlay_home, surface):
    stem = "eo" if surface == "core" else f"eo.{surface}"
    (overlay_home / f"{stem}.yml").write_text(
        yaml.safe_dump({"approval.denied": "Esperanto override"}), encoding="utf-8",
    )

    assert "eo" in i18n.supported_languages()
    assert _catalog_value(surface) == "Esperanto override"
    assert surface in i18n.language_surfaces()["eo"]


@pytest.mark.parametrize("surface", ["core", "tui", "desktop"])
def test_yaml_overlay_keeps_precedence_over_yml(overlay_home, surface):
    stem = "eo" if surface == "core" else f"eo.{surface}"
    for extension, message in (("yaml", "Primary override"), ("yml", "Secondary override")):
        (overlay_home / f"{stem}.{extension}").write_text(
            yaml.safe_dump({"approval.denied": message}), encoding="utf-8",
        )

    assert _catalog_value(surface) == "Primary override"

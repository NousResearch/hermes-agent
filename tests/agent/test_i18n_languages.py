"""Language identity: every bundled catalog has an endonym row and vice versa; ``language_options`` is
``en`` first, endonym-only, and reports where each language comes from."""

from __future__ import annotations

from agent import i18n
from agent.i18n_languages import BUNDLED_LANGUAGE_INFO, describe_language


def test_bundled_table_matches_bundled_catalogs():
    assert set(BUNDLED_LANGUAGE_INFO) == set(i18n.SUPPORTED_LANGUAGES)
    for lang, (endonym, rtl) in BUNDLED_LANGUAGE_INFO.items():
        assert endonym.strip() and endonym != lang, lang  # a real native name, never the bare id
        assert isinstance(rtl, bool)
    assert BUNDLED_LANGUAGE_INFO["ar"][1] is True


def test_language_options_are_en_first_sorted_and_sourced(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    i18n.reset_language_cache()
    try:
        options = i18n.language_options()
    finally:
        i18n.reset_language_cache()
    ids = [o["id"] for o in options]
    assert ids == ["en", *sorted(ids[1:])] and set(ids) == set(i18n.SUPPORTED_LANGUAGES)
    assert all(o["source"] == "bundled" for o in options)
    assert set(options[0]) == {"id", "endonym", "rtl", "source", "surfaces"}


def test_describe_language_precedence():
    # Pack metadata names a pack-only language; the bundled table wins for a bundled id even if a pack
    # re-declares it; an overlay-only language is reported as such and falls back to the bare id.
    pack = {"endonym": "Polski", "rtl": False, "source": "plugin:hermes-lang-pl"}
    assert describe_language("pl", pack=pack, overlay=False, surfaces=("core", "desktop")) == {
        "id": "pl", "endonym": "Polski", "rtl": False, "source": "plugin:hermes-lang-pl", "surfaces": ["core", "desktop"]}
    assert describe_language("ar", pack={**pack, "endonym": "Other", "rtl": False}, overlay=False, surfaces=())["endonym"] == "العربية"
    assert describe_language("ar", pack={**pack, "rtl": False}, overlay=False, surfaces=())["rtl"] is True
    assert describe_language("eo", pack=None, overlay=True, surfaces=()) == {
        "id": "eo", "endonym": "eo", "rtl": False, "source": "overlay", "surfaces": []}
    assert describe_language("de", pack=None, overlay=True, surfaces=())["source"] == "bundled"


def test_language_surfaces_report_each_layer(monkeypatch, tmp_path):
    """Bundled Korean has core + TUI strings and no Desktop catalog; a user overlay or a pack adds the
    surface it ships. ``i18n.languages`` carries this so the Desktop can leave out a language it would
    only show in English."""
    import hermes_yaml as yaml

    from agent import i18n_layers

    home = tmp_path / "home"
    (home / "locales").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    i18n_layers._reset_registry_for_tests()
    i18n.reset_language_cache()
    try:
        surfaces = i18n.language_surfaces()
        assert surfaces["ko"] == ("core", "tui")
        assert surfaces["en"] == ("core",)

        (home / "locales" / "ko.desktop.yaml").write_text(yaml.safe_dump({"settings": {"title": "설정"}}),
                                                         encoding="utf-8")
        i18n_layers.register_pack("pl", "desktop", {"settings.title": "Ustawienia"}, source="plugin:hermes-lang-pl")
        i18n.reset_language_cache()
        surfaces = i18n.language_surfaces()
        assert surfaces["ko"] == ("core", "tui", "desktop")
        assert surfaces["pl"] == ("desktop",)
        rows = {row["id"]: row for row in i18n.language_options()}
        assert rows["ko"]["surfaces"] == ["core", "tui", "desktop"]
        assert rows["pl"]["surfaces"] == ["desktop"]
        assert rows["ja"]["surfaces"] == ["core", "tui"]

        # A layer that loaded no strings advertises nothing: an empty or unparseable overlay, an empty pack.
        (home / "locales" / "ja.desktop.yaml").write_text("# nothing translated yet\n", encoding="utf-8")
        (home / "locales" / "de.desktop.yaml").write_text("settings: [unclosed\n", encoding="utf-8")
        i18n_layers.register_pack("uk", "desktop", {}, source="plugin:hermes-lang-uk")
        i18n.reset_language_cache()
        surfaces = i18n.language_surfaces()
        assert "desktop" not in surfaces["ja"]
        assert "desktop" not in surfaces["de"]
        assert "desktop" not in surfaces["uk"]
    finally:
        i18n_layers._reset_registry_for_tests()
        i18n.reset_language_cache()

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
    assert set(options[0]) == {"id", "endonym", "rtl", "source"}


def test_describe_language_precedence():
    # Pack metadata names a pack-only language; the bundled table wins for a bundled id even if a pack
    # re-declares it; an overlay-only language is reported as such and falls back to the bare id.
    pack = {"endonym": "Polski", "rtl": False, "source": "plugin:hermes-lang-pl"}
    assert describe_language("pl", pack=pack, overlay=False) == {
        "id": "pl", "endonym": "Polski", "rtl": False, "source": "plugin:hermes-lang-pl"}
    assert describe_language("ar", pack={**pack, "endonym": "Other", "rtl": False}, overlay=False)["endonym"] == "العربية"
    assert describe_language("ar", pack={**pack, "rtl": False}, overlay=False)["rtl"] is True
    assert describe_language("eo", pack=None, overlay=True) == {"id": "eo", "endonym": "eo", "rtl": False, "source": "overlay"}
    assert describe_language("de", pack=None, overlay=True)["source"] == "bundled"

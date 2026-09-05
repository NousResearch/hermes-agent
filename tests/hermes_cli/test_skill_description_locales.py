"""Locale-aware bundled-skill descriptions stay complete and compatible."""

from pathlib import Path

import yaml

from hermes_cli.skill_description_locales import (
    load_skill_description_catalog,
    localize_skill_description,
    normalize_skill_description_locale,
)

REPO = Path(__file__).resolve().parents[2]


def _bundled_skill_names() -> set[str]:
    names = set()
    for path in REPO.glob("skills/**/SKILL.md"):
        frontmatter = yaml.safe_load(path.read_text(encoding="utf-8").split("---", 2)[1])
        names.add(frontmatter["name"])
    return names


def test_zh_hant_catalog_covers_every_bundled_skill_exactly():
    catalog = load_skill_description_catalog("zh-hant")
    assert set(catalog) == _bundled_skill_names()
    assert all(isinstance(value, str) and value.strip() for value in catalog.values())


def test_locale_normalization_is_narrow_and_has_english_fallback():
    assert normalize_skill_description_locale("zh-Hant-TW") == "zh-hant"
    assert normalize_skill_description_locale("zh_TW") == "zh-hant"
    assert normalize_skill_description_locale("fr-FR") == "en"
    assert normalize_skill_description_locale(None) == "en"


def test_only_bundled_skills_receive_catalog_overrides():
    english = "Manage notes."
    translated = localize_skill_description("apple-notes", english, "zh-hant", bundled=True)
    assert translated != english
    assert localize_skill_description("apple-notes", english, "zh-hant", bundled=False) == english
    assert localize_skill_description("third-party", english, "zh-hant", bundled=True) == english


def test_locale_catalog_cache_is_isolated_by_normalized_locale():
    load_skill_description_catalog.cache_clear()
    zh_hant = load_skill_description_catalog("zh-hant")
    english = load_skill_description_catalog("en")
    assert zh_hant
    assert english == {}
    assert load_skill_description_catalog("zh-TW") is zh_hant
    assert load_skill_description_catalog("en-US") is english

"""hermes_cli.tips reads its catalog through i18n: count follows the English catalog, text follows the
active language, and a partial translation falls back to English per tip."""

from __future__ import annotations

import pytest

from agent import i18n
from agent.i18n import t
from hermes_cli import tips


@pytest.fixture(autouse=True)
def _fresh_caches():
    tips.reset_tips_cache()
    i18n.reset_language_cache()
    yield
    tips.reset_tips_cache()
    i18n.reset_language_cache()


def test_every_tip_key_resolves_to_catalog_text():
    keys = tips.tip_keys()
    assert keys, "the English catalog ships tips"
    assert all(t(k) != k for k in keys), "a bare key echo means the catalog is missing the entry"
    assert len(set(keys)) == len(keys)


def test_random_tip_and_placeholder_come_from_the_catalog():
    assert tips.get_random_tip() in {t(k) for k in tips.tip_keys()}
    assert tips.get_random_composer_placeholder() in {t(k) for k in tips.composer_placeholder_keys()}


def test_pickers_follow_active_language_with_english_fallback(tmp_path, monkeypatch):
    (tmp_path / "en.yaml").write_text(
        "tips:\n  t001: 'first'\n  t002: 'second'\n  placeholder:\n    p01: 'ask'\n", encoding="utf-8")
    (tmp_path / "de.yaml").write_text("tips:\n  t001: 'erstens'\n", encoding="utf-8")
    monkeypatch.setattr(i18n, "_locales_dir", lambda: tmp_path)
    monkeypatch.setenv("HERMES_LANGUAGE", "de")
    i18n.reset_language_cache()
    tips.reset_tips_cache()
    try:
        assert tips.tip_keys() == ("tips.t001", "tips.t002")  # count comes from English
        assert tips.composer_placeholder_keys() == ("tips.placeholder.p01",)
        seen = {tips.get_random_tip() for _ in range(50)}
        assert seen == {"erstens", "second"}  # translated where present, English otherwise
        assert tips.get_random_composer_placeholder() == "ask"
    finally:
        i18n.reset_language_cache()
        tips.reset_tips_cache()

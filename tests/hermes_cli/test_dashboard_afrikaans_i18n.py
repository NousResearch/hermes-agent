"""Dashboard Afrikaans i18n registration tests."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
WEB_I18N = ROOT / "web" / "src" / "i18n"


def test_afrikaans_locale_is_registered_in_dashboard_i18n():
    types = (WEB_I18N / "types.ts").read_text(encoding="utf-8")
    context = (WEB_I18N / "context.tsx").read_text(encoding="utf-8")
    index = (WEB_I18N / "index.ts").read_text(encoding="utf-8")
    locales = (WEB_I18N / "locales.ts").read_text(encoding="utf-8")

    union_codes = set(re.findall(r'"([a-z]{2})"', types.split("export interface", 1)[0]))
    metadata_codes = set(re.findall(r'code: "([a-z]{2})"', locales))

    assert "af" in union_codes
    assert union_codes == metadata_codes
    assert 'import { af } from "./af";' in context
    assert "isLocale(stored)" in context
    assert "LOCALE_OPTIONS" in index
    assert "getNextLocale" in index


def test_afrikaans_locale_is_available_in_language_switcher():
    switcher = (ROOT / "web" / "src" / "components" / "LanguageSwitcher.tsx").read_text(encoding="utf-8")
    locales = (WEB_I18N / "locales.ts").read_text(encoding="utf-8")
    en = (WEB_I18N / "en.ts").read_text(encoding="utf-8")
    zh = (WEB_I18N / "zh.ts").read_text(encoding="utf-8")
    af = (WEB_I18N / "af.ts").read_text(encoding="utf-8")

    assert "getLocaleOption(locale)" in switcher
    assert "getNextLocale(locale)" in switcher
    assert 'code: "af"' in locales
    assert 'flag: "🇿🇦"' in locales
    assert 'label: "AF"' in locales
    assert 'switchTo: "Switch language"' in en
    assert 'switchTo: "切换语言"' in zh
    assert 'switchTo: "Wissel taal"' in af


def test_afrikaans_readme_entrypoint_is_linked():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    af_readme = (ROOT / "README.af.md").read_text(encoding="utf-8")

    assert "README.af.md" in readme
    assert "language: af" in af_readme
    assert "modelgegenereerde antwoorde" in af_readme

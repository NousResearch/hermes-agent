"""Language identity for the bundled locales — endonym and script direction — plus the picker list.

One Python table; ``apps/shared/src/i18n.ts`` ``LOCALE_ENDONYMS`` / ``RTL_LOCALES`` must agree for the
ids both sides bundle (the desktop lane's test checks the relationship). Pack-only languages take
their endonym/rtl from the pack registration (``PluginContext.register_locale(endonym=..., rtl=...)``)
and fall back to the bare id.
"""

from __future__ import annotations

from typing import TypedDict


class LanguageOption(TypedDict):
    id: str
    endonym: str
    rtl: bool
    source: str


# id -> (endonym, rtl). Keep sorted by id; ``en`` is listed first by ``language_options`` regardless.
BUNDLED_LANGUAGE_INFO: dict[str, tuple[str, bool]] = {
    "af": ("Afrikaans", False),
    "ar": ("العربية", True),
    "de": ("Deutsch", False),
    "en": ("English", False),
    "es": ("Español", False),
    "fr": ("Français", False),
    "ga": ("Gaeilge", False),
    "hu": ("Magyar", False),
    "it": ("Italiano", False),
    "ja": ("日本語", False),
    "ko": ("한국어", False),
    "pt": ("Português", False),
    "ru": ("Русский", False),
    "tr": ("Türkçe", False),
    "uk": ("Українська", False),
    "zh": ("简体中文", False),
    "zh-hant": ("繁體中文", False),
}

BUNDLED_SOURCE = "bundled"
OVERLAY_SOURCE = "overlay"


def describe_language(lang: str, *, pack: dict | None, overlay: bool) -> LanguageOption:
    """One picker row. Precedence for endonym/rtl: bundled table → pack metadata → bare id / LTR.
    ``source`` names the highest layer that supplies the language (``plugin:<name>`` > ``overlay`` >
    ``bundled``) so a picker can say where a language came from."""
    bundled = BUNDLED_LANGUAGE_INFO.get(lang)
    endonym, rtl = bundled if bundled else ((pack or {}).get("endonym") or lang, bool((pack or {}).get("rtl")))
    if pack is not None:
        source = str(pack.get("source") or "plugin")
    elif overlay and bundled is None:
        source = OVERLAY_SOURCE
    else:
        source = BUNDLED_SOURCE
    return {"id": lang, "endonym": endonym, "rtl": rtl, "source": source}


def language_options() -> list[LanguageOption]:
    """``[{"id", "endonym", "rtl", "source"}, ...]`` for every supported language, ``en`` first then sorted
    by id — the list ``i18n.languages`` serves and every switcher renders (endonym only, no flags)."""
    from agent import i18n, i18n_layers
    from hermes_constants import get_hermes_home

    overlay_langs = i18n_layers.overlay_languages(get_hermes_home())
    return [
        describe_language(lang, pack=i18n_layers.pack_info(lang), overlay=lang in overlay_langs)
        for lang in i18n.supported_languages()
    ]


__all__ = ["BUNDLED_LANGUAGE_INFO", "LanguageOption", "describe_language", "language_options"]

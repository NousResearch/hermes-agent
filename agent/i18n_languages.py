"""Language identity for the bundled locales — endonym and script direction — plus the picker list.

``locales/registry.json`` owns bundled identities for Python and TypeScript. Pack-only languages take
their endonym/rtl from the pack registration (``PluginContext.register_locale(endonym=..., rtl=...)``)
and fall back to the bare id.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypedDict


class LanguageOption(TypedDict):
    id: str
    endonym: str
    rtl: bool
    source: str


# 内置语言的身份与别名来自共享注册表；插件语言仍由上游分层注册机制发现。
_registry_dir = Path(os.environ.get("HERMES_BUNDLED_LOCALES", ""))
if not (_registry_dir / "registry.json").is_file():
    _registry_dir = Path(__file__).resolve().parent.parent / "locales"
LOCALE_REGISTRY = json.loads((_registry_dir / "registry.json").read_text(encoding="utf-8"))
BUNDLED_LANGUAGE_INFO: dict[str, tuple[str, bool]] = {
    lang: (meta["name"], meta.get("direction") == "rtl")
    for lang, meta in LOCALE_REGISTRY["locales"].items()
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

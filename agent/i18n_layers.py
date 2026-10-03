"""Catalog layers above the bundled ``locales/`` tree: plugin language packs and the user overlay.

A language pack is per-language and multi-surface: ``<lang>.yaml`` (core, read by ``agent.i18n.t``),
``<lang>.tui.yaml`` and ``<lang>.desktop.yaml`` (opaque to Python: parsed, flattened, served to the
renderers over ``i18n.catalog``). Every layer is a flat ``{dotted.key: text}`` mapping and may be
PARTIAL — it only carries the keys it overrides.

Two layers live here:

- **packs** — registered by plugins through ``PluginContext.register_locale`` (last registered wins);
- **overlay** — ``$HERMES_HOME/locales/<lang>[.<surface>].yaml`` of the *current* profile home.

Merged views are cached per ``(lang, surface)`` / ``(home, lang, surface)``; :func:`clear_cache` (called
by ``agent.i18n.reset_language_cache``) drops them, and every pack mutation resets the facade's caches
so the next ``t()`` sees the new layer.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

CORE_SURFACE = "core"
SURFACES: tuple[str, ...] = (CORE_SURFACE, "tui", "desktop")

# ``en``, ``zh-hant``, ``pt-br``, ``sr-latn-rs``: lowercase language subtag + optional region/script parts.
_LANGUAGE_ID_RE = re.compile(r"^[a-z]{2,3}(-[a-z0-9]{2,8})*$")
# ``pl.yaml`` -> (pl, core); ``pl.tui.yaml`` -> (pl, tui). Anything else in a locales dir is ignored.
_LOCALE_FILE_RE = re.compile(r"^(?P<lang>[a-z]{2,3}(?:-[a-z0-9]{2,8})*)(?:\.(?P<surface>[a-z]+))?\.ya?ml$")


def normalize_language_id(value: Any) -> str:
    """Canonical spelling of a language id: lowercase, ``_`` -> ``-``, surrounding whitespace dropped."""
    return str(value).strip().lower().replace("_", "-") if isinstance(value, str) else ""


def is_language_id(value: Any) -> bool:
    return bool(_LANGUAGE_ID_RE.match(normalize_language_id(value)))


# ── parsing ───────────────────────────────────────────────────────────────────────────────────


def flatten(
    node: Any,
    prefix: str = "",
    out: dict[str, str] | None = None,
    _ancestors: frozenset[int] = frozenset(),
) -> dict[str, str]:
    """Nested mapping -> ``{dotted.key: text}``. Non-string, non-mapping leaves are dropped (catalogs are
    text-only); :func:`non_text_leaves` reports them for the validator.

    A mapping that (transitively) contains itself — e.g. a YAML anchor/alias cycle — raises
    :class:`ValueError`; *ancestors* tracks the current path only, so a shared anchor reused by
    sibling keys is still flattened once per key."""
    flat: dict[str, str] = {} if out is None else out
    if isinstance(node, Mapping):
        if id(node) in _ancestors:
            raise ValueError(f"cyclic anchor/alias reference at {prefix or '<root>'!r}")
        children = _ancestors | {id(node)}
        for key, value in node.items():
            flatten(value, f"{prefix}.{key}" if prefix else str(key), flat, children)
    elif isinstance(node, str):
        flat[prefix] = node
    return flat


def non_text_leaves(
    node: Any, prefix: str = "", _ancestors: frozenset[int] = frozenset()
) -> list[str]:
    """Dotted paths of leaves that are neither text nor a mapping (numbers, lists, booleans, nulls).

    Cyclic input raises :class:`ValueError`, mirroring :func:`flatten`."""
    if isinstance(node, Mapping):
        if id(node) in _ancestors:
            raise ValueError(f"cyclic anchor/alias reference at {prefix or '<root>'!r}")
        children = _ancestors | {id(node)}
        found: list[str] = []
        for key, value in node.items():
            found.extend(
                non_text_leaves(
                    value, f"{prefix}.{key}" if prefix else str(key), children
                )
            )
        return found
    return [] if isinstance(node, str) else [prefix or "<root>"]


def parse_locale_file(path: Path) -> dict[str, str]:
    """Parse one locale YAML into a flat catalog. Raises on unreadable/unparseable input or a non-mapping
    document — the strict form for registration and validation."""
    import hermes_yaml as yaml
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return flatten(data)


def load_locale_source(source: Any) -> dict[str, str]:
    """``register_locale`` input -> flat catalog: a mapping (nested or already flat) is flattened; a path
    is parsed with :func:`parse_locale_file`."""
    if isinstance(source, Mapping):
        return flatten(source)
    if isinstance(source, (str, Path)):
        return parse_locale_file(Path(source))
    raise TypeError(f"locale source must be a mapping or a path to a YAML file, got {type(source).__name__}")


def scan_locale_dir(directory: Path) -> list[tuple[str, str, Path]]:
    """``(lang, surface, path)`` for every ``<lang>[.<surface>].yaml`` under *directory* (sorted; empty when
    the directory is missing). Unknown surfaces are reported so a typo like ``pl.tiu.yaml`` is visible."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    found: list[tuple[str, str, Path]] = []
    for path in sorted(directory.iterdir()):
        match = _LOCALE_FILE_RE.match(path.name)
        if match is None or not path.is_file():
            continue
        surface = match.group("surface") or CORE_SURFACE
        if surface not in SURFACES:
            logger.warning("Ignoring locale file %s: unknown surface %r (expected one of %s)",
                           path, surface, ", ".join(SURFACES))
            continue
        found.append((match.group("lang"), surface, path))
    return found


# ── plugin packs ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PackEntry:
    """One registered pack layer. ``token`` orders registrations; the last one registered wins."""

    lang: str
    surface: str
    messages: Mapping[str, str]
    source: str
    endonym: str | None = None
    rtl: bool = False
    token: int = field(default=0, compare=False)


_lock = threading.RLock()
_tokens = count(1)
_packs: list[PackEntry] = []
_pack_cache: dict[tuple[str, str], dict[str, str]] = {}
_overlay_cache: dict[tuple[str, str, str], dict[str, str]] = {}
_overlay_languages_cache: dict[str, frozenset[str]] = {}


def _reset_facade() -> None:
    from agent.i18n import reset_language_cache  # lazy: the facade imports this module
    reset_language_cache()


def register_pack(lang: str, surface: str, messages: Mapping[str, str], *, source: str,
                  endonym: str | None = None, rtl: bool = False) -> PackEntry:
    """Add a pack layer for ``(lang, surface)``; later registrations shadow earlier ones key by key."""
    lang_id = normalize_language_id(lang)
    if not is_language_id(lang_id):
        raise ValueError(f"invalid language id {lang!r} (expected e.g. 'pl', 'pt-br', 'zh-hant')")
    if surface not in SURFACES:
        raise ValueError(f"unknown locale surface {surface!r} (expected one of {', '.join(SURFACES)})")
    entry = PackEntry(lang_id, surface, dict(messages), source, endonym or None, bool(rtl), next(_tokens))
    with _lock:
        _packs.append(entry)
    _reset_facade()
    logger.debug("i18n pack registered: %s/%s from %s (%d keys)", lang_id, surface, source, len(entry.messages))
    return entry


def unregister_pack(entry: PackEntry) -> bool:
    """Drop one registration (``PluginRegistration`` release). ``False`` when it was already gone."""
    with _lock:
        try:
            _packs.remove(entry)
        except ValueError:
            return False
    _reset_facade()
    return True


def pack_layer(lang: str, surface: str = CORE_SURFACE) -> dict[str, str]:
    """Merged pack messages for ``(lang, surface)`` in registration order (last wins); cached."""
    key = (lang, surface)
    with _lock:
        cached = _pack_cache.get(key)
        if cached is not None:
            return cached
        merged: dict[str, str] = {}
        for entry in _packs:
            if entry.lang == lang and entry.surface == surface:
                merged.update(entry.messages)
        _pack_cache[key] = merged
        return merged


def pack_languages() -> frozenset[str]:
    with _lock:
        return frozenset(entry.lang for entry in _packs)


def pack_info(lang: str) -> dict[str, Any] | None:
    """``{endonym, rtl, source}`` of the newest pack registration for *lang* (``None`` when no pack has
    it). ``endonym``/``rtl`` come from the newest registration that declared them."""
    with _lock:
        entries = [entry for entry in _packs if entry.lang == lang]
    if not entries:
        return None
    endonym = next((entry.endonym for entry in reversed(entries) if entry.endonym), None)
    rtl = next((entry.rtl for entry in reversed(entries) if entry.endonym is not None), entries[-1].rtl)
    return {"endonym": endonym, "rtl": rtl, "source": entries[-1].source}


# ── user overlay ──────────────────────────────────────────────────────────────────────────────


def overlay_dir(home: Path | str) -> Path:
    return Path(home) / "locales"


def overlay_layer(home: Path | str, lang: str, surface: str = CORE_SURFACE) -> dict[str, str]:
    """``<home>/locales/<lang>[.<surface>].yaml`` flattened; ``{}`` (logged) when missing or broken. Cached
    per home so a multiplexed process serving profile A then B then A never mixes overlays."""
    key = (str(home), lang, surface)
    with _lock:
        cached = _overlay_cache.get(key)
        if cached is not None:
            return cached
    name = f"{lang}.yaml" if surface == CORE_SURFACE else f"{lang}.{surface}.yaml"
    path = overlay_dir(home) / name
    flat: dict[str, str] = {}
    if path.is_file():
        try:
            flat = parse_locale_file(path)
        except Exception as exc:
            logger.warning("Failed to load i18n overlay %s: %s", path, exc)
            flat = {}
    with _lock:
        _overlay_cache[key] = flat
    return flat


def overlay_languages(home: Path | str) -> frozenset[str]:
    """Language ids that have at least one overlay file (any surface) under *home*."""
    key = str(home)
    with _lock:
        cached = _overlay_languages_cache.get(key)
        if cached is not None:
            return cached
    langs = frozenset(lang for lang, _surface, _path in scan_locale_dir(overlay_dir(home)))
    with _lock:
        _overlay_languages_cache[key] = langs
    return langs


def surface_catalog(home: Path | str, lang: str, surface: str = CORE_SURFACE) -> dict[str, str]:
    """Overlay + packs (packs win) for one surface — the layer ``i18n.catalog`` ships to a renderer, which
    merges it over its own bundled catalog. Bundled Python locales are NOT included."""
    merged = dict(overlay_layer(home, lang, surface))
    merged.update(pack_layer(lang, surface))
    return merged


def layered_languages(home: Path | str) -> frozenset[str]:
    """Languages supplied by any layer above bundled (overlay of *home* ∪ packs)."""
    return overlay_languages(home) | pack_languages()


def clear_cache() -> None:
    """Drop merged views (not the registrations); the facade's ``reset_language_cache`` calls this."""
    with _lock:
        _pack_cache.clear()
        _overlay_cache.clear()
        _overlay_languages_cache.clear()


def _reset_registry_for_tests() -> None:
    with _lock:
        _packs.clear()
    clear_cache()


def registered_packs() -> tuple[PackEntry, ...]:
    with _lock:
        return tuple(_packs)


__all__ = [
    "CORE_SURFACE", "SURFACES", "PackEntry", "normalize_language_id", "is_language_id", "flatten",
    "non_text_leaves", "parse_locale_file", "load_locale_source", "scan_locale_dir", "register_pack",
    "unregister_pack", "pack_layer", "pack_languages", "pack_info", "overlay_dir", "overlay_layer",
    "overlay_languages", "surface_catalog", "layered_languages", "clear_cache", "registered_packs",
]

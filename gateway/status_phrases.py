"""Human-friendly generic gateway status phrases: short chat-safe lines for the long-running
status surface without relaying raw model scratch text — only configured phrase strings are used;
tool args, commands, previews, and reasoning are never interpolated. Built-in defaults live in
``gateway/assets/status_phrases.yaml``; users add profile-relative catalogs under ``HERMES_HOME``
via ``status_phrases.yaml`` / ``status_phrases/*.yaml`` or ``display.status_phrases: {path:
<HERMES_HOME-relative>, mode: append|replace}``. Absolute paths and ``..`` escapes are ignored on
purpose so config stays profile-portable and cannot read arbitrary files."""

from __future__ import annotations

import random as _random
from collections.abc import Mapping, MutableSequence
from pathlib import Path
from typing import Any

import yaml

from hermes_constants import get_hermes_home

# Hermes UI surfaces, not app/vendor buckets.  Long-running-only: regular tool/thinking/interim
# chatter is deliberately not rewritten (too noisy in chat).
_STATUS_SURFACES = ("status", "generic")
_MAX_CUSTOM_PHRASES_PER_SURFACE = 80
_MAX_PHRASE_CHARS = 160
_CONVENTIONAL_RELATIVE_PATHS = ("status_phrases.yaml", "status_phrases")
_YAML_SUFFIXES = {".yaml", ".yml"}
_CONFIG_KEYS = ("generic_status_phrases", "status_phrases")  # legacy alias first

_FALLBACK_PHRASES: dict[str, list[str]] = {
    "status": ["still on it", "still working through it", "waiting for the result"],
    "generic": ["on it", "one sec", "checking that now"],
}


def _clean_phrase_list(value: Any) -> list[str]:
    cleaned: list[str] = []
    for item in value[:_MAX_CUSTOM_PHRASES_PER_SURFACE] if isinstance(value, list) else ():
        phrase = str(item or "").strip()
        if phrase and len(phrase) <= _MAX_PHRASE_CHARS and phrase not in cleaned:
            cleaned.append(phrase)
    return cleaned


def _merge_phrase_mapping(catalog: dict[str, list[str]], section: Mapping[str, Any], *,
                          inherited_mode: str | None = None) -> None:
    replace = str(section.get("mode") or inherited_mode or "append").strip().lower() == "replace"
    phrase_map = section.get("phrases") if isinstance(section.get("phrases"), Mapping) else section
    for surface in _STATUS_SURFACES:
        phrases = _clean_phrase_list(phrase_map.get(surface) if isinstance(phrase_map, Mapping) else None)
        if phrases:
            catalog[surface] = phrases if replace else [*catalog.get(surface, []), *phrases]


def _merge_phrase_file(catalog: dict[str, list[str]], path: Path, *, inherited_mode: str | None = None) -> None:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if isinstance(loaded, Mapping):
        _merge_phrase_mapping(catalog, loaded, inherited_mode=inherited_mode)


def _iter_phrase_files(base_dir: Path, raw_path: Any) -> list[Path]:
    """YAML files under ``base_dir/raw_path``; [] for absolute / ``..`` / escaping paths."""
    raw = str(raw_path or "").strip()
    candidate = Path(raw).expanduser()
    if not raw or candidate.is_absolute() or ".." in candidate.parts:
        return []
    base = base_dir.resolve()
    path = (base / candidate).resolve()
    try:
        path.relative_to(base)
    except ValueError:
        return []
    if path.is_file() and path.suffix.lower() in _YAML_SUFFIXES:
        return [path]
    if path.is_dir():
        return sorted(c for c in path.iterdir() if c.is_file() and c.suffix.lower() in _YAML_SUFFIXES)
    return []


def _merge_phrase_paths(catalog: dict[str, list[str]], paths: Any, *, base_dir: Path,
                        inherited_mode: str | None = None) -> None:
    if paths is None:
        return
    for raw_path in paths if isinstance(paths, list) else [paths]:
        for phrase_file in _iter_phrase_files(base_dir, raw_path):
            _merge_phrase_file(catalog, phrase_file, inherited_mode=inherited_mode)


def _copy_catalog(catalog: Mapping[str, list[str]]) -> dict[str, list[str]]:
    return {surface: list(phrases) for surface, phrases in catalog.items()}


_DEFAULT_PHRASES: dict[str, list[str]] = _copy_catalog(_FALLBACK_PHRASES)
_merge_phrase_file(
    _DEFAULT_PHRASES, Path(__file__).resolve().parent / "assets" / "status_phrases.yaml", inherited_mode="replace"
)


def _merge_phrase_config(catalog: dict[str, list[str]], section: Any, *, base_dir: Path) -> None:
    """Merge one display.status_phrases-style section (files first, then inline phrases)."""
    if not isinstance(section, Mapping):
        return
    mode = str(section.get("mode") or "append").strip().lower()
    for key in ("path", "paths"):
        _merge_phrase_paths(catalog, section.get(key), base_dir=base_dir, inherited_mode=mode)
    _merge_phrase_mapping(catalog, section)


def resolve_status_phrase_catalog(user_config: Mapping[str, Any] | None,
                                  platform_key: str | None = None) -> dict[str, list[str]]:
    """Resolve built-in + user-configured generic status phrases. Order mirrors gateway display
    settings: built-ins, conventional profile-relative user files, global
    ``display.status_phrases`` (or legacy alias ``generic_status_phrases``), then
    ``display.platforms.<platform>.status_phrases``."""
    catalog = _copy_catalog(_DEFAULT_PHRASES)
    hermes_home = get_hermes_home()
    _merge_phrase_paths(catalog, list(_CONVENTIONAL_RELATIVE_PATHS), base_dir=hermes_home)
    display = (user_config or {}).get("display") if isinstance(user_config, Mapping) else None
    if not isinstance(display, Mapping):
        return catalog
    sections, platforms = [display], display.get("platforms")
    if platform_key and isinstance(platforms, Mapping) and isinstance(platforms.get(platform_key), Mapping):
        sections.append(platforms[platform_key])
    for section in sections:
        for key in _CONFIG_KEYS:
            _merge_phrase_config(catalog, section.get(key), base_dir=hermes_home)
    return catalog


def classify_status_context(kind: str, *, tool_name: str | None = None, preview: str | None = None,
                            args: Any = None) -> str:
    """Classify an internal gateway event into a Hermes UI-surface bucket."""
    if str(kind or "").strip().lower() in {"heartbeat", "waiting", "long_running", "status"}:
        return "status"
    return "generic"


def choose_status_phrase(kind: str, *, tool_name: str | None = None, preview: str | None = None,
                         args: Any = None, recent: MutableSequence[str] | None = None,
                         rng: Any = None, catalog: Mapping[str, list[str]] | None = None) -> str:
    """Pick a short generic status phrase, avoiding recent repeats. ``preview`` and ``args`` are
    accepted for callback compatibility, but their raw contents are never embedded in the result."""
    phrase_catalog = catalog or _DEFAULT_PHRASES
    category = classify_status_context(kind, tool_name=tool_name, preview=preview, args=args)
    candidates = list(phrase_catalog.get(category) or phrase_catalog.get("generic") or _DEFAULT_PHRASES["generic"])
    if recent:
        recent_set = set(recent)
        candidates = [p for p in candidates if p not in recent_set] or candidates
    phrase = (rng or _random).choice(candidates)
    if recent is not None:
        recent.append(phrase)
        del recent[:-6]
    return phrase

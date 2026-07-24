"""Human-friendly generic gateway status phrases.

These helpers deliberately avoid relaying raw model scratch text.  They turn
Hermes' long-running gateway status surface into short status lines suitable
for chat surfaces.

Built-in defaults live in ``gateway/assets/status_phrases.yaml``. Users can add
portable, profile-relative phrase catalogs under ``HERMES_HOME`` either by using
conventional paths::

    ~/.hermes/status_phrases.yaml
    ~/.hermes/status_phrases/*.yaml

or by pointing config at a relative file/directory::

    display:
      status_phrases:
        path: status_phrases/whatsapp.yaml  # relative to HERMES_HOME
        mode: append                        # append (default) or replace

Absolute paths and ``..`` escapes are ignored on purpose so config stays
profile-portable and does not accidentally read arbitrary files.

Only configured phrase strings are used; raw tool args, commands, previews, and
reasoning text are never interpolated into the returned phrase.
"""

from __future__ import annotations

import random as _random
from collections.abc import Mapping, MutableSequence
from pathlib import Path
from typing import Any

import yaml

from hermes_constants import get_hermes_home

# These are Hermes UI surfaces, not app/vendor/domain buckets.  Keep this
# long-running-only: regular tool/thinking/interim chatter is intentionally not
# rewritten into generic placeholders because that gets noisy fast in chat.
_STATUS_SURFACES = ("status", "generic")
_MAX_CUSTOM_PHRASES_PER_SURFACE = 80
_MAX_PHRASE_CHARS = 160
_CONVENTIONAL_RELATIVE_PATHS = ("status_phrases.yaml", "status_phrases")

_FALLBACK_PHRASES: dict[str, list[str]] = {
    "status": ["still on it", "still working through it", "waiting for the result"],
    "generic": ["on it", "one sec", "checking that now"],
}


def _clean_phrase_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value[:_MAX_CUSTOM_PHRASES_PER_SURFACE]:
        phrase = str(item or "").strip()
        if not phrase or len(phrase) > _MAX_PHRASE_CHARS or phrase in seen:
            continue
        cleaned.append(phrase)
        seen.add(phrase)
    return cleaned


def _merge_phrase_mapping(catalog: dict[str, list[str]], section: Mapping[str, Any], *, inherited_mode: str | None = None) -> None:
    mode = str(section.get("mode") or inherited_mode or "append").strip().lower()
    replace = mode == "replace"
    phrase_map = section.get("phrases") if isinstance(section.get("phrases"), Mapping) else section
    for surface in _STATUS_SURFACES:
        phrases = _clean_phrase_list(phrase_map.get(surface) if isinstance(phrase_map, Mapping) else None)
        if not phrases:
            continue
        catalog[surface] = phrases if replace else [*catalog.get(surface, []), *phrases]


def _merge_phrase_file(catalog: dict[str, list[str]], path: Path, *, inherited_mode: str | None = None) -> None:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if isinstance(loaded, Mapping):
        _merge_phrase_mapping(catalog, loaded, inherited_mode=inherited_mode)


def _relative_path_under(base_dir: Path, raw_path: Any) -> Path | None:
    raw = str(raw_path or "").strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    base = base_dir.resolve()
    resolved = (base / candidate).resolve()
    try:
        resolved.relative_to(base)
    except ValueError:
        return None
    return resolved


def _iter_phrase_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
        return [path]
    if path.is_dir():
        return sorted(
            child for child in path.iterdir()
            if child.is_file() and child.suffix.lower() in {".yaml", ".yml"}
        )
    return []


def _merge_phrase_paths(
    catalog: dict[str, list[str]],
    paths: Any,
    *,
    base_dir: Path,
    inherited_mode: str | None = None,
) -> None:
    if paths is None:
        return
    raw_paths = paths if isinstance(paths, list) else [paths]
    for raw_path in raw_paths:
        resolved = _relative_path_under(base_dir, raw_path)
        if resolved is None:
            continue
        for phrase_file in _iter_phrase_files(resolved):
            _merge_phrase_file(catalog, phrase_file, inherited_mode=inherited_mode)


def _load_builtin_catalog() -> dict[str, list[str]]:
    catalog = {surface: list(phrases) for surface, phrases in _FALLBACK_PHRASES.items()}
    catalog_path = Path(__file__).resolve().parent / "assets" / "status_phrases.yaml"
    _merge_phrase_file(catalog, catalog_path, inherited_mode="replace")
    return catalog


_DEFAULT_PHRASES: dict[str, list[str]] = _load_builtin_catalog()


def _copy_default_catalog() -> dict[str, list[str]]:
    return {surface: list(phrases) for surface, phrases in _DEFAULT_PHRASES.items()}


def _merge_phrase_config(catalog: dict[str, list[str]], section: Any, *, base_dir: Path | None = None) -> None:
    """Merge one display.status_phrases-style section into ``catalog``."""
    if not isinstance(section, Mapping):
        return
    mode = str(section.get("mode") or "append").strip().lower()
    if base_dir is not None:
        _merge_phrase_paths(catalog, section.get("path"), base_dir=base_dir, inherited_mode=mode)
        _merge_phrase_paths(catalog, section.get("paths"), base_dir=base_dir, inherited_mode=mode)
    _merge_phrase_mapping(catalog, section)


def resolve_status_phrase_catalog(user_config: Mapping[str, Any] | None, platform_key: str | None = None) -> dict[str, list[str]]:
    """Resolve built-in + user-configured generic status phrases.

    Resolution order mirrors gateway display settings: built-ins, conventional
    profile-relative user files, global ``display.status_phrases`` (or legacy
    alias ``generic_status_phrases``), then
    ``display.platforms.<platform>.status_phrases``.
    """
    catalog = _copy_default_catalog()
    hermes_home = get_hermes_home()
    _merge_phrase_paths(catalog, list(_CONVENTIONAL_RELATIVE_PATHS), base_dir=hermes_home)

    display = (user_config or {}).get("display") if isinstance(user_config, Mapping) else None
    if not isinstance(display, Mapping):
        return catalog

    _merge_phrase_config(catalog, display.get("generic_status_phrases"), base_dir=hermes_home)
    _merge_phrase_config(catalog, display.get("status_phrases"), base_dir=hermes_home)

    platforms = display.get("platforms")
    if platform_key and isinstance(platforms, Mapping):
        platform_display = platforms.get(platform_key)
        if isinstance(platform_display, Mapping):
            _merge_phrase_config(catalog, platform_display.get("generic_status_phrases"), base_dir=hermes_home)
            _merge_phrase_config(catalog, platform_display.get("status_phrases"), base_dir=hermes_home)
    return catalog


def classify_status_context(
    kind: str,
    *,
    tool_name: str | None = None,
    preview: str | None = None,
    args: Any = None,
) -> str:
    """Classify an internal gateway event into a Hermes UI-surface bucket."""
    normalized = str(kind or "").strip().lower()
    if normalized in {"heartbeat", "waiting", "long_running", "status"}:
        return "status"
    return "generic"


def choose_status_phrase(
    kind: str,
    *,
    tool_name: str | None = None,
    preview: str | None = None,
    args: Any = None,
    recent: MutableSequence[str] | None = None,
    rng: Any = None,
    catalog: Mapping[str, list[str]] | None = None,
) -> str:
    """Pick a short generic status phrase, avoiding recent repeats.

    ``preview`` and ``args`` are accepted for callback compatibility, but their
    raw contents are never embedded in the returned phrase.
    """
    phrase_catalog = catalog or _DEFAULT_PHRASES
    category = classify_status_context(kind, tool_name=tool_name, preview=preview, args=args)
    candidates = list(phrase_catalog.get(category) or phrase_catalog.get("generic") or _DEFAULT_PHRASES["generic"])
    if recent:
        recent_set = set(recent)
        fresh = [phrase for phrase in candidates if phrase not in recent_set]
        if fresh:
            candidates = fresh
    picker = rng or _random
    phrase = picker.choice(candidates)
    if recent is not None:
        recent.append(phrase)
        del recent[:-6]
    return phrase

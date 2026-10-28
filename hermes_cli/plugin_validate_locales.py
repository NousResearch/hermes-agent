"""``hermes plugins validate`` — language-pack checks (``provides_locales``).

For every declared id the pack must ship ``locales/<id>.yaml`` that parses and is text-only (a number,
list or null leaf is an error: ``t()`` would silently drop it). Keys the English catalogs do not have
are a WARNING naming them — a pack written against a newer or older Hermes still installs, it just
carries dead keys. Core keys come from the bundled ``locales/en.yaml``; TUI/Desktop keys from
``locales/_keys.tui.json`` / ``locales/_keys.desktop.json`` (exported by the TS packages' build
scripts). When an export is absent the key check for that surface is skipped, never failed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from agent.i18n_layers import CORE_SURFACE, flatten, non_text_leaves, scan_locale_dir

logger = logging.getLogger(__name__)

_KEY_EXPORTS = {"tui": "_keys.tui.json", "desktop": "_keys.desktop.json"}


def _bundled_locales_dir() -> Path:
    from agent.i18n import _locales_dir
    return _locales_dir()


def reference_keys(surface: str) -> Optional[set[str]]:
    """English key set for *surface*: flattened ``en.yaml`` for core, the JSON export for tui/desktop.
    ``None`` when the reference is unavailable (the check is then skipped)."""
    locales = _bundled_locales_dir()
    try:
        if surface == CORE_SURFACE:
            from agent.i18n_layers import parse_locale_file
            return set(parse_locale_file(locales / "en.yaml"))
        export = locales / _KEY_EXPORTS[surface]
        if not export.is_file():
            return None
        data = json.loads(export.read_text(encoding="utf-8-sig"))
        keys = data.get("keys") if isinstance(data, dict) else data
        return {str(k) for k in keys} if isinstance(keys, list) else None
    except Exception as exc:  # a broken reference is a Hermes bug, not the pack's
        logger.debug("i18n reference keys for %s unavailable: %s", surface, exc)
        return None


def _load_document(path: Path) -> Any:
    import hermes_yaml as yaml
    with path.open("r", encoding="utf-8-sig") as handle:
        return yaml.safe_load(handle)


def check_language_packs(report, manifest: dict, plugin_dir: Path) -> None:
    """Locale-pack admission checks; a no-op (single passing line) when nothing is declared."""
    from hermes_cli.plugins_manifest import parse_provides_locales

    raw = manifest.get("provides_locales")
    declared, _meta = parse_provides_locales(raw, str(manifest.get("name") or plugin_dir.name))
    if raw is None:
        return
    if not declared:
        report.add("locales", False, "provides_locales declares no valid language id (expected e.g. pl, pt-br)")
        return
    locales_dir = plugin_dir / "locales"
    found = scan_locale_dir(locales_dir)
    by_lang: dict[str, list] = {}
    for lang_id, surface, path in found:
        by_lang.setdefault(lang_id, []).append((surface, path))
    for lang_id in declared:
        files = by_lang.get(lang_id, [])
        if not any(surface == CORE_SURFACE for surface, _ in files):
            report.add(f"locale {lang_id}", False, f"provides_locales declares {lang_id!r} but locales/{lang_id}.yaml is missing")
            continue
        for surface, path in files:
            _check_locale_file(report, lang_id, surface, path)
    undeclared = sorted(set(by_lang) - set(declared))
    if undeclared:
        report.warn(f"locales/ ships {', '.join(undeclared)} not listed in provides_locales "
                    "(loaded anyway; declare them so users can see the pack's languages)")


def _check_locale_file(report, lang_id: str, surface: str, path: Path) -> None:
    label = f"locale {lang_id}" if surface == CORE_SURFACE else f"locale {lang_id}.{surface}"
    try:
        document = _load_document(path)
    except Exception as exc:
        report.add(label, False, f"{path.name} failed to parse: {exc}")
        return
    if document is None:
        document = {}
    if not isinstance(document, dict):
        report.add(label, False, f"{path.name} must be a mapping, got {type(document).__name__}")
        return
    try:
        bad_leaves = non_text_leaves(document)
        if bad_leaves:
            report.add(label, False, f"{path.name} has non-text value(s) at: {', '.join(sorted(bad_leaves))}")
            return
        flat = flatten(document)
    except (ValueError, RecursionError) as exc:  # cyclic anchors, or nesting past Python's recursion limit
        report.add(label, False, f"{path.name} failed to flatten: {exc}")
        return
    reference = reference_keys(surface)
    if reference is None:
        report.add(label, True, f"{path.name}: {len(flat)} key(s); no {surface} key reference available, key check skipped")
        return
    unknown = sorted(set(flat) - reference)
    if unknown:
        report.warn(f"{path.name}: {len(unknown)} key(s) not in the English {surface} catalog (ignored at runtime): "
                    + ", ".join(unknown))
    report.add(label, True, f"{path.name}: {len(flat)} key(s), {len(flat) - len(unknown)} match the English {surface} catalog")


__all__ = ["check_language_packs", "reference_keys"]

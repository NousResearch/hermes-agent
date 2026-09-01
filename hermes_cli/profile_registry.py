"""P3.0 — Authoritative machine-readable profile registry.

One source contract: ``governance/profile-registry.yaml`` in the
KenseiAgent candidate.  Organisational metadata ONLY — model, provider,
tool, skill, memory and runtime configuration are deliberately excluded
(locked decision: the registry never duplicates runtime configuration).

Path ownership:
  * versioned source authority: ``governance/profile-registry.yaml`` (repo)
  * eventual deployment target: ``${HERMES_HOME}/governance/profile-registry.yaml``
  * build/tests:               explicit fixture or temporary HERMES_HOME

The data file is the single authority.  This module is the only parser;
the dashboard mirrors the shared schema/fixture, it never carries a
second registry data copy.

All functions are pure or read-only.  Nothing here creates, deletes,
retires or reconfigures a profile, and discrepancy reporting never
mutates state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

REGISTRY_FILENAME = "profile-registry.yaml"

SCHEMA_VERSION = 1

ALLOWED_KINDS = ("lead", "worker", "utility")
ALLOWED_LIFECYCLE = ("active", "standby", "frozen", "retired")

REQUIRED_FIELDS = ("name", "kind", "parent", "lifecycle", "domains", "gateway_unit")


class RegistryError(ValueError):
    """Raised when a registry document fails validation."""


def default_registry_path() -> Path:
    """Resolve the deployed registry path (never a hardcoded checkout)."""
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return Path(home) / "governance" / REGISTRY_FILENAME


def _as_name_list(value: Any, ctx: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    raise RegistryError(f"{ctx}: domains must be a string or list of strings")


def _validate_entry(entry: Any, index: int) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise RegistryError(f"profiles[{index}] must be a mapping")
    for req in REQUIRED_FIELDS:
        if req not in entry:
            raise RegistryError(f"profiles[{index}] ({entry.get('name')!r}) missing field: {req}")
    name = entry["name"]
    if not isinstance(name, str) or not name.strip():
        raise RegistryError(f"profiles[{index}].name must be a non-empty string")
    if entry["kind"] not in ALLOWED_KINDS:
        raise RegistryError(
            f"profiles[{index}] ({name}): kind must be one of {ALLOWED_KINDS}"
        )
    if entry["lifecycle"] not in ALLOWED_LIFECYCLE:
        raise RegistryError(
            f"profiles[{index}] ({name}): lifecycle must be one of {ALLOWED_LIFECYCLE}"
        )
    parent = entry["parent"]
    if parent is not None and (not isinstance(parent, str) or not parent.strip()):
        raise RegistryError(f"profiles[{index}] ({name}): parent must be null or a name")
    domains = _as_name_list(entry["domains"], name)
    gateway_unit = entry["gateway_unit"]
    if gateway_unit is not None and not isinstance(gateway_unit, str):
        raise RegistryError(f"profiles[{index}] ({name}): gateway_unit must be null or a string")
    return {
        "name": name,
        "kind": entry["kind"],
        "parent": parent,
        "lifecycle": entry["lifecycle"],
        "domains": domains,
        "gateway_unit": gateway_unit,
    }


def _check_graph(profiles: list[dict[str, Any]], root_name: str) -> None:
    names = [p["name"] for p in profiles]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise RegistryError(f"duplicate profile names: {', '.join(dupes)}")

    known = set(names) | {root_name}
    parent_map = {p["name"]: p["parent"] for p in profiles}

    # No self-parent; every named parent resolves; no cycles.
    for profile in profiles:
        name, parent = profile["name"], profile["parent"]
        if parent is None:
            continue  # only the root may (and only leads should) have null parent
        if parent == name:
            raise RegistryError(f"profile {name!r} cannot be its own parent")
        if parent != root_name and parent not in set(names):
            raise RegistryError(f"profile {name!r} references unknown parent {parent!r}")
        # Cycle walk (bounded by roster size).
        seen: set[str] = set()
        current: Optional[str] = name
        while current is not None and current != root_name:
            if current in seen:
                raise RegistryError(
                    f"parent graph cycle involving {name!r}"
                )
            seen.add(current)
            current = parent_map.get(current)
            if current is None:
                break  # reached a null-parent node (attached to root below)


def load_registry(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load + fully validate a registry document.

    Returns a normalised dict:
        {"schema_version", "root": {"name", "description"}, "profiles": [...]}.
    Raises RegistryError on any schema, enum, uniqueness, parent-link or
    cycle violation.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RegistryError("PyYAML is required to read the profile registry") from exc

    registry_path = Path(path)
    try:
        raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RegistryError(f"registry not found: {registry_path}") from exc
    except OSError as exc:
        raise RegistryError(f"cannot read registry {registry_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise RegistryError("registry document must be a mapping")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise RegistryError(
            f"registry schema_version must be {SCHEMA_VERSION}"
        )

    root = raw.get("root")
    if not isinstance(root, dict) or not isinstance(root.get("name"), str):
        raise RegistryError("registry root.name is required")
    root_name = root["name"]

    entries_raw = raw.get("profiles")
    if not isinstance(entries_raw, list) or not entries_raw:
        raise RegistryError("registry profiles must be a non-empty list")

    profiles = [_validate_entry(e, i) for i, e in enumerate(entries_raw)]
    _check_graph(profiles, root_name)

    # Deterministic ordering: sort by name.
    profiles.sort(key=lambda p: p["name"])
    return {
        "schema_version": SCHEMA_VERSION,
        "root": {"name": root_name, "description": root.get("description")},
        "profiles": profiles,
    }


def profile_index(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index profiles by name (deterministic, from load_registry output)."""
    return {p["name"]: p for p in registry["profiles"]}


# ---------------------------------------------------------------------------
# Pure renderer — registry -> Markdown
# ---------------------------------------------------------------------------


def render_registry_markdown(registry: dict[str, Any]) -> str:
    """Render the registry as deterministic Markdown.

    Pure function of the parsed registry.  Names, parents, kinds and
    lifecycle are emitted in sorted order; identical input yields a
    byte-identical document.
    """
    root = registry["root"]
    profiles = registry["profiles"]
    lines: list[str] = []
    lines.append("# Profile Tier Registry (generated)")
    lines.append("")
    lines.append(
        "**Source of truth:** `governance/profile-registry.yaml` — this file is "
        "generated; edit the YAML, never the Markdown."
    )
    lines.append("")
    root_desc = root.get("description") or ""
    lines.append(f"Root: **{root['name']}**" + (f" — {root_desc}" if root_desc else ""))
    lines.append("")

    def _row(p: dict[str, Any]) -> str:
        parent = p["parent"] or root["name"]
        return (
            f"| {p['name']} | {p['kind']} | {parent} | {p['lifecycle']} | "
            f"{', '.join(p['domains'])} | {p['gateway_unit'] or '—'} |"
        )

    lines.append("| Profile | Kind | Parent | Lifecycle | Domains | Gateway unit |")
    lines.append("|---|---|---|---|---|---|")
    lines.extend(_row(p) for p in profiles)
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Discrepancy reporting (strictly read-only)
# ---------------------------------------------------------------------------


def _discrepancy(kind: str, name: str, detail: str) -> dict[str, str]:
    return {"kind": kind, "profile": name, "detail": detail}


def compare_with_filesystem(
    registry: dict[str, Any], profiles_dir: str | os.PathLike[str],
) -> list[dict[str, str]]:
    """Compare registry entries with profile directories.  Never mutates.

    Surfaces unregistered filesystem profiles explicitly rather than
    silently attaching them to the root.
    """
    root_name = registry["root"]["name"]
    registered = {p["name"] for p in registry["profiles"]} | {root_name}
    discrepancies: list[dict[str, str]] = []
    try:
        on_disk = sorted(
            p.name for p in Path(profiles_dir).iterdir() if p.is_dir()
        )
    except OSError as exc:
        return [{"kind": "profiles_dir_unreadable", "profile": "", "detail": str(exc)}]
    registered_set = registered
    for dirname in on_disk:
        if dirname not in registered_set:
            discrepancies.append({
                "kind": "unregistered_profile",
                "profile": dirname,
                "detail": "profile directory exists on disk but is not in the registry",
            })
    for entry in registry["profiles"]:
        if entry["name"] not in on_disk and entry["name"] != root_name:
            discrepancies.append({
                "kind": "missing_profile_dir",
                "profile": entry["name"],
                "detail": "registered profile has no directory on disk",
            })
    return discrepancies


def compare_with_gateway_inventory(
    registry: dict[str, Any], systemd_units: Iterable[str],
) -> list[dict[str, str]]:
    """Compare registry expectations with an effective gateway inventory.

    ``systemd_units`` is a list of unit names (e.g. ``hermes-gateway-remii``).
    Read-only: the caller supplies the inventory; nothing is executed and no
    service state is changed here.
    """
    units = set(systemd_units)
    discrepancies: list[dict[str, str]] = []
    for entry in registry["profiles"]:
        unit = entry["gateway_unit"]
        if unit and unit not in units:
            discrepancies.append({
                "kind": "declared_unit_not_in_inventory",
                "profile": entry["name"],
                "detail": f"gateway_unit {unit} not present in effective inventory",
            })
    for unit in sorted(units):
        if unit == "hermes-gateway":
            continue
        owners = [
            p["name"] for p in registry["profiles"] if p["gateway_unit"] == unit
        ]
        if not owners:
            discrepancies.append({
                "kind": "unregistered_gateway_unit",
                "profile": "",
                "detail": f"unit {unit} is active but mapped to no registry profile",
            })
    return discrepancies
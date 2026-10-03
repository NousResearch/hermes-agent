"""Typed, reversible harness overlays.

The harness manifest is deliberately separate from ``config.yaml``.  It is a small,
versioned control plane for bounded, non-security-critical runtime knobs that an
operator (or an evolver) can inspect and change without rewriting source or the
user's config.  Overlays are only active when they were authored against the
currently running stock revision; an upgrade therefore invalidates them instead
of silently replaying values against a changed codebase.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import hermes_yaml as yaml

from hermes_constants import get_hermes_home
from utils import atomic_yaml_write, fast_safe_load, file_signature

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_MANIFEST_NAME = "harness.yaml"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class HarnessManifestError(ValueError):
    """The manifest cannot be safely interpreted."""


@dataclass(frozen=True)
class HarnessKey:
    path: str
    value_type: str
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    restart_required: bool = False
    safety: str = "tunable"
    description: str = ""


# This registry is intentionally small.  These settings already have stable runtime readers and
# do not grant capabilities, bypass approvals, alter credentials, or weaken redaction/blocklists.
# Keep security-sensitive surfaces out of this substrate even if they happen to be present in the
# normal config schema.
_REGISTRY: tuple[HarnessKey, ...] = (
    HarnessKey(
        "agent.max_turns", "integer", None, 1, 10000,
        description="Maximum agent iterations for one run.",
    ),
    HarnessKey(
        "agent.gateway_timeout", "integer", 1800, 0, 86400,
        description="Maximum seconds a gateway turn may run.",
    ),
    HarnessKey(
        "agent.gateway_turn_lease_timeout", "integer", 5, 0, 3600,
        description="Seconds before an idle gateway turn lease expires.",
    ),
    HarnessKey(
        "compression.threshold", "number", 0.50, 0.01, 0.99,
        description="Fraction of context at which compression begins.",
    ),
    HarnessKey(
        "compression.target_ratio", "number", 0.20, 0.01, 0.99,
        description="Fraction of the threshold retained after compression.",
    ),
    HarnessKey(
        "delegation.max_iterations", "integer", 250, 1, 10000,
        description="Maximum iterations allowed for delegated work.",
    ),
)
_REGISTRY_BY_PATH = {entry.path: entry for entry in _REGISTRY}


def manifest_path(home: Path | None = None) -> Path:
    return (home or get_hermes_home()) / _MANIFEST_NAME


def manifest_signature(home: Path | None = None) -> tuple[int, int, int, int]:
    """Return a cache-friendly signature for the manifest, or the all-zero sentinel."""
    try:
        stat_result = manifest_path(home).stat()
    except OSError:
        return (0, 0, 0, 0)
    return file_signature(stat_result)


def stock_revision() -> str:
    """Identify the running stock code revision without making manifest import expensive."""
    try:
        from hermes_cli.version_info import get_version_info

        info = get_version_info()
        return info.commit or info.derived_version or "unknown"
    except Exception:  # pragma: no cover - defensive for early bootstrap/install failures
        return "unknown"


def registry() -> tuple[HarnessKey, ...]:
    return _REGISTRY


def registry_entry(path: str) -> HarnessKey:
    try:
        return _REGISTRY_BY_PATH[path]
    except KeyError as exc:
        raise HarnessManifestError(
            f"Unknown or non-tunable harness key {path!r}. "
            f"Allowed keys: {', '.join(_REGISTRY_BY_PATH)}"
        ) from exc


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _coerce_value(entry: HarnessKey, value: Any) -> Any:
    if entry.value_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise HarnessManifestError(f"{entry.path} expects an integer")
    elif entry.value_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HarnessManifestError(f"{entry.path} expects a number")
        value = float(value)
    if value is None and entry.path == "agent.max_turns":
        return None
    if value is None:
        raise HarnessManifestError(f"{entry.path} cannot be null")
    if entry.minimum is not None and value < entry.minimum:
        raise HarnessManifestError(f"{entry.path} must be >= {entry.minimum:g}")
    if entry.maximum is not None and value > entry.maximum:
        raise HarnessManifestError(f"{entry.path} must be <= {entry.maximum:g}")
    return value


def _validate_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        raise HarnessManifestError(
            f"{label} must match {_NAME_RE.pattern} (letters, digits, '.', '_' and '-')"
        )
    return value


def _validate_manifest(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {"schema_version": SCHEMA_VERSION, "stock_revision": stock_revision(), "overlays": []}
    if not isinstance(raw, dict):
        raise HarnessManifestError("manifest top level must be a mapping")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise HarnessManifestError(
            f"unsupported harness manifest schema_version={raw.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )
    revision = raw.get("stock_revision")
    if not isinstance(revision, str) or not revision:
        raise HarnessManifestError("stock_revision must be a non-empty string")
    overlays = raw.get("overlays", [])
    if not isinstance(overlays, list):
        raise HarnessManifestError("overlays must be a list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(overlays):
        if not isinstance(item, dict):
            raise HarnessManifestError(f"overlay {index} must be a mapping")
        overlay_id = _validate_name(item.get("id"), f"overlay {index}.id")
        if overlay_id in seen:
            raise HarnessManifestError(f"duplicate overlay id {overlay_id!r}")
        seen.add(overlay_id)
        name = _validate_name(item.get("name", overlay_id), f"overlay {index}.name")
        reason = item.get("reason", "")
        authored_against = item.get("authored_against", revision)
        created_at = item.get("created_at", "")
        if not isinstance(reason, str) or len(reason) > 2000:
            raise HarnessManifestError(f"overlay {overlay_id!r}.reason must be a string <= 2000 chars")
        if not isinstance(authored_against, str) or not authored_against:
            raise HarnessManifestError(f"overlay {overlay_id!r}.authored_against must be a string")
        if not isinstance(created_at, str):
            raise HarnessManifestError(f"overlay {overlay_id!r}.created_at must be a string")
        values = item.get("values", {})
        if not isinstance(values, dict) or not values:
            raise HarnessManifestError(f"overlay {overlay_id!r}.values must be a non-empty mapping")
        clean_values: dict[str, Any] = {}
        for path, value in values.items():
            if not isinstance(path, str):
                raise HarnessManifestError(f"overlay {overlay_id!r} contains a non-string key")
            clean_values[path] = _coerce_value(registry_entry(path), value)
        normalized.append({
            "id": overlay_id,
            "name": name,
            "reason": reason,
            "authored_against": authored_against,
            "created_at": created_at,
            "values": clean_values,
        })
    return {"schema_version": SCHEMA_VERSION, "stock_revision": revision, "overlays": normalized}


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    path = path or manifest_path()
    try:
        with path.open(encoding="utf-8-sig") as handle:
            raw = fast_safe_load(handle)
    except FileNotFoundError:
        return _validate_manifest(None)
    except OSError as exc:
        raise HarnessManifestError(f"cannot read {path}: {exc}") from exc
    except Exception as exc:
        raise HarnessManifestError(f"cannot parse {path}: {exc}") from exc
    return _validate_manifest(raw)


def save_manifest(manifest: dict[str, Any], path: Path | None = None) -> None:
    path = path or manifest_path()
    clean = _validate_manifest(manifest)
    atomic_yaml_write(path, clean, sort_keys=False, create_mode=0o600)


def _active_overlays(manifest: dict[str, Any], revision: str) -> Iterable[dict[str, Any]]:
    for overlay in manifest["overlays"]:
        if overlay["authored_against"] == revision:
            yield overlay


def active_values(manifest: dict[str, Any], revision: str | None = None) -> dict[str, Any]:
    revision = revision or stock_revision()
    values: dict[str, Any] = {}
    for overlay in _active_overlays(manifest, revision):
        values.update(overlay["values"])
    return values


def apply_active_overlays(config: dict[str, Any], *, path: Path | None = None) -> dict[str, Any]:
    """Apply valid current-revision overlays, leaving stale overlays inert.

    Invalid user-authored manifests are ignored by the runtime (with a warning); a bad optional
    tuning file must not prevent Hermes from starting. The CLI remains strict and reports the
    exact validation error so the operator can repair or revert it.
    """
    try:
        manifest = load_manifest(path)
    except HarnessManifestError as exc:
        logger.warning("Ignoring invalid harness manifest: %s", exc)
        return config
    revision = stock_revision()
    stale = [o["id"] for o in manifest["overlays"] if o["authored_against"] != revision]
    if stale:
        logger.warning("Ignoring stale harness overlays for stock revision %s: %s", revision, ", ".join(stale))
    result = copy.deepcopy(config)
    for dotted, value in active_values(manifest, revision).items():
        cursor = result
        parts = dotted.split(".")
        for part in parts[:-1]:
            child = cursor.get(part)
            if not isinstance(child, dict):
                child = {}
                cursor[part] = child
            cursor = child
        cursor[parts[-1]] = value
    return result


def effective_values(manifest: dict[str, Any], revision: str | None = None) -> dict[str, Any]:
    revision = revision or stock_revision()
    values = {entry.path: entry.default for entry in _REGISTRY}
    values.update(active_values(manifest, revision))
    return values


def fingerprint(manifest: dict[str, Any] | None = None, revision: str | None = None) -> str:
    revision = revision or stock_revision()
    manifest = manifest or load_manifest()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stock_revision": revision,
        "overlays": [
            {"id": o["id"], "authored_against": o["authored_against"], "values": o["values"]}
            for o in manifest["overlays"]
            if o["authored_against"] == revision
        ],
        "effective_values": effective_values(manifest, revision),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def explain(path: str, manifest: dict[str, Any] | None = None, revision: str | None = None) -> dict[str, Any]:
    entry = registry_entry(path)
    revision = revision or stock_revision()
    manifest = manifest or load_manifest()
    sources = [
        {"overlay": overlay["id"], "reason": overlay["reason"], "value": overlay["values"][path]}
        for overlay in _active_overlays(manifest, revision)
        if path in overlay["values"]
    ]
    return {
        "key": entry.path,
        "type": entry.value_type,
        "default": entry.default,
        "active_value": effective_values(manifest, revision)[path],
        "minimum": entry.minimum,
        "maximum": entry.maximum,
        "restart_required": entry.restart_required,
        "safety": entry.safety,
        "description": entry.description,
        "stock_revision": revision,
        "sources": sources,
    }


def set_value(key_path: str, value: Any, *, overlay: str, reason: str, manifest_path_: Path | None = None) -> dict[str, Any]:
    entry = registry_entry(key_path)
    value = _coerce_value(entry, value)
    overlay = _validate_name(overlay, "overlay")
    if not isinstance(reason, str) or not reason.strip():
        raise HarnessManifestError("reason must be a non-empty string")
    path = manifest_path_ or manifest_path()
    manifest = load_manifest(path)
    revision = stock_revision()
    target = next((item for item in manifest["overlays"] if item["id"] == overlay), None)
    if target is None:
        target = {
            "id": overlay, "name": overlay, "reason": reason.strip(),
            "authored_against": revision, "created_at": _now(), "values": {},
        }
        manifest["overlays"].append(target)
    elif target["authored_against"] != revision:
        target["authored_against"] = revision
        target["created_at"] = _now()
    manifest["stock_revision"] = revision
    target["reason"] = reason.strip()
    target["values"][key_path] = value
    save_manifest(manifest, path)
    return manifest


def revert_overlay(overlay: str, manifest_path_: Path | None = None) -> dict[str, Any]:
    _validate_name(overlay, "overlay")
    path = manifest_path_ or manifest_path()
    manifest = load_manifest(path)
    before = len(manifest["overlays"])
    manifest["overlays"] = [item for item in manifest["overlays"] if item["id"] != overlay]
    if len(manifest["overlays"]) == before:
        raise HarnessManifestError(f"overlay {overlay!r} does not exist")
    save_manifest(manifest, path)
    return manifest


def show_state(manifest: dict[str, Any] | None = None, revision: str | None = None) -> dict[str, Any]:
    revision = revision or stock_revision()
    manifest = manifest or load_manifest()
    return {
        "path": str(manifest_path()),
        "schema_version": SCHEMA_VERSION,
        "stock_revision": revision,
        "manifest_stock_revision": manifest["stock_revision"],
        "fingerprint": fingerprint(manifest, revision),
        "values": effective_values(manifest, revision),
        "overlays": [
            {
                **overlay,
                "active": overlay["authored_against"] == revision,
            }
            for overlay in manifest["overlays"]
        ],
    }


def parse_cli_value(text: str) -> Any:
    try:
        value = yaml.safe_load(text)
    except Exception:
        value = text
    return text if value is None and text.strip().lower() not in {"null", "~"} else value


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)

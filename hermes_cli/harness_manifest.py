"""Typed, profile-scoped harness overlays and reproducible harness identity.

The overlay surface is intentionally narrower than ``config.yaml``.  Only
runtime-tuning roots are admitted; credentials, approvals, security policy,
tools, plugins, hooks, terminals, browsers, and messaging adapters are never
agent-writable harness options.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from hermes_cli.config_defaults import DEFAULT_CONFIG
from hermes_constants import get_hermes_home
from utils import atomic_yaml_write, fast_safe_load

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_MANIFEST_NAME = "harness.yaml"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_LOCK = threading.RLock()
_WARNED_STALE: set[tuple[str, str]] = set()
_WARNED_INVALID: set[str] = set()

# An allow-list is deliberate: a future security-sensitive config root must not
# become agent-writable merely because it was added to DEFAULT_CONFIG.
_TUNABLE_ROOTS = frozenset({
    "agent", "auxiliary", "compression", "context", "context_file_max_chars",
    "context_file_read_timeout", "curator", "delegation", "display",
    "file_read_max_chars", "human_delay", "logging", "max_concurrent_sessions",
    "max_live_sessions", "mcp_discovery_timeout", "mcp_single_query_discovery_timeout",
    "moa", "prompt_caching", "paste_collapse_char_threshold",
    "paste_collapse_threshold", "paste_collapse_threshold_fallback", "stt",
    "streaming", "timezone", "tool_loop_guardrails", "tool_output", "tts",
    "vision", "voice", "wake_word",
})

# Bounds are contracts only where the runtime already has a meaningful bounded
# domain. Other numeric entries remain typed but unbounded.
_BOUNDS: Dict[str, tuple[Optional[float], Optional[float]]] = {
    "compression.threshold": (0.0, 1.0),
    "compression.target_ratio": (0.0, 1.0),
    "agent.max_turns": (1, None),
    "context_file_read_timeout": (0.0, None),
    "file_read_max_chars": (1, None),
    "max_concurrent_sessions": (1, None),
    "max_live_sessions": (1, None),
    "mcp_discovery_timeout": (0.0, None),
    "mcp_single_query_discovery_timeout": (0.0, None),
}


class HarnessError(ValueError):
    """A harness manifest or requested change is invalid."""


def manifest_path(home: Optional[Path] = None) -> Path:
    return (home or get_hermes_home()) / _MANIFEST_NAME


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _walk_defaults(node: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(node, dict):
        for key in sorted(node):
            path = f"{prefix}.{key}" if prefix else key
            yield from _walk_defaults(node[key], path)
        return
    # ``None`` does not declare a useful runtime type, so it cannot be changed
    # through the typed harness until its config declaration becomes typed.
    if node is not None:
        yield prefix, node


def harness_registry() -> Dict[str, Dict[str, Any]]:
    """Return the registry derived from safe leaves in ``DEFAULT_CONFIG``."""
    registry: Dict[str, Dict[str, Any]] = {}
    for path, default in _walk_defaults(DEFAULT_CONFIG):
        if path.partition(".")[0] not in _TUNABLE_ROOTS:
            continue
        minimum, maximum = _BOUNDS.get(path, (None, None))
        registry[path] = {
            "name": path,
            "type": _type_name(default),
            "default": copy.deepcopy(default),
            "minimum": minimum,
            "maximum": maximum,
            # Applying a new harness to an existing conversation would change
            # request behavior under a cached prompt/tool prefix. New sessions
            # pick it up; existing sessions retain their recorded identity.
            "restart_required": True,
        }
    return registry


def schema_fingerprint() -> str:
    return _json_hash(harness_registry())


def stock_revision() -> str:
    from hermes_cli import __version__

    return f"{__version__}:{schema_fingerprint()[:12]}"


def _empty_manifest() -> Dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "overlays": [], "history": []}


def _validate_manifest(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return _empty_manifest()
    if not isinstance(raw, dict):
        raise HarnessError("harness.yaml must contain a mapping")
    if raw.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise HarnessError(
            f"unsupported harness schema_version {raw.get('schema_version')!r}; expected {SCHEMA_VERSION}")
    overlays = raw.get("overlays", [])
    history = raw.get("history", [])
    if not isinstance(overlays, list) or not isinstance(history, list):
        raise HarnessError("harness overlays and history must be lists")
    names: set[str] = set()
    normalized = _empty_manifest()
    normalized["history"] = copy.deepcopy(history)
    registry = harness_registry()
    for index, overlay in enumerate(overlays):
        if not isinstance(overlay, dict):
            raise HarnessError(f"overlay {index + 1} must be a mapping")
        name = str(overlay.get("name") or "")
        if not _NAME_RE.fullmatch(name):
            raise HarnessError(f"overlay {index + 1} has an invalid name")
        if name in names:
            raise HarnessError(f"duplicate overlay name: {name}")
        names.add(name)
        values = overlay.get("values", {})
        if not isinstance(values, dict):
            raise HarnessError(f"overlay {name!r} values must be a mapping")
        for key, value in values.items():
            _validate_value(str(key), value, registry)
        normalized["overlays"].append({
            "name": name,
            "authored_against": str(overlay.get("authored_against") or "unknown"),
            "created_at": str(overlay.get("created_at") or ""),
            "updated_at": str(overlay.get("updated_at") or overlay.get("created_at") or ""),
            "reason": str(overlay.get("reason") or ""),
            "values": copy.deepcopy(values),
        })
    return normalized


def load_manifest(*, home: Optional[Path] = None, strict: bool = True) -> Dict[str, Any]:
    path = manifest_path(home)
    try:
        with path.open(encoding="utf-8") as handle:
            return _validate_manifest(fast_safe_load(handle))
    except FileNotFoundError:
        return _empty_manifest()
    except Exception as exc:
        if strict:
            if isinstance(exc, HarnessError):
                raise
            raise HarnessError(f"cannot read {path}: {exc}") from exc
        key = str(path)
        if key not in _WARNED_INVALID:
            _WARNED_INVALID.add(key)
            logger.warning("Ignoring invalid harness manifest %s; using stock/config.yaml values: %s", path, exc)
        return _empty_manifest()


def manifest_signature(home: Optional[Path] = None) -> tuple[int, int, int, int]:
    """A four-int signature compatible with config loader cache signatures."""
    try:
        from utils import file_signature

        return file_signature(manifest_path(home).stat())
    except OSError:
        return (0, 0, 0, 0)


def _get_path(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    node: Any = mapping
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def _set_path(mapping: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = mapping
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = copy.deepcopy(value)


def _validate_value(key: str, value: Any, registry: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
    entry = (registry or harness_registry()).get(key)
    if entry is None:
        raise HarnessError(
            f"{key!r} is not a tunable harness option (security and capability surfaces are excluded)")
    expected = entry["type"]
    valid = {
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "string": isinstance(value, str),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
    }[expected]
    if not valid:
        raise HarnessError(f"{key} expects {expected}, got {_type_name(value)}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if entry["minimum"] is not None and value < entry["minimum"]:
            raise HarnessError(f"{key} must be >= {entry['minimum']}")
        if entry["maximum"] is not None and value > entry["maximum"]:
            raise HarnessError(f"{key} must be <= {entry['maximum']}")


def parse_value(key: str, raw: str) -> Any:
    import yaml

    try:
        value = yaml.safe_load(raw)
    except Exception as exc:
        raise HarnessError(f"invalid YAML value for {key}: {exc}") from exc
    _validate_value(key, value)
    return value


def apply_harness_overlays(base: Dict[str, Any], *, home: Optional[Path] = None) -> Dict[str, Any]:
    """Apply ordered overlays to a copy of *base*; invalid manifests fall back to stock."""
    result = copy.deepcopy(base)
    manifest = load_manifest(home=home, strict=False)
    current = stock_revision()
    manifest_key = str(manifest_path(home))
    for overlay in manifest["overlays"]:
        authored = overlay["authored_against"]
        if authored != current and (manifest_key, overlay["name"]) not in _WARNED_STALE:
            _WARNED_STALE.add((manifest_key, overlay["name"]))
            logger.warning(
                "Harness overlay %s was authored against %s, current stock is %s; applying with stale flag",
                overlay["name"], authored, current)
        for key, value in overlay["values"].items():
            _set_path(result, key, value)
    return result


def _safe_effective_values(effective: Mapping[str, Any]) -> Dict[str, Any]:
    missing = object()
    values: Dict[str, Any] = {}
    for key in harness_registry():
        value = _get_path(effective, key, missing)
        if value is not missing:
            values[key] = value
    return values


def harness_identity(effective: Optional[Mapping[str, Any]] = None, *, home: Optional[Path] = None) -> Dict[str, Any]:
    if effective is None:
        from hermes_cli.config import load_config_readonly

        effective = load_config_readonly()
    manifest = load_manifest(home=home, strict=False)
    current = stock_revision()
    lineage = [
        {"name": item["name"], "authored_against": item["authored_against"], "updated_at": item["updated_at"]}
        for item in manifest["overlays"]
    ]
    return {
        "stock_revision": current,
        "schema_fingerprint": schema_fingerprint(),
        "overlay_lineage": lineage,
        "effective_config_fingerprint": _json_hash(_safe_effective_values(effective)),
        "stale_overlays": [item["name"] for item in manifest["overlays"] if item["authored_against"] != current],
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_manifest(manifest: Dict[str, Any], *, home: Optional[Path] = None) -> None:
    path = manifest_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_yaml_write(path, manifest, sort_keys=False)
    # The CLI process may have loaded config before dispatching this command.
    try:
        from hermes_cli import config as config_module
        from hermes_cli import config_effective

        config_module._LOAD_CONFIG_CACHE.clear()
        config_effective._EFFECTIVE_CACHE.clear()
    except Exception:
        logger.debug("Could not invalidate config caches after harness write", exc_info=True)


def set_overlay(name: str, key: str, value: Any, reason: str, *, home: Optional[Path] = None) -> Dict[str, Any]:
    if not _NAME_RE.fullmatch(name):
        raise HarnessError("overlay name must be 1-64 letters, numbers, dots, underscores, or hyphens")
    if not reason.strip():
        raise HarnessError("--reason is required for harness provenance")
    _validate_value(key, value)
    with _LOCK:
        manifest = load_manifest(home=home, strict=True)
        now = _now()
        overlay = next((item for item in manifest["overlays"] if item["name"] == name), None)
        before = None
        if overlay is None:
            overlay = {
                "name": name,
                "authored_against": stock_revision(),
                "created_at": now,
                "updated_at": now,
                "reason": reason.strip(),
                "values": {},
            }
            manifest["overlays"].append(overlay)
        else:
            before = overlay["values"].get(key)
            overlay["updated_at"] = now
            overlay["reason"] = reason.strip()
        overlay["values"][key] = copy.deepcopy(value)
        manifest["history"].append({
            "action": "set", "overlay": name, "key": key, "before": before,
            "after": copy.deepcopy(value), "reason": reason.strip(), "at": now,
            "authored_against": overlay["authored_against"],
        })
        manifest["history"] = manifest["history"][-100:]
        _write_manifest(manifest, home=home)
        return copy.deepcopy(overlay)


def revert_overlay(name: Optional[str], reason: str, *, home: Optional[Path] = None) -> Dict[str, Any]:
    if not reason.strip():
        raise HarnessError("--reason is required for harness provenance")
    with _LOCK:
        manifest = load_manifest(home=home, strict=True)
        if not manifest["overlays"]:
            raise HarnessError("there are no harness overlays to revert")
        index = len(manifest["overlays"]) - 1
        if name:
            index = next((i for i, item in enumerate(manifest["overlays"]) if item["name"] == name), -1)
            if index < 0:
                raise HarnessError(f"overlay not found: {name}")
        removed = manifest["overlays"].pop(index)
        manifest["history"].append({
            "action": "revert", "overlay": removed["name"], "values": copy.deepcopy(removed["values"]),
            "reason": reason.strip(), "at": _now(), "authored_against": stock_revision(),
        })
        manifest["history"] = manifest["history"][-100:]
        _write_manifest(manifest, home=home)
        return removed


def _diff(effective: Mapping[str, Any]) -> list[Dict[str, Any]]:
    result = []
    for key, entry in harness_registry().items():
        value = _get_path(effective, key, entry["default"])
        if value != entry["default"]:
            result.append({"key": key, "stock": entry["default"], "effective": value})
    return result


def command_show(*, as_json: bool = False) -> None:
    from hermes_cli.config import load_config

    effective = load_config()
    payload = {
        "identity": harness_identity(effective),
        "manifest": load_manifest(strict=True),
        "effective": _safe_effective_values(effective),
    }
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return
    identity = payload["identity"]
    print(f"Stock revision: {identity['stock_revision']}")
    print(f"Effective fingerprint: {identity['effective_config_fingerprint']}")
    print(f"Schema fingerprint: {identity['schema_fingerprint']}")
    overlays = payload["manifest"]["overlays"]
    print("Overlays: " + (", ".join(item["name"] for item in overlays) if overlays else "(stock)"))
    if identity["stale_overlays"]:
        print("Stale overlays: " + ", ".join(identity["stale_overlays"]))


def command_diff(*, as_json: bool = False) -> None:
    from hermes_cli.config import load_config

    changes = _diff(load_config())
    if as_json:
        print(json.dumps(changes, indent=2, sort_keys=True, ensure_ascii=False))
        return
    if not changes:
        print("Harness matches stock.")
        return
    for item in changes:
        print(f"{item['key']}: {item['stock']!r} -> {item['effective']!r}")


def command_explain(key: str, *, as_json: bool = False) -> None:
    registry = harness_registry()
    if key not in registry:
        raise HarnessError(f"unknown or excluded harness option: {key}")
    from hermes_cli.config import load_config

    manifest = load_manifest(strict=True)
    effective = load_config()
    payload = dict(registry[key])
    payload["effective"] = _get_path(effective, key, payload["default"])
    payload["overlays"] = [
        {"name": item["name"], "value": item["values"][key], "reason": item["reason"],
         "authored_against": item["authored_against"]}
        for item in manifest["overlays"] if key in item["values"]
    ]
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return
    print(f"{key} ({payload['type']}, restart required: yes)")
    print(f"Stock: {payload['default']!r}")
    print(f"Effective: {payload['effective']!r}")
    for item in payload["overlays"]:
        print(f"Overlay {item['name']}: {item['value']!r} — {item['reason']}")

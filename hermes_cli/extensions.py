"""Inspect-only Hermes extension registry CLI support.

Phase 1 is deliberately read-only: parse manifests, search static indexes, and
render permission/risk receipts. No install/update/remove side effects here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

_VALID_KINDS = {"skill", "plugin", "mcp-server", "extension-pack"}
_REQUIRED_MANIFEST_FIELDS = {
    "manifest_version",
    "kind",
    "id",
    "name",
    "version",
    "description",
    "publisher",
    "compatibility",
    "contents",
    "permissions",
}
_EXTENSION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*/[a-z0-9][a-z0-9_.-]*$")


class ExtensionManifestError(ValueError):
    """Raised when an extension manifest is malformed or unsafe."""


class ExtensionRegistryError(ValueError):
    """Raised when a registry index is malformed."""


def load_extension_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate an inspect-only ``extension.yaml`` manifest."""
    manifest_path = Path(path)
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ExtensionManifestError(f"Invalid YAML: {exc}") from exc
    except OSError as exc:
        raise ExtensionManifestError(f"Could not read manifest: {exc}") from exc

    if not isinstance(raw, dict):
        raise ExtensionManifestError("Manifest must be a mapping")

    missing = sorted(_REQUIRED_MANIFEST_FIELDS - raw.keys())
    if missing:
        raise ExtensionManifestError(f"Missing required field(s): {', '.join(missing)}")

    kind = raw.get("kind")
    if kind not in _VALID_KINDS:
        raise ExtensionManifestError(
            f"Invalid kind {kind!r}; expected one of {', '.join(sorted(_VALID_KINDS))}"
        )

    extension_id = raw.get("id")
    if not isinstance(extension_id, str) or not _EXTENSION_ID_RE.fullmatch(extension_id):
        raise ExtensionManifestError(
            "Invalid id; expected namespace/name using lowercase letters, numbers, dots, underscores, or hyphens"
        )

    _expect_mapping(raw.get("publisher"), "publisher")
    _expect_mapping(raw.get("compatibility"), "compatibility")
    _validate_contents(raw.get("contents"))
    raw["permissions"] = _validate_permissions(raw.get("permissions"))
    raw.setdefault("risk", {})
    return raw


def load_registry_index(path: str | Path) -> dict[str, Any]:
    """Load a local static registry index JSON file."""
    index_path = Path(path)
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExtensionRegistryError(f"Invalid registry JSON: {exc}") from exc
    except OSError as exc:
        raise ExtensionRegistryError(f"Could not read registry index: {exc}") from exc

    if not isinstance(raw, dict):
        raise ExtensionRegistryError("Registry index must be a JSON object")
    if "schema_version" not in raw:
        raise ExtensionRegistryError("Registry index missing schema_version")
    extensions = raw.get("extensions")
    if not isinstance(extensions, list):
        raise ExtensionRegistryError("Registry index extensions must be a list")
    for idx, entry in enumerate(extensions):
        if not isinstance(entry, dict):
            raise ExtensionRegistryError(f"extensions[{idx}] must be an object")
        missing = [
            field
            for field in ("id", "name", "kind", "version", "description", "publisher")
            if field not in entry
        ]
        if missing:
            raise ExtensionRegistryError(
                f"extensions[{idx}] missing required field(s): {', '.join(missing)}"
            )
        if "tags" in entry and not isinstance(entry["tags"], list):
            raise ExtensionRegistryError(f"extensions[{idx}].tags must be a list")
    return raw


def search_registry(index: dict[str, Any], query: str) -> list[dict[str, Any]]:
    """Search registry entries by ID, name, description, tags, publisher, kind, risk, trust."""
    needle = query.strip().lower()
    entries = index.get("extensions", [])
    if not needle:
        return list(entries)
    return [entry for entry in entries if needle in _registry_haystack(entry)]


def render_search_results(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "No matching extensions found."

    header = f"{'ID':<40} {'Kind':<16} {'Risk':<8} {'Trust':<10} Description"
    lines = [header, "-" * len(header)]
    for entry in entries:
        lines.append(
            f"{entry['id']:<40} "
            f"{entry['kind']:<16} "
            f"{entry.get('risk_level', 'unknown'):<8} "
            f"{entry.get('trust_level', 'community'):<10} "
            f"{_truncate(str(entry.get('description', '')), 78)}"
        )
    return "\n".join(lines)


def render_inspect_receipt(manifest: dict[str, Any], *, source: str | None = None) -> str:
    permissions = manifest.get("permissions", {})
    contents = manifest.get("contents", {})
    risk = manifest.get("risk", {}) if isinstance(manifest.get("risk"), dict) else {}
    publisher = manifest.get("publisher", {}) if isinstance(manifest.get("publisher"), dict) else {}
    risk_level = risk.get("declared_level", "unknown")
    risk_rationale = risk.get("rationale", "No risk rationale declared.")

    lines = [
        f"# {manifest.get('name', 'Unnamed Extension')}",
        "",
        f"ID: {manifest.get('id')}",
        f"Kind: {manifest.get('kind')}",
        f"Version: {manifest.get('version')}",
        f"Publisher: {publisher.get('name', 'unknown')}",
        f"Risk: {risk_level}",
        f"Description: {manifest.get('description')}",
    ]
    if source:
        lines.append(f"Source: {source}")
    if manifest.get("repository"):
        lines.append(f"Repository: {manifest['repository']}")
    if manifest.get("homepage"):
        lines.append(f"Homepage: {manifest['homepage']}")

    lines.extend([
        "",
        "## Contents",
        *_render_contents(contents),
        "",
        "## Compatibility",
        *_render_mapping(manifest.get("compatibility", {})),
        "",
        "## Permission Preview",
        *_render_permissions(permissions, contents),
        "",
        "## Risk Rationale",
        str(risk_rationale),
        "",
        "## Installability Notes",
        "Phase 1 is inspect-only: no files, config, MCP servers, cron jobs, or plugins are installed.",
    ])
    return "\n".join(lines).rstrip() + "\n"


def extensions_command(args) -> int:
    """Dispatch ``hermes extensions`` subcommands."""
    action = getattr(args, "extensions_action", None)
    try:
        if action == "search":
            index = load_registry_index(args.index)
            matches = search_registry(index, args.query)
            print(render_search_results(matches))
            return 0 if matches else 1
        if action == "inspect":
            manifest = load_extension_manifest(args.manifest)
            print(render_inspect_receipt(manifest, source=str(args.manifest)), end="")
            return 0
    except (ExtensionManifestError, ExtensionRegistryError) as exc:
        print(f"Error: {exc}")
        return 2

    print("Usage: hermes extensions {search,inspect} ...")
    return 2


def _validate_contents(contents: Any) -> None:
    mapping = _expect_mapping(contents, "contents")
    for section, entries in mapping.items():
        if entries is None:
            continue
        if not isinstance(entries, list):
            raise ExtensionManifestError(f"contents.{section} must be a list")
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ExtensionManifestError(f"contents.{section}[{idx}] must be a mapping")
            if "path" in entry:
                _validate_relative_path(str(entry["path"]), f"contents.{section}[{idx}].path")


def _validate_permissions(permissions: Any) -> dict[str, Any]:
    raw = _expect_mapping(permissions, "permissions")
    env = _expect_list(raw.get("env", []), "permissions.env")
    for idx, item in enumerate(env):
        if not isinstance(item, dict):
            raise ExtensionManifestError(f"permissions.env[{idx}] must be a mapping")
        if not item.get("name"):
            raise ExtensionManifestError(f"permissions.env[{idx}].name is required")

    network = _expect_mapping(raw.get("network", {}), "permissions.network")
    _expect_list(network.get("hosts", []), "permissions.network.hosts")

    filesystem = _expect_mapping(raw.get("filesystem", {}), "permissions.filesystem")
    _expect_list(filesystem.get("reads", []), "permissions.filesystem.reads")
    _expect_list(filesystem.get("writes", []), "permissions.filesystem.writes")

    _expect_list(raw.get("toolsets", []), "permissions.toolsets")

    shell = _expect_mapping(raw.get("shell", {}), "permissions.shell")
    _expect_list(shell.get("allowed_commands", []), "permissions.shell.allowed_commands")
    if not isinstance(raw.get("outbound_messages", False), bool):
        raise ExtensionManifestError("permissions.outbound_messages must be a boolean")
    return raw


def _registry_haystack(entry: dict[str, Any]) -> str:
    values = [
        entry.get("id", ""),
        entry.get("name", ""),
        entry.get("description", ""),
        entry.get("publisher", ""),
        entry.get("kind", ""),
        entry.get("trust_level", ""),
        entry.get("risk_level", ""),
        *entry.get("tags", []),
    ]
    return "\n".join(str(value) for value in values).lower()


def _render_contents(contents: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for section, entries in contents.items():
        if not entries:
            continue
        lines.append(f"- {section}:")
        for entry in entries:
            if isinstance(entry, dict):
                label = entry.get("name") or entry.get("path") or entry
            else:
                label = entry
            lines.append(f"  - {label}")
    return lines or ["- none declared"]


def _render_mapping(mapping: Any) -> list[str]:
    if not isinstance(mapping, dict) or not mapping:
        return ["- none declared"]
    return [f"- {key}: {value}" for key, value in mapping.items()]


def _render_permissions(permissions: dict[str, Any], contents: dict[str, Any]) -> list[str]:
    network = permissions.get("network", {}) if isinstance(permissions.get("network"), dict) else {}
    filesystem = permissions.get("filesystem", {}) if isinstance(permissions.get("filesystem"), dict) else {}
    shell = permissions.get("shell", {}) if isinstance(permissions.get("shell"), dict) else {}
    lines = [
        "Env Vars:",
        *_indent(_render_env_vars(permissions.get("env", []))),
        "Network Hosts:",
        *_indent(network.get("hosts", []) or ["none"]),
        "Filesystem Reads:",
        *_indent(filesystem.get("reads", []) or ["none"]),
        "Filesystem Writes:",
        *_indent(filesystem.get("writes", []) or ["none"]),
        "Toolsets:",
        *_indent(permissions.get("toolsets", []) or ["none"]),
        "Shell Commands:",
        *_indent([*(shell.get("allowed_commands", []) or ["none"]), f"arbitrary shell: {'yes' if shell.get('arbitrary_shell') else 'no'}"]),
        "MCP Servers:",
        *_indent(_render_named_content(contents.get("mcp_servers", []))),
        "Cron Recipes:",
        *_indent(_render_path_content(contents.get("cron_recipes", []))),
        "Outbound Messages:",
        *_indent(["yes" if permissions.get("outbound_messages") else "no"]),
    ]
    return lines


def _render_env_vars(env_vars: Any) -> list[str]:
    if not env_vars:
        return ["none"]
    rendered: list[str] = []
    for env in env_vars:
        attrs = []
        if env.get("required"):
            attrs.append("required")
        if env.get("secret"):
            attrs.append("secret")
        suffix = f" ({', '.join(attrs)})" if attrs else ""
        rendered.append(f"{env.get('name')}{suffix}")
    return rendered


def _render_named_content(entries: Any) -> list[str]:
    if not entries:
        return ["none"]
    return [str(entry.get("name", entry)) if isinstance(entry, dict) else str(entry) for entry in entries]


def _render_path_content(entries: Any) -> list[str]:
    if not entries:
        return ["none"]
    return [str(entry.get("path", entry)) if isinstance(entry, dict) else str(entry) for entry in entries]


def _expect_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExtensionManifestError(f"{field_name} must be a mapping")
    return value


def _expect_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ExtensionManifestError(f"{field_name} must be a list")
    return value


def _validate_relative_path(value: str, field_name: str) -> None:
    path = PurePosixPath(value.replace("\\", "/"))
    if value.startswith("/") or path.is_absolute() or ".." in path.parts:
        raise ExtensionManifestError(f"Unsafe path traversal in {field_name}: {value}")


def _indent(values: list[Any]) -> list[str]:
    return [f"  - {value}" for value in values]


def _truncate(value: str, max_len: int) -> str:
    return value if len(value) <= max_len else value[: max_len - 1] + "…"

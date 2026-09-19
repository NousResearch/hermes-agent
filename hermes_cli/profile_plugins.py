"""Offline profile copies of installer-owned marketplace package files.

The caller holds the source installation lock through profile publication. This
module neither imports providers nor changes activation/settings. Inventories
are local integrity metadata, not authenticated provenance or security review.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import tempfile
from typing import NoReturn
from urllib.parse import urlsplit

import yaml

from hermes_cli import plugin_catalog, plugin_inventory
from hermes_cli.plugin_inventory import PluginInventoryError, checked_stat, read_regular

_METADATA = ".install-metadata.json"
_CATALOG = ".hermes-catalog.json"
_SHA = re.compile(r"[0-9a-fA-F]{40}")
_NAME = re.compile(r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)?")


class PluginSnapshotError(ValueError):
    """A package cannot safely be included in a profile clone."""


def _refuse(reason: str, record: dict | None = None) -> NoReturn:
    source, revision = "<saved-source-including-#subdir>", "<saved-40-character-commit-SHA>"
    if record is not None:
        source, revision = record["source"], record["revision"]
    raise PluginSnapshotError(
        f"Plugin snapshot refused: {reason}. Repair the source installation first: "
        "back up local changes/state outside plugins (force reinstall replaces the package), "
        "then in the source profile run "
        f"`hermes -p <profile> plugins install {shlex.quote(source)} --ref {shlex.quote(revision)} --force`. "
        "Keep the saved source INCLUDING #subdir and exact saved revision. "
        "Legacy installs without an installer inventory need explicit reinstall; "
        "never infer ownership from their used directories."
    ) from None


def _json(path: Path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _refuse("duplicate metadata key")
            result[key] = value
        return result
    return json.loads(read_regular(path), object_pairs_hook=unique)


def _source(source) -> str:
    if not isinstance(source, str) or not source or any(c.isspace() for c in source):
        _refuse("missing or malformed source metadata")
    repo, separator, subdir = source.partition("#")
    if separator:
        plugin_inventory.relative_file(subdir)
    if repo.startswith("git@") and re.fullmatch(r"git@[A-Za-z0-9.-]+:[A-Za-z0-9_./-]+", repo):
        return repo
    parsed = urlsplit(repo)
    if (parsed.scheme not in {"https", "http", "ssh", "file"} or parsed.query or parsed.fragment
            or parsed.password or (parsed.username and not (parsed.scheme == "ssh" and parsed.username == "git"))
            or not parsed.path or (parsed.scheme != "file" and not parsed.hostname)):
        _refuse("source is credential-bearing or malformed")
    return repo


def _offline_removals(home: Path):
    paths = [plugin_catalog.get_catalog_dir() / "removed.yaml", home / "cache" / "plugin-catalog.json"]
    entries, versions = [], {}
    for index, path in enumerate(paths):
        if not path.exists() and not path.is_symlink():
            if index == 0:
                _refuse("shipped removed-plugin list is missing")
            versions[path] = None
            continue
        checked_stat(path.parent)
        raw = read_regular(path)
        versions[path] = raw
        data = yaml.safe_load(raw) if index == 0 else _json(path)
        if not isinstance(data, dict) or not isinstance(data.get("removed"), list):
            _refuse("malformed removed-plugin list")
        for row in data["removed"]:
            if (not isinstance(row, dict) or not isinstance(row.get("name"), str) or not row["name"]
                    or any(not isinstance(row.get(k, ""), str) for k in ("repo", "reason", "date"))):
                _refuse("malformed removed-plugin entry")
            entries.append(plugin_catalog.RemovedEntry(**{k: row.get(k, "") for k in ("name", "repo", "reason", "date")}))
    return entries, versions


def _package_roots(root: Path, names) -> set[str]:
    """Check install roots only: unknown runtime contents are not our inventory."""
    checked_stat(root)
    found = set()
    categories = {name.split("/")[0] for name in names if "/" in name}
    for path in root.iterdir():
        if path.name == _METADATA:
            continue
        checked_stat(path)
        if not path.is_dir():
            _refuse("unowned plugin inventory entry")
        if path.name in categories:
            for child in path.iterdir():
                checked_stat(child)
                if not child.is_dir():
                    _refuse("unowned plugin category entry")
                found.add(path.name + "/" + child.name)
        else:
            found.add(path.name)
    return found


def _catalog(package: Path, record: dict, repo: str) -> bytes:
    sidecar = _json(package / _CATALOG)
    if (not isinstance(sidecar, dict)
            or set(sidecar) - {"catalog_name", "repo", "sha", "tier", "installed_at"}
            or not all(isinstance(sidecar.get(k), str) and sidecar[k] for k in ("catalog_name", "repo", "sha"))
            or not _SHA.fullmatch(sidecar["sha"])
            or any(not isinstance(v, str) for v in sidecar.values())):
        _refuse("malformed catalog provenance", record)
    if _source(sidecar["repo"]) != sidecar["repo"] or sidecar["repo"] != repo or sidecar["sha"].lower() != record["revision"].lower():
        _refuse("catalog identity disagrees with installed source/revision", record)
    return read_regular(package / _CATALOG)


def _manifest_names(package: Path) -> list[str]:
    names = []
    for filename in ("plugin.yaml", "plugin.yml", "plugin.json", ".claude-plugin/plugin.json"):
        path = package / filename
        if not path.exists():
            continue
        data = _json(path) if filename.endswith(".json") else yaml.safe_load(read_regular(path))
        if not isinstance(data, dict):
            _refuse("malformed plugin manifest")
        name = data.get("name")
        if name is not None:
            if not isinstance(name, str) or not name:
                _refuse("malformed plugin manifest name")
            names.append(name)
    return names


def copy_profile_plugins(source_home: Path, staging_home: Path) -> dict[str, list[str]]:
    """Return ``{copied: [keys], warnings: [text]}``; never publish partial copies.

    Caller MUST hold ``plugin_installation_lock(source_home)`` until the whole
    profile is published. Staging must be private and have no plugins directory.
    Catalog copies retain update/remove identity without Git or dependencies.
    Legacy Git and draft snapshot-only installs require explicit pristine repair.
    """
    source_home, staging_home = Path(source_home), Path(staging_home)
    root, destination = source_home / "plugins", staging_home / "plugins"
    report: dict[str, list[str]] = {"copied": [], "warnings": []}
    try:
        checked_stat(source_home)
        checked_stat(staging_home)
        if destination.exists() or destination.is_symlink():
            _refuse("destination plugins already exists")
        if not root.exists() and not root.is_symlink():
            return report
        checked_stat(root)
        if not any(root.iterdir()):
            return report
        metadata_bytes = read_regular(root / _METADATA)
        records = _json(root / _METADATA)
        if not isinstance(records, dict) or any(not _NAME.fullmatch(name) for name in records):
            _refuse("malformed install metadata")
        if _package_roots(root, records) != set(records):
            _refuse("missing, stale or ambiguous install metadata")
        removals, removal_versions = _offline_removals(source_home)
        sidecars = {}
        with tempfile.TemporaryDirectory(prefix=".plugin-snapshot-", dir=staging_home) as temporary:
            staged = Path(temporary) / "plugins"
            staged.mkdir()
            for name, record in records.items():
                if (not isinstance(record, dict) or not {"source", "revision", "pinned"} <= set(record)
                        or not isinstance(record["revision"], str) or not _SHA.fullmatch(record["revision"])
                        or type(record["pinned"]) is not bool):
                    _refuse("malformed install record")
                repo = _source(record["source"])
                if "files" not in record:
                    _refuse("missing installer-owned pristine file inventory", record)
                if set(record) - {"source", "revision", "pinned", "files"}:
                    _refuse("unsupported install record", record)
                package = root / name
                sidecars[name] = _catalog(package, record, repo)
                sidecar = json.loads(sidecars[name])
                plugin_inventory.copy_install_inventory(package, staged / name, record["files"])
                candidates = [name, package.name, repo, sidecar["catalog_name"], *_manifest_names(staged / name)]
                if any(plugin_catalog.match_removed(candidate, removals) for candidate in candidates):
                    _refuse("package matches offline removed-plugin list", record)
                (staged / name / _CATALOG).write_bytes(sidecars[name])
                (staged / name / _CATALOG).chmod(0o600)
                report["copied"].append(name)
                report["warnings"].append(
                    f"{name}: copied installer-owned files only; no Git history, runtime state or dependencies. "
                    "Catalog update/remove remain available. Local integrity is not a security review.")
            (staged / _METADATA).write_bytes(metadata_bytes)
            (staged / _METADATA).chmod(0o600)
            # Recheck only owned files/identities, never traverse private runtime trees.
            if read_regular(root / _METADATA) != metadata_bytes or _package_roots(root, records) != set(records):
                _refuse("source installation metadata changed during snapshot")
            for name, record in records.items():
                plugin_inventory.validate_install_inventory(root / name, record["files"])
                if read_regular(root / name / _CATALOG) != sidecars[name]:
                    _refuse("catalog identity changed during snapshot", record)
            for path, version in removal_versions.items():
                current = read_regular(path) if path.exists() or path.is_symlink() else None
                if current != version:
                    _refuse("removed-plugin list changed during snapshot")
            os.rename(staged, destination)
        return report
    except (PluginInventoryError, OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        # Parser and OS exceptions can contain private source lines or paths.
        _refuse(f"unreadable or changed package data ({type(exc).__name__})")

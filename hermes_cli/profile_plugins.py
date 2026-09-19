"""Offline, fail-closed snapshots of installed Git plugin packages.

The caller owns the installation lock and the unpublished profile staging home.
A clean standalone Git checkout or a locally recorded code snapshot supplies the
ownership boundary. Git internals, runtime data and interpreter environments are
never copied. Local integrity records are not authenticated provenance or a code audit.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
from typing import NoReturn
from urllib.parse import urlsplit

import yaml

from hermes_cli import plugin_catalog

_METADATA = ".install-metadata.json"
_CATALOG = ".hermes-catalog.json"
_SNAPSHOT = ".hermes-snapshot.json"
_MARKERS = {"plugin.yaml", "plugin.yml", "plugin.json", "__init__.py", ".git"}
_SHA = re.compile(r"[0-9a-fA-F]{40}")
_NAME = re.compile(r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)?")
_RUNTIME_NAMES = {
    ".env", ".venv", "venv", "env", "node_modules", "__pycache__", ".cache", "cache",
    "logs", "log", "data", "state", "sessions", "memories", "memory", "storage", "runtime",
    "credentials", "secrets", "auth.json", "config.yaml", "config.yml", "config.json",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ds_store", ".ssh", ".aws",
}
_RUNTIME_SUFFIXES = {".env", ".db", ".sqlite", ".sqlite3", ".log", ".pyc", ".pyo", ".pem", ".key", ".p12"}


class PluginSnapshotError(ValueError):
    """A package cannot safely be included in a profile clone."""


def _refuse(reason: str) -> NoReturn:
    raise PluginSnapshotError(
        f"Plugin snapshot refused: {reason}. Repair the source installation first: "
        + _repair_guidance()
    )


def _repair_guidance() -> str:
    return (
        "back up local changes/state outside plugins. For a nested category/package, move "
        "the entire nested package outside plugins, remove only its matching key from "
        "plugins/.install-metadata.json, and remove the category directory only if empty. "
        "The installer creates a flat plugins/package, not the old nested path. Then, in "
        "the profile being repaired, reinstall the complete repository with "
        "`hermes -p <profile> plugins install <source> --ref <40-character commit SHA> --force`; "
        "use the saved source/revision and update any configuration referring to the old path. "
        "For a subdirectory-only install, first move its package outside plugins and remove "
        "only its matching metadata key; omit the #subdir selector from the saved source. "
        "The repository root must itself be a supported plugin; otherwise obtain a standalone "
        "plugin repository from its author rather than reinstalling the same subdirectory. "
        "Manual, subdirectory-only and snapshots without an integrity manifest need a fresh "
        "standalone Git installation. Do not delete private data to bypass this check."
    )


def _stat(path: Path):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(path, "is_junction", lambda: False)():
        _refuse("symlink or junction in plugin paths")
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        _refuse("special file in plugin paths")
    if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
        _refuse("hard-linked file in plugin paths")
    return info


def _fingerprint(path: Path) -> tuple:
    info = _stat(path)
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read(path: Path) -> bytes:
    before = _fingerprint(path)
    if not stat.S_ISREG(before[2]):
        _refuse("expected a regular package file")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != before[:2]:
            _refuse("source changed while opening a package file")
        result = stream.read()
    if _fingerprint(path) != before:
        _refuse("source changed while reading a package file")
    return result


def _json(path: Path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _refuse("duplicate metadata key")
            result[key] = value
        return result
    return json.loads(_read(path), object_pairs_hook=unique)


def _runtime(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return any(p.lower() in _RUNTIME_NAMES
               or (p.lower().startswith(".env.") and p not in {".env.example", ".env.template"})
               or p.lower().startswith(("secrets.", "credentials."))
               or Path(p).suffix.lower() in _RUNTIME_SUFFIXES for p in parts)


def _tree(root: Path, *, inventory: bool = False) -> dict[str, tuple]:
    """Fingerprint payload and excluded roots, never descend into runtime trees.

    Inventory category names (notably ``memory``) are not runtime exclusions.
    Git internals still receive full link/mutation checks.
    """
    result = {".": _fingerprint(root)}
    def visit(directory, package):
        if inventory and not package and directory != root:
            package = bool({p.name for p in directory.iterdir()} & _MARKERS)
        for child in sorted(directory.iterdir()):
            info = _fingerprint(child)
            relative = child.relative_to(root)
            result[relative.as_posix()] = info
            excluded = package and ".git" not in relative.parts and _runtime(child.name)
            if stat.S_ISDIR(info[2]) and not excluded:
                visit(child, package)
    visit(root, not inventory)
    return result


def _source(source) -> str:
    if not isinstance(source, str) or not source or any(c.isspace() for c in source):
        _refuse("missing or malformed source metadata")
    if source.startswith("git@") and re.fullmatch(r"git@[A-Za-z0-9.-]+:[A-Za-z0-9_./-]+", source):
        return source
    parsed = urlsplit(source)
    if (parsed.scheme not in {"https", "http", "ssh", "file"} or parsed.query or parsed.fragment
            or parsed.password or (parsed.username and not (parsed.scheme == "ssh" and parsed.username == "git"))
            or not parsed.path or (parsed.scheme != "file" and not parsed.hostname)):
        _refuse("source is credential-bearing, malformed, or subdirectory-only")
    return source


def _offline_removals(home: Path):
    paths = [plugin_catalog.get_catalog_dir() / "removed.yaml", home / "cache" / "plugin-catalog.json"]
    entries, versions = [], {}
    for index, path in enumerate(paths):
        if not path.exists() and not path.is_symlink():
            if index == 0:
                _refuse("shipped removed-plugin list is missing")
            versions[path] = None
            continue
        _stat(path.parent)
        raw = _read(path)
        versions[path] = (raw, _fingerprint(path))
        data = yaml.safe_load(raw) if index == 0 else _json(path)
        if not isinstance(data, dict) or not isinstance(data.get("removed"), list):
            _refuse("malformed removed-plugin list")
        for row in data["removed"]:
            if (not isinstance(row, dict) or not isinstance(row.get("name"), str) or not row["name"]
                    or any(not isinstance(row.get(k, ""), str) for k in ("repo", "reason", "date"))):
                _refuse("malformed removed-plugin entry")
            entries.append(plugin_catalog.RemovedEntry(**{k: row.get(k, "") for k in ("name", "repo", "reason", "date")}))
    return entries, versions


def _git(root: Path, *args: str) -> bytes:
    # Read-only plumbing only: no checkout, filters, fsmonitor, hooks, fetch or lazy fetching.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_TERMINAL_PROMPT="0", GIT_NO_LAZY_FETCH="1", GIT_OPTIONAL_LOCKS="0")
    result = subprocess.run(
        ["git", "--no-replace-objects", "-c", "core.fsmonitor=false", "-c", f"core.hooksPath={os.devnull}",
         "--git-dir=" + str(root / ".git"), "--work-tree=" + str(root), *args],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False,
    )
    if result.returncode:
        _refuse("Git ownership/revision verification failed (Git diagnostics suppressed to protect credentials)")
    return result.stdout


def _git_payload(root: Path, record: dict, tree: dict) -> dict[str, tuple[bytes, int]]:
    git_dir = root / ".git"
    if not git_dir.is_dir() or git_dir.is_symlink():
        _refuse("package has no standalone Git ownership index")
    for forbidden in ("commondir", "gitdir", "worktrees", "modules", "objects/info/alternates", "objects/info/http-alternates"):
        if (git_dir / forbidden).exists():
            _refuse("worktrees, submodules or external Git object stores are unsupported")
    if _git(root, "rev-parse", "HEAD").decode().strip().lower() != record["revision"].lower():
        _refuse("installed revision disagrees with metadata")
    origin = _git(root, "config", "--local", "--get", "remote.origin.url").decode().strip()
    if _source(origin) != record["source"]:
        _refuse("Git origin disagrees with source metadata")
    committed = {}
    for entry in _git(root, "ls-tree", "-rz", "--full-tree", "HEAD").split(b"\0"):
        if entry:
            header, name = entry.split(b"\t", 1)
            mode, kind, oid = header.decode().split()
            if kind != "blob" or mode not in {"100644", "100755"}:
                _refuse("tracked symlink, submodule or special file")
            committed[name.decode("utf-8")] = (mode, oid)
    indexed = {}
    for entry in _git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if entry:
            header, name = entry.split(b"\t", 1)
            mode, oid, stage = header.decode().split()
            if stage != "0":
                _refuse("unmerged Git index")
            indexed[name.decode("utf-8")] = (mode, oid)
    if not committed or indexed != committed:
        _refuse("Git index is dirty or empty")
    payload = {}
    for name, (mode, oid) in committed.items():
        parts = PurePosixPath(name).parts
        if (not parts or name.startswith("/") or any(p in {".", "..", ".git"} for p in parts)
                or "\\" in name or _runtime(name) or name in {_CATALOG, _METADATA, _SNAPSHOT}):
            _refuse("tracked runtime/secret-like payload or unsafe package path")
        data = _read(root / name)
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if actual != oid:
            _refuse("working package differs from its tracked revision")
        if bool(_stat(root / name).st_mode & 0o111) != (mode == "100755"):
            _refuse("working package executable mode differs from its tracked revision")
        payload[name] = (data, int(mode, 8) & 0o777)
    for name, info in tree.items():
        if name == "." or name == ".git" or name.startswith(".git/"):
            continue
        if stat.S_ISDIR(info[2]) or name in committed or name == _CATALOG or _runtime(name):
            continue
        _refuse("unowned file in package (including ignored files)")
    return payload


def _snapshot_manifest(record: dict, payload: dict) -> dict:
    return {
        "version": 1,
        "installation": {k: v for k, v in record.items() if k != "snapshot_sha256"},
        "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "mode": mode}
                  for name, (data, mode) in sorted(payload.items())},
    }


def _snapshot_payload(root: Path, record: dict, tree: dict) -> dict:
    if stat.S_IMODE(_stat(root / _SNAPSHOT).st_mode) != 0o600:
        _refuse("snapshot manifest mode changed")
    raw = _read(root / _SNAPSHOT)
    if hashlib.sha256(raw).hexdigest() != record["snapshot_sha256"]:
        _refuse("snapshot manifest differs from installation record")
    payload = {}
    for name, info in tree.items():
        if name == ".git" or name.startswith(".git/"):
            _refuse("snapshot unexpectedly contains Git internals")
        if name == "." or stat.S_ISDIR(info[2]) or _runtime(name) or name == _SNAPSHOT:
            continue
        if "\\" in name or name == _METADATA:
            _refuse("unsafe snapshot payload path")
        payload[name] = (_read(root / name), stat.S_IMODE(info[2]))
    if _json(root / _SNAPSHOT) != _snapshot_manifest(record, payload):
        _refuse("snapshot payload or installation metadata changed")
    return payload


def _inventory(root: Path) -> list[str]:
    packages = []
    def visit(directory: Path, depth: int):
        names = {p.name for p in directory.iterdir()}
        if directory != root and names & _MARKERS:
            packages.append(directory.relative_to(root).as_posix())
            return
        for child in sorted(directory.iterdir()):
            if directory == root and child.name == _METADATA:
                continue
            if _runtime(child.name) and not child.is_dir():
                continue
            if not child.is_dir() or depth >= 2 or child.name.startswith("."):
                _refuse("unowned plugin inventory entry")
            before = len(packages)
            visit(child, depth + 1)
            if before == len(packages):
                _refuse("unowned or empty plugin directory")
    visit(root, 0)
    return packages


def _write_payload(destination: Path, payload: dict[str, tuple[bytes, int]]) -> None:
    for name, (data, mode) in payload.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
        path.chmod(mode)


def _manifest_names(package: Path) -> list[str]:
    names = []
    for filename in ("plugin.yaml", "plugin.yml", "plugin.json", ".claude-plugin/plugin.json"):
        path = package / filename
        if not path.exists():
            continue
        data = _json(path) if filename.endswith(".json") else yaml.safe_load(_read(path))
        if not isinstance(data, dict):
            _refuse("malformed plugin manifest")
        name = data.get("name")
        if name is not None:
            if not isinstance(name, str) or not name:
                _refuse("malformed plugin manifest name")
            names.append(name)
    return names


def copy_profile_plugins(source_home: Path, staging_home: Path) -> dict[str, list[str]]:
    """Snapshot verified packages; return ``{copied: [keys], warnings: [text]}``.

    Caller MUST hold ``plugin_installation_lock(source_home)`` from
    ``hermes_cli.plugin_installation`` until the whole profile is published.
    Requires a private staging home without ``plugins``. On failure no plugin tree is
    published. Does not import plugins, install dependencies, grant consent or enable
    anything. Clones have no Git repository: source/revision/pin and local integrity
    metadata support repeat cloning, not authenticated provenance or updates.
    """
    source_home, staging_home = Path(source_home), Path(staging_home)
    root, destination = source_home / "plugins", staging_home / "plugins"
    report: dict[str, list[str]] = {"copied": [], "warnings": []}
    try:
        _stat(source_home)
        _stat(staging_home)
        if destination.exists() or destination.is_symlink():
            _refuse("destination plugins already exists")
        if not root.exists() and not root.is_symlink():
            return report
        before = _tree(root, inventory=True)
        packages = _inventory(root)
        if not packages and _METADATA not in before:
            return report
        records = _json(root / _METADATA)
        if not isinstance(records, dict) or set(records) != set(packages):
            _refuse("missing, stale or ambiguous install metadata")
        removals, removal_versions = _offline_removals(source_home)
        output = {}
        with tempfile.TemporaryDirectory(prefix=".plugin-snapshot-", dir=staging_home) as temporary:
            staged = Path(temporary) / "plugins"
            staged.mkdir()
            for name in packages:
                record = records[name]
                if (not _NAME.fullmatch(name) or not isinstance(record, dict)
                        or not {"source", "revision", "pinned"} <= set(record)
                        or set(record) - {"source", "revision", "pinned", "snapshot_sha256", "catalog_provenance"}
                        or not isinstance(record.get("revision"), str) or not _SHA.fullmatch(record["revision"])
                        or type(record.get("pinned")) is not bool):
                    _refuse("malformed or unsupported install metadata")
                snapshot = record.get("snapshot_sha256")
                if ("snapshot_sha256" in record and (not isinstance(snapshot, str) or not re.fullmatch(r"[0-9a-f]{64}", snapshot))
                        or ("catalog_provenance" in record and snapshot is None)):
                    _refuse("malformed snapshot installation metadata")
                _source(record.get("source"))
                package = root / name
                package_tree = _tree(package)
                sidecar = _json(package / _CATALOG) if (package / _CATALOG).exists() else None
                history = record.get("catalog_provenance")
                for provenance in (sidecar, history):
                    if provenance is None:
                        continue
                    sidecar_to_check = provenance
                    if (not isinstance(sidecar_to_check, dict)
                            or set(sidecar_to_check) - {"catalog_name", "repo", "sha", "tier", "installed_at"}
                            or not all(isinstance(sidecar_to_check.get(k), str) and sidecar_to_check[k] for k in ("catalog_name", "repo", "sha"))
                            or not _SHA.fullmatch(sidecar_to_check["sha"])
                            or any(not isinstance(v, str) for v in sidecar_to_check.values())):
                        _refuse("malformed catalog provenance")
                    _source(sidecar_to_check["repo"])
                candidates = [name, package.name, record["source"], *_manifest_names(package)]
                if sidecar:
                    candidates.extend((sidecar["catalog_name"], sidecar["repo"]))
                if history:
                    candidates.extend((history["catalog_name"], history["repo"]))
                for candidate in candidates:
                    if plugin_catalog.match_removed(candidate, removals):
                        _refuse("package matches the offline removed-plugin list; remove it from the clone source")
                payload = (_snapshot_payload(package, record, package_tree) if snapshot is not None
                           else _git_payload(package, record, package_tree))
                new_record = dict(record)
                if sidecar:
                    if sidecar["sha"].lower() == record["revision"].lower() and sidecar["repo"] == record["source"]:
                        payload[_CATALOG] = (json.dumps(sidecar, indent=2).encode() + b"\n", 0o600)
                    else:
                        # The installer records installed SHA, not a reviewed/approved SHA.
                        # Preserve stale history separately from current-install metadata.
                        new_record["catalog_provenance"] = sidecar
                if "catalog_provenance" in new_record:
                    report["warnings"].append(f"{name}: catalog provenance differs from installed revision; retained as historical installation metadata.")
                manifest = json.dumps(_snapshot_manifest(new_record, payload), sort_keys=True, indent=2).encode() + b"\n"
                new_record["snapshot_sha256"] = hashlib.sha256(manifest).hexdigest()
                payload[_SNAPSHOT] = (manifest, 0o600)
                _write_payload(staged / name, payload)
                output[name] = new_record
                report["copied"].append(name)
                report["warnings"].append(
                    f"{name}: code-only snapshot, no Git history or dependencies copied. "
                    "Local integrity metadata is not a security review. To restore updates in the destination, "
                    + _repair_guidance() + " Review dependencies separately.")
            (staged / _METADATA).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            (staged / _METADATA).chmod(0o600)
            if _tree(root, inventory=True) != before:
                _refuse("source files or plugin inventory changed during snapshot")
            for path, version in removal_versions.items():
                current = (_read(path), _fingerprint(path)) if path.exists() or path.is_symlink() else None
                if current != version:
                    _refuse("removed-plugin list changed during snapshot")
            os.rename(staged, destination)
        return report
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError, subprocess.SubprocessError) as exc:
        _refuse(f"unreadable or malformed package data ({type(exc).__name__})")

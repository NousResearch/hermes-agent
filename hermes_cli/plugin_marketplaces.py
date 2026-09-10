"""Profile-scoped Git plugin marketplaces.

Custom marketplaces use Claude's ``.claude-plugin/marketplace.json`` as the
catalogue format. Only repository-local ``./path`` sources are supported.
Repositories are fetched through the selected profile host's existing Git
credentials and reduced to a bounded, non-secret JSON cache for Desktop.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re

import subprocess
import tempfile
import threading
import time
import urllib.parse
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from hermes_cli._subprocess_compat import noninteractive_git_env
from utils import (
    secure_atomic_write_text,
    secure_open_file,
    secure_parent_directory,
    secure_unlink,
)

logger = logging.getLogger(__name__)

_REGISTRY_VERSION = 1
_CACHE_TTL = 6 * 60 * 60
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_ENTRIES = 500
_MAX_NAME_LENGTH = 128
_MAX_TEXT_LENGTH = 4096
_SOURCE_ID_RE = re.compile(r"^[0-9a-f]{16}$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_PROCESS_REGISTRY_LOCK = threading.RLock()
_PROCESS_REFRESH_LOCK = threading.RLock()


class MarketplaceError(ValueError):
    """A marketplace URL, checkout, or manifest is invalid."""


@dataclass(frozen=True)
class MarketplaceCatalogEntry:
    name: str
    repo: str
    sha: str
    description: str
    maintainer: str
    subdir: str
    source_id: str
    source_name: str
    display_name: str
    version: str
    tree_sha: str
    compatible: bool
    tier: str = "community"


def _registry_path() -> Path:
    return get_hermes_home() / "plugin-marketplaces.json"


def _cache_dir() -> Path:
    return get_hermes_home() / "cache" / "plugin-marketplaces"


def _ensure_cache_dir() -> Path:
    path = _cache_dir()
    try:
        with secure_parent_directory(
            path / ".sentinel", get_hermes_home(), create=True
        ):
            pass
    except OSError as exc:
        raise MarketplaceError(
            f"Marketplace cache path contains a symlink or is unsafe: {exc}"
        ) from exc
    return path


def _open_marketplace_lock(path: Path):
    from hermes_cli.plugin_install_state import PluginOperationError, _open_lock_path

    try:
        return _open_lock_path(path)
    except PluginOperationError as exc:
        raise MarketplaceError(str(exc)) from exc


@contextmanager
def _registry_lock():
    """Serialize profile-local registry read-modify-write transactions."""
    path = _registry_path().with_suffix(".lock")
    with _PROCESS_REGISTRY_LOCK, _open_marketplace_lock(path) as handle:
        from hermes_cli.plugin_install_state import _lock_file, _unlock_file

        _lock_file(handle)
        try:
            yield
        finally:
            _unlock_file(handle)


@contextmanager
def _refresh_lock(source_id: str):
    """Serialize one source's clone/parse/cache transaction across processes."""
    source_id = _validated_source_id(source_id)
    cache = _ensure_cache_dir()
    path = cache / f".{source_id}.refresh.lock"
    # ponytail: process-global lock; split per source if refresh throughput matters.
    with _PROCESS_REFRESH_LOCK, _open_marketplace_lock(path) as handle:
        from hermes_cli.plugin_install_state import _lock_file, _unlock_file

        _lock_file(handle)
        try:
            yield
        finally:
            _unlock_file(handle)


def _source_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _validated_source_id(value: Any) -> str:
    if not isinstance(value, str) or not _SOURCE_ID_RE.fullmatch(value):
        raise MarketplaceError(
            "Plugin marketplace registry contains an invalid source ID."
        )
    return value


def _text(value: Any, *, label: str, maximum: int = _MAX_TEXT_LENGTH) -> str:
    if not isinstance(value, str):
        raise MarketplaceError(f"{label} must be a string.")
    value = value.strip()
    if not value:
        raise MarketplaceError(f"{label} must not be empty.")
    if len(value) > maximum:
        raise MarketplaceError(f"{label} exceeds the {maximum}-character limit.")
    if any(ord(char) < 32 for char in value):
        raise MarketplaceError(f"{label} contains control characters.")
    return value


def _optional_text(value: Any, *, label: str, maximum: int = _MAX_TEXT_LENGTH) -> str:
    if value in (None, ""):
        return ""
    return _text(value, label=label, maximum=maximum)


def _read_json(path: Path, *, label: str) -> Any:
    try:
        if path.stat().st_size > _MAX_MANIFEST_BYTES:
            raise MarketplaceError(f"{label} exceeds the 1 MB limit.")
        return json.loads(path.read_text(encoding="utf-8"))
    except MarketplaceError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise MarketplaceError(f"Could not read {label}: {exc}") from exc


def _read_control_json(path: Path, *, label: str) -> Any:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = secure_open_file(path, get_hermes_home(), flags)
    except OSError as exc:
        raise MarketplaceError(f"Could not read {label}: {exc}") from exc
    try:
        info = os.fstat(fd)
        if info.st_size > _MAX_MANIFEST_BYTES:
            raise MarketplaceError(f"{label} exceeds the 1 MB limit.")
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            fd = -1
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise MarketplaceError(f"Could not read {label}: {exc}") from exc
    finally:
        if fd >= 0:
            os.close(fd)


def _read_registry() -> list[dict[str, str]]:
    path = _registry_path()
    if path.is_symlink():
        raise MarketplaceError("Plugin marketplace registry must not be a symlink.")
    if not path.exists():
        return []
    value = _read_control_json(path, label="plugin marketplace registry")
    if not isinstance(value, dict) or value.get("version") != _REGISTRY_VERSION:
        raise MarketplaceError("Plugin marketplace registry has an unsupported format.")
    sources = value.get("marketplaces")
    if not isinstance(sources, list):
        raise MarketplaceError(
            "Plugin marketplace registry must contain a marketplaces list."
        )
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise MarketplaceError(
                "Plugin marketplace registry contains an invalid source."
            )
        sid = _validated_source_id(source.get("id"))
        name = source.get("name")
        url = source.get("url")
        if not isinstance(name, str) or not name or not isinstance(url, str) or not url:
            raise MarketplaceError(
                "Plugin marketplace registry contains an invalid source."
            )
        normalized = _normalize_url(url, allow_file=url.startswith("file://"))
        if sid != _source_id(normalized) or sid in seen:
            raise MarketplaceError(
                "Plugin marketplace registry contains an invalid source."
            )
        seen.add(sid)
        out.append({
            "id": sid,
            "name": _text(name, label="marketplace name", maximum=_MAX_NAME_LENGTH),
            "url": normalized,
        })
    return out


def _write_registry(sources: list[dict[str, str]]) -> None:
    path = _registry_path()
    if path.is_symlink():
        raise MarketplaceError("Plugin marketplace registry must not be a symlink.")
    secure_atomic_write_text(
        path,
        json.dumps(
            {"marketplaces": sources, "version": _REGISTRY_VERSION},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        get_hermes_home(),
    )


def _normalize_url(url: str, *, allow_file: bool = False) -> str:
    from hermes_cli.plugins_cmd import _resolve_git_url, _scrub_git_url

    candidate = (url or "").strip()
    if not candidate:
        raise MarketplaceError("Marketplace URL is required.")
    if "\\" in candidate or any(ord(char) < 32 for char in candidate):
        raise MarketplaceError("Marketplace URL contains unsupported characters.")
    raw = urllib.parse.urlsplit(candidate)
    if raw.query or raw.fragment:
        raise MarketplaceError("Marketplace URL must not contain a query or fragment.")
    try:
        git_url, subdir = _resolve_git_url(candidate)
    except ValueError as exc:
        raise MarketplaceError(str(exc)) from exc
    if subdir:
        raise MarketplaceError("Marketplace URL must point to a repository root.")
    parsed = urllib.parse.urlsplit(git_url)
    if parsed.username or parsed.password:
        raise MarketplaceError("Marketplace URLs must not contain credentials.")
    if parsed.query or parsed.fragment:
        raise MarketplaceError("Marketplace URL must not contain a query or fragment.")
    if parsed.scheme == "file" and allow_file:
        return git_url
    if parsed.scheme != "https":
        raise MarketplaceError("Marketplace URL must use https://.")
    return _scrub_git_url(git_url).rstrip("/")


def _git(repo: Path, *args: str) -> str:
    from hermes_cli.plugins_cmd import _resolve_git_executable, _safe_git_error

    git = _resolve_git_executable()
    if not git:
        raise MarketplaceError("git is not installed or not in PATH.")
    try:
        result = subprocess.run(
            [git, *args],
            cwd=repo,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            stdin=subprocess.DEVNULL,
            env=noninteractive_git_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise MarketplaceError("Git command timed out after 20 seconds.") from exc
    if result.returncode != 0:
        raise MarketplaceError(_safe_git_error(result))
    return result.stdout.strip().lower()


def _clone(url: str, target: Path) -> None:
    from hermes_cli.plugins_cmd import _resolve_git_executable, _safe_git_error

    git = _resolve_git_executable()
    if not git:
        raise MarketplaceError("git is not installed or not in PATH.")
    try:
        result = subprocess.run(
            [git, "clone", "--depth", "1", url, str(target)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            stdin=subprocess.DEVNULL,
            env=noninteractive_git_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise MarketplaceError("Marketplace clone timed out after 60 seconds.") from exc
    if result.returncode != 0:
        raise MarketplaceError(f"Git clone failed:\n{_safe_git_error(result, url)}")


def _author_name(value: Any, fallback: str) -> str:
    if isinstance(value, dict):
        value = value.get("name")
    if isinstance(value, str) and value.strip():
        return _text(value, label="plugin author")
    return fallback


def _manifest_file(root: Path, path: Path, label: str) -> Path:
    if path.is_symlink():
        raise MarketplaceError(f"{label} must not be a symlink.")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise MarketplaceError(
            f"{label} must stay inside the marketplace repository."
        ) from exc
    if not path.is_file():
        raise MarketplaceError(f"Repository has no {label}.")
    return path


def _compatibility(plugin_root: Path, plugin_name: str) -> tuple[bool, str]:
    for filename in ("plugin.yaml", "plugin.yml", "plugin.json"):
        manifest = plugin_root / filename
        if manifest.is_symlink():
            return False, f"{filename} must not be a symlink"
        if manifest.is_file():
            try:
                from hermes_cli.plugins_cmd import _read_manifest

                parsed = _read_manifest(plugin_root)
            except Exception as exc:
                return False, str(exc)
            if not parsed:
                return False, f"Could not validate {filename}"
            if parsed.get("name") != plugin_name:
                return False, "Hermes plugin name must match the marketplace entry"
            return True, ""
    return False, "Requires a root Hermes plugin.json or plugin.yaml manifest"


def _parse_checkout(root: Path, source: dict[str, str]) -> list[dict[str, Any]]:
    from hermes_cli.plugins_cmd import PluginOperationError, _resolve_subdir_within

    marketplace_path = _manifest_file(
        root,
        root / ".claude-plugin" / "marketplace.json",
        ".claude-plugin/marketplace.json",
    )
    manifest = _read_json(marketplace_path, label="marketplace manifest")
    if not isinstance(manifest, dict):
        raise MarketplaceError("Marketplace manifest must be a JSON object.")
    name = (
        _text(
            manifest.get("name"),
            label="marketplace name",
            maximum=_MAX_NAME_LENGTH,
        )
        .replace("-", " ")
        .replace("_", " ")
        .title()
    )
    plugins = manifest.get("plugins")
    if not isinstance(plugins, list) or not plugins:
        raise MarketplaceError("Marketplace manifest must contain at least one plugin.")
    if len(plugins) > _MAX_ENTRIES:
        raise MarketplaceError(f"Marketplace exceeds the {_MAX_ENTRIES}-plugin limit.")
    owner = _author_name(manifest.get("owner"), name)
    head = _git(root, "rev-parse", "HEAD")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in plugins:
        if not isinstance(raw, dict):
            raise MarketplaceError("Marketplace plugins must be JSON objects.")
        plugin_name = _text(
            raw.get("name"),
            label="plugin name",
            maximum=_MAX_NAME_LENGTH,
        )
        if plugin_name in seen:
            raise MarketplaceError(
                f"Marketplace contains duplicate plugin '{plugin_name}'."
            )
        seen.add(plugin_name)
        source_path = raw.get("source")
        if not isinstance(source_path, str):
            raise MarketplaceError(
                f"Marketplace plugin '{plugin_name}' uses an unsupported object source."
            )
        if "\\" in source_path or not source_path.startswith("./"):
            raise MarketplaceError(
                f"Marketplace plugin '{plugin_name}' source must be a relative './...' path."
            )
        subdir = source_path[2:].strip("/")
        if not subdir:
            raise MarketplaceError(
                f"Marketplace plugin '{plugin_name}' source is empty."
            )
        relative = Path(subdir)
        if any(
            (root.joinpath(*relative.parts[:index])).is_symlink()
            for index in range(1, len(relative.parts) + 1)
        ):
            raise MarketplaceError(
                f"Marketplace plugin '{plugin_name}' root must not contain symlinks."
            )
        try:
            plugin_root = _resolve_subdir_within(root, subdir)
        except PluginOperationError as exc:
            raise MarketplaceError(
                f"Marketplace plugin '{plugin_name}' source must stay inside the marketplace repository."
            ) from exc
        if _git(root, "cat-file", "-t", f"HEAD:{subdir}") != "tree":
            raise MarketplaceError(
                f"Marketplace plugin '{plugin_name}' source must be a directory tree."
            )
        claude_manifest = plugin_root / ".claude-plugin" / "plugin.json"
        metadata: dict[str, Any] = {}
        if claude_manifest.exists() or claude_manifest.is_symlink():
            value = _read_json(
                _manifest_file(
                    root, claude_manifest, f"plugin manifest for {plugin_name}"
                ),
                label=f"plugin manifest for {plugin_name}",
            )
            if not isinstance(value, dict):
                raise MarketplaceError(
                    f"Plugin manifest for '{plugin_name}' must be a JSON object."
                )
            metadata = value
        compatible, reason = _compatibility(plugin_root, plugin_name)
        entries.append({
            "compatible": compatible,
            "description": _optional_text(
                raw.get("description") or metadata.get("description"),
                label=f"description for {plugin_name}",
            ),
            "display_name": _optional_text(
                raw.get("displayName")
                or raw.get("display_name")
                or metadata.get("displayName")
                or plugin_name,
                label=f"display name for {plugin_name}",
                maximum=_MAX_NAME_LENGTH,
            ),
            "incompatibility_reason": reason,
            "maintainer": _author_name(
                raw.get("author") or metadata.get("author"), owner
            ),
            "name": plugin_name,
            "repo": source["url"],
            "sha": head,
            "source_id": source["id"],
            "source_name": name,
            "subdir": subdir,
            "tree_sha": _git(root, "rev-parse", f"HEAD:{subdir}"),
            "version": _optional_text(
                raw.get("version") or metadata.get("version"),
                label=f"version for {plugin_name}",
                maximum=_MAX_NAME_LENGTH,
            ),
        })
    source["name"] = name
    return entries


def _cache_path(source_id: str) -> Path:
    return _cache_dir() / f"{_validated_source_id(source_id)}.json"


def _remove_cache(source_id: str) -> None:
    try:
        secure_unlink(_cache_path(source_id), get_hermes_home(), missing_ok=True)
    except OSError as exc:
        logger.warning("Marketplace removed; cache cleanup deferred: %s", exc)


def _write_cache(source: dict[str, str], entries: list[dict[str, Any]]) -> None:
    path = _cache_path(source["id"])
    if path.is_symlink():
        raise MarketplaceError("Plugin marketplace cache must not be a symlink.")
    secure_atomic_write_text(
        path,
        json.dumps({"entries": entries, "source": source}, indent=2, sort_keys=True)
        + "\n",
        get_hermes_home(),
    )


def _read_cache(source: dict[str, str]) -> dict[str, Any] | None:
    path = _cache_path(source["id"])
    _ensure_cache_dir()
    if path.is_symlink():
        return None
    if not path.is_file():
        return None
    try:
        value = _read_control_json(path, label=f"cache for {source['name']}")
    except MarketplaceError:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("entries"), list):
        return None
    cached_source = value.get("source")
    if not isinstance(cached_source, dict) or cached_source.get("url") != source["url"]:
        return None
    for entry in value["entries"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("compatible"), bool):
            return None
        required = (
            "name",
            "repo",
            "sha",
            "source_id",
            "source_name",
            "subdir",
            "tree_sha",
        )
        if not all(isinstance(entry.get(key), str) for key in required):
            return None
        if entry["repo"] != source["url"] or entry["source_id"] != source["id"]:
            return None
        if not _SHA_RE.fullmatch(entry["sha"]) or not _SHA_RE.fullmatch(
            entry["tree_sha"]
        ):
            return None
        subdir = entry["subdir"]
        if (
            not subdir
            or "\\" in subdir
            or subdir.startswith("/")
            or ".." in Path(subdir).parts
        ):
            return None
        for key in (
            "description",
            "display_name",
            "incompatibility_reason",
            "maintainer",
            "version",
        ):
            if not isinstance(entry.get(key, ""), str):
                return None
    return {**source, "available": True, "entries": value["entries"], "stale": False}


def _refresh(source: dict[str, str]) -> dict[str, Any]:
    with _refresh_lock(source["id"]):
        resolved = dict(source)
        _ensure_cache_dir()
        with tempfile.TemporaryDirectory(prefix="marketplace-") as tmp:
            clone = Path(tmp) / "repo"
            _clone(resolved["url"], clone)
            entries = _parse_checkout(clone, resolved)
        _write_cache(resolved, entries)
        return {**resolved, "available": True, "entries": entries, "stale": False}


def add_marketplace(url: str, *, allow_file: bool = False) -> dict[str, Any]:
    """Validate, persist, and return one Git marketplace."""
    normalized = _normalize_url(url, allow_file=allow_file)
    sid = _source_id(normalized)
    source = {
        "id": sid,
        "name": normalized.rsplit("/", 1)[-1],
        "url": normalized,
    }
    refreshed = _refresh(source)
    source["name"] = str(refreshed["name"])
    with _registry_lock():
        sources = _read_registry()
        existing = next((saved for saved in sources if saved["id"] == sid), None)
        if existing is None:
            sources.append({key: source[key] for key in ("id", "name", "url")})
        else:
            existing["name"] = source["name"]
        _write_registry(sources)
    return refreshed


def list_marketplaces(*, force: bool = False) -> list[dict[str, Any]]:
    """Return saved marketplaces, retaining validated stale snapshots on errors."""
    out: list[dict[str, Any]] = []
    for source in _read_registry():
        cached = _read_cache(source)
        cache_path = _cache_path(source["id"])
        fresh = False
        if cached is not None and not force:
            try:
                fresh = time.time() - cache_path.stat().st_mtime < _CACHE_TTL
            except OSError:
                pass
        if fresh and cached is not None:
            out.append(cached)
            continue
        try:
            refreshed = _refresh(source)
            if refreshed["name"] != source["name"]:
                with _registry_lock():
                    sources = _read_registry()
                    for saved in sources:
                        if saved["id"] == source["id"]:
                            saved["name"] = str(refreshed["name"])
                    _write_registry(sources)
            out.append(refreshed)
        except MarketplaceError as exc:
            if cached is not None:
                out.append({**cached, "error": str(exc), "stale": True})
            else:
                out.append({
                    **source,
                    "available": False,
                    "entries": [],
                    "error": str(exc),
                    "stale": False,
                })
    return out


def remove_marketplace(source_id: str) -> bool:
    source_id = _validated_source_id(source_id)
    with _registry_lock():
        sources = _read_registry()
        kept = [source for source in sources if source["id"] != source_id]
        if len(kept) == len(sources):
            return False
        _write_registry(kept)
    _remove_cache(source_id)
    return True


@contextmanager
def marketplace_authority(source_id: str, url: str):
    """Hold source registration stable across an install commit."""
    with _registry_lock():
        yield any(
            source["id"] == source_id and source["url"] == url
            for source in _read_registry()
        )


def get_marketplace_entry(
    source_id: str, name: str, *, force: bool = False
) -> dict[str, Any] | None:
    for source in list_marketplaces(force=force):
        if (
            source["id"] != source_id
            or not source.get("available")
            or (force and source.get("stale"))
        ):
            continue
        for entry in source.get("entries", []):
            if isinstance(entry, dict) and entry.get("name") == name:
                # Refresh runs without the registry lock. Re-check authority
                # before returning so a concurrently removed source cannot
                # authorize an install from its detached refresh result.
                with _registry_lock():
                    if not any(
                        saved["id"] == source_id and saved["url"] == source["url"]
                        for saved in _read_registry()
                    ):
                        _remove_cache(source_id)
                        return None
                return entry
    return None


def public_marketplace(value: dict[str, Any]) -> dict[str, Any]:
    """Remove clone/install coordinates before data reaches the renderer."""
    entry_fields = {
        "compatible",
        "description",
        "display_name",
        "incompatibility_reason",
        "maintainer",
        "name",
        "source_id",
        "source_name",
        "version",
    }
    return {
        key: value[key]
        for key in ("available", "entries", "error", "id", "name", "stale")
        if key in value
    } | {
        "entries": [
            {key: entry.get(key) for key in entry_fields}
            for entry in value.get("entries", [])
            if isinstance(entry, dict)
        ]
    }


def as_catalog_entry(value: dict[str, Any]):
    """Adapt trusted marketplace metadata to the existing catalog contract."""
    return MarketplaceCatalogEntry(
        name=str(value["name"]),
        repo=str(value["repo"]),
        sha=str(value["sha"]),
        description=str(value.get("description") or ""),
        maintainer=str(value.get("maintainer") or value.get("source_name") or ""),
        tier="community",
        subdir=str(value.get("subdir") or ""),
        source_id=str(value.get("source_id") or ""),
        source_name=str(value.get("source_name") or ""),
        display_name=str(value.get("display_name") or value.get("name") or ""),
        version=str(value.get("version") or ""),
        tree_sha=str(value.get("tree_sha") or ""),
        compatible=bool(value.get("compatible")),
    )

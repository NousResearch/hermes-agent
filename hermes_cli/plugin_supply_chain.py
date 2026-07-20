"""Pure plugin provenance and declared-capability reporting helpers.

This module inspects manifest data and filesystem markers only.  It never imports
plugin code, reads environment/configuration files, or performs network access.
"""

from __future__ import annotations

import errno
import json
import os
import re
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, cast
from urllib.parse import urlsplit

LOCK_FILENAME = ".hermes-plugin-lock.json"
MAX_LOCK_BYTES = 64 * 1024
_CAPABILITY_WARNING = "CAPABILITY_REPORT_IS_NOT_SECURITY_AUDIT"
_FULL_COMMIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
_PROVENANCE_FIELDS = {
    "source_url",
    "subdir",
    "resolved_commit",
    "requested_ref",
    "inspected_at",
}


@dataclass(frozen=True)
class PluginProvenance:
    source_url: str
    subdir: str | None
    resolved_commit: str
    requested_ref: str | None
    inspected_at: str


@dataclass(frozen=True)
class PluginCapabilityReport:
    hooks: tuple[str, ...]
    tools: tuple[str, ...]
    required_env: tuple[str, ...]
    has_dashboard: bool
    has_after_install: bool
    warnings: tuple[str, ...]


def validate_full_commit_sha(value: object) -> str:
    """Return *value* if it is a canonical full Git commit SHA."""
    if not isinstance(value, str) or _FULL_COMMIT_SHA.fullmatch(value) is None:
        raise ValueError("commit SHA must be exactly 40 lowercase hexadecimal characters")
    return value


def _sorted_strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(sorted({item for item in value if isinstance(item, str) and item}))


def _required_env_names(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    names: set[str] = set()
    for item in value:
        if isinstance(item, str) and item:
            names.add(item)
        elif isinstance(item, Mapping):
            item_mapping = cast(Mapping[object, object], item)
            name = item_mapping.get("name")
            if isinstance(name, str) and name:
                names.add(name)
    return tuple(sorted(names))


def build_capability_report(
    plugin_dir: str | os.PathLike[str], manifest: Mapping[str, Any]
) -> PluginCapabilityReport:
    """Report declared capabilities without importing or executing plugin code."""
    root = Path(plugin_dir)
    return PluginCapabilityReport(
        hooks=_sorted_strings(manifest.get("hooks")),
        tools=_sorted_strings(manifest.get("provides_tools")),
        required_env=_required_env_names(manifest.get("requires_env")),
        has_dashboard=(root / "dashboard" / "manifest.json").is_file(),
        has_after_install=(root / "after-install.md").is_file(),
        warnings=(_CAPABILITY_WARNING,),
    )


def _validate_subdir(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("provenance subdir must be a string or null")
    if value == "":
        return value
    if "\\" in value:
        raise ValueError("provenance subdir must use safe relative path segments")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise ValueError("provenance subdir must use safe relative path segments")
    return value


def validate_source_url(value: object) -> str:
    """Return a clone-safe provenance URL, rejecting secrets and URL metadata."""
    if not isinstance(value, str) or not value:
        raise ValueError("provenance source_url must be a non-empty string")
    if "://" in value:
        parsed = urlsplit(value)
        if parsed.query or parsed.fragment:
            raise ValueError("provenance source_url must not contain a query or fragment")
        if parsed.password is not None or (
            parsed.username is not None
            and not (parsed.scheme.lower() == "ssh" and parsed.username == "git")
        ):
            raise ValueError("provenance source_url must not contain credentials")
    elif "@" in value:
        if (
            re.fullmatch(r"git@[^\s@:/?#]+:[^\s?#]+", value) is None
            or value.count("@") != 1
        ):
            raise ValueError("provenance source_url contains unsafe or malformed userinfo")
    else:
        parsed = urlsplit(value)
        if parsed.query or parsed.fragment:
            raise ValueError("provenance source_url must not contain a query or fragment")
    return value


def _validated_provenance(data: Mapping[str, object]) -> PluginProvenance:
    if set(data) != _PROVENANCE_FIELDS:
        raise ValueError("provenance lock has missing or unexpected fields")
    requested_ref = data["requested_ref"]
    inspected_at = data["inspected_at"]
    if requested_ref is not None and not isinstance(requested_ref, str):
        raise ValueError("provenance requested_ref must be a string or null")
    if not isinstance(inspected_at, str) or not inspected_at:
        raise ValueError("provenance inspected_at must be a non-empty string")
    return PluginProvenance(
        source_url=validate_source_url(data["source_url"]),
        subdir=_validate_subdir(data["subdir"]),
        resolved_commit=validate_full_commit_sha(data["resolved_commit"]),
        requested_ref=requested_ref,
        inspected_at=inspected_at,
    )


def _lock_path(plugin_dir: str | os.PathLike[str]) -> Path:
    root = Path(plugin_dir)
    lock_path = root / LOCK_FILENAME
    if lock_path.parent.resolve() != root.resolve():
        raise ValueError("provenance lock location must remain under plugin_dir")
    return lock_path


def write_provenance_lock(
    plugin_dir: str | os.PathLike[str], provenance: PluginProvenance
) -> Path:
    """Atomically write deterministic, credential-free provenance JSON."""
    validated = _validated_provenance(asdict(provenance))
    lock_path = _lock_path(plugin_dir)
    payload = json.dumps(asdict(validated), sort_keys=True, indent=2) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=lock_path.parent,
            prefix=f"{LOCK_FILENAME}.",
            delete=False,
        ) as temporary:
            os.chmod(temporary.name, 0o600)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, lock_path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return lock_path


def read_provenance_lock(
    plugin_dir: str | os.PathLike[str],
) -> PluginProvenance | None:
    """Read a lock, returning ``None`` only when absent and rejecting bad data."""
    lock_path = _lock_path(plugin_dir)
    flags = os.O_RDONLY
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    flags |= nofollow
    try:
        before = None if nofollow else lock_path.lstat()
        if before is not None and stat.S_ISLNK(before.st_mode):
            raise ValueError("provenance lock must be a regular file")
        descriptor = os.open(lock_path, flags)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            return None
        raise ValueError("provenance lock is malformed or unreadable") from exc

    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("provenance lock must be a regular file")
        if metadata.st_nlink != 1:
            raise ValueError("provenance lock must have a single link")
        if before is not None and (
            stat.S_ISLNK(before.st_mode)
            or (before.st_dev, before.st_ino) != (metadata.st_dev, metadata.st_ino)
        ):
            raise ValueError("provenance lock must be a regular file")
        chunks: list[bytes] = []
        remaining = MAX_LOCK_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > MAX_LOCK_BYTES:
            raise ValueError("provenance lock is malformed or unreadable")
        data = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("provenance lock is malformed or unreadable") from exc
    finally:
        os.close(descriptor)
    if not isinstance(data, dict):
        raise ValueError("provenance lock must contain a JSON object")
    return _validated_provenance(data)

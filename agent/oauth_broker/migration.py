"""Redacted, atomic profile migration for the OAuth broker rollout.

`plan_migration` builds a dry-run snapshot: profile set, group counts, file
hashes, and per-entry metadata (ids, labels, sources, priorities, status
timestamps) — never secret values. `apply_migration` validates every hash,
then rewrites one profile at a time (staged in memory, atomic replace,
journaled); a write failure automatically restores the entire profile batch
from exact original bytes. `rollback_migration` removes the broker
references and restores legacy priority/enabled state by stable entry id.

Legacy entries are archived, never deleted: they keep their secret fields
on disk, gain ``disabled: true``, and drop out of pool rotation until a
rollback re-enables them (docs/design/oauth-broker.md §七).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from agent.oauth_broker.models import ACCOUNT_ALIASES, CLIENT_KEY_KEYCHAIN_SERVICE

GROUP_ORDER = {
    "A": ("A", "B", "C"),
    "B": ("B", "C", "A"),
    "C": ("C", "A", "B"),
}

CLIENT_KEY_SECRET_SOURCE = f"keychain://{CLIENT_KEY_KEYCHAIN_SERVICE}/local"

_PROVIDER = "openai-codex"
SNAPSHOT_SCHEMA_VERSION = 2
JOURNAL_SCHEMA_VERSION = 1
_MAX_JOURNAL_BYTES = 1024 * 1024
_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
# Only keys PRESENT on the entry are captured, so restore can reproduce the
# exact original shape (an entry that never had `priority` or was already
# `disabled` comes back exactly that way).
_LEGACY_SNAPSHOT_FIELDS = (
    "id",
    "label",
    "source",
    "priority",
    "disabled",
    "last_status",
    "last_status_at",
)


_BROKER_RUNTIME_FIELDS = {
    "access_token",
    "refresh_token",
    "last_status",
    "last_status_at",
    "last_error_code",
    "last_error_reason",
    "last_error_message",
    "last_error_reset_at",
    "expires_at",
    "expires_at_ms",
    "last_refresh",
    "inference_base_url",
    "agent_key",
    "agent_key_expires_at",
    "request_count",
    "disabled",
}


class MigrationError(RuntimeError):
    pass


def _validate_port(port) -> int:
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise MigrationError(f"broker port must be an integer in 1..65535, got {port!r}")
    return port


def _validate_profile_name(name) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name in (".", "..")
        or name != Path(name).name
        or _PROFILE_NAME_RE.fullmatch(name) is None
    ):
        raise MigrationError(f"invalid profile name {name!r}")
    return name


def _validate_snapshot(snapshot: dict) -> tuple[int, List[str]]:
    """Validate all non-secret migration control metadata before file access."""
    if not isinstance(snapshot, dict):
        raise MigrationError("migration snapshot is not an object")
    expected_top_level = {
        "snapshot_schema_version",
        "mode",
        "provider",
        "port",
        "groups",
        "group_counts",
        "profiles",
    }
    if set(snapshot) != expected_top_level:
        raise MigrationError("migration snapshot top-level fields are invalid")
    if snapshot.get("snapshot_schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise MigrationError("unsupported migration snapshot schema version")
    if snapshot.get("mode") != "dry-run":
        raise MigrationError("migration snapshot mode must be dry-run")
    if snapshot.get("provider") != _PROVIDER:
        raise MigrationError("migration snapshot provider mismatch")
    profiles = snapshot.get("profiles")
    groups = snapshot.get("groups")
    if not isinstance(profiles, dict) or not profiles or not isinstance(groups, dict):
        raise MigrationError("migration snapshot profiles/groups are invalid")
    if any(not isinstance(name, str) for name in profiles) or any(
        not isinstance(name, str) for name in groups
    ):
        raise MigrationError("migration snapshot profile names are invalid")
    if set(profiles) != set(groups):
        raise MigrationError("migration snapshot profile/group sets differ")
    port = _validate_port(snapshot.get("port"))
    names = sorted(profiles)
    expected_counts = {alias: 0 for alias in ACCOUNT_ALIASES}
    expected_profile_fields = {
        "group",
        "auth_sha256",
        "auth_canonical_sha256",
        "added_entry_ids",
        "legacy",
    }
    allowed_legacy_fields = set(_LEGACY_SNAPSHOT_FIELDS)
    for name in names:
        _validate_profile_name(name)
        profile = profiles[name]
        group = groups[name]
        if (
            not isinstance(profile, dict)
            or set(profile) != expected_profile_fields
            or not isinstance(group, str)
            or group not in GROUP_ORDER
            or profile.get("group") != group
        ):
            raise MigrationError(f"profile {name} has invalid group metadata")
        expected_counts[group] += 1
        for hash_field in ("auth_sha256", "auth_canonical_sha256"):
            auth_hash = profile.get(hash_field)
            if not isinstance(auth_hash, str) or re.fullmatch(
                r"[0-9a-f]{64}", auth_hash
            ) is None:
                raise MigrationError(
                    f"profile {name} has invalid {hash_field} metadata"
                )
        expected_added = [f"broker-{alias}" for alias in GROUP_ORDER[group]]
        if profile.get("added_entry_ids") != expected_added:
            raise MigrationError(f"profile {name} has invalid broker entry metadata")
        legacy = profile.get("legacy")
        if not isinstance(legacy, list) or not legacy:
            raise MigrationError(f"profile {name} has invalid legacy metadata")
        ids = []
        for entry in legacy:
            if (
                not isinstance(entry, dict)
                or not set(entry).issubset(allowed_legacy_fields)
                or "id" not in entry
            ):
                raise MigrationError(
                    f"profile {name} has invalid legacy metadata fields"
                )
            if "label" in entry and not isinstance(entry["label"], str):
                raise MigrationError(
                    f"profile {name} has invalid legacy label metadata"
                )
            if "source" in entry and not isinstance(entry["source"], str):
                raise MigrationError(
                    f"profile {name} has invalid legacy source metadata"
                )
            if "priority" in entry and (
                isinstance(entry["priority"], bool)
                or not isinstance(entry["priority"], int)
            ):
                raise MigrationError(
                    f"profile {name} has invalid legacy priority metadata"
                )
            if "disabled" in entry and not isinstance(entry["disabled"], bool):
                raise MigrationError(
                    f"profile {name} has invalid legacy disabled metadata"
                )
            if "last_status" in entry and not (
                entry["last_status"] is None
                or isinstance(entry["last_status"], str)
            ):
                raise MigrationError(
                    f"profile {name} has invalid legacy status metadata"
                )
            if "last_status_at" in entry:
                status_at = entry["last_status_at"]
                if isinstance(status_at, bool) or not (
                    status_at is None
                    or isinstance(status_at, (int, float, str))
                ):
                    raise MigrationError(
                        f"profile {name} has invalid legacy status timestamp"
                    )
            ids.append(entry.get("id"))
        if (
            any(not isinstance(entry_id, str) or not entry_id for entry_id in ids)
            or len(set(ids)) != len(ids)
        ):
            raise MigrationError(f"profile {name} has invalid or duplicate legacy ids")
    group_counts = snapshot.get("group_counts")
    if (
        not isinstance(group_counts, dict)
        or set(group_counts) != set(expected_counts)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in group_counts.values()
        )
        or group_counts != expected_counts
    ):
        raise MigrationError("migration snapshot group counts are invalid")
    return port, names


def broker_entry_payload(alias: str, *, port: int, priority: int) -> dict:
    if alias not in ACCOUNT_ALIASES:
        raise MigrationError(f"unknown account alias {alias!r}")
    base_url = f"http://127.0.0.1:{_validate_port(port)}/accounts/{alias}/backend-api/codex"
    return {
        "id": f"broker-{alias}",
        "label": f"broker-{alias}",
        "auth_type": "api_key",
        "priority": priority,
        "source": "keychain_reference",
        "base_url": base_url,
        "secret_source": CLIENT_KEY_SECRET_SOURCE,
    }


def _auth_path(profiles_root: Path, name: str) -> Path:
    """Resolve one profile's auth.json, rejecting symlinks and escapes."""
    _validate_profile_name(name)
    root = Path(profiles_root)
    if root.is_symlink():
        raise MigrationError("profiles root uses symlinks; refusing")
    if name == "default":
        path = root / "auth.json"
        if path.is_symlink():
            raise MigrationError("profile default uses symlinks; refusing")
        if path.parent.resolve() != root.resolve():
            raise MigrationError("profile default escapes the profiles root; refusing")
        return path

    base = root / "profiles"
    if base.is_symlink():
        raise MigrationError("profiles root uses symlinks; refusing")
    profile_dir = base / name
    path = profile_dir / "auth.json"
    if profile_dir.is_symlink() or path.is_symlink():
        raise MigrationError(f"profile {name} uses symlinks; refusing")
    if not path.resolve().is_relative_to(base.resolve()):
        raise MigrationError(f"profile {name} escapes the profiles root; refusing")
    return path


def _decode_store(raw: bytes, path: Path) -> dict:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise MigrationError(f"cannot read profile store {path.name}") from exc
    if not isinstance(payload, dict):
        raise MigrationError(f"profile store {path.name} is not an object")
    return payload


def _canonical_store_sha256(store: dict) -> str:
    encoded = json.dumps(
        store,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_store(path: Path) -> tuple[dict, bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise MigrationError(f"cannot read profile store {path.name}") from exc
    return _decode_store(raw, path), raw


def _pool_entries(store: dict, path: Path) -> List[dict]:
    pool = store.get("credential_pool")
    entries = pool.get(_PROVIDER) if isinstance(pool, dict) else None
    if not isinstance(entries, list) or not entries:
        raise MigrationError(f"profile {path.parent.name} has no {_PROVIDER} pool")
    if any(not isinstance(entry, dict) for entry in entries):
        raise MigrationError(
            f"profile {path.parent.name} has a non-object {_PROVIDER} pool entry"
        )
    return entries


def _durable_write_bytes(path: Path, data: bytes) -> None:
    """Atomic, durable, owner-only write with full-write semantics.

    A randomized exclusive temp file in the target directory prevents planted
    fixed-name symlinks. The target changes only after every byte is written
    and fsynced; the parent directory is then fsynced for rename durability.
    """
    path = Path(path)
    staging: Optional[Path] = None
    try:
        fd, staging_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        staging = Path(staging_name)
        try:
            os.fchmod(fd, 0o600)
            view = memoryview(data)
            written = 0
            while written < len(view):
                count = os.write(fd, view[written:])
                if count <= 0:
                    raise OSError("short write made no progress")
                written += count
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(staging, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        if staging is not None:
            try:
                os.unlink(staging)
            except OSError:
                pass
        raise


def _write_profile_auth(path: Path, store: dict) -> None:
    _durable_write_bytes(path, json.dumps(store, indent=2).encode("utf-8"))


def _discover_profiles(profiles_root: Path) -> List[str]:
    root = Path(profiles_root)
    base = root / "profiles"
    if root.is_symlink() or base.is_symlink():
        raise MigrationError("profiles root uses symlinks; refusing")

    names: List[str] = []
    default_path = root / "auth.json"
    if default_path.is_symlink():
        raise MigrationError("profile default uses symlinks; refusing")
    if default_path.exists():
        if not default_path.is_file():
            raise MigrationError("default profile auth.json is not a regular file")
        names.append("default")

    if base.exists() and not base.is_dir():
        raise MigrationError(f"profiles path under {profiles_root} is not a directory")
    if base.is_dir():
        for entry in sorted(base.iterdir()):
            auth_path = entry / "auth.json"
            if not auth_path.is_file():
                continue
            if entry.is_symlink() or auth_path.is_symlink():
                raise MigrationError(f"profile {entry.name} uses symlinks; refusing")
            profile_name = _validate_profile_name(entry.name)
            if profile_name == "default" and "default" in names:
                raise MigrationError(
                    "default profile exists both at the Hermes root and under profiles/"
                )
            names.append(profile_name)

    if not names:
        raise MigrationError(f"no active profile auth stores under {profiles_root}")
    return sorted(names)


def plan_migration(
    profiles_root: Path, groups: Mapping[str, str], *, port: int
) -> dict:
    """Validate the rollout and return the redacted dry-run snapshot."""
    _validate_port(port)
    if any(not isinstance(name, str) for name in groups):
        raise MigrationError("group map contains an invalid profile name")
    discovered = _discover_profiles(profiles_root)
    if set(groups) != set(discovered):
        raise MigrationError(
            "group map does not match the active profile set "
            f"(groups={sorted(groups)}, discovered={discovered})"
        )
    if any(
        not isinstance(group, str) or group not in GROUP_ORDER
        for group in groups.values()
    ):
        raise MigrationError("group map contains an invalid group; expected A/B/C")

    group_counts = {alias: 0 for alias in ACCOUNT_ALIASES}
    profiles: Dict[str, dict] = {}
    for name in discovered:
        path = _auth_path(profiles_root, name)
        store, raw = _read_store(path)
        entries = _pool_entries(store, path)
        broker_ids = {f"broker-{alias}" for alias in ACCOUNT_ALIASES}
        existing = sorted(
            e.get("id") for e in entries if e.get("id") in broker_ids
        )
        if existing:
            raise MigrationError(
                f"profile {name} already has broker entries {existing}"
            )
        seen_ids: set = set()
        legacy_snaps: List[dict] = []
        for entry in entries:
            entry_id = entry.get("id")
            if not isinstance(entry_id, str) or not entry_id:
                raise MigrationError(
                    f"profile {name} has a pool entry without a stable id"
                )
            if entry_id in seen_ids:
                raise MigrationError(
                    f"profile {name} has duplicate pool entry id {entry_id!r}"
                )
            seen_ids.add(entry_id)
            legacy_snaps.append(
                {
                    field: entry[field]
                    for field in _LEGACY_SNAPSHOT_FIELDS
                    if field in entry
                }
            )
        group = groups[name]
        group_counts[group] += 1
        order = GROUP_ORDER[group]
        profiles[name] = {
            "group": group,
            "auth_sha256": hashlib.sha256(raw).hexdigest(),
            "auth_canonical_sha256": _canonical_store_sha256(store),
            "added_entry_ids": [f"broker-{alias}" for alias in order],
            "legacy": legacy_snaps,
        }
    return {
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "mode": "dry-run",
        "provider": _PROVIDER,
        "port": int(port),
        "groups": dict(groups),
        "group_counts": group_counts,
        "profiles": profiles,
    }


def save_snapshot(snapshot: dict, path: Path) -> Path:
    _validate_snapshot(snapshot)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _durable_write_bytes(path, json.dumps(snapshot, indent=2).encode("utf-8"))
    return path


def _transformed_store(store: dict, path: Path, group: str, port: int) -> dict:
    entries = _pool_entries(store, path)
    order = GROUP_ORDER[group]
    brokers = [
        broker_entry_payload(alias, port=port, priority=index)
        for index, alias in enumerate(order)
    ]
    archived = []
    for offset, entry in enumerate(entries):
        updated = dict(entry)
        updated["disabled"] = True
        updated["priority"] = len(brokers) + offset
        archived.append(updated)
    result = dict(store)
    pool = dict(result.get("credential_pool") or {})
    pool[_PROVIDER] = brokers + archived
    result["credential_pool"] = pool
    return result


def _restored_store(store: dict, path: Path, profile_snapshot: dict) -> dict:
    entries = _pool_entries(store, path)
    added = set(profile_snapshot["added_entry_ids"])
    by_id = {legacy["id"]: legacy for legacy in profile_snapshot["legacy"]}
    restored = []
    for entry in entries:
        if entry.get("id") in added:
            continue
        updated = dict(entry)
        snapshot_entry = by_id.get(updated.get("id"))
        if snapshot_entry is not None:
            # Exact-state restore: reproduce the captured presence AND value
            # of `disabled` and `priority` (absent keys stay absent).
            if "disabled" in snapshot_entry:
                updated["disabled"] = snapshot_entry["disabled"]
            else:
                updated.pop("disabled", None)
            if "priority" in snapshot_entry:
                updated["priority"] = snapshot_entry["priority"]
            else:
                updated.pop("priority", None)
        restored.append(updated)
    result = dict(store)
    pool = dict(result.get("credential_pool") or {})
    pool[_PROVIDER] = restored
    result["credential_pool"] = pool
    return result


def _restored_candidate_from_migrated(
    store: dict,
    path: Path,
    profile_snapshot: dict,
    *,
    port: int,
) -> dict:
    """Return the exact logical pre-migration store or reject any third state."""
    entries = _pool_entries(store, path)
    added_ids = list(profile_snapshot["added_entry_ids"])
    legacy = list(profile_snapshot["legacy"])
    legacy_ids = [entry["id"] for entry in legacy]
    current_ids = [entry.get("id") for entry in entries]
    if current_ids != added_ids + legacy_ids:
        raise MigrationError(
            f"profile {path.parent.name} is neither original nor exact migrated state"
        )

    entries_by_id = {entry["id"]: entry for entry in entries}
    for index, entry_id in enumerate(added_ids):
        alias = str(entry_id).removeprefix("broker-")
        expected = broker_entry_payload(alias, port=port, priority=index)
        actual = entries_by_id[entry_id]
        unknown_fields = set(actual) - set(expected) - _BROKER_RUNTIME_FIELDS
        if unknown_fields or any(
            actual.get(key) != value for key, value in expected.items()
        ):
            raise MigrationError(
                f"profile {path.parent.name} broker entry {entry_id!r} drifted"
            )
        if any(
            actual.get(field) not in (None, "")
            for field in ("access_token", "refresh_token", "agent_key")
        ):
            raise MigrationError(
                f"profile {path.parent.name} broker entry {entry_id!r} contains a persisted secret"
            )
        if actual.get("disabled") not in (None, False):
            raise MigrationError(
                f"profile {path.parent.name} broker entry {entry_id!r} is disabled"
            )
        request_count = actual.get("request_count", 0)
        if isinstance(request_count, bool) or not isinstance(request_count, int) or request_count < 0:
            raise MigrationError(
                f"profile {path.parent.name} broker entry {entry_id!r} has invalid request_count"
            )

    for offset, snapshot_entry in enumerate(legacy):
        actual = entries[len(added_ids) + offset]
        if actual.get("disabled") is not True or actual.get("priority") != (
            len(added_ids) + offset
        ):
            raise MigrationError(
                f"profile {path.parent.name} legacy migration metadata drifted"
            )
        for field in _LEGACY_SNAPSHOT_FIELDS:
            if field in {"priority", "disabled"}:
                continue
            if (field in actual) != (field in snapshot_entry) or (
                field in snapshot_entry and actual[field] != snapshot_entry[field]
            ):
                raise MigrationError(
                    f"profile {path.parent.name} legacy entry metadata drifted"
                )

    restored = _restored_store(store, path, profile_snapshot)
    if _canonical_store_sha256(restored) != profile_snapshot[
        "auth_canonical_sha256"
    ]:
        raise MigrationError(
            f"profile {path.parent.name} secret-bearing state drifted during migration"
        )
    return restored


def _snapshot_sha256(snapshot: dict) -> str:
    encoded = json.dumps(
        snapshot,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _effective_journal_path(
    profiles_root: Path,
    journal_path: Optional[Path],
    *,
    operation: str,
) -> Path:
    if journal_path is not None:
        path = Path(journal_path)
    else:
        path = Path(profiles_root) / f".oauth-broker-{operation}.journal"
    if path.is_symlink() or path.parent.is_symlink() or not path.parent.is_dir():
        raise MigrationError("migration journal path is not a safe regular-file target")
    return path


def _read_journal(path: Path, snapshot: dict, names: List[str], operation: str) -> Optional[dict]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise MigrationError("cannot safely open migration journal") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_JOURNAL_BYTES:
            raise MigrationError("migration journal is not a bounded regular file")
        if info.st_mode & 0o077:
            raise MigrationError("migration journal permissions are not owner-only")
        chunks = []
        remaining = _MAX_JOURNAL_BYTES + 1
        while remaining:
            chunk = os.read(fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > _MAX_JOURNAL_BYTES:
            raise MigrationError("migration journal exceeds the size limit")
    finally:
        os.close(fd)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise MigrationError("migration journal is malformed") from exc
    expected = {
        "journal_schema_version",
        "operation",
        "snapshot_sha256",
        "profiles",
        "written",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected
        or payload.get("journal_schema_version") != JOURNAL_SCHEMA_VERSION
        or payload.get("operation") != operation
        or payload.get("snapshot_sha256") != _snapshot_sha256(snapshot)
        or payload.get("profiles") != names
        or not isinstance(payload.get("written"), list)
        or any(name not in names for name in payload["written"])
        or len(set(payload["written"])) != len(payload["written"])
    ):
        raise MigrationError("migration journal does not match this snapshot")
    return payload


def _write_journal(
    path: Path,
    snapshot: dict,
    names: List[str],
    written: List[str],
    *,
    operation: str,
) -> None:
    payload = {
        "journal_schema_version": JOURNAL_SCHEMA_VERSION,
        "operation": operation,
        "snapshot_sha256": _snapshot_sha256(snapshot),
        "profiles": names,
        "written": list(written),
    }
    _durable_write_bytes(
        path,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8"),
    )


def _remove_journal(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError as exc:
        raise MigrationError("migration journal removal was not durable") from exc


def _recover_interrupted_apply(
    profiles_root: Path,
    snapshot: dict,
    journal_path: Path,
    *,
    port: int,
    names: List[str],
) -> Optional[Dict[str, bytes]]:
    if _read_journal(journal_path, snapshot, names, "apply") is None:
        return None
    recovered: Dict[str, bytes] = {}
    for name in names:
        path = _auth_path(profiles_root, name)
        store, raw = _read_store(path)
        profile_snapshot = snapshot["profiles"][name]
        if _canonical_store_sha256(store) == profile_snapshot["auth_canonical_sha256"]:
            recovered[name] = raw
            continue
        restored = _restored_candidate_from_migrated(
            store,
            path,
            profile_snapshot,
            port=port,
        )
        _write_profile_auth(path, restored)
        verified, verified_raw = _read_store(path)
        if _canonical_store_sha256(verified) != profile_snapshot[
            "auth_canonical_sha256"
        ]:
            raise MigrationError(f"profile {name} interrupted apply recovery failed")
        recovered[name] = verified_raw
    return recovered


def apply_migration(
    profiles_root: Path,
    snapshot: dict,
    *,
    journal_path: Optional[Path] = None,
) -> dict:
    """Apply after full preflight, with restart-safe full-fleet recovery."""
    port, names = _validate_snapshot(snapshot)
    if _discover_profiles(profiles_root) != names:
        raise MigrationError(
            "active profile set changed since the snapshot; re-run the dry-run"
        )
    effective_journal = _effective_journal_path(
        profiles_root,
        journal_path,
        operation="apply",
    )
    recovered_bytes = _recover_interrupted_apply(
        profiles_root,
        snapshot,
        effective_journal,
        port=port,
        names=names,
    )

    staged: Dict[str, dict] = {}
    original_bytes: Dict[str, bytes] = {}
    for name in names:
        path = _auth_path(profiles_root, name)
        store, raw = _read_store(path)
        if recovered_bytes is None:
            if hashlib.sha256(raw).hexdigest() != snapshot["profiles"][name][
                "auth_sha256"
            ]:
                raise MigrationError(
                    f"profile {name} changed since the snapshot (hash drift); "
                    "re-run the dry-run"
                )
        elif raw != recovered_bytes[name]:
            raise MigrationError(
                f"profile {name} changed during interrupted apply recovery"
            )
        original_bytes[name] = raw
        staged[name] = _transformed_store(
            store, path, snapshot["profiles"][name]["group"], port
        )

    written: List[str] = []
    _write_journal(
        effective_journal,
        snapshot,
        names,
        written,
        operation="apply",
    )
    try:
        for name in names:
            _write_profile_auth(_auth_path(profiles_root, name), staged[name])
            written.append(name)
            _write_journal(
                effective_journal,
                snapshot,
                names,
                written,
                operation="apply",
            )
    except BaseException as failure:
        restore_failures: List[str] = []
        for name in names:
            try:
                _durable_write_bytes(
                    _auth_path(profiles_root, name), original_bytes[name]
                )
            except BaseException:
                restore_failures.append(name)
        if restore_failures:
            raise MigrationError(
                "apply failed and full-fleet auto-restore was incomplete for "
                f"profiles {restore_failures}; the journal was preserved"
            ) from failure
        _remove_journal(effective_journal)
        if not isinstance(failure, Exception):
            raise
        raise MigrationError(
            "apply failed; auto-restored every profile to its exact original bytes"
        ) from failure

    _remove_journal(effective_journal)
    return {
        "applied": True,
        "written": written,
        "journal_path": effective_journal,
    }


def rollback_migration(
    profiles_root: Path,
    snapshot: dict,
    *,
    journal_path: Optional[Path] = None,
) -> dict:
    """Restore the full fleet, including a restart-safe mixed rollback state."""
    port, names = _validate_snapshot(snapshot)
    if _discover_profiles(profiles_root) != names:
        raise MigrationError(
            "active profile set changed since the snapshot; refusing rollback"
        )
    effective_journal = _effective_journal_path(
        profiles_root,
        journal_path,
        operation="rollback",
    )
    _read_journal(effective_journal, snapshot, names, "rollback")

    current_bytes: Dict[str, bytes] = {}
    restored_stores: Dict[str, dict] = {}
    for name in names:
        path = _auth_path(profiles_root, name)
        store, raw = _read_store(path)
        current_bytes[name] = raw
        profile_snapshot = snapshot["profiles"][name]
        if _canonical_store_sha256(store) == profile_snapshot[
            "auth_canonical_sha256"
        ]:
            restored_stores[name] = store
        else:
            restored_stores[name] = _restored_candidate_from_migrated(
                store,
                path,
                profile_snapshot,
                port=port,
            )

    written: List[str] = []
    _write_journal(
        effective_journal,
        snapshot,
        names,
        written,
        operation="rollback",
    )
    try:
        for name in names:
            _write_profile_auth(
                _auth_path(profiles_root, name),
                restored_stores[name],
            )
            written.append(name)
            _write_journal(
                effective_journal,
                snapshot,
                names,
                written,
                operation="rollback",
            )
    except BaseException as failure:
        reapply_failures: List[str] = []
        for name in names:
            try:
                _durable_write_bytes(
                    _auth_path(profiles_root, name), current_bytes[name]
                )
            except BaseException:
                reapply_failures.append(name)
        if reapply_failures:
            raise MigrationError(
                "rollback failed and full-fleet reapply was incomplete for "
                f"profiles {reapply_failures}; the journal was preserved"
            ) from failure
        _remove_journal(effective_journal)
        if not isinstance(failure, Exception):
            raise
        raise MigrationError(
            "rollback failed; reapplied exact pre-rollback bytes to every profile"
        ) from failure

    _remove_journal(effective_journal)
    return {"restored": written, "journal_path": effective_journal}


__all__ = [
    "CLIENT_KEY_SECRET_SOURCE",
    "GROUP_ORDER",
    "MigrationError",
    "apply_migration",
    "broker_entry_payload",
    "plan_migration",
    "rollback_migration",
    "save_snapshot",
]

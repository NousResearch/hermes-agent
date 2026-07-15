"""Gateway-UID first-wave state proof for the production prerequisite collector.

This child is executed with the exact production gateway uid/gid and a scrubbed
environment.  It performs only bounded reads plus ephemeral create/read/replace/
delete probes.  It does not classify work, choose tools, or inspect secrets.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import sys
from pathlib import Path
from typing import Any, Mapping


PROBE_SCHEMA = "muncho-production-gateway-state-proof.v1"
PRODUCTION_HOME = Path("/opt/adventico-ai-platform/hermes-home")
STATE_DIRECTORY = Path("/var/lib/hermes-cloud-gateway")
MAX_STATE_FILE_BYTES = 8 * 1024 * 1024


class GatewayStateProbeError(RuntimeError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identity(path: Path, *, allow_missing: bool = False) -> Mapping[str, Any]:
    try:
        item = path.lstat()
    except FileNotFoundError:
        if allow_missing:
            return {
                "path": str(path),
                "exists": False,
                "size": 0,
                "owner_uid": os.geteuid(),
                "group_gid": os.getegid(),
                "mode": None,
                "readable": True,
            }
        raise
    if (
        not stat.S_ISREG(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_nlink != 1
        or item.st_size > MAX_STATE_FILE_BYTES
    ):
        raise GatewayStateProbeError("gateway_state_file_invalid")
    with path.open("rb") as stream:
        payload = stream.read(MAX_STATE_FILE_BYTES + 1)
    if len(payload) != item.st_size or len(payload) > MAX_STATE_FILE_BYTES:
        raise GatewayStateProbeError("gateway_state_file_changed")
    return {
        "path": str(path),
        "exists": True,
        "size": item.st_size,
        "owner_uid": item.st_uid,
        "group_gid": item.st_gid,
        "mode": f"{stat.S_IMODE(item.st_mode):04o}",
        "readable": True,
    }


def _atomic_roundtrip(directory: Path, *, prefix: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_hex(24).encode("ascii")
    first = directory / f".{prefix}.{secrets.token_hex(8)}.first"
    second = directory / f".{prefix}.{secrets.token_hex(8)}.second"
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(first, flags, 0o600)
        try:
            os.write(descriptor, nonce)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if first.read_bytes() != nonce:
            raise GatewayStateProbeError("gateway_state_roundtrip_failed")
        descriptor = os.open(second, flags, 0o600)
        try:
            os.write(descriptor, nonce[::-1])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(second, first)
        if first.read_bytes() != nonce[::-1]:
            raise GatewayStateProbeError("gateway_state_atomic_replace_failed")
        first.unlink()
        parent = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        first.unlink(missing_ok=True)
        second.unlink(missing_ok=True)


def _memory_proof() -> Mapping[str, Any]:
    from tools.memory_tool import MemoryStore, get_memory_dir

    directory = get_memory_dir()
    if directory != PRODUCTION_HOME / "memories":
        raise GatewayStateProbeError("gateway_memory_home_drifted")
    store = MemoryStore()
    store.load_from_disk()
    probe = directory / f".muncho-prerequisite-memory-{secrets.token_hex(8)}.md"
    try:
        store._write_file(probe, ["first"])
        if store._read_file(probe) != ["first"]:
            raise GatewayStateProbeError("gateway_memory_create_load_failed")
        store._write_file(probe, ["second"])
        if store._read_file(probe) != ["second"]:
            raise GatewayStateProbeError("gateway_memory_atomic_rewrite_failed")
    finally:
        probe.unlink(missing_ok=True)
    return {
        "home": str(directory),
        "memory": _identity(directory / "MEMORY.md", allow_missing=True),
        "user": _identity(directory / "USER.md", allow_missing=True),
        "built_in_load": True,
        "built_in_atomic_create_rewrite": True,
    }


def _config_proof() -> Mapping[str, Any]:
    import yaml

    path = PRODUCTION_HOME / "config.yaml"
    raw = path.read_bytes()
    if not 0 < len(raw) <= 2 * 1024 * 1024:
        raise GatewayStateProbeError("gateway_config_invalid")
    try:
        config = yaml.safe_load(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, yaml.YAMLError) as exc:
        raise GatewayStateProbeError("gateway_config_invalid") from exc
    if not isinstance(config, Mapping):
        raise GatewayStateProbeError("gateway_config_invalid")
    memory = config.get("memory")
    if (
        not isinstance(memory, Mapping)
        or memory.get("memory_enabled") is not True
        or memory.get("user_profile_enabled") is not True
    ):
        raise GatewayStateProbeError("gateway_memory_config_not_enabled")
    return {
        "path": str(path),
        "memory_enabled": True,
        "user_profile_enabled": True,
        "readable": True,
    }


def _skills_proof(release: Path) -> Mapping[str, Any]:
    from agent.skill_utils import iter_skill_index_files

    bundled = release / "skills"
    paths = list(iter_skill_index_files(bundled, "SKILL.md"))
    if not 1 <= len(paths) <= 4096:
        raise GatewayStateProbeError("gateway_bundled_skills_index_invalid")
    records: list[Mapping[str, Any]] = []
    for path in paths:
        raw = path.read_bytes()
        if not 0 < len(raw) <= 2 * 1024 * 1024:
            raise GatewayStateProbeError("gateway_bundled_skill_invalid")
        records.append(
            {
                "path": str(path.relative_to(bundled)),
                "size": len(raw),
                "sha256": _sha256(raw),
            }
        )
    user = PRODUCTION_HOME / "skills"
    _atomic_roundtrip(user, prefix="muncho-prerequisite-skill")
    return {
        "bundled_path": str(bundled),
        "bundled_count": len(records),
        "bundled_index_sha256": _sha256(_canonical_bytes(records)),
        "user_path": str(user),
        "user_atomic_roundtrip": True,
    }


def _session_db_proof() -> Mapping[str, Any]:
    from hermes_state import SessionDB

    path = PRODUCTION_HOME / "state.db"
    database = SessionDB(db_path=path)
    try:
        mode = str(database._conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        if mode != "wal" or database._fts_enabled is not True:
            raise GatewayStateProbeError("gateway_session_db_not_wal_fts")
        result = database.search_messages(
            "muncho_prerequisite_fts_probe_no_match_expected",
            limit=1,
        )
        if not isinstance(result, list) or len(result) > 1:
            raise GatewayStateProbeError("gateway_session_db_fts_query_failed")
    finally:
        database.close()
    item = path.lstat()
    return {
        "path": str(path),
        "journal_mode": "wal",
        "fts5_enabled": True,
        "real_fts_query": True,
        "owner_uid": item.st_uid,
        "group_gid": item.st_gid,
        "mode": f"{stat.S_IMODE(item.st_mode):04o}",
    }


def _state_directory_proof() -> Mapping[str, Any]:
    item = STATE_DIRECTORY.lstat()
    if (
        not stat.S_ISDIR(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_uid != os.geteuid()
        or item.st_gid != os.getegid()
        or stat.S_IMODE(item.st_mode) != 0o700
    ):
        raise GatewayStateProbeError("gateway_state_directory_identity_invalid")
    _atomic_roundtrip(STATE_DIRECTORY, prefix="muncho-prerequisite-state")
    return {
        "path": str(STATE_DIRECTORY),
        "owner_uid": item.st_uid,
        "group_gid": item.st_gid,
        "mode": "0700",
        "atomic_roundtrip": True,
    }


def collect_gateway_state_proof(release: Path) -> Mapping[str, Any]:
    try:
        release = release.resolve(strict=True)
        release.relative_to(Path("/opt/adventico-ai-platform/hermes-agent-releases"))
    except (OSError, ValueError) as exc:
        raise GatewayStateProbeError("gateway_release_identity_invalid") from exc
    if Path(os.environ.get("HERMES_HOME", "")) != PRODUCTION_HOME:
        raise GatewayStateProbeError("gateway_home_not_exact")
    if Path.cwd() != release:
        raise GatewayStateProbeError("gateway_working_directory_not_exact")
    unsigned = {
        "schema": PROBE_SCHEMA,
        "gateway_uid": os.geteuid(),
        "gateway_gid": os.getegid(),
        "hermes_home": str(PRODUCTION_HOME),
        "config": _config_proof(),
        "memory": _memory_proof(),
        "skills": _skills_proof(release),
        "session_db": _session_db_proof(),
        "state_directory": _state_directory_proof(),
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {**unsigned, "proof_sha256": _sha256(_canonical_bytes(unsigned))}


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        return 2
    try:
        proof = collect_gateway_state_proof(Path(arguments[0]))
    except (OSError, RuntimeError, ValueError):
        return 2
    print(_canonical_bytes(proof).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

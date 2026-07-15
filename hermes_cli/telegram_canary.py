"""Synthetic, non-private Telegram gateway delivery canary.

The canary reuses the production authorization and Telegram send paths. It
injects one pre-send network failure (which cannot reach Telegram), delivers a
payload that must be split, and submits the same idempotency key twice so the
second attempt is suppressed locally. Raw chat/user identifiers and bot tokens
are never written to the receipt.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import marshal
import os
import re
import subprocess
import sys
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import CodeType
from typing import Any

from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.config import Platform
from gateway.platforms.base import SendResult, utf16_len
from gateway.session import SessionSource
from hermes_constants import get_hermes_home


CANARY_SCHEMA = "hermes.telegram-canary/v1"
_RUNTIME_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_DESTINATION_ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
_PROCESS_LOCKS: dict[str, threading.RLock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class LiveCanaryClaim:
    """Non-sensitive state needed to bind one live gateway delivery."""

    run_id: str
    idempotency_key_sha256: str
    created_at: str
    runtime_sha: str
    destination_alias: str
    source_message_id_hash: str
    source_update_id_hash: str


def canary_paths(hermes_home: Path | None = None) -> tuple[Path, Path]:
    """Return the fixed receipt and producer-state paths for this profile."""
    root = hermes_home if hermes_home is not None else get_hermes_home()
    receipt_root = root / "receipts"
    return (
        receipt_root / "telegram-canary.jsonl",
        receipt_root / "telegram-canary-state.json",
    )


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_live_payload(runtime_sha: str) -> str:
    """Build the deterministic non-private payload emitted by the quick command."""
    if not _RUNTIME_SHA_RE.fullmatch(runtime_sha):
        raise ValueError("runtime_sha must be an exact 40-character lowercase Git SHA")
    header = (
        "Hermes synthetic Telegram gateway canary. "
        "Non-private and not a production or domain acceptance receipt. "
        f"Runtime {runtime_sha}.\n"
    )
    return header + ("A" * 5000)


def _pyc_is_safe_for_tracked_source(cache_path: Path, source_path: Path) -> bool:
    """Return True when a PEP 3147 cache cannot alter the tracked source.

    Timestamp/size metadata is insufficient: an equal-size malicious source can
    be compiled and the tracked source's timestamp restored. Compare the loaded
    code object to a fresh compilation instead, using the optimization level
    encoded in the cache filename. Raw marshal bytes are not canonical because
    equivalent code objects can encode string-reference tables differently.

    PEP 552 metadata still determines whether ordinary import machinery can
    select the cache at all. A timestamp/size mismatch or a checked-hash
    mismatch makes the cache stale, so Python compiles the clean tracked source
    instead. Unchecked-hash and metadata-current caches remain content-attested.
    """
    optimization = 0
    match = re.search(r"\.opt-([0-9]+)\.pyc$", cache_path.name)
    if match:
        optimization = int(match.group(1))
        if optimization not in {1, 2}:
            return False
    try:
        cached = cache_path.read_bytes()
        if len(cached) <= 16 or cached[:4] != importlib.util.MAGIC_NUMBER:
            return False
        source_bytes = source_path.read_bytes()
        flags = int.from_bytes(cached[4:8], "little")
        if flags & ~0b11:
            return False
        if flags & 0b1:
            if (
                flags & 0b10
                and cached[8:16] != importlib.util.source_hash(source_bytes)
            ):
                return True
        else:
            source_stat = source_path.stat()
            cached_mtime = int.from_bytes(cached[8:12], "little")
            cached_size = int.from_bytes(cached[12:16], "little")
            if cached_mtime != (int(source_stat.st_mtime) & 0xFFFFFFFF):
                return True
            if cached_size != (source_stat.st_size & 0xFFFFFFFF):
                return True
        cached_code = marshal.loads(cached[16:])
        expected_code = compile(
            source_bytes,
            str(source_path),
            "exec",
            dont_inherit=True,
            optimize=optimization,
        )
        return isinstance(cached_code, CodeType) and cached_code == expected_code
    except (EOFError, OSError, SyntaxError, TypeError, ValueError):
        return False


def verify_running_runtime_sha(
    expected_sha: str,
    *,
    source_root: Path | None = None,
) -> bool:
    """Verify the executing checkout is the configured clean Git revision.

    Tracked, staged, and untracked files must all be clean. Ignored importable
    artifacts outside dependency/build roots are rejected too: a sourceless
    ``sitecustomize.pyc`` can change the executing runtime while ordinary Git
    status remains empty. Git is invoked without a shell and every command has
    a short timeout.
    """
    if not _RUNTIME_SHA_RE.fullmatch(str(expected_sha)):
        return False
    root = (source_root or Path(__file__).resolve().parents[1]).resolve()
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
        if head.returncode != 0 or head.stdout.strip() != expected_sha:
            return False
        clean = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignore-submodules=none",
            ],
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        if clean.returncode != 0 or clean.stdout:
            return False
        ignored = subprocess.run(
            [
                "git",
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
                "-z",
            ],
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        if ignored.returncode != 0:
            return False
        tracked = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        if tracked.returncode != 0:
            return False
        tracked_files = {
            os.fsdecode(path)
            for path in tracked.stdout.split(b"\0")
            if path
        }
        dependency_roots = {".venv", "venv", ".tox", ".nox", "node_modules"}
        executable_suffixes = {
            ".py",
            ".pyc",
            ".pyo",
            ".pth",
            ".so",
            ".dylib",
            ".pyd",
            ".egg-link",
        }
        for raw_path in ignored.stdout.split(b"\0"):
            if not raw_path:
                continue
            relative = Path(os.fsdecode(raw_path))
            if relative.parts and relative.parts[0] in dependency_roots:
                continue
            if relative.suffix.lower() == ".pyc" and relative.parent.name == "__pycache__":
                cache_tag = sys.implementation.cache_tag
                pytest_match = (
                    re.fullmatch(
                        rf"(?P<stem>.+)\.{re.escape(cache_tag)}-pytest-"
                        r"[A-Za-z0-9][A-Za-z0-9.]*\.pyc",
                        relative.name,
                    )
                    if cache_tag and relative.parts[0] == "tests"
                    else None
                )
                if pytest_match:
                    pytest_source = (
                        relative.parent.parent
                        / f"{pytest_match.group('stem')}.py"
                    ).as_posix()
                    # Pytest assertion-rewrite caches use a nonstandard name
                    # that Python's ordinary import machinery will not load.
                    # They are test-only artifacts, so tolerate them only under
                    # tests/ and only when bound to an exact tracked source.
                    if pytest_source in tracked_files:
                        continue
                    return False
                try:
                    source = Path(
                        importlib.util.source_from_cache(str(root / relative))
                    )
                    tracked_source = source.relative_to(root).as_posix()
                except (ValueError, OSError):
                    return False
                # Ordinary interpreter caches are safe to tolerate only when
                # their source is tracked and their code payload exactly
                # matches compiling that clean source. Sourceless/root bytecode,
                # forged caches, and caches for ignored modules fail closed.
                if tracked_source in tracked_files and _pyc_is_safe_for_tracked_source(
                    root / relative,
                    source,
                ):
                    continue
                return False
            if relative.suffix.lower() in executable_suffixes:
                return False
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def claim_live_canary(
    *,
    runtime_sha: str,
    destination_alias: str,
    message_id: Any,
    update_id: Any,
    state_path: Path,
    created_at: str | None = None,
) -> tuple[LiveCanaryClaim | None, bool]:
    """Claim one producer idempotency key before any subprocess or send.

    Returns ``(claim, duplicate)``. A retained pending claim intentionally
    suppresses a retry after a crash because Telegram acceptance would be
    ambiguous at that boundary.
    """
    if not _RUNTIME_SHA_RE.fullmatch(str(runtime_sha)):
        raise ValueError("runtime_sha must be an exact 40-character lowercase Git SHA")
    if not _DESTINATION_ALIAS_RE.fullmatch(str(destination_alias)):
        raise ValueError("destination_alias must be a bounded safe alias")
    message_text = str(message_id or "").strip()
    update_text = str(update_id or "").strip()
    if not message_text or not update_text:
        raise ValueError("live canary requires both Telegram message and update IDs")

    message_hash = _sha256_text(message_text)
    update_hash = _sha256_text(update_text)
    key = _sha256_text(
        "|".join(
            (
                CANARY_SCHEMA,
                runtime_sha,
                destination_alias,
                message_hash,
                update_hash,
            )
        )
    )
    with _exclusive_file_lock(state_path):
        state = _load_state(state_path)
        if key in state["runs"]:
            return None, True

        timestamp = created_at or _utc_now()
        run_uuid = str(uuid.UUID(hex=key.removeprefix("sha256:")[:32]))
        claim = LiveCanaryClaim(
            run_id=run_uuid,
            idempotency_key_sha256=key,
            created_at=timestamp,
            runtime_sha=runtime_sha,
            destination_alias=destination_alias,
            source_message_id_hash=message_hash,
            source_update_id_hash=update_hash,
        )
        state["runs"][key] = {
            "status": "pending",
            "updated_at": timestamp,
            "claim": asdict(claim),
        }
        _write_state(state_path, state)
        return claim, False


def mark_live_canary_pre_send_failure(
    claim: LiveCanaryClaim,
    *,
    state_path: Path,
    reason: str,
) -> None:
    """Record a bounded non-sensitive failure without allowing an unsafe replay."""
    with _exclusive_file_lock(state_path):
        state = _load_state(state_path)
        entry = state["runs"].get(claim.idempotency_key_sha256)
        if isinstance(entry, dict):
            entry["status"] = "pre_send_failed"
            entry["failure"] = str(reason)[:160]
            entry["updated_at"] = _utc_now()
            _write_state(state_path, state)


def finalize_live_canary(
    claim: LiveCanaryClaim,
    *,
    result: SendResult | None,
    payload: str,
    authentication: dict[str, bool],
    duplicate_probe_suppressed: bool,
    receipt_path: Path,
    state_path: Path,
) -> tuple[dict[str, Any], str]:
    """Bind the existing Telegram pipeline's final result to one durable receipt."""
    from plugins.platforms.telegram.adapter import prepare_legacy_text_chunks

    raw = result.raw_response if result and isinstance(result.raw_response, dict) else {}
    raw_message_ids = [str(item) for item in raw.get("message_ids", [])]
    message_hashes = [_sha256_text(item) for item in raw_message_ids]
    attempt_counts = [int(item) for item in raw.get("attempt_counts", [])]
    chunk_units = [int(item) for item in raw.get("chunk_utf16_units", [])]
    chunk_hashes = [str(item) for item in raw.get("chunk_sha256", [])]
    chunk_count = int(raw.get("chunk_count", 0) or 0)
    injected_failures = int(raw.get("synthetic_pre_send_failures", 0) or 0)
    acknowledged = bool(result and result.success)
    expected_chunks = prepare_legacy_text_chunks(payload)
    expected_units = [utf16_len(chunk) for chunk in expected_chunks]
    expected_hashes = [_sha256_text(chunk) for chunk in expected_chunks]
    exact_chunk_plan = bool(
        chunk_count == len(expected_chunks)
        and chunk_units == expected_units
        and chunk_hashes == expected_hashes
    )
    all_chunks_acknowledged = bool(
        acknowledged
        and chunk_count >= 2
        and len(message_hashes) == chunk_count
        and len(attempt_counts) == chunk_count
        and len(chunk_units) == chunk_count
        and len(chunk_hashes) == chunk_count
        and all(units <= 4096 for units in chunk_units)
        and exact_chunk_plan
    )
    retry_verified = bool(
        injected_failures == 1
        and attempt_counts
        and attempt_counts[0] == 2
        and all(attempt == 1 for attempt in attempt_counts[1:])
    )
    auth_verified = bool(authentication) and all(authentication.values())
    passed = bool(
        auth_verified
        and duplicate_probe_suppressed
        and retry_verified
        and all_chunks_acknowledged
    )
    receipt = {
        "schema": CANARY_SCHEMA,
        "mode": "live_gateway_pipeline",
        "run_id": claim.run_id,
        "created_at": claim.created_at,
        "completed_at": _utc_now(),
        "runtime_sha": claim.runtime_sha,
        "synthetic": True,
        "private_data": False,
        "qualifies_for_external_acceptance": False,
        "destination_alias": claim.destination_alias,
        "result": "pass" if passed else "fail",
        "checks": {
            "authentication": dict(authentication),
            "idempotency": {
                "scope": "canary_producer",
                "claim_before_delivery": True,
                "idempotency_key_sha256": claim.idempotency_key_sha256,
                "source_message_id_hash": claim.source_message_id_hash,
                "source_update_id_hash": claim.source_update_id_hash,
                "duplicate_probe_suppressed": bool(duplicate_probe_suppressed),
            },
            "retry": {
                "injected_pre_send_failures": injected_failures,
                "attempt_counts": attempt_counts,
                "safe_retry_verified": retry_verified,
            },
            "length": {
                "input_utf16_units": utf16_len(payload),
                "input_content_sha256": _sha256_text(payload),
                "chunk_count": chunk_count,
                "chunk_utf16_units": chunk_units,
                "chunk_sha256": chunk_hashes,
                "max_chunk_utf16_units": 4096,
                "all_chunks_acknowledged": all_chunks_acknowledged,
            },
            "delivery": {
                "acknowledged": acknowledged,
                "confirmation": "api_acknowledged" if acknowledged else "failed",
                "message_id_hashes": message_hashes,
                "thread_fallback": bool(raw.get("thread_fallback", False)),
            },
        },
    }
    receipt_sha256 = append_private_receipt(receipt_path, receipt)

    with _exclusive_file_lock(state_path):
        state = _load_state(state_path)
        entry = state["runs"].get(claim.idempotency_key_sha256)
        if isinstance(entry, dict):
            entry["status"] = "receipt_written"
            entry["result"] = receipt["result"]
            entry["receipt_record_sha256"] = _sha256_text(
                json.dumps(receipt, sort_keys=True, separators=(",", ":"))
            )
            entry["updated_at"] = receipt["completed_at"]
            _write_state(state_path, state)
    return receipt, receipt_sha256


def _parse_csv_env(name: str) -> set[str]:
    return {
        item.strip()
        for item in os.getenv(name, "").split(",")
        if item.strip()
    }


def strict_single_owner_id() -> str | None:
    """Return the sole Telegram owner ID, or ``None`` when not fail-closed."""
    allowed_ids = _parse_csv_env("TELEGRAM_ALLOWED_USERS") | _parse_csv_env(
        "GATEWAY_ALLOWED_USERS"
    )
    allow_all = any(
        os.getenv(name, "").strip().lower() in {"true", "1", "yes"}
        for name in ("TELEGRAM_ALLOW_ALL_USERS", "GATEWAY_ALLOW_ALL_USERS")
    )
    if allow_all or "*" in allowed_ids or len(allowed_ids) != 1:
        return None
    return next(iter(allowed_ids))


def _auth_checks() -> dict[str, bool]:
    """Exercise the production gateway DM authorization method fail-closed."""
    owner_id = strict_single_owner_id()

    checker = GatewayAuthorizationMixin()
    checker.adapters = {}
    checker.pairing_stores = {}
    checker.pairing_store = None

    def source(user_id: str) -> SessionSource:
        return SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="synthetic-canary-auth",
            chat_type="dm",
            user_id=user_id,
            user_name=None,
        )

    owner_allowed = bool(owner_id) and checker._is_user_authorized(source(owner_id))
    unknown_denied = not checker._is_user_authorized(
        source("synthetic-canary-unknown-user")
    )
    return {
        "gateway_path_exercised": True,
        "strict_single_owner_allowlist": owner_id is not None,
        "owner_allowed": owner_allowed,
        "unknown_denied": unknown_denied,
    }


def _secure_parent(path: Path) -> None:
    parent = path.parent
    if parent.exists() and parent.is_symlink():
        raise ValueError(f"refusing symlinked receipt directory: {parent}")
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent.chmod(0o700)


def _reject_symlink(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"refusing symlinked canary file: {path}")


def _fsync_directory(path: Path) -> None:
    """Durably persist directory-entry changes where the platform supports it."""
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_all(fd: int, payload: bytes) -> None:
    """Write every byte to a regular file or fail without silent truncation."""
    remaining = memoryview(payload)
    while remaining:
        written = os.write(fd, remaining)
        if written <= 0:
            raise OSError("short write while persisting canary state")
        remaining = remaining[written:]


@contextmanager
def _exclusive_file_lock(path: Path):
    """Serialize one receipt/state transaction across processes and threads."""
    _secure_parent(path)
    lock_path = path.with_name(f".{path.name}.lock")
    _reject_symlink(lock_path)
    lock_key = str(lock_path.resolve(strict=False))
    with _PROCESS_LOCKS_GUARD:
        process_lock = _PROCESS_LOCKS.setdefault(lock_key, threading.RLock())
    process_lock.acquire()
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(lock_path, flags, 0o600)
        os.fchmod(fd, 0o600)
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"0")
                    os.fsync(fd)
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                if os.name == "nt":
                    import msvcrt

                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
    finally:
        process_lock.release()


def _load_state(path: Path) -> dict[str, Any]:
    _secure_parent(path)
    _reject_symlink(path)
    if not path.exists():
        return {"schema": CANARY_SCHEMA, "runs": {}}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("canary state is unreadable; refusing delivery") from exc
    if loaded.get("schema") != CANARY_SCHEMA or not isinstance(loaded.get("runs"), dict):
        raise ValueError("canary state has an unsupported schema; refusing delivery")
    return loaded


def _write_state(path: Path, state: dict[str, Any]) -> None:
    _secure_parent(path)
    _reject_symlink(path)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(temp, flags, 0o600)
    try:
        payload = json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n"
        _write_all(fd, payload.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    temp.replace(path)
    path.chmod(0o600)
    _fsync_directory(path.parent)


def append_private_receipt(path: Path, receipt: dict[str, Any]) -> str:
    """Append one receipt record and return the SHA-256 of the full file."""
    with _exclusive_file_lock(path):
        _secure_parent(path)
        _reject_symlink(path)
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
        try:
            line = json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
            _write_all(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        path.chmod(0o600)
        _fsync_directory(path.parent)
        return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Emit only the deterministic payload; the running gateway owns delivery."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    payload_parser = subparsers.add_parser("payload")
    payload_parser.add_argument("--runtime-sha", required=True)
    args = parser.parse_args(argv)
    if args.command == "payload":
        sys.stdout.write(build_live_payload(args.runtime_sha))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

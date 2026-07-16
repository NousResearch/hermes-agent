"""Synthetic, non-private Telegram gateway delivery canary.

The canary reuses the production authorization and Telegram send paths. It
injects one pre-send network failure (which cannot reach Telegram), delivers a
payload that must be split, and submits the same idempotency key twice so the
second attempt is suppressed locally. Raw chat/user identifiers and bot tokens
are never written to the receipt.
"""

from __future__ import annotations

import _imp
import hashlib
import importlib.util
import json
import marshal
import os
import re
import stat
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
_MAX_STATE_BYTES = 8 * 1024 * 1024
_MAX_RECEIPT_BYTES = 32 * 1024 * 1024
_MAX_RECEIPT_RECORD_BYTES = 256 * 1024
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
    instead. Hash staleness also follows the interpreter's active
    ``--check-hash-based-pycs`` policy; modes that skip validation remain
    content-attested along with metadata-current caches.
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
            hash_policy = _imp.check_hash_based_pycs
            runtime_checks_hash = hash_policy == "always" or (
                hash_policy == "default" and bool(flags & 0b10)
            )
            if runtime_checks_hash and (
                cached[8:16] != importlib.util.source_hash(source_bytes)
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


def _trusted_git_executable() -> str | None:
    """Find Git without consulting the caller's mutable PATH."""
    candidates: list[Path] = []
    if os.name == "nt":
        candidates.extend(
            (
                Path(r"C:\Program Files\Git\cmd\git.exe"),
                Path(r"C:\Program Files (x86)\Git\cmd\git.exe"),
            )
        )
    else:
        candidates.extend((Path("/usr/bin/git"), Path("/bin/git")))
    for directory in os.defpath.split(os.pathsep):
        if directory and Path(directory).is_absolute():
            candidates.append(Path(directory) / ("git.exe" if os.name == "nt" else "git"))
    for candidate in candidates:
        try:
            info = candidate.lstat()
            if stat.S_ISREG(info.st_mode) and os.access(candidate, os.X_OK):
                return str(candidate)
        except OSError:
            continue
    return None


def _trusted_git_environment(git: str) -> dict[str, str]:
    """Return a minimal environment with caller-controlled Git state removed."""
    env: dict[str, str] = {
        "PATH": os.pathsep.join(
            dict.fromkeys(
                [
                    str(Path(git).parent),
                    *[
                        item
                        for item in os.defpath.split(os.pathsep)
                        if item and Path(item).is_absolute()
                    ],
                ]
            )
        ),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_COUNT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
        "LC_ALL": "C",
    }
    return env


def _run_trusted_git(
    git: str,
    root: Path,
    *args: str,
    text: bool = False,
) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        [git, *args],
        cwd=root,
        env=_trusted_git_environment(git),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=text,
        timeout=5,
        check=False,
    )


def _parse_head_tree(payload: bytes) -> dict[str, tuple[str, str]] | None:
    entries: dict[str, tuple[str, str]] = {}
    try:
        for raw_entry in payload.split(b"\0"):
            if not raw_entry:
                continue
            metadata, raw_path = raw_entry.split(b"\t", 1)
            mode, kind, object_id = metadata.split(b" ", 2)
            path = os.fsdecode(raw_path)
            if (
                kind != b"blob"
                or mode not in {b"100644", b"100755", b"120000"}
                or not path
                or path in entries
            ):
                return None
            entries[path] = (os.fsdecode(mode), os.fsdecode(object_id))
    except (UnicodeError, ValueError):
        return None
    return entries


def _parse_index(payload: bytes) -> dict[str, tuple[str, str]] | None:
    entries: dict[str, tuple[str, str]] = {}
    try:
        for raw_entry in payload.split(b"\0"):
            if not raw_entry:
                continue
            metadata, raw_path = raw_entry.split(b"\t", 1)
            mode, object_id, stage = metadata.split(b" ", 2)
            path = os.fsdecode(raw_path)
            if stage != b"0" or not path or path in entries:
                return None
            entries[path] = (os.fsdecode(mode), os.fsdecode(object_id))
    except (UnicodeError, ValueError):
        return None
    return entries


def _stable_tracked_blob_id(
    root: Path,
    relative_text: str,
    mode: str,
    object_format: str,
) -> str | None:
    """Hash one tracked path directly from the working tree without Git filters."""
    relative = Path(relative_text)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        return None
    candidate = root.joinpath(*relative.parts)
    try:
        current = root
        for component in relative.parts[:-1]:
            current = current / component
            parent_info = current.lstat()
            if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
                return None
        before = candidate.lstat()
        if mode == "120000":
            if os.name == "nt" and stat.S_ISREG(before.st_mode):
                content = candidate.read_bytes()
            elif stat.S_ISLNK(before.st_mode):
                content = os.fsencode(os.readlink(candidate))
            else:
                return None
            digest = hashlib.new(object_format)
            digest.update(f"blob {len(content)}\0".encode("ascii"))
            digest.update(content)
            return digest.hexdigest()
        if not stat.S_ISREG(before.st_mode):
            return None
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if hasattr(os, "O_NONBLOCK"):
            flags |= os.O_NONBLOCK
        fd = os.open(candidate, flags)
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode):
                return None
            digest = hashlib.new(object_format)
            digest.update(f"blob {opened.st_size}\0".encode("ascii"))
            total = 0
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                digest.update(chunk)
            final = os.fstat(fd)
        finally:
            os.close(fd)
        named = candidate.lstat()
        identity = lambda info: (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
        if total != opened.st_size or identity(opened) != identity(final):
            return None
        if identity(final) != identity(named):
            return None
        return digest.hexdigest()
    except (OSError, ValueError):
        return None


def verify_running_runtime_sha(
    expected_sha: str,
    *,
    source_root: Path | None = None,
) -> bool:
    """Verify the executing checkout is the configured clean Git revision.

    Tracked, staged, and untracked files must all be clean. Every tracked
    regular file is also hashed directly against the exact HEAD tree, so index
    flags and Git status optimizations cannot hide different executing bytes. Ignored importable
    artifacts outside dependency/build roots are rejected too: a sourceless
    ``sitecustomize.pyc`` can change the executing runtime while ordinary Git
    status remains empty. Git is resolved outside the caller's PATH, invoked
    with a scrubbed environment and no shell, and every command has a short timeout.
    """
    if not _RUNTIME_SHA_RE.fullmatch(str(expected_sha)):
        return False
    root = (source_root or Path(__file__).resolve().parents[1]).resolve()
    git = _trusted_git_executable()
    if git is None:
        return False
    try:
        top = _run_trusted_git(git, root, "rev-parse", "--show-toplevel", text=True)
        if top.returncode != 0 or Path(top.stdout.strip()).resolve() != root:
            return False
        head = _run_trusted_git(git, root, "rev-parse", "--verify", "HEAD", text=True)
        if head.returncode != 0 or head.stdout.strip() != expected_sha:
            return False
        object_format_result = _run_trusted_git(
            git, root, "rev-parse", "--show-object-format", text=True
        )
        object_format = object_format_result.stdout.strip()
        if object_format_result.returncode != 0 or object_format not in hashlib.algorithms_available:
            return False
        clean = _run_trusted_git(
            git,
            root,
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        )
        if clean.returncode != 0 or clean.stdout:
            return False
        ignored = _run_trusted_git(
            git,
            root,
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
        )
        if ignored.returncode != 0:
            return False
        tree_result = _run_trusted_git(git, root, "ls-tree", "-r", "-z", "HEAD")
        index_result = _run_trusted_git(git, root, "ls-files", "--stage", "-z")
        if tree_result.returncode != 0 or index_result.returncode != 0:
            return False
        tree = _parse_head_tree(tree_result.stdout)
        index = _parse_index(index_result.stdout)
        if tree is None or index != tree:
            return False
        for path, (mode, object_id) in tree.items():
            if _stable_tracked_blob_id(root, path, mode, object_format) != object_id:
                return False
        tracked_files = set(tree)
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
        existing = state["runs"].get(key)
        if existing is not None:
            if not isinstance(existing, dict):
                raise ValueError("canary state contains an invalid retained claim")
            if existing.get("status") in {"finalizing", "receipt_written"}:
                claim_data = existing.get("claim")
                try:
                    retained_claim = LiveCanaryClaim(**claim_data)
                except (TypeError, ValueError):
                    raise ValueError(
                        "canary state contains an invalid retained claim"
                    ) from None
                if (
                    retained_claim.idempotency_key_sha256 != key
                    or retained_claim.runtime_sha != runtime_sha
                    or retained_claim.destination_alias != destination_alias
                    or retained_claim.source_message_id_hash != message_hash
                    or retained_claim.source_update_id_hash != update_hash
                ):
                    raise ValueError("canary state retained claim does not match replay")
                return retained_claim, True
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
    with _exclusive_file_lock(state_path):
        state = _load_state(state_path)
        entry = state["runs"].get(claim.idempotency_key_sha256)
        if (
            not isinstance(entry, dict)
            or entry.get("claim") != asdict(claim)
            or entry.get("status") not in {
                "pending",
                "finalizing",
                "receipt_written",
            }
        ):
            raise ValueError("finalize requires the exact pending canary claim")

        status = entry["status"]
        if status == "pending":
            # Seal the exact post-delivery record in producer state before the
            # append. If the process dies between those writes, a retry must
            # recover this record rather than accepting a same-key record with
            # different checks or inventing a new completion timestamp.
            entry["status"] = "finalizing"
            entry["receipt"] = receipt
            entry["receipt_record_sha256"] = _sha256_text(
                json.dumps(receipt, sort_keys=True, separators=(",", ":"))
            )
            entry["updated_at"] = receipt["completed_at"]
            _write_state(state_path, state)
            expected_receipt = receipt
        else:
            expected_receipt = entry.get("receipt")
            if not isinstance(expected_receipt, dict):
                raise ValueError("finalized canary state is missing its sealed receipt")
            expected_record_sha = _sha256_text(
                json.dumps(expected_receipt, sort_keys=True, separators=(",", ":"))
            )
            if entry.get("receipt_record_sha256") != expected_record_sha:
                raise ValueError("finalized canary state has a receipt hash mismatch")

        persisted_receipt, receipt_sha256 = _append_or_find_receipt(
            receipt_path,
            expected_receipt,
            dedupe=True,
            require_existing=status == "receipt_written",
        )
        record_sha = _sha256_text(
            json.dumps(persisted_receipt, sort_keys=True, separators=(",", ":"))
        )
        if entry.get("receipt_record_sha256") != record_sha:
            raise ValueError("finalized canary state does not match its receipt")
        entry["status"] = "receipt_written"
        entry["result"] = persisted_receipt["result"]
        entry["updated_at"] = persisted_receipt["completed_at"]
        _write_state(state_path, state)
    return persisted_receipt, receipt_sha256


def recover_sealed_live_canary(
    claim: LiveCanaryClaim,
    *,
    receipt_path: Path,
    state_path: Path,
) -> tuple[dict[str, Any], str]:
    """Complete an already-sealed receipt without repeating delivery.

    A process may fail after the post-delivery receipt was sealed into producer
    state but before the append or final state readback. A replay of the exact
    Telegram update remains delivery-suppressed, but can use this function to
    finish the durable receipt transaction from the sealed record.
    """
    with _exclusive_file_lock(state_path):
        state = _load_state(state_path)
        entry = state["runs"].get(claim.idempotency_key_sha256)
        if (
            not isinstance(entry, dict)
            or entry.get("claim") != asdict(claim)
            or entry.get("status") not in {"finalizing", "receipt_written"}
        ):
            raise ValueError("recovery requires the exact sealed canary claim")
        expected_receipt = entry.get("receipt")
        if not isinstance(expected_receipt, dict):
            raise ValueError("sealed canary state is missing its receipt")
        expected_record_sha = _sha256_text(
            json.dumps(expected_receipt, sort_keys=True, separators=(",", ":"))
        )
        if entry.get("receipt_record_sha256") != expected_record_sha:
            raise ValueError("sealed canary state has a receipt hash mismatch")

        persisted_receipt, receipt_sha256 = _append_or_find_receipt(
            receipt_path,
            expected_receipt,
            dedupe=True,
            require_existing=entry.get("status") == "receipt_written",
        )
        record_sha = _sha256_text(
            json.dumps(persisted_receipt, sort_keys=True, separators=(",", ":"))
        )
        if record_sha != expected_record_sha:
            raise ValueError("sealed canary state does not match its receipt")
        entry["status"] = "receipt_written"
        entry["result"] = persisted_receipt["result"]
        entry["updated_at"] = persisted_receipt["completed_at"]
        _write_state(state_path, state)
        return persisted_receipt, receipt_sha256


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


def _absolute_canary_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _is_owned_by_current_user(info: os.stat_result) -> bool:
    get_euid = getattr(os, "geteuid", None)
    return get_euid is None or info.st_uid == get_euid()


def _verify_parent_path(parent: Path, opened: os.stat_result) -> None:
    try:
        named = parent.lstat()
    except OSError as exc:
        raise ValueError("canary parent changed during canary file access") from exc
    if (
        stat.S_ISLNK(named.st_mode)
        or not stat.S_ISDIR(named.st_mode)
        or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        raise ValueError("canary parent changed during canary file access")


@contextmanager
def _secure_parent_handle(path: Path):
    """Open the immediate parent through no-follow directory descriptors."""
    absolute = _absolute_canary_path(path)
    parent = absolute.parent
    supports_dir_fd = (
        os.name != "nt"
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and os.open in getattr(os, "supports_dir_fd", set())
    )
    if not supports_dir_fd:
        current = Path(parent.anchor)
        for component in parent.parts[1:]:
            current = current / component
            try:
                info = current.lstat()
            except FileNotFoundError:
                current.mkdir(mode=0o700)
                info = current.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"refusing unsafe canary directory: {current}")
        info = parent.lstat()
        if not _is_owned_by_current_user(info):
            raise ValueError("canary parent must be owned by the current user")
        parent.chmod(0o700)
        info = parent.lstat()
        try:
            yield None, absolute.name, absolute, info
        finally:
            _verify_parent_path(parent, info)
        return

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(parent.anchor, directory_flags)
    try:
        for component in parent.parts[1:]:
            try:
                child = os.open(component, directory_flags, dir_fd=fd)
            except FileNotFoundError:
                os.mkdir(component, 0o700, dir_fd=fd)
                child = os.open(component, directory_flags, dir_fd=fd)
            os.close(fd)
            fd = child
        info = os.fstat(fd)
        if not stat.S_ISDIR(info.st_mode) or not _is_owned_by_current_user(info):
            raise ValueError("canary parent must be an owned directory")
        os.fchmod(fd, 0o700)
        info = os.fstat(fd)
        try:
            yield fd, absolute.name, absolute, info
        finally:
            _verify_parent_path(parent, info)
    finally:
        os.close(fd)


def _open_at(parent_fd: int | None, absolute: Path, name: str, flags: int) -> int:
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if parent_fd is None:
        return os.open(absolute, flags, 0o600)
    return os.open(name, flags, 0o600, dir_fd=parent_fd)


def _named_stat(parent_fd: int | None, absolute: Path, name: str) -> os.stat_result:
    if parent_fd is None:
        return absolute.lstat()
    return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)


def _validate_private_regular_fd(
    fd: int,
    parent_fd: int | None,
    absolute: Path,
    name: str,
) -> os.stat_result:
    opened = os.fstat(fd)
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or not _is_owned_by_current_user(opened)
        or stat.S_IMODE(opened.st_mode) & 0o077
    ):
        raise ValueError("canary file must be an owner-private regular single-link file")
    if hasattr(os, "fchmod"):
        os.fchmod(fd, 0o600)
    else:
        absolute.chmod(0o600)
    named = _named_stat(parent_fd, absolute, name)
    if (
        not stat.S_ISREG(named.st_mode)
        or named.st_nlink != 1
        or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        raise ValueError("canary file changed during canary file access")
    return os.fstat(fd)


def _verify_private_regular_identity(
    fd: int,
    parent_fd: int | None,
    absolute: Path,
    name: str,
    opened: os.stat_result,
) -> None:
    final = os.fstat(fd)
    named = _named_stat(parent_fd, absolute, name)
    if _identity(opened) != _identity(final) or _identity(final) != _identity(named):
        raise ValueError("canary file changed during canary file access")


def _read_fd_stable(fd: int, limit: int) -> bytes:
    def read_once() -> bytes:
        os.lseek(fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, min(1024 * 1024, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                raise ValueError("canary file exceeds its bounded size")
        return b"".join(chunks)

    before = os.fstat(fd)
    first = read_once()
    middle = os.fstat(fd)
    second = read_once()
    final = os.fstat(fd)
    if first != second or _identity(before) != _identity(middle) or _identity(middle) != _identity(final):
        raise ValueError("canary file changed during bounded read")
    return second


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
    lock_path = _absolute_canary_path(path).with_name(f".{path.name}.lock")
    lock_key = str(lock_path)
    with _PROCESS_LOCKS_GUARD:
        process_lock = _PROCESS_LOCKS.setdefault(lock_key, threading.RLock())
    process_lock.acquire()
    try:
        with _secure_parent_handle(lock_path) as (parent_fd, name, absolute, _):
            fd = _open_at(parent_fd, absolute, name, os.O_RDWR | os.O_CREAT)
            try:
                opened = _validate_private_regular_fd(fd, parent_fd, absolute, name)
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
                _verify_private_regular_identity(fd, parent_fd, absolute, name, os.fstat(fd))
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
    try:
        with _secure_parent_handle(path) as (parent_fd, name, absolute, _):
            try:
                fd = _open_at(parent_fd, absolute, name, os.O_RDONLY)
            except FileNotFoundError:
                return {"schema": CANARY_SCHEMA, "runs": {}}
            try:
                opened = _validate_private_regular_fd(fd, parent_fd, absolute, name)
                payload = _read_fd_stable(fd, _MAX_STATE_BYTES)
                _verify_private_regular_identity(fd, parent_fd, absolute, name, opened)
            finally:
                os.close(fd)
        loaded = json.loads(payload.decode("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("canary state is unreadable; refusing delivery") from exc
    if loaded.get("schema") != CANARY_SCHEMA or not isinstance(loaded.get("runs"), dict):
        raise ValueError("canary state has an unsupported schema; refusing delivery")
    return loaded


def _write_state(path: Path, state: dict[str, Any]) -> None:
    payload = (json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if len(payload) > _MAX_STATE_BYTES:
        raise ValueError("canary state exceeds its bounded size")
    with _secure_parent_handle(path) as (parent_fd, name, absolute, _):
        temp_name = f".{name}.{uuid.uuid4().hex}.tmp"
        temp_absolute = absolute.with_name(temp_name)
        fd = _open_at(
            parent_fd,
            temp_absolute,
            temp_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
        )
        try:
            opened = _validate_private_regular_fd(
                fd, parent_fd, temp_absolute, temp_name
            )
            _write_all(fd, payload)
            os.fsync(fd)
            _verify_private_regular_identity(
                fd, parent_fd, temp_absolute, temp_name, os.fstat(fd)
            )
        finally:
            os.close(fd)
        try:
            try:
                existing = _open_at(parent_fd, absolute, name, os.O_RDONLY)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                try:
                    _validate_private_regular_fd(existing, parent_fd, absolute, name)
                finally:
                    os.close(existing)
            if parent_fd is None:
                os.replace(temp_absolute, absolute)
            else:
                os.replace(temp_name, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            readback = _open_at(parent_fd, absolute, name, os.O_RDONLY)
            try:
                opened = _validate_private_regular_fd(
                    readback, parent_fd, absolute, name
                )
                if _read_fd_stable(readback, _MAX_STATE_BYTES) != payload:
                    raise ValueError("canary state readback mismatch")
                _verify_private_regular_identity(
                    readback, parent_fd, absolute, name, opened
                )
            finally:
                os.close(readback)
            if parent_fd is not None:
                os.fsync(parent_fd)
        finally:
            try:
                if parent_fd is None:
                    temp_absolute.unlink()
                else:
                    os.unlink(temp_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass


def _quarantine_partial_tail(
    parent_fd: int | None,
    absolute: Path,
    name: str,
    fragment: bytes,
) -> None:
    digest = hashlib.sha256(fragment).hexdigest()
    quarantine_name = f".{name}.partial-{digest}.bin"
    quarantine_absolute = absolute.with_name(quarantine_name)
    try:
        fd = _open_at(
            parent_fd,
            quarantine_absolute,
            quarantine_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
        )
    except FileExistsError:
        fd = _open_at(parent_fd, quarantine_absolute, quarantine_name, os.O_RDONLY)
        try:
            opened = _validate_private_regular_fd(
                fd, parent_fd, quarantine_absolute, quarantine_name
            )
            if _read_fd_stable(fd, _MAX_RECEIPT_RECORD_BYTES) != fragment:
                raise ValueError("partial-tail quarantine does not match preserved evidence")
            _verify_private_regular_identity(
                fd, parent_fd, quarantine_absolute, quarantine_name, opened
            )
        finally:
            os.close(fd)
        return
    try:
        _validate_private_regular_fd(fd, parent_fd, quarantine_absolute, quarantine_name)
        _write_all(fd, fragment)
        os.fsync(fd)
        _verify_private_regular_identity(
            fd,
            parent_fd,
            quarantine_absolute,
            quarantine_name,
            os.fstat(fd),
        )
    finally:
        os.close(fd)


def _prepare_receipt_records(
    fd: int,
    parent_fd: int | None,
    absolute: Path,
    name: str,
) -> list[dict[str, Any]]:
    payload = _read_fd_stable(fd, _MAX_RECEIPT_BYTES)
    if payload and not payload.endswith(b"\n"):
        tail_start = payload.rfind(b"\n") + 1
        fragment = payload[tail_start:]
        if len(fragment) > _MAX_RECEIPT_RECORD_BYTES:
            raise ValueError("partial canary receipt exceeds its bounded record size")
        try:
            candidate = json.loads(fragment.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            _quarantine_partial_tail(parent_fd, absolute, name, fragment)
            os.ftruncate(fd, tail_start)
        else:
            if not isinstance(candidate, dict) or candidate.get("schema") != CANARY_SCHEMA:
                raise ValueError("partial canary receipt has an unsupported schema")
            os.lseek(fd, 0, os.SEEK_END)
            _write_all(fd, b"\n")
        os.fsync(fd)
        payload = _read_fd_stable(fd, _MAX_RECEIPT_BYTES)
    records: list[dict[str, Any]] = []
    for line in payload.splitlines():
        if len(line) > _MAX_RECEIPT_RECORD_BYTES:
            raise ValueError("canary receipt record exceeds its bounded size")
        try:
            record = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("canary receipt contains an invalid committed record") from exc
        if not isinstance(record, dict) or record.get("schema") != CANARY_SCHEMA:
            raise ValueError("canary receipt contains an unsupported record")
        records.append(record)
    return records


def _receipt_binding(receipt: dict[str, Any]) -> tuple[Any, ...]:
    checks = receipt.get("checks")
    idempotency = checks.get("idempotency") if isinstance(checks, dict) else None
    if not isinstance(idempotency, dict):
        idempotency = {}
    return (
        receipt.get("schema"),
        receipt.get("run_id"),
        receipt.get("created_at"),
        receipt.get("runtime_sha"),
        receipt.get("destination_alias"),
        idempotency.get("idempotency_key_sha256"),
        idempotency.get("source_message_id_hash"),
        idempotency.get("source_update_id_hash"),
    )


def _append_or_find_receipt(
    path: Path,
    receipt: dict[str, Any],
    *,
    dedupe: bool,
    require_existing: bool = False,
) -> tuple[dict[str, Any], str]:
    line = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if len(line) > _MAX_RECEIPT_RECORD_BYTES:
        raise ValueError("canary receipt record exceeds its bounded size")
    with _exclusive_file_lock(path):
        with _secure_parent_handle(path) as (parent_fd, name, absolute, _):
            fd = _open_at(parent_fd, absolute, name, os.O_RDWR | os.O_APPEND | os.O_CREAT)
            try:
                _validate_private_regular_fd(fd, parent_fd, absolute, name)
                records = _prepare_receipt_records(fd, parent_fd, absolute, name)
                persisted = receipt
                if dedupe:
                    matches = [
                        item
                        for item in records
                        if _receipt_binding(item)[5] == _receipt_binding(receipt)[5]
                    ]
                    if len(matches) > 1 or (
                        matches and _receipt_binding(matches[0]) != _receipt_binding(receipt)
                    ):
                        raise ValueError("canary receipt conflicts with its pending claim")
                    if matches:
                        if matches[0] != receipt:
                            raise ValueError(
                                "canary receipt conflicts with its pending claim"
                            )
                        persisted = matches[0]
                    elif require_existing:
                        raise ValueError("finalized canary receipt is missing")
                    else:
                        os.lseek(fd, 0, os.SEEK_END)
                        if os.fstat(fd).st_size + len(line) > _MAX_RECEIPT_BYTES:
                            raise ValueError("canary receipt file exceeds its bounded size")
                        _write_all(fd, line)
                        os.fsync(fd)
                else:
                    os.lseek(fd, 0, os.SEEK_END)
                    if os.fstat(fd).st_size + len(line) > _MAX_RECEIPT_BYTES:
                        raise ValueError("canary receipt file exceeds its bounded size")
                    _write_all(fd, line)
                    os.fsync(fd)
                payload = _read_fd_stable(fd, _MAX_RECEIPT_BYTES)
                opened = os.fstat(fd)
                _verify_private_regular_identity(
                    fd, parent_fd, absolute, name, opened
                )
                if parent_fd is not None:
                    os.fsync(parent_fd)
                return persisted, hashlib.sha256(payload).hexdigest()
            finally:
                os.close(fd)


def append_private_receipt(path: Path, receipt: dict[str, Any]) -> str:
    """Append one receipt record and return the SHA-256 of the full file."""
    _, file_sha = _append_or_find_receipt(path, receipt, dedupe=False)
    return file_sha


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

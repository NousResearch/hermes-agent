"""Turn-bound staging attachment registry for discrete bubble streaming.

Provides secure, authenticated attachment registration, private sandbox staging,
cryptographic integrity hashing (SHA-256), TOCTOU prevention, single-use delivery tracking,
and controlled lifecycle cleanup.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("gateway.attachment_registry")

# Maximum permitted attachment size (50 MB)
MAX_ATTACHMENT_BYTES = 50 * 1024 * 1024
MAX_ATTACHMENTS_PER_TURN = 5

_DENIED_SOURCE_BASENAMES = {
    ".env", ".envrc", "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa",
    "known_hosts", "credentials", "credentials.json", "service_account.json",
    "authorized_keys", "master.key", "shadow", "passwd",
}

_DENIED_SOURCE_EXTENSIONS = {
    ".pem", ".key", ".pkcs12", ".pfx", ".kdbx",
}


def is_pid_alive(pid: int) -> bool:
    """Check whether a process with the given PID is currently alive on the host."""
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                return False
            exit_code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)
            return exit_code.value == 259  # STILL_ACTIVE
        except Exception:  # health: allow BLE001 -- process query failure fallback
            return True
    else:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except Exception:  # health: allow BLE001 -- posix process signal fallback
            return True


class AttachmentState(str, Enum):
    """Lifecycle state machine for turn-bound attachments."""

    REGISTERED = "REGISTERED"
    CLAIMED = "CLAIMED"
    DISPATCHING = "DISPATCHING"
    SUCCEEDED = "SUCCEEDED"
    FAILED_PRE_DISPATCH = "FAILED_PRE_DISPATCH"
    OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"
    CLEANUP_ELIGIBLE = "CLEANUP_ELIGIBLE"


@dataclass
class AttachmentRecord:
    """Metadata and delivery lifecycle tracking for an authorized staged attachment."""

    token: str
    chat_id: str
    turn_id: Optional[str]
    producer: str
    original_path: str
    staged_path: str
    sha256: str
    file_size: int
    file_name: str
    caption: str = ""
    created_at: float = field(default_factory=time.monotonic)
    state: AttachmentState = AttachmentState.REGISTERED
    attempts: int = 0
    max_attempts: int = 3
    outcome_unknown_at: Optional[float] = None
    quarantine_duration_seconds: float = 120.0

    @property
    def in_flight(self) -> bool:
        return self.state in (AttachmentState.CLAIMED, AttachmentState.DISPATCHING)

    @in_flight.setter
    def in_flight(self, val: bool) -> None:
        if val:
            if self.state in (AttachmentState.SUCCEEDED, AttachmentState.OUTCOME_UNKNOWN, AttachmentState.CLEANUP_ELIGIBLE):
                raise ValueError(f"Cannot set in_flight=True on terminal attachment state {self.state}")
            if self.state not in (AttachmentState.CLAIMED, AttachmentState.DISPATCHING):
                self.state = AttachmentState.CLAIMED
        else:
            if self.state in (AttachmentState.CLAIMED, AttachmentState.DISPATCHING):
                self.state = AttachmentState.REGISTERED

    @property
    def delivered(self) -> bool:
        return self.state == AttachmentState.SUCCEEDED

    @delivered.setter
    def delivered(self, val: bool) -> None:
        if val:
            if self.state in (AttachmentState.OUTCOME_UNKNOWN, AttachmentState.CLEANUP_ELIGIBLE):
                raise ValueError(f"Cannot set delivered=True on terminal attachment state {self.state}")
            self.state = AttachmentState.SUCCEEDED
        else:
            if self.state == AttachmentState.SUCCEEDED:
                raise ValueError("Cannot regress terminal state SUCCEEDED via delivered=False")


class AttachmentRegistry:
    """Thread-safe registry managing verified artifacts produced by trusted internal generators."""

    def __init__(self, base_staging_dir: Optional[Path] = None) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, AttachmentRecord] = {}
        self._staged_path_to_token: dict[str, str] = {}
        self._chat_turn_index: dict[str, set[str]] = {}
        self._base_staging_dir = (
            base_staging_dir or Path(tempfile.gettempdir()) / "airhermes_staging"
        )
        self._ensure_base_staging_dir()
        self.recover_abandoned_staging()

    def recover_abandoned_staging(
        self,
        max_age_seconds: float = 86400,
        min_abandoned_age_seconds: float = 60.0,
    ) -> int:
        """Scan and clean up abandoned staging directories from prior runs or crashed processes.

        Safety guarantees:
        - Never follows unexpected symlinks or junctions.
        - Directory ownership and boundary inside _base_staging_dir are verified.
        - Never deletes directories belonging to active, running PIDs.
        - Dead-PID directories are cleaned only if older than min_abandoned_age_seconds.
        - Legacy directories are cleaned if older than max_age_seconds.
        - Idempotent and safe across multiple invocations.
        """
        recovered = 0
        now = time.time()
        try:
            if not self._base_staging_dir.exists():
                return 0
            resolved_base = self._base_staging_dir.resolve()
            for child in list(self._base_staging_dir.iterdir()):
                # Symlink / junction protection: never follow or traverse into reparse points
                if os.path.islink(child) or child.is_symlink():
                    try:
                        child.unlink(missing_ok=True)
                        recovered += 1
                    except Exception:  # health: allow S110, BLE001 -- unlink symlink cleanup fallback
                        pass
                    continue

                if not child.is_dir():
                    continue

                # Directory boundary check
                try:
                    if child.resolve().parent != resolved_base:
                        logger.warning("[AttachmentRegistry] Skipping directory outside base staging: %s", child)
                        continue
                except Exception:  # health: allow S110, BLE001 -- invalid directory resolve fallback
                    continue

                # Process identity check: proc_<pid>_<opaque>
                match = re.match(r"^proc_(?P<pid>\d+)_[a-f0-9]+$", child.name, re.IGNORECASE)
                should_remove = False

                if match:
                    owner_pid = int(match.group("pid"))
                    if owner_pid == os.getpid():
                        # Current process: active session, never touch in recovery
                        continue
                    if is_pid_alive(owner_pid):
                        # Another process is actively running: never delete active process's artifacts
                        continue
                    # Process is confirmed DEAD
                    try:
                        mtime = child.stat().st_mtime
                        if now - mtime >= min_abandoned_age_seconds:
                            should_remove = True
                    except Exception:  # health: allow S110, BLE001 -- stat error fallback
                        pass
                else:
                    # Legacy directory format without proc_ prefix
                    try:
                        mtime = child.stat().st_mtime
                        if now - mtime >= max_age_seconds:
                            should_remove = True
                    except Exception:  # health: allow S110, BLE001 -- stat error fallback
                        pass

                if should_remove:
                    try:
                        shutil.rmtree(child, ignore_errors=True)
                        recovered += 1
                        logger.info("[AttachmentRegistry] Recovered abandoned staging dir: %s", child.name)
                    except Exception as e:  # health: allow BLE001 -- abandoned directory removal failure
                        logger.debug("[AttachmentRegistry] Failed removing abandoned dir %s: %s", child, e)

        except Exception as e:  # health: allow BLE001 -- staging recovery scan failure
            logger.debug("[AttachmentRegistry] Error during abandoned staging recovery: %s", e)
        return recovered

    def _resolve_token(self, token_or_path: str) -> Optional[str]:
        if token_or_path in self._records:
            return token_or_path
        try:
            norm_p = str(Path(token_or_path).resolve())
            return self._staged_path_to_token.get(norm_p)
        except Exception:  # health: allow S110, BLE001 -- token lookup fallback
            return self._staged_path_to_token.get(token_or_path)

    def _ensure_base_staging_dir(self) -> None:
        try:
            self._base_staging_dir.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(self._base_staging_dir, 0o700)
            except (OSError, NotImplementedError):  # health: allow S110 -- ignore chmod on unsupported platforms
                pass
        except Exception as e:  # health: allow BLE001 -- staging base directory creation warning
            logger.warning("[AttachmentRegistry] Failed creating base staging dir: %s", e)

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _is_safe_source_file(src: Path) -> tuple[bool, str]:
        """Validate source file security before copying into staging."""
        try:
            if not src.exists():
                return False, f"Source file does not exist: {src}"
            if not src.is_file():
                return False, f"Source path is not a regular file: {src}"
            # Resolve symlinks to detect hidden targets
            resolved = src.resolve(strict=True)
            if not resolved.is_file():
                return False, f"Resolved source is not a regular file: {resolved}"

            # Check prohibited basenames and extensions
            lower_name = resolved.name.lower()
            if lower_name in _DENIED_SOURCE_BASENAMES or lower_name.startswith(".env"):
                return False, f"Access denied: sensitive configuration file {resolved.name}"

            ext = resolved.suffix.lower()
            if ext in _DENIED_SOURCE_EXTENSIONS:
                return False, f"Access denied: restricted credential extension {ext}"

            # Check size boundary
            size = resolved.stat().st_size
            if size <= 0:
                return False, "File is empty (0 bytes)"
            if size > MAX_ATTACHMENT_BYTES:
                return False, f"File size {size} exceeds maximum limit of {MAX_ATTACHMENT_BYTES} bytes"

            return True, ""
        except Exception as e:  # health: allow BLE001 -- source file validation failure
            return False, f"Security check exception: {e}"

    def stage_and_authorize(
        self,
        chat_id: str,
        source_path: str,
        producer: str,
        file_name: Optional[str] = None,
        caption: str = "",
        turn_id: Optional[str] = None,
    ) -> tuple[Optional[AttachmentRecord], Optional[str]]:
        """Perform pre-copy validation, private staging copy, SHA-256 calculation, and registration."""
        src = Path(source_path)
        is_safe, reason = self._is_safe_source_file(src)
        if not is_safe:
            logger.warning("[AttachmentRegistry] Denied staging for %s: %s", source_path, reason)
            return None, reason

        with self._lock:
            # Enforce turn artifact count limits
            turn_key = f"{chat_id}:{turn_id or 'default'}"
            current_tokens = self._chat_turn_index.get(turn_key, set())
            if len(current_tokens) >= MAX_ATTACHMENTS_PER_TURN:
                err = f"Maximum attachments limit ({MAX_ATTACHMENTS_PER_TURN}) reached for turn {turn_key}"
                logger.warning("[AttachmentRegistry] %s", err)
                return None, err

            # Create isolated per-process opaque staging subdirectory (proc_<pid>_<token>)
            opaque_dir_name = f"proc_{os.getpid()}_{secrets.token_hex(12)}"
            staging_subdir = self._base_staging_dir / opaque_dir_name
            try:
                staging_subdir.mkdir(parents=True, exist_ok=False)
                try:
                    os.chmod(staging_subdir, 0o700)
                except (OSError, NotImplementedError):  # health: allow S110 -- ignore chmod on unsupported platforms
                    pass
            except Exception as e:  # health: allow BLE001 -- staging subdirectory creation failure
                err = f"Failed creating isolated staging directory: {e}"
                logger.error("[AttachmentRegistry] %s", err)
                return None, err

            # Sanitize display filename
            clean_name = file_name or src.name
            clean_name = re.sub(r'[\r\n\t\\/:]', '_', clean_name).strip() or "attachment"
            staged_file = staging_subdir / clean_name

            # Physical private copy (never symlink or hardlink)
            try:
                shutil.copyfile(src, staged_file)
                try:
                    os.chmod(staged_file, 0o600)
                except (OSError, NotImplementedError):  # health: allow S110 -- ignore chmod on unsupported platforms
                    pass
                digest = self._compute_sha256(staged_file)
                staged_size = staged_file.stat().st_size
            except Exception as e:  # health: allow BLE001 -- staging copy and hashing failure
                shutil.rmtree(staging_subdir, ignore_errors=True)
                err = f"Failed copying and hashing staged artifact: {e}"
                logger.error("[AttachmentRegistry] %s", err)
                return None, err

            token = f"att_sec_{secrets.token_hex(12)}"
            record = AttachmentRecord(
                token=token,
                chat_id=chat_id,
                turn_id=turn_id,
                producer=producer,
                original_path=str(src.resolve()),
                staged_path=str(staged_file.resolve()),
                sha256=digest,
                file_size=staged_size,
                file_name=clean_name,
                caption=caption,
            )

            self._records[token] = record
            self._staged_path_to_token[record.staged_path] = token
            if turn_key not in self._chat_turn_index:
                self._chat_turn_index[turn_key] = set()
            self._chat_turn_index[turn_key].add(token)

            logger.info(
                "[AttachmentRegistry] Staged and authorized %s from %s as token %s (%d bytes, sha256=%s...)",
                clean_name, producer, token, staged_size, digest[:12],
            )
            return record, None

    def is_staged_artifact(self, path_or_token: str) -> bool:
        """Check whether a path or token corresponds to a staged, authorized artifact."""
        with self._lock:
            if path_or_token in self._records:
                return True
            try:
                norm_p = str(Path(path_or_token).resolve())
                return norm_p in self._staged_path_to_token
            except Exception:  # health: allow S110, BLE001 -- artifact path check fallback
                return False

    def get_record(self, path_or_token: str) -> Optional[AttachmentRecord]:
        """Lookup an attachment record by token or staged path."""
        with self._lock:
            record = self._records.get(path_or_token)
            if record:
                return record
            try:
                norm_p = str(Path(path_or_token).resolve())
                token = self._staged_path_to_token.get(norm_p)
                return self._records.get(token) if token else None
            except Exception:  # health: allow S110, BLE001 -- record lookup fallback
                return None

    def verify_for_delivery(
        self,
        chat_id: str,
        path: str,
        turn_id: Optional[str] = None,
    ) -> tuple[bool, Optional[str], Optional[AttachmentRecord]]:
        """Verify that a path corresponds to an authorized, unchanged staged artifact for the given chat/turn.

        Atomically claims the attachment (state -> CLAIMED) under lock to prevent concurrent redemptions.
        """
        try:
            norm_path = str(Path(path).resolve())
        except Exception:  # health: allow S110, BLE001 -- path resolve fallback
            norm_path = str(path)

        with self._lock:
            token = self._staged_path_to_token.get(norm_path)
            if not token:
                # Also check direct token lookup
                token = path if path.startswith("att_sec_") else None
            record = self._records.get(token) if token else None

            if not record:
                return False, f"Path {path} is not an authorized staged attachment", None

            # Verify destination chat isolation
            if record.chat_id != chat_id:
                return False, f"Chat mismatch: attachment authorized for {record.chat_id}, attempted {chat_id}", None

            # Verify turn binding if provided
            if turn_id and record.turn_id and record.turn_id != turn_id:
                return False, f"Turn mismatch: authorized for turn {record.turn_id}, attempted {turn_id}", None

            # Single-use check: already delivered attachments cannot be re-sent
            if record.state == AttachmentState.SUCCEEDED:
                return False, f"Attachment {token} has already been delivered (single-use enforced)", None

            # Concurrency check: in-flight items cannot be claimed again
            if record.state in (AttachmentState.CLAIMED, AttachmentState.DISPATCHING):
                return False, f"Attachment {token} is currently in-flight", None

            if record.state == AttachmentState.OUTCOME_UNKNOWN:
                return False, f"Attachment {token} delivery outcome is unknown (cannot safely retry)", None

            if record.state == AttachmentState.CLEANUP_ELIGIBLE:
                return False, f"Attachment {token} is marked cleanup-eligible (cannot deliver)", None

            if record.attempts >= record.max_attempts:
                return False, f"Attachment {token} exceeded max delivery attempts ({record.max_attempts})", None

            # Physical TOCTOU verification
            staged_p = Path(record.staged_path)
            if not staged_p.is_file():
                return False, f"Staged file disappeared before delivery: {staged_p}", None

            # Re-compute digest of current physical file to prevent content replacement
            current_digest = self._compute_sha256(staged_p)
            if current_digest != record.sha256:
                return False, f"TOCTOU violation: staged file hash {current_digest} does not match expected {record.sha256}", None

            # Atomically claim for delivery
            record.state = AttachmentState.CLAIMED
            return True, None, record

    def mark_dispatching(self, token_or_path: str) -> bool:
        """Atomically transition a CLAIMED attachment to DISPATCHING before native transmission."""
        with self._lock:
            token = self._resolve_token(token_or_path)
            record = self._records.get(token) if token else None
            if record and record.state == AttachmentState.CLAIMED:
                record.state = AttachmentState.DISPATCHING
                return True
            return False

    def record_delivery_result(self, token_or_path: str, success: bool) -> None:
        """Update delivery lifecycle state upon sidecar dispatch result."""
        with self._lock:
            token = self._resolve_token(token_or_path)
            record = self._records.get(token) if token else None
            if not record:
                return

            if success:
                record.state = AttachmentState.SUCCEEDED
                logger.info("[AttachmentRegistry] Successfully delivered attachment %s", token)
            else:
                record.attempts += 1
                if record.attempts >= record.max_attempts:
                    record.state = AttachmentState.CLEANUP_ELIGIBLE
                    logger.warning(
                        "[AttachmentRegistry] Delivery attempts exceeded (%d) for %s; marked CLEANUP_ELIGIBLE",
                        record.attempts, token,
                    )
                else:
                    record.state = AttachmentState.FAILED_PRE_DISPATCH
                    logger.warning(
                        "[AttachmentRegistry] Delivery attempt %d failed for attachment %s; marked FAILED_PRE_DISPATCH",
                        record.attempts, token,
                    )

    def mark_cancelled(self, token_or_path: str, in_native_dispatch: bool = False) -> None:
        """Transition attachment state upon asyncio cancellation."""
        with self._lock:
            token = self._resolve_token(token_or_path)
            record = self._records.get(token) if token else None
            if not record:
                return

            if in_native_dispatch:
                record.state = AttachmentState.OUTCOME_UNKNOWN
                record.outcome_unknown_at = time.monotonic()
                logger.warning(
                    "[AttachmentRegistry] Dispatch cancelled in-flight for %s; state marked OUTCOME_UNKNOWN (quarantined %.1fs)",
                    token, record.quarantine_duration_seconds,
                )
            else:
                record.state = AttachmentState.FAILED_PRE_DISPATCH
                logger.info(
                    "[AttachmentRegistry] Dispatch cancelled pre-flight for %s; state marked FAILED_PRE_DISPATCH",
                    token,
                )

    def cleanup_turn(self, chat_id: str, turn_id: Optional[str] = None, force_deferred: bool = False) -> None:
        """Clean up staged files for completed turns while safely leaving active in-flight items alone.

        OUTCOME_UNKNOWN files are quarantined for quarantine_duration_seconds to allow external
        native senders (e.g. Sidecar) to finish reading from disk without premature deletion.
        """
        turn_key = f"{chat_id}:{turn_id or 'default'}"
        now = time.monotonic()
        with self._lock:
            tokens = list(self._chat_turn_index.get(turn_key, set()))
            for tok in tokens:
                record = self._records.get(tok)
                if not record:
                    continue

                # 1. Do not delete files while a task is actively claiming or dispatching
                if record.state in (AttachmentState.CLAIMED, AttachmentState.DISPATCHING):
                    continue

                # 2. Bounded deferred cleanup for OUTCOME_UNKNOWN
                if record.state == AttachmentState.OUTCOME_UNKNOWN:
                    elapsed = now - (record.outcome_unknown_at or now)
                    if not force_deferred and elapsed < record.quarantine_duration_seconds:
                        logger.info(
                            "[AttachmentRegistry] Preserving OUTCOME_UNKNOWN attachment %s during quarantine (%.1fs < %.1fs)",
                            tok, elapsed, record.quarantine_duration_seconds,
                        )
                        continue
                    # Quarantine expired: transition to CLEANUP_ELIGIBLE
                    record.state = AttachmentState.CLEANUP_ELIGIBLE

                # 3. Clean up physical file for terminal / finished states
                staged_p = Path(record.staged_path)
                parent_dir = staged_p.parent
                try:
                    if staged_p.exists():
                        staged_p.unlink(missing_ok=True)
                    if parent_dir.exists() and parent_dir != self._base_staging_dir:
                        # Safety check: ensure parent_dir is directly inside _base_staging_dir
                        if parent_dir.resolve().parent == self._base_staging_dir.resolve():
                            shutil.rmtree(parent_dir, ignore_errors=True)
                except Exception as e:  # health: allow BLE001 -- non-fatal turn attachment cleanup error
                    logger.debug("[AttachmentRegistry] Cleanup error for %s: %s", tok, e)

                self._staged_path_to_token.pop(record.staged_path, None)
                self._records.pop(tok, None)
                if turn_key in self._chat_turn_index:
                    self._chat_turn_index[turn_key].discard(tok)

            if turn_key in self._chat_turn_index and not self._chat_turn_index[turn_key]:
                self._chat_turn_index.pop(turn_key, None)

    def cleanup_all(self) -> None:
        """Shutdown or emergency cleanup of all non-in-flight staging directories."""
        with self._lock:
            for tok, record in list(self._records.items()):
                if record.state not in (AttachmentState.CLAIMED, AttachmentState.DISPATCHING):
                    try:
                        p = Path(record.staged_path)
                        p.unlink(missing_ok=True)
                        if p.parent != self._base_staging_dir:
                            shutil.rmtree(p.parent, ignore_errors=True)
                    except Exception:  # health: allow S110, BLE001 -- emergency cleanup ignore error
                        pass
                    self._records.pop(tok, None)
                    self._staged_path_to_token.pop(record.staged_path, None)
            self._chat_turn_index.clear()


# Global module singleton instance
registry = AttachmentRegistry()

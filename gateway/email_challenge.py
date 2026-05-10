"""Persistent email challenge-response store."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from hermes_constants import get_hermes_home


logger = logging.getLogger(__name__)
DEFAULT_EMAIL_CHALLENGE_TTL_SECONDS = 900
MAX_PENDING_CHALLENGES_PER_SENDER = 25
MAX_PENDING_CHALLENGES_TOTAL = 500
MAX_PENDING_CHALLENGE_CACHED_ATTACHMENT_BYTES = 25_000_000

_CACHE_PATH_MARKERS = {
    "audio": "audio_",
    "documents": "doc_",
    "images": "img_",
    "videos": "video_",
}


def cleanup_challenge_cached_attachments(event_data: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Delete cached local attachment files referenced by challenge-owned event data.

    Returns attachment metadata for files that could not be unlinked so callers
    that own the persistent store can retain a small cleanup tombstone and retry
    later without keeping the original body/subject/event payload.
    """
    failed: list[Dict[str, Any]] = []
    attachments = event_data.get("attachments") if isinstance(event_data, dict) else None
    if not isinstance(attachments, list):
        return failed
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        path = _challenge_cached_attachment_path(attachment.get("path"))
        if path is None:
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning("[Email] Could not delete cached challenge attachment %s: %s", path, exc)
            failed.append({
                "path": str(path),
                "type": attachment.get("type"),
                "media_type": attachment.get("media_type"),
            })
    return failed


def _challenge_cached_attachment_path(raw_path: Any) -> Optional[Path]:
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute() or path.name in ("", ".", ".."):
        return None
    try:
        resolved_path = path.resolve()
    except OSError:
        return None
    from gateway.platforms import base

    cache_roots = {
        "audio": base.AUDIO_CACHE_DIR,
        "documents": base.DOCUMENT_CACHE_DIR,
        "images": base.IMAGE_CACHE_DIR,
        "videos": base.VIDEO_CACHE_DIR,
    }
    for cache_kind, prefix in _CACHE_PATH_MARKERS.items():
        if not path.name.startswith(prefix):
            continue
        try:
            root = cache_roots[cache_kind].resolve()
        except OSError:
            continue
        if resolved_path.is_relative_to(root):
            return path
    return None


class EmailChallengeStore:
    """Small JSON-backed one-time challenge store for email gateway auth."""

    def __init__(self, path: Optional[str] = None, ttl_seconds: int = DEFAULT_EMAIL_CHALLENGE_TTL_SECONDS):
        self.path = Path(path) if path else get_hermes_home() / "email_challenges.json"
        self.ttl_seconds = ttl_seconds

    def create(self, sender: str, subject: str, message_id: str, event_data: Dict[str, Any]) -> Optional[str]:
        with self._locked_data() as data:
            self._cleanup_expired_locked(data)
            sender = sender.lower()
            self._trim_sender_challenges_locked(data, sender)
            if self._pending_count_locked(data) >= MAX_PENDING_CHALLENGES_TOTAL:
                logger.warning("[Email] Email challenge store full; dropping challenge for %s", sender)
                return None
            pending_bytes = self._pending_cached_attachment_bytes_locked(data)
            event_bytes = self._event_cached_attachment_bytes(event_data)
            if pending_bytes + event_bytes > MAX_PENDING_CHALLENGE_CACHED_ATTACHMENT_BYTES:
                logger.warning("[Email] Email challenge attachment cache quota full; dropping challenge for %s", sender)
                cleanup_challenge_cached_attachments(event_data)
                return None
            code = self._new_code(data)
            data["challenges"].append({
                "code_hash": self._code_hash(code),
                "sender": sender,
                "subject": subject,
                "message_id": message_id,
                "created_at": time.time(),
                "used": False,
                "event": event_data,
            })
            return code

    def confirm(self, sender: str, code: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        with self._locked_data() as data:
            sender = sender.lower()
            now = time.time()
            found = None
            for challenge in data["challenges"]:
                if self._challenge_matches_code(challenge, code):
                    found = challenge
                    break

            if not found:
                return "not_found", None
            if found.get("used"):
                return "used", None
            if found.get("sender") != sender:
                return "sender_mismatch", None

            created_at = self._created_at(found)
            if created_at is None or now - created_at > self.ttl_seconds:
                failed_cleanup = cleanup_challenge_cached_attachments(found.get("event") or {})
                self._record_cleanup_pending_locked(data, failed_cleanup)
                found["used"] = True
                self._scrub_sensitive_challenge_locked(found)
                return "expired", None

            event = copy.deepcopy(found.get("event") or None)
            found["used"] = True
            self._scrub_sensitive_challenge_locked(found)
            return "ok", event

    def remove(self, sender: str, code: str) -> bool:
        """Remove a pending challenge and delete its cached attachments."""
        with self._locked_data() as data:
            sender = sender.lower()
            retained = []
            removed = False
            for challenge in data["challenges"]:
                if not removed and challenge.get("sender") == sender and self._challenge_matches_code(challenge, code):
                    failed_cleanup = cleanup_challenge_cached_attachments(challenge.get("event") or {})
                    self._record_cleanup_pending_locked(data, failed_cleanup)
                    removed = True
                    continue
                retained.append(challenge)
            data["challenges"] = retained
            return removed

    def cleanup_expired(self) -> None:
        if not self.path.exists():
            return
        with self._locked_data() as data:
            self._cleanup_expired_locked(data)

    def _new_code(self, data: Dict[str, Any]) -> str:
        existing_hashes = {challenge.get("code_hash") for challenge in data["challenges"]}
        legacy_codes = {challenge.get("code") for challenge in data["challenges"]}
        while True:
            code = secrets.token_urlsafe(12)
            if code not in legacy_codes and self._code_hash(code) not in existing_hashes:
                return code

    def _challenge_matches_code(self, challenge: Dict[str, Any], code: str) -> bool:
        code_hash = challenge.get("code_hash")
        if isinstance(code_hash, str):
            return secrets.compare_digest(code_hash, self._code_hash(code))
        legacy_code = challenge.get("code")
        return isinstance(legacy_code, str) and secrets.compare_digest(legacy_code, code)

    def _code_hash(self, code: str) -> str:
        return hashlib.sha256(code.encode("utf-8")).hexdigest()

    def _cleanup_expired_locked(self, data: Dict[str, Any]) -> bool:
        self._retry_cleanup_pending_locked(data)
        now = time.time()
        before = len(data["challenges"])
        retained = []
        for challenge in data["challenges"]:
            created_at = self._created_at(challenge)
            keep = not challenge.get("used") and created_at is not None and now - created_at <= self.ttl_seconds
            if keep:
                retained.append(challenge)
            else:
                failed_cleanup = cleanup_challenge_cached_attachments(challenge.get("event") or {})
                self._record_cleanup_pending_locked(data, failed_cleanup)
                self._scrub_sensitive_challenge_locked(challenge)
        data["challenges"] = retained
        return len(data["challenges"]) != before

    def _record_cleanup_pending_locked(self, data: Dict[str, Any], attachments: list[Dict[str, Any]]) -> None:
        valid = []
        for attachment in attachments:
            if not isinstance(attachment, dict):
                continue
            path = _challenge_cached_attachment_path(attachment.get("path"))
            if path is None:
                continue
            valid.append({
                "path": str(path),
                "type": attachment.get("type"),
                "media_type": attachment.get("media_type"),
            })
        if not valid:
            return
        data.setdefault("cleanup_pending", []).append({"attachments": valid})

    def _retry_cleanup_pending_locked(self, data: Dict[str, Any]) -> None:
        pending = data.get("cleanup_pending")
        if not isinstance(pending, list) or not pending:
            data.pop("cleanup_pending", None)
            return
        retained = []
        for item in pending:
            if not isinstance(item, dict):
                continue
            failed_cleanup = cleanup_challenge_cached_attachments(item)
            if failed_cleanup:
                retained.append({"attachments": failed_cleanup})
        if retained:
            data["cleanup_pending"] = retained
        else:
            data.pop("cleanup_pending", None)

    def _scrub_sensitive_challenge_locked(self, challenge: Dict[str, Any]) -> None:
        challenge.pop("event", None)
        challenge.pop("subject", None)
        challenge.pop("message_id", None)
        challenge.pop("body", None)
        challenge.pop("attachments", None)

    def _trim_sender_challenges_locked(self, data: Dict[str, Any], sender: str) -> None:
        pending_for_sender = [
            challenge
            for challenge in data["challenges"]
            if challenge.get("sender") == sender and not challenge.get("used")
        ]
        overflow = len(pending_for_sender) - MAX_PENDING_CHALLENGES_PER_SENDER + 1
        if overflow <= 0:
            return
        pending_for_sender.sort(key=lambda challenge: self._created_at(challenge) or 0)
        to_remove = {id(challenge) for challenge in pending_for_sender[:overflow]}
        for challenge in pending_for_sender[:overflow]:
            failed_cleanup = cleanup_challenge_cached_attachments(challenge.get("event") or {})
            self._record_cleanup_pending_locked(data, failed_cleanup)
        data["challenges"] = [challenge for challenge in data["challenges"] if id(challenge) not in to_remove]

    def _pending_count_locked(self, data: Dict[str, Any]) -> int:
        return sum(1 for challenge in data["challenges"] if not challenge.get("used"))

    def _pending_cached_attachment_bytes_locked(self, data: Dict[str, Any]) -> int:
        return sum(
            self._event_cached_attachment_bytes(challenge.get("event") or {})
            for challenge in data["challenges"]
            if not challenge.get("used")
        )

    def _event_cached_attachment_bytes(self, event_data: Dict[str, Any]) -> int:
        attachments = event_data.get("attachments") if isinstance(event_data, dict) else None
        if not isinstance(attachments, list):
            return 0
        total = 0
        for attachment in attachments:
            if not isinstance(attachment, dict):
                continue
            path = _challenge_cached_attachment_path(attachment.get("path"))
            if path is None:
                continue
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total

    def _created_at(self, challenge: Dict[str, Any]) -> Optional[float]:
        if "created_at" not in challenge:
            logger.warning("[Email] Ignoring email challenge without timestamp in %s", self.path)
            return None
        try:
            return float(challenge.get("created_at"))
        except (TypeError, ValueError):
            logger.warning("[Email] Ignoring malformed email challenge timestamp in %s", self.path)
            return None

    @contextlib.contextmanager
    def _locked_data(self) -> Iterator[Dict[str, Any]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(f".{self.path.name}.lock")
        with lock_path.open("a+") as lock_file:
            self._lock_file(lock_file)
            try:
                self._loaded_store_was_sanitized = False
                data = self._load()
                self._harden_permissions()
                original = self._serialized(data)
                yield data
                if self._loaded_store_was_sanitized or self._serialized(data) != original:
                    self._save(data)
            finally:
                self._loaded_store_was_sanitized = False
                self._unlock_file(lock_file)

    def _serialized(self, data: Dict[str, Any]) -> str:
        return json.dumps(data, indent=2, sort_keys=True)

    def _load(self) -> Dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"challenges": []}
        except json.JSONDecodeError:
            logger.warning("[Email] Ignoring corrupt email challenge store: %s", self.path)
            self._discard_invalid_store()
            return {"challenges": []}
        except OSError as exc:
            logger.warning("[Email] Could not read email challenge store %s: %s", self.path, exc)
            raise
        if not isinstance(raw, dict) or not isinstance(raw.get("challenges"), list):
            logger.warning("[Email] Ignoring invalid email challenge store shape: %s", self.path)
            self._discard_invalid_store()
            return {"challenges": []}
        challenges, challenges_sanitized = self._sanitize_loaded_challenges(raw["challenges"])
        data = {"challenges": challenges}
        cleanup_pending = raw.get("cleanup_pending")
        if isinstance(cleanup_pending, list):
            data["cleanup_pending"] = cleanup_pending
        if challenges_sanitized or data != raw:
            logger.warning("[Email] Sanitizing invalid email challenge store entries: %s", self.path)
            self._loaded_store_was_sanitized = True
        return data

    def _sanitize_loaded_challenges(self, raw_challenges: list[Any]) -> tuple[list[Dict[str, Any]], bool]:
        challenges: list[Dict[str, Any]] = []
        sanitized = False
        for raw_challenge in raw_challenges:
            if not isinstance(raw_challenge, dict):
                sanitized = True
                continue
            challenge = self._sanitize_loaded_challenge(raw_challenge)
            if challenge is None:
                sanitized = True
                continue
            if challenge != raw_challenge:
                sanitized = True
            challenges.append(challenge)
        return challenges, sanitized

    def _sanitize_loaded_challenge(self, challenge: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        code_hash = challenge.get("code_hash")
        legacy_code = challenge.get("code")
        has_code_hash = isinstance(code_hash, str) and len(code_hash) == 64 and all(
            char in "0123456789abcdefABCDEF" for char in code_hash
        )
        has_legacy_code = isinstance(legacy_code, str) and bool(legacy_code)
        sender = challenge.get("sender")
        used = challenge.get("used")
        if not (has_code_hash or has_legacy_code) or not isinstance(sender, str) or not sender:
            return None
        if used not in (True, False):
            return None
        has_valid_created_at = self._created_at(challenge) is not None
        if not has_valid_created_at:
            return None

        sanitized: Dict[str, Any] = {}
        if has_code_hash:
            sanitized["code_hash"] = code_hash
        if has_legacy_code:
            sanitized["code"] = legacy_code
        sanitized["sender"] = sender
        sanitized["created_at"] = challenge["created_at"]
        sanitized["used"] = used

        if not used:
            subject = challenge.get("subject")
            message_id = challenge.get("message_id")
            event = challenge.get("event")
            if not isinstance(subject, str) or not isinstance(message_id, str) or not isinstance(event, dict):
                return None
            sanitized["subject"] = subject
            sanitized["message_id"] = message_id
            sanitized["event"] = event

        return sanitized

    def _discard_invalid_store(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.warning("[Email] Could not remove invalid email challenge store %s: %s", self.path, exc)

    def _harden_permissions(self) -> None:
        if os.name == "nt" or not self.path.exists():
            return
        try:
            os.chmod(self.path, 0o600)
        except OSError as exc:
            logger.warning("[Email] Could not harden email challenge store permissions %s: %s", self.path, exc)

    def _save(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{secrets.token_hex(4)}.tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(tmp, flags, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
        except Exception:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        tmp.replace(self.path)
        os.chmod(self.path, 0o600)

    def _lock_file(self, lock_file: Any) -> None:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

    def _unlock_file(self, lock_file: Any) -> None:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

"""
DM Pairing System

Code-based approval flow for authorizing new users on messaging platforms.
Instead of static allowlists with user IDs, unknown users receive a one-time
pairing code that the bot owner approves via the CLI.

Security features (based on OWASP + NIST SP 800-63-4 guidance):
  - 8-char codes from 32-char unambiguous alphabet (no 0/O/1/I)
  - Cryptographic randomness via secrets.choice()
  - 1-hour code expiry
  - Max 3 pending codes per platform
  - Rate limiting: 1 request per user per 10 minutes
  - Lockout after 5 failed approval attempts (1 hour)
  - File permissions: chmod 0600 on all data files
  - Codes are never logged to stdout

Storage: ~/.hermes/pairing/
"""

import hashlib
import json
import os
import secrets
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from gateway.whatsapp_identity import (
    expand_whatsapp_aliases,
    normalize_whatsapp_identifier,
)
from hermes_constants import get_hermes_dir
from utils import atomic_replace


# Unambiguous alphabet -- excludes 0/O, 1/I to prevent confusion
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
CODE_LENGTH = 8

# Timing constants
CODE_TTL_SECONDS = 3600             # Codes expire after 1 hour
RATE_LIMIT_SECONDS = 600            # 1 request per user per 10 minutes
LOCKOUT_SECONDS = 3600              # Lockout duration after too many failures

# Limits
MAX_PENDING_PER_PLATFORM = 3        # Max pending codes per platform
MAX_FAILED_ATTEMPTS = 5             # Failed approvals before lockout

PAIRING_DIR = get_hermes_dir("platforms/pairing", "pairing")


def _secure_write(path: Path, data: str) -> None:
    """Write data to file with restrictive permissions (owner read/write only).

    Uses a temp-file + atomic rename so readers always see either the old
    complete file or the new one — never a partial write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        try:
            os.chmod(tmp_path, 0o600)
        except OSError:
            pass
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class PairingStore:
    """
    Manages pairing codes and approved user lists.

    Data files per platform:
      - {platform}-pending.json   : pending pairing requests
      - {platform}-approved.json  : approved (paired) users
      - _rate_limits.json         : rate limit tracking

    When constructed with ``profile="<name>"``, storage lives under
    ``<HERMES_HOME>/profiles/<name>/pairing/`` (per-profile, used by
    multiplexing gateways so each profile has its own whitelist).
    Without a profile, storage is the global ``<HERMES_HOME>/pairing/``
    directory (backward-compat for the ``hermes pairing`` CLI).
    """

    def __init__(self, profile: Optional[str] = None):
        # Resolve storage directory lazily — tests use a temp HERMES_HOME
        # and PairingStore may be constructed before the env is set.
        if profile:
            from hermes_constants import get_hermes_home
            self._dir = get_hermes_home() / "profiles" / profile / "pairing"
        else:
            self._dir = PAIRING_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        # Protects all read-modify-write cycles. The gateway runs multiple
        # platform adapters concurrently in threads sharing one PairingStore.
        self._lock = threading.RLock()
        self._profile = profile  # for diagnostics / log lines

    @property
    def profile(self) -> Optional[str]:
        """Profile name this store is scoped to, or None for the global store."""
        return self._profile

    def _pending_path(self, platform: str) -> Path:
        return self._dir / f"{platform}-pending.json"

    def _approved_path(self, platform: str) -> Path:
        return self._dir / f"{platform}-approved.json"

    def _rate_limit_path(self) -> Path:
        return self._dir / "_rate_limits.json"

    def _load_json(self, path: Path) -> dict:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_json(self, path: Path, data: dict) -> None:
        _secure_write(path, json.dumps(data, indent=2, ensure_ascii=False))

    def _normalize_user_id(self, platform: str, user_id: str) -> str:
        """Normalize platform-specific user IDs before persisting them."""
        raw_user_id = str(user_id or "").strip()
        if platform == "whatsapp":
            return normalize_whatsapp_identifier(raw_user_id) or raw_user_id
        return raw_user_id

    def _user_id_aliases(self, platform: str, user_id: str) -> set[str]:
        """Return all known equivalent user IDs for auth/rate-limit checks."""
        raw_user_id = str(user_id or "").strip()
        if not raw_user_id:
            return set()

        aliases = {raw_user_id, self._normalize_user_id(platform, raw_user_id)}
        if platform == "whatsapp":
            aliases.update(expand_whatsapp_aliases(raw_user_id))
        aliases.discard("")
        return aliases

    def _user_ids_match(self, platform: str, left: str, right: str) -> bool:
        """Return True when two user IDs represent the same principal."""
        left_aliases = self._user_id_aliases(platform, left)
        right_aliases = self._user_id_aliases(platform, right)
        return bool(left_aliases and right_aliases and (left_aliases & right_aliases))

    # ----- Approved users -----

    def is_approved(self, platform: str, user_id: str) -> bool:
        """Check if a user is approved (paired) on a platform."""
        approved = self._load_json(self._approved_path(platform))
        for approved_user_id in approved:
            if self._user_ids_match(platform, approved_user_id, user_id):
                return True
        return False

    def list_approved(self, platform: str = None) -> list:
        """List approved users, optionally filtered by platform."""
        results = []
        platforms = [platform] if platform else self._all_platforms("approved")
        for p in platforms:
            approved = self._load_json(self._approved_path(p))
            for uid, info in approved.items():
                results.append({"platform": p, "user_id": uid, **info})
        return results

    def _approve_user(self, platform: str, user_id: str, user_name: str = "") -> None:
        """Add a user to the approved list. Must be called under self._lock."""
        approved = self._load_json(self._approved_path(platform))
        normalized_user_id = self._normalize_user_id(platform, user_id)
        duplicate_ids = [
            approved_user_id
            for approved_user_id in approved
            if self._user_ids_match(platform, approved_user_id, normalized_user_id)
        ]
        for approved_user_id in duplicate_ids:
            del approved[approved_user_id]

        approved[normalized_user_id] = {
            "user_name": user_name,
            "approved_at": time.time(),
        }
        self._save_json(self._approved_path(platform), approved)

    def revoke(self, platform: str, user_id: str) -> bool:
        """Remove a user from the approved list. Returns True if found."""
        path = self._approved_path(platform)
        with self._lock:
            approved = self._load_json(path)
            matching_ids = [
                approved_user_id
                for approved_user_id in approved
                if self._user_ids_match(platform, approved_user_id, user_id)
            ]
            if matching_ids:
                for approved_user_id in matching_ids:
                    del approved[approved_user_id]
                self._save_json(path, approved)
                return True
        return False

    # ----- Pending codes -----

    @staticmethod
    def _hash_code(code: str, salt: bytes) -> str:
        """Hash a pairing code with the given salt using SHA-256."""
        return hashlib.sha256(salt + code.encode("utf-8")).hexdigest()

    def generate_code(
        self,
        platform: str,
        user_id: str,
        user_name: str = "",
    ) -> Optional[str]:
        """Generate a one-time pairing code for a user. Returns None on rate limit."""
        with self._lock:
            if self._is_locked_out(platform):
                return None
            if self._is_rate_limited(platform, user_id):
                return None
            pending = self._load_json(self._pending_path(platform))
            pending_codes = pending.get("codes", [])
            now = time.time()
            pending_codes = [c for c in pending_codes if c["expires_at"] > now]
            if len(pending_codes) >= MAX_PENDING_PER_PLATFORM:
                return None
            code = "".join(secrets.choice(ALPHABET) for _ in range(CODE_LENGTH))
            salt = secrets.token_bytes(16)
            normalized = self._normalize_user_id(platform, user_id)
            pending_codes.append({
                "code_hash": self._hash_code(code, salt),
                "salt_hex": salt.hex(),
                "user_id": normalized,
                "user_name": user_name,
                "created_at": now,
                "expires_at": now + CODE_TTL_SECONDS,
            })
            pending["codes"] = pending_codes
            self._save_json(self._pending_path(platform), pending)
        return code

    def approve_code(self, platform: str, code: str) -> Optional[Dict[str, str]]:
        """Validate a code, mark the user approved, and return their info."""
        with self._lock:
            if self._is_locked_out(platform):
                return None
            code = code.upper().strip()
            pending = self._load_json(self._pending_path(platform))
            now = time.time()
            for entry in pending.get("codes", []):
                if entry["expires_at"] <= now:
                    continue
                salt = bytes.fromhex(entry["salt_hex"])
                if self._hash_code(code, salt) == entry["code_hash"]:
                    pending["codes"] = [
                        c for c in pending["codes"] if c is not entry
                    ]
                    self._save_json(self._pending_path(platform), pending)
                    self._approve_user(platform, entry["user_id"], entry.get("user_name", ""))
                    self._reset_failures(platform)
                    return {"user_id": entry["user_id"], "user_name": entry.get("user_name", "")}
            self._record_failure(platform)
        return None

    def _is_locked_out(self, platform: str) -> bool:
        rl = self._load_json(self._rate_limit_path())
        until = rl.get("_lockout_until", {}).get(platform, 0)
        return until > time.time()

    def _is_rate_limited(self, platform: str, user_id: str) -> bool:
        rl = self._load_json(self._rate_limit_path())
        now = time.time()
        for uid, last_seen in list(rl.items()):
            if uid.startswith("_"):
                continue
            if ":" in uid and uid.split(":", 1)[0] == platform:
                if self._user_ids_match(platform, uid.split(":", 1)[1], user_id):
                    if now - float(last_seen) < RATE_LIMIT_SECONDS:
                        return True
        return False

    def _record_rate_limit(self, platform: str, user_id: str) -> None:
        rl = self._load_json(self._rate_limit_path())
        key = f"{platform}:{self._normalize_user_id(platform, user_id)}"
        rl[key] = time.time()
        self._save_json(self._rate_limit_path(), rl)

    def _record_failure(self, platform: str) -> None:
        rl = self._load_json(self._rate_limit_path())
        key = f"_failures:{platform}"
        rl[key] = rl.get(key, 0) + 1
        if rl[key] >= MAX_FAILED_ATTEMPTS:
            until_key = "_lockout_until"
            rl.setdefault(until_key, {})[platform] = time.time() + LOCKOUT_SECONDS
            rl[key] = 0
        self._save_json(self._rate_limit_path(), rl)

    def _reset_failures(self, platform: str) -> None:
        rl = self._load_json(self._rate_limit_path())
        key = f"_failures:{platform}"
        if key in rl:
            del rl[key]
            self._save_json(self._rate_limit_path(), rl)

    def clear_pending(self, platform: str = None) -> int:
        """Remove expired (or all) pending codes. Returns count removed."""
        with self._lock:
            count = 0
            platforms = [platform] if platform else self._all_platforms("pending")
            for p in platforms:
                pending = self._load_json(self._pending_path(p))
                if not pending:
                    continue
                if platform is None and "expired_only" not in pending:
                    count += len(pending.get("codes", []))
                    self._save_json(self._pending_path(p), {"codes": []})
                else:
                    now = time.time()
                    before = len(pending.get("codes", []))
                    pending["codes"] = [
                        c for c in pending.get("codes", []) if c["expires_at"] > now
                    ]
                    count += before - len(pending["codes"])
                    self._save_json(self._pending_path(p), pending)
        return count

    def list_pending(self, platform: str = None) -> list:
        """List pending pairing requests (active only)."""
        results = []
        platforms = [platform] if platform else self._all_platforms("pending")
        for p in platforms:
            pending = self._load_json(self._pending_path(p))
            now = time.time()
            for entry in pending.get("codes", []):
                if entry["expires_at"] <= now:
                    continue
                results.append({
                    "platform": p,
                    "user_id": entry["user_id"],
                    "user_name": entry.get("user_name", ""),
                    "code_age_minutes": int((now - entry["created_at"]) / 60),
                })
        return results

    def _all_platforms(self, kind: str) -> list:
        """Return all platform names that have a pairing file of the given kind."""
        if not PAIRING_DIR.exists():
            return []
        suffix = "-pending.json" if kind == "pending" else "-approved.json"
        return [p.stem.replace("-pending", "").replace("-approved", "")
                for p in PAIRING_DIR.glob(f"*{suffix}")]


# Backwards-compat shim for tests/CLI that imported the helper at module level.
def _is_authorized_for_pairing(_platform: str) -> bool:  # pragma: no cover
    """No-op shim — owner-only operations are now CLI-gated, not import-gated."""
    return True

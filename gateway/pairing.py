"""
DM Pairing System

Code-based approval flow for authorizing new users on messaging platforms.
Instead of static allowlists with user IDs, unknown users receive a one-time
pairing code that the bot owner approves via the CLI.

Two pairing flows are supported:

  Reactive (default):
    Unknown user messages the bot → bot generates a code tied to that user →
    owner runs ``hermes pairing approve <platform> <code>`` to approve.

  Proactive (invite codes):
    Owner runs ``hermes pairing generate <platform>`` → receives a single-use
    invite code → shares it out-of-band → recipient sends the code as their
    first message → auto-approved.

Security features (based on OWASP + NIST SP 800-63-4 guidance):
  - 8-char codes from 32-char unambiguous alphabet (no 0/O/1/I)
  - Cryptographic randomness via secrets.choice()
  - 1-hour code expiry (reactive), 24-hour expiry (invite codes)
  - Max 3 pending codes per platform (reactive), 5 invite codes
  - Rate limiting: 1 request per user per 10 minutes
  - Lockout after 5 failed approval attempts (1 hour)
  - File permissions: chmod 0600 on all data files
  - Codes are never logged to stdout

Storage: ~/.hermes/pairing/
"""

import json
import os
import secrets
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_dir


# Unambiguous alphabet -- excludes 0/O, 1/I to prevent confusion
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
CODE_LENGTH = 8

# Timing constants
CODE_TTL_SECONDS = 3600             # Codes expire after 1 hour
RATE_LIMIT_SECONDS = 600            # 1 request per user per 10 minutes
LOCKOUT_SECONDS = 3600              # Lockout duration after too many failures

# Limits
MAX_PENDING_PER_PLATFORM = 3        # Max pending codes per platform
MAX_INVITE_CODES_PER_PLATFORM = 5   # Max invite codes per platform
MAX_FAILED_ATTEMPTS = 5             # Failed approvals before lockout

# Invite code TTL — longer than reactive codes since the owner needs time
# to share the code out-of-band.
INVITE_TTL_SECONDS = 86400          # 24 hours

PAIRING_DIR = get_hermes_dir("platforms/pairing", "pairing")


def _secure_write(path: Path, data: str) -> None:
    """Write data to file with restrictive permissions (owner read/write only).

    Uses a temp-file + atomic rename so readers always see either the old
    complete file or the new one — never a partial write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass  # Windows doesn't support chmod the same way
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class PairingStore:
    """
    Manages pairing codes and approved user lists.

    Data files per platform:
      - {platform}-pending.json   : pending pairing requests (reactive flow)
      - {platform}-invites.json   : invite codes (proactive flow)
      - {platform}-approved.json  : approved (paired) users
      - _rate_limits.json         : rate limit tracking
    """

    def __init__(self):
        PAIRING_DIR.mkdir(parents=True, exist_ok=True)
        # Protects all read-modify-write cycles. The gateway runs multiple
        # platform adapters concurrently in threads sharing one PairingStore.
        self._lock = threading.RLock()

    def _pending_path(self, platform: str) -> Path:
        return PAIRING_DIR / f"{platform}-pending.json"

    def _invites_path(self, platform: str) -> Path:
        return PAIRING_DIR / f"{platform}-invites.json"

    def _approved_path(self, platform: str) -> Path:
        return PAIRING_DIR / f"{platform}-approved.json"

    def _rate_limit_path(self) -> Path:
        return PAIRING_DIR / "_rate_limits.json"

    def _load_json(self, path: Path) -> dict:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_json(self, path: Path, data: dict) -> None:
        _secure_write(path, json.dumps(data, indent=2, ensure_ascii=False))

    # ----- Approved users -----

    def is_approved(self, platform: str, user_id: str) -> bool:
        """Check if a user is approved (paired) on a platform."""
        approved = self._load_json(self._approved_path(platform))
        return user_id in approved

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
        approved[user_id] = {
            "user_name": user_name,
            "approved_at": time.time(),
        }
        self._save_json(self._approved_path(platform), approved)

    def revoke(self, platform: str, user_id: str) -> bool:
        """Remove a user from the approved list. Returns True if found."""
        path = self._approved_path(platform)
        with self._lock:
            approved = self._load_json(path)
            if user_id in approved:
                del approved[user_id]
                self._save_json(path, approved)
                return True
        return False

    # ----- Pending codes -----

    def generate_code(
        self, platform: str, user_id: str, user_name: str = ""
    ) -> Optional[str]:
        """
        Generate a pairing code for a new user.

        Returns the code string, or None if:
          - User is rate-limited (too recent request)
          - Max pending codes reached for this platform
          - User/platform is in lockout due to failed attempts
        """
        with self._lock:
            self._cleanup_expired(platform)

            # Check lockout
            if self._is_locked_out(platform):
                return None

            # Check rate limit for this specific user
            if self._is_rate_limited(platform, user_id):
                return None

            # Check max pending
            pending = self._load_json(self._pending_path(platform))
            if len(pending) >= MAX_PENDING_PER_PLATFORM:
                return None

            # Generate cryptographically random code
            code = "".join(secrets.choice(ALPHABET) for _ in range(CODE_LENGTH))

            # Store pending request
            pending[code] = {
                "user_id": user_id,
                "user_name": user_name,
                "created_at": time.time(),
            }
            self._save_json(self._pending_path(platform), pending)

            # Record rate limit
            self._record_rate_limit(platform, user_id)

            return code

    def approve_code(self, platform: str, code: str) -> Optional[dict]:
        """
        Approve a pairing code. Adds the user to the approved list.

        Returns {user_id, user_name} on success, None if code is invalid/expired.
        """
        with self._lock:
            self._cleanup_expired(platform)
            code = code.upper().strip()

            pending = self._load_json(self._pending_path(platform))
            if code not in pending:
                self._record_failed_attempt(platform)
                return None

            entry = pending.pop(code)
            self._save_json(self._pending_path(platform), pending)

            # Add to approved list
            self._approve_user(platform, entry["user_id"], entry.get("user_name", ""))

            return {
                "user_id": entry["user_id"],
                "user_name": entry.get("user_name", ""),
            }

    def list_pending(self, platform: str = None) -> list:
        """List pending pairing requests, optionally filtered by platform."""
        results = []
        platforms = [platform] if platform else self._all_platforms("pending")
        for p in platforms:
            self._cleanup_expired(p)
            pending = self._load_json(self._pending_path(p))
            for code, info in pending.items():
                age_min = int((time.time() - info["created_at"]) / 60)
                results.append({
                    "platform": p,
                    "code": code,
                    "user_id": info["user_id"],
                    "user_name": info.get("user_name", ""),
                    "age_minutes": age_min,
                })
        return results

    def clear_pending(self, platform: str = None) -> int:
        """Clear all pending requests. Returns count removed."""
        with self._lock:
            count = 0
            platforms = [platform] if platform else self._all_platforms("pending")
            for p in platforms:
                pending = self._load_json(self._pending_path(p))
                count += len(pending)
                self._save_json(self._pending_path(p), {})
        return count

    # ----- Invite codes (proactive flow) -----

    def generate_invite_code(self, platform: str) -> Optional[str]:
        """
        Generate a single-use invite code for a platform.

        The bot owner shares this code out-of-band.  When an unauthorized user
        sends the code as their first message, they are auto-approved.

        Returns the code string, or None if the invite limit is reached.
        """
        with self._lock:
            self._cleanup_expired_invites(platform)

            invites = self._load_json(self._invites_path(platform))
            if len(invites) >= MAX_INVITE_CODES_PER_PLATFORM:
                return None

            code = "".join(secrets.choice(ALPHABET) for _ in range(CODE_LENGTH))
            invites[code] = {"created_at": time.time()}
            self._save_json(self._invites_path(platform), invites)
            return code

    def claim_invite_code(
        self, platform: str, code: str, user_id: str, user_name: str = ""
    ) -> bool:
        """
        Try to claim an invite code.  If valid, the user is approved and the
        code is consumed (single-use).

        Returns True if the code was valid and the user was approved.
        """
        with self._lock:
            self._cleanup_expired_invites(platform)
            code = code.upper().strip()

            invites = self._load_json(self._invites_path(platform))
            if code not in invites:
                return False

            # Consume the code
            del invites[code]
            self._save_json(self._invites_path(platform), invites)

            # Approve the user
            self._approve_user(platform, user_id, user_name)
            return True

    def list_invites(self, platform: str = None) -> list:
        """List outstanding invite codes, optionally filtered by platform."""
        results = []
        platforms = [platform] if platform else self._all_platforms("invites")
        for p in platforms:
            self._cleanup_expired_invites(p)
            invites = self._load_json(self._invites_path(p))
            for code, info in invites.items():
                age_min = int((time.time() - info["created_at"]) / 60)
                ttl_min = max(0, int((INVITE_TTL_SECONDS - (time.time() - info["created_at"])) / 60))
                results.append({
                    "platform": p,
                    "code": code,
                    "age_minutes": age_min,
                    "ttl_minutes": ttl_min,
                })
        return results

    def clear_invites(self, platform: str = None) -> int:
        """Clear all invite codes. Returns count removed."""
        with self._lock:
            count = 0
            platforms = [platform] if platform else self._all_platforms("invites")
            for p in platforms:
                invites = self._load_json(self._invites_path(p))
                count += len(invites)
                self._save_json(self._invites_path(p), {})
        return count

    def _cleanup_expired_invites(self, platform: str) -> None:
        """Remove expired invite codes."""
        path = self._invites_path(platform)
        invites = self._load_json(path)
        now = time.time()
        expired = [
            code for code, info in invites.items()
            if (now - info["created_at"]) > INVITE_TTL_SECONDS
        ]
        if expired:
            for code in expired:
                del invites[code]
            self._save_json(path, invites)

    # ----- Rate limiting and lockout -----

    def _is_rate_limited(self, platform: str, user_id: str) -> bool:
        """Check if a user has requested a code too recently."""
        limits = self._load_json(self._rate_limit_path())
        key = f"{platform}:{user_id}"
        last_request = limits.get(key, 0)
        return (time.time() - last_request) < RATE_LIMIT_SECONDS

    def _record_rate_limit(self, platform: str, user_id: str) -> None:
        """Record the time of a pairing request for rate limiting."""
        limits = self._load_json(self._rate_limit_path())
        key = f"{platform}:{user_id}"
        limits[key] = time.time()
        self._save_json(self._rate_limit_path(), limits)

    def _is_locked_out(self, platform: str) -> bool:
        """Check if a platform is in lockout due to failed approval attempts."""
        limits = self._load_json(self._rate_limit_path())
        lockout_key = f"_lockout:{platform}"
        lockout_until = limits.get(lockout_key, 0)
        return time.time() < lockout_until

    def _record_failed_attempt(self, platform: str) -> None:
        """Record a failed approval attempt. Triggers lockout after MAX_FAILED_ATTEMPTS."""
        limits = self._load_json(self._rate_limit_path())
        fail_key = f"_failures:{platform}"
        fails = limits.get(fail_key, 0) + 1
        limits[fail_key] = fails
        if fails >= MAX_FAILED_ATTEMPTS:
            lockout_key = f"_lockout:{platform}"
            limits[lockout_key] = time.time() + LOCKOUT_SECONDS
            limits[fail_key] = 0  # Reset counter
            print(f"[pairing] Platform {platform} locked out for {LOCKOUT_SECONDS}s "
                  f"after {MAX_FAILED_ATTEMPTS} failed attempts", flush=True)
        self._save_json(self._rate_limit_path(), limits)

    # ----- Cleanup -----

    def _cleanup_expired(self, platform: str) -> None:
        """Remove expired pending codes."""
        path = self._pending_path(platform)
        pending = self._load_json(path)
        now = time.time()
        expired = [
            code for code, info in pending.items()
            if (now - info["created_at"]) > CODE_TTL_SECONDS
        ]
        if expired:
            for code in expired:
                del pending[code]
            self._save_json(path, pending)

    def _all_platforms(self, suffix: str) -> list:
        """List all platforms that have data files of a given suffix."""
        platforms = []
        for f in PAIRING_DIR.iterdir():
            if f.name.endswith(f"-{suffix}.json"):
                platform = f.name.replace(f"-{suffix}.json", "")
                if not platform.startswith("_"):
                    platforms.append(platform)
        return platforms

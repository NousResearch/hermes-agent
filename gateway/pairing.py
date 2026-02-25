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

import json
import os
import secrets
import time
from pathlib import Path
from typing import Optional

# Unambiguous alphabet -- excludes 0/O, 1/I to prevent confusion
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
CODE_LENGTH = 8

# Timing constants
CODE_TTL_SECONDS = 3600  # Codes expire after 1 hour
RATE_LIMIT_SECONDS = 600  # 1 request per user per 10 minutes
LOCKOUT_SECONDS = 3600  # Lockout duration after too many failures

# Limits
MAX_PENDING_PER_PLATFORM = 3  # Max pending codes per platform
MAX_FAILED_ATTEMPTS = 5  # Failed approvals before lockout

PAIRING_DIR = Path(os.path.expanduser("~/.hermes/pairing"))


def _secure_write(path: Path, data: str) -> None:
    """Write data to file with restrictive permissions (owner read/write only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass  # Windows doesn't support chmod the same way


class PairingStore:
    """
    Manages pairing codes and approved user lists.

    Data files per platform:
      - {platform}-pending.json   : pending pairing requests
      - {platform}-approved.json  : approved (paired) users
      - _rate_limits.json         : rate limit tracking
    """

    def __init__(self):
        PAIRING_DIR.mkdir(parents=True, exist_ok=True)

    def _pending_path(self, platform: str) -> Path:
        return PAIRING_DIR / f"{platform}-pending.json"

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
        """Add a user to the approved list."""
        approved = self._load_json(self._approved_path(platform))
        approved[user_id] = {
            "user_name": user_name,
            "approved_at": time.time(),
        }
        self._save_json(self._approved_path(platform), approved)

    def revoke(self, platform: str, user_id: str) -> bool:
        """Remove a user from the approved list. Returns True if found."""
        path = self._approved_path(platform)
        approved = self._load_json(path)
        if user_id in approved:
            del approved[user_id]
            self._save_json(path, approved)
            return True
        return False

    # ----- Pending codes -----

    def generate_code(self, platform: str, user_id: str, user_name: str = "") -> Optional[str]:
        """
        Generate a pairing code for a new user.

        Returns the code string, or None if:
          - User is rate-limited (too recent request)
          - Max pending codes reached for this platform
          - User/platform is in lockout due to failed attempts
        """
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
                results.append(
                    {
                        "platform": p,
                        "code": code,
                        "user_id": info["user_id"],
                        "user_name": info.get("user_name", ""),
                        "age_minutes": age_min,
                    }
                )
        return results

    def clear_pending(self, platform: str = None) -> int:
        """Clear all pending requests. Returns count removed."""
        count = 0
        platforms = [platform] if platform else self._all_platforms("pending")
        for p in platforms:
            pending = self._load_json(self._pending_path(p))
            count += len(pending)
            self._save_json(self._pending_path(p), {})
        return count

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
            print(
                f"[pairing] Platform {platform} locked out for {LOCKOUT_SECONDS}s "
                f"after {MAX_FAILED_ATTEMPTS} failed attempts",
                flush=True,
            )
        self._save_json(self._rate_limit_path(), limits)

    # ----- Cleanup -----

    def _cleanup_expired(self, platform: str) -> None:
        """Remove expired pending codes."""
        path = self._pending_path(platform)
        pending = self._load_json(path)
        now = time.time()
        expired = [code for code, info in pending.items() if (now - info["created_at"]) > CODE_TTL_SECONDS]
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

"""
DM Pairing System

Most platforms use a code-based approval flow: unknown users receive a one-time
pairing code that the bot owner approves via the CLI.

Kasia is different. Unknown Kasia users are queued as pending contact requests
keyed by canonical address so Hermes can avoid auto-spending on first contact
while still showing both the `.kas` identity and the full `kaspa:` address.

Storage: ~/.hermes/pairing/
"""

import json
import os
import secrets
import time
from pathlib import Path
from typing import Optional

from gateway.kasia_config import normalized_kasia_address_variants
from gateway.kasia_identity import (
    normalize_kasia_kns_name,
    resolve_kasia_identity,
)
from hermes_cli.config import get_hermes_home


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

PAIRING_DIR = get_hermes_home() / "pairing"
_KASIA_PLATFORM = "kasia"


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

    def _normalize_platform(self, platform: str) -> str:
        return str(platform or "").strip().lower()

    def _is_kasia_platform(self, platform: str) -> bool:
        return self._normalize_platform(platform) == _KASIA_PLATFORM

    def _find_kasia_entry(self, entries: dict, target: str) -> tuple[Optional[str], Optional[dict]]:
        normalized_kns = normalize_kasia_kns_name(target)
        target_variants = normalized_kasia_address_variants(target)
        for key, info in entries.items():
            canonical_address = info.get("canonical_address") or info.get("user_id") or key
            entry_variants = normalized_kasia_address_variants(canonical_address)
            entry_kns = normalize_kasia_kns_name(info.get("kns_name"))
            if normalized_kns and entry_kns == normalized_kns:
                return key, info
            if target_variants and entry_variants and target_variants & entry_variants:
                return key, info
        return None, None

    def _build_kasia_entry(
        self,
        *,
        target: str,
        display_name: str = "",
        existing: Optional[dict] = None,
    ) -> Optional[dict]:
        identity = resolve_kasia_identity(target, display_name=display_name)
        if not identity.canonical_address:
            return None

        existing = existing or {}
        record = identity.to_record()
        if existing.get("kns_name") and not record.get("kns_name"):
            record["kns_name"] = existing.get("kns_name")
        if existing.get("display_name") and record.get("display_name") == record.get("canonical_address"):
            record["display_name"] = existing.get("display_name")
            record["user_name"] = existing.get("display_name")
        if existing.get("original_target") and not record.get("original_target"):
            record["original_target"] = existing.get("original_target")
        return record

    # ----- Approved users -----

    def is_approved(self, platform: str, user_id: str) -> bool:
        """Check if a user is approved (paired) on a platform."""
        platform = self._normalize_platform(platform)
        approved = self._load_json(self._approved_path(platform))
        if self._is_kasia_platform(platform):
            key, _ = self._find_kasia_entry(approved, user_id)
            return key is not None
        return user_id in approved

    def list_approved(self, platform: str = None) -> list:
        """List approved users, optionally filtered by platform."""
        results = []
        platforms = [self._normalize_platform(platform)] if platform else self._all_platforms("approved")
        for p in platforms:
            approved = self._load_json(self._approved_path(p))
            for uid, info in approved.items():
                results.append({"platform": p, "user_id": uid, **info})
        return results

    def _approve_user(self, platform: str, user_id: str, user_name: str = "") -> None:
        """Add a user to the approved list."""
        platform = self._normalize_platform(platform)
        approved = self._load_json(self._approved_path(platform))
        approved[user_id] = {
            "user_name": user_name,
            "approved_at": time.time(),
        }
        self._save_json(self._approved_path(platform), approved)

    def revoke(self, platform: str, user_id: str) -> bool:
        """Remove a user from the approved list. Returns True if found."""
        platform = self._normalize_platform(platform)
        path = self._approved_path(platform)
        approved = self._load_json(path)
        if self._is_kasia_platform(platform):
            key, _ = self._find_kasia_entry(approved, user_id)
            if key:
                del approved[key]
                self._save_json(path, approved)
                return True
            return False
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
        platform = self._normalize_platform(platform)
        if self._is_kasia_platform(platform):
            return None

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

    def get_pending_code(self, platform: str, user_id: str) -> Optional[str]:
        """Return an existing unexpired pending code for a user, if any."""
        platform = self._normalize_platform(platform)
        if self._is_kasia_platform(platform):
            return None
        self._cleanup_expired(platform)
        pending = self._load_json(self._pending_path(platform))
        for code, info in pending.items():
            if info.get("user_id") == user_id:
                return code
        return None

    def record_pending_request(
        self,
        platform: str,
        user_id: str,
        user_name: str = "",
    ) -> Optional[dict]:
        """Queue a pending Kasia contact request keyed by canonical address."""
        platform = self._normalize_platform(platform)
        if not self._is_kasia_platform(platform):
            return None

        pending = self._load_json(self._pending_path(platform))
        existing_key, existing = self._find_kasia_entry(pending, user_id)
        entry = self._build_kasia_entry(
            target=user_id,
            display_name=user_name,
            existing=existing,
        )
        if not entry:
            return None

        now = time.time()
        entry["created_at"] = existing.get("created_at", now) if existing else now
        entry["last_seen_at"] = now
        pending[entry["canonical_address"]] = entry
        if existing_key and existing_key != entry["canonical_address"]:
            pending.pop(existing_key, None)
        self._save_json(self._pending_path(platform), pending)

        age_min = int((now - entry["created_at"]) / 60)
        return {"platform": platform, "age_minutes": age_min, **entry}

    def approve_identity(self, platform: str, target: str) -> Optional[dict]:
        """Approve a Kasia contact by address or KNS name."""
        platform = self._normalize_platform(platform)
        if not self._is_kasia_platform(platform):
            return None

        pending = self._load_json(self._pending_path(platform))
        pending_key, pending_entry = self._find_kasia_entry(pending, target)
        record = self._build_kasia_entry(
            target=target,
            display_name=(pending_entry or {}).get("display_name", ""),
            existing=pending_entry,
        )
        if not record and pending_entry:
            record = dict(pending_entry)
        if not record or not record.get("canonical_address"):
            return None

        if pending_key:
            pending.pop(pending_key, None)
            self._save_json(self._pending_path(platform), pending)

        approved = self._load_json(self._approved_path(platform))
        approved[record["canonical_address"]] = {
            **record,
            "approved_at": time.time(),
        }
        self._save_json(self._approved_path(platform), approved)
        return {"platform": platform, **approved[record["canonical_address"]]}

    def approve_code(self, platform: str, code: str) -> Optional[dict]:
        """
        Approve a pairing code. Adds the user to the approved list.

        Returns {user_id, user_name} on success, None if code is invalid/expired.
        """
        platform = self._normalize_platform(platform)
        if self._is_kasia_platform(platform):
            return self.approve_identity(platform, code)

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
        platforms = [self._normalize_platform(platform)] if platform else self._all_platforms("pending")
        for p in platforms:
            self._cleanup_expired(p)
            pending = self._load_json(self._pending_path(p))
            for key, info in pending.items():
                age_min = int((time.time() - info["created_at"]) / 60)
                entry = {
                    "platform": p,
                    "user_id": info.get("user_id") or key,
                    "user_name": info.get("user_name", ""),
                    "age_minutes": age_min,
                    **info,
                }
                if not self._is_kasia_platform(p):
                    entry["code"] = key
                results.append(entry)
        return results

    def clear_pending(self, platform: str = None) -> int:
        """Clear all pending requests. Returns count removed."""
        count = 0
        platforms = [self._normalize_platform(platform)] if platform else self._all_platforms("pending")
        for p in platforms:
            pending = self._load_json(self._pending_path(p))
            count += len(pending)
            self._save_json(self._pending_path(p), {})
        return count

    # ----- Rate limiting and lockout -----

    def _is_rate_limited(self, platform: str, user_id: str) -> bool:
        """Check if a user has requested a code too recently."""
        platform = self._normalize_platform(platform)
        limits = self._load_json(self._rate_limit_path())
        key = f"{platform}:{user_id}"
        last_request = limits.get(key, 0)
        return (time.time() - last_request) < RATE_LIMIT_SECONDS

    def _record_rate_limit(self, platform: str, user_id: str) -> None:
        """Record the time of a pairing request for rate limiting."""
        platform = self._normalize_platform(platform)
        limits = self._load_json(self._rate_limit_path())
        key = f"{platform}:{user_id}"
        limits[key] = time.time()
        self._save_json(self._rate_limit_path(), limits)

    def _is_locked_out(self, platform: str) -> bool:
        """Check if a platform is in lockout due to failed approval attempts."""
        platform = self._normalize_platform(platform)
        limits = self._load_json(self._rate_limit_path())
        lockout_key = f"_lockout:{platform}"
        lockout_until = limits.get(lockout_key, 0)
        return time.time() < lockout_until

    def _record_failed_attempt(self, platform: str) -> None:
        """Record a failed approval attempt. Triggers lockout after MAX_FAILED_ATTEMPTS."""
        platform = self._normalize_platform(platform)
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
        platform = self._normalize_platform(platform)
        if self._is_kasia_platform(platform):
            return
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

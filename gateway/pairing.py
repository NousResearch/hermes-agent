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
import logging
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
from hermes_constants import get_hermes_dir, get_hermes_home
from utils import atomic_replace

logger = logging.getLogger(__name__)


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


# Platform value -> its per-platform allowlist env var. When an operator has
# already configured an allowlist for a platform, approving a pairing code also
# writes the user into that allowlist (and revoking removes them), so the
# operator's own list stays the single visible/editable source of truth instead
# of drifting from an opaque approved.json (#23778 consolidation, option i).
# Platforms absent from this map (or with no allowlist configured) keep the
# pairing store as the sole grant record, honored by the authz union.
_PLATFORM_ALLOWLIST_ENV = {
    "telegram": "TELEGRAM_ALLOWED_USERS",
    "discord": "DISCORD_ALLOWED_USERS",
    "whatsapp": "WHATSAPP_ALLOWED_USERS",
    "whatsapp_cloud": "WHATSAPP_CLOUD_ALLOWED_USERS",
    "slack": "SLACK_ALLOWED_USERS",
    "signal": "SIGNAL_ALLOWED_USERS",
    "email": "EMAIL_ALLOWED_USERS",
    "sms": "SMS_ALLOWED_USERS",
    "mattermost": "MATTERMOST_ALLOWED_USERS",
    "matrix": "MATRIX_ALLOWED_USERS",
    "dingtalk": "DINGTALK_ALLOWED_USERS",
    "feishu": "FEISHU_ALLOWED_USERS",
    "wecom": "WECOM_ALLOWED_USERS",
    "wecom_callback": "WECOM_CALLBACK_ALLOWED_USERS",
    "weixin": "WEIXIN_ALLOWED_USERS",
    "bluebubbles": "BLUEBUBBLES_ALLOWED_USERS",
    "qqbot": "QQ_ALLOWED_USERS",
    "yuanbao": "YUANBAO_ALLOWED_USERS",
}


def _allowlist_env_for_platform(platform: str) -> Optional[str]:
    """Return the per-platform allowlist env var name, or None.

    Falls back to the platform registry for plugin platforms so a plugin's
    own ``allowed_users_env`` is honored too.
    """
    platform = (platform or "").lower().strip()
    env_var = _PLATFORM_ALLOWLIST_ENV.get(platform)
    if env_var:
        return env_var
    try:
        from gateway.platform_registry import platform_registry

        entry = platform_registry.get(platform)
        if entry and entry.allowed_users_env:
            return entry.allowed_users_env
    except Exception:
        pass
    return None


def _split_allowlist(raw: str) -> list:
    return [uid.strip() for uid in raw.split(",") if uid.strip()]


def _sync_allowlist_add(platform: str, user_id: str) -> None:
    """Add ``user_id`` to the platform allowlist env var IF one is configured.

    Option (i): only materialize the grant into the allowlist when the operator
    already runs an allowlist for this platform. On an open gateway (no
    allowlist) we do nothing — the pairing store remains the grant record and
    the authz union honors it, so we never silently convert an open gateway into
    a locked one on first pairing.
    """
    env_var = _allowlist_env_for_platform(platform)
    if not env_var:
        return
    current = os.getenv(env_var, "").strip()
    if not current:
        return  # No allowlist configured — leave the gateway open (option i).
    ids = _split_allowlist(current)
    if "*" in ids or str(user_id) in ids:
        return  # Already covered.
    ids.append(str(user_id))
    try:
        from hermes_cli.config import save_env_value

        save_env_value(env_var, ",".join(ids))
    except Exception:
        # Best-effort: the pairing store grant still authorizes via the union,
        # so a failure here degrades to "grant recorded but not mirrored".
        pass


def _sync_allowlist_remove(platform: str, user_id: str) -> None:
    """Remove ``user_id`` from the platform allowlist env var if present."""
    env_var = _allowlist_env_for_platform(platform)
    if not env_var:
        return
    current = os.getenv(env_var, "").strip()
    if not current:
        return
    ids = _split_allowlist(current)
    remaining = [i for i in ids if i != str(user_id)]
    if len(remaining) == len(ids):
        return  # Not present.
    try:
        from hermes_cli.config import save_env_value, remove_env_value

        if remaining:
            save_env_value(env_var, ",".join(remaining))
        else:
            remove_env_value(env_var)
    except Exception:
        pass


def _load_json_file(path: Path) -> dict:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _merge_pairing_dir(active_dir: Path, alternate_dir: Path) -> None:
    """Merge split legacy/new pairing data into the active PairingStore dir.

    Older installs use ``{HERMES_HOME}/pairing`` while newer code/docs may
    write ``{HERMES_HOME}/platforms/pairing``. If both directories exist, the
    gateway must not silently ignore approved users sitting in the inactive
    location; otherwise already-paired Feishu users get asked for a fresh code.
    """
    if not alternate_dir.exists() or active_dir.resolve() == alternate_dir.resolve():
        return
    active_dir.mkdir(parents=True, exist_ok=True)
    for src in alternate_dir.glob("*.json"):
        if not src.is_file():
            continue
        dest = active_dir / src.name
        merged = _load_json_file(src)
        if not merged:
            continue
        current = _load_json_file(dest)
        before = dict(current)
        # Active data wins on key conflict; otherwise union the inactive data.
        merged.update(current)
        if merged != before:
            _secure_write(dest, json.dumps(merged, indent=2, ensure_ascii=False))


def _migrate_split_pairing_dirs() -> None:
    home = get_hermes_home()
    old_dir = home / "pairing"
    new_dir = home / "platforms" / "pairing"
    active = PAIRING_DIR
    alternate = new_dir if active.resolve() == old_dir.resolve() else old_dir
    _merge_pairing_dir(active, alternate)


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
        atomic_replace(tmp_path, path)
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
        if not profile:
            # Heal installs whose global pairing data ended up split across
            # the legacy and new directories (per-profile stores never had
            # the legacy/new split).
            _migrate_split_pairing_dirs()
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
            except PermissionError as e:
                # Surface this loudly: a 0600 file owned by a different user
                # (classic Docker symptom: `docker exec` runs as root and writes
                # the file, then the gateway process — running as `hermes` after
                # gosu drop — can't read it) would otherwise be swallowed by
                # the generic OSError branch below, silently leaving the user
                # marked unauthorized. See issue #10270.
                try:
                    st = path.stat()
                    owner_info = f"owner_uid={st.st_uid} mode={oct(st.st_mode)[-4:]}"
                except OSError:
                    owner_info = "<stat failed>"
                # os.geteuid doesn't exist on Windows; the Docker scenario is
                # POSIX-only, but the gateway (and this fallback) runs anywhere.
                euid = os.geteuid() if hasattr(os, "geteuid") else "n/a"
                logger.warning(
                    "Pairing file %s exists but is not readable as uid=%s (%s; %s). "
                    "If you ran `docker exec <container> hermes pairing approve ...` as root, "
                    "re-run with `docker exec -u hermes <container> ...` and "
                    "chown the existing file to the hermes user, or restart the "
                    "container so the entrypoint can fix ownership.",
                    path, euid, owner_info, e,
                )
                return {}
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

        # Mirror the grant into the operator's allowlist when one is configured
        # (option i), so the pairing store and the allowlist stay a single
        # visible source of truth. No-op on open gateways.
        _sync_allowlist_add(platform, normalized_user_id)

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
                # Keep the allowlist mirror in sync: revoking a paired user
                # also removes the entry the approval added (option i). No-op if
                # the user was added to the allowlist by other means.
                _sync_allowlist_remove(platform, user_id)
                return True
        return False

    # ----- Pending codes -----

    @staticmethod
    def _hash_code(code: str, salt: bytes) -> str:
        """Hash a pairing code with the given salt using SHA-256."""
        return hashlib.sha256(salt + code.encode("utf-8")).hexdigest()

    def generate_code(
        self, platform: str, user_id: str, user_name: str = ""
    ) -> Optional[str]:
        """
        Generate a pairing code for a new user.

        Returns the code string, or None if:
          - User is rate-limited (too recent request)
          - Max pending codes reached for this platform
          - User/platform is in lockout due to failed attempts

        The code is NOT stored in plaintext.  Only a salted SHA-256 hash is
        persisted so that reading the pending file does not reveal codes.
        """
        with self._lock:
            self._cleanup_expired(platform)
            normalized_user_id = self._normalize_user_id(platform, user_id)

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

            # Hash the code with a random salt before storing
            salt = os.urandom(16)
            code_hash = self._hash_code(code, salt)

            # Use a unique entry id as the key (not the code itself)
            entry_id = secrets.token_hex(8)

            # Store pending request with hashed code
            pending[entry_id] = {
                "hash": code_hash,
                "salt": salt.hex(),
                "user_id": normalized_user_id,
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

        Returns ``{user_id, user_name}`` on success, ``None`` if the code is
        invalid/expired OR the platform is currently locked out after
        ``MAX_FAILED_ATTEMPTS`` failed approvals (#10195). Callers can
        disambiguate with ``_is_locked_out(platform)``.

        Verification: the user-provided code is hashed with each stored
        entry's salt and compared to the stored hash using constant-time
        comparison. Pre-hash entries (legacy plaintext-key format from
        pre-upgrade pending.json files) are silently ignored — they get
        pruned at TTL by ``_cleanup_expired``.
        """
        with self._lock:
            self._cleanup_expired(platform)
            code = code.upper().strip()

            # Lockout check — must run before the pending lookup so a
            # valid code (e.g. one already sitting in pending) cannot be
            # accepted once the lockout fires. Without this, the lockout
            # only blocks `generate_code`, not `approve_code` — nullifying
            # the brute-force protection for any code already issued.
            if self._is_locked_out(platform):
                return None

            pending = self._load_json(self._pending_path(platform))

            # Find the entry whose hash matches the provided code.
            # Tolerate legacy plaintext-key entries (no salt/hash) and
            # malformed entries — skip them rather than KeyError, so an
            # in-place upgrade across an existing pending.json doesn't
            # crash on the first approve call. Legacy entries get pruned
            # at their TTL by _cleanup_expired.
            matched_key = None
            matched_entry = None
            for entry_id, entry in pending.items():
                if not isinstance(entry, dict):
                    continue
                if "salt" not in entry or "hash" not in entry:
                    continue
                try:
                    salt = bytes.fromhex(entry["salt"])
                except ValueError:
                    continue
                candidate_hash = self._hash_code(code, salt)
                if secrets.compare_digest(candidate_hash, entry["hash"]):
                    matched_key = entry_id
                    matched_entry = entry
                    break

            if matched_key is None:
                self._record_failed_attempt(platform)
                return None

            del pending[matched_key]
            self._save_json(self._pending_path(platform), pending)

            # Add to approved list
            self._approve_user(platform, matched_entry["user_id"],
                               matched_entry.get("user_name", ""))

            return {
                "user_id": matched_entry["user_id"],
                "user_name": matched_entry.get("user_name", ""),
            }

    def list_pending(self, platform: str = None) -> list:
        """List pending pairing requests, optionally filtered by platform.

        Codes are stored hashed — the ``code`` field is replaced with the
        first 8 hex characters of the hash so admins can distinguish entries
        without revealing the original code. Legacy plaintext-key entries
        (pre-hash format) are shown with a "legacy" placeholder so admins
        can see them age out without crashing on a missing ``hash`` field.
        """
        results = []
        with self._lock:
            platforms = [platform] if platform else self._all_platforms("pending")
            for p in platforms:
                self._cleanup_expired(p)
                pending = self._load_json(self._pending_path(p))
                for entry_id, info in pending.items():
                    if not isinstance(info, dict):
                        continue
                    created_at = info.get("created_at")
                    if not isinstance(created_at, (int, float)):
                        continue
                    age_min = int((time.time() - created_at) / 60)
                    hash_val = info.get("hash")
                    code_display = hash_val[:8] if isinstance(hash_val, str) else "legacy"
                    results.append({
                        "platform": p,
                        "code": code_display,
                        "user_id": info.get("user_id", ""),
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

    # ----- Rate limiting and lockout -----

    def _is_rate_limited(self, platform: str, user_id: str) -> bool:
        """Check if a user has requested a code too recently."""
        limits = self._load_json(self._rate_limit_path())
        for alias in self._user_id_aliases(platform, user_id):
            key = f"{platform}:{alias}"
            last_request = limits.get(key, 0)
            if (time.time() - last_request) < RATE_LIMIT_SECONDS:
                return True
        return False

    def _record_rate_limit(self, platform: str, user_id: str) -> None:
        """Record the time of a pairing request for rate limiting."""
        limits = self._load_json(self._rate_limit_path())
        now = time.time()
        for alias in self._user_id_aliases(platform, user_id):
            key = f"{platform}:{alias}"
            limits[key] = now
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
        """Remove expired pending codes.

        Tolerant of malformed / legacy entries — anything without a numeric
        ``created_at`` is treated as expired (it's effectively unusable
        with the new hash-keyed schema anyway).
        """
        path = self._pending_path(platform)
        pending = self._load_json(path)
        now = time.time()
        expired = []
        for entry_id, info in pending.items():
            if not isinstance(info, dict):
                expired.append(entry_id)
                continue
            created_at = info.get("created_at")
            if not isinstance(created_at, (int, float)):
                expired.append(entry_id)
                continue
            if (now - created_at) > CODE_TTL_SECONDS:
                expired.append(entry_id)
        if expired:
            for entry_id in expired:
                del pending[entry_id]
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

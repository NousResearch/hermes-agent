"""DM pairing: code-based approval of new users on messaging platforms.

Unknown users receive a one-time pairing code that the bot owner approves via the
CLI, instead of static user-ID allowlists. Security properties (OWASP + NIST SP
800-63-4): 8-char codes from a 32-char unambiguous alphabet via ``secrets``, 1-hour
expiry, max 3 pending per platform, 1 request per user per 10 min, lockout after 5
failed approvals, chmod 0600 data files, codes never logged. Storage: ~/.hermes/pairing/
"""

import contextlib
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

from gateway.whatsapp_identity import expand_whatsapp_aliases, normalize_whatsapp_identifier
from hermes_constants import get_default_hermes_root, get_hermes_dir, get_hermes_home
from utils import atomic_replace

logger = logging.getLogger(__name__)


# Unambiguous alphabet -- excludes 0/O, 1/I to prevent confusion
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
CODE_LENGTH = 8

CODE_TTL_SECONDS = 3600             # Codes expire after 1 hour
RATE_LIMIT_SECONDS = 600            # 1 request per user per 10 minutes
LOCKOUT_SECONDS = 3600              # Lockout duration after too many failures
MAX_PENDING_PER_PLATFORM = 3        # Max pending codes per platform
MAX_FAILED_ATTEMPTS = 5             # Failed approvals before lockout

# Default pairing directory override. Deliberately ``None``: an eagerly computed
# path would freeze the HERMES_HOME/profile context at gateway boot, ignoring later
# context-local overrides, so the gateway and ``hermes pairing`` CLI wrote different
# directories. ``_default_pairing_dir()`` resolves fresh per call; tests patch this.
PAIRING_DIR = None


# Default (non-profile-scoped) pairing directory. Left unresolved (``None``) here rather than computed
# eagerly: this module is imported once by the long-lived gateway process at container/process boot, and
# computing the path eagerly freezes it to whatever HERMES_HOME/profile context existed at that exact import
# moment for the rest of the process's lifetime -- even if a context-local override (see
# hermes_constants.set_hermes_home_override) is established afterward. A freshly-started, short-lived
# process (e.g. the ``hermes pairing`` CLI) re-imports this module later with the final environment already
# in place, so it never observes the stale value -- the resulting asymmetry is what made pending pairing
# codes issued by the gateway unrecoverable while CLI-side writes to the same directory kept working
# (NousResearch/hermes-agent#93449). ``_default_pairing_dir()`` below resolves this fresh on every call in
# production. Tests patch this attribute directly to a concrete path for isolation (e.g.
# ``patch("gateway.pairing.PAIRING_DIR", tmp_path)``); that continues to work unchanged, since a patched
# (non-``None``) value takes precedence over recomputing.
def _default_pairing_dir() -> Path:
    return PAIRING_DIR if PAIRING_DIR is not None else get_hermes_dir("platforms/pairing", "pairing")


# Platform value -> allowlist env var. Approving a code also writes the user into
# an already-configured allowlist (revoke removes them) so the operator's list stays
# the visible source of truth. Platforms absent here (or with no allowlist
# configured) keep the pairing store as the sole grant record (authz union).
# See #23778.
_PLATFORM_ALLOWLIST_ENV = {
    "telegram": "TELEGRAM_ALLOWED_USERS", "discord": "DISCORD_ALLOWED_USERS",
    "whatsapp": "WHATSAPP_ALLOWED_USERS", "whatsapp_cloud": "WHATSAPP_CLOUD_ALLOWED_USERS",
    "slack": "SLACK_ALLOWED_USERS", "signal": "SIGNAL_ALLOWED_USERS",
    "email": "EMAIL_ALLOWED_USERS", "sms": "SMS_ALLOWED_USERS",
    "mattermost": "MATTERMOST_ALLOWED_USERS", "matrix": "MATRIX_ALLOWED_USERS",
    "dingtalk": "DINGTALK_ALLOWED_USERS", "feishu": "FEISHU_ALLOWED_USERS",
    "wecom": "WECOM_ALLOWED_USERS", "wecom_callback": "WECOM_CALLBACK_ALLOWED_USERS",
    "weixin": "WEIXIN_ALLOWED_USERS", "bluebubbles": "BLUEBUBBLES_ALLOWED_USERS",
    "qqbot": "QQ_ALLOWED_USERS", "yuanbao": "YUANBAO_ALLOWED_USERS",
}


def _allowlist_env_for_platform(platform: str) -> Optional[str]:
    """Allowlist env var name for ``platform`` (plugin registry fallback), or None."""
    platform = (platform or "").lower().strip()
    if env_var := _PLATFORM_ALLOWLIST_ENV.get(platform):
        return env_var
    with contextlib.suppress(Exception):
        from gateway.platform_registry import platform_registry
        return platform_registry.get(platform).allowed_users_env or None
    return None


def _split_allowlist(raw: str) -> list:
    return [uid.strip() for uid in raw.split(",") if uid.strip()]


def _platform_uses_whatsapp_identity(platform: str) -> bool:
    """True for Baileys WhatsApp and Meta Cloud — same phone/JID identity rules."""
    return (platform or "").strip().lower() in {"whatsapp", "whatsapp_cloud"}


def _normalize_user_id(platform: str, user_id: str) -> str:
    """Normalize platform-specific user IDs before persisting / comparing them."""
    raw_user_id = str(user_id or "").strip()
    return (normalize_whatsapp_identifier(raw_user_id) or raw_user_id) if _platform_uses_whatsapp_identity(platform) else raw_user_id


def _user_id_aliases(platform: str, user_id: str) -> set[str]:
    """All known equivalent user IDs for auth / allowlist matching."""
    raw_user_id = str(user_id or "").strip()
    if not raw_user_id:
        return set()
    aliases = {raw_user_id, _normalize_user_id(platform, raw_user_id)}
    if _platform_uses_whatsapp_identity(platform):
        aliases.update(expand_whatsapp_aliases(raw_user_id))
    aliases.discard("")
    return aliases


def _user_ids_match(platform: str, left: str, right: str) -> bool:
    """True when two user IDs represent the same principal."""
    left_aliases = _user_id_aliases(platform, left)
    return bool(left_aliases and left_aliases & _user_id_aliases(platform, right))


def _matching_ids(platform: str, approved: dict, user_id: str) -> list:
    return [uid for uid in approved if _user_ids_match(platform, uid, user_id)]


def _read_allowlist_env(env_var: str) -> str:
    """Read a platform allowlist env var through the profile secret scope.

    Under multiplexing the process env may hold ANOTHER profile's allowlist, so a scoped
    miss must return empty rather than borrow it; unscoped callers keep the legacy
    ``os.getenv`` read. Writes (``save_env_value``/``remove_env_value``) target the
    active profile's ``.env`` / installed scope, not ``os.environ``.

    See #88441.
    """
    with contextlib.suppress(Exception):
        from agent.secret_scope import UnscopedSecretError, get_secret
        try:
            return (get_secret(env_var) or "").strip()
        except UnscopedSecretError:
            pass
    return (os.getenv(env_var) or "").strip()


def _configured_allowlist(platform: str):
    """``(env_var, ids)`` for a platform whose allowlist is configured, else None.

    An unconfigured allowlist means an open gateway: the pairing store stays the
    sole grant record and we must never lock the gateway by materializing one.
    """
    env_var = _allowlist_env_for_platform(platform)
    current = _read_allowlist_env(env_var) if env_var else ""
    return (env_var, _split_allowlist(current)) if current else None


def _write_allowlist_env(env_var: str, ids: list) -> None:
    """Best-effort persist (empty list removes the key); the pairing store grant still authorizes via the union."""
    with contextlib.suppress(Exception):
        from hermes_cli.config import save_env_value, remove_env_value
        save_env_value(env_var, ",".join(ids)) if ids else remove_env_value(env_var)


def _sync_allowlist_add(platform: str, user_id: str) -> None:
    """Add ``user_id`` to the platform allowlist env var IF one is configured."""
    configured = _configured_allowlist(platform)
    if configured is None:
        return
    env_var, ids = configured
    if "*" in ids or str(user_id) in ids:
        return
    _write_allowlist_env(env_var, [*ids, str(user_id)])


def _iter_live_gateway_adapters():
    """Yield adapters from the in-process GatewayRunner, if one is running."""
    runner = None
    with contextlib.suppress(Exception):
        from gateway.run import _gateway_runner_ref
        runner = _gateway_runner_ref()
    if runner is None:
        return
    mappings = [getattr(runner, "adapters", None) or {}, *(getattr(runner, "_profile_adapters", None) or {}).values()]
    for mapping in mappings:
        for adapter in (mapping or {}).values():
            if adapter is not None:
                yield adapter


def _adapter_platform_name(adapter) -> str:
    value = getattr(getattr(adapter, "platform", None), "value", None)
    return str(value or getattr(adapter, "name", None) or "").strip().lower()


def _purge_allowlist_entries(entries, platform: str, user_id: str):
    """Drop alias-equivalent allowlist entries while preserving ``*``."""
    def keep(entry) -> bool:
        return str(entry).strip() == "*" or not _user_ids_match(platform, str(entry), str(user_id))
    if isinstance(entries, str):
        return ",".join(filter(keep, _split_allowlist(entries)))
    if isinstance(entries, (set, frozenset)):
        return set(filter(keep, entries))
    if isinstance(entries, (list, tuple)):
        return list(filter(keep, entries))
    return entries


def _sync_live_adapter_allowlist_remove(platform: str, user_id: str) -> None:
    """Clear revoked principals from in-process adapter ``_allow_from`` snapshots,
    so intake does not keep authorizing from a stale snapshot until restart."""
    platform_name = (platform or "").strip().lower()
    if not platform_name or not str(user_id or "").strip():
        return
    for adapter in _iter_live_gateway_adapters():
        if _adapter_platform_name(adapter) != platform_name:
            continue
        if hasattr(adapter, "_allow_from"):
            with contextlib.suppress(Exception):
                adapter._allow_from = _purge_allowlist_entries(set(adapter._allow_from or ()), platform_name, user_id)
        extra = getattr(getattr(adapter, "config", None), "extra", None)
        if isinstance(extra, dict) and "allow_from" in extra:
            with contextlib.suppress(Exception):
                extra["allow_from"] = _purge_allowlist_entries(extra.get("allow_from"), platform_name, user_id)


def _sync_allowlist_remove(platform: str, user_id: str) -> None:
    """Remove ``user_id`` (and WhatsApp alias equivalents) from the allowlist.

    Approve mirrors a normalized phone while revoke is often given a JID/device
    form, so matching uses alias rules -- exact delete would leave the sender authorized.
    An unconfigured allowlist is left alone (config-only snapshots are not touched).
    """
    configured = _configured_allowlist(platform)
    if configured is None:
        return
    env_var, ids = configured
    remaining = _purge_allowlist_entries(ids, platform, user_id)
    if len(remaining) == len(ids):
        return  # Not present.
    _write_allowlist_env(env_var, remaining)
    _sync_live_adapter_allowlist_remove(platform, user_id)


def _load_json_file(path: Path) -> dict:
    """Read a JSON object; {} when missing, malformed, unreadable, or not a dict.

    PermissionError is logged loudly: a 0600 file owned by another uid (Docker:
    ``docker exec`` as root wrote it, the gosu-dropped gateway can't read it)
    would otherwise silently leave the user unauthorized.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except PermissionError as e:
        try:
            # Surface this loudly: a 0600 file owned by a different user (classic Docker symptom: `docker
            # exec` runs as root and writes the file, then the gateway process — running as `hermes` after
            # gosu drop — can't read it) would otherwise be swallowed by the generic OSError branch below,
            # silently leaving the user marked unauthorized. See issue #10270.
            st = path.stat()
            owner_info = f"owner_uid={st.st_uid} mode={oct(st.st_mode)[-4:]}"
        except OSError:
            owner_info = "<stat failed>"
        euid = os.geteuid() if hasattr(os, "geteuid") else "n/a"  # no geteuid on Windows
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


def _save_json_file(path: Path, data: dict) -> None:
    _secure_write(path, json.dumps(data, indent=2, ensure_ascii=False))


def _migrate_split_pairing_dirs(*, home: Optional[Path] = None, active: Optional[Path] = None) -> None:
    """Merge split legacy (``pairing``) / new (``platforms/pairing``) data into the active dir.

    If both exist, approved users in the inactive location must not be silently
    ignored (they would be asked for a fresh code). Active data wins on key conflict.
    """
    home = home or get_hermes_home()
    old_dir = home / "pairing"
    active = active if active is not None else _default_pairing_dir()
    alternate = home / "platforms" / "pairing" if active.resolve() == old_dir.resolve() else old_dir
    if not alternate.exists() or active.resolve() == alternate.resolve():
        return
    active.mkdir(parents=True, exist_ok=True)
    for src in alternate.glob("*.json"):
        merged = _load_json_file(src) if src.is_file() else {}
        if not merged:
            continue
        current = _load_json_file(active / src.name)
        merged.update(current)
        if merged != current:
            _save_json_file(active / src.name, merged)


def _secure_write(path: Path, data: str) -> None:
    """Write 0600 via temp file + atomic rename so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp_path, path)
        with contextlib.suppress(OSError):  # Windows doesn't support chmod the same way
            os.chmod(path, 0o600)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def _is_hashed_entry(entry) -> bool:
    return isinstance(entry, dict) and "salt" in entry and "hash" in entry


def _entry_created_at(info):
    """Numeric ``created_at`` of a pending entry, or None for malformed/legacy entries."""
    created_at = info.get("created_at") if isinstance(info, dict) else None
    return created_at if isinstance(created_at, (int, float)) else None


class PairingStore:
    """Pairing codes and approved user lists.

    Files per platform: ``{platform}-pending.json``, ``{platform}-approved.json``, plus
    shared ``_rate_limits.json``. With ``profile="<name>"`` storage resolves from that
    profile's HERMES_HOME exactly as ``hermes -p <name> pairing ...`` does, so multiplex
    gateways and profile-scoped CLI approvals share one whitelist.
    """

    def __init__(self, profile: Optional[str] = None):
        profile_home = None
        if profile:
            root = get_default_hermes_root()
            profile_home = root if profile == "default" else root / "profiles" / profile
        self._dir = get_hermes_dir("platforms/pairing", "pairing", home=profile_home) if profile else _default_pairing_dir()
        self._dir.mkdir(parents=True, exist_ok=True)
        # Merge the alternate old/new layout so upgrades cannot split approvals.
        _migrate_split_pairing_dirs(home=profile_home, active=self._dir)
        self._lock = threading.RLock()  # adapters run concurrently in threads sharing one store
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

    def _cleanup_expired(self, platform: str) -> None:
        """Remove expired pending codes; malformed/legacy entries (no numeric ``created_at``) count as expired."""
        path = self._pending_path(platform)
        pending = self._load_json(path)
        now = time.time()
        live = {
            k: v for k, v in pending.items()
            if (created := _entry_created_at(v)) is not None and (now - created) <= CODE_TTL_SECONDS
        }
        if len(live) != len(pending):
            self._save_json(path, live)

    _load_json = staticmethod(_load_json_file)
    _save_json = staticmethod(_save_json_file)

    def _platforms(self, platform: Optional[str], suffix: str) -> list:
        return [platform] if platform else self._all_platforms(suffix)

    # ----- Approved users -----

    def is_approved(self, platform: str, user_id: str) -> bool:
        """Check if a user is approved (paired) on a platform."""
        return bool(_matching_ids(platform, self._load_json(self._approved_path(platform)), user_id))

    def list_approved(self, platform: str = None) -> list:
        """List approved users, optionally filtered by platform."""
        return [
            {"platform": p, "user_id": uid, **info}
            for p in self._platforms(platform, "approved")
            for uid, info in self._load_json(self._approved_path(p)).items()
        ]

    def _approve_user(self, platform: str, user_id: str, user_name: str = "") -> None:
        """Add a user to the approved list. Must be called under self._lock."""
        approved = self._load_json(self._approved_path(platform))
        normalized_user_id = _normalize_user_id(platform, user_id)
        for approved_user_id in _matching_ids(platform, approved, normalized_user_id):
            del approved[approved_user_id]
        approved[normalized_user_id] = {"user_name": user_name, "approved_at": time.time()}
        self._save_json(self._approved_path(platform), approved)
        # Mirror the grant into the operator's allowlist when one is configured.
        _sync_allowlist_add(platform, normalized_user_id)

    def revoke(self, platform: str, user_id: str) -> bool:
        """Remove a user from the approved list. Returns True if found."""
        path = self._approved_path(platform)
        with self._lock:
            approved = self._load_json(path)
            matching_ids = _matching_ids(platform, approved, user_id)
            if not matching_ids:
                return False
            for approved_user_id in matching_ids:
                del approved[approved_user_id]
            self._save_json(path, approved)
            # Keep the allowlist mirror in sync (no-op if added by other means).
            _sync_allowlist_remove(platform, user_id)
            return True

    # ----- Pending codes -----

    @staticmethod
    def _hash_code(code: str, salt: bytes) -> str:
        return hashlib.sha256(salt + code.encode("utf-8")).hexdigest()

    def _finish_approval(self, platform: str, pending: dict, matched_key: str, matched_entry: dict) -> dict:
        """Remove a pending request and approve its user. Must hold self._lock."""
        del pending[matched_key]
        self._save_json(self._pending_path(platform), pending)
        # A successful approval proves legitimacy, so the persisted brute-force streak
        # must not carry over (isolated typos would accumulate into a spurious lockout).
        self._reset_failed_attempts(platform)
        result = {"user_id": matched_entry["user_id"], "user_name": matched_entry.get("user_name", "")}
        self._approve_user(platform, result["user_id"], result["user_name"])
        return result

    def generate_code(self, platform: str, user_id: str, user_name: str = "") -> Optional[str]:
        """Generate a pairing code for a new user.

        Returns None if the user is rate-limited, the platform hit MAX_PENDING_PER_PLATFORM,
        or the platform is locked out. Only a salted SHA-256 hash of the code is persisted.
        """
        with self._lock:
            self._cleanup_expired(platform)
            normalized_user_id = _normalize_user_id(platform, user_id)
            if self._is_locked_out(platform) or self._is_rate_limited(platform, user_id):
                return None
            pending = self._load_json(self._pending_path(platform))
            if len(pending) >= MAX_PENDING_PER_PLATFORM:
                return None
            code = "".join(secrets.choice(ALPHABET) for _ in range(CODE_LENGTH))
            salt = os.urandom(16)
            # Keyed by a random entry id, not the code itself.
            pending[secrets.token_hex(8)] = {
                "hash": self._hash_code(code, salt), "salt": salt.hex(),
                "user_id": normalized_user_id, "user_name": user_name, "created_at": time.time(),
            }
            self._save_json(self._pending_path(platform), pending)
            self._record_rate_limit(platform, user_id)
            return code

    def approve_code(self, platform: str, code: str) -> Optional[dict]:
        """Approve a pairing code and add its user to the approved list.

        Returns ``{user_id, user_name}``, or ``None`` if the code is invalid/expired OR the
        platform is locked out (disambiguate with ``_is_locked_out``). Constant-time
        salted-hash compare; legacy plaintext entries are ignored and pruned at TTL.

        See #10195.
        """
        with self._lock:
            self._cleanup_expired(platform)
            code = code.upper().strip()
            # Before the lookup, or an already-issued valid code would bypass lockout.
            if self._is_locked_out(platform):
                return None
            pending = self._load_json(self._pending_path(platform))
            # Skip legacy/malformed entries so an in-place upgrade doesn't crash.
            for entry_id, entry in pending.items():
                if not _is_hashed_entry(entry):
                    continue
                try:
                    salt = bytes.fromhex(entry["salt"])
                except ValueError:
                    continue
                if secrets.compare_digest(self._hash_code(code, salt), entry["hash"]):
                    return self._finish_approval(platform, pending, entry_id, entry)
            self._record_failed_attempt(platform)
            return None

    @staticmethod
    def looks_like_request_id(value: str) -> bool:
        """True when ``value`` is shaped like a ``list_pending`` request id (16 hex chars);
        pairing codes are 8 uppercase chars, so callers accepting either can dispatch on this."""
        value = str(value or "").strip()
        return len(value) == 16 and all(c in "0123456789abcdefABCDEF" for c in value)

    def approve_request(self, platform: str, request_id: str) -> Optional[dict]:
        """Approve a pending request by its server-side request id (admin surfaces that
        must never reveal the DM'd code). Returns ``{user_id, user_name}`` or ``None``.

        Neither counts toward nor is gated by the brute-force lockout: a request id is only
        obtainable by an authenticated admin, so a stale id is "the row expired", not an
        attack -- counting it would let a few GUI clicks lock the operator out.
        """
        with self._lock:
            self._cleanup_expired(platform)
            request_id = str(request_id or "").strip().lower()
            if not request_id:
                return None
            pending = self._load_json(self._pending_path(platform))
            for entry_id, entry in pending.items():
                if _is_hashed_entry(entry) and secrets.compare_digest(str(entry_id).lower(), request_id):
                    return self._finish_approval(platform, pending, entry_id, entry)
            return None

    def list_pending(self, platform: str = None) -> list:
        """List pending requests (codes are never returned; each exposes a ``request_id``
        for :meth:`approve_request`; legacy pre-hash entries report an empty id)."""
        results = []
        with self._lock:
            for p in self._platforms(platform, "pending"):
                self._cleanup_expired(p)
                for entry_id, info in self._load_json(self._pending_path(p)).items():
                    created_at = _entry_created_at(info)
                    if created_at is None:
                        continue
                    is_modern = isinstance(info.get("hash"), str) and isinstance(info.get("salt"), str)
                    results.append({
                        "platform": p,
                        "request_id": str(entry_id) if is_modern else "",
                        "user_id": info.get("user_id", ""), "user_name": info.get("user_name", ""),
                        "age_minutes": int((time.time() - created_at) / 60),
                    })
        return results

    def clear_pending(self, platform: str = None) -> int:
        """Clear all pending requests. Returns count removed."""
        with self._lock:
            count = 0
            for p in self._platforms(platform, "pending"):
                count += len(self._load_json(self._pending_path(p)))
                self._save_json(self._pending_path(p), {})
        return count

    # ----- Rate limiting and lockout -----

    def _limits(self) -> dict:
        return self._load_json(self._rate_limit_path())

    def _save_limits(self, limits: dict) -> None:
        self._save_json(self._rate_limit_path(), limits)

    def _is_rate_limited(self, platform: str, user_id: str) -> bool:
        """Whether a user (under any alias) has requested a code too recently."""
        limits = self._limits()
        return any(
            (time.time() - limits.get(f"{platform}:{alias}", 0)) < RATE_LIMIT_SECONDS
            for alias in _user_id_aliases(platform, user_id)
        )

    def _record_rate_limit(self, platform: str, user_id: str) -> None:
        limits = self._limits()
        now = time.time()
        for alias in _user_id_aliases(platform, user_id):
            limits[f"{platform}:{alias}"] = now
        self._save_limits(limits)

    def _is_locked_out(self, platform: str) -> bool:
        return time.time() < self._limits().get(f"_lockout:{platform}", 0)

    def _record_failed_attempt(self, platform: str) -> None:
        """Record a failed approval attempt; triggers lockout after MAX_FAILED_ATTEMPTS."""
        limits = self._limits()
        fail_key = f"_failures:{platform}"
        fails = limits.get(fail_key, 0) + 1
        limits[fail_key] = fails
        if fails >= MAX_FAILED_ATTEMPTS:
            limits[f"_lockout:{platform}"] = time.time() + LOCKOUT_SECONDS
            limits[fail_key] = 0
            print(f"[pairing] Platform {platform} locked out for {LOCKOUT_SECONDS}s "
                  f"after {MAX_FAILED_ATTEMPTS} failed attempts", flush=True)
        self._save_limits(limits)

    def _reset_failed_attempts(self, platform: str) -> None:
        """Clear the failed-approval counter after a success (it tracks *consecutive* failures)."""
        limits = self._limits()
        fail_key = f"_failures:{platform}"
        if limits.get(fail_key):
            limits[fail_key] = 0
            self._save_limits(limits)

    def _all_platforms(self, suffix: str) -> list:
        """Platforms that have a ``-<suffix>.json`` data file (``_``-prefixed files are shared state)."""
        tail = f"-{suffix}.json"
        platforms = (f.name.replace(tail, "") for f in self._dir.iterdir() if f.name.endswith(tail))
        return [p for p in platforms if not p.startswith("_")]

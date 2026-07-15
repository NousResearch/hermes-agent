"""Per-account refresh coordination for the OAuth broker.

Each account alias owns one ``AccountSlot`` holding one in-process
``asyncio.Lock`` plus a per-alias advisory lock file, so a refresh token is
consumed by at most one refresher at a time (they are single-use upstream).
The Keychain grant is re-read after both locks are held, and a rotated grant
is written back to the Keychain before the new access token is handed out.

Failure semantics (docs/design/oauth-broker.md §八):

* ``invalid_grant`` / ``refresh_token_reused`` / ``token_revoked`` make the
  slot terminal-unhealthy; the refresh endpoint is never retried for it.
* Everything else (timeouts, 5xx, rate limits) stays retryable.
* One slot's lock or terminal state never affects the other aliases.
"""

from __future__ import annotations

import asyncio
import base64
from contextlib import contextmanager
import hashlib
import json
import logging
import os
import secrets
import stat
import threading
import time
from pathlib import Path
from typing import Callable, Optional

try:  # windows-footgun: ok — broker is Darwin-only; flock is advisory extra
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None  # type: ignore[assignment]

from agent.credential_persistence import _fingerprint_value
from agent.oauth_broker.models import (
    ACCOUNT_ALIASES,
    GrantStoreError,
    OAuthGrant,
    RedactedAccountStatus,
)

logger = logging.getLogger(__name__)

# Mirrors hermes_cli.auth.CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS without
# importing that (heavy) module at broker import time.
DEFAULT_REFRESH_SKEW_SECONDS = 120

# Used only when a refresh result carries no expiry and the new access token
# is not a decodable JWT.
DEFAULT_ACCESS_TOKEN_TTL_SECONDS = 3600

# Terminal OAuth refresh outcomes per the approved design (§八.8).
TERMINAL_REFRESH_ERROR_CODES = frozenset(
    {"invalid_grant", "refresh_token_reused", "token_revoked"}
)

# Bounded Keychain-write attempts per persistence trigger (§八.7). A refresh
# whose upstream rotation succeeded but whose Keychain write keeps failing
# leaves the slot serving the rotated grant from memory, marked
# ``persistence_degraded``; later requests retry with the same bound.
PERSISTENCE_RETRY_ATTEMPTS = 3
PERSISTENCE_RETRY_COOLDOWN_SECONDS = 5.0
RECOVERY_MARKER_VERSION = 1
RECOVERY_MARKER_MAX_BYTES = 4096
DEFINITIVE_NO_ROTATION_ERROR_CODES = frozenset(
    {"codex_rate_limited", "codex_refresh_failed"}
)
SAFE_GRANT_STORE_ERROR_CATEGORIES = frozenset(
    {
        "invalid_alias",
        "invalid_grant_payload",
        "invalid_payload",
        "not_found",
        "os_error",
        "schema_version",
        "unavailable",
        "unknown_field",
    }
)


class SlotRefreshError(RuntimeError):
    """Refresh failure carrying alias, category, and terminality only."""

    def __init__(self, *, alias: str, category: str, terminal: bool) -> None:
        self.alias = alias
        self.category = category
        self.terminal = terminal
        kind = "terminal" if terminal else "retryable"
        super().__init__(
            f"oauth broker slot {alias}: {kind} refresh failure ({category})"
        )


def _jwt_expiry(token: str) -> Optional[float]:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return None
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))
        exp = claims.get("exp")
        if isinstance(exp, bool) or not isinstance(exp, (int, float)):
            return None
        return float(exp)
    except Exception:
        return None


def _default_refresh_fn(access_token: str, refresh_token: str) -> dict:
    """Production refresh boundary: Hermes's pure Codex OAuth refresh."""
    from hermes_cli.auth import refresh_codex_oauth_pure

    return refresh_codex_oauth_pure(access_token, refresh_token)


def _open_secure_directory(path: Path) -> int:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("secure state path is not a directory")
        os.fchmod(fd, 0o700)
    except BaseException:
        os.close(fd)
        raise
    return fd


class _AccountLockAcquireCancelled(Exception):
    pass


def _acquire_account_lock_fd(
    state_dir: Path,
    alias: str,
    cancel_event: Optional[threading.Event] = None,
) -> int:
    if alias not in ACCOUNT_ALIASES:
        raise ValueError(f"unknown account alias {alias!r}")
    directory_fd = _open_secure_directory(Path(state_dir) / "locks")
    try:
        lock_flags = os.O_RDWR | os.O_CREAT
        lock_flags |= getattr(os, "O_CLOEXEC", 0)
        lock_flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(
            f"{alias}.lock",
            lock_flags,
            0o600,
            dir_fd=directory_fd,
        )
    finally:
        os.close(directory_fd)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("account lock path is not a regular file")
        os.fchmod(fd, 0o600)
        if fcntl is not None:
            if cancel_event is None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            else:
                while True:
                    if cancel_event.is_set():
                        raise _AccountLockAcquireCancelled
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        if cancel_event.is_set():
                            raise _AccountLockAcquireCancelled
                        break
                    except BlockingIOError:
                        cancel_event.wait(0.01)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _release_account_lock_fd(fd: int) -> None:
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@contextmanager
def account_process_lock(state_dir: Path, alias: str):
    """Serialize every Keychain mutation for one alias across processes."""

    fd = _acquire_account_lock_fd(Path(state_dir), alias)
    try:
        yield
    finally:
        _release_account_lock_fd(fd)


class AccountSlot:
    def __init__(
        self,
        alias: str,
        *,
        grant_store,
        state_dir: Path,
        refresh_fn: Optional[Callable[[str, str], dict]] = None,
        skew_seconds: int = DEFAULT_REFRESH_SKEW_SECONDS,
        clock: Callable[[], float] = time.time,
        initial_grant: Optional[OAuthGrant] = None,
    ) -> None:
        if alias not in ACCOUNT_ALIASES:
            raise ValueError(f"account alias must be one of {ACCOUNT_ALIASES}")
        self.alias = alias
        self._grant_store = grant_store
        self._state_dir = Path(state_dir)
        self._lock_path = self._state_dir / "locks" / f"{alias}.lock"
        self._recovery_marker_path = (
            self._state_dir / "recovery" / f"{alias}.json"
        )
        self._refresh_fn = refresh_fn or _default_refresh_fn
        self._skew_seconds = max(0, int(skew_seconds))
        self._clock = clock
        self._async_lock = asyncio.Lock()
        # Optionally preloaded by the run path so startup validation and
        # /health/detailed reflect the Keychain state before first traffic.
        self._cached: Optional[OAuthGrant] = initial_grant
        self._generation = 0
        self._terminal_category: Optional[str] = None
        self._last_refresh_at: Optional[float] = None
        self._last_refresh_result: Optional[str] = None
        self._persistence_degraded = False
        self._next_persistence_retry_at = 0.0
        # Retained in memory while a rotated grant is pending persistence.
        # It lets this process recognize an external re-auth even if another
        # process already removed the shared on-disk recovery marker.
        self._pending_stale_refresh_fingerprint: Optional[str] = None
        if initial_grant is not None and self._recovery_marker_blocks(initial_grant):
            # Production run preloads grants before binding. Validate that
            # preloaded state synchronously so status/readiness and the fast
            # token path can never bypass a crash-recovery marker.
            self._terminal_category = "persistence_recovery_required"

    # -- public API --------------------------------------------------------

    async def get_access_token(self, *, force_refresh: bool = False) -> str:
        self._raise_if_terminal()
        generation_seen = self._generation
        if not force_refresh:
            cached = self._cached
            if (
                cached is not None
                and not self._expiring(cached)
                and (
                    not self._persistence_degraded
                    or self._clock() < self._next_persistence_retry_at
                )
            ):
                return cached.access_token
        async with self._async_lock:
            self._raise_if_terminal()
            cached = self._cached
            # A refresh completed while this caller queued for the lock. The
            # rotated token is fresh for forced callers too — forcing again
            # would burn another single-use refresh token for nothing. Its
            # persistence attempts also just ran, so waiters don't pile on.
            if (
                self._generation != generation_seen
                and cached is not None
                and not self._expiring(cached)
            ):
                return cached.access_token
            if (
                not force_refresh
                and cached is not None
                and not self._expiring(cached)
            ):
                if self._persistence_degraded:
                    await self._retry_persistence_locked()
                    cached = self._cached or cached
                return cached.access_token
            return await self._locked_refresh(force_refresh)

    async def force_refresh(self) -> str:
        """Single forced refresh for callers without a failed-token identity."""
        return await self.get_access_token(force_refresh=True)

    async def refresh_after_unauthorized(self, failed_access_token: str) -> str:
        """Refresh only if the 401 belongs to the slot's current generation.

        A delayed 401 for an older access token reuses the already-rotated
        cached grant instead of burning another single-use refresh token.
        """
        self._raise_if_terminal()
        async with self._async_lock:
            self._raise_if_terminal()
            cached = self._cached
            if (
                cached is not None
                and cached.access_token != failed_access_token
                and not self._expiring(cached)
            ):
                return cached.access_token
            return await self._locked_refresh(
                True, failed_access_token=failed_access_token
            )

    def account_id(self) -> Optional[str]:
        """ChatGPT account id of the loaded grant (identifier, not a secret)."""
        grant = self._cached
        return grant.account_id if grant else None

    def status(self) -> RedactedAccountStatus:
        grant = self._cached
        return RedactedAccountStatus(
            alias=self.alias,
            present=grant is not None,
            healthy=self._terminal_category is None,
            terminal_category=self._terminal_category,
            expires_at=grant.expires_at if grant else None,
            last_refresh_at=self._last_refresh_at,
            last_refresh_result=self._last_refresh_result,
            access_token_fingerprint=(
                _fingerprint_value(grant.access_token) if grant else None
            ),
            persistence_degraded=self._persistence_degraded,
        )

    # -- internals ----------------------------------------------------------

    def _raise_if_terminal(self) -> None:
        if self._terminal_category is not None:
            raise SlotRefreshError(
                alias=self.alias,
                category=self._terminal_category,
                terminal=True,
            )

    def _expiring(self, grant: OAuthGrant) -> bool:
        return grant.expires_at <= self._clock() + self._skew_seconds

    async def _locked_refresh(
        self,
        force_refresh: bool,
        *,
        failed_access_token: Optional[str] = None,
    ) -> str:
        preflight_marker_written = False
        refresh_attempt_guarded = False
        fd = await self._acquire_file_lock_async()
        try:
            try:
                if self._persistence_degraded and self._cached is not None:
                    # Before consuming the pending in-memory refresh chain again,
                    # re-check the Keychain under the shared alias lock. A human
                    # login/logout is authoritative and must never be overwritten.
                    grant = await self._resolve_degraded_authority_locked()
                    if not self._persistence_degraded:
                        # A different Keychain grant is a newer external
                        # generation. This request belongs to the old pending
                        # generation, so even bare force_refresh must stop here.
                        return grant.access_token
                else:
                    grant = await asyncio.to_thread(
                        self._grant_store.load, self.alias
                    )
                    if await asyncio.to_thread(
                        self._recovery_marker_blocks, grant
                    ):
                        self._terminal_category = "persistence_recovery_required"
                        raise SlotRefreshError(
                            alias=self.alias,
                            category="persistence_recovery_required",
                            terminal=True,
                        )
                    if (
                        failed_access_token is not None
                        and grant.access_token != failed_access_token
                        and not self._expiring(grant)
                    ):
                        # Another process already rotated the generation that
                        # produced this delayed 401.
                        self._adopt(grant, result=None)
                        return grant.access_token
                    if not force_refresh and not self._expiring(grant):
                        # Another process already rotated the grant on the
                        # Keychain; adopt it without spending a refresh token.
                        self._adopt(grant, result=None)
                        return grant.access_token
                if not self._persistence_degraded:
                    # Cover the transport-ambiguity window: once the request
                    # is dispatched, the old refresh token may already have
                    # been consumed even if no response reaches this process.
                    await asyncio.to_thread(
                        self._write_recovery_marker, grant.refresh_token
                    )
                    self._pending_stale_refresh_fingerprint = (
                        self._refresh_token_fingerprint(grant.refresh_token)
                    )
                    preflight_marker_written = True
                refresh_attempt_guarded = True
                updated = await asyncio.to_thread(
                    self._refresh_fn, grant.access_token, grant.refresh_token
                )
                new_grant = self._merge(grant, updated)
                # Persist before serving when possible. A rotation that
                # cannot be persisted is still served from memory — dropping
                # it would strand the freshly-consumed refresh chain.
                persisted = await asyncio.to_thread(self._try_persist, new_grant)
                if persisted:
                    try:
                        await asyncio.to_thread(self._clear_recovery_marker)
                    except Exception:
                        # The Keychain already contains the replacement. A
                        # stale marker compares unequal on restart and is then
                        # removed; do not turn safe persistence into a 503.
                        logger.warning(
                            "oauth broker slot %s: could not remove obsolete recovery marker",
                            self.alias,
                        )
                    preflight_marker_written = False
                    refresh_attempt_guarded = False
                    self._pending_stale_refresh_fingerprint = None
                self._adopt(new_grant, result="ok")
                self._persistence_degraded = not persisted
                self._next_persistence_retry_at = (
                    0.0
                    if persisted
                    else self._clock() + PERSISTENCE_RETRY_COOLDOWN_SECONDS
                )
                if persisted:
                    logger.info("oauth broker slot %s: refresh ok", self.alias)
                else:
                    logger.warning(
                        "oauth broker slot %s: refresh ok but keychain "
                        "persistence degraded; serving rotated grant from "
                        "memory and retrying on later requests",
                        self.alias,
                    )
                return new_grant.access_token
            except SlotRefreshError:
                raise
            except Exception as exc:
                category, terminal = self._classify(exc)
                if refresh_attempt_guarded and not terminal:
                    if self._definitive_no_rotation_response(exc):
                        if preflight_marker_written:
                            try:
                                await asyncio.to_thread(
                                    self._clear_recovery_marker
                                )
                                preflight_marker_written = False
                                self._pending_stale_refresh_fingerprint = None
                            except Exception:
                                category = "refresh_outcome_unknown"
                                terminal = True
                    else:
                        # No explicit token-endpoint response proves that the
                        # old refresh token survived. Reusing it could consume
                        # a single-use chain twice, so fail closed.
                        category = "refresh_outcome_unknown"
                        terminal = True
                self._last_refresh_at = self._clock()
                self._last_refresh_result = category
                if terminal:
                    self._terminal_category = category
                    logger.warning(
                        "oauth broker slot %s: terminal refresh failure (%s)",
                        self.alias,
                        category,
                    )
                else:
                    logger.warning(
                        "oauth broker slot %s: refresh failed (%s); retryable",
                        self.alias,
                        category,
                    )
                logger.debug(
                    "oauth broker slot %s: refresh error type %s",
                    self.alias,
                    type(exc).__name__,
                )
                # ``from None``: upstream error text may embed server-supplied
                # content; the normalized category is the whole story.
                raise SlotRefreshError(
                    alias=self.alias, category=category, terminal=terminal
                ) from None
        finally:
            self._release_file_lock(fd)

    @staticmethod
    def _refresh_token_fingerprint(refresh_token: str) -> str:
        return "sha256:" + hashlib.sha256(
            refresh_token.encode("utf-8")
        ).hexdigest()

    def _write_recovery_marker(self, stale_refresh_token: str) -> None:
        """Durably record a non-secret fingerprint before token consumption."""
        path = self._recovery_marker_path
        payload = json.dumps(
            {
                "version": RECOVERY_MARKER_VERSION,
                "state": "refresh_persistence_pending",
                "stale_refresh_fingerprint": self._refresh_token_fingerprint(
                    stale_refresh_token
                ),
            },
            sort_keys=True,
        ).encode("utf-8")
        directory_fd = _open_secure_directory(path.parent)
        temporary_name = f".{path.name}.{secrets.token_hex(8)}.tmp"
        marker_fd: Optional[int] = None
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            marker_fd = os.open(
                temporary_name,
                flags,
                0o600,
                dir_fd=directory_fd,
            )
            os.fchmod(marker_fd, 0o600)
            remaining = memoryview(payload)
            while remaining:
                written = os.write(marker_fd, remaining)
                if written <= 0:
                    raise OSError("short write while staging recovery marker")
                remaining = remaining[written:]
            os.fsync(marker_fd)
            os.close(marker_fd)
            marker_fd = None
            os.replace(
                temporary_name,
                path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            os.fsync(directory_fd)
        except BaseException:
            if marker_fd is not None:
                try:
                    os.close(marker_fd)
                except OSError:
                    pass
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except OSError:
                pass
            raise
        finally:
            os.close(directory_fd)

    def _recovery_marker_blocks(self, grant: OAuthGrant) -> bool:
        path = self._recovery_marker_path
        try:
            directory_fd = _open_secure_directory(path.parent)
        except OSError:
            return True
        try:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            try:
                marker_fd = os.open(path.name, flags, dir_fd=directory_fd)
            except FileNotFoundError:
                return False
            except OSError:
                return True
            try:
                marker_stat = os.fstat(marker_fd)
                if (
                    not stat.S_ISREG(marker_stat.st_mode)
                    or marker_stat.st_size > RECOVERY_MARKER_MAX_BYTES
                ):
                    return True
                chunks = []
                total = 0
                while total <= RECOVERY_MARKER_MAX_BYTES:
                    chunk = os.read(
                        marker_fd,
                        RECOVERY_MARKER_MAX_BYTES + 1 - total,
                    )
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                if total > RECOVERY_MARKER_MAX_BYTES:
                    return True
                raw = b"".join(chunks)
            finally:
                os.close(marker_fd)
        finally:
            os.close(directory_fd)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, ValueError):
            return True
        if not isinstance(payload, dict):
            return True
        expected = payload.get("stale_refresh_fingerprint")
        valid = (
            set(payload) == {
                "version",
                "state",
                "stale_refresh_fingerprint",
            }
            and payload.get("version") == RECOVERY_MARKER_VERSION
            and payload.get("state") == "refresh_persistence_pending"
            and isinstance(expected, str)
            and len(expected) == 71
            and expected.startswith("sha256:")
            and all(character in "0123456789abcdef" for character in expected[7:])
        )
        if not valid:
            return True
        if expected == self._refresh_token_fingerprint(grant.refresh_token):
            return True
        # A different Keychain token proves re-authentication or successful
        # persistence by another process. The stale marker is now obsolete.
        try:
            self._clear_recovery_marker()
        except OSError:
            logger.warning(
                "oauth broker slot %s: could not remove obsolete recovery marker",
                self.alias,
            )
        return False

    def _clear_recovery_marker(self) -> None:
        path = self._recovery_marker_path
        directory_fd = _open_secure_directory(path.parent)
        try:
            try:
                os.unlink(path.name, dir_fd=directory_fd)
            except FileNotFoundError:
                return
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        fd = _open_secure_directory(path)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _try_persist(self, grant: OAuthGrant) -> bool:
        for attempt in range(1, PERSISTENCE_RETRY_ATTEMPTS + 1):
            try:
                self._grant_store.replace(self.alias, grant)
                return True
            except Exception:
                logger.debug(
                    "oauth broker slot %s: keychain persistence attempt "
                    "%d/%d failed",
                    self.alias,
                    attempt,
                    PERSISTENCE_RETRY_ATTEMPTS,
                )
        return False

    async def _resolve_degraded_authority_locked(self) -> OAuthGrant:
        pending = self._cached
        stale_fingerprint = self._pending_stale_refresh_fingerprint
        if pending is None or stale_fingerprint is None:
            self._terminal_category = "persistence_recovery_required"
            raise SlotRefreshError(
                alias=self.alias,
                category="persistence_recovery_required",
                terminal=True,
            )
        try:
            current = await asyncio.to_thread(self._grant_store.load, self.alias)
        except GrantStoreError as exc:
            if exc.category == "not_found":
                try:
                    await asyncio.to_thread(self._clear_recovery_marker)
                except OSError:
                    logger.warning(
                        "oauth broker slot %s: could not remove obsolete recovery marker",
                        self.alias,
                    )
                self._cached = None
                self._generation += 1
                self._persistence_degraded = False
                self._next_persistence_retry_at = 0.0
                self._pending_stale_refresh_fingerprint = None
                self._terminal_category = None
                raise SlotRefreshError(
                    alias=self.alias,
                    category="not_found",
                    terminal=False,
                ) from None
            category = (
                exc.category
                if exc.category in SAFE_GRANT_STORE_ERROR_CATEGORIES
                else "grant_store_error"
            )
            raise SlotRefreshError(
                alias=self.alias, category=category, terminal=False
            ) from None
        except Exception:
            raise SlotRefreshError(
                alias=self.alias, category="grant_store_error", terminal=False
            ) from None

        if self._refresh_token_fingerprint(current.refresh_token) != stale_fingerprint:
            try:
                await asyncio.to_thread(self._clear_recovery_marker)
            except OSError:
                logger.warning(
                    "oauth broker slot %s: could not remove obsolete recovery marker",
                    self.alias,
                )
            self._adopt(current, result=None)
            self._persistence_degraded = False
            self._next_persistence_retry_at = 0.0
            self._pending_stale_refresh_fingerprint = None
            logger.info(
                "oauth broker slot %s: adopted externally updated keychain grant",
                self.alias,
            )
            return current
        return pending

    async def _retry_persistence_locked(self) -> None:
        """Run at most one bounded persistence batch per cooldown window."""
        grant = self._cached
        if grant is None:
            self._persistence_degraded = False
            self._next_persistence_retry_at = 0.0
            return
        if self._clock() < self._next_persistence_retry_at:
            return
        fd = await self._acquire_file_lock_async()
        try:
            # A human re-auth or another broker process may have replaced the
            # stale Keychain grant while this instance served its pending
            # in-memory grant. Compare against the in-memory pre-refresh
            # fingerprint even if another process already removed the shared
            # recovery marker.
            stale_fingerprint = self._pending_stale_refresh_fingerprint
            if stale_fingerprint is None:
                self._terminal_category = "persistence_recovery_required"
                raise SlotRefreshError(
                    alias=self.alias,
                    category="persistence_recovery_required",
                    terminal=True,
                )
            try:
                current = await asyncio.to_thread(self._grant_store.load, self.alias)
            except Exception:
                # Without a trustworthy current value, writing could overwrite
                # a concurrent re-auth. Keep serving the pending grant from
                # memory and retry after the cooldown.
                self._next_persistence_retry_at = (
                    self._clock() + PERSISTENCE_RETRY_COOLDOWN_SECONDS
                )
                return
            if self._refresh_token_fingerprint(current.refresh_token) != stale_fingerprint:
                try:
                    await asyncio.to_thread(self._clear_recovery_marker)
                except OSError:
                    logger.warning(
                        "oauth broker slot %s: could not remove obsolete recovery marker",
                        self.alias,
                    )
                self._adopt(current, result=None)
                self._persistence_degraded = False
                self._next_persistence_retry_at = 0.0
                self._pending_stale_refresh_fingerprint = None
                logger.info(
                    "oauth broker slot %s: adopted externally updated keychain grant",
                    self.alias,
                )
                return
            if await asyncio.to_thread(self._try_persist, grant):
                try:
                    await asyncio.to_thread(self._clear_recovery_marker)
                except OSError:
                    logger.warning(
                        "oauth broker slot %s: could not remove obsolete recovery marker",
                        self.alias,
                    )
                self._persistence_degraded = False
                self._next_persistence_retry_at = 0.0
                self._pending_stale_refresh_fingerprint = None
                logger.info(
                    "oauth broker slot %s: rotated grant persisted to "
                    "keychain after degradation",
                    self.alias,
                )
            else:
                self._next_persistence_retry_at = (
                    self._clock() + PERSISTENCE_RETRY_COOLDOWN_SECONDS
                )
        finally:
            self._release_file_lock(fd)

    def _adopt(self, grant: OAuthGrant, *, result: Optional[str]) -> None:
        self._cached = grant
        self._generation += 1
        if result is not None:
            self._last_refresh_at = self._clock()
            self._last_refresh_result = result

    def _merge(self, old: OAuthGrant, updated) -> OAuthGrant:
        if not isinstance(updated, dict):
            raise ValueError("refresh result must be a dict")
        access = str(updated.get("access_token") or "").strip()
        if not access:
            raise ValueError("refresh result missing access_token")
        refresh = str(updated.get("refresh_token") or "").strip()
        expires_at = updated.get("expires_at")
        if isinstance(expires_at, bool) or not isinstance(
            expires_at, (int, float)
        ):
            expires_at = _jwt_expiry(access)
        if expires_at is None:
            expires_at = self._clock() + DEFAULT_ACCESS_TOKEN_TTL_SECONDS
        return OAuthGrant(
            access_token=access,
            refresh_token=refresh or old.refresh_token,
            expires_at=float(expires_at),
            account_id=old.account_id,
        )

    @staticmethod
    def _definitive_no_rotation_response(exc: Exception) -> bool:
        code = getattr(exc, "code", None)
        return (
            isinstance(code, str)
            and code.strip() in DEFINITIVE_NO_ROTATION_ERROR_CODES
            and not bool(getattr(exc, "relogin_required", False))
        )

    def _classify(self, exc: Exception) -> tuple[str, bool]:
        raw_code = getattr(exc, "code", None)
        code = raw_code.strip() if isinstance(raw_code, str) else ""
        if code in TERMINAL_REFRESH_ERROR_CODES:
            return code, True
        if bool(getattr(exc, "relogin_required", False)):
            return "auth_relogin_required", True
        if isinstance(exc, GrantStoreError):
            category = str(exc.category or "").strip()
            if category in SAFE_GRANT_STORE_ERROR_CATEGORIES:
                return category, False
            return "grant_store_error", False
        if isinstance(exc, TimeoutError):
            return "timeout", False
        if code in DEFINITIVE_NO_ROTATION_ERROR_CODES:
            return code, False
        return "refresh_error", False

    async def _acquire_file_lock_async(self) -> int:
        cancel_event = threading.Event()
        worker = asyncio.create_task(
            asyncio.to_thread(self._acquire_file_lock, cancel_event)
        )
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancel_event.set()
            try:
                fd = await worker
            except _AccountLockAcquireCancelled:
                pass
            else:
                self._release_file_lock(fd)
            raise

    def _acquire_file_lock(
        self, cancel_event: Optional[threading.Event] = None
    ) -> int:
        return _acquire_account_lock_fd(
            self._state_dir, self.alias, cancel_event=cancel_event
        )

    def _release_file_lock(self, fd: int) -> None:
        _release_account_lock_fd(fd)


__all__ = [
    "DEFAULT_REFRESH_SKEW_SECONDS",
    "TERMINAL_REFRESH_ERROR_CODES",
    "AccountSlot",
    "SlotRefreshError",
    "account_process_lock",
]

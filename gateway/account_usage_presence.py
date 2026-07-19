"""Provider account usage rendered through messaging identity/presence surfaces.

Provider collection remains in :mod:`agent.account_usage`; this module owns only
poll scheduling, a crash-safe ownership journal, and platform fan-out.
"""

from __future__ import annotations

import asyncio
import contextvars
from dataclasses import dataclass, replace
from enum import Enum
import json
import logging
import math
import os
from pathlib import Path
import stat
import tempfile
import threading
import time
from typing import Any, Awaitable, Callable, Mapping, Optional, cast

from agent.account_usage import (
    AccountUsageFetchOutcome,
    AccountUsageSnapshot,
    fetch_account_usage_outcome,
    retry_after_seconds,
)
from gateway.config import AccountUsagePresenceConfig
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_STATE_VERSION = 3
_MAX_STATE_BYTES = 64 * 1024
_MAX_STATE_ENTRIES = 32
_STATE_PHASES = frozenset({"pending", "owned"})


@dataclass(frozen=True)
class AccountUsagePresenceCapabilities:
    """Identity surfaces an adapter can mutate at runtime."""

    display_name: bool = False
    activity: bool = False

    @property
    def any(self) -> bool:
        return bool(self.display_name or self.activity)


@dataclass(frozen=True)
class AccountUsagePresencePayload:
    """Provider-neutral value passed from the controller to platform adapters."""

    label: str
    remaining_percent: Optional[int]
    cached: bool = False

    @classmethod
    def unknown(cls) -> "AccountUsagePresencePayload":
        return cls(label="Usage", remaining_percent=None)


class AccountUsagePresenceRestoreResult(str, Enum):
    """Result of comparing a remote identity with one journaled ownership value."""

    RESTORED = "restored"
    ALREADY_BASELINE = "already_baseline"
    EXTERNAL = "external"
    RETRY = "retry"

    @property
    def can_retire(self) -> bool:
        return self is not AccountUsagePresenceRestoreResult.RETRY


class AccountUsagePresenceApplyResult(str, Enum):
    """CAS result for a persistent identity-surface generation update."""

    APPLIED = "applied"
    EXTERNAL = "external"
    RETRY = "retry"


@dataclass(frozen=True)
class _JournalEntry:
    baseline: dict[str, Any]
    owned: dict[str, Any]
    phase: str
    previous_owned: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "owned": self.owned,
            "phase": self.phase,
            "previous_owned": self.previous_owned,
        }


def account_usage_presence_state_path(home: Optional[Path] = None) -> Path:
    """Return the private journal path for one Hermes profile."""

    root = Path(home) if home is not None else get_hermes_home()
    return root / "state" / "account-usage-presence" / "journal.json"


def account_usage_presence_payload_from_snapshot(
    snapshot: AccountUsageSnapshot,
    *,
    window_label: Optional[str] = None,
) -> Optional[AccountUsagePresencePayload]:
    """Select one percentage window and convert it to a compact payload."""

    selected = None
    requested = str(window_label or "").strip().casefold()
    for window in snapshot.windows:
        if window.used_percent is None:
            continue
        if requested and str(window.label).strip().casefold() != requested:
            continue
        selected = window
        break

    if selected is None or selected.used_percent is None:
        return None
    used_value = float(selected.used_percent)
    if not math.isfinite(used_value):
        return None
    used_percent = max(0.0, min(100.0, used_value))
    return AccountUsagePresencePayload(
        label=selected.label,
        remaining_percent=int(round(100.0 - used_percent)),
    )


def _platform_name(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip().lower()


def _valid_state_mapping(value: Any) -> bool:
    if not isinstance(value, dict) or not value or len(value) > 16:
        return False
    for key, item in value.items():
        if not isinstance(key, str) or not key or len(key) > 128:
            return False
        if isinstance(item, str):
            if len(item) > 1024:
                return False
        elif item is None or isinstance(item, (bool, int)):
            pass
        elif isinstance(item, float):
            if not math.isfinite(item):
                return False
        else:
            return False
    return True


def _parse_journal(raw: Any) -> dict[str, _JournalEntry]:
    if not isinstance(raw, dict) or set(raw) != {"version", "entries"}:
        raise ValueError("journal must contain exactly version and entries")
    version = raw["version"]
    if version not in {2, _STATE_VERSION}:
        raise ValueError("unsupported journal version")
    entries = raw["entries"]
    if not isinstance(entries, dict) or len(entries) > _MAX_STATE_ENTRIES:
        raise ValueError("journal entries must be a bounded object")

    parsed: dict[str, _JournalEntry] = {}
    for key, value in entries.items():
        if not isinstance(key, str) or not key or len(key) > 256:
            raise ValueError("invalid journal state key")
        expected_fields = {"baseline", "owned", "phase"}
        if version == _STATE_VERSION:
            expected_fields.add("previous_owned")
        if not isinstance(value, dict) or set(value) != expected_fields:
            raise ValueError("invalid journal entry schema")
        if not _valid_state_mapping(value["baseline"]):
            raise ValueError("invalid journal baseline")
        if not _valid_state_mapping(value["owned"]):
            raise ValueError("invalid journal owned state")
        phase = value["phase"]
        if phase not in _STATE_PHASES:
            raise ValueError("invalid journal phase")
        previous_owned = value.get("previous_owned")
        if previous_owned is not None and not _valid_state_mapping(previous_owned):
            raise ValueError("invalid previous journal owned state")
        if phase == "owned" and previous_owned is not None:
            raise ValueError("owned journal entry cannot retain previous state")
        parsed[key] = _JournalEntry(
            baseline=dict(value["baseline"]),
            owned=dict(value["owned"]),
            phase=phase,
            previous_owned=(
                dict(previous_owned) if previous_owned is not None else None
            ),
        )
    return parsed


def _safe_regular_identity(path: Path) -> Optional[tuple[int, int]]:
    """Return a regular file identity, None if absent, and reject link tricks."""

    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("journal path is not a single-link regular file")
    if info.st_size > _MAX_STATE_BYTES:
        raise ValueError("journal exceeds size limit")
    return (info.st_dev, info.st_ino)


def _read_state_no_follow(path: Path) -> bytes:
    expected = _safe_regular_identity(path)
    if expected is None:
        raise FileNotFoundError(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != expected
            or opened.st_size > _MAX_STATE_BYTES
        ):
            raise ValueError("journal changed or is unsafe while opening")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            data = handle.read(_MAX_STATE_BYTES + 1)
        if len(data) > _MAX_STATE_BYTES:
            raise ValueError("journal exceeds size limit")
        return data
    finally:
        os.close(fd)


def _write_state_atomically(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace a private journal without following its target."""

    parent = path.parent
    parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    parent_info = os.lstat(parent)
    if not stat.S_ISDIR(parent_info.st_mode) or stat.S_ISLNK(parent_info.st_mode):
        raise ValueError("journal directory is not a real directory")
    os.chmod(parent, 0o700)

    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(encoded) > _MAX_STATE_BYTES:
        raise ValueError("journal exceeds size limit")

    initial_identity = _safe_regular_identity(path)
    fd, temporary_name = tempfile.mkstemp(prefix=".journal-", dir=parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.close(fd)
        fd = -1

        if _safe_regular_identity(path) != initial_identity:
            raise RuntimeError("journal target changed during atomic write")
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        try:
            directory_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class AccountUsagePresenceController:
    """Fetch one provider snapshot and fan it out to selected adapters."""

    def __init__(
        self,
        config: AccountUsagePresenceConfig,
        adapters: Callable[[], Mapping[Any, Any]],
        *,
        fetcher: Optional[Callable[[str], Any]] = None,
        state_path: Optional[Path] = None,
        monotonic: Callable[[], float] = time.monotonic,
        fetch_timeout_seconds: float = 20.0,
        adapter_timeout_seconds: float = 15.0,
        recovery_interval_seconds: float = 30.0,
    ) -> None:
        self.config = config
        self._adapters = adapters
        self._fetcher = fetcher or fetch_account_usage_outcome
        self._state_path = state_path or account_usage_presence_state_path()
        self._monotonic = monotonic
        self._fetch_timeout_seconds = max(0.01, float(fetch_timeout_seconds))
        self._adapter_timeout_seconds = max(0.01, float(adapter_timeout_seconds))
        self._recovery_interval_seconds = max(0.01, float(recovery_interval_seconds))
        self._task: Optional[asyncio.Task] = None
        self._fetch_thread: Optional[threading.Thread] = None
        self._state_loaded = False
        self._journal_blocked = False
        self._journal: dict[str, _JournalEntry] = {}
        self._owned_in_process: set[str] = set()
        self._last_payload: Optional[AccountUsagePresencePayload] = None
        self._last_success_at: Optional[float] = None
        self._last_applied: dict[str, tuple[int, AccountUsagePresencePayload]] = {}
        self._adapter_retry_until: dict[str, float] = {}
        self._provider_retry_until = 0.0

    @property
    def task(self) -> Optional[asyncio.Task]:
        return self._task

    @property
    def recovery_pending(self) -> bool:
        self._load_state()
        return bool(self._journal) or self._journal_blocked

    async def start(self) -> None:
        """Recover prior ownership, then poll only when explicitly configured."""

        self._load_state()
        if self._journal_blocked:
            return
        if self._journal:
            await self.recover_saved_baselines()

        if not self.config.is_configured:
            if self._journal and (self._task is None or self._task.done()):
                self._task = asyncio.create_task(
                    self._run_recovery(),
                    name="account-usage-presence-recovery",
                )
            return

        if self._task is None or self._task.done():
            self._task = asyncio.create_task(
                self._run(),
                name="account-usage-presence",
            )

    async def stop(self) -> None:
        """Cancel refreshes and CAS-restore persistent identity surfaces."""

        task = self._task
        self._task = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self.recover_saved_baselines()

    async def recover_saved_baselines(self) -> None:
        """Retry pending recovery; safe to call after an adapter reconnect."""

        await self._restore_saved_baselines()

    async def _run(self) -> None:
        try:
            while True:
                await self.refresh_once()
                await asyncio.sleep(self.config.update_interval_seconds)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Account-usage presence loop stopped unexpectedly", exc_info=True)

    async def _run_recovery(self) -> None:
        try:
            while self._journal and not self._journal_blocked:
                await self._restore_saved_baselines()
                if not self._journal:
                    return
                await asyncio.sleep(self._recovery_interval_seconds)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Account-usage presence recovery loop stopped", exc_info=True)

    @staticmethod
    def _publish_fetch_outcome(
        future: asyncio.Future,
        outcome: tuple[bool, Any],
    ) -> None:
        if not future.done():
            future.set_result(outcome)

    async def _fetch_outcome(self, provider: str) -> AccountUsageFetchOutcome:
        """Run the synchronous collector in one abandonable daemon worker."""

        running = self._fetch_thread
        if running is not None and running.is_alive():
            raise TimeoutError("previous provider account-usage fetch is still running")

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        context = contextvars.copy_context()

        def collect() -> None:
            try:
                outcome: tuple[bool, Any] = (
                    True,
                    context.run(self._fetcher, provider),
                )
            except Exception as exc:
                outcome = (False, exc)
            try:
                loop.call_soon_threadsafe(self._publish_fetch_outcome, future, outcome)
            except RuntimeError:
                pass

        worker = threading.Thread(
            target=collect,
            name="hermes-account-usage-presence-fetch",
            daemon=True,
        )
        self._fetch_thread = worker
        try:
            worker.start()
        except Exception:
            self._fetch_thread = None
            raise

        try:
            succeeded, value = await asyncio.wait_for(
                asyncio.shield(future),
                timeout=self._fetch_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError("provider account-usage fetch timed out") from exc

        if not succeeded:
            if isinstance(value, BaseException):
                return AccountUsageFetchOutcome(
                    retry_after_seconds=retry_after_seconds(value),
                    failed=True,
                )
            return AccountUsageFetchOutcome(failed=True)
        if isinstance(value, AccountUsageFetchOutcome):
            return value
        if value is None or isinstance(value, AccountUsageSnapshot):
            return AccountUsageFetchOutcome(snapshot=value)
        raise TypeError("account-usage fetcher returned an unsupported value")

    async def refresh_once(self) -> None:
        """Fetch and apply one snapshot; public for lifecycle tests."""

        if not self.config.is_configured:
            return
        self._load_state()
        if self._journal_blocked:
            return

        provider = self.config.provider
        if not provider:
            return
        now = self._monotonic()
        if now < self._provider_retry_until:
            outcome = AccountUsageFetchOutcome(failed=True)
        else:
            try:
                outcome = await self._fetch_outcome(provider)
            except Exception:
                logger.warning(
                    "Account-usage presence fetch failed for provider %s",
                    provider,
                    exc_info=True,
                )
                outcome = AccountUsageFetchOutcome(failed=True)
            if outcome.retry_after_seconds is not None:
                self._provider_retry_until = now + outcome.retry_after_seconds
                logger.warning(
                    "Account-usage provider %s rate-limited; retrying in %.0fs",
                    provider,
                    outcome.retry_after_seconds,
                )

        live_payload = (
            account_usage_presence_payload_from_snapshot(
                outcome.snapshot,
                window_label=self.config.window_label,
            )
            if outcome.snapshot is not None
            else None
        )
        if live_payload is not None:
            self._last_payload = live_payload
            self._last_success_at = now
            payload = live_payload
        elif (
            self._last_payload is not None
            and self._last_success_at is not None
            and now - self._last_success_at < self.config.stale_after_seconds
        ):
            payload = replace(self._last_payload, cached=True)
        else:
            payload = AccountUsagePresencePayload.unknown()

        await self._apply_to_selected_adapters(payload, now=now)

    def _iter_adapters(self):
        for key, adapter in list(self._adapters().items()):
            yield _platform_name(key), adapter

    @staticmethod
    def _state_key(platform: str, adapter: Any) -> str:
        method = getattr(adapter, "account_usage_presence_state_key", None)
        if callable(method):
            try:
                value = str(method() or "").strip()
                if value:
                    return value
            except Exception:
                logger.debug(
                    "Account-usage presence state-key lookup failed for %s",
                    platform,
                    exc_info=True,
                )
        return platform

    async def _await_adapter(self, result: Awaitable[Any]) -> Any:
        return await asyncio.wait_for(result, timeout=self._adapter_timeout_seconds)

    async def _apply_to_selected_adapters(
        self,
        payload: AccountUsagePresencePayload,
        *,
        now: float,
    ) -> None:
        selected = set(self.config.platforms)
        for platform, adapter in self._iter_adapters():
            if platform not in selected:
                continue
            capabilities = getattr(
                adapter,
                "account_usage_presence_capabilities",
                AccountUsagePresenceCapabilities(),
            )
            if not isinstance(capabilities, AccountUsagePresenceCapabilities):
                logger.warning(
                    "Ignoring invalid account-usage presence capabilities from %s",
                    platform,
                )
                continue
            if not capabilities.any:
                logger.info(
                    "Account-usage presence unsupported by platform %s; skipping",
                    platform,
                )
                continue

            state_key = self._state_key(platform, adapter)
            if now < self._adapter_retry_until.get(state_key, 0.0):
                continue
            if self._last_applied.get(state_key) == (id(adapter), payload):
                continue
            entry = self._journal.get(state_key)
            if entry is not None and entry.phase == "pending":
                # A pending transition may have changed the remote value even
                # when its API call raised. Fail closed until stop/restart
                # recovery reconciles the journaled generations.
                continue
            if entry is not None and state_key not in self._owned_in_process:
                continue

            prior_owned_entry = (
                entry if entry is not None and entry.phase == "owned" else None
            )
            baseline = entry.baseline if entry is not None else None
            if entry is None:
                capture = getattr(adapter, "capture_account_usage_presence_baseline", None)
                if callable(capture):
                    try:
                        captured = await self._await_adapter(cast(Awaitable[Any], capture()))
                    except Exception:
                        logger.warning(
                            "Could not capture %s account-usage presence baseline",
                            platform,
                            exc_info=True,
                        )
                        continue
                    if captured is not None:
                        if not _valid_state_mapping(captured):
                            logger.warning(
                                "Ignoring invalid %s account-usage presence baseline",
                                platform,
                            )
                            continue
                        baseline = dict(captured)

            pending_entry: Optional[_JournalEntry] = None
            guarded_apply: Optional[Callable[..., Awaitable[Any]]] = None
            if baseline is not None:
                build_owned = getattr(
                    adapter,
                    "build_account_usage_presence_owned_state",
                    None,
                )
                if not callable(build_owned):
                    logger.warning(
                        "Persistent account-usage surface %s has no ownership renderer",
                        platform,
                    )
                    continue
                guarded_apply = getattr(
                    adapter,
                    "apply_account_usage_presence_if_owned",
                    None,
                )
                if not callable(guarded_apply):
                    logger.warning(
                        "Persistent account-usage surface %s has no guarded ownership updater",
                        platform,
                    )
                    continue
                try:
                    owned = build_owned(payload, baseline)
                except Exception:
                    logger.warning(
                        "Could not render %s account-usage ownership state",
                        platform,
                        exc_info=True,
                    )
                    continue
                if not _valid_state_mapping(owned):
                    logger.warning(
                        "Ignoring invalid %s account-usage ownership state",
                        platform,
                    )
                    continue
                pending_entry = _JournalEntry(
                    baseline=dict(baseline),
                    owned=dict(owned),
                    phase="pending",
                    previous_owned=(
                        dict(prior_owned_entry.owned)
                        if prior_owned_entry is not None
                        else None
                    ),
                )
                candidate = dict(self._journal)
                candidate[state_key] = pending_entry
                if not self._persist_journal(candidate):
                    continue
                self._journal = candidate

            apply = getattr(adapter, "apply_account_usage_presence", None)
            if not callable(apply):
                continue
            try:
                if baseline is not None:
                    assert guarded_apply is not None
                    expected_owned = (
                        prior_owned_entry.owned
                        if prior_owned_entry is not None
                        else baseline
                    )
                    raw_apply_result = await self._await_adapter(
                        cast(
                            Awaitable[Any],
                            guarded_apply(payload, baseline, expected_owned),
                        )
                    )
                    try:
                        apply_result = AccountUsagePresenceApplyResult(
                            raw_apply_result
                        )
                    except (TypeError, ValueError):
                        logger.warning(
                            "Ignoring invalid guarded account-usage apply result from %s",
                            platform,
                        )
                        apply_result = AccountUsagePresenceApplyResult.RETRY
                else:
                    changed = await self._await_adapter(
                        cast(Awaitable[Any], apply(payload, baseline))
                    )
                    apply_result = (
                        AccountUsagePresenceApplyResult.APPLIED
                        if changed
                        else AccountUsagePresenceApplyResult.RETRY
                    )
            except Exception as exc:
                retry_after = retry_after_seconds(exc)
                if retry_after is not None:
                    self._adapter_retry_until[state_key] = now + retry_after
                    logger.warning(
                        "Account-usage presence rate-limited by %s; retrying in %.0fs",
                        platform,
                        retry_after,
                    )
                else:
                    logger.warning(
                        "Account-usage presence update failed for %s",
                        platform,
                        exc_info=True,
                    )
                continue

            if apply_result is AccountUsagePresenceApplyResult.EXTERNAL:
                candidate = dict(self._journal)
                candidate.pop(state_key, None)
                if self._persist_journal(candidate):
                    self._journal = candidate
                    self._owned_in_process.discard(state_key)
                    self._last_applied.pop(state_key, None)
                logger.info(
                    "Account-usage presence no longer owns %s; preserving external value",
                    state_key,
                )
                continue

            if apply_result is not AccountUsagePresenceApplyResult.APPLIED:
                if pending_entry is not None:
                    candidate = dict(self._journal)
                    if prior_owned_entry is None:
                        candidate.pop(state_key, None)
                    else:
                        candidate[state_key] = prior_owned_entry
                    if self._persist_journal(candidate):
                        self._journal = candidate
                continue

            if pending_entry is not None:
                self._owned_in_process.add(state_key)
                candidate = dict(self._journal)
                candidate[state_key] = replace(
                    pending_entry,
                    phase="owned",
                    previous_owned=None,
                )
                if self._persist_journal(candidate):
                    self._journal = candidate
            self._last_applied[state_key] = (id(adapter), payload)

    async def _restore_one(
        self,
        platform: str,
        adapter: Any,
        entry: _JournalEntry,
    ) -> AccountUsagePresenceRestoreResult:
        restore = getattr(adapter, "restore_account_usage_presence", None)
        if not callable(restore):
            return AccountUsagePresenceRestoreResult.RETRY
        owned_candidates = [entry.owned]
        if (
            entry.phase == "pending"
            and entry.previous_owned is not None
            and entry.previous_owned != entry.owned
        ):
            # A crash or journal-write failure can leave a transition pending
            # while the remote surface is either the new desired value or the
            # previously owned value. Both are safe CAS expectations for
            # restoring the captured baseline.
            owned_candidates.append(entry.previous_owned)

        for owned in owned_candidates:
            try:
                value = await self._await_adapter(
                    cast(Awaitable[Any], restore(entry.baseline, owned))
                )
            except Exception:
                logger.warning(
                    "Account-usage presence restore failed for %s",
                    platform,
                    exc_info=True,
                )
                return AccountUsagePresenceRestoreResult.RETRY
            try:
                result = AccountUsagePresenceRestoreResult(value)
            except (TypeError, ValueError):
                logger.warning(
                    "Ignoring invalid account-usage restore result from %s",
                    platform,
                )
                return AccountUsagePresenceRestoreResult.RETRY
            if result is not AccountUsagePresenceRestoreResult.EXTERNAL:
                return result
        return AccountUsagePresenceRestoreResult.EXTERNAL

    async def _restore_saved_baselines(self) -> None:
        self._load_state()
        if self._journal_blocked or not self._journal:
            return

        connected: dict[str, tuple[str, Any]] = {}
        for platform, adapter in self._iter_adapters():
            connected[self._state_key(platform, adapter)] = (platform, adapter)

        jobs: dict[str, asyncio.Task] = {}
        for state_key, entry in self._journal.items():
            connected_entry = connected.get(state_key)
            if connected_entry is None:
                continue
            platform, adapter = connected_entry
            jobs[state_key] = asyncio.create_task(
                self._restore_one(platform, adapter, entry)
            )
        if not jobs:
            return

        results = await asyncio.gather(*jobs.values())
        retired = {
            state_key
            for state_key, result in zip(jobs, results)
            if result.can_retire
        }
        for state_key, result in zip(jobs, results):
            if result is AccountUsagePresenceRestoreResult.EXTERNAL:
                logger.info(
                    "Account-usage presence no longer owns %s; preserving external value",
                    state_key,
                )
        if not retired:
            return

        candidate = {
            key: value for key, value in self._journal.items() if key not in retired
        }
        if not self._persist_journal(candidate):
            return
        self._journal = candidate
        self._owned_in_process.difference_update(retired)
        for key in retired:
            self._last_applied.pop(key, None)
            self._adapter_retry_until.pop(key, None)

    def _load_state(self) -> None:
        if self._state_loaded:
            return
        self._state_loaded = True
        try:
            data = _read_state_no_follow(self._state_path)
            raw = json.loads(data.decode("utf-8"))
            self._journal = _parse_journal(raw)
        except FileNotFoundError:
            return
        except Exception:
            self._journal_blocked = True
            logger.error(
                "Unsafe or invalid account-usage presence journal at %s; "
                "all identity mutation is disabled until an operator inspects it",
                self._state_path,
                exc_info=True,
            )

    def _persist_journal(self, entries: Mapping[str, _JournalEntry]) -> bool:
        if self._journal_blocked:
            return False
        value = {
            "version": _STATE_VERSION,
            "entries": {
                key: entry.to_dict() for key, entry in sorted(entries.items())
            },
        }
        try:
            _write_state_atomically(self._state_path, value)
            return True
        except Exception:
            self._journal_blocked = True
            logger.error(
                "Could not safely persist account-usage presence journal at %s; "
                "further identity mutation is disabled",
                self._state_path,
                exc_info=True,
            )
            return False

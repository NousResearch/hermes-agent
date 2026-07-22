"""Host-global cross-process concurrency cap for ``claude -p`` spawns.

Hermes profiles/gateways run as separate processes and this host may also run
other Claude Code consumers on the same Max login (for example OpenClaw or
other local Claude Code sessions). Hermes can only cap **its own** concurrent
``claude -p`` children — leave headroom for the rest of the host.

Mechanism mirrors ``hermes_cli.active_sessions`` (JSON registry + advisory
``fcntl``/``msvcrt`` file lock) but:

  * Lives under the **host-global** shared root
    (``~/.hermes/shared/claude_cli_slots/`` via
    :func:`hermes_constants.get_default_hermes_root`), not per-profile
    ``HERMES_HOME``, so every Hermes profile/process shares one pool.
  * **Waits** (bounded) when saturated, then raises
    :class:`ClaudeCliConcurrencyError` so the conversation loop can
    activate the profile's fallback chain (grok/gpt) instead of hanging.

Config (user-facing ``config.yaml``)::

    model:
      claude_cli:
        max_concurrent: 3                 # default 3; 0/null = unbounded
        acquire_timeout_seconds: 45       # wait then fall back

Internal env bridge (not user-facing docs; tests / emergency override)::

    HERMES_CLAUDE_CLI_MAX_CONCURRENT
    HERMES_CLAUDE_CLI_ACQUIRE_TIMEOUT
    HERMES_CLAUDE_CLI_SLOT_DIR   # redirect store (required under pytest)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

logger = logging.getLogger(__name__)

# Defaults — leave headroom on a Max login shared with other Claude Code consumers.
DEFAULT_MAX_CONCURRENT = 3
DEFAULT_ACQUIRE_TIMEOUT_SECONDS = 45.0
_POLL_INTERVAL_SECONDS = 0.15
_STALE_MAX_AGE_SECONDS = 6 * 3600  # hard reap even if pid looks alive

# Slot registry filename under the shared dir.
_STATE_FILENAME = "slots.json"
_LOCK_FILENAME = "slots.lock"

# Sentinel so callers can pass max_concurrent=None to mean "unbounded"
# without it being confused with "use config default".
_MISSING: Any = object()


# ---------------------------------------------------------------------------
# Config resolution (config.yaml user-facing; env = internal bridge)
# ---------------------------------------------------------------------------


def _coerce_positive_int(value: Any, *, default: int, key: str) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        logger.warning("Ignoring invalid %s=%r (expected positive int)", key, value)
        return default
    try:
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError(value)
            parsed = int(value)
        elif isinstance(value, str):
            parsed = int(value.strip(), 10)
        else:
            parsed = int(value)
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid %s=%r (expected positive int)", key, value)
        return default
    return parsed


def _coerce_nonneg_float(value: Any, *, default: float, key: str) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid %s=%r (expected number)", key, value)
        return default
    if parsed < 0:
        return default
    return parsed


def _load_model_claude_cli_cfg() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
        model = cfg.get("model")
        if not isinstance(model, dict):
            return {}
        section = model.get("claude_cli")
        return section if isinstance(section, dict) else {}
    except Exception:
        logger.debug("claude_cli concurrency: config load failed", exc_info=True)
        return {}


def resolve_claude_cli_max_concurrent(
    override: Optional[int] = None,
) -> Optional[int]:
    """Return the concurrent ``claude -p`` cap, or None when unbounded.

    Priority: explicit override → ``HERMES_CLAUDE_CLI_MAX_CONCURRENT`` →
    ``model.claude_cli.max_concurrent`` → default 3.

    ``0`` / negative in config or env means unbounded (disabled).
    """
    if override is not None:
        n = int(override)
        return None if n <= 0 else n

    env_raw = (os.environ.get("HERMES_CLAUDE_CLI_MAX_CONCURRENT") or "").strip()
    if env_raw:
        try:
            n = int(env_raw, 10)
            return None if n <= 0 else n
        except ValueError:
            logger.warning(
                "Ignoring invalid HERMES_CLAUDE_CLI_MAX_CONCURRENT=%r", env_raw
            )

    section = _load_model_claude_cli_cfg()
    if "max_concurrent" in section:
        raw = section.get("max_concurrent")
        if raw is None:
            return DEFAULT_MAX_CONCURRENT
        n = _coerce_positive_int(
            raw, default=DEFAULT_MAX_CONCURRENT, key="model.claude_cli.max_concurrent"
        )
        # 0 from user means unbounded; _coerce_positive_int would return
        # default for invalid, so handle 0 explicitly.
        try:
            parsed = int(raw)
            if parsed <= 0:
                return None
            return parsed
        except (TypeError, ValueError):
            return n if n > 0 else DEFAULT_MAX_CONCURRENT

    return DEFAULT_MAX_CONCURRENT


def resolve_claude_cli_acquire_timeout(
    override: Optional[float] = None,
) -> float:
    """Seconds to wait for a free slot before raising concurrency error."""
    if override is not None:
        return max(0.0, float(override))

    env_raw = (os.environ.get("HERMES_CLAUDE_CLI_ACQUIRE_TIMEOUT") or "").strip()
    if env_raw:
        try:
            return max(0.0, float(env_raw))
        except ValueError:
            logger.warning(
                "Ignoring invalid HERMES_CLAUDE_CLI_ACQUIRE_TIMEOUT=%r", env_raw
            )

    section = _load_model_claude_cli_cfg()
    if "acquire_timeout_seconds" in section:
        return _coerce_nonneg_float(
            section.get("acquire_timeout_seconds"),
            default=DEFAULT_ACQUIRE_TIMEOUT_SECONDS,
            key="model.claude_cli.acquire_timeout_seconds",
        )
    return DEFAULT_ACQUIRE_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# Shared store path (host-global, not per-profile)
# ---------------------------------------------------------------------------


def _slot_dir() -> Path:
    """Directory for the host-global claude_cli slot registry.

    Override via ``HERMES_CLAUDE_CLI_SLOT_DIR`` (required under pytest so
    tests never touch the real ``~/.hermes/shared/`` store).
    """
    override = (os.environ.get("HERMES_CLAUDE_CLI_SLOT_DIR") or "").strip()
    if override:
        return Path(override).expanduser()

    from hermes_constants import get_default_hermes_root

    path = get_default_hermes_root() / "shared" / "claude_cli_slots"

    # Seat belt: refuse to touch the real user store during tests.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        raise RuntimeError(
            "Refusing to touch real claude_cli slot store during tests: "
            f"{path}. Set HERMES_CLAUDE_CLI_SLOT_DIR to a tmp_path."
        )
    return path


def _state_path() -> Path:
    return _slot_dir() / _STATE_FILENAME


def _lock_path() -> Path:
    return _slot_dir() / _LOCK_FILENAME


# ---------------------------------------------------------------------------
# File lock (same style as hermes_cli.active_sessions / auth._file_lock)
# ---------------------------------------------------------------------------


class _FileLock:
    """Cross-process exclusive advisory lock on a companion ``.lock`` file."""

    def __init__(self, path: Path):
        self.path = path
        self._fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+b")
        if os.name == "nt":
            try:
                import msvcrt

                # msvcrt.locking needs at least 1 byte of content.
                if self._fh.seek(0, os.SEEK_END) == 0:
                    self._fh.write(b" ")
                    self._fh.flush()
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_LOCK, 1)
            except Exception as exc:
                self._fh.close()
                self._fh = None
                raise RuntimeError("claude_cli slot file lock unavailable") from exc
        else:
            try:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
            except Exception as exc:
                self._fh.close()
                self._fh = None
                raise RuntimeError("claude_cli slot file lock unavailable") from exc
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fh is None:
            return
        if os.name == "nt":
            try:
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
        else:
            try:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        try:
            self._fh.close()
        finally:
            self._fh = None


# ---------------------------------------------------------------------------
# PID liveness + stale reaping
# ---------------------------------------------------------------------------


def _process_start_time(pid: int) -> Optional[float]:
    try:
        import psutil  # type: ignore

        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pid_alive(pid: Any, process_start_time: Any = None) -> bool:
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    try:
        from gateway.status import _pid_exists

        exists = bool(_pid_exists(pid_int))
    except Exception:
        # Conservative: if we cannot check, treat as alive so we don't
        # over-admit under a broken psutil import.
        return True
    if not exists:
        return False
    expected_start = _optional_float(process_start_time)
    if expected_start is None:
        return True
    current_start = _process_start_time(pid_int)
    if current_start is None:
        return True
    return abs(current_start - expected_start) < 0.001


def _read_entries(path: Path) -> list[dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return []
    except Exception:
        logger.warning("Ignoring corrupt claude_cli slot registry at %s", path)
        return []
    entries = data.get("entries") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return []
    return [e for e in entries if isinstance(e, dict)]


def _write_entries(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"entries": entries}, fh, sort_keys=True)
    os.replace(tmp, path)


def _prune_stale(
    entries: list[dict[str, Any]], *, now: Optional[float] = None
) -> list[dict[str, Any]]:
    """Drop slots whose holder PID is dead or whose age exceeds the hard cap."""
    now = time.time() if now is None else now
    kept: list[dict[str, Any]] = []
    for entry in entries:
        if not _pid_alive(entry.get("pid"), entry.get("process_start_time")):
            continue
        started = _optional_float(entry.get("started_at")) or 0.0
        if started and (now - started) > _STALE_MAX_AGE_SECONDS:
            continue
        kept.append(entry)
    return kept


# ---------------------------------------------------------------------------
# Public lease API
# ---------------------------------------------------------------------------


@dataclass
class ClaudeCliSlotLease:
    """Held slot for one in-flight ``claude -p`` turn."""

    lease_id: str
    enabled: bool = True
    released: bool = False
    max_concurrent: Optional[int] = None

    def release(self) -> None:
        if self.released or not self.enabled:
            self.released = True
            return
        release_claude_cli_slot(self)


def _resolve_cap_arg(max_concurrent: Any) -> Optional[int]:
    """Interpret an explicit max_concurrent arg or fall back to config.

    * ``_MISSING`` → read config/env/default
    * ``None`` or ``<= 0`` → unbounded
    * positive int → that cap
    """
    if max_concurrent is _MISSING:
        return resolve_claude_cli_max_concurrent()
    if max_concurrent is None:
        return None
    try:
        n = int(max_concurrent)
    except (TypeError, ValueError):
        return resolve_claude_cli_max_concurrent()
    return None if n <= 0 else n


def try_acquire_claude_cli_slot(
    *,
    max_concurrent: Any = _MISSING,
    metadata: Optional[dict[str, Any]] = None,
) -> Tuple[Optional[ClaudeCliSlotLease], Optional[str]]:
    """Non-blocking attempt to claim one host-global claude_cli slot.

    Returns ``(lease, None)`` on success, ``(None, reason)`` when full.
    When the cap is disabled (None), returns a no-op lease.

    Pass ``max_concurrent=None`` (or ``0``) for unbounded; omit the arg to
    resolve from config/env/default.
    """
    cap = _resolve_cap_arg(max_concurrent)
    lease_id = uuid.uuid4().hex
    if cap is None:
        return (
            ClaudeCliSlotLease(
                lease_id=lease_id, enabled=False, max_concurrent=None
            ),
            None,
        )

    now = time.time()
    entry: dict[str, Any] = {
        "lease_id": lease_id,
        "pid": os.getpid(),
        "process_start_time": _process_start_time(os.getpid()),
        "started_at": now,
        "updated_at": now,
    }
    if metadata:
        entry["metadata"] = {
            str(k): v for k, v in metadata.items() if isinstance(k, str)
        }

    state_path = _state_path()
    with _FileLock(_lock_path()):
        raw = _read_entries(state_path)
        entries = _prune_stale(raw, now=now)
        pruned = len(raw) - len(entries)
        if pruned:
            logger.info("claude_cli slots: pruned %d stale lease(s)", pruned)
        active = len(entries)
        if active >= cap:
            _write_entries(state_path, entries)
            reason = (
                f"claude_cli concurrency cap reached "
                f"({active}/{cap} host-wide Hermes sessions). "
                f"Wait for a free slot or raise model.claude_cli.max_concurrent."
            )
            logger.info(
                "claude_cli slot full: active=%d max=%d", active, cap
            )
            return None, reason
        entries.append(entry)
        _write_entries(state_path, entries)

    return (
        ClaudeCliSlotLease(
            lease_id=lease_id, enabled=True, max_concurrent=cap
        ),
        None,
    )


def release_claude_cli_slot(lease: ClaudeCliSlotLease) -> None:
    """Release a previously acquired slot. Idempotent."""
    if lease.released:
        return
    if not lease.enabled:
        lease.released = True
        return
    state_path = _state_path()
    try:
        with _FileLock(_lock_path()):
            entries = _prune_stale(_read_entries(state_path))
            kept = [
                e
                for e in entries
                if str(e.get("lease_id") or "") != lease.lease_id
            ]
            if len(kept) != len(entries):
                _write_entries(state_path, kept)
    except Exception:
        logger.debug("claude_cli slot release failed", exc_info=True)
    finally:
        lease.released = True


def acquire_claude_cli_slot(
    *,
    max_concurrent: Any = _MISSING,
    timeout_seconds: Optional[float] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> ClaudeCliSlotLease:
    """Block until a slot is free or *timeout_seconds* elapses.

    On timeout raises :class:`agent.transports.claude_cli.ClaudeCliConcurrencyError`
    (classifiable so Hermes' fallback chain can take over).
    """
    from agent.transports.claude_cli import ClaudeCliConcurrencyError

    cap = _resolve_cap_arg(max_concurrent)
    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else resolve_claude_cli_acquire_timeout()
    )
    if cap is None:
        return ClaudeCliSlotLease(
            lease_id=uuid.uuid4().hex, enabled=False, max_concurrent=None
        )

    deadline = time.monotonic() + max(0.0, float(timeout))
    last_reason: Optional[str] = None
    while True:
        lease, reason = try_acquire_claude_cli_slot(
            max_concurrent=cap, metadata=metadata
        )
        if lease is not None:
            if lease.enabled:
                logger.debug(
                    "claude_cli slot acquired lease=%s max=%s",
                    lease.lease_id[:8],
                    cap,
                )
            return lease
        last_reason = reason
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(_POLL_INTERVAL_SECONDS, max(remaining, 0.01)))

    msg = last_reason or (
        f"claude_cli concurrency cap reached (max={cap}). "
        "No free slot within acquire timeout."
    )
    raise ClaudeCliConcurrencyError(
        message=(
            f"{msg} Timed out after {timeout:.1f}s waiting for a free "
            f"claude -p slot. Hermes will try the profile fallback if configured."
        ),
        max_concurrent=cap,
        timeout_seconds=float(timeout),
    )


@contextmanager
def claude_cli_slot(
    *,
    max_concurrent: Any = _MISSING,
    timeout_seconds: Optional[float] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Iterator[ClaudeCliSlotLease]:
    """Context manager: acquire a host-global slot for the duration of a turn.

    Releases on normal exit, exception, and is safe if the holder crashes
    (stale PID reaping on next acquire).
    """
    lease = acquire_claude_cli_slot(
        max_concurrent=max_concurrent,
        timeout_seconds=timeout_seconds,
        metadata=metadata,
    )
    try:
        yield lease
    finally:
        lease.release()


def claude_cli_slot_registry_snapshot() -> list[dict[str, Any]]:
    """Pruned slot registry for diagnostics/tests."""
    state_path = _state_path()
    with _FileLock(_lock_path()):
        entries = _prune_stale(_read_entries(state_path))
        _write_entries(state_path, entries)
        return list(entries)


# Process-local guard so tests can force unbounded without config.
_force_unbounded = threading.local()


def force_unbounded_for_tests(active: bool = True) -> None:
    """Test helper: skip the host cap for the current thread."""
    _force_unbounded.active = active


def is_force_unbounded() -> bool:
    return bool(getattr(_force_unbounded, "active", False))


__all__ = [
    "DEFAULT_MAX_CONCURRENT",
    "DEFAULT_ACQUIRE_TIMEOUT_SECONDS",
    "ClaudeCliSlotLease",
    "resolve_claude_cli_max_concurrent",
    "resolve_claude_cli_acquire_timeout",
    "try_acquire_claude_cli_slot",
    "acquire_claude_cli_slot",
    "release_claude_cli_slot",
    "claude_cli_slot",
    "claude_cli_slot_registry_snapshot",
    "force_unbounded_for_tests",
    "is_force_unbounded",
]

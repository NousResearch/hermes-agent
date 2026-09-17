"""Crash-aware macOS Power Protect ownership.

``pmset -a disablesleep`` writes a persistent, machine-wide setting.  This
module adds an ownership journal around that setting so concurrent Hermes
processes share one lease and a later Hermes launch can repair a stale lease
left by a crash, SIGKILL, or power loss.  It never changes a SleepDisabled=1
setting unless a Hermes journal proves Hermes owns it.

The normal turn-scoped inhibitor remains ``caffeinate`` and does not use this
module or require administrator privileges.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

STATE_VERSION = 1
_STATE_NAME = "stay-awake-power.json"
_LOCK_NAME = "stay-awake-power.lock"

ReadSleepDisabled = Callable[[], int]
SetSleepDisabled = Callable[[int], None]


class PowerProtectError(RuntimeError):
    """The ownership journal is missing, corrupt, or cannot be reconciled."""


def _state_root() -> Path:
    """Return a user-scoped shared state root for all Hermes profiles."""
    configured = os.environ.get("HERMES_POWER_PROTECT_STATE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    # PowerManagement is host-wide, so profile-local HERMES_HOME paths must not
    # split ownership state between profiles running on the same Mac user.
    return Path.home() / ".hermes" / "runtime" / "stay-awake"


def _state_path(root: Path) -> Path:
    return root / _STATE_NAME


def _lock_path(root: Path) -> Path:
    return root / _LOCK_NAME


def _process_start_time(pid: int) -> float | None:
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _owner_alive(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    try:
        pid = int(record["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    try:
        import psutil

        process = psutil.Process(pid)
        if not process.is_running():
            return False
        expected = record.get("start_time")
        if (
            expected is not None
            and abs(float(process.create_time()) - float(expected)) > 1.0
        ):
            return False
        return True
    except Exception:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True


def _same_owner(record: Any, pid: int, start_time: float | None) -> bool:
    if not isinstance(record, dict):
        return False
    pid_value = record.get("pid")
    if not isinstance(pid_value, (int, str)):
        return False
    try:
        if int(pid_value) != pid:
            return False
    except (TypeError, ValueError):
        return False
    expected = record.get("start_time")
    return (
        expected is None
        or start_time is None
        or abs(float(expected) - float(start_time)) <= 1.0
    )


def _validate_state(data: Any) -> dict[str, Any]:
    if (
        not isinstance(data, dict)
        or isinstance(data.get("version"), bool)
        or data.get("version") != STATE_VERSION
    ):
        raise PowerProtectError("invalid Hermes Power Protect state version")
    previous = data.get("previous_sleep_disabled")
    owners = data.get("owners")
    if (
        isinstance(previous, bool)
        or previous not in (0, 1)
        or not isinstance(owners, dict)
    ):
        raise PowerProtectError("invalid Hermes Power Protect state shape")
    for key, value in owners.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            raise PowerProtectError("invalid Hermes Power Protect owner record")
        try:
            pid = int(value["pid"])
            if pid <= 0 or isinstance(value["pid"], bool) or key != str(pid):
                raise ValueError
            start_time = value.get("start_time")
            if start_time is not None and not math.isfinite(float(start_time)):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise PowerProtectError(
                "invalid Hermes Power Protect owner record"
            ) from None
        # Dead, well-formed owners are pruned under the lock. Malformed records
        # fail closed instead of being interpreted as stale.
    return data


@contextmanager
def _locked(root: Path) -> Iterator[None]:
    import fcntl

    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = _lock_path(root)
    with lock_path.open("a+") as lock:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _load(root: Path) -> dict[str, Any] | None:
    path = _state_path(root)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as stream:
            return _validate_state(json.load(stream))
    except PowerProtectError:
        raise
    except Exception as exc:
        raise PowerProtectError(
            f"cannot read Hermes Power Protect state: {exc}"
        ) from exc


def _write(root: Path, state: dict[str, Any]) -> None:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(prefix=".stay-awake-power-", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(state, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(raw_path, 0o600)
        os.replace(raw_path, _state_path(root))
    finally:
        try:
            os.unlink(raw_path)
        except FileNotFoundError:
            pass


def _remove(root: Path) -> None:
    try:
        _state_path(root).unlink()
    except FileNotFoundError:
        pass


class PowerProtectLease:
    """One process-owned lease on the persistent macOS sleep setting."""

    def __init__(
        self,
        read_sleep_disabled: ReadSleepDisabled,
        set_sleep_disabled: SetSleepDisabled,
        *,
        root: Path | None = None,
    ) -> None:
        self._read = read_sleep_disabled
        self._set = set_sleep_disabled
        self._root = root or _state_root()
        self._pid = os.getpid()
        self._start_time = _process_start_time(self._pid)
        self._acquired = False
        self.recovered_stale = False

    def acquire(self) -> None:
        if self._acquired:
            return
        with _locked(self._root):
            state = _load(self._root)
            if state is not None:
                before_count = len(state["owners"])
                state = self._reap_dead_owners(state)
                self.recovered_stale = (
                    state is None or len(state["owners"]) < before_count
                )
            if state is None:
                previous = self._read()
                if previous not in (0, 1):
                    raise PowerProtectError(
                        f"unexpected SleepDisabled value: {previous!r}"
                    )
                state = {
                    "version": STATE_VERSION,
                    "previous_sleep_disabled": previous,
                    "owners": {},
                }
                owners = state["owners"]
                owners[str(self._pid)] = {
                    "pid": self._pid,
                    "start_time": self._start_time,
                }
                # Journal ownership before changing the persistent setting. If
                # the process is killed between these operations, the next Hermes
                # launch can still identify and repair the interrupted attempt.
                _write(self._root, state)
                if previous != 1:
                    try:
                        self._set(1)
                    except Exception:
                        _remove(self._root)
                        raise
            else:
                if self._read() != 1:
                    self._set(1)
                owners = state.get("owners")
                if not isinstance(owners, dict):
                    raise PowerProtectError("invalid Hermes Power Protect owners")
                owners[str(self._pid)] = {
                    "pid": self._pid,
                    "start_time": self._start_time,
                }
                _write(self._root, state)
            self._acquired = True

    def release(self) -> None:
        if not self._acquired:
            return
        try:
            with _locked(self._root):
                state = _load(self._root)
                if state is None:
                    return
                owners = state["owners"]
                record = owners.get(str(self._pid))
                if not _same_owner(record, self._pid, self._start_time):
                    return
                owners.pop(str(self._pid), None)
                if owners:
                    _write(self._root, state)
                    return
                previous = int(state["previous_sleep_disabled"])
                if self._read() == 1 and previous == 0:
                    self._set(previous)
                _remove(self._root)
        finally:
            self._acquired = False

    def _reap_dead_owners(self, state: dict[str, Any]) -> dict[str, Any] | None:
        original_keys = set(state["owners"])
        owners = {
            key: value for key, value in state["owners"].items() if _owner_alive(value)
        }
        state["owners"] = owners
        if owners:
            if set(owners) != original_keys:
                _write(self._root, state)
            return state
        previous = int(state["previous_sleep_disabled"])
        if self._read() == 1 and previous == 0:
            self._set(previous)
        _remove(self._root)
        return None


def recover_stale_power_protect(
    read_sleep_disabled: ReadSleepDisabled,
    set_sleep_disabled: SetSleepDisabled,
    *,
    root: Path | None = None,
) -> bool:
    """Repair a stale Hermes lease before a new session begins.

    Returns ``True`` when a stale Hermes-owned record was found and removed.
    Unknown ``SleepDisabled=1`` values are never changed without a Hermes
    ownership record.
    """
    resolved_root = root or _state_root()
    if not _state_path(resolved_root).exists():
        return False
    with _locked(resolved_root):
        state = _load(resolved_root)
        if state is None:
            return False
        before_count = len(state["owners"])
        lease = PowerProtectLease(
            read_sleep_disabled, set_sleep_disabled, root=resolved_root
        )
        after = lease._reap_dead_owners(state)
        return after is None or len(after["owners"]) < before_count

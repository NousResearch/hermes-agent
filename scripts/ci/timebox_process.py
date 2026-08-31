"""Run a command inside a bounded, recursively supervised process tree."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from types import FrameType
from typing import Protocol

import psutil


TIMEOUT_EXIT_CODE = 124
KILL_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)
_FINAL_REAP_SECONDS = 5
_DESCENDANT_REFRESH_SECONDS = 1.0
_SignalHandler = Callable[[int, FrameType | None], object] | int | None


class _CancellationSignal(BaseException):
    """Interrupt a blocking wait while retaining the signal to forward."""

    def __init__(self, signal_number: int) -> None:
        super().__init__(signal_number)
        self.signal_number = signal_number


class _CancellationForwarder:
    """Raise once so repeated cancellation cannot interrupt tree cleanup."""

    def __init__(self) -> None:
        self._raised = False
        self._defer_depth = 0
        self._pending_signal: int | None = None

    def __call__(self, signal_number: int, _frame: FrameType | None) -> None:
        if self._raised:
            return
        self._raised = True
        if self._defer_depth:
            self._pending_signal = signal_number
            return
        raise _CancellationSignal(signal_number)

    @contextmanager
    def defer(self) -> Iterator[None]:
        """Delay the first cancellation until bootstrap owns the child handle."""
        self._defer_depth += 1
        try:
            yield
        finally:
            self._defer_depth -= 1
            if self._defer_depth == 0 and self._pending_signal is not None:
                pending_signal = self._pending_signal
                self._pending_signal = None
                raise _CancellationSignal(pending_signal)


def _terminate_group(pid: int, signal_number: int) -> None:
    """Signal a complete POSIX process group without racing a normal exit."""
    if os.name == "nt":
        # Windows has no killpg equivalent. taskkill's /T follows the child
        # tree, which gives local Windows verification the same no-orphans
        # guarantee as the Linux CI process group.
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    try:
        os.killpg(pid, signal_number)  # windows-footgun: ok - guarded above
    except ProcessLookupError:
        pass


def _snapshot_descendants(pid: int) -> list[psutil.Process]:
    """Retain handles for descendants even if their group leader exits."""
    try:
        return psutil.Process(pid).children(recursive=True)
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return []


class _DescendantTracker:
    """Retain descendant identities before a short-lived group leader exits."""

    def __init__(
        self,
        pid: int,
        *,
        refresh_seconds: float = _DESCENDANT_REFRESH_SECONDS,
        snapshot_descendants: Callable[[int], Sequence[psutil.Process]] = (
            _snapshot_descendants
        ),
    ) -> None:
        self._pid = pid
        self._refresh_seconds = refresh_seconds
        self._snapshot_descendants = snapshot_descendants
        self._tracked: dict[tuple[int, float], psutil.Process] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"timebox-descendants-{pid}",
            daemon=True,
        )

    def _refresh(self) -> None:
        for process in self._snapshot_descendants(self._pid):
            try:
                identity = (process.pid, process.create_time())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
            with self._lock:
                self._tracked[identity] = process

    def _run(self) -> None:
        while not self._stop_event.wait(self._refresh_seconds):
            self._refresh()

    def start(self) -> None:
        self._refresh()
        self._thread.start()

    def stop(self) -> list[psutil.Process]:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(1.0, self._refresh_seconds * 2))
        self._refresh()
        with self._lock:
            return list(self._tracked.values())


class _ProcessTracker(Protocol):
    def start(self) -> None: ...

    def stop(self) -> list[psutil.Process]: ...


def _signal_descendants(
    descendants: Sequence[psutil.Process], signal_number: int
) -> None:
    """Signal deepest descendants first, including separate sessions."""
    for descendant in reversed(descendants):
        try:
            descendant.send_signal(signal_number)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue


def _remaining_seconds(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _wait_for_root(process: subprocess.Popen[bytes], deadline: float) -> bool:
    """Return whether the group leader survived until *deadline*."""
    try:
        process.wait(timeout=_remaining_seconds(deadline))
    except subprocess.TimeoutExpired:
        return process.poll() is None
    return False


def _wait_for_descendants(
    descendants: Sequence[psutil.Process], deadline: float
) -> list[psutil.Process]:
    """Return retained descendants still alive at the common deadline."""
    if not descendants:
        return []
    _, alive = psutil.wait_procs(
        list(descendants), timeout=_remaining_seconds(deadline)
    )
    return alive


def _terminate_complete_tree(
    process: subprocess.Popen[bytes],
    *,
    descendants: Sequence[psutil.Process],
    initial_signal: int,
    grace_seconds: int,
    terminate_group: Callable[[int, int], None],
) -> None:
    """Stop the leader and every retained descendant within one grace period."""
    terminate_group(process.pid, initial_signal)
    _signal_descendants(descendants, initial_signal)

    deadline = time.monotonic() + grace_seconds
    root_alive = _wait_for_root(process, deadline)
    alive_descendants = _wait_for_descendants(descendants, deadline)
    if not root_alive and not alive_descendants:
        return

    if root_alive:
        terminate_group(process.pid, KILL_SIGNAL)
    _signal_descendants(alive_descendants, KILL_SIGNAL)

    if root_alive:
        process.wait()
    if alive_descendants:
        psutil.wait_procs(alive_descendants, timeout=_FINAL_REAP_SECONDS)


def run_with_timebox(
    command: Sequence[str],
    *,
    timeout_seconds: int,
    grace_seconds: int,
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    terminate_group: Callable[[int, int], None] = _terminate_group,
    tracker_factory: Callable[[int], _ProcessTracker] = _DescendantTracker,
    startup_guard: Callable[[], AbstractContextManager[None]] = nullcontext,
) -> int:
    """Run *command* and terminate its complete process tree on timeout."""
    process: subprocess.Popen[bytes] | None = None
    tracker: _ProcessTracker | None = None

    def stop_tracker() -> list[psutil.Process]:
        return tracker.stop() if tracker is not None else []

    def terminate_process_tree(initial_signal: int) -> None:
        if process is None:
            return
        _terminate_complete_tree(
            process,
            descendants=stop_tracker(),
            initial_signal=initial_signal,
            grace_seconds=grace_seconds,
            terminate_group=terminate_group,
        )

    try:
        with startup_guard():
            process = popen(list(command), start_new_session=os.name != "nt")
            tracker = tracker_factory(process.pid)
            tracker.start()
        result = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        print(
            f"::error::Fork test shard exceeded the {timeout_seconds}-second watchdog.",
            flush=True,
        )
        terminate_process_tree(signal.SIGTERM)
        return TIMEOUT_EXIT_CODE
    except _CancellationSignal as cancellation:
        terminate_process_tree(cancellation.signal_number)
        raise
    except KeyboardInterrupt:
        terminate_process_tree(signal.SIGINT)
        raise
    except BaseException:
        terminate_process_tree(signal.SIGTERM)
        raise
    else:
        stop_tracker()
        return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--grace-seconds", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if args.timeout_seconds <= 0 or args.grace_seconds < 0:
        parser.error("timeout must be positive, with a non-negative grace period")

    previous_handlers: dict[int, _SignalHandler] = {}
    forward_cancellation = _CancellationForwarder()
    for signal_name in ("SIGTERM", "SIGHUP", "SIGINT"):
        signal_number = getattr(signal, signal_name, None)
        if isinstance(signal_number, int):
            previous_handlers[signal_number] = signal.signal(
                signal_number, forward_cancellation
            )
    try:
        return run_with_timebox(
            command,
            timeout_seconds=args.timeout_seconds,
            grace_seconds=args.grace_seconds,
            startup_guard=forward_cancellation.defer,
        )
    except _CancellationSignal as cancellation:
        return 128 + cancellation.signal_number
    finally:
        for signal_number, handler in previous_handlers.items():
            signal.signal(signal_number, handler)


if __name__ == "__main__":
    raise SystemExit(main())

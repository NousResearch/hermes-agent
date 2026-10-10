from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_cli.gateway_windows_supervisor as supervisor


class _FakeOwner:
    acquired = True
    released = 0

    def __init__(self, _path: Path) -> None:
        pass

    def acquire(self) -> bool:
        return type(self).acquired

    def release(self) -> None:
        type(self).released += 1


def test_repeated_exit_75_is_bounded(monkeypatch, tmp_path):
    """Exit 75 is retryable, not privileged: watchdog/ownership failures also use it."""
    calls = []

    class Proc:
        def __init__(self, argv):
            calls.append(list(argv))

        def wait(self):
            return 75

    _FakeOwner.acquired = True
    _FakeOwner.released = 0
    monkeypatch.setattr(supervisor, "_OwnerLock", _FakeOwner)
    monkeypatch.setattr(supervisor.subprocess, "Popen", Proc)
    monkeypatch.setattr(supervisor, "_ack_stop", lambda *a: False)
    monkeypatch.setattr(supervisor, "_sleep_with_stop", lambda *a: False)

    rc = supervisor.supervise(
        ["python", "-c", "pass"],
        home=tmp_path,
        restart_delay_ms=0,
        failure_window_s=300,
        max_failures=3,
    )

    assert rc == 75
    assert len(calls) == 3
    assert _FakeOwner.released == 1


def test_mixed_retryable_failures_share_one_budget(monkeypatch, tmp_path):
    codes = iter([17, 75, 19])
    calls = []

    class Proc:
        def __init__(self, argv):
            calls.append(list(argv))

        def wait(self):
            return next(codes)

    _FakeOwner.acquired = True
    monkeypatch.setattr(supervisor, "_OwnerLock", _FakeOwner)
    monkeypatch.setattr(supervisor.subprocess, "Popen", Proc)
    monkeypatch.setattr(supervisor, "_ack_stop", lambda *a: False)
    monkeypatch.setattr(supervisor, "_sleep_with_stop", lambda *a: False)

    assert supervisor.supervise(
        ["gateway"], home=tmp_path, restart_delay_ms=0,
        failure_window_s=300, max_failures=3,
    ) == 19
    assert len(calls) == 3


def test_non_owner_never_spawns_or_consumes_stop(monkeypatch, tmp_path):
    _FakeOwner.acquired = False
    monkeypatch.setattr(supervisor, "_OwnerLock", _FakeOwner)
    monkeypatch.setattr(
        supervisor.subprocess, "Popen",
        lambda *a, **k: pytest.fail("non-owner must not spawn a child"),
    )
    monkeypatch.setattr(
        supervisor, "_ack_stop",
        lambda *a: pytest.fail("non-owner must not consume stop state"),
    )

    assert supervisor.supervise(["gateway"], home=tmp_path) == 0


def test_stop_nonce_is_acknowledged_before_marker_is_removed(tmp_path):
    marker, ack, _owner = supervisor._paths(tmp_path)
    marker.parent.mkdir(parents=True)
    marker.write_text("nonce-123\n", encoding="utf-8")

    assert supervisor._ack_stop(marker, ack)
    assert not marker.exists()
    assert ack.read_text(encoding="utf-8") == "nonce-123"


@pytest.mark.platforms("windows")
def test_owner_lock_is_exclusive_and_recoverable(tmp_path):
    """Real Windows byte-range lock: one lifecycle owner, then clean takeover after release."""
    _marker, _ack, lock_path = supervisor._paths(tmp_path)
    first = supervisor._OwnerLock(lock_path)
    second = supervisor._OwnerLock(lock_path)

    assert first.acquire()
    assert not second.acquire()
    first.release()
    assert second.acquire()
    second.release()

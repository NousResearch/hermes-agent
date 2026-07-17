"""Focused RED tests for durable Antigravity runtime resolution."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import types
from pathlib import Path

import pytest

import agent.antigravity_runtime as ar


def _runtime_root(home: Path) -> Path:
    return home / ".gemini" / "antigravity-cli"


def _write_state(home: Path, payload: dict) -> Path:
    path = _runtime_root(home) / "log" / "hermes-antigravity-runtime.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def tmp_path():
    root = Path.cwd() / ".pytest-runtime-fixtures"
    for relative in (
        "home/.gemini/antigravity-cli/log/hermes-antigravity-runtime.json",
        "home/.gemini/antigravity-cli/log/hermes-antigravity-managed.log",
        "home/.gemini/antigravity-cli/cache/default_project_id.txt",
        "runtime.lock",
    ):
        path = root / relative
        if path.exists():
            path.unlink()
    return root


@pytest.fixture(autouse=True)
def _hermetic_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))
    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_METADATA_SERVICE_TIMEOUT", "1")
    monkeypatch.setenv("AWS_METADATA_SERVICE_NUM_ATTEMPTS", "1")
    monkeypatch.setenv("TIRITH_ENABLED", "false")


@pytest.fixture
def runtime_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-no-profile-home"))
    monkeypatch.setattr(ar, "_windows_listener_is_agy", lambda address, expected_pid=None: True)
    return home


def _make_manager(
    tmp_path: Path, env: dict[str, str], *, platform: str = "win32"
) -> "ar.AntigravityRuntimeManager":
    return ar.AntigravityRuntimeManager(
        command="agy",
        cwd=str(tmp_path),
        env_factory=lambda: dict(env),
        platform=platform,
    )


def test_live_env_address_is_retained(runtime_home, tmp_path, monkeypatch):
    env = {
        "HOME": str(runtime_home),
        "ANTIGRAVITY_LS_ADDRESS": "http://localhost:61727",
        "ANTIGRAVITY_PROJECT_ID": "proj-env",
    }
    manager = _make_manager(tmp_path, env)
    start_calls: list[float] = []

    monkeypatch.setattr(
        ar,
        "probe_antigravity_address",
        lambda address, timeout_seconds=0.2: address == "http://localhost:61727",
    )
    monkeypatch.setattr(
        manager,
        "_start_managed_runtime",
        lambda timeout_seconds: start_calls.append(timeout_seconds),
    )

    child_env = manager.prepare_env(timeout_seconds=5)

    assert child_env["ANTIGRAVITY_LS_ADDRESS"] == "http://localhost:61727"
    assert child_env["ANTIGRAVITY_PROJECT_ID"] == "proj-env"
    assert start_calls == []


def test_stale_env_address_is_replaced_by_live_managed_state(runtime_home, tmp_path, monkeypatch):
    env = {
        "HOME": str(runtime_home),
        "ANTIGRAVITY_LS_ADDRESS": "http://localhost:61727",
        "ANTIGRAVITY_PROJECT_ID": "proj-env",
    }
    manager = _make_manager(tmp_path, env)
    _write_state(
        runtime_home,
        {
            "status": "ready",
            "address": "http://localhost:62111",
            "project_id": "proj-state",
            "daemon_pid": 1001,
            "agy_pid": 1002,
            "log_path": str(_runtime_root(runtime_home) / "log" / "managed.log"),
        },
    )

    monkeypatch.setattr(
        ar,
        "probe_antigravity_address",
        lambda address, timeout_seconds=0.2: address == "http://localhost:62111",
    )

    child_env = manager.prepare_env(timeout_seconds=5)

    assert child_env["ANTIGRAVITY_LS_ADDRESS"] == "http://localhost:62111"
    assert child_env["ANTIGRAVITY_PROJECT_ID"] == "proj-env"


def test_project_id_falls_back_to_default_project_cache(runtime_home, tmp_path, monkeypatch):
    env = {
        "HOME": str(runtime_home),
        "ANTIGRAVITY_LS_ADDRESS": "http://localhost:62111",
    }
    cache_path = _runtime_root(runtime_home) / "cache" / "default_project_id.txt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("proj-from-cache\n", encoding="utf-8")
    manager = _make_manager(tmp_path, env)

    monkeypatch.setattr(ar, "probe_antigravity_address", lambda address, timeout_seconds=0.2: True)

    child_env = manager.prepare_env(timeout_seconds=5)

    assert child_env["ANTIGRAVITY_PROJECT_ID"] == "proj-from-cache"


def test_http_port_parser_extracts_real_log_line_shape():
    line = "2026-07-17T02:10:31.318Z INFO Language server listening on random port at 61727 for HTTP"

    assert ar.parse_antigravity_log_address(line) == "http://localhost:61727"


def test_auto_start_is_invoked_only_when_no_live_address_exists(runtime_home, tmp_path, monkeypatch):
    env = {
        "HOME": str(runtime_home),
        "ANTIGRAVITY_LS_ADDRESS": "http://localhost:61727",
        "ANTIGRAVITY_PROJECT_ID": "proj-env",
    }
    manager = _make_manager(tmp_path, env)
    start_calls: list[float] = []

    monkeypatch.setattr(
        ar,
        "probe_antigravity_address",
        lambda address, timeout_seconds=0.2: address == "http://localhost:63123",
    )

    def _fake_start(timeout_seconds: float):
        start_calls.append(timeout_seconds)
        _write_state(
            runtime_home,
            {
                "status": "ready",
                "address": "http://localhost:63123",
                "project_id": "proj-env",
                "daemon_pid": 2001,
                "agy_pid": 2002,
                "log_path": str(_runtime_root(runtime_home) / "log" / "managed.log"),
            },
        )

    monkeypatch.setattr(manager, "_start_managed_runtime", _fake_start)

    child_env = manager.prepare_env(timeout_seconds=5)

    assert child_env["ANTIGRAVITY_LS_ADDRESS"] == "http://localhost:63123"
    assert len(start_calls) == 1
    assert 0 < start_calls[0] - time.monotonic() <= 5


def test_concurrent_prepare_env_calls_do_not_duplicate_runtime_launch(
    runtime_home, tmp_path, monkeypatch
):
    env = {
        "HOME": str(runtime_home),
        "ANTIGRAVITY_PROJECT_ID": "proj-env",
    }
    manager = _make_manager(tmp_path, env)
    launch_started = threading.Event()
    release_launch = threading.Event()
    launch_calls: list[float] = []
    errors: list[BaseException] = []
    results: list[str] = []

    monkeypatch.setattr(
        ar,
        "probe_antigravity_address",
        lambda address, timeout_seconds=0.2: address == "http://localhost:64123",
    )

    def _fake_start(timeout_seconds: float):
        launch_calls.append(timeout_seconds)
        launch_started.set()
        assert release_launch.wait(timeout=1), "launch never released"
        _write_state(
            runtime_home,
            {
                "status": "ready",
                "address": "http://localhost:64123",
                "project_id": "proj-env",
                "daemon_pid": 3001,
                "agy_pid": 3002,
                "log_path": str(_runtime_root(runtime_home) / "log" / "managed.log"),
            },
        )

    monkeypatch.setattr(manager, "_start_managed_runtime", _fake_start)

    def _worker():
        try:
            result = manager.prepare_env(timeout_seconds=5)
            results.append(result["ANTIGRAVITY_LS_ADDRESS"])
        except BaseException as exc:  # pragma: no cover - captured for assertions below
            errors.append(exc)

    first = threading.Thread(target=_worker)
    second = threading.Thread(target=_worker)
    first.start()
    assert launch_started.wait(timeout=1), "first launch never started"
    second.start()
    time.sleep(0.05)
    release_launch.set()
    first.join(timeout=1)
    second.join(timeout=1)

    assert not errors
    assert len(launch_calls) == 1
    assert 0 < launch_calls[0] - time.monotonic() <= 5
    assert results == ["http://localhost:64123", "http://localhost:64123"]


def test_non_windows_startup_failure_is_actionable(runtime_home, tmp_path, monkeypatch):
    env = {
        "HOME": str(runtime_home),
        "ANTIGRAVITY_PROJECT_ID": "proj-env",
    }
    manager = _make_manager(tmp_path, env, platform="linux")

    monkeypatch.setattr(ar, "probe_antigravity_address", lambda address, timeout_seconds=0.2: False)

    with pytest.raises(RuntimeError, match="(?i)(windows|ANTIGRAVITY_LS_ADDRESS|agy -i)"):
        manager.prepare_env(timeout_seconds=5)


def test_managed_child_spawn_truncates_stale_log_before_launch(runtime_home, tmp_path, monkeypatch):
    daemon = ar._ManagedAntigravityDaemon(command="agy", cwd=str(tmp_path), home=runtime_home)
    log_path = _runtime_root(runtime_home) / "log" / "hermes-antigravity-managed.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "Language server listening on random port at 61234 for HTTP\n",
        encoding="utf-8",
    )

    seen: dict[str, object] = {}

    def _fake_spawn(command, *, cwd, env):
        seen["command"] = command
        seen["cwd"] = cwd
        seen["home"] = env.get("HOME")
        seen["log_contents"] = log_path.read_text(encoding="utf-8")
        return object()

    monkeypatch.setitem(
        sys.modules,
        "winpty",
        types.SimpleNamespace(PtyProcess=types.SimpleNamespace(spawn=_fake_spawn)),
    )

    daemon._spawn_child()

    assert seen["cwd"] == str(tmp_path)
    assert seen["home"] == str(runtime_home)
    assert seen["log_contents"] == ""


def test_windows_managed_state_rejects_unrelated_listener_owner(
    runtime_home, tmp_path, monkeypatch
):
    env = {
        "HOME": str(runtime_home),
        "ANTIGRAVITY_PROJECT_ID": "proj-env",
    }
    manager = _make_manager(tmp_path, env, platform="win32")
    _write_state(
        runtime_home,
        {
            "status": "ready",
            "address": "http://localhost:62111",
            "project_id": "proj-state",
            "daemon_pid": 1001,
            "agy_pid": 4321,
            "log_path": str(_runtime_root(runtime_home) / "log" / "managed.log"),
        },
    )
    manager._current_home = runtime_home

    expected_pids: list[int | None] = []
    monkeypatch.setattr(ar, "probe_antigravity_address", lambda address, timeout_seconds=0.2: True)
    monkeypatch.setattr(
        ar,
        "_windows_listener_is_agy",
        lambda address, *, expected_pid=None: expected_pids.append(expected_pid) or False,
        raising=False,
    )

    assert manager._discover_live_address_from_state() is None
    assert expected_pids == [4321]


class _BudgetClock:
    def __init__(self) -> None:
        self.now = 100.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_log_discovery_stops_when_deadline_budget_is_exhausted(
    runtime_home, tmp_path, monkeypatch
):
    env = {
        "HOME": str(runtime_home),
        "ANTIGRAVITY_PROJECT_ID": "proj-env",
    }
    clock = _BudgetClock()
    manager = ar.AntigravityRuntimeManager(
        command="agy",
        cwd=str(tmp_path),
        env_factory=lambda: dict(env),
        platform="win32",
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    log_dir = _runtime_root(runtime_home) / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(10):
        path = log_dir / f"candidate-{idx}.log"
        path.write_text("placeholder\n", encoding="utf-8")
        ts = time.time() - idx
        os.utime(path, (ts, ts))

    probe_budgets: list[tuple[float, float]] = []

    def _fake_probe(address, timeout_seconds=0.2):
        remaining_before_probe = 100.35 - clock.now
        probe_budgets.append((timeout_seconds, remaining_before_probe))
        clock.now += min(0.11, timeout_seconds)
        return False

    class _StopLaunch(Exception):
        pass

    monkeypatch.setattr(ar, "_tail_log_for_address", lambda path: "http://localhost:65000")
    monkeypatch.setattr(ar, "probe_antigravity_address", _fake_probe)
    monkeypatch.setattr(manager, "_start_managed_runtime", lambda timeout_seconds: (_ for _ in ()).throw(_StopLaunch()))

    with pytest.raises(_StopLaunch):
        manager.prepare_env(timeout_seconds=0.35)

    assert len(probe_budgets) <= 4
    assert all(timeout <= remaining + 1e-9 for timeout, remaining in probe_budgets)
    assert clock.now <= 100.35 + 1e-9


def test_prepare_env_uses_single_absolute_deadline_across_discovery_and_wait(
    runtime_home, tmp_path, monkeypatch
):
    env = {
        "HOME": str(runtime_home),
        "ANTIGRAVITY_PROJECT_ID": "proj-env",
    }
    clock = _BudgetClock()
    manager = ar.AntigravityRuntimeManager(
        command="agy",
        cwd=str(tmp_path),
        env_factory=lambda: dict(env),
        platform="win32",
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    phase = {"count": 0}

    def _fake_logs(*, deadline=None):
        del deadline
        phase["count"] += 1
        if phase["count"] == 1:
            clock.now += 0.6
        return None

    monkeypatch.setattr(
        manager,
        "_discover_live_address_from_state",
        lambda *, deadline=None: None,
    )
    monkeypatch.setattr(manager, "_discover_live_address_from_logs", _fake_logs)
    monkeypatch.setattr(manager, "_start_managed_runtime", lambda deadline: None)

    with pytest.raises(RuntimeError, match="caller timeout expired"):
        manager.prepare_env(timeout_seconds=1.0)

    assert clock.now <= 101.01


def test_windows_runtime_lock_closes_handle_on_unexpected_post_lock_exception(
    tmp_path, monkeypatch
):
    lock = ar._WindowsRuntimeLock(tmp_path / "runtime.lock")
    closed: list[bool] = []

    class _FakeHandle:
        def seek(self, offset):
            return None

        def fileno(self):
            return 7

        def truncate(self):
            return None

        def write(self, payload):
            raise RuntimeError("boom")

        def flush(self):
            return None

        def close(self):
            closed.append(True)

    monkeypatch.setitem(
        sys.modules,
        "msvcrt",
        types.SimpleNamespace(LK_LOCK=1, LK_NBLCK=2, LK_UNLCK=3, locking=lambda *args: None),
    )
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: _FakeHandle())

    with pytest.raises(RuntimeError, match="boom"):
        lock.acquire(blocking=False)

    assert closed == [True]
    assert lock._handle is None


def test_windows_runtime_lock_closes_handle_on_base_exception(tmp_path, monkeypatch):
    lock = ar._WindowsRuntimeLock(tmp_path / "runtime.lock")
    closed: list[bool] = []

    class _FakeHandle:
        def seek(self, offset):
            return None

        def fileno(self):
            return 7

        def truncate(self):
            return None

        def write(self, payload):
            raise KeyboardInterrupt()

        def flush(self):
            return None

        def close(self):
            closed.append(True)

    monkeypatch.setitem(
        sys.modules,
        "msvcrt",
        types.SimpleNamespace(LK_LOCK=1, LK_NBLCK=2, LK_UNLCK=3, locking=lambda *args: None),
    )
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: _FakeHandle())

    with pytest.raises(KeyboardInterrupt):
        lock.acquire(blocking=False)

    assert closed == [True]
    assert lock._handle is None


def test_windows_runtime_lock_release_always_closes_and_resets(tmp_path, monkeypatch):
    lock = ar._WindowsRuntimeLock(tmp_path / "runtime.lock")
    closed: list[bool] = []

    class _FakeHandle:
        def seek(self, offset):
            raise ValueError("closed unexpectedly")

        def fileno(self):
            return 7

        def close(self):
            closed.append(True)

    monkeypatch.setitem(
        sys.modules,
        "msvcrt",
        types.SimpleNamespace(LK_UNLCK=3, locking=lambda *args: None),
    )
    lock._handle = _FakeHandle()

    with pytest.raises(ValueError, match="closed unexpectedly"):
        lock.release()

    assert closed == [True]
    assert lock._handle is None


def test_windows_runtime_lock_does_not_mask_seek_oserror_as_contention(tmp_path, monkeypatch):
    lock = ar._WindowsRuntimeLock(tmp_path / "runtime.lock")
    closed: list[bool] = []
    locking_calls: list[bool] = []

    class _FakeHandle:
        def seek(self, offset):
            raise OSError("seek failed")

        def fileno(self):
            return 7

        def close(self):
            closed.append(True)

    monkeypatch.setitem(
        sys.modules,
        "msvcrt",
        types.SimpleNamespace(
            LK_LOCK=1,
            LK_NBLCK=2,
            locking=lambda *args: locking_calls.append(True),
        ),
    )
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: _FakeHandle())

    with pytest.raises(OSError, match="seek failed"):
        lock.acquire(blocking=False)

    assert locking_calls == []
    assert closed == [True]
    assert lock._handle is None

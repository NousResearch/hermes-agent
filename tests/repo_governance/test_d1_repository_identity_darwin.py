"""Darwin descriptor-relative spawn/lifecycle tests."""
from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

import pytest

CANDIDATE = Path("/Users/ykliu/.hermes/profiles/dev/artifacts/repo-governance/2026-08-11-repo-governance-d1-3-i3-candidate")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class _FakeCFunction:
    def __init__(self, call):
        self._call = call

    def __call__(self, *args):
        return self._call(*args)


class _FakeSpawnLibSystem:
    def __init__(self, pid):
        self.posix_spawn_file_actions_init = _FakeCFunction(lambda _actions: 0)
        self.posix_spawn_file_actions_addfchdir = _FakeCFunction(lambda *_args: 0)
        self.posix_spawn_file_actions_adddup2 = _FakeCFunction(lambda *_args: 0)
        self.posix_spawn_file_actions_addclose = _FakeCFunction(lambda *_args: 0)
        self.posix_spawn_file_actions_destroy = _FakeCFunction(lambda _actions: 0)

        def spawn(pid_pointer, *_args):
            pid_pointer._obj.value = pid
            return 0

        self.posix_spawn = _FakeCFunction(spawn)


@pytest.mark.parametrize(
    ("crossing_site", "monotonic_ns_samples"),
    (
        ("term_grace", (0, 249_000_000, 251_000_000, 250_000_000, 500_000_000)),
        ("cleanup_deadline", (0, 250_000_000, 499_000_000, 501_000_000, 500_000_000)),
    ),
)
def test_deadline_crossing_never_sleeps_negative_in_cleanup_loops(
    monkeypatch, crossing_site, monotonic_ns_samples
):
    from repo_governance import darwin_repository_identity as darwin

    child_pid = 4242
    pipes = iter(((10, 11), (12, 13)))
    monotonic_samples = iter((0.0, 2.0))
    ns_samples = iter(monotonic_ns_samples)
    signals = []
    waits = []
    sleeps = []

    monkeypatch.setattr(darwin, "_libsystem", lambda: _FakeSpawnLibSystem(child_pid))
    monkeypatch.setattr(darwin.os, "pipe", lambda: next(pipes))
    monkeypatch.setattr(darwin.os, "set_inheritable", lambda *_args: None)
    monkeypatch.setattr(darwin.os, "set_blocking", lambda *_args: None)
    monkeypatch.setattr(darwin.os, "close", lambda _fd: None)
    monkeypatch.setattr(darwin.select, "select", lambda *_args: ((), (), ()))
    monkeypatch.setattr(darwin.time, "monotonic", lambda: next(monotonic_samples))
    monkeypatch.setattr(darwin.time, "monotonic_ns", lambda: next(ns_samples))

    def strict_sleep(seconds):
        sleeps.append(seconds)
        if seconds < 0:
            raise ValueError("sleep length must be non-negative")

    def exact_pid_kill(pid, sig):
        assert pid == child_pid and pid > 0
        signals.append((pid, sig))

    def exact_pid_wait(pid, options):
        assert pid == child_pid and pid > 0
        assert options == os.WNOHANG
        waits.append((crossing_site, pid, options))
        return 0, 0

    monkeypatch.setattr(darwin.time, "sleep", strict_sleep)
    monkeypatch.setattr(darwin.os, "kill", exact_pid_kill)
    monkeypatch.setattr(darwin.os, "waitpid", exact_pid_wait)

    with pytest.raises(TimeoutError, match="git operation deadline"):
        darwin._spawn_git_at_fd(99, darwin.GIT_COMMANDS[3], timeout_seconds=1.0)

    assert sleeps and all(0 <= seconds <= 0.01 for seconds in sleeps)
    assert len(sleeps) == 1
    assert waits == [(crossing_site, child_pid, os.WNOHANG)]
    assert signals == [(child_pid, signal.SIGTERM), (child_pid, signal.SIGKILL)]


class _ForkingLibSystem:
    def __init__(self, child_seconds, *, ignore_sigterm=False):
        self.posix_spawn_file_actions_init = _FakeCFunction(lambda _actions: 0)
        self.posix_spawn_file_actions_addfchdir = _FakeCFunction(lambda *_args: 0)
        self.posix_spawn_file_actions_adddup2 = _FakeCFunction(lambda *_args: 0)
        self.posix_spawn_file_actions_addclose = _FakeCFunction(lambda *_args: 0)
        self.posix_spawn_file_actions_destroy = _FakeCFunction(lambda _actions: 0)

        def spawn(pid_pointer, *_args):
            child = os.fork()
            if child == 0:
                if ignore_sigterm:
                    signal.signal(signal.SIGTERM, signal.SIG_IGN)
                os.closerange(3, 256)
                time.sleep(child_seconds)
                os._exit(0)
            pid_pointer._obj.value = child
            return 0

        self.posix_spawn = _FakeCFunction(spawn)


@pytest.mark.live_system_guard_bypass
def test_eof_from_live_child_still_obeys_absolute_operation_deadline(monkeypatch):
    from repo_governance import darwin_repository_identity as darwin

    monkeypatch.setattr(darwin, "_libsystem", lambda: _ForkingLibSystem(2.0))
    fd = os.open(REPOSITORY_ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="git operation deadline"):
            darwin._spawn_git_at_fd(fd, darwin.GIT_COMMANDS[3], timeout_seconds=0.05)
    finally:
        os.close(fd)
    assert time.monotonic() - started < 0.75


@pytest.mark.live_system_guard_bypass
def test_deadline_cleanup_terms_then_conditionally_kills_and_boundedly_reaps_exact_child(monkeypatch):
    from repo_governance import darwin_repository_identity as darwin

    monkeypatch.setattr(darwin, "_libsystem", lambda: _ForkingLibSystem(2.0, ignore_sigterm=True))
    real_kill = os.kill
    real_waitpid = os.waitpid
    signals = []
    waits = []
    interrupted = {"kill": False, "wait": False}

    def recording_kill(pid, sig):
        assert pid > 0
        signals.append((pid, sig, time.monotonic_ns()))
        if not interrupted["kill"]:
            interrupted["kill"] = True
            raise InterruptedError
        return real_kill(pid, sig)

    def recording_waitpid(pid, options):
        assert pid > 0
        assert options == os.WNOHANG
        waits.append((pid, options, time.monotonic_ns()))
        if not interrupted["wait"]:
            interrupted["wait"] = True
            raise InterruptedError
        return real_waitpid(pid, options)

    monkeypatch.setattr(darwin.os, "kill", recording_kill)
    monkeypatch.setattr(darwin.os, "waitpid", recording_waitpid)
    fd = os.open(REPOSITORY_ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        with pytest.raises(TimeoutError, match="git operation deadline"):
            darwin._spawn_git_at_fd(fd, darwin.GIT_COMMANDS[3], timeout_seconds=0.05)
    finally:
        os.close(fd)

    delivered = [row for row in signals if row[1] in (signal.SIGTERM, signal.SIGKILL)]
    assert [row[1] for row in delivered] == [signal.SIGTERM, signal.SIGTERM, signal.SIGKILL]
    assert len({row[0] for row in delivered}) == 1
    assert delivered[-1][2] - delivered[1][2] >= 250_000_000
    assert waits and all(row[0] == delivered[-1][0] and row[1] == os.WNOHANG for row in waits)


def test_darwin_spawn_uses_fchdir_exact_commands_and_reaps_actual_child():
    from repo_governance import darwin_repository_identity as darwin
    profile = json.loads((CANDIDATE / "bindings/r6-profile.json").read_text())
    assert darwin.GIT_COMMANDS == tuple(tuple(row["argv"]) for row in profile["commandContract"]["commands"])
    fd = os.open(REPOSITORY_ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        observation = darwin._spawn_git_at_fd(fd, darwin.GIT_COMMANDS[3], timeout_seconds=5.0)
    finally:
        os.close(fd)
    assert observation["stdout"] == b"true\n"
    assert observation["stderr"] == b""
    assert observation["exitCode"] == 0
    assert observation["exactChildReaped"] is True
    assert observation["remainingOpenFds"] == ()


def test_native_abi_registry_has_18_rows_and_probe_covers_lifecycle():
    from repo_governance import darwin_repository_identity as darwin
    registry = json.loads((CANDIDATE / "native-abi-vectors.v2.json").read_text())
    assert len(registry["expectedIds"]) == 18
    results = darwin._run_native_abi_probe_for_test()
    assert set(results) == set(registry["expectedIds"])
    assert all(row["passed"] and row["residue"] == 0 for row in results.values())

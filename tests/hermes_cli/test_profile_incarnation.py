"""Behavioral tests for persisted named-profile incarnation identity."""

from __future__ import annotations

import multiprocessing
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from hermes_cli import profile_incarnation, profile_lifecycle, profiles
from hermes_cli.profile_incarnation import (
    PROFILE_INCARNATION_FILENAME,
    ensure_profile_incarnation,
    read_profile_incarnation,
)


def _backfill_worker(profile_home: str, gate, results) -> None:
    gate.wait(timeout=10)
    results.put(ensure_profile_incarnation(profile_home))


def _assert_concurrent_process_backfill(
    tmp_path: Path,
    monkeypatch,
) -> None:
    hermes_home = tmp_path / ".hermes"
    profile_home = hermes_home / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    context = multiprocessing.get_context("spawn")
    gate = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_backfill_worker,
            args=(str(profile_home), gate, results),
        )
        for _ in range(6)
    ]
    try:
        for process in processes:
            process.start()
        gate.set()
        for process in processes:
            process.join(timeout=15)
        assert [process.exitcode for process in processes] == [0] * len(processes)

        observed = [results.get(timeout=2) for _ in processes]
        assert None not in observed
        assert len(set(observed)) == 1
        assert read_profile_incarnation(profile_home) == observed[0]
        assert not list(profile_home.glob(f"{PROFILE_INCARNATION_FILENAME}.*.tmp"))
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        results.close()
        results.join_thread()


@pytest.mark.linux_only
def test_concurrent_process_backfill_on_linux(tmp_path: Path, monkeypatch) -> None:
    _assert_concurrent_process_backfill(tmp_path, monkeypatch)


@pytest.mark.macos_only
def test_concurrent_process_backfill_on_macos(tmp_path: Path, monkeypatch) -> None:
    _assert_concurrent_process_backfill(tmp_path, monkeypatch)


@pytest.mark.windows_only
def test_concurrent_process_backfill_on_windows(tmp_path: Path, monkeypatch) -> None:
    _assert_concurrent_process_backfill(tmp_path, monkeypatch)


def _assert_legacy_backfill_excludes_profile_recreation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    profile_home = profiles.create_profile("worker", no_alias=True, no_skills=True)
    (profile_home / PROFILE_INCARNATION_FILENAME).unlink()

    entered_read = threading.Event()
    release_read = threading.Event()
    real_read = profile_incarnation.read_profile_incarnation
    observed: list[str | None] = []
    errors: list[BaseException] = []

    def blocking_read(home: Path | str) -> str | None:
        current = real_read(home)
        if (
            Path(home) == profile_home
            and current is None
            and not entered_read.is_set()
        ):
            entered_read.set()
            if not release_read.wait(timeout=5):
                raise TimeoutError("legacy backfill barrier was not released")
        return current

    monkeypatch.setattr(profile_incarnation, "read_profile_incarnation", blocking_read)

    def backfill() -> None:
        try:
            observed.append(ensure_profile_incarnation(profile_home))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=backfill)
    thread.start()
    assert entered_read.wait(timeout=5)
    script = (
        "from hermes_cli import profile_lifecycle, profiles; "
        "profile_lifecycle._PROFILE_LIFECYCLE_LOCK_TIMEOUT_SECONDS=0.2; "
        "\ntry:\n profiles.delete_profile('worker', yes=True)\n"
        "except TimeoutError:\n raise SystemExit(0)\n"
        "profiles.create_profile('worker', no_alias=True, no_skills=True); "
        "raise SystemExit(1)"
    )
    try:
        recreate = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "HERMES_HOME": str(hermes_home)},
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    finally:
        release_read.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert recreate.returncode == 0, recreate.stdout + recreate.stderr
    assert errors == []
    assert len(observed) == 1
    assert observed[0] is not None
    assert real_read(profile_home) == observed[0]


def _assert_resource_lease_timeout_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    profile_home = profiles.create_profile("worker", no_alias=True, no_skills=True)
    incarnation = read_profile_incarnation(profile_home)
    assert incarnation is not None
    script = (
        "import os; from pathlib import Path; "
        "from hermes_cli import profile_lifecycle; "
        "from hermes_cli.profile_incarnation import profile_incarnation_lease; "
        "profile_lifecycle._PROFILE_LIFECYCLE_LOCK_TIMEOUT_SECONDS=0.2; "
        "home=Path(os.environ['PROFILE_HOME']); token=os.environ['INCARNATION']; "
        "\ntry:\n"
        " with profile_incarnation_lease(home, token):\n  pass\n"
        "except FileNotFoundError:\n raise SystemExit(0)\n"
        "except TimeoutError:\n raise SystemExit(2)\n"
        "raise SystemExit(1)"
    )
    with profile_lifecycle._cross_process_profile_mutation_lock():
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={
                **os.environ,
                "HERMES_HOME": str(hermes_home),
                "PROFILE_HOME": str(profile_home),
                "INCARNATION": incarnation,
            },
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.linux_only
def test_legacy_backfill_excludes_profile_recreation_on_linux(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_legacy_backfill_excludes_profile_recreation(tmp_path, monkeypatch)


@pytest.mark.macos_only
def test_legacy_backfill_excludes_profile_recreation_on_macos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_legacy_backfill_excludes_profile_recreation(tmp_path, monkeypatch)


@pytest.mark.windows_only
def test_legacy_backfill_excludes_profile_recreation_on_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_legacy_backfill_excludes_profile_recreation(tmp_path, monkeypatch)


@pytest.mark.linux_only
def test_resource_lease_timeout_fails_closed_on_linux(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_resource_lease_timeout_fails_closed(tmp_path, monkeypatch)


@pytest.mark.macos_only
def test_resource_lease_timeout_fails_closed_on_macos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_resource_lease_timeout_fails_closed(tmp_path, monkeypatch)


@pytest.mark.windows_only
def test_resource_lease_timeout_fails_closed_on_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_resource_lease_timeout_fails_closed(tmp_path, monkeypatch)

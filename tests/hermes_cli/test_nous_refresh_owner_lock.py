"""Cross-PROCESS refresh ownership for the Nous single-use refresh token.

A single-use refresh token may be redeemed exactly once. Hermes routinely runs more than one process
against the same account (the desktop backend's ``hermes serve`` AND the messaging gateway, or two
installs sharing a HERMES_HOME), and each runs its own keepalive. ``_nous_shared_store_lock``
serializes each read and each WRITE, but not the read -> POST -> write SEQUENCE: two processes both
read the same grant, both redeem it, and the Portal retires the original and revokes the whole
session chain as a token-theft signal (``refresh_token_reused``).

``_nous_refresh_owner_lock`` makes that window exclusive across processes. These tests drive the
lock through real subprocesses (a thread lock would not prove cross-process behaviour).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _child_script() -> str:
    """A child that acquires the owner lock, records overlap, and holds it briefly."""
    return textwrap.dedent(
        """
        import sys, time, os
        sys.path.insert(0, sys.argv[1])
        from hermes_cli.auth_nous import _nous_refresh_owner_lock

        marker = sys.argv[2]
        hold = float(sys.argv[3])
        with _nous_refresh_owner_lock(timeout_seconds=30.0):
            with open(marker, "a", encoding="utf-8") as fh:
                fh.write("enter\\n")
                fh.flush()
            time.sleep(hold)
            with open(marker, "a", encoding="utf-8") as fh:
                fh.write("exit\\n")
        """
    )


@pytest.fixture()
def shared_home(tmp_path, monkeypatch):
    """A HERMES_HOME whose shared auth dir is under the tmp path."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_owner_lock_serializes_two_processes(shared_home, tmp_path):
    """Two real processes must never hold the lock at once."""
    marker = tmp_path / "events.txt"
    env = dict(os.environ)
    env["HERMES_HOME"] = str(shared_home)

    script = tmp_path / "child.py"
    script.write_text(_child_script(), encoding="utf-8")

    procs = [
        subprocess.Popen(
            [sys.executable, str(script), str(REPO_ROOT), str(marker), "0.7"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        for _ in range(2)
    ]
    for proc in procs:
        out, err = proc.communicate(timeout=60)
        assert proc.returncode == 0, f"child failed: {err.decode(errors='replace')}"

    events = marker.read_text(encoding="utf-8").split()
    assert events.count("enter") == 2, events
    # Strict interleaving: enter/exit/enter/exit — never enter/enter/.../exit/exit.
    assert events == ["enter", "exit", "enter", "exit"], (
        f"the owner lock did not serialize the two processes: {events}")


def test_owner_lock_is_reentrant_within_one_process(shared_home):
    """A nested acquisition in the same thread must not deadlock."""
    from hermes_cli.auth_nous import _nous_refresh_owner_lock

    with _nous_refresh_owner_lock(timeout_seconds=5.0) as outer:
        assert outer is True
        with _nous_refresh_owner_lock(timeout_seconds=5.0) as inner:
            assert inner is True


def test_timeout_yields_false_instead_of_raising(shared_home):
    """Losing the ownership race must DEGRADE, never raise.

    ``_file_lock`` raises ``TimeoutError`` on contention. If that escaped a refresh path it would
    abort the caller's whole credential resolve — the opposite of the guard's purpose. The context
    manager must swallow it and report ``owned=False`` so the caller adopts a peer's rotation
    instead of redeeming a single-use token a second time.
    """
    import threading
    import time

    from hermes_cli.auth_nous import _nous_refresh_owner_lock

    holding = threading.Event()
    release = threading.Event()

    def _hold():
        with _nous_refresh_owner_lock(timeout_seconds=30.0):
            holding.set()
            release.wait(30.0)

    holder = threading.Thread(target=_hold, daemon=True)
    holder.start()
    assert holding.wait(10.0), "holder never acquired the lock"
    try:
        with _nous_refresh_owner_lock(timeout_seconds=1.0) as owned:
            assert owned is False, "a contended acquisition must report owned=False"
    finally:
        release.set()
        holder.join(10.0)


def test_uncontended_acquisition_reports_owned(shared_home):
    """The happy path must report ownership so the caller is allowed to POST."""
    from hermes_cli.auth_nous import _nous_refresh_owner_lock

    with _nous_refresh_owner_lock(timeout_seconds=5.0) as owned:
        assert owned is True


def test_owner_lock_path_lives_beside_the_shared_store(shared_home):
    from hermes_cli.auth_nous import (
        NOUS_REFRESH_OWNER_LOCK_FILENAME,
        _nous_refresh_owner_lock_path,
        _nous_shared_auth_dir,
    )

    assert _nous_refresh_owner_lock_path().parent == _nous_shared_auth_dir()
    assert _nous_refresh_owner_lock_path().name == NOUS_REFRESH_OWNER_LOCK_FILENAME


def test_second_process_waits_rather_than_posting_concurrently(shared_home, tmp_path):
    """The waiter must observe the holder's write before it can proceed.

    Emulates the real transaction: the first process "rotates" (writes a new refresh token under
    the lock), the second must see that value when it acquires — which is what lets it adopt the
    rotation instead of redeeming the spent token.
    """
    store = shared_home / "shared" / "nous_auth.json"
    store.parent.mkdir(parents=True, exist_ok=True)

    script = textwrap.dedent(
        """
        import sys, json, time
        sys.path.insert(0, sys.argv[1])
        from hermes_cli.auth_nous import _nous_refresh_owner_lock

        store = sys.argv[2]
        tag = sys.argv[3]
        delay = float(sys.argv[4])
        with _nous_refresh_owner_lock(timeout_seconds=30.0):
            if tag == "first":
                time.sleep(delay)
                json.dump({"refresh_token": "rotated-by-first"}, open(store, "w"))
            else:
                # Must run after the first process persisted its rotation.
                data = json.load(open(store))
                assert data["refresh_token"] == "rotated-by-first", data
        """
    )
    script_path = tmp_path / "child2.py"
    script_path.write_text(script, encoding="utf-8")

    env = dict(os.environ)
    env["HERMES_HOME"] = str(shared_home)

    first = subprocess.Popen(
        [sys.executable, str(script_path), str(REPO_ROOT), str(store), "first", "1.0"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    time.sleep(0.3)  # let the first process take the lock
    second = subprocess.Popen(
        [sys.executable, str(script_path), str(REPO_ROOT), str(store), "second", "0.0"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    out1, err1 = first.communicate(timeout=60)
    out2, err2 = second.communicate(timeout=60)
    assert first.returncode == 0, err1.decode(errors="replace")
    assert second.returncode == 0, (
        "the second process did not observe the first process's rotation; "
        f"stderr: {err2.decode(errors='replace')}")
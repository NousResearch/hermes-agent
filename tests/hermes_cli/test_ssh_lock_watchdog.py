"""TDD for Desktop SSH lock/log watchdog (D6)."""

from __future__ import annotations

import inspect
import json
import os
import re
import threading
from pathlib import Path

from hermes_cli import web_server


def test_watchdog_start_is_noop_without_owner_nonce(monkeypatch):
    monkeypatch.setattr(web_server, "_SSH_OWNER_NONCE", None)
    started = []
    monkeypatch.setattr(
        threading,
        "Thread",
        lambda *args, **kwargs: started.append(kwargs) or (_ for _ in ()).throw(AssertionError("thread")),
    )
    web_server._start_ssh_lock_watchdog()
    assert started == []


def test_watchdog_waits_for_missing_lock_and_does_not_exit():
    action, seen, missing = web_server._ssh_lock_watchdog_decision(
        seen_lock=False,
        lock_exists=False,
        lock_readable=False,
        lock_nonce=None,
        our_nonce="0123456789abcdef",
        log_nlink=1,
        missing_polls=0,
    )
    assert action == "wait"
    assert seen is False
    assert missing == 0


def test_watchdog_still_waits_after_many_absent_polls():
    action, seen, missing = web_server._ssh_lock_watchdog_decision(
        seen_lock=False,
        lock_exists=False,
        lock_readable=False,
        lock_nonce=None,
        our_nonce="0123456789abcdef",
        log_nlink=1,
        missing_polls=80,
    )
    assert action == "wait"
    assert seen is False


def test_watchdog_exits_after_two_missing_polls_once_seen():
    first, seen, missing = web_server._ssh_lock_watchdog_decision(
        seen_lock=True,
        lock_exists=False,
        lock_readable=False,
        lock_nonce=None,
        our_nonce="0123456789abcdef",
        log_nlink=1,
        missing_polls=0,
    )
    assert first == "continue"
    assert seen is True
    assert missing == 1

    second, seen, missing = web_server._ssh_lock_watchdog_decision(
        seen_lock=True,
        lock_exists=False,
        lock_readable=False,
        lock_nonce=None,
        our_nonce="0123456789abcdef",
        log_nlink=1,
        missing_polls=missing,
    )
    assert second == "exit"
    assert missing == 2


def test_watchdog_exits_when_lock_nonce_differs():
    action, seen, missing = web_server._ssh_lock_watchdog_decision(
        seen_lock=True,
        lock_exists=True,
        lock_readable=True,
        lock_nonce="fedcba9876543210",
        our_nonce="0123456789abcdef",
        log_nlink=1,
        missing_polls=0,
    )
    assert action == "exit"
    assert seen is True
    assert missing == 0


def test_watchdog_exits_when_log_nlink_is_zero():
    action, _, _ = web_server._ssh_lock_watchdog_decision(
        seen_lock=True,
        lock_exists=True,
        lock_readable=True,
        lock_nonce="0123456789abcdef",
        our_nonce="0123456789abcdef",
        log_nlink=0,
        missing_polls=0,
    )
    assert action == "exit"


def test_watchdog_keeps_serving_when_lock_unreadable():
    action, seen, missing = web_server._ssh_lock_watchdog_decision(
        seen_lock=True,
        lock_exists=True,
        lock_readable=False,
        lock_nonce=None,
        our_nonce="0123456789abcdef",
        log_nlink=1,
        missing_polls=3,
    )
    assert action == "continue"
    assert seen is True
    assert missing == 0


def test_watchdog_matching_lock_resets_missing_polls():
    action, seen, missing = web_server._ssh_lock_watchdog_decision(
        seen_lock=True,
        lock_exists=True,
        lock_readable=True,
        lock_nonce="0123456789abcdef",
        our_nonce="0123456789abcdef",
        log_nlink=1,
        missing_polls=1,
    )
    assert action == "continue"
    assert seen is True
    assert missing == 0


def test_watchdog_observes_unlinked_log_nlink(tmp_path: Path):
    log_path = tmp_path / "0123456789abcdef.log"
    log_path.write_text("ready\n", encoding="utf-8")
    fd = os.open(log_path, os.O_RDONLY)
    try:
        os.unlink(log_path)
        assert os.fstat(fd).st_nlink == 0
        assert web_server._ssh_log_is_unlinked(fd) is True
    finally:
        os.close(fd)


def test_watchdog_reads_spawn_nonce_from_lock_file(tmp_path: Path):
    lock_path = tmp_path / "backend.lock.json"
    lock_path.write_text(json.dumps({"spawnNonce": "0123456789abcdef"}), encoding="utf-8")
    exists, readable, nonce = web_server._ssh_lock_snapshot(str(lock_path))
    assert exists is True
    assert readable is True
    assert nonce == "0123456789abcdef"


def test_watchdog_unreadable_lock_snapshot(tmp_path: Path):
    lock_path = tmp_path / "backend.lock.json"
    lock_path.write_text("{not-json", encoding="utf-8")
    exists, readable, nonce = web_server._ssh_lock_snapshot(str(lock_path))
    assert exists is True
    assert readable is False
    assert nonce is None


def test_watchdog_contract_has_no_age_pkill_or_parent_pid():
    source = inspect.getsource(web_server._start_ssh_lock_watchdog)
    source += inspect.getsource(web_server._ssh_lock_watchdog_decision)
    body = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith(('"', "'", "#"))
    )
    lowered = body.lower()
    assert "hermes_parent_pid" not in lowered
    assert "pkill" not in lowered
    assert "killpg" not in lowered
    assert re.search(r"\bage\b", lowered) is None
    assert "serve-ssh-lock-watchdog" in source
    assert "os._exit" in source
    signature = inspect.signature(web_server._start_ssh_lock_watchdog)
    assert list(signature.parameters) == []


def test_watchdog_poll_default_is_two_seconds():
    source = inspect.getsource(web_server._start_ssh_lock_watchdog)
    assert "2.0" in source or "2" in source

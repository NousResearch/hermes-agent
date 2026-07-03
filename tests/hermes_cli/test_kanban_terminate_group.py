"""Focused tests for the killpg group-termination patch in
_terminate_reclaimed_worker. Run against the patched module."""
import os
import signal
import types

import hermes_cli.kanban_db as kb


def _claimer_host():
    # claim_lock must start with "<host>:" for host-local termination.
    return kb._claimer_id().split(":", 1)[0]


def test_group_kill_used_on_real_path(monkeypatch):
    """signal_fn=None → deliver via killpg(getpgid(pid)), not os.kill."""
    pgid_calls = []
    monkeypatch.setattr(kb.os, "getpgid", lambda pid: 4242)
    monkeypatch.setattr(kb.os, "killpg", lambda pgid, sig: pgid_calls.append((pgid, sig)))
    # os.kill must NOT be used on the group path; make it explode if called.
    monkeypatch.setattr(kb.os, "kill", lambda *a: (_ for _ in ()).throw(AssertionError("os.kill used")))
    # Leader reported dead after first SIGTERM → terminated True, no SIGKILL.
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)

    info = kb._terminate_reclaimed_worker(1234, f"{_claimer_host()}:999")
    assert info["terminated"] is True
    assert pgid_calls == [(4242, signal.SIGTERM)]


def test_falls_back_to_leader_pid_when_group_gone(monkeypatch):
    """killpg raising ProcessLookupError → fall back to os.kill(pid)."""
    kill_calls = []
    def _boom_pgid(pgid, sig):
        raise ProcessLookupError
    monkeypatch.setattr(kb.os, "getpgid", lambda pid: 4242)
    monkeypatch.setattr(kb.os, "killpg", _boom_pgid)
    monkeypatch.setattr(kb.os, "kill", lambda pid, sig: kill_calls.append((pid, sig)))
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)

    info = kb._terminate_reclaimed_worker(1234, f"{_claimer_host()}:999")
    # Fallback kill(pid) raised nothing → then _pid_alive False → terminated.
    assert info["terminated"] is True
    assert kill_calls == [(1234, signal.SIGTERM)]


def test_signal_fn_hook_preserved_single_pid(monkeypatch):
    """Injected signal_fn (tests) stays single-pid — killpg never touched."""
    monkeypatch.setattr(kb.os, "killpg", lambda *a: (_ for _ in ()).throw(AssertionError("killpg used")))
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    seen = []
    info = kb._terminate_reclaimed_worker(
        1234, f"{_claimer_host()}:999", signal_fn=lambda p, s: seen.append((p, s))
    )
    assert info["terminated"] is True
    assert seen == [(1234, signal.SIGTERM)]

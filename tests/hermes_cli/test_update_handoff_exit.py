"""Windows hand-off child must hard-exit once the update is durably done (#93581).

The re-exec'd venv child (spawned by
``_reexec_dependency_sync_off_windows_shim`` with ``HERMES_UPDATE_REEXEC=1``)
completes all update work — the receipt records ``success`` / ``completed at
command boundary`` — but then hangs in interpreter shutdown on a leftover
non-daemon thread, freezing the PowerShell window for minutes. The fix: on
the hand-off path only, after the receipt is finalized, the lock released,
and stdio restored, flush and ``os._exit(code)`` instead of unwinding.

These tests pin: the hard exit fires (with the right code) only when the
re-exec marker env is set, it happens after lock release + stdio restore,
early ``SystemExit`` codes propagate to it, and real exceptions keep the
normal raise path (traceback intact, no hard exit).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hermes_cli.main as main_mod
from hermes_cli import update_cmd
from hermes_cli.main import cmd_update


class _FakeLock:
    def __init__(self, events):
        self._events = events

    def acquire(self):
        self._events.append("acquire")
        return True

    def release(self):
        self._events.append("release")


# Events from the most recent _run_cmd_update call, also filled in when
# cmd_update propagates an exception (the return value is unreachable then).
_LAST = {}


def _run_cmd_update(monkeypatch, impl, *, reexec: bool):
    """Run cmd_update with everything external mocked; return the events."""
    events = {"order": [], "exit_codes": [], "receipts": []}

    def fake_impl(args, gateway_mode=False):
        events["order"].append("impl")
        impl(args, gateway_mode=gateway_mode)

    def fake_finalize_io(state):
        events["order"].append("restore-stdio")

    def fake_receipt(code, reason):
        events["receipts"].append((code, reason))

    def fake_exit(code):
        events["order"].append("hard-exit")
        events["exit_codes"].append(code)

    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: False)
    monkeypatch.setattr("hermes_cli.config.detect_install_method", lambda root: "git")
    monkeypatch.setattr("hermes_cli.update_lock.UpdateLock", lambda: _FakeLock(events["order"]))
    monkeypatch.setattr(update_cmd, "_cmd_update_impl", fake_impl)
    monkeypatch.setattr(main_mod, "_install_hangup_protection", lambda gateway_mode=False: None)
    monkeypatch.setattr(main_mod, "_finalize_update_output", fake_finalize_io)
    monkeypatch.setattr("hermes_cli.update_receipt.finalize_pending_update_receipt", fake_receipt)
    monkeypatch.setattr("os._exit", fake_exit)
    if reexec:
        monkeypatch.setenv("HERMES_UPDATE_REEXEC", "1")
    else:
        monkeypatch.delenv("HERMES_UPDATE_REEXEC", raising=False)

    args = SimpleNamespace(plan=False, check=False, gateway=False, branch=None)
    try:
        cmd_update(args)
    finally:
        _LAST.clear()
        _LAST.update(events)
    return events


def _noop_impl(args, gateway_mode=False):
    return None


def test_handoff_child_hard_exits_zero_after_success(monkeypatch):
    events = _run_cmd_update(monkeypatch, _noop_impl, reexec=True)
    assert events["exit_codes"] == [0]
    assert events["receipts"] == [(0, "completed at command boundary")]
    # The hard exit is the last thing, after lock release and stdio restore.
    assert events["order"] == ["acquire", "impl", "release", "restore-stdio", "hard-exit"]


def test_non_handoff_run_never_hard_exits(monkeypatch):
    events = _run_cmd_update(monkeypatch, _noop_impl, reexec=False)
    assert events["exit_codes"] == []
    assert "hard-exit" not in events["order"]
    assert events["receipts"] == [(0, "completed at command boundary")]


def test_handoff_child_propagates_early_systemexit_code(monkeypatch):
    def early_refusal(args, gateway_mode=False):
        raise SystemExit(3)

    with pytest.raises(SystemExit) as excinfo:
        _run_cmd_update(monkeypatch, early_refusal, reexec=True)
    assert excinfo.value.code == 3
    # The finally-block hard exit ran (before the re-raise propagated)
    # and carried the early exit's code, not a blanket 0.
    assert _LAST["exit_codes"] == [3]


def test_handoff_child_systemexit_none_means_zero(monkeypatch):
    def bare_exit(args, gateway_mode=False):
        raise SystemExit(None)

    with pytest.raises(SystemExit):
        _run_cmd_update(monkeypatch, bare_exit, reexec=True)
    assert _LAST["exit_codes"] == [0]


def test_unhandled_exception_keeps_raise_path_no_hard_exit(monkeypatch):
    def boom(args, gateway_mode=False):
        raise RuntimeError("update tail exploded")

    # With os._exit patched to record, the re-raised RuntimeError reaches
    # pytest with the finally block already run — and no hard exit fires.
    with pytest.raises(RuntimeError, match="update tail exploded"):
        _run_cmd_update(monkeypatch, boom, reexec=True)
    assert "hard-exit" not in _LAST["order"]
    assert _LAST["receipts"] == [(1, "RuntimeError: update tail exploded")]

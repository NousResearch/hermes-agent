"""Tests: the plugin update-check cadence — clock-gated, receipt-surfaced,
apply-explicit. All seams injected; hermetic."""

from __future__ import annotations

import time

import pytest

import hermes_cli.plugins_cadence as cad


@pytest.fixture
def homed(tmp_path, monkeypatch):
    """Cadence state (markers) inside a temp hermes home."""
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    return tmp_path


class _Result:
    def __init__(self, name, klass="git", update_available=None, needs_fixing=None):
        self.name = name
        self.klass = klass
        self.update_available = update_available
        self.needs_fixing = needs_fixing

    def to_json(self):
        return {"name": self.name, "class": self.klass}


def _no_results(plugins_dir, **k):
    return []


def test_clock_gate_due_when_never_run(homed):
    assert cad.check_due(now=1000.0, interval_hours=24) is True


def test_clock_gate_not_due_within_interval(homed, monkeypatch):
    marker = homed / "plugin-update-checks"
    marker.mkdir()
    (marker / "last-run").write_text("x", encoding="utf-8")
    # fresh mtime
    recent = time.time() - 100
    import os

    os.utime(marker / "last-run", (recent, recent))
    assert cad.check_due(now=time.time(), interval_hours=24) is False


def test_zero_interval_disables(homed):
    assert cad.check_due(now=time.time(), interval_hours=0) is False


def test_interval_reads_config(homed):
    assert cad.check_interval_hours(lambda s, k: None) == 24
    assert cad.check_interval_hours(lambda s, k: 6) == 6.0
    assert cad.check_interval_hours(lambda s, k: 0) == 0.0
    assert cad.check_interval_hours(lambda s, k: "garbage") == 24


def test_run_writes_receipt_and_marker(homed):
    calls = {"checks": 0}
    receipt_calls = []

    import pm.receipt as receipt_mod

    orig_begin, orig_finalize = receipt_mod.begin, receipt_mod.finalize
    monkey = pytest.MonkeyPatch()
    monkey.setattr(receipt_mod, "begin", lambda kind: receipt_calls.append(("begin", kind)))
    monkey.setattr(
        receipt_mod, "finalize",
        lambda outcome, exit_code=0: receipt_calls.append(("finalize", outcome)),
    )
    try:
        def fake_run_checks(plugins_dir):
            calls["checks"] += 1
            return [_Result("plug", update_available=True)]

        results = cad.run_scheduled_check(
            run_checks_fn=fake_run_checks,
            plugins_dir=homed / "plugins",
            now=time.time(),
        )
        assert calls["checks"] == 1
        assert results[0].name == "plug"
        assert ("begin", "plugin-check") in receipt_calls
        assert ("finalize", "updates-available") in receipt_calls
        # marker stamped → second run is a no-op
        assert cad.run_scheduled_check(
            run_checks_fn=fake_run_checks,
            plugins_dir=homed / "plugins",
            now=time.time(),
        ) is None
        assert calls["checks"] == 1
    finally:
        monkey.undo()


def test_needs_fixing_logged_not_applied(homed):
    applied = []

    import pm.receipt as receipt_mod

    monkey = pytest.MonkeyPatch()
    monkey.setattr(receipt_mod, "begin", lambda kind: None)
    monkey.setattr(receipt_mod, "finalize", lambda outcome, exit_code=0: None)
    try:
        results = cad.run_scheduled_check(
            run_checks_fn=lambda d: [_Result("plug", needs_fixing="mismatch")],
            plugins_dir=homed / "plugins",
            apply_updates_fn=applied.append,
            now=time.time(),
        )
        assert applied == []  # needs-fixing is never auto-applied
    finally:
        monkey.undo()


def test_auto_apply_only_git_rows_and_only_when_opted_in(homed):
    applied = []

    import pm.receipt as receipt_mod

    monkey = pytest.MonkeyPatch()
    monkey.setattr(receipt_mod, "begin", lambda kind: None)
    monkey.setattr(receipt_mod, "finalize", lambda outcome, exit_code=0: None)
    try:
        results = [_Result("gitplug", klass="git", update_available=True),
                   _Result("pipplug", klass="pip", update_available=True)]
        # opted OUT: nothing applied
        cad.run_scheduled_check(
            run_checks_fn=lambda d: results,
            plugins_dir=homed / "plugins",
            apply_updates_fn=applied.append,
            config_get=lambda s, k: (False if k == "auto_apply" else None),
            now=time.time(),
        )
        assert applied == []  # auto_apply False → nothing

        # opted IN: only the git row applies
        cad._markers_dir().joinpath("last-run").unlink()
        cad.run_scheduled_check(
            run_checks_fn=lambda d: results,
            plugins_dir=homed / "plugins",
            apply_updates_fn=applied.append,
            config_get=lambda s, k: (True if k == "auto_apply" else None),
            now=time.time(),
        )
        assert applied == ["gitplug"]
    finally:
        monkey.undo()


def test_check_failure_is_never_fatal_and_stamps_marker(homed):
    import pm.receipt as receipt_mod

    monkey = pytest.MonkeyPatch()
    monkey.setattr(receipt_mod, "begin", lambda kind: None)
    monkey.setattr(receipt_mod, "finalize", lambda outcome, exit_code=0: None)
    try:
        def boom(d):
            raise RuntimeError("network down")

        results = cad.run_scheduled_check(
            run_checks_fn=boom,
            plugins_dir=homed / "plugins",
            now=time.time(),
        )
        assert results == []
        # marker stamped → a failing check doesn't hammer the network
        assert (homed / "plugin-update-checks" / "last-run").is_file()
    finally:
        monkey.undo()

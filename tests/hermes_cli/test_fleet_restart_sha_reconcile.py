"""``fleet_restart_pending`` must reconcile with the live fleet's stamped ``code_sha``.

The marker is a pull→restart obligation, but the restart itself can be performed by
anything — a supervisor, an ops script, an agent's fallback restart — so updater-side
bookkeeping cannot be the only thing that retires it. The marker records its own
``expected_sha``, and every live gateway stamps the code it is actually running into
``gateway_state.json.code_sha`` (``gateway/status.py``, reported by
``collect_fleet_versions()`` — the same field the updater's own verify stage trusts).
These tests pin the three contract cases:

1. marker present, every live gateway serves ``expected_sha`` → no warning, marker
   retired (one line logged), nothing restarted;
2. marker present, the fleet lags (stale/unknown rows, or only a partial match) →
   warning and catch-up restart unchanged;
3. no marker → behavior identical to before.

No gateway is started and no service is touched: the fleet snapshot is the stamped
runtime state, stubbed here.
"""

from __future__ import annotations

import logging

import pytest

import hermes_cli.update_receipt as update_receipt
from hermes_cli import main as hermes_main
from hermes_cli import update_cmd, update_cmd_fleet

from tests.hermes_cli.test_update_fleet_restart_pending import (
    _make_up_to_date_side_effect, _patch_update_deps, _update_args,
)

SHA = "e" * 40
OTHER_SHA = "7" * 40
WARNING = "did not restart running gateways"


def _marker():
    return update_cmd._fleet_restart_pending_marker_path()


def _arm_marker(monkeypatch, *, expected_sha=SHA, checkout_sha=SHA):
    """Write the marker and make the checkout HEAD agree with it."""
    update_cmd._write_fleet_restart_pending_marker(expected_sha=expected_sha)
    assert _marker().is_file()
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: checkout_sha)
    monkeypatch.setattr(update_cmd_fleet, "_current_checkout_sha", lambda: checkout_sha)


def _fleet(monkeypatch, rows):
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda **_kwargs: rows)


def _row(profile="default", *, sha=SHA, state="current", pid=42):
    return {
        "profile": profile, "pid": pid, "code_sha": sha,
        "code_version": "0.21.0", "state": state,
    }


def _restart_recorder(monkeypatch):
    seen = {"ran": False}

    def _restart():
        seen["ran"] = True
        return True

    monkeypatch.setattr(update_cmd, "_run_pending_fleet_restart", _restart)
    monkeypatch.setattr(update_cmd_fleet, "_run_pending_fleet_restart", _restart)
    return seen


def _retirement_records(caplog):
    return [record.getMessage() for record in caplog.records if "retired" in record.getMessage()]


# ── 1. marker satisfied by the live fleet → silent, retired, no restart ──────


def test_startup_silent_and_retires_marker_when_live_fleet_serves_expected_sha(
    monkeypatch, capsys, caplog
):
    _arm_marker(monkeypatch)
    _fleet(monkeypatch, [_row()])

    with caplog.at_level(logging.INFO, logger="hermes_cli.update_cmd"):
        update_cmd_fleet._warn_pending_fleet_restart_on_startup()

    captured = capsys.readouterr()
    assert captured.err == "" and captured.out == ""
    assert not _marker().exists(), "a satisfied marker must not survive to warn again"
    assert any(SHA[:10] in line for line in _retirement_records(caplog)), caplog.text


def test_update_no_op_when_marker_already_satisfied(monkeypatch, tmp_path, capsys):
    args = _update_args()
    _patch_update_deps(monkeypatch, tmp_path, _make_up_to_date_side_effect(sha=SHA))
    _arm_marker(monkeypatch)
    _fleet(monkeypatch, [_row()])
    seen = _restart_recorder(monkeypatch)

    hermes_main.cmd_update(args)

    out = capsys.readouterr().out
    assert seen["ran"] is False, "the fleet already serves expected_sha: no restart"
    assert WARNING not in out
    assert not _marker().exists()


def test_no_marker_keeps_existing_startup_behavior(monkeypatch, capsys, caplog):
    """Nothing pending: no warning, nothing logged, no state written."""
    _fleet(monkeypatch, [_row()])
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: SHA)
    monkeypatch.setattr(update_cmd_fleet, "_current_checkout_sha", lambda: SHA)

    with caplog.at_level(logging.INFO, logger="hermes_cli.update_cmd"):
        assert update_cmd_fleet._pending_fleet_restart_needed() is False
        update_cmd_fleet._warn_pending_fleet_restart_on_startup()

    captured = capsys.readouterr()
    assert captured.err == "" and captured.out == ""
    assert _retirement_records(caplog) == []
    assert not _marker().exists()


# ── 2. the fleet really lags → warning + catch-up unchanged ─────────────────


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param([_row(sha=OTHER_SHA, state="stale")], id="whole-fleet-stale"),
        pytest.param([_row("alpha"), _row("beta", sha=OTHER_SHA, state="stale")], id="partial-match"),
        pytest.param([_row(sha=None, state="unknown")], id="unknown-identity"),
        pytest.param([], id="probe-empty"),
    ],
)
def test_update_still_warns_and_catches_up_when_fleet_lags(
    monkeypatch, tmp_path, capsys, caplog, rows
):
    args = _update_args()
    _patch_update_deps(monkeypatch, tmp_path, _make_up_to_date_side_effect(sha=SHA))
    _arm_marker(monkeypatch)
    _fleet(monkeypatch, rows)
    seen = _restart_recorder(monkeypatch)

    with caplog.at_level(logging.INFO, logger="hermes_cli.update_cmd"):
        hermes_main.cmd_update(args)

    out = capsys.readouterr().out
    assert WARNING in out, "a lagging fleet must keep the protection"
    assert seen["ran"] is True
    assert _retirement_records(caplog) == [], "nothing was retired"


def test_startup_still_warns_when_fleet_lags(monkeypatch, capsys):
    _arm_marker(monkeypatch)
    _fleet(monkeypatch, [_row("alpha"), _row("beta", sha=OTHER_SHA, state="stale")])

    update_cmd_fleet._warn_pending_fleet_restart_on_startup()

    assert WARNING in capsys.readouterr().err
    assert _marker().exists()

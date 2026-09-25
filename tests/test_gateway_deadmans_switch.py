"""Deadman's switch tests (LAG-677).

The gateway startup block must verify the transcriber stack once per gateway
start: fresh daemon heartbeat AND a live watchdog process → silent; anything
else → exactly one macOS notification. Never raises (a deadman failure must
never affect the startup it observes), never touches anything but its own
env-read state and subprocess seams.

Import-lightness contract (mirrors test_startup_watchdog.py): the module is
stdlib-only and must never import gateway or hermes_cli.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hermes_gateway_deadmans_switch as dd  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """No real env leakage; every test gets its own home."""
    for name in (
        dd.ENV_DEADMAN,
        dd.ENV_DEADMAN_HEARTBEAT,
        dd.ENV_DEADMAN_LABEL,
        dd.ENV_DEADMAN_STALE_S,
    ):
        monkeypatch.delenv(name, raising=False)


NOW = 1_000_000.0


def _touch_heartbeat(path: Path, *, age_s: float = 5.0, writer: str = "daemon") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "writer": writer}
    path.write_text(json.dumps(payload), encoding="utf-8")
    stamp = NOW - age_s  # the file was written age_s seconds before "now"
    os.utime(path, (stamp, stamp))


HEALTHY_PRESENCE = {dd.DEFAULT_LABEL: True, dd.DEFAULT_WATCHDOG_LABEL: True}
NO_WATCHDOG = {dd.DEFAULT_LABEL: True, dd.DEFAULT_WATCHDOG_LABEL: False}
NOTHING_LOADED = {dd.DEFAULT_LABEL: False, dd.DEFAULT_WATCHDOG_LABEL: False}


# ── pure helpers ──────────────────────────────────────────────────────────────


def test_heartbeat_writer_defaults_to_daemon():
    assert dd.heartbeat_writer(None) == "daemon"
    assert dd.heartbeat_writer({}) == "daemon"
    assert dd.heartbeat_writer({"writer": "manual"}) == "manual"
    assert dd.heartbeat_writer({"writer": 3}) == "daemon"


def test_labels_exact_match_not_substring():
    out = dd.labels_in_launchctl_list(
        "65621\t75\tai.hermes.gateway\n-\t0\tcom.alex.transcriber.watchdog\n"
        "89506\t0\tcom.alex.transcriber\n",
        (dd.DEFAULT_LABEL, dd.DEFAULT_WATCHDOG_LABEL),
    )
    assert out == {dd.DEFAULT_LABEL: True, dd.DEFAULT_WATCHDOG_LABEL: True}


def test_labels_missing_watchdog_row():
    out = dd.labels_in_launchctl_list(
        "89506\t0\tcom.alex.transcriber\n",
        (dd.DEFAULT_LABEL, dd.DEFAULT_WATCHDOG_LABEL),
    )
    assert out == {dd.DEFAULT_LABEL: True, dd.DEFAULT_WATCHDOG_LABEL: False}


def test_stale_threshold_clamps_and_defaults(monkeypatch):
    monkeypatch.delenv(dd.ENV_DEADMAN_STALE_S, raising=False)
    assert dd._stale_threshold() == dd.DEFAULT_STALE_S
    monkeypatch.setenv(dd.ENV_DEADMAN_STALE_S, "120")
    assert dd._stale_threshold() == 120.0
    monkeypatch.setenv(dd.ENV_DEADMAN_STALE_S, "5")  # below floor → floor
    assert dd._stale_threshold() == dd._MIN_STALE_S
    monkeypatch.setenv(dd.ENV_DEADMAN_STALE_S, "99999")  # above cap → cap
    assert dd._stale_threshold() == dd._MAX_STALE_S
    monkeypatch.setenv(dd.ENV_DEADMAN_STALE_S, "garbage")
    assert dd._stale_threshold() == dd.DEFAULT_STALE_S


# ── assessment ───────────────────────────────────────────────────────────────


def test_assess_healthy_is_silent(tmp_path):
    hb = tmp_path / "hb.json"
    _touch_heartbeat(hb)
    state = dd.assess(
        heartbeat_path=str(hb),
        label=dd.DEFAULT_LABEL,
        watchdog_label=dd.DEFAULT_WATCHDOG_LABEL,
        now=NOW,
        launchctl=HEALTHY_PRESENCE,
    )
    assert state["healthy"] is True
    assert state["reasons"] == []


def test_assess_fresh_heartbeat_but_watchdog_dead(tmp_path):
    """The core incident: the transcriber watchdog was dead through the outage."""
    hb = tmp_path / "hb.json"
    _touch_heartbeat(hb)
    state = dd.assess(
        heartbeat_path=str(hb),
        label=dd.DEFAULT_LABEL,
        watchdog_label=dd.DEFAULT_WATCHDOG_LABEL,
        now=NOW,
        launchctl=NO_WATCHDOG,
    )
    assert state["healthy"] is False
    assert state["heartbeat_age_s"] == 5.0
    assert state["watchdog_process_present"] is False
    assert state["reasons"] == ["watchdog_process_absent"]


def test_assess_watchdog_alive_but_heartbeat_stale(tmp_path):
    hb = tmp_path / "hb.json"
    _touch_heartbeat(hb, age_s=dd.DEFAULT_STALE_S + 10)
    state = dd.assess(
        heartbeat_path=str(hb),
        label=dd.DEFAULT_LABEL,
        watchdog_label=dd.DEFAULT_WATCHDOG_LABEL,
        now=NOW,
        launchctl=HEALTHY_PRESENCE,
    )
    assert state["healthy"] is False
    assert state["watchdog_process_present"] is True
    assert any(r.startswith("heartbeat_stale") for r in state["reasons"])


def test_assess_missing_heartbeat_is_unhealthy(tmp_path):
    state = dd.assess(
        heartbeat_path=str(tmp_path / "absent.json"),
        label=dd.DEFAULT_LABEL,
        watchdog_label=dd.DEFAULT_WATCHDOG_LABEL,
        now=NOW,
        launchctl=HEALTHY_PRESENCE,
    )
    assert state["healthy"] is False
    assert "heartbeat_missing_or_unreadable" in state["reasons"]


def test_assess_manual_heartbeat_writer_is_unhealthy(tmp_path):
    """HB-12: an ops script refreshed the heartbeat — not daemon liveness."""
    hb = tmp_path / "hb.json"
    _touch_heartbeat(hb, writer="manual")
    state = dd.assess(
        heartbeat_path=str(hb),
        label=dd.DEFAULT_LABEL,
        watchdog_label=dd.DEFAULT_WATCHDOG_LABEL,
        now=NOW,
        launchctl=HEALTHY_PRESENCE,
    )
    assert state["healthy"] is False
    assert "heartbeat_writer_is_manual" in state["reasons"]


def test_assess_launchctl_failure_is_fail_closed(tmp_path, monkeypatch):
    """launchctl unreadable → the watchdog cannot be proven alive → unhealthy."""
    hb = tmp_path / "hb.json"
    _touch_heartbeat(hb)
    monkeypatch.setattr(dd, "launchctl_presence", lambda labels: None)
    state = dd.assess(
        heartbeat_path=str(hb),
        label=dd.DEFAULT_LABEL,
        watchdog_label=dd.DEFAULT_WATCHDOG_LABEL,
        now=NOW,
        launchctl=None,
    )
    assert state["healthy"] is False
    assert "watchdog_unverifiable (launchctl failed)" in state["reasons"]


# ── the switch ───────────────────────────────────────────────────────────────


def _switch(tmp_path, monkeypatch, launchctl, *, pause_path=None):
    hb = tmp_path / "hb.json"
    _touch_heartbeat(hb)
    monkeypatch.setenv(dd.ENV_DEADMAN_HEARTBEAT, str(hb))
    if pause_path is None:
        pause_path = str(tmp_path / "no-such-sentinel")
    return dd.run_gateway_deadmans_switch(
        now=NOW, launchctl=launchctl, pause_path=pause_path
    )


def test_switch_healthy_never_notifies(tmp_path, monkeypatch):
    state = _switch(tmp_path, monkeypatch, HEALTHY_PRESENCE)
    assert state["fired"] is False
    assert state["skipped"] is False
    assert state["healthy"] is True


def test_switch_fires_when_watchdog_dead(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dd, "notify", lambda m, t="x": calls.append(m) or True)
    state = _switch(tmp_path, monkeypatch, NO_WATCHDOG)
    assert state["fired"] is True
    assert len(calls) == 1
    assert "watchdog_process_absent" in calls[0]


def test_switch_silent_under_maintenance_pause(tmp_path, monkeypatch):
    sentinel = tmp_path / "sentinel"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("", encoding="utf-8")
    stamp = NOW - 60.0  # well within the 4h TTL
    os.utime(sentinel, (stamp, stamp))
    state = _switch(tmp_path, monkeypatch, NO_WATCHDOG, pause_path=str(sentinel))
    assert state == {
        "fired": False,
        "skipped": True,
        "reason": "maintenance_pause_sentinel",
    }


def test_switch_ignores_expired_pause_sentinel(tmp_path, monkeypatch):
    """WD-10 revision: a 5h-old sentinel is a forgotten one, not a pause."""
    calls = []
    monkeypatch.setattr(dd, "notify", lambda m, t="x": calls.append(m) or True)
    sentinel = tmp_path / "sentinel"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("", encoding="utf-8")
    stamp = NOW - (dd.PAUSE_TTL_S + 60.0)
    os.utime(sentinel, (stamp, stamp))
    state = _switch(tmp_path, monkeypatch, NO_WATCHDOG, pause_path=str(sentinel))
    assert state["fired"] is True  # expired sentinel must not silence the deadman


def test_switch_inert_without_heartbeat_file(tmp_path, monkeypatch):
    """Machines without the transcriber stack are untouched (LAG-677 gate)."""
    calls = []
    monkeypatch.setattr(dd, "notify", lambda m, t="x": calls.append(m) or True)
    monkeypatch.setenv(dd.ENV_DEADMAN_HEARTBEAT, str(tmp_path / "absent.json"))
    state = dd.run_gateway_deadmans_switch(
        now=NOW, launchctl=NOTHING_LOADED, pause_path=str(tmp_path / "s")
    )
    assert state == {"fired": False, "skipped": True, "reason": "heartbeat_absent"}
    assert calls == []


def test_switch_env_disable(tmp_path, monkeypatch):
    monkeypatch.setenv(dd.ENV_DEADMAN, "0")
    state = _switch(tmp_path, monkeypatch, NO_WATCHDOG)
    assert state == {"fired": False, "skipped": True, "reason": "disabled_by_env"}


def test_switch_never_raises_on_internal_error(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("assessment exploded")

    monkeypatch.setattr(dd, "assess", boom)
    state = _switch(tmp_path, monkeypatch, HEALTHY_PRESENCE)
    assert state["fired"] is False
    assert state["skipped"] is True
    assert state["reason"].startswith("internal_error:")


def test_switch_notify_failure_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "notify", lambda m, t="x": False)
    state = _switch(tmp_path, monkeypatch, NO_WATCHDOG)
    assert state["fired"] is False
    assert state["notified"] is False
    assert state["healthy"] is False


# ── wiring + import-lightness contract ───────────────────────────────────────


def test_main_wires_deadman_into_gateway_run_block():
    """The startup-liveness block must arm the deadman for gateway runs."""
    import hermes_cli.main as m

    src = Path(m.__file__).read_text(encoding="utf-8")
    assert "_argv_is_gateway_run(sys.argv[1:])" in src
    assert "hermes_gateway_deadmans_switch" in src
    assert "run_gateway_deadmans_switch" in src
    # The deadman arm must sit INSIDE the gateway-run gate (before main()).
    block = src.split("def _argv_is_gateway_run", 1)[1].split(
        "def _exit_after_oneshot"
    )[0]
    assert "run_gateway_deadmans_switch" in block


def test_deadmans_switch_module_is_stdlib_only():
    src = Path(dd.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "import yaml",
        "import requests",
        "from hermes_cli",
        "import gateway",
    ):
        assert forbidden not in src, forbidden

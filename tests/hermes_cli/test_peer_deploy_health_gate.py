"""Regression tests for issue #130708: peer deploy health gate vs stale platforms.

The Tier-2 gate must derive both ``ALL`` and ``CONNECTED`` from platform entries
stamped by the CURRENT gateway process (exact ``(pid, start_time)`` writer
identity). A stale row left by a dead process (e.g. a removed adapter still
present in ``gateway_state.json``) must never fail — or be blamed by — the
gate.
"""

from __future__ import annotations

import json
import subprocess
import sys

from hermes_cli import peer_deploy


LIVE_PID = 1707940
LIVE_START = 12345678.0


def _live(name="google_chat", state="connected", **extra):
    entry = {
        "state": state,
        "writer_pid": LIVE_PID,
        "writer_start_time": LIVE_START,
    }
    entry.update(extra)
    return entry


def _runtime(*, platforms, pid=LIVE_PID, start_time=LIVE_START):
    return {"pid": pid, "start_time": start_time, "platforms": platforms}


def _issue_runtime():
    """The live host shape from #130708: two live connected platforms plus a
    9-day-stale ``feishu`` row in ``retrying`` from a dead pid."""
    return _runtime(platforms={
        "google_chat": _live("google_chat", "connected"),
        "api_server": _live("api_server", "connected"),
        "feishu": {
            "state": "retrying",
            "error_message": "Feishu websocket link lost; rebuilding",
            "writer_pid": 1749354,
            "writer_start_time": 11111111.0,
        },
    })


# ── stale entries are excluded ──────────────────────────────────────────────


def test_stale_writer_pid_mismatch_excluded():
    owned = peer_deploy.owned_platforms(
        {"feishu": {"state": "retrying", "writer_pid": 1749354,
                    "writer_start_time": LIVE_START}},
        LIVE_PID, LIVE_START)
    assert owned == {}


def test_stale_writer_start_time_mismatch_excluded():
    owned = peer_deploy.owned_platforms(
        {"feishu": {"state": "retrying", "writer_pid": LIVE_PID,
                    "writer_start_time": 11111111.0}},
        LIVE_PID, LIVE_START)
    assert owned == {}


def test_legacy_entries_without_writer_identity_excluded():
    owned = peer_deploy.owned_platforms(
        {"feishu": {"state": "retrying"},
         "plain": {"state": "connected"}},
        LIVE_PID, LIVE_START)
    assert owned == {}


def test_non_dict_platform_values_excluded():
    owned = peer_deploy.owned_platforms(
        {"feishu": "connected", "other": None, "n": 3,
         "ok": _live("ok", "connected")},
        LIVE_PID, LIVE_START)
    assert sorted(owned) == ["ok"]


def test_non_dict_platforms_map_yields_empty():
    assert peer_deploy.owned_platforms(None, LIVE_PID, LIVE_START) == {}
    assert peer_deploy.owned_platforms(["feishu"], LIVE_PID, LIVE_START) == {}


# ── live entries are verified ─────────────────────────────────────────────


def test_live_matching_entries_are_owned():
    owned = peer_deploy.owned_platforms(
        {"google_chat": _live("google_chat", "connected"),
         "api_server": _live("api_server", "connected")},
        LIVE_PID, LIVE_START)
    assert sorted(owned) == ["api_server", "google_chat"]


def test_live_disconnected_owned_platform_is_still_listed():
    """A platform the current gateway owns but has not connected must still
    fail the gate (the Tier-2 intent): ownership filters rows, never states."""
    runtime = _runtime(platforms={
        "google_chat": _live("google_chat", "retrying"),
    })
    healthy, all_p, conn_p, missing = peer_deploy.evaluate_platform_gate(runtime)
    assert healthy is False
    assert all_p == ["google_chat"]
    assert conn_p == []
    assert missing == ["google_chat"]


def test_state_matching_is_case_insensitive_with_status_fallback():
    assert peer_deploy.platform_state({"state": "Connected"}) == "connected"
    assert peer_deploy.platform_state({"status": "connected"}) == "connected"
    assert peer_deploy.platform_state({"state": "retrying"}) == "retrying"
    assert peer_deploy.platform_state("connected") == ""
    assert peer_deploy.platform_state(None) == ""


# ── gate outcomes ─────────────────────────────────────────────────────────


def test_gate_succeeds_with_stale_entries_present():
    """The #130708 scenario: live owned platforms all connected, one stale
    dead-process row present — the gate must pass."""
    healthy, all_p, conn_p, missing = peer_deploy.evaluate_platform_gate(_issue_runtime())
    assert healthy is True
    assert all_p == ["api_server", "google_chat"]
    assert conn_p == ["api_server", "google_chat"]
    assert missing == []


def test_gate_failure_blames_only_live_owned_platforms():
    """When a live platform is down AND a stale row exists, the error must
    name only the live platform — never the tombstone."""
    runtime = _runtime(platforms={
        "google_chat": _live("google_chat", "connected"),
        "api_server": _live("api_server", "retrying"),
        "feishu": {"state": "retrying", "writer_pid": 1749354,
                   "writer_start_time": 11111111.0},
    })
    healthy, all_p, conn_p, missing = peer_deploy.evaluate_platform_gate(runtime)
    assert healthy is False
    assert all_p == ["api_server", "google_chat"]
    assert conn_p == ["google_chat"]
    assert missing == ["api_server"]
    assert "feishu" not in missing


def test_gate_empty_owned_reports_healthy_fail_open():
    """No owned rows (no live identity, or a pre-stamping state file) must not
    fail forever — same fail-open choice as the dashboard's ownership filter."""
    healthy, all_p, conn_p, missing = peer_deploy.evaluate_platform_gate(
        {"pid": LIVE_PID, "start_time": LIVE_START, "platforms": {
            "feishu": {"state": "retrying"}}})
    assert (healthy, all_p, conn_p, missing) == (True, [], [], [])
    healthy, _, _, _ = peer_deploy.evaluate_platform_gate({})
    assert healthy is True
    healthy, _ = peer_deploy.platforms_healthy([], [])
    assert healthy is True


def test_platforms_healthy_set_equality():
    assert peer_deploy.platforms_healthy(["a", "b"], ["b", "a"]) == (True, [])
    ok, missing = peer_deploy.platforms_healthy(["a", "b"], ["a"])
    assert ok is False and missing == ["b"]


def test_parse_probe_output_round_trip():
    all_p, conn_p = peer_deploy.parse_probe_output("ALL api_server,google_chat\nCONNECTED api_server,google_chat\n")
    assert all_p == ["api_server", "google_chat"]
    assert conn_p == ["api_server", "google_chat"]
    all_p, conn_p = peer_deploy.parse_probe_output("ALL none\nCONNECTED none\n")
    assert (all_p, conn_p) == ([], [])


# ── the remote probe snippet itself ─────────────────────────────────────────


def test_probe_script_filters_by_writer_identity():
    script = peer_deploy.HEALTH_PROBE_SCRIPT
    assert "writer_pid" in script
    assert "writer_start_time" in script
    assert "d.get('pid')" in script and "d.get('start_time')" in script
    # Both populations derive from the filtered map, not the raw one.
    assert "sorted(owned)" in script
    assert "sorted(conn)" in script
    assert "sorted(plats)" not in script


def _run_probe(payload, tmp_path):
    state_file = tmp_path / "gateway_state.json"
    state_file.write_text(json.dumps(payload), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-c", peer_deploy.HEALTH_PROBE_SCRIPT, str(state_file)],
        capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    return peer_deploy.parse_probe_output(proc.stdout)


def test_probe_end_to_end_ignores_stale_row(tmp_path):
    """Execute the actual remote snippet against the #130708 payload: the
    stale ``feishu`` row must appear in neither population."""
    all_p, conn_p = _run_probe(_issue_runtime(), tmp_path)
    assert all_p == ["api_server", "google_chat"]
    assert conn_p == ["api_server", "google_chat"]
    assert peer_deploy.platforms_healthy(all_p, conn_p) == (True, [])


def test_probe_end_to_end_still_catches_live_disconnect(tmp_path):
    runtime = _runtime(platforms={
        "google_chat": _live("google_chat", "connected"),
        "api_server": _live("api_server", "retrying"),
        "feishu": {"state": "retrying", "writer_pid": 1749354,
                   "writer_start_time": 11111111.0},
    })
    all_p, conn_p = _run_probe(runtime, tmp_path)
    ok, missing = peer_deploy.platforms_healthy(all_p, conn_p)
    assert ok is False
    assert missing == ["api_server"]


def test_owned_matches_dashboard_helper():
    """Parity with the dashboard reader: the same map filtered here and by
    ``_owned_profile_platforms`` must agree."""
    from hermes_cli.web_server_gateway import _owned_profile_platforms

    plats = {
        "google_chat": _live("google_chat", "connected"),
        "feishu": {"state": "retrying", "writer_pid": 1749354,
                   "writer_start_time": 11111111.0},
        "legacy": {"state": "connected"},
    }
    assert peer_deploy.owned_platforms(plats, LIVE_PID, LIVE_START) == \
        _owned_profile_platforms((LIVE_PID, LIVE_START), plats)

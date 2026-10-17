"""Tests for the CLI half of deferred restart (hermes_cli/defer_restart.py).

Exit-code contract under test (see the module docstring):
0 accepted/duplicate/superseded · 2 usage · 3 no target · 4 no supervisor
5 cooldown · 6 outcome unknown (must NOT be retried blindly).
"""

from __future__ import annotations

import json

import pytest

import hermes_cli.defer_restart as hd
from gateway import defer_restart as dr


FAKE_GENERATION = {"pid": 4242, "start_time": 1_760_000_000}
FAKE_IDENTITY = {
    "kind": "launchd",
    "service_id": "hermes-gateway",
    "hermes_home": "/tmp/hermes-test",
}


@pytest.fixture()
def cli_env(monkeypatch, tmp_path):
    """Sandboxed request dir + fixed identity/target + captured signal."""
    requests_dir = tmp_path / "restart_requests"
    monkeypatch.setattr(dr, "requests_dir", lambda: requests_dir)
    monkeypatch.setattr(dr, "compute_service_identity", lambda: dict(FAKE_IDENTITY))
    monkeypatch.setattr(hd, "_resolve_target_generation", lambda: dict(FAKE_GENERATION))
    monkeypatch.setattr(hd, "_supervisor_precheck", lambda identity: (True, "fake supervisor"))

    signals: list[tuple[int, int]] = []

    def fake_kill(pid, sig):
        signals.append((pid, sig))

    monkeypatch.setattr(hd.os, "kill", fake_kill)
    # Fast confirm polling for tests.
    monkeypatch.setattr(hd, "CONFIRM_TIMEOUT_S", 0.5)
    monkeypatch.setattr(hd, "CONFIRM_POLL_S", 0.01)
    return {"requests_dir": requests_dir, "signals": signals}


def _run(args=None):
    try:
        hd.cmd_defer_restart(args or object())
    except SystemExit as exc:
        return exc.code
    raise AssertionError("cmd_defer_restart must exit")


# --- early exits ---------------------------------------------------------------


def test_exit_3_when_no_target(cli_env, monkeypatch):
    monkeypatch.setattr(hd, "_resolve_target_generation", lambda: None)
    assert _run() == hd.EXIT_NO_TARGET
    assert cli_env["signals"] == []


def test_exit_4_when_no_supervisor_identity(cli_env, monkeypatch):
    monkeypatch.setattr(dr, "compute_service_identity", lambda: None)
    assert _run() == hd.EXIT_NO_SUPERVISOR
    assert cli_env["signals"] == []


def test_exit_4_when_precheck_fails(cli_env, monkeypatch):
    monkeypatch.setattr(hd, "_supervisor_precheck", lambda identity: (False, "no KeepAlive"))
    assert _run() == hd.EXIT_NO_SUPERVISOR
    assert cli_env["signals"] == []


def test_exit_5_when_cooldown_active(cli_env, monkeypatch):
    monkeypatch.setattr(dr, "cooldown_active_for",
                        lambda key, now=None: {"request_id": "old", "accepted_at": 1.0})
    assert _run() == hd.EXIT_COOLDOWN
    assert cli_env["signals"] == []


# --- happy path and confirmation -----------------------------------------------


def test_happy_path_writes_request_and_signals_target(cli_env, monkeypatch):
    monkeypatch.setattr(dr, "read_result",
                        lambda rid: {"outcome": dr.OUTCOME_ACCEPTED, "request_id": rid})

    assert _run() == hd.EXIT_ACCEPTED
    # Signal went to the resolved target with SIGUSR1.
    import signal as signal_mod

    assert cli_env["signals"] == [(FAKE_GENERATION["pid"], signal_mod.SIGUSR1)]
    # Request file is persisted, 0600, and carries the full binding.
    pending = list(cli_env["requests_dir"].glob("pending-*.json"))
    assert len(pending) == 1
    assert pending[0].stat().st_mode & 0o777 == 0o600
    payload = json.loads(pending[0].read_text())
    assert payload["schema_version"] == dr.PROTOCOL_SCHEMA_VERSION
    assert payload["service_identity"] == FAKE_IDENTITY
    assert payload["target_generation"] == FAKE_GENERATION
    assert payload["request_id"] == pending[0].name[len("pending-"):-len(".json")]
    assert payload["actor_ref"]["pid"] > 0


def test_duplicate_and_superseded_map_to_exit_0(cli_env, monkeypatch):
    for outcome in (dr.OUTCOME_DUPLICATE, dr.OUTCOME_SUPERSEDED):
        monkeypatch.setattr(dr, "read_result",
                            lambda rid, o=outcome: {"outcome": o, "request_id": rid})
        assert _run() == hd.EXIT_ACCEPTED


def test_exit_6_on_confirm_timeout(cli_env, monkeypatch):
    monkeypatch.setattr(dr, "read_result", lambda rid: None)
    assert _run() == hd.EXIT_UNKNOWN


def test_exit_5_when_gateway_rejects_cooldown(cli_env, monkeypatch):
    monkeypatch.setattr(dr, "read_result",
                        lambda rid: {"outcome": dr.OUTCOME_REJECTED_COOLDOWN, "request_id": rid})
    assert _run() == hd.EXIT_COOLDOWN


@pytest.mark.parametrize("outcome", [
    dr.OUTCOME_REJECTED_INVALID,
    dr.OUTCOME_REJECTED_TARGET_MISMATCH,
    dr.OUTCOME_EXPIRED,
])
def test_exit_6_when_gateway_rejects(cli_env, monkeypatch, outcome):
    monkeypatch.setattr(dr, "read_result",
                        lambda rid, o=outcome: {"outcome": o, "request_id": rid, "detail": "x"})
    assert _run() == hd.EXIT_UNKNOWN

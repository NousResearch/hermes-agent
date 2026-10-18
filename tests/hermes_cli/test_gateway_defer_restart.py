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
    requests_dir.mkdir(parents=True, exist_ok=True)  # like the real requests_dir()
    monkeypatch.setattr(dr, "requests_dir", lambda: requests_dir)
    monkeypatch.setattr(dr, "compute_service_identity", lambda: dict(FAKE_IDENTITY))
    monkeypatch.setattr(hd, "_resolve_target_generation", lambda: dict(FAKE_GENERATION))
    monkeypatch.setattr(hd, "_supervisor_precheck", lambda identity, pid: (True, "fake supervisor"))

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
    monkeypatch.setattr(hd, "_supervisor_precheck", lambda identity, pid: (False, "no KeepAlive"))
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
    assert payload["actor_ref"].startswith("cli:")  # bounded audit string (schema)


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


def test_exit_4_when_gateway_reports_stopping(cli_env, monkeypatch):
    """A plain stop without a restart path is a refusal, never a success."""
    monkeypatch.setattr(dr, "read_result",
                        lambda rid: {"outcome": dr.OUTCOME_REJECTED_STOPPING, "request_id": rid})
    assert _run() == hd.EXIT_NO_SUPERVISOR


# --- supervisor pre-check semantics (review gate 1, item 4) ---------------------


def test_launchd_keepalive_only_bare_true_is_accepted():
    """v1 accepts ONLY bare ``KeepAlive=true``: dict forms are conditional and
    their loaded state cannot be proven to recover exit 75 (probe, 2026-10-17:
    ``launchctl print`` drops the ``keepalive`` property token for dicts)."""
    assert hd._launchd_keepalive_recovers_exit75(True) is True
    for refused in (False, None,
                    {"SuccessfulExit": True}, {"SuccessfulExit": False},
                    {"Crash": True}, {"SuccessfulExit": False, "Crash": True}):
        assert hd._launchd_keepalive_recovers_exit75(refused) is False


def test_launchd_loaded_properties_parse():
    """The loaded-job ``properties`` line is the only observable policy evidence;
    a missing line must refuse (undeterminable), never pass."""
    assert hd._launchd_loaded_properties_have_keepalive(
        "    path = /x/ai.hermes.gateway.plist\n"
        "    properties = keepalive | runatload | inferred program\n") is True
    assert hd._launchd_loaded_properties_have_keepalive(
        "    properties = runatload | inferred program\n") is False
    assert hd._launchd_loaded_properties_have_keepalive("    state = running\n") is None


def test_s6_is_conservatively_refused_in_v1(monkeypatch):
    """s6 distinguishes up from wantedup — without a binding proof, exit 4."""
    monkeypatch.setattr(hd, "_supervisor_precheck", hd._supervisor_precheck)  # real code path
    ok, why = hd._supervisor_precheck({"kind": "s6", "service_id": "svc:gw",
                                       "hermes_home": "/tmp/x"}, 4242)
    assert ok is False
    assert "s6" in why


@pytest.mark.parametrize("restart,prevented,succeeded,expect_ok", [
    ("always", "", "", True),
    ("on-failure", "", "", True),
    ("always", "75", "", False),
    ("on-failure", "", "75", False),
    ("on-abnormal", "", "", False),
    ("on-abort", "", "", False),
    ("on-watchdog", "", "", False),
    ("on-success", "", "", False),
    ("no", "", "", False),
])
def test_systemd_policy_must_recover_exit75(restart, prevented, succeeded, expect_ok, monkeypatch):
    import subprocess

    props = (f"FragmentPath=/etc/systemd/system/x.service\nRestart={restart}\n"
             f"RestartPreventExitStatus={prevented}\nSuccessExitStatus={succeeded}\n"
             f"ActiveState=active\nMainPID={FAKE_GENERATION['pid']}\n")

    monkeypatch.setattr(hd.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, props, ""))
    ok, why = hd._precheck_systemd("hermes-gateway", FAKE_GENERATION["pid"])
    assert ok is expect_ok


def test_systemd_pid_binding_rejects_hand_started_gateway(monkeypatch):
    import subprocess

    props = ("FragmentPath=/etc/systemd/system/x.service\nRestart=always\n"
             "RestartPreventExitStatus=\nSuccessExitStatus=\nActiveState=active\nMainPID=11111\n")
    monkeypatch.setattr(hd.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, props, ""))
    ok, why = hd._precheck_systemd("hermes-gateway", FAKE_GENERATION["pid"])
    assert ok is False
    assert "MainPID" in why


# --- real-chain tests (gate 2, G2-R1): bash assembly → real entrypoint -----------
# A cron payload can assemble the command at runtime ("$left-$right"), which lexical
# guards cannot see. The chain below is REAL: bash variable expansion → `hermes` on
# PATH → the actual cmd_defer_restart entrypoint. Stubs sit exactly at the side-effect
# boundaries (request publish + signal), so "zero restart side effects" is measured,
# not assumed. The control variant proves the chain reaches those boundaries.

_CHAIN_JOB = (
    "#!/bin/sh\n"
    "left=defer\n"
    "right=restart\n"
    'hermes gateway "$left-$right"\n'
)

_CHAIN_SHIM = '''#!/usr/bin/env python3
import os, sys, types
sys.path.insert(0, os.environ["CHAIN_REPO"])
from pathlib import Path

LOG = Path(os.environ["CHAIN_LOG"])

def _log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(msg + "\\n")

import gateway.defer_restart as dr
import hermes_cli.defer_restart as hd

# Boundary stubs: everything below here is the restart side effect the test measures.
dr.cleanup_expired = lambda *a, **k: None
dr.compute_service_identity = lambda: {"kind": "stub", "service_id": "hermes-gateway",
                                       "hermes_home": "/tmp/stub"}
hd._resolve_target_generation = lambda: {"pid": 4242, "start_time": 1_760_000_000}
hd._supervisor_precheck = lambda identity, pid: (True, "stub supervisor")
dr.cooldown_active_for = lambda key: None

def _write(payload):
    _log("REQUEST_PUBLISHED " + str(payload.get("request_id")))
    return True
hd._write_request_file = _write
os.kill = lambda pid, sig: _log("SIGNAL %s %s" % (pid, sig))
dr.read_result = lambda req_id: {"request_id": req_id, "outcome": "accepted"}

try:
    hd.cmd_defer_restart(types.SimpleNamespace(max_wait=90.0))
except SystemExit as exc:
    sys.exit(int(exc.code or 0))
'''


def _run_real_chain(tmp_path, *, cron_context: bool):
    """Execute the job script through bash with a real `hermes` CLI shim on PATH."""
    import os
    import subprocess as sp
    import sys
    from pathlib import Path

    repo = Path(hd.__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shim_py = bin_dir / "shim.py"
    shim_py.write_text(_CHAIN_SHIM)
    shim = bin_dir / "hermes"
    shim.write_text(f'#!/bin/bash\nexec "{sys.executable}" "{shim_py}" "$@"\n')
    shim.chmod(0o755)
    job = tmp_path / "job.sh"
    job.write_text(_CHAIN_JOB)
    log = tmp_path / "side_effects.log"
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["CHAIN_REPO"] = str(repo)
    env["CHAIN_LOG"] = str(log)
    env.pop("HERMES_CRON_SESSION", None)
    env.pop("_HERMES_CRON_EXTERNAL_WORKER", None)
    if cron_context:
        # Real cron sessions publish "1" (scheduler sets the ContextVar to "1");
        # `_trusted_cron_provenance` truthiness matches that contract.
        env["HERMES_CRON_SESSION"] = "1"
    proc = sp.run(["bash", str(job)], env=env, capture_output=True, text=True, timeout=120)
    side_effects = log.read_text() if log.exists() else ""
    return proc, side_effects


def test_dynamic_assembly_in_cron_chain_reaches_entrypoint_and_is_refused(tmp_path):
    """G2-R1 chain A, cron variant: the assembled command reaches the REAL entrypoint
    and the trusted-cron-provenance refusal stops it — zero restart side effects."""
    proc, side_effects = _run_real_chain(tmp_path, cron_context=True)
    assert proc.returncode == 7
    assert "trusted cron provenance" in proc.stdout + proc.stderr
    assert "REQUEST_PUBLISHED" not in side_effects
    assert "SIGNAL" not in side_effects


def test_dynamic_assembly_control_reaches_request_and_signal(tmp_path):
    """G2-R1 chain A, control (no cron marker): the same chain DOES reach the
    request-publish and signal boundaries — proving the sentinel stubs are live and
    the cron variant's zero is a real zero."""
    proc, side_effects = _run_real_chain(tmp_path, cron_context=False)
    assert proc.returncode == 0
    assert "REQUEST_PUBLISHED" in side_effects
    assert "SIGNAL 4242" in side_effects

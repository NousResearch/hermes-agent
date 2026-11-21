"""Behaviour contracts for the C1 fix: prove-human gates on the same outcome validity as a run.

Docker, the real TTY and the attestation file are all faked: these tests prove the gating logic in
`_phase_prove` / `_map_rc` only. The fake outcome bytes are not evidence of any runtime, snapshot or
source provenance.
"""

import json
from dataclasses import replace

import pytest

from isolated_tests import attestation, main, policy, probe, proof
from isolated_tests.executor import Outcome

PROBE_OK_STDOUT = ("ISOTEST_PROBE " + json.dumps({k: True for k in probe.REQUIRED_CHECKS}) + "\n").encode()

PROBE_OK = Outcome(returncode=0, timed_out=False, interrupted=None, stdout=PROBE_OK_STDOUT, stderr=b"",
                   stdout_truncated=False, stderr_truncated=False, readers_done=True,
                   cleanup_status="removed")
WALL_OK = Outcome(returncode=None, timed_out=True, interrupted=None, stdout=b"", stderr=b"",
                  stdout_truncated=False, stderr_truncated=False, readers_done=True,
                  cleanup_status="removed")

UNTRUSTED = {
    "cleanup_unverified": {"cleanup_status": "unverified"},
    "cleanup_label_mismatch": {"cleanup_status": "label_mismatch"},
    "cleanup_not_configured": {"cleanup_status": "not_configured"},
    "interrupted": {"interrupted": "SIGINT"},
    "readers_not_done": {"readers_done": False},
    "stdout_truncated": {"stdout_truncated": True},
    "stderr_truncated": {"stderr_truncated": True},
}


def _probe_stdout(checks: dict) -> bytes:
    return ("ISOTEST_PROBE " + json.dumps(checks) + "\n").encode()


class _Tty:
    def __init__(self):
        self.text = []

    def write(self, s):
        self.text.append(s)

    def flush(self):
        pass


@pytest.fixture
def harness(tmp_path, monkeypatch):
    calls = {"execute": [], "confirm": 0, "write": 0, "outcomes": []}

    def fake_execute(ctx, cli_home, run_id, argv, wall=None):
        calls["execute"].append(wall)
        return calls["outcomes"].pop(0)

    def fake_confirm(tty, prompt, timeout=None):
        calls["confirm"] += 1
        return "code"

    def fake_attest_write(*args, **kwargs):
        calls["write"] += 1

    monkeypatch.setattr(main, "_execute", fake_execute)
    monkeypatch.setattr(main, "_argv", lambda *a, **k: ["docker", "run"])
    monkeypatch.setattr(main, "_write_new", lambda path, data: None)
    monkeypatch.setattr(main, "_manifest_sha", lambda ctx, rel: "0" * 64)
    monkeypatch.setattr(proof, "confirm", fake_confirm)
    monkeypatch.setattr(attestation, "write", fake_attest_write)

    def run(*outcomes):
        calls["outcomes"] = list(outcomes)
        ctx = {"out": tmp_path, "python": "/opt/py/bin/python3", "binding": {"fake": True}}
        rc = main._phase_prove(ctx, "0" * 16, tmp_path / "cli", _Tty())
        return rc, calls

    return run


@pytest.mark.parametrize("fields", list(UNTRUSTED.values()) + [{"returncode": 125}],
                         ids=list(UNTRUSTED) + ["rc125"])
def test_untrusted_probe_is_invalid_and_stops_before_wall(harness, fields):
    rc, calls = harness(replace(PROBE_OK, **fields))
    assert rc == policy.EXIT_INVALID
    assert calls["write"] == 0
    assert calls["confirm"] == 1
    assert len(calls["execute"]) == 1  # no wall canary


@pytest.mark.parametrize("fields", [
    {"returncode": None, "timed_out": True},
    {"returncode": 1},
    {"stdout": _probe_stdout({k: True for k in probe.REQUIRED_CHECKS[1:]})},
    {"stdout": _probe_stdout({**{k: True for k in probe.REQUIRED_CHECKS}, probe.REQUIRED_CHECKS[0]: False})},
], ids=["timed_out", "nonzero", "check_missing", "check_false"])
def test_failed_probe_keeps_proof_failed(harness, fields):
    rc, calls = harness(replace(PROBE_OK, **fields))
    assert rc == policy.EXIT_PROOF_FAILED
    assert calls["write"] == 0
    assert len(calls["execute"]) == 1


TRUNCATED_PROBE = b'ISOTEST_PROBE {"scratch_write'
MALFORMED_PROBE = b"ISOTEST_PROBE {not json}\n"


@pytest.mark.parametrize("fields, expected", [
    ({"stdout_truncated": True, "stdout": TRUNCATED_PROBE}, policy.EXIT_INVALID),
    ({"cleanup_status": "unverified", "stdout": MALFORMED_PROBE}, policy.EXIT_INVALID),
    ({"stdout": MALFORMED_PROBE}, policy.EXIT_PROOF_FAILED),
    ({"returncode": None, "timed_out": True, "stdout": TRUNCATED_PROBE}, policy.EXIT_PROOF_FAILED),
], ids=["truncated_invalid", "unverified_malformed_invalid", "clean_malformed", "timed_out_truncated"])
def test_unparseable_probe_line_is_gated_not_internal(harness, fields, expected):
    """Regression for C1-F1: the real parser met a broken ISOTEST_PROBE line before the gate (rc 7)."""
    rc, calls = harness(replace(PROBE_OK, **fields))
    assert rc == expected
    assert calls["execute"] == [None]  # no wall canary
    assert calls["write"] == 0
    assert calls["confirm"] == 1


@pytest.mark.parametrize("fields", list(UNTRUSTED.values()), ids=list(UNTRUSTED))
def test_untrusted_wall_canary_writes_nothing(harness, fields):
    rc, calls = harness(PROBE_OK, replace(WALL_OK, **fields))
    assert rc == policy.EXIT_INVALID
    assert calls["write"] == 0
    assert calls["confirm"] == 1


@pytest.mark.parametrize("fields", [
    {"timed_out": False, "returncode": 0},
    {"cleanup_status": "absent"},
], ids=["not_timed_out", "cleanup_absent"])
def test_wall_canary_not_proven_writes_nothing(harness, fields):
    rc, calls = harness(PROBE_OK, replace(WALL_OK, **fields))
    assert rc == policy.EXIT_PROOF_FAILED
    assert calls["write"] == 0
    assert calls["confirm"] == 1


def test_trusted_probe_and_expected_wall_timeout_write_once(harness):
    rc, calls = harness(PROBE_OK, WALL_OK)
    assert rc == policy.EXIT_OK
    assert calls["write"] == 1
    assert calls["confirm"] == 2
    assert calls["execute"] == [None, probe.WALL_CANARY_SECONDS]


@pytest.mark.parametrize("fields, expected", [
    ({}, policy.EXIT_OK),
    ({"returncode": 1}, policy.EXIT_TARGET_FAILED),
    ({"returncode": policy.CONTAINER_PREREQ_EXIT}, policy.EXIT_PREREQUISITE_MISSING),
    ({"returncode": 126}, policy.EXIT_PREREQUISITE_MISSING),
    ({"returncode": 127}, policy.EXIT_PREREQUISITE_MISSING),
    ({"returncode": None, "timed_out": True}, policy.EXIT_TIMEOUT),
    ({"returncode": 125}, policy.EXIT_INVALID),
    ({"cleanup_status": "unverified"}, policy.EXIT_INVALID),
    ({"returncode": None, "timed_out": True, "stdout_truncated": True}, policy.EXIT_INVALID),
], ids=["ok", "target_failed", "prereq87", "prereq126", "prereq127", "timeout", "rc125",
        "unverified", "invalid_beats_timeout"])
def test_map_rc_contract_unchanged(fields, expected):
    assert main._map_rc(replace(PROBE_OK, **fields)) == expected

"""Tests for the shared-metrics smoke artifact."""

import json
import subprocess
from pathlib import Path

import pytest

from gateway import status
from hermes_cli.observability import shared_metrics_contract as contract
from hermes_cli.observability.shared_metrics_update import update_receipt_fields
from scripts import smoke_nemo_relay_shared_metrics as smoke

# The smoke spawns a real chat and is not run by CI; these pin its expectations to the contract, so a
# renamed metric, a changed field set or a new projection rule fails here instead of rotting there.
_EXACT_EXPECTATIONS = {
    **smoke.V5_EXPECTED_DIMENSIONS,
    **smoke.GATEWAY_TURN_EXPECTED_DIMENSIONS,
    **smoke.FAILURE_EXPECTED_DIMENSIONS,
}


@pytest.mark.parametrize(
    "relative_path",
    [
        Path(".venv") / "bin" / "hermes",
        Path(".venv") / "Scripts" / "hermes.exe",
    ],
)
def test_resolve_hermes_executable_from_repository_venv(
    tmp_path,
    monkeypatch,
    relative_path,
):
    executable = tmp_path / relative_path
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(smoke.shutil, "which", lambda _name: None)

    assert smoke._resolve_hermes_executable(tmp_path) == executable


def test_resolve_hermes_executable_reports_missing_binary(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke.shutil, "which", lambda _name: None)

    with pytest.raises(SystemExit, match="or on PATH"):
        smoke._resolve_hermes_executable(tmp_path)


def test_smoke_expects_only_contract_metrics_with_their_current_fields():
    assert smoke.EXPECTED_METRIC_NAMES <= contract.COUNTER_METRICS
    assert set(_EXACT_EXPECTATIONS) <= smoke.EXPECTED_METRIC_NAMES
    for name, dimensions in _EXACT_EXPECTATIONS.items():
        assert frozenset(dimensions) == contract._METRIC_FIELDS[name], name
        assert contract.counter_dimensions_are_valid(name, dimensions), name
    assert set(smoke.EXPECTED_STARTUP_SURFACES) <= contract.STARTUP_SURFACES


@pytest.mark.platforms("posix")  # flock-held gateway lock
@pytest.mark.parametrize("start_time_shift", [0, 10_000], ids=["live-non-gateway", "recycled-pid"])
def test_gateway_stop_spares_a_process_that_is_not_the_homes_live_gateway(tmp_path, start_time_shift):
    """A held lock and a gateway record naming this home do not make the live PID a gateway."""
    import fcntl

    decoy = subprocess.Popen(["sleep", "300"])
    try:
        record = json.dumps({
            "pid": decoy.pid, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
            "start_time": status._get_process_start_time(decoy.pid) + start_time_shift,
            "hermes_home": str(tmp_path),
        })
        (tmp_path / "gateway.pid").write_text(record)
        with open(tmp_path / "gateway.lock", "w") as lock:
            lock.write(record)
            lock.flush()
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            smoke._stop_auto_started_gateway(tmp_path, timeout=5)
        assert decoy.poll() is None
    finally:
        decoy.kill()
        decoy.wait()


def test_smoke_update_expectations_match_the_receipt_projection():
    run, stage_rows = update_receipt_fields(smoke.UPDATE_RECEIPT)

    assert run == smoke.FAILURE_EXPECTED_DIMENSIONS["hermes.update.run"]
    assert sorted((row["stage"], row["outcome"]) for row in stage_rows) == smoke.expected_update_stages()

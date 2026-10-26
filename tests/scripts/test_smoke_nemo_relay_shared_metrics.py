"""Tests for the shared-metrics smoke artifact."""

from pathlib import Path

import pytest

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


def test_smoke_update_expectations_match_the_receipt_projection():
    run, stage_rows = update_receipt_fields(smoke.UPDATE_RECEIPT)

    assert run == smoke.FAILURE_EXPECTED_DIMENSIONS["hermes.update.run"]
    assert sorted((row["stage"], row["outcome"]) for row in stage_rows) == smoke.expected_update_stages()

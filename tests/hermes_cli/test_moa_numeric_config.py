"""Malformed numeric MoA settings must degrade to defaults, not break the CLI or JSON."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from hermes_cli.moa_config import normalize_moa_config


@pytest.mark.parametrize("fanout", [
    {"mode": "every_n", "n": float("inf")},
    {"mode": "every_n", "n": float("-inf")},
    {"mode": "every_n", "n": "inf"},
    {"mode": "every_n", "n": "-inf"},
    "every_n:inf",
    "every_n:-inf",
])
def test_moa_list_tolerates_nonfinite_fanout(tmp_path, fanout):
    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({"moa": {"fanout": fanout}}))
    env = {**os.environ, "HERMES_HOME": str(home)}
    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "moa", "list"],
        cwd=Path(__file__).resolve().parents[2],
        env=env, capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    normalized = normalize_moa_config({"fanout": fanout})
    assert normalized["fanout"] == normalize_moa_config({})["fanout"]


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), float("-inf"), "NaN", "inf", "-inf"])
def test_nonfinite_temperatures_do_not_escape_into_json(temperature):
    raw = {
        "reference_temperature": temperature,
        "aggregator_temperature": temperature,
    }
    normalized = normalize_moa_config(raw)
    assert normalized["reference_temperature"] is None
    assert normalized["aggregator_temperature"] is None
    json.dumps(normalized, allow_nan=False)

    finite = normalize_moa_config({
        "reference_temperature": 0,
        "aggregator_temperature": "0.75",
        "fanout": {"mode": "every_n", "n": "3.0"},
    })
    assert finite["reference_temperature"] == 0
    assert finite["aggregator_temperature"] == 0.75
    assert finite["fanout"] == "every_n:3"

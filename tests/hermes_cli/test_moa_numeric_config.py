"""Malformed numeric MoA settings must degrade to defaults, not break the CLI or JSON."""

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


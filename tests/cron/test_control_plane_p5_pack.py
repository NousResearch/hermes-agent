from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from cron_control.p5 import load_p5_allowlist, load_p5_dataset, run_p5_canaries


P5_DIR = Path(__file__).resolve().parents[2] / "docs" / "cron-control" / "p5"


def test_p5_pack_validator_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(P5_DIR / "validate_phase5.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "phase-5 rollout pack validation passed" in result.stdout


def test_p5_canary_bundle_runs_openai_codex_only_and_rolls_back() -> None:
    dataset = load_p5_dataset()
    allowlist = load_p5_allowlist()
    summary = run_p5_canaries()

    assert len(dataset) == 30
    assert summary["dataset"]["rows"] == 30
    assert summary["all_passed"] is True
    assert summary["allowlist"]["policy_id"] == "p5-canary-allowlist-v1"
    assert all(route["route_id"].startswith("openai-codex/") for route in allowlist["routes"])
    assert summary["canaries"]["quarantine"]["ok"] is True
    assert summary["canaries"]["reset"]["ok"] is True
    assert summary["canaries"]["model_switch"]["ok"] is True
    assert summary["canaries"]["model_switch"]["job_after_switch"]["provider"] == "openai-codex"
    assert summary["canaries"]["model_switch"]["job_after_switch"]["model"] == "gpt-5.6-terra"
    assert summary["canaries"]["model_switch"]["job_after_rollback"]["provider"] == "openai-codex"
    assert summary["canaries"]["model_switch"]["job_after_rollback"]["model"] == "gpt-5.4"

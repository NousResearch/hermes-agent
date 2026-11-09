import json
import subprocess
import sys
from pathlib import Path

from scripts.aion_pr_intake_adapter_v0_1_check import check_fixture_dir, validate_packet

FIXTURE_DIR = Path("tests/fixtures/aion-pr-intake-adapter-v0.1")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_valid_fixture_passes():
    packet = load(FIXTURE_DIR / "valid/hermes-kanban-pr-intake.valid.json")
    assert validate_packet(packet) == []


def test_invalid_fixtures_fail_closed():
    invalid_files = sorted((FIXTURE_DIR / "invalid").glob("*.invalid.json"))
    assert {p.name for p in invalid_files} == {
        "label-only-execution-claim.invalid.json",
        "missing-assignee.invalid.json",
        "missing-dispatcher-pickup.invalid.json",
        "missing-audit-verdict.invalid.json",
        "forbidden-runtime-flag.invalid.json",
        "full-unattended-overclaim.invalid.json",
    }
    for path in invalid_files:
        assert validate_packet(load(path)), path


def test_fixture_suite_reports_pass_when_invalids_fail_closed():
    result = check_fixture_dir(FIXTURE_DIR)
    assert result["verdict"] == "PASS"
    assert result["fixture_count"] == 7
    assert result["valid_fixture_count"] == 1
    assert result["invalid_fixture_count"] == 6


def test_checker_cli_outputs_pass():
    proc = subprocess.run(
        [sys.executable, "scripts/aion_pr_intake_adapter_v0_1_check.py"],
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["verdict"] == "PASS"
    assert payload["fixture_count"] == 7

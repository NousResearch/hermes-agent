import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.demo_workflow import build_demo_output


def test_build_demo_output_creates_valid_end_to_end_result(tmp_path: Path):
    log_path = tmp_path / "demo-router.jsonl"

    result = build_demo_output(log_path)

    assert log_path.exists()
    assert result["summary"]["total_decisions"] == 3
    assert result["summary"]["total_feedback_events"] == 3
    assert result["patched_config_valid"] is True
    assert result["patched_config_errors"] == []
    assert result["patch_generated_count"] >= 1
    assert len(result["routes"]) == 3
    assert any(item["primary_model"] == "ollama" for item in result["routes"])

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    payloads = [json.loads(line) for line in lines]
    assert sum(1 for row in payloads if row.get("event_type") == "feedback") == 3


def test_demo_script_outputs_json(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "demo_workflow.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["summary"]["total_decisions"] == 3
    assert payload["patched_config_valid"] is True

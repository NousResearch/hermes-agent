import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_feedback_cli_appends_event(tmp_path: Path):
    log_path = tmp_path / "router.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "log_router_feedback.py"),
            "--log-path",
            str(log_path),
            "--request-id",
            "req_123",
            "--outcome",
            "success",
            "--actual-model-used",
            "gpt-5.4",
            "--user-rating",
            "5",
            "--notes",
            "worked well",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.returncode == 0
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    event = json.loads(lines[0])
    assert event["event_type"] == "feedback"
    assert event["request_id"] == "req_123"
    assert event["outcome"] == "success"
    assert event["actual_model_used"] == "gpt-5.4"
    assert event["user_rating"] == 5
    assert event["notes"] == "worked well"


def test_feedback_cli_marks_fallback(tmp_path: Path):
    log_path = tmp_path / "router.jsonl"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "log_router_feedback.py"),
            "--log-path",
            str(log_path),
            "--request-id",
            "req_456",
            "--outcome",
            "fallback_used",
            "--fallback-used",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    event = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert event["fallback_used"] is True

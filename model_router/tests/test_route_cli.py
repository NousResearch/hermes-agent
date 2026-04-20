import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "route_cli.py"), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def test_cli_primary_only_for_coding():
    result = run_cli(
        "--task-type", "coding",
        "--mode", "execute",
        "--priority", "high",
        "--has-code",
        "--primary-only",
    )

    assert result.stdout.strip() == "gpt-5.4"


def test_cli_json_output_for_policy_override_chat_medium_critical():
    result = run_cli(
        "--task-type", "chat",
        "--mode", "draft",
        "--priority", "medium",
        "--quota", "critical",
        "--json",
    )

    payload = json.loads(result.stdout)
    assert payload["primary_model"] == "claude-sonnet-4.6"
    assert any("policy_override" in item for item in payload["trace"])


def test_cli_writes_telemetry_log(tmp_path: Path):
    log_path = tmp_path / "router.jsonl"

    result = run_cli(
        "--task-type", "coding",
        "--mode", "execute",
        "--priority", "high",
        "--has-code",
        "--log-path", str(log_path),
        "--request-id", "req-test-1",
        "--primary-only",
    )

    assert result.stdout.strip() == "gpt-5.4"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["request_id"] == "req-test-1"
    assert payload["decision"]["primary_model"] == "gpt-5.4"

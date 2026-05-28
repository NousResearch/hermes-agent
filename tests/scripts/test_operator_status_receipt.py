import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "operator_status_receipt.py"
spec = importlib.util.spec_from_file_location("operator_status_receipt", MODULE_PATH)
operator_status_receipt = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = operator_status_receipt
spec.loader.exec_module(operator_status_receipt)


def test_parse_launchctl_extracts_runtime_fields():
    text = """
    state = running
    program = /tmp/hermes/venv/bin/python
    working directory = /tmp/hermes
    stdout path = /tmp/profile/logs/gateway.log
    stderr path = /tmp/profile/logs/gateway.error.log
    pid = 12345
    """

    parsed = operator_status_receipt.parse_launchctl(text)

    assert parsed["state"] == "running"
    assert parsed["program"] == "/tmp/hermes/venv/bin/python"
    assert parsed["working_directory"] == "/tmp/hermes"
    assert parsed["stdout_path"] == "/tmp/profile/logs/gateway.log"
    assert parsed["stderr_path"] == "/tmp/profile/logs/gateway.error.log"
    assert parsed["pid"] == "12345"


def test_timestamped_issue_lines_honors_since(tmp_path):
    log = tmp_path / "gateway.log"
    log.write_text(
        "\n".join(
            [
                "2026-05-28 17:00:00,000 ERROR root: Non-retryable client error: old",
                "Traceback (most recent call last):",
                "2026-05-28 18:00:00,000 ERROR root: Non-retryable client error: 'NoneType' object is not iterable",
                "2026-05-28 18:01:00,000 INFO gateway.run: Gateway running with 1 platform(s)",
            ]
        )
    )

    hits = operator_status_receipt.timestamped_issue_lines(log, since="2026-05-28 18:00:00")

    assert len(hits) == 1
    assert "NoneType" in hits[0]


def test_render_markdown_includes_live_cwd():
    receipt = {
        "generated_at": "2026-05-28T22:00:00+00:00",
        "profile": "sawyer",
        "repo": {
            "root": "/tmp/hermes",
            "branch": "codex/test",
            "head": "abc123",
            "status_short": "## codex/test",
        },
        "launchctl": {"pid": "123", "working_directory": "/tmp/hermes"},
        "status_flags": {
            "slack_configured": True,
            "gateway_running": True,
            "openai_codex_logged_in": True,
        },
        "logs": {
            "since": "2026-05-28 18:00:00",
            "stdout_issue_count": 0,
            "stderr_issue_count": 0,
        },
    }

    markdown = operator_status_receipt.render_markdown(receipt)

    assert "Gateway cwd: `/tmp/hermes`" in markdown
    assert "Slack configured: `True`" in markdown

"""Focused contract tests for the interactive API-run watcher."""

from __future__ import annotations

import json

from hermes_cli.run_watch import watch_run


class _Response:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self.lines)


def test_watch_run_displays_redacted_approval_and_posts_correlated_choice():
    requests = []

    def opener(request, timeout):
        requests.append((request, timeout))
        if request.get_method() == "GET":
            payload = {"event": "approval.request", "request_id": "req-1", "command": "echo [REDACTED]",
                       "description": "dangerous shell command", "choices": ["once", "deny"]}
            return _Response([
                f"data: {json.dumps(payload)}\n".encode(),
                b"\n",
                b'data: {"event":"run.completed"}\n',
                b"\n",
            ])
        return _Response([])

    output = []
    status = watch_run(
        "run-1", "http://127.0.0.1:8642", "test-key", opener=opener,
        prompt=lambda _text: "once", write=output.append,
    )

    assert status == "completed"
    assert any("echo [REDACTED]" in line for line in output)
    approval = requests[1][0]
    assert approval.full_url == "http://127.0.0.1:8642/v1/runs/run-1/approval"
    assert json.loads(approval.data) == {"choice": "once", "request_id": "req-1"}
    assert approval.get_header("Authorization") == "Bearer test-key"


def test_watch_run_rejects_non_loopback_http_before_sending_credentials():
    try:
        watch_run("run-1", "http://remote.example", "test-key", opener=lambda *_: None)
    except ValueError as exc:
        assert "HTTPS" in str(exc)
    else:
        raise AssertionError("plaintext remote endpoint was accepted")

import json
from unittest.mock import Mock, patch

import pytest

from agent.background_jobs import (
    BackgroundTaskRequest,
    background_backend_enabled,
    enqueue_background_task,
    enqueue_background_task_via_command,
    get_background_backend,
    preview_prompt,
)


def _request() -> BackgroundTaskRequest:
    return BackgroundTaskRequest(
        task_id="bg_test",
        prompt="map the repo",
        origin="gateway",
        platform="telegram",
        session_id="bg_test",
        user_id="123",
        chat_id="456",
    )


def test_preview_prompt_truncates_cleanly():
    assert preview_prompt("abc", limit=10) == "abc"
    assert preview_prompt("x" * 12, limit=10) == ("x" * 10) + "..."


def test_backend_aliases_normalize_to_command(monkeypatch):
    monkeypatch.setenv("HERMES_BACKGROUND_BACKEND", "minions")
    assert get_background_backend() == "command"
    assert background_backend_enabled() is True


def test_enqueue_background_task_local_is_noop(monkeypatch):
    monkeypatch.delenv("HERMES_BACKGROUND_BACKEND", raising=False)
    submission = enqueue_background_task(_request())
    assert submission.backend == "local"
    assert submission.task_id == "bg_test"
    assert submission.accepted is True


def test_command_backend_requires_command(monkeypatch):
    monkeypatch.setenv("HERMES_BACKGROUND_BACKEND", "command")
    monkeypatch.delenv("HERMES_BACKGROUND_ENQUEUE_CMD", raising=False)
    with pytest.raises(ValueError):
        enqueue_background_task(_request())


def test_command_backend_parses_json_response(monkeypatch):
    monkeypatch.setenv("HERMES_BACKGROUND_ENQUEUE_CMD", "enqueue")
    proc = Mock(returncode=0, stdout=json.dumps({
        "task_id": "remote-123",
        "backend": "minions",
        "queue": "background",
        "message": "accepted",
    }), stderr="")
    with patch("agent.background_jobs.subprocess.run", return_value=proc) as mock_run:
        submission = enqueue_background_task_via_command(_request(), env={
            "HERMES_BACKGROUND_BACKEND": "command",
            "HERMES_BACKGROUND_ENQUEUE_CMD": "enqueue",
        })

    payload = json.loads(mock_run.call_args.kwargs["input"])
    assert payload["version"]
    assert payload["kind"] == "background"
    assert payload["payload"]["prompt"] == "map the repo"
    assert payload["callback"]["type"] == "platform"
    assert submission.task_id == "remote-123"
    assert submission.backend == "minions"
    assert submission.queue == "background"
    assert submission.message == "accepted"
    assert mock_run.call_args.kwargs["input"]


def test_command_backend_surfaces_failures():
    proc = Mock(returncode=9, stdout="", stderr="boom")
    with patch("agent.background_jobs.subprocess.run", return_value=proc):
        with pytest.raises(RuntimeError, match="boom"):
            enqueue_background_task_via_command(_request(), env={
                "HERMES_BACKGROUND_BACKEND": "command",
                "HERMES_BACKGROUND_ENQUEUE_CMD": "enqueue",
            })

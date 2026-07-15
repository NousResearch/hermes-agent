from __future__ import annotations

import json
from unittest.mock import patch

from agent.codex_runtime import (
    _codex_attempt_environment,
    _codex_turn_started_callback_from_environment,
)


class _Response:
    status = 204

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


def test_configured_turn_callback_posts_actual_runtime_ids(monkeypatch) -> None:
    monkeypatch.setenv(
        "HERMES_CODEX_TURN_STARTED_URL", "http://127.0.0.1:8765/codex-turn"
    )
    monkeypatch.setenv("HERMES_CODEX_TURN_STARTED_TOKEN", "callback-secret")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-live")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "run-live")
    monkeypatch.setenv("HERMES_PROFILE", "engineering")

    with patch("agent.codex_runtime.urlopen", return_value=_Response()) as opened:
        callback = _codex_turn_started_callback_from_environment()
        assert callback is not None
        callback("thread-live", "turn-live")

    request = opened.call_args.args[0]
    assert request.full_url == "http://127.0.0.1:8765/codex-turn"
    assert request.headers["Authorization"] == "Bearer callback-secret"
    assert json.loads(request.data) == {
        "hermes_task_id": "task-live",
        "hermes_run_id": "run-live",
        "hermes_profile": "engineering",
        "codex_thread_id": "thread-live",
        "codex_turn_id": "turn-live",
    }
    assert _codex_attempt_environment()["OMNICLAW_ATTEMPT_TOKEN"].startswith("eyJ")


def test_unconfigured_turn_callback_is_disabled(monkeypatch) -> None:
    monkeypatch.delenv("HERMES_CODEX_TURN_STARTED_URL", raising=False)
    assert _codex_turn_started_callback_from_environment() is None

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.action_requests import build_action_buttons, store_action_request
from scripts.sentry_create_pr_from_packet import build_codex_prompt


def test_build_action_buttons_stores_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    buttons = build_action_buttons(
        [{"label": "Create PR", "kind": "sentry", "action": "create_pr"}],
        {"project": "incremnt", "issue": {"id": "INCR-1"}},
    )

    assert buttons == [{"label": "Create PR", "callback_data": buttons[0]["callback_data"]}]
    assert buttons[0]["callback_data"].startswith("ar:sentry-")
    request_id = buttons[0]["callback_data"].split(":", 1)[1]
    stored = json.loads((tmp_path / "action_requests" / f"{request_id}.json").read_text())
    assert stored["kind"] == "sentry"
    assert stored["action"] == "create_pr"
    assert stored["payload"]["project"] == "incremnt"


def test_sentry_create_pr_prompt_contains_required_guardrails(tmp_path, monkeypatch):
    repo = tmp_path / "onemore"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "sentry_repo_map.json").write_text(json.dumps({"incremnt": str(repo)}))

    prompt = build_codex_prompt(
        {
            "id": "sentry-test",
            "payload": {
                "project": "incremnt",
                "issue": {"id": "INCR-1", "url": "https://sentry.example/issues/1"},
                "title": "Crash in onboarding",
                "environment": "production",
            },
        }
    )

    assert "Commit, push, AND create PR" in prompt
    assert "Do not merge" in prompt
    assert str(repo) in prompt
    assert "Crash in onboarding" in prompt

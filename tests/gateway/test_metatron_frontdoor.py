import json
import asyncio
import urllib.error

from gateway.metatron_frontdoor import (
    detect_metatron_action,
    format_metatron_telegram_response,
    perform_metatron_action,
)


def test_detects_bootstrap_action():
    assert detect_metatron_action("Metatron, initialize THE hub") == {
        "type": "bootstrap_hub",
    }


def test_detects_coding_task_action():
    assert detect_metatron_action("create coding task to fix Paperclip OOM") == {
        "type": "create_coding_task",
        "message": "fix Paperclip OOM",
    }


def test_ignores_general_questions():
    assert detect_metatron_action("tell me a joke") is None


def test_detects_blocked_status_question():
    assert detect_metatron_action("what's blocked?") == {
        "type": "ask_metatron",
        "message": "what's blocked?",
    }


def test_perform_coding_task_posts_to_paperclip(monkeypatch):
    calls = []

    def fake_post_json(url, payload, timeout):
        calls.append((url, payload, timeout))
        return {
            "issue": {"identifier": "THE-12", "id": "issue-1", "title": "Fix OOM"},
            "project": {"name": "Paperclip Platform"},
            "assignee": {"name": "Code Architect"},
            "route": {"reason": "Runtime/deployment language"},
        }

    monkeypatch.setattr("gateway.metatron_frontdoor._post_json", fake_post_json)

    result = asyncio.run(
        perform_metatron_action(
            {"type": "create_coding_task", "message": "fix Paperclip OOM"},
            base_url="http://paperclip.test/api/metatron",
        )
    )

    assert calls == [
        (
            "http://paperclip.test/api/metatron/coding-tasks",
            {"message": "fix Paperclip OOM", "requestedBy": "telegram-metatron-frontdoor"},
            10.0,
        )
    ]
    assert result["issue"]["identifier"] == "THE-12"


def test_perform_blocked_question_posts_to_orchestrate(monkeypatch):
    calls = []

    def fake_post_json(url, payload, timeout):
        calls.append((url, payload, timeout))
        return {"response": "Blocked: Paperclip dev server is down."}

    monkeypatch.setattr("gateway.metatron_frontdoor._post_json", fake_post_json)

    result = asyncio.run(
        perform_metatron_action(
            {"type": "ask_metatron", "message": "what's blocked?"},
            base_url="http://paperclip.test/api/metatron",
        )
    )

    assert calls == [
        (
            "http://paperclip.test/api/metatron/orchestrate",
            {"message": "what's blocked?", "requestedBy": "telegram-metatron-frontdoor"},
            10.0,
        )
    ]
    assert result["response"] == "Blocked: Paperclip dev server is down."


def test_formats_created_issue_response():
    text = format_metatron_telegram_response({
        "issue": {"identifier": "THE-12", "id": "issue-1"},
        "project": {"name": "Paperclip Platform"},
        "assignee": {"name": "Code Architect"},
        "route": {"reason": "Runtime/deployment language"},
    })

    assert "Created THE-12" in text
    assert "Code Architect" in text


def test_formats_orchestrate_response():
    assert format_metatron_telegram_response({"response": "No blockers."}) == "No blockers."


def test_perform_action_returns_blocked_message_on_connection_error(monkeypatch):
    def fake_post_json(_url, _payload, _timeout):
        raise urllib.error.URLError("down")

    monkeypatch.setattr("gateway.metatron_frontdoor._post_json", fake_post_json)

    result = asyncio.run(
        perform_metatron_action(
            {"type": "bootstrap_hub"},
            base_url="http://paperclip.test/api/metatron",
        )
    )

    assert result["blocked"] is True
    assert "Paperclip Metatron hub is unreachable" in result["message"]

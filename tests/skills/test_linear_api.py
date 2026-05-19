"""Tests for Linear script helpers."""

import importlib.util
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


SKILL_API = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/linear/scripts/linear_api.py"
)


def _load_linear_api_module():
    spec = importlib.util.spec_from_file_location("linear_api_test", SKILL_API)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def _mock_urlopen(payloads: list[dict], captured_calls: list[dict]):
    iterator = iter(payloads)

    def _fake_urlopen(request, **_):
        captured_calls.append(json.loads(request.data.decode("utf-8")))
        return _make_response(next(iterator))

    return _fake_urlopen


@pytest.fixture
def linear_api_module(monkeypatch):
    monkeypatch.setenv("LINEAR_API_KEY", "lin_api_test")
    return _load_linear_api_module()


def test_create_issue_resolves_label_and_assignee(linear_api_module):
    calls: list[dict] = []

    payloads = [
        {"data": {"teams": {"nodes": [{"id": "team-id", "key": "ENG", "name": "Engineers"}]}}},
        {"data": {"issueLabels": {"nodes": [{"id": "label-id", "name": "bug"}]}}},
        {"data": {"users": {"nodes": [{"id": "user-id", "name": "Alice", "email": "alice@example.com"}]}}},
        {"data": {"issueCreate": {"success": True, "issue": {"identifier": "ENG-1"}}}},
    ]

    with patch("urllib.request.urlopen", side_effect=_mock_urlopen(payloads, calls)):
        linear_api_module.main([
            "create-issue",
            "--title",
            "Fix login",
            "--team",
            "ENG",
            "--label",
            "Bug",
            "--assignee",
            "alice@example.com",
        ])

    assert calls[0]["query"].startswith("query")
    assert "issueLabels" in calls[1]["query"]
    assert "users" in calls[2]["query"]
    assert "issueCreate" in calls[3]["query"]
    mutation_input = calls[3]["variables"]["input"]
    assert mutation_input["teamId"] == "team-id"
    assert mutation_input["labelIds"] == ["label-id"]
    assert mutation_input["assigneeId"] == "user-id"


@pytest.mark.parametrize("field_name", ["label", "assignee"])
def test_create_issue_not_found_exits(linear_api_module, field_name):
    err = io.StringIO()
    calls: list[dict] = []

    if field_name == "label":
        payloads = [
            {"data": {"teams": {"nodes": [{"id": "team-id", "key": "ENG", "name": "Engineers"}]}}},
            {"data": {"issueLabels": {"nodes": [{"id": "other-label", "name": "feature"}]}}},
        ]
        args = [
            "create-issue",
            "--title",
            "Fix login",
            "--team",
            "ENG",
            "--label",
            "Bug",
        ]
        expected_message = "Label not found: Bug"
    else:
        payloads = [
            {"data": {"teams": {"nodes": [{"id": "team-id", "key": "ENG", "name": "Engineers"}]}}},
            {"data": {"users": {"nodes": [{"id": "user-id", "name": "Alice", "email": "alice@example.com"}]}}},
        ]
        args = [
            "create-issue",
            "--title",
            "Fix login",
            "--team",
            "ENG",
            "--assignee",
            "Missing Person",
        ]
        expected_message = "Assignee not found: Missing Person"

    with patch.object(linear_api_module.sys, "stderr", err), patch(
        "urllib.request.urlopen", side_effect=_mock_urlopen(payloads, calls)
    ):
        with pytest.raises(SystemExit) as exc:
            linear_api_module.main(args)

    assert exc.value.code == 1
    assert expected_message in err.getvalue()
    assert "issueCreate" not in calls[-1]["query"]

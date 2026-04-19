"""Tests for optional-skills/productivity/cognition-retention/scripts/*.py

Covers: CLI argument parsing, error handling, output formatting,
and MCP JSON-RPC call construction. All HTTP calls are mocked —
no network access needed to run these tests.
"""

import json
import os
import sys
from io import BytesIO
from pathlib import Path
from unittest import mock

import pytest

# Add the scripts dir so we can import modules directly
SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "productivity"
    / "cognition-retention"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))


# ── Helpers ──────────────────────────────────────────────────────────


def _mock_urlopen(response_data: dict):
    """Return a context-manager mock for urllib.request.urlopen."""
    mcp_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [{"type": "text", "text": json.dumps(response_data)}]
        },
    }
    body = json.dumps(mcp_response).encode()
    cm = mock.MagicMock()
    cm.__enter__ = mock.Mock(return_value=BytesIO(body))
    cm.__exit__ = mock.Mock(return_value=False)
    return cm


def _set_key(monkeypatch):
    """Set the API key both in env AND in already-imported module globals."""
    key = "cog_me_test_key_123"
    url = "https://www.cognitionus.com/api/integrations/claude-code/mcp"
    monkeypatch.setenv("COGNITION_API_KEY", key)
    monkeypatch.setenv("COGNITION_URL", url)
    # Patch module-level constants (cached at import time)
    for mod_name in ["cognition_log", "cognition_briefing", "cognition_review",
                     "cognition_brain", "cognition_batch"]:
        if mod_name in sys.modules:
            monkeypatch.setattr(sys.modules[mod_name], "COGNITION_KEY", key)
            monkeypatch.setattr(sys.modules[mod_name], "COGNITION_URL", url)


# ── cognition_log.py ─────────────────────────────────────────────────


class TestCognitionLog:
    def test_missing_api_key(self, monkeypatch, capsys):
        monkeypatch.delenv("COGNITION_API_KEY", raising=False)
        import cognition_log

        with mock.patch("sys.argv", ["cognition_log", "test-concept"]):
            with pytest.raises(SystemExit):
                cognition_log.main()

    def test_missing_concept_arg(self, monkeypatch, capsys):
        _set_key(monkeypatch)
        import cognition_log

        with mock.patch("sys.argv", ["cognition_log"]):
            with pytest.raises(SystemExit):
                cognition_log.main()

    def test_successful_log_new(self, monkeypatch, capsys):
        _set_key(monkeypatch)
        import cognition_log

        response = {"success": True, "isNew": True, "newRetention": 85, "nextReviewAt": "2026-04-25"}
        with mock.patch("urllib.request.urlopen", return_value=_mock_urlopen(response)):
            with mock.patch("sys.argv", ["cognition_log", "react-hooks", "--topic", "React", "--score", "0.85", "--weight", "active"]):
                cognition_log.main()
        out = capsys.readouterr().out
        assert "NEW" in out
        assert "react-hooks" in out
        assert "85%" in out

    def test_successful_log_updated(self, monkeypatch, capsys):
        _set_key(monkeypatch)
        import cognition_log

        response = {"success": True, "isNew": False, "newRetention": 92, "nextReviewAt": "2026-04-28"}
        with mock.patch("urllib.request.urlopen", return_value=_mock_urlopen(response)):
            with mock.patch("sys.argv", ["cognition_log", "docker-compose"]):
                cognition_log.main()
        out = capsys.readouterr().out
        assert "UPDATED" in out
        assert "92%" in out

    def test_source_integration_set(self, monkeypatch):
        _set_key(monkeypatch)
        import cognition_log

        response = {"success": True, "isNew": True, "newRetention": 70, "nextReviewAt": "2026-04-22"}
        captured_request = {}

        def mock_urlopen(req, **kwargs):
            captured_request["data"] = json.loads(req.data)
            return _mock_urlopen(response)

        with mock.patch("urllib.request.urlopen", side_effect=mock_urlopen):
            with mock.patch("sys.argv", ["cognition_log", "test-concept"]):
                cognition_log.main()

        args = captured_request["data"]["params"]["arguments"]
        assert args["source_integration"] == "hermes_agent"


# ── cognition_briefing.py ────────────────────────────────────────────


class TestCognitionBriefing:
    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("COGNITION_API_KEY", raising=False)
        import cognition_briefing

        with pytest.raises(SystemExit):
            cognition_briefing.main()

    def test_healthy_briefing(self, monkeypatch, capsys):
        _set_key(monkeypatch)
        import cognition_briefing

        response = {
            "overallRetention": 85,
            "weakest": [{"concept": "docker", "topic": "DevOps", "retention": 45}],
            "dueForReview": [{"concept": "k8s", "topic": "DevOps"}],
            "teammateNudges": [{"concept": "graphql", "topic": "API"}],
            "briefing": "Overall 85%. Weakest: docker (45%)",
        }
        with mock.patch("urllib.request.urlopen", return_value=_mock_urlopen(response)):
            with mock.patch("sys.argv", ["cognition_briefing"]):
                cognition_briefing.main()
        out = capsys.readouterr().out
        assert "85%" in out
        assert "Solid" in out
        assert "docker" in out
        assert "k8s" in out
        assert "graphql" in out

    def test_at_risk_label(self, monkeypatch, capsys):
        _set_key(monkeypatch)
        import cognition_briefing

        response = {"overallRetention": 45, "weakest": [], "dueForReview": [], "teammateNudges": [], "briefing": ""}
        with mock.patch("urllib.request.urlopen", return_value=_mock_urlopen(response)):
            with mock.patch("sys.argv", ["cognition_briefing"]):
                cognition_briefing.main()
        out = capsys.readouterr().out
        assert "At risk" in out

    def test_empty_briefing(self, monkeypatch, capsys):
        _set_key(monkeypatch)
        import cognition_briefing

        response = {"overallRetention": 0, "weakest": [], "dueForReview": [], "teammateNudges": [], "briefing": "No data."}
        with mock.patch("urllib.request.urlopen", return_value=_mock_urlopen(response)):
            with mock.patch("sys.argv", ["cognition_briefing"]):
                cognition_briefing.main()
        out = capsys.readouterr().out
        assert "0%" in out


# ── cognition_review.py ──────────────────────────────────────────────


class TestCognitionReview:
    def test_no_reviews_needed(self, monkeypatch, capsys):
        _set_key(monkeypatch)
        import cognition_review

        response = {"suggestions": []}
        with mock.patch("urllib.request.urlopen", return_value=_mock_urlopen(response)):
            with mock.patch("sys.argv", ["cognition_review"]):
                cognition_review.main()
        out = capsys.readouterr().out
        assert "Nothing to review" in out

    def test_urgency_icons(self, monkeypatch, capsys):
        _set_key(monkeypatch)
        import cognition_review

        response = {
            "suggestions": [
                {"concept": "a", "topic": "X", "urgency": "high", "currentRetention": 30, "overdue": True, "reviewCount": 2},
                {"concept": "b", "topic": "Y", "urgency": "medium", "currentRetention": 55, "overdue": False, "reviewCount": 5},
                {"concept": "c", "topic": "Z", "urgency": "low", "currentRetention": 75, "overdue": False, "reviewCount": 10},
            ]
        }
        with mock.patch("urllib.request.urlopen", return_value=_mock_urlopen(response)):
            with mock.patch("sys.argv", ["cognition_review"]):
                cognition_review.main()
        out = capsys.readouterr().out
        assert "🔴" in out
        assert "🟡" in out
        assert "🟢" in out
        assert "OVERDUE" in out


# ── cognition_brain.py ───────────────────────────────────────────────


class TestCognitionBrain:
    def test_full_report(self, monkeypatch, capsys):
        _set_key(monkeypatch)
        import cognition_brain

        # Brain calls 3 MCP tools sequentially
        responses = [
            # get_user_retention
            {"overallRetention": 72, "totalNodes": 50, "topics": [
                {"topic": "Python", "avgRetention": 88, "nodeCount": 15, "status": "strong"},
                {"topic": "DevOps", "avgRetention": 52, "nodeCount": 8, "status": "critical"},
            ]},
            # get_weak_topics
            {"weakTopics": [{"concept": "docker", "topic": "DevOps", "retention": 42}]},
            # suggest_review
            {"suggestions": [{"concept": "docker", "topic": "DevOps", "urgency": "high", "currentRetention": 42}]},
        ]
        call_count = {"n": 0}

        def mock_urlopen(req, **kwargs):
            idx = call_count["n"]
            call_count["n"] += 1
            return _mock_urlopen(responses[idx % len(responses)])

        with mock.patch("urllib.request.urlopen", side_effect=mock_urlopen):
            with mock.patch("sys.argv", ["cognition_brain"]):
                cognition_brain.main()

        out = capsys.readouterr().out
        assert "72%" in out
        assert "Python" in out
        assert "DevOps" in out
        assert "docker" in out
        assert "cognitionus.com" in out


# ── cognition_batch.py ───────────────────────────────────────────────


class TestCognitionBatch:
    def test_missing_file_arg(self, monkeypatch):
        _set_key(monkeypatch)
        import cognition_batch

        with mock.patch("sys.argv", ["cognition_batch"]):
            with pytest.raises(SystemExit):
                cognition_batch.main()

    def test_successful_batch(self, monkeypatch, capsys, tmp_path):
        _set_key(monkeypatch)
        import cognition_batch

        concepts_file = tmp_path / "concepts.json"
        concepts_file.write_text(json.dumps([
            {"concept": "a", "topic": "X"},
            {"concept": "b", "topic": "Y"},
        ]))

        response = {"success": True, "ingested": 2, "created": 2, "updated": 0, "errors": []}
        with mock.patch("urllib.request.urlopen", return_value=_mock_urlopen(response)):
            with mock.patch("sys.argv", ["cognition_batch", str(concepts_file)]):
                cognition_batch.main()

        out = capsys.readouterr().out
        assert "2 ingested" in out
        assert "2 new" in out
        assert "0 errors" in out

    def test_hermes_agent_source(self, monkeypatch, tmp_path):
        _set_key(monkeypatch)
        import cognition_batch

        concepts_file = tmp_path / "c.json"
        concepts_file.write_text(json.dumps([{"concept": "test"}]))

        captured = {}

        def mock_urlopen(req, **kwargs):
            captured["data"] = json.loads(req.data)
            return _mock_urlopen({"ingested": 1, "created": 1, "errors": []})

        with mock.patch("urllib.request.urlopen", side_effect=mock_urlopen):
            with mock.patch("sys.argv", ["cognition_batch", str(concepts_file)]):
                cognition_batch.main()

        events = captured["data"]["params"]["arguments"]["events"]
        assert events[0]["source_integration"] == "hermes_agent"


# ── MCP JSON-RPC format ──────────────────────────────────────────────


class TestMCPProtocol:
    def test_jsonrpc_format(self, monkeypatch):
        _set_key(monkeypatch)
        import cognition_log

        captured = {}

        def mock_urlopen(req, **kwargs):
            captured["data"] = json.loads(req.data)
            captured["headers"] = dict(req.headers)
            return _mock_urlopen({"success": True, "isNew": True, "newRetention": 80, "nextReviewAt": "x"})

        with mock.patch("urllib.request.urlopen", side_effect=mock_urlopen):
            with mock.patch("sys.argv", ["cognition_log", "test"]):
                cognition_log.main()

        payload = captured["data"]
        assert payload["jsonrpc"] == "2.0"
        assert payload["method"] == "tools/call"
        assert payload["params"]["name"] == "log_learning"
        assert "Authorization" in captured["headers"]
        assert captured["headers"]["Authorization"].startswith("Bearer cog_me_")

    def test_accept_header_includes_sse(self, monkeypatch):
        _set_key(monkeypatch)
        import cognition_log

        captured = {}

        def mock_urlopen(req, **kwargs):
            captured["accept"] = req.get_header("Accept")
            return _mock_urlopen({"success": True, "isNew": True, "newRetention": 80, "nextReviewAt": "x"})

        with mock.patch("urllib.request.urlopen", side_effect=mock_urlopen):
            with mock.patch("sys.argv", ["cognition_log", "test"]):
                cognition_log.main()

        assert "text/event-stream" in captured["accept"]

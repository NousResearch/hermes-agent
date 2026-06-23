import asyncio
import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _valid_claude_handoff_body() -> str:
    return "\n".join(
        [
            "# Session context",
            "Claude investigated the current issue.",
            "# Key decisions",
            "Hermes remains the memory writer.",
            "# Open questions",
            "None.",
            "# Artifacts produced",
            "A short handoff packet.",
            "# Relevant existing memories",
            "No direct memory writes from Claude.",
            "# What Hermes should know",
            "Review this packet before promotion.",
        ]
    )


class TestClaudeHandoffMemory:
    def test_requires_canonical_sections(self):
        bad_body = "# Session context\nOnly one section."
        error = APIServerAdapter._claude_handoff_sections(bad_body)
        assert error is not None
        assert "required section headings" in error

    def test_rejects_mismatched_handoff_name_surface(self):
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(
            return_value={
                "description": "Test handoff",
                "body": _valid_claude_handoff_body(),
                "surface": "claudeai",
                "session_date": "2026-06-23",
                "name": "handoff-claudecode-linux-2026-06-23-abc123",
            }
        )

        response = asyncio.run(adapter._handle_claude_handoff_memory(request))
        payload = json.loads(response.text)

        assert response.status == 400
        assert payload["accepted"] is False
        assert payload["error"] == "handoff name surface does not match payload surface"

    def test_writes_pending_handoff_memory(self, monkeypatch):
        captured = {}

        class FakeClarenceDB:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def write_memory(self, **kwargs):
                captured.update(kwargs)
                return {"status": "created", "memory_id": 42}

        fake_module = types.ModuleType("clarence_db")
        setattr(fake_module, "ClarenceDB", FakeClarenceDB)
        monkeypatch.setitem(sys.modules, "clarence_db", fake_module)

        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(
            return_value={
                "description": "Test handoff",
                "body": _valid_claude_handoff_body(),
                "surface": "claudeai",
                "session_date": "2026-06-23",
                "source_session_id": "claude-session-1",
            }
        )

        response = asyncio.run(adapter._handle_claude_handoff_memory(request))
        payload = json.loads(response.text)

        assert response.status == 200
        assert payload["accepted"] is True
        assert payload["durable"] is True
        assert payload["memory_id"] == 42
        assert payload["name"].startswith("handoff-claudeai-2026-06-23-")
        assert captured["name"] == payload["name"]
        assert captured["type"] == "reference"
        assert captured["kind"] == "state"
        assert captured["author_agent"] == "claude-external"
        assert captured["tags"] == [
            "handoff-pending",
            "claude-handoff",
            "claudeai",
            "2026-06-23",
        ]

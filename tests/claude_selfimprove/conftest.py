"""Sandbox fixtures for the claude_selfimprove test suite.

Every test gets a synthetic ``HOME`` (so ``~/.claude`` and
``~/.claude-hermes`` resolve inside a tempdir) and a synthetic
``HERMES_HOME`` (so pipeline state never touches a real profile), matching
the isolation pattern used elsewhere in this repository
(``tests/conftest.py``, ``tests/agent/test_curator_backup.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Isolate HOME and HERMES_HOME, return a helper namespace."""
    home = tmp_path / "home"
    hermes_home = tmp_path / "hermes_home"
    home.mkdir()
    hermes_home.mkdir()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("CLAUDE_SELFIMPROVE_CLAUDE_DIR", raising=False)
    monkeypatch.delenv("CLAUDE_SELFIMPROVE_CLAUDE_HERMES_DIR", raising=False)

    class Sandbox:
        def __init__(self) -> None:
            self.home = home
            self.hermes_home = hermes_home
            self.claude_home = home / ".claude"
            self.claude_hermes_home = home / ".claude-hermes"

        def write_transcript(
            self, root: str, project: str, session_id: str, lines: list[dict]
        ) -> Path:
            """Write a synthetic Claude Code transcript JSONL file.

            ``root`` is "claude" or "claude-hermes". Each dict in ``lines``
            becomes one JSON line, matching the real transcript shape closely
            enough for the scanner/heuristics under test (a ``type``, a
            ``message`` with ``role``/``content``, and a session id).
            """
            base = self.claude_home if root == "claude" else self.claude_hermes_home
            proj_dir = base / "projects" / project
            proj_dir.mkdir(parents=True, exist_ok=True)
            path = proj_dir / f"{session_id}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for line in lines:
                    fh.write(json.dumps(line) + "\n")
            return path

    return Sandbox()


def user_turn(text: str, session_id: str = "sess-1") -> dict:
    return {
        "type": "user",
        "sessionId": session_id,
        "message": {"role": "user", "content": text},
    }


def assistant_turn(text: str, session_id: str = "sess-1") -> dict:
    return {
        "type": "assistant",
        "sessionId": session_id,
        "message": {"role": "assistant", "content": text},
    }

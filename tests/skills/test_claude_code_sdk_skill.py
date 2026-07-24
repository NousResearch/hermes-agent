"""Behavior tests for the optional claude-code-sdk skill."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "autonomous-ai-agents"
    / "claude-code-sdk"
    / "scripts"
    / "session_manager.py"
)


@pytest.fixture(scope="module")
def session_manager():
    spec = importlib.util.spec_from_file_location(
        "test_claude_code_sdk_session_manager", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


@pytest.fixture
def isolated_state(session_manager, tmp_path, monkeypatch):
    state_dir = tmp_path / "profile" / "skill-state" / "claude-code-sdk"
    monkeypatch.setattr(session_manager, "STATE_DIR", state_dir)
    monkeypatch.setattr(session_manager, "SESSIONS_FILE", state_dir / "sessions.json")
    monkeypatch.setattr(session_manager, "LOCK_FILE", state_dir / ".sessions.lock")
    monkeypatch.setattr(
        session_manager, "SESSION_LOCK_DIR", state_dir / ".session-locks"
    )
    monkeypatch.setattr(session_manager, "COST_LOG", state_dir / "cost.log")
    return state_dir


def test_hermes_home_honors_profile_and_platform_fallbacks(session_manager):
    assert session_manager._get_hermes_home(
        environ={"HERMES_HOME": "/profiles/work"},
        platform="linux",
        home=Path("/users/test"),
    ) == Path("/profiles/work")
    assert session_manager._get_hermes_home(
        environ={}, platform="darwin", home=Path("/users/test")
    ) == Path("/users/test/.hermes")
    assert session_manager._get_hermes_home(
        environ={"LOCALAPPDATA": "C:/Users/Test/AppData/Local"},
        platform="win32",
        home=Path("C:/Users/Test"),
    ) == Path("C:/Users/Test/AppData/Local/hermes")
    assert session_manager._get_hermes_home(
        environ={}, platform="win32", home=Path("C:/Users/Test")
    ) == Path("C:/Users/Test/AppData/Local/hermes")


def test_same_handle_lock_rejects_overlap(session_manager, isolated_state):
    if not (session_manager._HAS_FCNTL or session_manager._HAS_MSVCRT):
        pytest.skip("supported-platform locking primitive is unavailable")

    with session_manager._session_lock("same-handle"):
        with pytest.raises(session_manager.SessionBusyError):
            with session_manager._session_lock("same-handle"):
                pass

    with session_manager._session_lock("same-handle"):
        pass


def test_run_query_forwards_current_sdk_options(session_manager, tmp_path):
    observed = {}

    class FakeOptions:
        def __init__(self, **kwargs):
            observed["options"] = kwargs

    class FakeTextBlock:
        def __init__(self, text):
            self.text = text

    class FakeAssistantMessage:
        def __init__(self, content):
            self.content = content

    class FakeResultMessage:
        def __init__(self, cost, session_id):
            self.total_cost_usd = cost
            self.session_id = session_id
            self.is_error = False

    class FakeClient:
        def __init__(self, options):
            observed["client_options"] = options

        async def connect(self):
            observed["connected"] = True

        async def query(self, message):
            observed["message"] = message

        async def receive_response(self):
            yield FakeAssistantMessage([FakeTextBlock("SDK_OK")])
            yield FakeResultMessage(0.125, "claude-session-id")

        async def disconnect(self):
            observed["disconnected"] = True

    fake_sdk = (
        FakeAssistantMessage,
        FakeOptions,
        FakeClient,
        FakeResultMessage,
        FakeTextBlock,
    )
    with patch.object(session_manager, "_import_sdk", return_value=fake_sdk):
        result = asyncio.run(
            session_manager._run_query(
                str(tmp_path),
                "do the work",
                "resume-me",
                max_turns=7,
                max_budget_usd=1.5,
            )
        )

    assert observed["options"] == {
        "cwd": str(tmp_path),
        "resume": "resume-me",
        "max_turns": 7,
        "max_budget_usd": 1.5,
    }
    assert observed["message"] == "do the work"
    assert observed["connected"] is True
    assert observed["disconnected"] is True
    assert result == ("SDK_OK", 0.125, "claude-session-id")


def test_run_query_raises_for_sdk_error_result(session_manager, tmp_path):
    class FakeOptions:
        def __init__(self, **kwargs):
            pass

    class FakeAssistantMessage:
        pass

    class FakeTextBlock:
        pass

    class FakeResultMessage:
        is_error = True
        result = "Invalid authentication credentials"
        errors = []
        api_error_status = 401
        total_cost_usd = 0.0
        session_id = "failed-session"

    class FakeClient:
        def __init__(self, options):
            pass

        async def connect(self):
            pass

        async def query(self, message):
            pass

        async def receive_response(self):
            yield FakeResultMessage()

        async def disconnect(self):
            pass

    fake_sdk = (
        FakeAssistantMessage,
        FakeOptions,
        FakeClient,
        FakeResultMessage,
        FakeTextBlock,
    )
    with patch.object(session_manager, "_import_sdk", return_value=fake_sdk):
        with pytest.raises(
            RuntimeError, match=r"API status 401.*Invalid authentication"
        ):
            asyncio.run(
                session_manager._run_query(str(tmp_path), "hello", resume_id=None)
            )


def test_query_updates_resume_and_cost_state(
    session_manager, isolated_state, tmp_path, capsys, monkeypatch
):
    now = session_manager._now_iso()
    record = session_manager.SessionRecord(
        handle="abc123",
        project_path=str(tmp_path),
        created_at=now,
        last_activity=now,
    )
    session_manager._save_store(
        session_manager.SessionStore(sessions={record.handle: record})
    )

    observed = {}

    async def fake_run_query(project_path, message, resume_id, **kwargs):
        observed.update(
            project_path=project_path,
            message=message,
            resume_id=resume_id,
            options=kwargs,
        )
        return "done", 0.25, "claude-123"

    monkeypatch.setattr(session_manager, "_run_query", fake_run_query)
    args = argparse.Namespace(
        handle="abc123",
        message="fix it",
        max_turns=4,
        max_budget_usd=2.0,
        timeout=30,
    )

    assert session_manager.cmd_query(args) == 0
    payload = json.loads(capsys.readouterr().out)
    saved = session_manager._load_store().sessions["abc123"]

    assert observed == {
        "project_path": str(tmp_path),
        "message": "fix it",
        "resume_id": None,
        "options": {"max_turns": 4, "max_budget_usd": 2.0},
    }
    assert payload["text"] == "done"
    assert payload["total_cost_usd"] == 0.25
    assert saved.claude_session_id == "claude-123"
    assert saved.message_count == 1
    assert saved.total_cost_usd == 0.25


def test_query_parser_accepts_and_validates_limits(session_manager):
    parser = session_manager.build_parser()
    args = parser.parse_args([
        "query",
        "abc123",
        "hello",
        "--max-turns",
        "5",
        "--max-budget-usd",
        "1.25",
        "--timeout",
        "45",
    ])
    assert args.max_turns == 5
    assert args.max_budget_usd == 1.25
    assert args.timeout == 45

    with pytest.raises(SystemExit):
        parser.parse_args(["query", "abc123", "hello", "--max-turns", "0"])


def test_doctor_names_skill_and_package_correctly(
    session_manager, isolated_state, capsys
):
    with (
        patch.object(
            session_manager.importlib.metadata, "version", return_value="0.2.119"
        ),
        patch.object(session_manager, "_import_sdk", return_value=(object,) * 5),
    ):
        assert session_manager.cmd_doctor(argparse.Namespace()) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "status": "ok",
        "skill": "claude-code-sdk",
        "package": "claude-agent-sdk",
        "package_version": "0.2.119",
        "state_dir": str(isolated_state),
    }

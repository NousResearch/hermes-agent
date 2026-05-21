"""Tests for built-in workflow slash commands that wrap skills."""

from queue import Queue

import cli as cli_mod
from cli import HermesCLI


def test_cli_workflow_wrapper_queues_skill_invocation(monkeypatch):
    instance = object.__new__(HermesCLI)
    instance.session_id = "test-session"
    instance._pending_input = Queue()

    monkeypatch.setattr(
        cli_mod,
        "get_skill_commands",
        lambda: {"/autopilot": {"name": "autopilot"}},
    )
    monkeypatch.setattr(cli_mod, "resolve_skill_command_key", lambda name: f"/{name}")

    calls = {}

    def fake_build(cmd_key, user_instruction, task_id=None):
        calls["cmd_key"] = cmd_key
        calls["user_instruction"] = user_instruction
        calls["task_id"] = task_id
        return "LOADED AUTOPILOT"

    monkeypatch.setattr(cli_mod, "build_skill_invocation_message", fake_build)

    instance.process_command("/autopilot finish the goal")

    assert calls == {
        "cmd_key": "/autopilot",
        "user_instruction": "finish the goal",
        "task_id": "test-session",
    }
    assert instance._pending_input.get_nowait() == "LOADED AUTOPILOT"


def test_cli_deepinterview_alias_uses_canonical_skill(monkeypatch):
    instance = object.__new__(HermesCLI)
    instance.session_id = "test-session"
    instance._pending_input = Queue()

    monkeypatch.setattr(
        cli_mod,
        "get_skill_commands",
        lambda: {"/deep-interview": {"name": "deep-interview"}},
    )
    monkeypatch.setattr(
        cli_mod,
        "resolve_skill_command_key",
        lambda name: "/deep-interview" if name == "deep-interview" else None,
    )
    monkeypatch.setattr(
        cli_mod,
        "build_skill_invocation_message",
        lambda cmd_key, user_instruction, task_id=None: f"{cmd_key}:{user_instruction}:{task_id}",
    )

    instance.process_command("/deepinterview requirements")

    assert instance._pending_input.get_nowait() == "/deep-interview:requirements:test-session"


def test_cli_devflow_wrapper_queues_skill_invocation(monkeypatch):
    instance = object.__new__(HermesCLI)
    instance.session_id = "test-session"
    instance._pending_input = Queue()

    monkeypatch.setattr(
        cli_mod,
        "get_skill_commands",
        lambda: {"/devflow": {"name": "devflow"}},
    )
    monkeypatch.setattr(cli_mod, "resolve_skill_command_key", lambda name: f"/{name}")
    monkeypatch.setattr(
        cli_mod,
        "build_skill_invocation_message",
        lambda cmd_key, user_instruction, task_id=None: f"{cmd_key}:{user_instruction}:{task_id}",
    )

    instance.process_command("/devflow implement the reviewed workflow")

    assert instance._pending_input.get_nowait() == (
        "/devflow:implement the reviewed workflow:test-session"
    )

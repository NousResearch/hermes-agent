from hermes_cli.cli_agent_setup_mixin import (
    _remote_kanban_private_work,
    _remote_kanban_toolsets,
)


def test_remote_kanban_private_work_requires_task_and_protected_provider(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert _remote_kanban_private_work("openai-codex") is False

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_123")
    assert _remote_kanban_private_work("openai-codex") is True
    assert _remote_kanban_private_work("nous") is True
    assert _remote_kanban_private_work("ollama-launch") is False


def test_remote_kanban_toolsets_are_bounded_to_worker_capabilities():
    assert _remote_kanban_toolsets(["computer-use", "memory"]) == [
        "terminal",
        "file",
        "web",
    ]

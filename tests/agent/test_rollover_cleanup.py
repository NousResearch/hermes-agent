"""Resource ownership contracts for automatic rollover cleanup."""

import threading
from types import SimpleNamespace


def _agent(owner_task_id):
    return SimpleNamespace(
        session_id="durable-session-id",
        _process_owner_task_ids={owner_task_id},
        _active_children_lock=threading.Lock(),
        _active_children=[],
        _memory_manager=None,
        _session_messages=[],
        context_compressor=SimpleNamespace(on_session_end=lambda *_args: None),
        release_clients=lambda: None,
    )


def _install_cleanup_isolates(monkeypatch, processes, running_processes=()):
    from tools import browser_tool_lifecycle, file_tools
    from tools.computer_use import tool as computer_use
    from tools.process_registry import process_registry
    from tools import terminal_tool

    monkeypatch.setattr(process_registry, "list_sessions", lambda: processes)
    monkeypatch.setattr(
        process_registry, "running_owned_by",
        lambda owner: [process for process in running_processes if process.owner_task_id == owner],
    )
    monkeypatch.setattr(browser_tool_lifecycle, "cleanup_browser", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(computer_use, "release_computer_use_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(file_tools, "clear_file_ops_cache", lambda _owner: None)
    monkeypatch.setattr(
        terminal_tool, "_session_scope",
        lambda: terminal_tool._SessionScope(env_type="ssh", persistent=False),
    )
    return terminal_tool


def test_rollover_cleanup_resolves_the_gateway_terminal_environment_owner(monkeypatch):
    from agent.rollover_cleanup import close_for_rollover
    from gateway.session_context import scoped_session_key

    owner_task_id = "rollover-turn-owner"
    session_key = "agent:main:rollover-lane"
    processes = []
    terminal_tool = _install_cleanup_isolates(monkeypatch, processes)
    cleanup_attempts = []
    env = SimpleNamespace(cleanup=lambda: cleanup_attempts.append("cleanup"))

    with scoped_session_key(session_key):
        env_key = terminal_tool._resolve_container_task_id(owner_task_id)
        assert env_key == "session:agent:main:rollover-lane"
        terminal_tool._active_environments[env_key] = env
        try:
            close_for_rollover(_agent(owner_task_id), lambda _process: True)
            assert env_key not in terminal_tool._active_environments
            assert cleanup_attempts == ["cleanup"]
        finally:
            terminal_tool._active_environments.pop(env_key, None)
            terminal_tool._last_activity.pop(env_key, None)
            terminal_tool._creation_locks.pop(env_key, None)


def test_rollover_cleanup_preserves_environment_used_by_independent_process(monkeypatch):
    from agent.rollover_cleanup import close_for_rollover
    from gateway.session_context import scoped_session_key

    owner_task_id = "rollover-turn-owner"
    session_key = "agent:main:rollover-lane"
    running_process = SimpleNamespace(
        owner_task_id="independent-process-owner",
        task_id="session:agent:main:rollover-lane",
    )
    terminal_tool = _install_cleanup_isolates(
        monkeypatch,
        [{"owner_task_id": "independent-process-owner", "status": "running"}],
        [running_process],
    )
    cleanup_attempts = []
    env = SimpleNamespace(cleanup=lambda: cleanup_attempts.append("cleanup"))

    with scoped_session_key(session_key):
        env_key = terminal_tool._resolve_container_task_id(owner_task_id)
        terminal_tool._active_environments[env_key] = env
        try:
            close_for_rollover(_agent(owner_task_id), lambda _process: True)
            assert terminal_tool._active_environments[env_key] is env
            assert cleanup_attempts == []
        finally:
            terminal_tool._active_environments.pop(env_key, None)
            terminal_tool._last_activity.pop(env_key, None)
            terminal_tool._creation_locks.pop(env_key, None)

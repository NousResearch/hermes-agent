"""Checked session resource settlement, without terminating independent infrastructure."""
from __future__ import annotations


def close_for_rollover(agent, callback_is_settled) -> None:
    """Run off-loop in the owning profile. A failed cleanup never authorizes cutover.

    Unlike close(), this neither ends the durable session nor kills processes. The
    store owns the first operation; independently running infrastructure owns the
    lifetime of the environments it is using.
    """
    from tools.browser_tool_lifecycle import cleanup_browser
    from tools.computer_use.tool import release_computer_use_session
    from tools.file_tools import clear_file_ops_cache
    from tools.process_registry import process_registry
    from tools.terminal_tool import _resolve_container_task_id
    from tools.terminal_tool_lifecycle import cleanup_vm

    with agent._active_children_lock:
        if agent._active_children:
            raise RuntimeError("Child-agent cleanup has not settled")
    owners = set(getattr(agent, "_process_owner_task_ids", ()))
    owners.add(agent.session_id)
    all_processes = process_registry.list_sessions()
    processes = [p for p in all_processes if p["owner_task_id"] in owners]
    for process in processes:
        if (process.get("notify_on_complete") or process.get("watch_patterns")) and not callback_is_settled(process):
            raise RuntimeError("A callback-dependent process has not settled")
    infrastructure_owners = {p["owner_task_id"] for p in processes if p["status"] == "running"}
    running_process_owners = {p["owner_task_id"] for p in all_processes if p["status"] == "running"}
    running_environment_owners = {
        process.task_id
        for owner in running_process_owners
        for process in process_registry.running_owned_by(owner)
        if process.task_id
    }
    for owner in owners:
        cleanup_browser(owner, strict=True)
        release_computer_use_session(owner, strict=True)
        terminal_owner = _resolve_container_task_id(owner)
        if (owner not in infrastructure_owners and terminal_owner not in infrastructure_owners
                and terminal_owner not in running_environment_owners):
            cleanup_vm(terminal_owner, strict=True)
        clear_file_ops_cache(owner)

    manager = agent._memory_manager
    if manager is not None:
        if not manager.flush_pending(timeout=10):
            raise RuntimeError("Memory writes have not settled")
        manager.on_session_end(agent._session_messages or [], strict=True)
        if not manager.flush_pending(timeout=10):
            raise RuntimeError("End-of-session memory writes have not settled")
        manager.shutdown_all(strict=True)
        if manager.shutdown_drain_state["status"] != "drained":
            raise RuntimeError("Memory shutdown did not drain")
    agent.context_compressor.on_session_end(agent.session_id, agent._session_messages or [])
    agent._memory_provider_shutdown = True
    # Native retirement preserves cross-thread transport safety. No old-session row
    # is finalized here: a failed store commit still leaves that owner authoritative.
    agent.release_clients()

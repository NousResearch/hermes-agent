"""Mirror committed built-in ``skill_manage`` writes to external memory providers.

Split out of ``agent/agent_runtime_helpers.py`` (code-health size ratchet); the helper only needs
attributes off the agent, so it is a leaf module.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def mirror_skill_write_to_memory_providers(
    agent: Any,
    function_name: str,
    tool_result: Any,
    tool_args: dict[str, Any],
    *,
    task_id: str = "",
    tool_call_id: Optional[str] = None,
) -> None:
    """Mirror a committed built-in skill_manage write to external providers.

    Shared by both executor paths (the sequential inline dispatcher in
    agent/tool_executor.py and the concurrent ``invoke_tool`` below) so skill
    writes are observed consistently regardless of which executor ran the
    tool. No-op for every other tool. All gating (committed-only, mutating
    actions, action→payload mapping) lives behind the manager interface
    (``MemoryManager.notify_skill_tool_write``).
    """
    if function_name != "skill_manage":
        return
    memory_manager = getattr(agent, "_memory_manager", None)
    if not memory_manager:
        return
    memory_manager.notify_skill_tool_write(
        tool_result,
        tool_args,
        build_metadata=lambda: agent._build_memory_write_metadata(
            task_id=task_id,
            tool_call_id=tool_call_id,
            tool_name="skill_manage",
        ),
    )

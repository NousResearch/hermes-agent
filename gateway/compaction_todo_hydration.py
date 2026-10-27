"""Structured todo hydration for detached gateway compaction paths."""

from types import SimpleNamespace
from typing import Any

from tools.todo_tool import TODO_INJECTION_HEADER, TodoStore


def _row_text(msg: Any) -> str:
    content = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(content, list):
        content = "\n".join(str(p.get("text") or "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return content if isinstance(content, str) else ""


def hydrate_todos_for_compaction(agent: Any, messages: list) -> Any:
    """Restore structured state, or carry the latest rendered block without interpreting it.

    The durable reader includes paired results summarised out of the visible transcript. Without
    one, only structured results AFTER the newest folded snapshot can supersede that snapshot.
    """
    if not isinstance(getattr(agent, "_todo_store", None), TodoStore):
        return agent
    start = next((i + 1 for i in range(len(messages) - 1, -1, -1)
                  if messages[i].get("role") == "user" and TODO_INJECTION_HEADER in _row_text(messages[i])), 0)
    history = messages[start:]
    getter = getattr(getattr(agent, "_session_db", None), "get_latest_todo_result", None)
    content = getter(agent.session_id) if callable(getter) and agent.session_id else None
    if isinstance(content, str):
        # The durable reader already verified pairing. Replay through the model's existing parser
        # and TodoStore validation rather than introducing another result decoder.
        history = [
            {"role": "assistant", "tool_calls": [{"id": "compaction-todo", "type": "function",
             "function": {"name": "todo_list", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "compaction-todo", "content": content},
        ]
    found = agent._latest_todo_response(history)
    agent._compaction_todo_snapshot = None
    agent._compaction_todo_authoritative = found is not None
    if found is not None:
        todos, revision = found
        agent._todo_store.restore(todos, revision=revision)
    elif start:
        text = _row_text(messages[start - 1])
        from agent.conversation_compression import _PRUNED_SKILL_RELOAD_NOTICE_HEADER
        # The boundary fold appends a fresh skill-reload notice; carrying the old one too would stack them.
        snapshot = text[text.index(TODO_INJECTION_HEADER):]
        agent._compaction_todo_snapshot = snapshot.split("\n\n" + _PRUNED_SKILL_RELOAD_NOTICE_HEADER, 1)[0]
    return agent


def fold_todos_for_compaction(messages: list, compressed: list, session_db: Any, session_id: str) -> list:
    """Use the agent's hydration parser and boundary fold without building a runtime agent."""
    from run_agent import AIAgent
    from agent.agent_runtime_helpers import repair_message_sequence
    from agent.conversation_compression import _fold_todo_snapshot

    state = SimpleNamespace(_todo_store=TodoStore(), _session_db=session_db, session_id=session_id,
                            _tool_response_matches_todo_call=AIAgent._tool_response_matches_todo_call)
    state._latest_todo_response = lambda history: AIAgent._latest_todo_response(state, history)
    state._repair_message_sequence = lambda rows: repair_message_sequence(state, rows)
    hydrate_todos_for_compaction(state, messages)
    _fold_todo_snapshot(state, compressed)
    return compressed

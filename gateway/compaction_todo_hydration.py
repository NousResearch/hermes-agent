"""Structured todo hydration for detached gateway compaction paths."""

from functools import partial
from typing import Any

from agent.conversation_compression_todo import compaction_todo_snapshot, fold_todo_snapshot
from agent.todo_hydration import decode_todo_result, latest_todo_response
from tools.todo_tool import TODO_INJECTION_HEADER, TodoStore


def _row_text(msg: Any) -> str:
    content = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(content, list):
        content = "\n".join(str(p.get("text") or "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return content if isinstance(content, str) else ""


def _hydrate_compaction_store(store: TodoStore, messages: list, session_db: Any, session_id: str) -> tuple[str | None, bool]:
    """Restore structured state, or carry the latest rendered block without interpreting it.

    The durable reader includes paired results summarised out of the visible transcript. Without
    one, only structured results AFTER the newest folded snapshot can supersede that snapshot.
    """
    start = next((i + 1 for i in range(len(messages) - 1, -1, -1)
                  if messages[i].get("role") == "user" and TODO_INJECTION_HEADER in _row_text(messages[i])), 0)
    getter = getattr(session_db, "get_latest_todo_result", None)
    content = getter(session_id) if callable(getter) and session_id else None
    # Durable results are already paired; visible history must still verify the assistant call.
    found = (decode_todo_result(content, session_id=session_id) if isinstance(content, str)
             else latest_todo_response(messages[start:], session_id=session_id))
    snapshot = None
    if found is not None:
        todos, revision = found
        store.restore(todos, revision=revision)
    elif start:
        text = _row_text(messages[start - 1])
        from agent.conversation_compression import _PRUNED_SKILL_RELOAD_NOTICE_HEADER
        # The boundary fold appends a fresh skill-reload notice; carrying the old one too would stack them.
        snapshot = text[text.index(TODO_INJECTION_HEADER):]
        snapshot = snapshot.split("\n\n" + _PRUNED_SKILL_RELOAD_NOTICE_HEADER, 1)[0]
    return snapshot, found is not None


def hydrate_todos_for_compaction(agent: Any, messages: list) -> Any:
    """Seed an existing detached agent without requiring any of its private decoder methods."""
    if isinstance(getattr(agent, "_todo_store", None), TodoStore):
        agent._compaction_todo_snapshot, agent._compaction_todo_authoritative = _hydrate_compaction_store(
            agent._todo_store, messages, getattr(agent, "_session_db", None), agent.session_id,
        )
    return agent


def fold_todos_for_compaction(messages: list, compressed: list, session_db: Any, session_id: str) -> list:
    """Hydrate and fold explicit todo state without constructing or impersonating an agent."""
    from agent.agent_runtime_helpers import repair_message_sequence

    store = TodoStore()
    fallback, authoritative = _hydrate_compaction_store(store, messages, session_db, session_id)
    snapshot, authoritative = compaction_todo_snapshot(store, fallback=fallback, authoritative=authoritative)
    fold_todo_snapshot(compressed, snapshot, authoritative=authoritative,
                       repair_sequence=partial(repair_message_sequence, None))
    return compressed

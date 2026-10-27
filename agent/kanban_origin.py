"""Executor-owned provenance: bind only committed user rows, never tool arguments."""
from contextlib import contextmanager

from agent.context_compressor import user_originated_turn_view
from agent.session_persistence import _is_ephemeral_scaffolding
from tools.kanban_provenance import bind_conversation_origin, conversation_origin


@contextmanager
def executor_conversation_origin(agent, function_name: str, messages: list | None):
    """Flush before creation, then resolve the selected live dict's exact committed row.

    A disabled/failed persistence path creates no excerpt or guessed row anchor. The
    durable payload (not api_content) owns clean/native persist overrides.
    """
    origin = None
    if function_name == "kanban_create" and messages and not getattr(agent, "_persist_disabled", False):
        db = getattr(agent, "_session_db", None)
        if db is not None and agent._flush_messages_to_session_db(messages) is True:
            selected = next((msg for msg in reversed(messages)
                             if not _is_ephemeral_scaffolding(msg)
                             and user_originated_turn_view(msg) is not None), None)
            row_id = selected.get("_row_id") if selected else None
            if isinstance(row_id, int):
                around = db.get_messages_around(agent.session_id, row_id, window=0)
                row = next((r for r in around.get("window", [])
                            if r["id"] == row_id and r["role"] == "user"), None)
                if row and selected is not None:
                    origin = conversation_origin(agent.session_id) or {}
                    origin["message_row_id"] = row_id
                    origin.pop("message_id", None)  # The selected row, not the initial ingress, owns this anchor.
                    if row.get("platform_message_id"):
                        origin["message_id"] = row["platform_message_id"]
                    # Session persistence owns native-part projection and clean overrides.
                    # JSON-looking user prose stays literal text, never a second envelope.
                    durable_view = user_originated_turn_view(dict(selected, content=row.get("content")))
                    origin["prompt_excerpt"] = durable_view.get("content") if durable_view else None
    from hermes_cli.kanban_db_origin import sanitize_origin
    with bind_conversation_origin(sanitize_origin(origin)):
        yield

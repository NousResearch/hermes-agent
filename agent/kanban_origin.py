"""Executor-owned provenance: bind only committed user rows, never tool arguments."""
from contextlib import contextmanager

from agent.agent_init_session import agent_interface
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
        store = agent_interface(agent)
        if store is not None and agent._flush_messages_to_session_db(messages) is True:
            selected = next((msg for msg in reversed(messages)
                             if not _is_ephemeral_scaffolding(msg)
                             and user_originated_turn_view(msg) is not None), None)
            row_id = selected.get("_row_id") if selected else None
            if isinstance(row_id, int):
                around = store.messages.read_around(agent.session_id, row_id, window=0)
                row = next((r for r in around.window if r.id == row_id and r.role == "user"), None)
                if row and selected is not None:
                    origin = conversation_origin(agent.session_id) or {}
                    origin["message_row_id"] = row_id
                    origin.pop("message_id", None)  # The selected row, not the initial ingress, owns this anchor.
                    if row.platform_message_id:
                        origin["message_id"] = row.platform_message_id
                    # Session persistence owns native-part projection and clean overrides.
                    # JSON-looking user prose stays literal text, never a second envelope.
                    durable_view = user_originated_turn_view(dict(selected, content=row.content))
                    origin["prompt_excerpt"] = durable_view.get("content") if durable_view else None
    from hermes_cli.kanban_db_origin import sanitize_origin
    with bind_conversation_origin(sanitize_origin(origin)):
        yield


def dispatch_with_origin(agent, messages: list | None, function_name: str, *args, **kwargs):
    """``model_tools.handle_function_call`` under this call's conversation origin (the concurrent path)."""
    import model_tools
    with executor_conversation_origin(agent, function_name, messages):
        return model_tools.handle_function_call(function_name, *args, **kwargs)

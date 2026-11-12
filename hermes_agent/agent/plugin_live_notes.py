"""Tell an open chat about a plugin that went live while it runs, once, without breaking its cache.

An install from any surface (the onboarding plugins card, Settings, ``hermes plugins install`` with a
serve backend) makes the plugin's MCP tools and skills usable in this profile's open chats at once
(``hermes_cli.plugins_activation.load_and_go_live``) and queues the notice here, one per plugin key: a
second install or update of the same plugin replaces its notice. The notice rides the next tool result
of the running turn (``agent/tool_executor.py``), after its content: a new row not yet flushed, so no
cached byte moves. When no tool runs again this turn, the next user message carries it instead
(``agent/turn_context.py::_merge_gateway_notes``). Either way it is popped, so it is delivered once.
"""

from __future__ import annotations

import threading
from typing import Any

from agent.message_content import flatten_message_text

_QUEUED = "_plugin_live_notes"
_IN_BATCH = "_plugin_notes_in_tool_batch"
# The install RPC thread queues while the agent thread delivers.
_LOCK = threading.Lock()


def _queued(agent: Any) -> dict[str, str]:
    notes = getattr(agent, _QUEUED, None)
    return notes if isinstance(notes, dict) else {}


def queue_plugin_notice(agent: Any, plugin_key: str, note: str) -> None:
    """Queue ``note`` for ``agent``'s next tool result or user turn; it replaces this plugin's older one."""
    with _LOCK:
        notes = {key: text for key, text in _queued(agent).items() if key != plugin_key}
        notes[plugin_key] = note
        setattr(agent, _QUEUED, notes)


def pop_plugin_notices(agent: Any) -> str:
    """Every queued notice, oldest plugin first, and clear the queue."""
    with _LOCK:
        notes = _queued(agent)
        if notes:
            setattr(agent, _QUEUED, {})
    return "\n\n".join(notes.values())


def attach_to_tool_message(agent: Any, message: dict) -> None:
    """Append every queued notice to ``message``, the tool result about to be appended and flushed. It
    goes after the content, so a result wrapped as untrusted data keeps the notice outside the wrapper."""
    notes = pop_plugin_notices(agent)
    if not notes:
        return
    _append(message, notes)
    in_batch = getattr(agent, _IN_BATCH, None)
    in_batch = in_batch if isinstance(in_batch, dict) else {}
    setattr(agent, _IN_BATCH, {**in_batch, message.get("tool_call_id"): notes})


def keep_after_turn_budget(agent: Any, batch: list[dict]) -> None:
    """Put a notice back on a batch row whose content the turn budget spilled to a file: the inline
    preview keeps only the head, and the notice sits at the tail. The flushed row already holds it."""
    in_batch = getattr(agent, _IN_BATCH, None)
    if not isinstance(in_batch, dict) or not in_batch:
        return
    setattr(agent, _IN_BATCH, {})
    for message in batch:
        notes = in_batch.get(message.get("tool_call_id"))
        if notes and notes not in flatten_message_text(message.get("content")):
            _append(message, notes)


def _append(message: dict, notes: str) -> None:
    content = message.get("content")
    if isinstance(content, list):
        message["content"] = [*content, {"type": "text", "text": notes}]
    else:
        message["content"] = f"{content or ''}\n\n{notes}"

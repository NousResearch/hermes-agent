"""The gateway's askers for ``PluginContext.ask_choice`` (``hermes_cli/plugin_choices.py``).

Both reuse the clarify tool's own prompt, so every surface renders a plugin command's choice the
way it renders the agent's questions:

* a messaging chat gets ``adapter.send_clarify``: buttons where the adapter has them (Telegram,
  Discord, Slack), the base numbered list elsewhere. A tap resolves it through
  ``resolve_gateway_clarify``; a typed reply reaches the clarify text-intercept even though the
  command still holds the chat (the adapter's clarify bypass), and an undeliverable card falls back
  to the numbered text, as for the clarify tool;
* a local session (classic CLI, Ink TUI, Desktop) gets the ``clarify`` server request on its
  viewers' sockets, answered by their clarify panel.

The handler runs on an executor thread (``_hm_run_plugin_handler``), so blocking here is safe; the
wait follows the clarify timeout.
"""
from __future__ import annotations

import asyncio
import uuid
from contextlib import nullcontext
from typing import Optional

from gateway.config import Platform


def chat_choices(runner, source, session_key: str):
    """Bind the messaging asker for a command *source* sent, unless the caller bound one already (a
    local viewer's ``slash.exec``) or the source is a local session (it has no chat to post in)."""
    from hermes_cli.plugin_choices import asker_bound, bind_asker
    if asker_bound() or source.platform == Platform.LOCAL:
        return nullcontext()
    loop = asyncio.get_running_loop()
    return bind_asker(lambda question, labels: _ask_chat(runner, source, session_key, loop, question, labels))


def _ask_chat(runner, source, session_key, loop, question, labels) -> Optional[str]:
    from gateway.run_turn_runner_clarify_delivery import _clarify_send_then_wait, text_fallback_coro
    from tools import clarify_gateway
    adapter = runner._delivery_adapter_for(source)
    if adapter is None:
        return None
    clarify_id = uuid.uuid4().hex[:10]
    send = dict(chat_id=source.chat_id, question=question, choices=labels, clarify_id=clarify_id,
                session_key=session_key, metadata=runner._thread_metadata_for_source(source))
    clarify_gateway.register(clarify_id, session_key, question, labels)

    def fallback():
        coro = text_fallback_coro(adapter, **send)
        return None if coro is None else asyncio.run_coroutine_threadsafe(coro, loop)
    card = asyncio.run_coroutine_threadsafe(adapter.send_clarify(**send), loop)
    response, answered = _clarify_send_then_wait(card, clarify_id=clarify_id, session_key=session_key,
                                                 clarify_mod=clarify_gateway, fallback=fallback)
    if answered:
        return response
    retire = getattr(type(adapter), "retire_clarify_card", None)
    if callable(retire):  # a persistent card must stop looking answerable
        from gateway.run_turn_runner import _clarify_expired_notice
        asyncio.run_coroutine_threadsafe(retire(adapter, clarify_id, _clarify_expired_notice()), loop)
    return None


def viewer_choices(live, session_id: str):
    """Bind the local-session asker: the ``clarify`` server request to *live*'s attached viewers."""
    from hermes_cli.plugin_choices import bind_asker
    return bind_asker(lambda question, labels: _ask_viewers(live, session_id, question, labels))


def _ask_viewers(live, session_id, question, labels) -> Optional[str]:
    from gateway.session_server_requests import ask_viewer_result
    from tools.clarify_gateway import get_clarify_timeout
    timeout = get_clarify_timeout()
    result = ask_viewer_result(live.event_stream, session_id, "clarify",
                               {"questions": [{"qid": "q0", "question": question, "choices": labels,
                                               "multi_select": False}]},
                               timeout=timeout if timeout > 0 else None, qids=["q0"])
    answer = ((result or {}).get("answers") or {}).get("q0")
    return answer if isinstance(answer, str) else None

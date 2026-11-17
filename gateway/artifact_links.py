"""Hosted artifact viewer links for chat replies.

A chat cannot render an artifact, so the reply must carry its viewer page. The model is told to
share it, but delivery cannot depend on that: :func:`append_viewer_links` adds the link of every
artifact this turn's tool calls made, changed or showed whose viewer URL the reply lacks. Only
``viewer_url`` is ever added (``tools.hosted_artifacts.viewer_link``), never the view ticket.
"""

from __future__ import annotations

from typing import Any

from tools.hosted_artifacts import ViewerLink, viewer_link


def _tool_names_by_call_id(messages: list[dict[str, Any]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or ():
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict):
                continue
            call_id = call.get("id") or call.get("call_id")
            if call_id and isinstance(function.get("name"), str):
                names[str(call_id)] = function["name"]
    return names


def turn_viewer_links(messages: Any, history_offset: int) -> list[ViewerLink]:
    """Viewer links from this turn's tool results (``messages[history_offset:]``), first seen first,
    one per URL. A list compression shrank below the offset yields nothing: its rows can no longer be
    told apart from earlier turns', and an earlier turn's link must not ride this reply."""
    if not isinstance(messages, list):
        return []
    turn = [m for m in messages[history_offset:] if isinstance(m, dict)]
    names = _tool_names_by_call_id(turn)
    links: dict[str, ViewerLink] = {}
    for msg in turn:
        if msg.get("role") != "tool":
            continue
        name = msg.get("name") or names.get(str(msg.get("tool_call_id") or ""))
        link = viewer_link(name, msg.get("content"))
        if link is not None:
            links.setdefault(link.url, link)
    return list(links.values())


def append_viewer_links(final_response: str, messages: Any, history_offset: int) -> str:
    """*final_response* plus one line per artifact link of this turn that it does not already contain."""
    missing = [link.text() for link in turn_viewer_links(messages, history_offset)
               if link.url not in final_response]
    if not missing:
        return final_response
    return "\n\n".join([final_response.rstrip(), "\n".join(missing)]) if final_response.strip() else "\n".join(missing)

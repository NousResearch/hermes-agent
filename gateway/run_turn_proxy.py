"""Construct proxy chat requests with text-only history and session continuity headers."""

from __future__ import annotations

from typing import Dict, List


def _proxy_request_payload(message: str, context_prompt: str, history: list,
                           session_id: str, proxy_key: str) -> tuple[dict, dict]:
    """Build the remote chat request while retaining its session continuity headers."""
    # OpenAI chat format. The remote keeps continuity via X-Hermes-Session-Id; send the current
    # message plus a compact text-only history for a remote that has none yet.
    api_messages: List[Dict[str, str]] = [{"role": "system", "content": context_prompt}] if context_prompt else []
    api_messages += [
        {"role": msg.get("role"), "content": msg.get("content")}
        for msg in history if msg.get("role") in {"user", "assistant"} and msg.get("content")
    ]
    api_messages.append({"role": "user", "content": message})

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if proxy_key:
        headers["Authorization"] = f"Bearer {proxy_key}"
    if session_id:
        headers["X-Hermes-Session-Id"] = session_id
    body = {"model": "hermes-agent", "messages": api_messages, "stream": True}
    return body, headers

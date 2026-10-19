"""Build an authenticated gateway request with its verified recall identity."""
from dataclasses import dataclass
from typing import Any

from gateway.recall_scope import (
    GATEWAY_PROXY_MARKER_HEADER, GATEWAY_PROXY_ORIGIN_HEADER,
    GATEWAY_PROXY_SESSION_KEY_HEADER, gateway_proxy_request_context,
)


@dataclass(frozen=True)
class ProxyRequest:
    url: str
    headers: dict[str, str]
    body: dict[str, Any]


def build_proxy_request(proxy_url, proxy_key, source, session_id, session_key,
                        message, context_prompt, history) -> ProxyRequest:
    path, origin, key = gateway_proxy_request_context(source, session_id, session_key)
    messages = [{"role": "system", "content": context_prompt}] if context_prompt else []
    messages.extend(
        {"role": row.get("role"), "content": row.get("content")}
        for row in history if row.get("role") in {"user", "assistant"} and row.get("content")
    )
    messages.append({"role": "user", "content": message})
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {proxy_key}"}
    if origin is not None:
        headers[GATEWAY_PROXY_MARKER_HEADER] = "1"
        headers[GATEWAY_PROXY_ORIGIN_HEADER] = origin
        headers[GATEWAY_PROXY_SESSION_KEY_HEADER] = key
    if session_id:
        headers["X-Hermes-Session-Id"] = session_id
    return ProxyRequest(proxy_url + path, headers,
                        {"model": "hermes-agent", "messages": messages, "stream": True})

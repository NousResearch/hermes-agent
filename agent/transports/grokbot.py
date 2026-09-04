"""Native Grok Bot (Cursor/sand) Hermes transport. No OpenAI proxy.

``api_mode: grokbot`` talks ConnectRPC protobuf in-process via
``agent.grokbot.native``. Auth is ``~/.grokbot/session.json``.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from agent.transports.base import ProviderTransport
from agent.transports import register_transport

_CURSOR_HOST = "api2.cursor.sh"


def load_native():
    from agent.grokbot import native

    return native


def build_grokbot_client(agent: Any = None):
    return load_native().build_grokbot_client(agent)


def _hostname(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    try:
        return (urlparse(raw).hostname or "").lower()
    except Exception:
        return ""


def grokbot_runtime_active(
    agent: Any = None,
    *,
    api_mode: str = "",
    provider: str = "",
    base_url: str = "",
) -> bool:
    """True when this agent/request should use the Grok Bot facade.

    Matches explicit api_mode/provider, or the exact Cursor inference host.
    Substring matching is not used. ``https://evil-api2.cursor.sh.example``
    must not hijack.
    """
    mode = (api_mode or getattr(agent, "api_mode", "") or "").strip().lower()
    if mode == "grokbot":
        return True
    prov = (
        provider
        or getattr(agent, "provider", "")
        or getattr(agent, "requested_provider", "")
        or ""
    ).strip().lower()
    if prov in {"grokbot", "grok-bot"}:
        return True
    url = (base_url or getattr(agent, "base_url", "") or "").strip()
    if _hostname(url) == _CURSOR_HOST:
        return True
    kwargs = getattr(agent, "_client_kwargs", None) or {}
    if isinstance(kwargs, dict) and _hostname(str(kwargs.get("base_url") or "")) == _CURSOR_HOST:
        return True
    return False


class GrokbotTransport(ProviderTransport):
    def __init__(self):
        self._inner = load_native().GrokbotTransport()

    @property
    def api_mode(self) -> str:
        return "grokbot"

    def convert_messages(self, messages, **kwargs):
        return self._inner.convert_messages(messages, **kwargs)

    def convert_tools(self, tools):
        return self._inner.convert_tools(tools)

    def build_kwargs(self, model, messages, tools=None, **params):
        return self._inner.build_kwargs(model, messages, tools=tools, **params)

    def normalize_response(self, response, **kwargs):
        return self._inner.normalize_response(response, **kwargs)

    def validate_response(self, response) -> bool:
        return self._inner.validate_response(response)

    def extract_cache_stats(self, response):
        return self._inner.extract_cache_stats(response)

    def map_finish_reason(self, raw_reason: str) -> str:
        return self._inner.map_finish_reason(raw_reason)


register_transport("grokbot", GrokbotTransport)

"""OpenRouter xAI ``:online`` must not duplicate the client ``web_search`` tool.

Regression for #76481: an OpenRouter ``x-ai/…:online`` request already carries
OpenRouter's own server-side ``web_search`` tool, so forwarding Hermes' client
``web_search`` as well makes xAI reject the request with
``HTTP 400: Duplicate tool names: web_search``.  The client ``web_search`` is
dropped only for that exact provider+namespace+suffix combination; every other
tool and every other request keeps its tools unchanged.
"""

import pytest

from agent.transports.chat_completions import (
    ChatCompletionsTransport,
    _omit_openrouter_online_web_search,
)
from providers import get_provider_profile

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _fn_tool(name):
    return {"type": "function", "function": {"name": name, "parameters": {}}}


def _flat_tool(name):
    return {"name": name, "parameters": {}}


def _tool_names(tools):
    out = []
    for t in tools or []:
        fn = t.get("function")
        out.append(fn["name"] if isinstance(fn, dict) else t.get("name"))
    return out


# ── Unit tests on the filter helper ─────────────────────────────────────────


class TestOmitHelper:
    def test_drops_client_web_search_openai_shape(self):
        tools = [_fn_tool("web_search"), _fn_tool("web_extract"), _fn_tool("terminal")]
        out = _omit_openrouter_online_web_search(
            tools, "x-ai/grok-4.5:online", OPENROUTER_BASE
        )
        assert _tool_names(out) == ["web_extract", "terminal"]

    def test_drops_client_web_search_flat_shape(self):
        tools = [_flat_tool("web_search"), _flat_tool("web_extract")]
        out = _omit_openrouter_online_web_search(
            tools, "x-ai/grok-4.5:online", OPENROUTER_BASE
        )
        assert _tool_names(out) == ["web_extract"]

    def test_namespace_prefixed_model_still_matches(self):
        tools = [_fn_tool("web_search"), _fn_tool("web_extract")]
        out = _omit_openrouter_online_web_search(
            tools, "openrouter/x-ai/grok-4.5:online", OPENROUTER_BASE
        )
        assert _tool_names(out) == ["web_extract"]

    def test_bare_grok_keeps_web_search(self):
        tools = [_fn_tool("web_search"), _fn_tool("web_extract")]
        out = _omit_openrouter_online_web_search(
            tools, "x-ai/grok-4.5", OPENROUTER_BASE
        )
        assert _tool_names(out) == ["web_search", "web_extract"]

    def test_non_openrouter_xai_keeps_web_search(self):
        # Direct xAI endpoint, not OpenRouter — no server-side collision.
        tools = [_fn_tool("web_search")]
        out = _omit_openrouter_online_web_search(
            tools, "x-ai/grok-4.5:online", "https://api.x.ai/v1"
        )
        assert _tool_names(out) == ["web_search"]

    def test_non_xai_openrouter_online_keeps_web_search(self):
        tools = [_fn_tool("web_search")]
        out = _omit_openrouter_online_web_search(
            tools, "perplexity/sonar:online", OPENROUTER_BASE
        )
        assert _tool_names(out) == ["web_search"]

    def test_empty_and_none_tools_pass_through(self):
        assert _omit_openrouter_online_web_search(None, "x-ai/grok-4.5:online", OPENROUTER_BASE) is None
        assert _omit_openrouter_online_web_search([], "x-ai/grok-4.5:online", OPENROUTER_BASE) == []


# ── Integration through build_kwargs (the real OpenRouter profile path) ──────


@pytest.fixture
def transport():
    return ChatCompletionsTransport()


def _messages():
    return [{"role": "user", "content": "search the web"}]


class TestBuildKwargs:
    def test_online_request_omits_client_web_search(self, transport):
        kw = transport.build_kwargs(
            model="x-ai/grok-4.5:online",
            messages=_messages(),
            tools=[_fn_tool("web_search"), _fn_tool("web_extract")],
            provider_profile=get_provider_profile("openrouter"),
            base_url=OPENROUTER_BASE,
        )
        assert _tool_names(kw.get("tools")) == ["web_extract"]

    def test_bare_grok_request_keeps_client_web_search(self, transport):
        kw = transport.build_kwargs(
            model="x-ai/grok-4.5",
            messages=_messages(),
            tools=[_fn_tool("web_search"), _fn_tool("web_extract")],
            provider_profile=get_provider_profile("openrouter"),
            base_url=OPENROUTER_BASE,
        )
        assert _tool_names(kw.get("tools")) == ["web_search", "web_extract"]

"""Regression guard: fallback activation must rebuild the right Anthropic SDK client.

``agent.client_lifecycle._swap_fallback_clients`` performs the runtime client
swap behind ``try_activate_fallback`` / ``bind_route_entry``. Its
``anthropic_messages`` branch must dispatch Vertex-hosted Claude through
``build_anthropic_vertex_client`` (and populate ``_vertex_project_id`` /
``_vertex_region`` so ``_create_request_anthropic_client`` keys as ``"vertex"``
even when the session started on Gemini ``chat_completions``), matching
``agent_init``, ``_build_switched_client``, and ``_rebuild_anthropic_client``.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest

VERTEX_BASE = (
    "https://aiplatform.googleapis.com/v1/projects/proj-1"
    "/locations/global/publishers/anthropic"
)
BEDROCK_BASE = "https://bedrock-runtime.eu-central-1.amazonaws.com"


@pytest.fixture
def spies(monkeypatch):
    """Record which Anthropic client factory each code path calls."""
    calls = {"vertex": [], "direct": [], "bedrock": []}

    vertex_mod = types.ModuleType("agent.anthropic_vertex_adapter")

    def _build_vertex(project_id, region="global", timeout=None, **kw):
        calls["vertex"].append({"project_id": project_id, "region": region})
        c = MagicMock(name="AnthropicVertex")
        c.base_url = "https://aiplatform.googleapis.com/v1/"
        return c

    def _get_vertex_cfg(*a, **kw):
        return ("proj-1", "global")

    vertex_mod.build_anthropic_vertex_client = _build_vertex
    vertex_mod.get_anthropic_vertex_config = _get_vertex_cfg
    vertex_mod.is_anthropic_vertex_model = lambda m: str(m or "").startswith("anthropic/")
    monkeypatch.setitem(sys.modules, "agent.anthropic_vertex_adapter", vertex_mod)

    import agent.anthropic_adapter as direct_mod

    def _build_direct(api_key, base_url=None, timeout=None, **kw):
        calls["direct"].append({"api_key": api_key, "base_url": base_url})
        c = MagicMock(name="Anthropic")
        c.base_url = base_url
        return c

    def _build_bedrock(region, **kw):
        calls["bedrock"].append({"region": region})
        return MagicMock(name="AnthropicBedrock")

    monkeypatch.setattr(direct_mod, "build_anthropic_client", _build_direct)
    monkeypatch.setattr(
        direct_mod, "build_anthropic_bedrock_client", _build_bedrock, raising=False
    )
    import agent.anthropic_credentials as creds_mod

    monkeypatch.setattr(creds_mod, "resolve_anthropic_token", lambda **kw: "", raising=False)
    monkeypatch.setattr(creds_mod, "_is_oauth_token", lambda k: False, raising=False)
    return calls


def _agent(provider, base_url, model="google/gemini-4-argon", api_mode="chat_completions"):
    from run_agent import AIAgent

    a = object.__new__(AIAgent)
    a.model = model
    a.provider = provider
    a.requested_provider = provider
    a.api_mode = api_mode
    a.base_url = base_url
    a.api_key = "placeholder"
    a.client = MagicMock(name="openai-client")
    a._client_kwargs = {"api_key": "placeholder", "base_url": base_url}
    a._anthropic_client = None
    a._anthropic_api_key = ""
    a._anthropic_base_url = None
    a._is_anthropic_oauth = False
    a._config_context_length = None
    a._oauth_1m_beta_disabled = False
    a._credential_pool = None
    a._credential_pool_entry_id = None
    return a


def _fb_client(api_key="vertex-adc", base_url=VERTEX_BASE):
    c = MagicMock(name="aux-client")
    c.api_key = api_key
    c.base_url = base_url
    return c


def test_fallback_to_vertex_claude_uses_vertex_sdk_and_sets_attrs(spies):
    """Fallback from Gemini (chat_completions) to Claude on Vertex must use AnthropicVertex."""
    from agent.client_lifecycle import _swap_fallback_clients

    agent = _agent(
        "vertex",
        "https://aiplatform.googleapis.com/v1beta1/projects/proj-1/locations/global/endpoints/openapi",
    )
    agent.model = "claude-opus-5-5"
    agent.api_mode = "anthropic_messages"

    _swap_fallback_clients(
        agent, _fb_client(), "vertex", "claude-opus-5-5", VERTEX_BASE, "anthropic_messages"
    )

    assert spies["vertex"] == [{"project_id": "proj-1", "region": "global"}]
    assert spies["direct"] == []
    assert agent._vertex_project_id == "proj-1"
    assert agent._vertex_region == "global"
    assert agent.client is None

    # Request-local client must also dispatch to AnthropicVertex after the fallback swap.
    spies["vertex"].clear()
    key = agent._request_anthropic_client_key()
    assert key[:3] == ("vertex", "proj-1", "global")
    agent._build_anthropic_client_for_key(key)
    assert spies["vertex"] == [{"project_id": "proj-1", "region": "global"}]
    assert spies["direct"] == []


def test_fallback_to_bedrock_claude_uses_bedrock_sdk(spies):
    """Bedrock fallback must dispatch via AnthropicBedrock."""
    from agent.client_lifecycle import _swap_fallback_clients

    agent = _agent("bedrock", BEDROCK_BASE, model="claude-opus-5-5", api_mode="anthropic_messages")
    _swap_fallback_clients(
        agent,
        _fb_client(api_key="aws", base_url=BEDROCK_BASE),
        "bedrock",
        "claude-opus-5-5",
        BEDROCK_BASE,
        "anthropic_messages",
    )

    assert spies["bedrock"] == [{"region": "eu-central-1"}]
    assert spies["direct"] == []
    assert spies["vertex"] == []


def test_fallback_to_native_anthropic_uses_direct_sdk(spies):
    """Native Anthropic fallback must still use build_anthropic_client."""
    from agent.client_lifecycle import _swap_fallback_clients

    agent = _agent("vertex", VERTEX_BASE)
    agent.provider = "anthropic"
    agent.model = "claude-opus-5-5"
    agent.api_mode = "anthropic_messages"

    _swap_fallback_clients(
        agent,
        _fb_client(api_key="sk-ant-test", base_url="https://api.anthropic.com"),
        "anthropic",
        "claude-opus-5-5",
        "https://api.anthropic.com",
        "anthropic_messages",
    )

    assert spies["direct"] == [{"api_key": "sk-ant-test", "base_url": "https://api.anthropic.com"}]
    assert spies["vertex"] == []


def test_swap_fallback_clients_agrees_with_rebuild_anthropic_client(spies):
    """Cross-site invariant: _swap_fallback_clients and _rebuild_anthropic_client agree per provider."""
    from agent.client_lifecycle import _swap_fallback_clients

    for provider, base in (
        ("vertex", VERTEX_BASE),
        ("bedrock", BEDROCK_BASE),
        ("anthropic", "https://api.anthropic.com"),
    ):
        for k in spies:
            spies[k].clear()
        agent = _agent(provider, base, model="claude-opus-5-5", api_mode="anthropic_messages")
        _swap_fallback_clients(
            agent, _fb_client(api_key="tok", base_url=base), provider, "claude-opus-5-5", base, "anthropic_messages"
        )
        via_fallback = {k: len(v) > 0 for k, v in spies.items()}

        for k in spies:
            spies[k].clear()
        agent2 = _agent(provider, base, model="claude-opus-5-5", api_mode="anthropic_messages")
        agent2._vertex_project_id = "proj-1"
        agent2._vertex_region = "global"
        agent2._bedrock_region = "eu-central-1"
        agent2._rebuild_anthropic_client()
        via_rebuild = {k: len(v) > 0 for k, v in spies.items()}

        assert via_fallback == via_rebuild, (
            f"provider={provider!r}: _swap_fallback_clients and _rebuild_anthropic_client "
            f"disagree on SDK dispatch ({via_fallback} vs {via_rebuild})"
        )

"""Regression guard: ``/model`` must rebuild the right Anthropic SDK client.

``agent.agent_runtime_helpers.switch_model`` performs the runtime swap behind
the ``/model`` command (CLI, gateway and TUI all funnel into it). Its
``anthropic_messages`` branch rebuilt the client with
``build_anthropic_client()`` unconditionally, ignoring the provider — unlike
``agent_init``, ``_rebuild_anthropic_client`` and
``_create_request_anthropic_client``, which all dispatch on it.

Bedrock- and Vertex-hosted Claude speak the Anthropic Messages protocol but
authenticate through their cloud's own SDK, with a base_url that has no
``/v1/messages`` route. So a ``/model`` switch on either provider replaced a
working cloud client with a direct Anthropic one and every subsequent call
failed — on Vertex, HTTP 404 against
``…/publishers/anthropic/v1/messages``.

The tests assert the cross-site invariant (all rebuild paths agree on which
SDK a provider gets) rather than snapshotting any one implementation.
"""

import sys
import types
from unittest.mock import MagicMock, patch

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
    monkeypatch.setattr(direct_mod, "resolve_anthropic_token", lambda: "", raising=False)
    monkeypatch.setattr(direct_mod, "_is_oauth_token", lambda k: False, raising=False)
    return calls


def _agent(provider, base_url, model="claude-opus-4-8"):
    """Bare AIAgent carrying only what the swap path reads.

    switch_model explicitly supports this shape — see its ``_MISSING``
    sentinel comment about tests constructing agents via ``__new__``.
    """
    from run_agent import AIAgent

    a = object.__new__(AIAgent)
    a.model = model
    a.provider = provider
    a.requested_provider = provider
    a.api_mode = "anthropic_messages"
    a.base_url = base_url
    a.api_key = "placeholder"
    a.client = None
    a._client_kwargs = {}
    a._anthropic_client = MagicMock(name="pre-existing client")
    a._anthropic_api_key = "placeholder"
    a._anthropic_base_url = base_url
    a._is_anthropic_oauth = False
    a._config_context_length = None
    a._oauth_1m_beta_disabled = False
    a._credential_pool = None
    a._credential_pool_entry_id = None
    a._vertex_project_id = "proj-1"
    a._vertex_region = "global"
    a._bedrock_region = "eu-central-1"
    a._primary_runtime = {}
    a._fallback_activated = False
    a._fallback_index = 0
    return a


def _switch(agent, new_model, new_provider, base_url):
    """Drive switch_model far enough to observe the client rebuild.

    Everything after the rebuild (context-length probing, cache policy,
    compressor refresh) is unrelated to provider dispatch; stub it so a bare
    agent can get through. If the tail still raises we swallow it — the
    dispatch has already happened and the spies have recorded it.
    """
    from agent import agent_runtime_helpers as arh
    from run_agent import AIAgent

    with (
        patch.object(AIAgent, "_ensure_lmstudio_runtime_loaded", lambda self, *a, **k: None, create=True),
        patch.object(AIAgent, "_lmstudio_load_was_unverified", lambda self, *a, **k: False, create=True),
        patch.object(AIAgent, "_effective_lmstudio_context_length", lambda self, *a, **k: None, create=True),
        patch.object(AIAgent, "_anthropic_prompt_cache_policy", lambda self, *a, **k: (False, False), create=True),
        patch.object(AIAgent, "_apply_client_headers_for_base_url", lambda self, *a, **k: None, create=True),
        patch.object(AIAgent, "_create_openai_client", lambda self, *a, **k: MagicMock(), create=True),
        patch("agent.credential_pool.load_pool", lambda *a, **k: None, create=True),
    ):
        try:
            arh.switch_model(
                agent, new_model, new_provider,
                api_key="", base_url=base_url, api_mode="anthropic_messages",
            )
        except Exception:
            # Post-rebuild tail is out of scope for these assertions.
            pass


def test_switch_to_vertex_claude_uses_vertex_sdk(spies):
    """/model to another Claude on Vertex must rebuild via AnthropicVertex."""
    agent = _agent("vertex", VERTEX_BASE)
    _switch(agent, "claude-sonnet-4-5", "vertex", VERTEX_BASE)

    assert spies["vertex"], (
        "regression: /model switch on Claude-on-Vertex rebuilt via the direct "
        f"Anthropic adapter ({spies['direct']}) — every later call would 404 "
        f"against {VERTEX_BASE}/v1/messages"
    )
    assert spies["direct"] == []
    assert spies["vertex"][0] == {"project_id": "proj-1", "region": "global"}


def test_switch_to_bedrock_claude_uses_bedrock_sdk(spies):
    """Same bug class: Bedrock must rebuild via AnthropicBedrock."""
    agent = _agent("bedrock", BEDROCK_BASE)
    _switch(agent, "claude-sonnet-4-5", "bedrock", BEDROCK_BASE)

    assert spies["bedrock"], f"expected Bedrock SDK, got direct={spies['direct']}"
    assert spies["direct"] == []
    # Region comes from the endpoint being switched TO, not the stale attr.
    assert spies["bedrock"][0]["region"] == "eu-central-1"


def test_switch_to_native_anthropic_still_uses_direct_sdk(spies):
    """Guard the other direction — native Anthropic must NOT regress."""
    agent = _agent("anthropic", "https://api.anthropic.com")
    _switch(agent, "claude-opus-4-8", "anthropic", "https://api.anthropic.com")

    assert spies["direct"], "native Anthropic must use the direct adapter"
    assert spies["vertex"] == []
    assert spies["bedrock"] == []


def test_no_client_is_built_against_the_vertex_publisher_base_url(spies):
    """The publisher URL is display-only; no direct client may target it."""
    agent = _agent("vertex", VERTEX_BASE)
    _switch(agent, "claude-sonnet-4-5", "vertex", VERTEX_BASE)

    assert not [c for c in spies["direct"] if c["base_url"] == VERTEX_BASE]


def test_switching_into_vertex_from_another_provider_resolves_config(spies):
    """Switching INTO vertex must not depend on attrs stashed at init.

    A session that started on native Anthropic never sets
    ``_vertex_project_id`` / ``_vertex_region``, so the branch has to fall
    back to the shared vertex config chain.
    """
    agent = _agent("anthropic", "https://api.anthropic.com")
    del agent._vertex_project_id
    del agent._vertex_region

    _switch(agent, "claude-sonnet-4-5", "vertex", VERTEX_BASE)

    assert spies["vertex"] == [{"project_id": "proj-1", "region": "global"}]
    assert spies["direct"] == []


def test_switch_model_agrees_with_rebuild_anthropic_client(spies):
    """The invariant that broke: rebuild paths must agree per provider."""
    for provider, base in (
        ("vertex", VERTEX_BASE),
        ("bedrock", BEDROCK_BASE),
        ("anthropic", "https://api.anthropic.com"),
    ):
        for key in spies:
            spies[key].clear()
        agent = _agent(provider, base)
        _switch(agent, "claude-sonnet-4-5", provider, base)
        via_switch = {k: len(v) > 0 for k, v in spies.items()}

        for key in spies:
            spies[key].clear()
        agent2 = _agent(provider, base)
        agent2._rebuild_anthropic_client()
        via_rebuild = {k: len(v) > 0 for k, v in spies.items()}

        assert via_switch == via_rebuild, (
            f"provider={provider!r}: switch_model and "
            f"_rebuild_anthropic_client disagree on SDK dispatch "
            f"({via_switch} vs {via_rebuild})"
        )

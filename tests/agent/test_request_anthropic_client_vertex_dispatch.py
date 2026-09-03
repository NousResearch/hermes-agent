"""Regression guard: the request-local Anthropic client must honour the
Claude-on-Vertex provider dispatch.

Upstream #67142 introduced ``AIAgent._create_request_anthropic_client`` — a
per-request client that carries every in-flight ``anthropic_messages`` call
(see the two call sites in ``agent/chat_completion_helpers.py``). It was
written to mirror ``_rebuild_anthropic_client``, but only reproduced the
direct-Anthropic and Bedrock branches. On a Claude-on-Vertex deployment the
missing branch made every turn fall through to a direct Anthropic client
constructed with the *display-only* Vertex publisher base_url, so the SDK
POSTed to::

    …/publishers/anthropic/v1/messages          (HTTP 404)

instead of the real Vertex publisher route::

    …/publishers/anthropic/models/<model>:rawPredict

The shared ``_anthropic_client`` was correctly an ``AnthropicVertex``, which
is why agent init and every auxiliary task looked healthy while the main
conversation loop 404'd on the first call.

These tests assert the *invariant* — that the request-local builder and the
shared rebuild agree on which SDK a given provider gets — rather than
snapshotting either implementation.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def agent():
    """A minimal AIAgent stand-in carrying only the attributes the two client
    builders read. Avoids the full __init__ (network/config/plugins)."""
    from run_agent import AIAgent

    a = object.__new__(AIAgent)
    a.api_mode = "anthropic_messages"
    a.provider = "vertex"
    a.model = "claude-opus-4-8"
    a._anthropic_api_key = "vertex-adc"
    a._anthropic_base_url = (
        "https://aiplatform.googleapis.com/v1/projects/proj-1"
        "/locations/global/publishers/anthropic"
    )
    a._vertex_project_id = "proj-1"
    a._vertex_region = "global"
    a._oauth_1m_beta_disabled = False
    a._anthropic_client = None
    return a


@pytest.fixture
def spies(monkeypatch):
    """Patch both adapter factories and record which one gets called."""
    calls = {"vertex": [], "direct": [], "bedrock": []}

    vertex_mod = types.ModuleType("agent.anthropic_vertex_adapter")

    def _build_vertex(project_id, region="global", timeout=None, **kw):
        calls["vertex"].append({"project_id": project_id, "region": region})
        c = MagicMock(name="AnthropicVertex")
        c.base_url = "https://aiplatform.googleapis.com/v1/"
        return c

    vertex_mod.build_anthropic_vertex_client = _build_vertex

    import agent.anthropic_adapter as direct_mod

    def _build_direct(api_key, base_url=None, timeout=None, **kw):
        calls["direct"].append({"api_key": api_key, "base_url": base_url})
        c = MagicMock(name="Anthropic")
        c.base_url = base_url
        return c

    def _build_bedrock(region, **kw):
        calls["bedrock"].append({"region": region})
        return MagicMock(name="AnthropicBedrock")

    monkeypatch.setitem(sys.modules, "agent.anthropic_vertex_adapter", vertex_mod)
    monkeypatch.setattr(direct_mod, "build_anthropic_client", _build_direct)
    monkeypatch.setattr(
        direct_mod, "build_anthropic_bedrock_client", _build_bedrock, raising=False
    )
    return calls


def _make_request_client(agent):
    with patch.object(
        type(agent), "_try_refresh_anthropic_client_credentials", lambda self: None
    ):
        return agent._create_request_anthropic_client(reason="test")


def test_request_client_uses_vertex_sdk_for_vertex_provider(agent, spies):
    """provider=vertex must build via the AnthropicVertex adapter."""
    _make_request_client(agent)

    assert spies["vertex"] == [{"project_id": "proj-1", "region": "global"}]
    assert spies["direct"] == [], (
        "regression: request-local client fell through to the direct Anthropic "
        "adapter on a Claude-on-Vertex agent — every call would 404 against "
        f"{agent._anthropic_base_url}/v1/messages"
    )


def test_request_client_never_targets_the_display_only_base_url(agent, spies):
    """The publisher base_url is display-only; no client may be built on it."""
    _make_request_client(agent)

    built_on_display_url = [
        c for c in spies["direct"] if c["base_url"] == agent._anthropic_base_url
    ]
    assert not built_on_display_url


def test_request_client_and_rebuild_agree_on_provider_dispatch(agent, spies):
    """The invariant: both builders must pick the same SDK for a provider.

    This is what actually broke — a provider special-cased in
    ``_rebuild_anthropic_client`` but not in the request-local builder.
    """
    for provider in ("vertex", "anthropic", "bedrock"):
        agent.provider = provider
        if provider == "bedrock":
            agent._bedrock_region = "us-east-1"

        for key in spies:
            spies[key].clear()
        _make_request_client(agent)
        request_path = {k: len(v) for k, v in spies.items()}

        for key in spies:
            spies[key].clear()
        agent._rebuild_anthropic_client()
        rebuild_path = {k: len(v) for k, v in spies.items()}

        assert request_path == rebuild_path, (
            f"provider={provider!r}: request-local builder and "
            f"_rebuild_anthropic_client disagree on SDK dispatch "
            f"({request_path} vs {rebuild_path})"
        )


def test_vertex_region_falls_back_to_global(agent, spies):
    """A missing/empty region must not reach the SDK as None."""
    agent._vertex_region = None
    _make_request_client(agent)
    assert spies["vertex"][0]["region"] == "global"


# ── Cache-key discrimination ───────────────────────────────────────────────
#
# Upstream added a single-slot cache in front of the request-local builder
# and moved the provider dispatch onto ``key[0]`` from
# ``_request_anthropic_client_key()``. That makes the key function part of
# the dispatch: a provider missing a branch there keys as ``"direct"`` and
# can never reach its own branch in the builder, no matter that the branch
# exists. These tests pin that contract.


def test_vertex_keys_distinctly_from_direct(agent):
    """provider=vertex must not share the ``"direct"`` cache bucket.

    If it does, the ``elif key[0] == "vertex"`` branch in
    ``_create_request_anthropic_client`` is unreachable and every turn 404s
    against the display-only publisher base_url.
    """
    assert agent._request_anthropic_client_key()[0] == "vertex"

    agent.provider = "anthropic"
    assert agent._request_anthropic_client_key()[0] == "direct"


def test_vertex_key_covers_project_region_and_timeout(agent):
    """Anything baked into the Vertex client must invalidate the cache slot.

    ``build_anthropic_vertex_client`` bakes project, region and timeout into
    the client, so a change to any of them must produce a different key —
    otherwise a warm slot hands back a client pointed at the wrong publisher
    route (or carrying a stale ``/model`` timeout).
    """
    base = agent._request_anthropic_client_key()

    agent._vertex_project_id = "proj-2"
    assert agent._request_anthropic_client_key() != base

    agent._vertex_project_id = "proj-1"
    agent._vertex_region = "us-east5"
    assert agent._request_anthropic_client_key() != base

    agent._vertex_region = "global"
    assert agent._request_anthropic_client_key() == base
    with patch(
        "run_agent.get_provider_request_timeout", lambda *a, **kw: 999999
    ):
        assert agent._request_anthropic_client_key() != base

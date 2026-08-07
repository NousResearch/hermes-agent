"""Regression guard: ``_replace_primary_openai_client`` must succeed for
``minimax-oauth`` (and other providers that go through the OAuth credential
resolver) without raising ``The api_key client option must be set``.

When the conversation-loop's stale-stream detector kills a stuck HTTP
connection, ``run_agent._replace_primary_openai_client`` is invoked to
rebuild the OpenAI client with the *current* credentials. For API-key
providers, ``self._client_kwargs["api_key"]`` is populated at startup and
stays valid. For OAuth providers like ``minimax-oauth``, the bearer token
is held in a credential pool / singleton resolver — NOT in
``self._client_kwargs``. The previous client was built at startup with
whatever the resolver returned at that moment, but on a rebuild the code
re-reads ``self._client_kwargs`` which is empty for the api_key slot,
so ``OpenAI(api_key=None, ...)`` raises:

    openai.OpenAIError: The api_key client option must be set either by
    passing api_key to the client or by setting the OPENAI_API_KEY
    environment variable

Symptom: stream goes stale → 300s wait → cleanup thread tries to rebuild
→ "Failed to rebuild shared OpenAI client (stale_stream_pool_cleanup)"
warning → next user message sees "Interrupted during API call" with
hours of wall-clock wasted (visible in agent.log as
``Stream stale for 300s ... Killing connection`` followed by
``Failed to rebuild shared OpenAI client ... The api_key client option
must be set``).

Fix: ``_replace_primary_openai_client`` must re-resolve credentials via
the appropriate provider-specific resolver BEFORE building the new
client, mirroring the pattern already used by
``_try_refresh_xai_oauth_client_credentials`` and
``_try_refresh_nous_client_credentials``.
"""

from types import SimpleNamespace
from unittest.mock import patch

from run_agent import AIAgent


def _make_agent(provider: str = "minimax-oauth"):
    return AIAgent(
        api_key="initial-bearer-from-startup",
        base_url="https://api.minimax.io/v1",
        model="MiniMax-M3",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        provider=provider,
    )


def _make_fake_openai_factory(constructed):
    class _FakeOpenAI:
        def __init__(self, **kwargs):
            self._kwargs = kwargs
            self._closed = False
            constructed.append(self)

        def close(self):
            self._closed = True

    return _FakeOpenAI


def test_replace_primary_openai_client_resolves_minimax_oauth_credentials():
    """A minimax-oauth agent whose ``_client_kwargs["api_key"]`` is empty
    (because the bearer token lived in the resolver at startup) must
    still be able to rebuild its client on stale-stream recovery.

    The fix: ``_replace_primary_openai_client`` re-calls
    ``resolve_minimax_oauth_runtime_credentials`` so the rebuilt client
    gets a fresh token. We don't care about the token value — just that
    the rebuild doesn't raise and that ``OpenAI`` is constructed with a
    non-empty api_key.
    """
    agent = _make_agent(provider="minimax-oauth")
    constructed = []
    fake_openai = _make_fake_openai_factory(constructed)

    # Simulate the post-startup state: original kwargs no longer carry
    # the live bearer (it was returned by the resolver at startup, not
    # cached here). This is exactly the situation the old code hit.
    agent._client_kwargs = {
        "base_url": "https://api.minimax.io/v1",
        # NO "api_key" — the live token lives in the resolver pool
    }

    fake_creds = {
        "api_key": "freshly-resolved-minimax-bearer",
        "base_url": "https://api.minimax.io/v1",
    }

    with patch("run_agent.OpenAI", fake_openai), \
         patch(
             "hermes_cli.auth.resolve_minimax_oauth_runtime_credentials",
             return_value=fake_creds,
         ):
        # First, seed an initial client so _replace has something to
        # tear down (mirrors test_create_openai_client_reuse pattern).
        agent.client = agent._create_openai_client(
            {"api_key": "initial-bearer-from-startup",
             "base_url": "https://api.minimax.io/v1"},
            reason="seed",
            shared=True,
        )

        # Now the actual scenario: rebuild after stale stream kill.
        ok = agent._replace_primary_openai_client(reason="stale_stream_pool_cleanup")

    assert ok is True, (
        "_replace_primary_openai_client returned False for minimax-oauth — "
        "this is the exact failure that produced hours of 'waiting for model "
        "response' freezes in production (agent.log: 'Failed to rebuild "
        "shared OpenAI client ... The api_key client option must be set')."
    )
    assert len(constructed) >= 2, "expected at least 2 OpenAI constructions (seed + rebuild)"

    # The rebuild MUST have produced a client with a real api_key, not
    # the literal value the OpenAI SDK rejects.
    rebuilt = constructed[-1]
    rebuilt_key = rebuilt._kwargs.get("api_key")
    assert rebuilt_key, (
        f"rebuilt client has no api_key (kwargs={rebuilt._kwargs!r}); "
        f"the SDK would have raised 'The api_key client option must be set'."
    )
    assert rebuilt_key != "no-key-required", (
        "rebuilt client was constructed with the placeholder "
        "'no-key-required' fallback, which would 401 on real requests."
    )


def test_replace_primary_openai_client_keeps_existing_api_key_when_resolver_unavailable():
    """When the credential resolver itself raises (transient OAuth portal
    outage, missing token, etc.), the rebuild must NOT silently leave the
    agent with a broken client — but it also must not crash the cleanup
    thread. Acceptable behavior: fall back to whatever api_key is still
    in ``self._client_kwargs`` (which the agent already proved works at
    startup), and warn loudly. Previously the fallback was to raise,
    which killed the cleanup thread and left the user with a stuck
    'waiting for model response' state for the rest of the session.
    """
    agent = _make_agent(provider="minimax-oauth")
    constructed = []
    fake_openai = _make_fake_openai_factory(constructed)

    # Realistic post-startup state: kwargs still carry the bearer from
    # the last successful resolve.
    agent._client_kwargs = {
        "api_key": "still-cached-bearer",
        "base_url": "https://api.minimax.io/v1",
    }

    def _resolver_raises(*a, **kw):
        raise RuntimeError("OAuth portal is down (simulated)")

    with patch("run_agent.OpenAI", fake_openai), \
         patch(
             "hermes_cli.auth.resolve_minimax_oauth_runtime_credentials",
             side_effect=_resolver_raises,
         ):
        agent.client = agent._create_openai_client(
            dict(agent._client_kwargs),
            reason="seed",
            shared=True,
        )
        ok = agent._replace_primary_openai_client(reason="stale_stream_pool_cleanup")

    assert ok is True, (
        "rebuild returned False after a resolver failure — the cleanup "
        "thread is now broken and the next chat will see 'Interrupted "
        "during API call' for the rest of the session."
    )
    assert len(constructed) >= 2
    rebuilt = constructed[-1]
    # We accept the cached bearer as a degraded-but-functional fallback
    # rather than crashing. Better a stale token for one request than a
    # dead client for the whole session.
    assert rebuilt._kwargs.get("api_key") == "still-cached-bearer", (
        f"expected fallback to cached api_key, got {rebuilt._kwargs.get('api_key')!r}"
    )


def test_replace_primary_openai_client_also_refreshes_xai_oauth_credentials():
    """The new ``_refresh_kwargs_for_oauth_provider`` dispatcher is
    deliberately generic (covers ALL OAuth providers, not just minimax).
    This test pins the behaviour for xai-oauth so a future refactor that
    accidentally narrows the dispatch back to minimax-only is caught.

    Pattern mirrors the minimax test above: empty post-startup kwargs,
    patch the xai resolver, assert the rebuilt client has a real api_key.
    """
    agent = AIAgent(
        api_key="initial-xai-bearer",
        base_url="https://api.x.ai/v1",
        model="grok-3",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        provider="xai-oauth",
    )
    constructed = []
    fake_openai = _make_fake_openai_factory(constructed)

    agent._client_kwargs = {
        "base_url": "https://api.x.ai/v1",
        # NO "api_key" — the live bearer lives in the resolver pool
    }

    fake_creds = {
        "api_key": "freshly-resolved-xai-bearer",
        "base_url": "https://api.x.ai/v1",
    }

    with patch("run_agent.OpenAI", fake_openai), \
         patch(
             "hermes_cli.auth.resolve_xai_oauth_runtime_credentials",
             return_value=fake_creds,
         ):
        agent.client = agent._create_openai_client(
            {"api_key": "initial-xai-bearer", "base_url": "https://api.x.ai/v1"},
            reason="seed",
            shared=True,
        )
        ok = agent._replace_primary_openai_client(reason="stale_stream_pool_cleanup")

    assert ok is True, (
        "_replace_primary_openai_client returned False for xai-oauth — "
        "the generic dispatcher should refresh xai-oauth the same way it "
        "refreshes minimax-oauth."
    )
    rebuilt = constructed[-1]
    assert rebuilt._kwargs.get("api_key") == "freshly-resolved-xai-bearer", (
        f"xai-oauth rebuild did not pick up the resolver's fresh bearer; "
        f"got api_key={rebuilt._kwargs.get('api_key')!r}"
    )


def test_replace_primary_openai_client_noop_for_api_key_providers():
    """The dispatcher must not touch ``self._client_kwargs`` for providers
    that aren't in the oauth_resolvers map (i.e. plain API-key providers
    like openrouter, anthropic, openai). For those, the existing kwargs
    are still valid and the refresh is a no-op.
    """
    agent = AIAgent(
        api_key="sk-static-test-key",
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-3.5-sonnet",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        provider="openrouter",
    )
    constructed = []
    fake_openai = _make_fake_openai_factory(constructed)

    agent._client_kwargs = {
        "api_key": "sk-static-test-key",
        "base_url": "https://openrouter.ai/api/v1",
    }

    with patch("run_agent.OpenAI", fake_openai):
        agent.client = agent._create_openai_client(
            dict(agent._client_kwargs),
            reason="seed",
            shared=True,
        )
        ok = agent._replace_primary_openai_client(reason="stale_stream_pool_cleanup")

    assert ok is True
    rebuilt = constructed[-1]
    assert rebuilt._kwargs.get("api_key") == "sk-static-test-key", (
        f"api-key provider rebuild changed the api_key unexpectedly: "
        f"{rebuilt._kwargs.get('api_key')!r}"
    )

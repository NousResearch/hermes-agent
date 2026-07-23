"""Recovery/restore/switch paths must rebuild Claude-on-Vertex primaries
through the provider-aware chokepoint.

Regression tests for the PR-review finding that transient-transport
recovery and fallback restoration rebuilt every ``anthropic_messages``
primary with ``build_anthropic_client()`` — silently pointing a Vertex
(or Bedrock) Claude session at api.anthropic.com with the placeholder
key, where every subsequent request 401s. All client construction is
monkeypatched at the chokepoint; no SDKs or network involved.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ── the chokepoint itself ────────────────────────────────────────────────────


class TestBuildAnthropicClientForProvider:
    def test_vertex_dispatches_to_anthropic_vertex(self, monkeypatch):
        import agent.anthropic_adapter as aa
        import agent.vertex_adapter as va

        creds = object()
        monkeypatch.setattr(
            va, "get_vertex_anthropic_config", lambda *a, **k: (creds, "proj-1", "global")
        )
        built = {}
        monkeypatch.setattr(
            aa, "build_anthropic_vertex_client",
            lambda project, region, credentials=None: built.update(
                project=project, region=region, credentials=credentials
            ) or "VERTEX_CLIENT",
        )
        client = aa.build_anthropic_client_for_provider("vertex", "vertex-oauth", None)
        assert client == "VERTEX_CLIENT"
        assert built == {"project": "proj-1", "region": "global", "credentials": creds}

    def test_vertex_falls_back_to_agent_cached_attrs(self, monkeypatch):
        import agent.anthropic_adapter as aa
        import agent.vertex_adapter as va

        monkeypatch.setattr(
            va, "get_vertex_anthropic_config", lambda *a, **k: (None, None, None)
        )
        built = {}
        monkeypatch.setattr(
            aa, "build_anthropic_vertex_client",
            lambda project, region, credentials=None: built.update(
                project=project, region=region, credentials=credentials
            ) or "VERTEX_CLIENT",
        )
        cached_creds = object()
        agent_stub = MagicMock()
        agent_stub._vertex_project_id = "cached-proj"
        agent_stub._vertex_region = "us-east5"
        agent_stub._vertex_credentials = cached_creds

        client = aa.build_anthropic_client_for_provider(
            "vertex", "vertex-oauth", None, agent=agent_stub
        )
        assert client == "VERTEX_CLIENT"
        assert built == {
            "project": "cached-proj", "region": "us-east5", "credentials": cached_creds,
        }

    def test_vertex_refreshes_agent_caches(self, monkeypatch):
        import agent.anthropic_adapter as aa
        import agent.vertex_adapter as va

        creds = object()
        monkeypatch.setattr(
            va, "get_vertex_anthropic_config", lambda *a, **k: (creds, "proj-2", "europe-west1")
        )
        monkeypatch.setattr(
            aa, "build_anthropic_vertex_client", lambda *a, **k: "VERTEX_CLIENT"
        )
        agent_stub = MagicMock()
        aa.build_anthropic_client_for_provider("vertex", "vertex-oauth", None, agent=agent_stub)
        assert agent_stub._vertex_project_id == "proj-2"
        assert agent_stub._vertex_region == "europe-west1"
        assert agent_stub._vertex_credentials is creds

    def test_bedrock_dispatches_with_agent_region(self, monkeypatch):
        import agent.anthropic_adapter as aa

        monkeypatch.setattr(
            aa, "build_anthropic_bedrock_client", lambda region: f"BEDROCK:{region}"
        )
        agent_stub = MagicMock()
        agent_stub._bedrock_region = "eu-west-3"
        client = aa.build_anthropic_client_for_provider(
            "bedrock", "aws-sdk", None, agent=agent_stub
        )
        assert client == "BEDROCK:eu-west-3"

    def test_bedrock_parses_region_from_base_url(self, monkeypatch):
        import agent.anthropic_adapter as aa

        monkeypatch.setattr(
            aa, "build_anthropic_bedrock_client", lambda region: f"BEDROCK:{region}"
        )
        client = aa.build_anthropic_client_for_provider(
            "bedrock", "aws-sdk", "https://bedrock-runtime.ap-southeast-2.amazonaws.com"
        )
        assert client == "BEDROCK:ap-southeast-2"

    def test_other_providers_use_plain_client(self, monkeypatch):
        import agent.anthropic_adapter as aa

        seen = {}
        monkeypatch.setattr(
            aa, "build_anthropic_client",
            lambda api_key, base_url, timeout=None, drop_context_1m_beta=False: seen.update(
                api_key=api_key, base_url=base_url, timeout=timeout, drop=drop_context_1m_beta
            ) or "PLAIN_CLIENT",
        )
        client = aa.build_anthropic_client_for_provider(
            "anthropic", "sk-ant-x", "https://api.anthropic.com",
            timeout=42.0, drop_context_1m_beta=True,
        )
        assert client == "PLAIN_CLIENT"
        assert seen == {
            "api_key": "sk-ant-x", "base_url": "https://api.anthropic.com",
            "timeout": 42.0, "drop": True,
        }


# ── recovery / restore / switch paths route through the chokepoint ──────────


def _patch_chokepoint(monkeypatch):
    """Record chokepoint invocations; return the recorded-calls list."""
    import agent.anthropic_adapter as aa

    calls = []

    def _fake(provider, api_key, base_url, *, timeout=None,
              drop_context_1m_beta=False, agent=None):
        calls.append({"provider": provider, "api_key": api_key})
        return f"CLIENT_FOR:{provider}"

    monkeypatch.setattr(aa, "build_anthropic_client_for_provider", _fake)
    return calls


class _StubAgent:
    """Minimal agent shape for the runtime-helper paths under test."""

    def __init__(self, provider="vertex"):
        self.log_prefix = ""
        self.quiet_mode = True
        self.model = "claude-fable-5"
        self.provider = provider
        self.base_url = "https://aiplatform.googleapis.com/v1"
        self.api_mode = "anthropic_messages"
        self.api_key = "vertex-oauth"
        self.client = None
        self._client_kwargs = {}
        self._transport_cache = {}
        self._fallback_activated = False
        self._credential_pool = None
        self._config_context_length = None
        self._anthropic_client = None
        self._anthropic_api_key = "vertex-oauth"
        self._anthropic_base_url = None
        self._is_anthropic_oauth = False
        self._use_prompt_caching = False
        self._use_native_cache_layout = False
        self.context_compressor = MagicMock()
        self._primary_runtime = {
            "model": self.model,
            "provider": provider,
            "base_url": self.base_url,
            "api_mode": "anthropic_messages",
            "api_key": "vertex-oauth",
            "client_kwargs": {},
            "anthropic_api_key": "vertex-oauth",
            "anthropic_base_url": None,
            "is_anthropic_oauth": False,
            "use_prompt_caching": False,
            "use_native_cache_layout": False,
            "compressor_model": "m",
            "compressor_context_length": 100000,
            "compressor_base_url": "",
            "compressor_api_key": "",
            "compressor_provider": provider,
        }

    def _is_openrouter_url(self):
        return False

    def _close_openai_client(self, *a, **k):
        pass

    def _create_openai_client(self, *a, **k):
        return MagicMock()

    def _vprint(self, *a, **k):
        pass

    def _anthropic_prompt_cache_policy(self, **k):
        return False, False

    def _ensure_lmstudio_runtime_loaded(self):
        pass


def test_transient_recovery_rebuilds_vertex_via_chokepoint(monkeypatch):
    import agent.agent_runtime_helpers as helpers

    calls = _patch_chokepoint(monkeypatch)
    monkeypatch.setattr(helpers.time, "sleep", lambda *_: None)

    agent_stub = _StubAgent(provider="vertex")
    err = type("APIConnectionError", (Exception,), {})()
    ok = helpers.try_recover_primary_transport(
        agent_stub, err, retry_count=3, max_retries=3
    )
    assert ok is True
    assert agent_stub._anthropic_client == "CLIENT_FOR:vertex"
    assert calls and calls[0]["provider"] == "vertex"


def test_transient_recovery_keeps_plain_path_for_anthropic(monkeypatch):
    import agent.agent_runtime_helpers as helpers

    calls = _patch_chokepoint(monkeypatch)
    monkeypatch.setattr(helpers.time, "sleep", lambda *_: None)

    agent_stub = _StubAgent(provider="anthropic")
    agent_stub._primary_runtime["anthropic_api_key"] = "sk-ant-y"
    err = type("ReadTimeout", (Exception,), {})()
    ok = helpers.try_recover_primary_transport(
        agent_stub, err, retry_count=3, max_retries=3
    )
    assert ok is True
    assert calls and calls[0]["provider"] == "anthropic"
    assert calls[0]["api_key"] == "sk-ant-y"


def test_restore_primary_runtime_rebuilds_vertex_via_chokepoint(monkeypatch):
    import agent.agent_runtime_helpers as helpers

    calls = _patch_chokepoint(monkeypatch)

    agent_stub = _StubAgent(provider="vertex")
    agent_stub._fallback_activated = True  # restoring FROM a fallback
    helpers.restore_primary_runtime(agent_stub)
    assert agent_stub._anthropic_client == "CLIENT_FOR:vertex"
    assert calls and calls[0]["provider"] == "vertex"


def test_switch_model_to_vertex_claude_uses_chokepoint(monkeypatch):
    import agent.agent_runtime_helpers as helpers

    calls = _patch_chokepoint(monkeypatch)

    agent_stub = _StubAgent(provider="anthropic")
    helpers.switch_model(
        agent_stub, "claude-fable-5", "vertex",
        base_url="https://aiplatform.googleapis.com/v1",
        api_mode="anthropic_messages",
    )
    assert agent_stub._anthropic_client == "CLIENT_FOR:vertex"
    assert calls and calls[0]["provider"] == "vertex"
    # SDK-auth placeholder key pinned (matches agent_init) — a stale key from
    # the previous provider must not leak into the vertex runtime.
    assert agent_stub._anthropic_api_key == "vertex-oauth"
    assert agent_stub.api_key == "vertex-oauth"


def test_rebuild_anthropic_client_delegates(monkeypatch):
    """AIAgent._rebuild_anthropic_client routes through the chokepoint."""
    import run_agent as ra

    calls = _patch_chokepoint(monkeypatch)

    stub = _StubAgent(provider="vertex")
    ra.AIAgent._rebuild_anthropic_client(stub)
    assert stub._anthropic_client == "CLIENT_FOR:vertex"
    assert calls and calls[0]["provider"] == "vertex"


# ── model-aware api_mode determination (the desktop-session 404) ─────────────
#
# determine_api_mode("vertex") is provider-level chat_completions; the desktop
# model-set flow persisted that and clobbered the model-aware resolution,
# building an OpenAI client against the Anthropic base_url → HTTP 404 on a
# brand-new session. Claude on vertex must resolve anthropic_messages even
# when only provider+model are known.

@pytest.mark.parametrize("model,expected", [
    ("claude-fable-5", "anthropic_messages"),
    ("anthropic/claude-fable-5", "anthropic_messages"),
    ("claude-sonnet-5", "anthropic_messages"),
    ("google/gemini-3.1-pro-preview", "chat_completions"),
    ("moonshotai/kimi-k2-thinking-maas", "chat_completions"),
    ("", "chat_completions"),  # no model → provider-level default
])
def test_determine_api_mode_vertex_is_model_aware(model, expected):
    from hermes_cli.providers import determine_api_mode

    assert determine_api_mode("vertex", model=model) == expected


def test_determine_api_mode_vertex_alias_is_model_aware():
    from hermes_cli.providers import determine_api_mode

    assert (
        determine_api_mode("google-vertex", model="claude-fable-5")
        == "anthropic_messages"
    )


def test_switch_model_vertex_claude_autodetects_anthropic_mode(monkeypatch):
    """The live failure: switch to vertex claude with api_mode unset must not
    fall back to the chat_completions OpenAI client."""
    import agent.agent_runtime_helpers as helpers

    calls = _patch_chokepoint(monkeypatch)

    agent_stub = _StubAgent(provider="anthropic")
    helpers.switch_model(
        agent_stub, "claude-fable-5", "vertex",
        base_url="https://aiplatform.googleapis.com/v1",
        api_mode="",  # ← unset: must be determined model-aware
    )
    assert agent_stub.api_mode == "anthropic_messages"
    assert agent_stub._anthropic_client == "CLIENT_FOR:vertex"
    assert calls and calls[0]["provider"] == "vertex"


def test_request_anthropic_client_routes_vertex(monkeypatch):
    """The per-request streaming client (stale/interrupt watchdog ownership
    contract) must be provider-aware: this was the live 404 — every streamed
    request on a vertex Claude session built a plain Anthropic client and
    POSTed {aiplatform_host}/v1/messages (Google HTML 404), including the
    very first call of a fresh session."""
    import run_agent as ra

    calls = _patch_chokepoint(monkeypatch)

    stub = _StubAgent(provider="vertex")
    stub._try_refresh_anthropic_client_credentials = lambda: False
    client = ra.AIAgent._create_request_anthropic_client(
        stub, reason="chat_completion_stream_request"
    )
    assert client == "CLIENT_FOR:vertex"
    assert calls and calls[0]["provider"] == "vertex"


def test_request_anthropic_client_keeps_plain_for_anthropic(monkeypatch):
    import run_agent as ra

    calls = _patch_chokepoint(monkeypatch)

    stub = _StubAgent(provider="anthropic")
    stub._anthropic_api_key = "sk-ant-z"
    stub._try_refresh_anthropic_client_credentials = lambda: False
    client = ra.AIAgent._create_request_anthropic_client(stub, reason="x")
    assert client == "CLIENT_FOR:anthropic"
    assert calls[0]["api_key"] == "sk-ant-z"


# ── fallback activation for Claude-on-Vertex (the second live failure) ────────
#
# try_activate_fallback has its OWN api_mode detection block, separate from
# determine_api_mode. It had no Claude-on-Vertex case, so a vertex fallback
# model (e.g. fable-5 Overloaded → sonnet-5) defaulted to chat_completions,
# skipped the anthropic_messages client build, and 404'd on {host}/v1/messages.

class _FallbackStubAgent(_StubAgent):
    def __init__(self):
        super().__init__(provider="vertex")
        self.model = "claude-fable-5"
        self._fallback_index = 0
        self._fallback_chain = [{"provider": "vertex", "model": "claude-sonnet-5"}]
        self._unavailable_fallback_keys = None
        self._rate_limited_until = 0
        self.context_compressor = None  # skip the network-y context-length probe
        self.reasoning_config = None
        self._pending_fallback_notice = None

    def _try_activate_fallback(self, reason=None):  # recursion guard: should not fire
        raise AssertionError("unexpected chain recursion — happy path should not skip")

    def _is_azure_openai_url(self, _u):
        return False

    def _is_direct_openai_url(self, _u):
        return False

    def _provider_model_requires_responses_api(self, *a, **k):
        return False

    def _buffer_status(self, *a, **k):
        pass


class _FakeFbClient:
    base_url = "https://aiplatform.googleapis.com/v1"
    api_key = "vertex-oauth"


def test_fallback_to_vertex_claude_builds_anthropic_vertex(monkeypatch):
    import agent.chat_completion_helpers as helpers
    import agent.auxiliary_client as aux
    import agent.anthropic_adapter as aa

    calls = []

    def _fake_choke(provider, api_key, base_url, *, timeout=None,
                    drop_context_1m_beta=False, agent=None):
        calls.append(provider)
        return f"CLIENT_FOR:{provider}"

    monkeypatch.setattr(aux, "resolve_provider_client",
                        lambda *a, **k: (_FakeFbClient(), "claude-sonnet-5"))
    monkeypatch.setattr(helpers, "_fallback_entry_unavailable_without_network",
                        lambda *a, **k: None)
    monkeypatch.setattr(aa, "build_anthropic_client_for_provider", _fake_choke)
    monkeypatch.setattr(helpers, "rewrite_prompt_model_identity", lambda *a, **k: None)
    monkeypatch.setattr(helpers, "_reset_stale_streak", lambda *a, **k: None)

    agent = _FallbackStubAgent()
    ok = helpers.try_activate_fallback(agent, reason=None)

    assert ok is True
    assert agent.model == "claude-sonnet-5"
    assert agent.provider == "vertex"
    # The core assertions: detected as anthropic_messages and built via the
    # provider chokepoint (AnthropicVertex), NOT left on chat_completions.
    assert agent.api_mode == "anthropic_messages"
    assert agent._anthropic_client == "CLIENT_FOR:vertex"
    assert agent.client is None
    assert calls == ["vertex"]

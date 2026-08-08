"""Regression tests for wave-1 shard-s4 mixin extraction (w1a).

Two clusters moved verbatim out of ``run_agent.py`` into
``plugins/agent/mixins/``:

* ``request_client_lifecycle_mixin.py`` — per-request wire-client lifecycle
  (clusters c1+c2, 10 methods): image-part detection, copilot headers,
  request-client cache slot, create/close/abort for OpenAI-wire and
  Anthropic request-local clients.
* ``route_client_config_mixin.py`` — route-derived client-config helpers
  (cluster c3, 3 methods): URL-specific default headers, user-default
  header merge, TLS/extra-header recompute on credential rotation, plus
  the moved module-level helpers ``_routermint_headers`` /
  ``_qwen_portal_headers`` / ``_QWEN_CODE_VERSION``.

The moved methods are byte-identical to the originals; ``AIAgent`` now
inherits them via ``class AIAgent(RequestClientLifecycleMixin,
RouteClientConfigMixin)``. These tests pin the *pure* moved logic using the
bare-adapter pattern (``object.__new__`` + stub config, mirroring how
existing tests in this directory construct agents without ``__init__``).

Cross-cut helpers that deliberately STAY on ``AIAgent`` (the shared-client
core from shard s3: ``_openai_client_lock``, ``_create_openai_client``,
``_close_openai_client``, ``_is_openai_client_closed``,
``_force_close_tcp_sockets``, ``_client_log_context``,
``_ensure_primary_openai_client``) are stubbed on the bare instance so the
moved methods can be exercised standalone.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from plugins.agent.mixins.request_client_lifecycle_mixin import (
    RequestClientLifecycleMixin,
)
from plugins.agent.mixins.route_client_config_mixin import (
    RouteClientConfigMixin,
    _QWEN_CODE_VERSION,
    _qwen_portal_headers,
    _routermint_headers,
)

# ---------------------------------------------------------------------------
# Cluster c1+c2 — request client lifecycle
# ---------------------------------------------------------------------------


class TestApiKwargsHaveImageParts:
    def test_image_part_in_messages(self):
        api_kwargs = {
            "messages": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
            ]
        }
        assert RequestClientLifecycleMixin._api_kwargs_have_image_parts(api_kwargs) is True

    def test_input_image_part_in_responses_input(self):
        api_kwargs = {"input": [{"type": "input_image", "image_url": "data:image/png;base64,x"}]}
        assert RequestClientLifecycleMixin._api_kwargs_have_image_parts(api_kwargs) is True

    def test_text_only_returns_false(self):
        api_kwargs = {"messages": [{"type": "text", "text": "hi"}, {"type": "text", "text": "bye"}]}
        assert RequestClientLifecycleMixin._api_kwargs_have_image_parts(api_kwargs) is False

    def test_nested_image_part_in_tool_content(self):
        api_kwargs = {
            "messages": [
                {
                    "role": "tool",
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "image_url", "image_url": {"url": "u"}},
                    ],
                }
            ]
        }
        assert RequestClientLifecycleMixin._api_kwargs_have_image_parts(api_kwargs) is True

    def test_non_dict_returns_false(self):
        assert RequestClientLifecycleMixin._api_kwargs_have_image_parts(None) is False
        assert RequestClientLifecycleMixin._api_kwargs_have_image_parts("nope") is False
        assert RequestClientLifecycleMixin._api_kwargs_have_image_parts([]) is False


class TestRequestClientCacheRef:
    def test_lazy_init_creates_slot(self):
        agent = object.__new__(RequestClientLifecycleMixin)
        cache = agent._request_client_cache_ref()
        assert cache == {"client": None, "kwargs": None, "poisoned": False, "in_use": False}

    def test_same_cache_returned_on_repeat(self):
        agent = object.__new__(RequestClientLifecycleMixin)
        assert agent._request_client_cache_ref() is agent._request_client_cache_ref()

    def test_does_not_clobber_existing_slot(self):
        agent = object.__new__(RequestClientLifecycleMixin)
        agent._request_client_cache = {"client": "c", "kwargs": "k", "poisoned": True, "in_use": True}
        assert agent._request_client_cache_ref() is agent._request_client_cache


class TestCreateRequestOpenaiClient:
    def _bare_agent(self, **attrs):
        agent = object.__new__(RequestClientLifecycleMixin)
        agent._client_kwargs = {
            "api_key": "k",
            "base_url": "https://api.openai.com/v1",
            "max_retries": 0,
        }
        agent.provider = "openai"
        # s3 shared-client core stays on AIAgent — stub the contract here.
        agent._ensure_primary_openai_client = lambda **kw: SimpleNamespace(
            __class__=object, base_url="https://api.openai.com/v1"
        )
        agent._openai_client_lock = lambda: _NullLock()
        agent._request_client_cache_ref = RequestClientLifecycleMixin._request_client_cache_ref.__get__(
            agent, RequestClientLifecycleMixin
        )
        agent._is_openai_client_closed = lambda client: False
        agent._create_openai_client = lambda kwargs, **kw: SimpleNamespace(kwargs=kwargs)
        agent._close_openai_client = lambda *a, **kw: None
        agent._api_kwargs_have_image_parts = staticmethod(
            RequestClientLifecycleMixin._api_kwargs_have_image_parts
        )
        agent._copilot_headers_for_request = lambda **kw: {"User-Agent": "test"}
        for k, v in attrs.items():
            setattr(agent, k, v)
        return agent

    def test_sets_max_retries_zero(self):
        agent = self._bare_agent()
        client = agent._create_request_openai_client(reason="chat_completion_request")
        assert client.kwargs["max_retries"] == 0

    def test_caches_client_for_reuse_after_clean_close(self):
        agent = self._bare_agent()
        agent._REQUEST_CLIENT_REUSE_REASONS = frozenset({"request_complete"})
        a = agent._create_request_openai_client(reason="chat_completion_request")
        # clean finish on the owning thread: in_use=False, client kept
        agent._close_request_openai_client(a, reason="request_complete")
        b = agent._create_request_openai_client(reason="chat_completion_request")
        assert a is b
        assert agent._request_client_cache_ref()["in_use"] is True

    def test_kwargs_change_builds_fresh_client(self):
        agent = self._bare_agent()
        a = agent._create_request_openai_client(reason="r1")
        agent._client_kwargs["base_url"] = "https://other.example/v1"
        b = agent._create_request_openai_client(reason="r2")
        assert a is not b


class TestCloseRequestOpenaiClient:
    def _bare_agent(self, closes):
        agent = object.__new__(RequestClientLifecycleMixin)
        agent._openai_client_lock = lambda: _NullLock()
        agent._request_client_cache_ref = RequestClientLifecycleMixin._request_client_cache_ref.__get__(
            agent, RequestClientLifecycleMixin
        )
        agent._close_openai_client = lambda client, reason, shared: closes.append(reason)
        agent._REQUEST_CLIENT_REUSE_REASONS = frozenset({"request_complete"})
        return agent

    def test_reuse_reason_keeps_wire_client(self):
        closes = []
        agent = self._bare_agent(closes)
        client = SimpleNamespace()
        cache = agent._request_client_cache_ref()
        cache["client"] = client
        cache["in_use"] = True
        agent._close_request_openai_client(client, reason="request_complete")
        assert closes == []
        assert agent._request_client_cache_ref()["in_use"] is False

    def test_non_reuse_reason_really_closes(self):
        closes = []
        agent = self._bare_agent(closes)
        client = SimpleNamespace()
        cache = agent._request_client_cache_ref()
        cache["client"] = client
        cache["in_use"] = True
        agent._close_request_openai_client(client, reason="request_error")
        assert closes == ["request_error"]
        assert agent._request_client_cache_ref()["client"] is None


class TestMroWiring:
    """The moved methods must be reachable through AIAgent's MRO."""

    # method name -> mixin class that now owns it
    _OWNERS = {
        "_api_kwargs_have_image_parts": RequestClientLifecycleMixin,
        "_copilot_headers_for_request": RequestClientLifecycleMixin,
        "_request_client_cache_ref": RequestClientLifecycleMixin,
        "_create_request_openai_client": RequestClientLifecycleMixin,
        "_close_request_openai_client": RequestClientLifecycleMixin,
        "_close_cached_request_openai_client": RequestClientLifecycleMixin,
        "_abort_request_openai_client": RequestClientLifecycleMixin,
        "_create_request_anthropic_client": RequestClientLifecycleMixin,
        "_close_request_anthropic_client": RequestClientLifecycleMixin,
        "_abort_request_anthropic_client": RequestClientLifecycleMixin,
        "_apply_client_headers_for_base_url": RouteClientConfigMixin,
        "_apply_user_default_headers": RouteClientConfigMixin,
        "_reapply_route_client_config": RouteClientConfigMixin,
    }

    def test_methods_are_wired_through_mro(self):
        import run_agent

        for name, mixin in self._OWNERS.items():
            assert getattr(run_agent.AIAgent, name) is getattr(mixin, name), name

    def test_module_helpers_reexported_through_run_agent(self):
        import run_agent

        assert run_agent._routermint_headers is _routermint_headers
        assert run_agent._qwen_portal_headers is _qwen_portal_headers
        assert run_agent._QWEN_CODE_VERSION == _QWEN_CODE_VERSION


# ---------------------------------------------------------------------------
# Cluster c3 — route client config + moved module-level helpers
# ---------------------------------------------------------------------------


class TestRouteModuleHelpers:
    def test_qwen_version_constant(self):
        assert _QWEN_CODE_VERSION == "0.14.1"

    def test_qwen_portal_headers_shape(self):
        headers = _qwen_portal_headers()
        assert headers["User-Agent"].startswith(f"QwenCode/{_QWEN_CODE_VERSION}")
        assert headers["X-DashScope-CacheControl"] == "enable"
        assert headers["X-DashScope-AuthType"] == "qwen-oauth"

    def test_routermint_headers_has_hermes_ua(self):
        headers = _routermint_headers()
        assert headers["User-Agent"].startswith("HermesAgent/")


class TestApplyClientHeadersForBaseUrl:
    def _bare_agent(self, base_url="https://api.openai.com/v1"):
        agent = object.__new__(RouteClientConfigMixin)
        agent._client_kwargs = {"api_key": "k", "base_url": base_url}
        agent.provider = "openai"
        agent.api_mode = "chat_completions"
        return agent

    def test_openrouter_headers_applied(self):
        agent = self._bare_agent("https://openrouter.ai/api/v1")
        agent._apply_client_headers_for_base_url("https://openrouter.ai/api/v1")
        headers = agent._client_kwargs.get("default_headers")
        assert isinstance(headers, dict)

    def test_unknown_base_url_clears_headers_when_no_profile(self, monkeypatch):
        import providers

        monkeypatch.setattr(providers, "get_provider_profile", lambda name: None)
        agent = self._bare_agent("https://generic.example/v1")
        agent._client_kwargs["default_headers"] = {"User-Agent": "old"}
        agent._apply_client_headers_for_base_url("https://generic.example/v1")
        assert "default_headers" not in agent._client_kwargs

    def test_qwen_route_uses_moved_portal_headers(self):
        agent = self._bare_agent("https://portal.qwen.ai")
        agent._apply_client_headers_for_base_url("https://portal.qwen.ai")
        ua = agent._client_kwargs["default_headers"]["User-Agent"]
        assert ua.startswith(f"QwenCode/{_QWEN_CODE_VERSION}")

    def test_routermint_route_uses_moved_helper(self):
        agent = self._bare_agent("https://api.routermint.com")
        agent._apply_client_headers_for_base_url("https://api.routermint.com")
        ua = agent._client_kwargs["default_headers"]["User-Agent"]
        assert ua.startswith("HermesAgent/")


class TestApplyUserDefaultHeaders:
    def _bare_agent(self):
        agent = object.__new__(RouteClientConfigMixin)
        agent._client_kwargs = {"api_key": "k"}
        agent.api_mode = "chat_completions"
        return agent

    def test_merges_user_headers(self, monkeypatch):
        agent = self._bare_agent()
        calls = []

        def fake_merge(headers):
            calls.append(headers)
            merged = dict(headers or {})
            merged["User-Agent"] = "curl/8.7.1"
            return merged

        import agent.auxiliary_client as aux

        monkeypatch.setattr(aux, "_apply_user_default_headers", fake_merge)
        agent._client_kwargs["default_headers"] = {"X-Foo": "bar"}
        agent._apply_user_default_headers()
        assert calls == [{"X-Foo": "bar"}]
        assert agent._client_kwargs["default_headers"]["User-Agent"] == "curl/8.7.1"

    def test_anthropic_mode_is_noop(self):
        agent = self._bare_agent()
        agent.api_mode = "anthropic_messages"
        agent._apply_user_default_headers()  # must not raise


class TestReapplyRouteClientConfig:
    def test_recomputes_headers_and_clears_tls(self):
        agent = object.__new__(RouteClientConfigMixin)
        agent.base_url = "https://openrouter.ai/api/v1"
        agent._client_kwargs = {
            "api_key": "k",
            "base_url": "https://openrouter.ai/api/v1",
            "ssl_verify": True,
            "ssl_ca_cert": "/tmp/ca.pem",
        }
        agent._apply_client_headers_for_base_url = lambda *a, **kw: agent._client_kwargs.setdefault(
            "default_headers", {"User-Agent": "recomputed"}
        )
        agent._reapply_route_client_config(route_changed=False)
        assert "ssl_verify" not in agent._client_kwargs
        assert "ssl_ca_cert" not in agent._client_kwargs
        assert agent._client_kwargs.get("default_headers") == {"User-Agent": "recomputed"}


class _NullLock:
    """Context manager lock stub (mirrors AIAgent._openai_client_lock contract)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

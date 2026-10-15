"""Vision capability and fallback behavior through the real auxiliary route."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.auxiliary_client import call_llm, async_call_llm, resolve_vision_provider_client

@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip provider env vars so each test starts clean."""
    for key in (
        "OPENROUTER_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_KEY",
        "OPENAI_MODEL", "LLM_MODEL", "NOUS_INFERENCE_BASE_URL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN",
        "NVIDIA_API_KEY", "NVIDIA_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    # Module-level unhealthy cache (10-min TTL) leaks between tests;
    # earlier tests that call _mark_provider_unhealthy() poison the
    # cache for later ones, causing _resolve_auto_route to skip providers
    # that the test patched to return valid clients.
    import agent.auxiliary_client as _aux_mod
    _aux_mod._aux_unhealthy_until.clear()
    _aux_mod._aux_unhealthy_logged_at.clear()
    yield
    _aux_mod._aux_unhealthy_until.clear()
    _aux_mod._aux_unhealthy_logged_at.clear()

class TestVisionCapabilityWalk:
    """Public calls use real configuration, selection, ladder, conversion and SDK shaping."""

    @pytest.fixture
    def routes(self, monkeypatch):
        from copy import deepcopy

        import httpx
        from agent import auxiliary_client as aux

        primary = {"provider": "openrouter", "model": "openai/gpt-4o"}

        def entry(model, host="main.example", provider="custom"):
            return {"provider": provider, "model": model,
                    "base_url": f"https://{host}/v1", "api_key": "candidate-key",
                    "api_mode": "chat_completions"}

        config = {
            "model": {"provider": "openrouter", "default": primary["model"]},
            "providers": {
                "openrouter": {"models": {primary["model"]: {"supports_vision": True}}},
                "custom": {"models": {"known-text": {"supports_vision": False}}},
            },
            "auxiliary": {"vision": {
                "provider": "auto", "model": primary["model"],
                "fallback_chain": [primary, {"provider": "anthropic", "model": "unavailable"}],
            }},
            "fallback_providers": [
                primary, entry("known-text"), entry("text", provider="kimi-coding"),
                entry("main-a"), entry("main-b"),
            ],
        }
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": "https://image.example/photo.png"}},
        ]}]
        tools = [{"type": "function", "function": {
            "name": "annotate", "parameters": {"type": "object"},
        }}]
        reasoning = {"enabled": True, "effort": "low"}
        route_info, requests, errors = {}, [], []
        faults = {primary["model"]: (429, "Rate limit exceeded"),
                  "main-a": (400, "image_url unsupported; text-only model")}
        monkeypatch.setenv("OPENROUTER_API_KEY", "primary-key")
        # Refresh is an external credential operation. No ladder/selector is mocked.
        refresh = MagicMock(return_value=False)
        monkeypatch.setattr(aux, "_refresh_provider_credentials", refresh)

        def respond(request):
            assert request.url.path.endswith("/chat/completions")
            payload = json.loads(request.content)
            requests.append((request, payload, dict(route_info)))
            fault = faults.get(payload["model"])
            if fault:
                status, message = fault
                return httpx.Response(status, request=request, json={"error": {"message": message}})
            return httpx.Response(200, request=request, json={
                "id": "vision-result", "object": "chat.completion", "created": 0,
                "model": payload["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "described"},
                             "finish_reason": "stop"}],
            })

        async def async_send(client, request, **kwargs):
            return respond(request)

        import openai
        for cls in (openai.OpenAI, openai.AsyncOpenAI):
            original = cls._make_status_error

            def record_error(client, *args, _original=original, **kwargs):
                error = _original(client, *args, **kwargs)
                errors.append(error)
                return error

            monkeypatch.setattr(cls, "_make_status_error", record_error)
        monkeypatch.setattr(httpx.Client, "send", lambda client, request, **kw: respond(request))
        monkeypatch.setattr(httpx.AsyncClient, "send", async_send)

        async def invoke(async_mode, **overrides):
            (aux.get_hermes_home() / "config.yaml").write_text(json.dumps(config))
            kwargs: dict = dict(task="vision", messages=messages, tools=tools, temperature=0.2,
                          timeout=19, max_tokens=73, reasoning_config=reasoning,
                          extra_body={"test_marker": "preserved"}, route_info=route_info)
            kwargs.update(overrides)
            return await async_call_llm(**kwargs) if async_mode else call_llm(**kwargs)

        return SimpleNamespace(
            aux=aux, config=config, entry=entry, primary=primary, faults=faults, invoke=invoke,
            requests=requests, errors=errors, refresh=refresh, route_info=route_info,
            messages=messages, tools=tools, reasoning=reasoning,
            original=deepcopy((messages, tools, reasoning)),
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
    async def test_main_false_does_not_override_candidate_true(self, routes, async_mode):
        from copy import deepcopy

        r = routes
        r.config["model"].update(
            provider="custom:main-only", default="main-text", supports_vision=False,
            base_url="https://main-only.example/v1", api_key="main-only-key",
        )
        r.config["fallback_providers"] = [r.entry("main-b")]
        r.config["providers"]["custom"]["models"]["main-b"] = {"supports_vision": True}
        original_config = deepcopy(r.config)

        response = await r.invoke(async_mode)

        assert response.choices[0].message.content == "described"
        assert [p["model"] for _, p, _ in r.requests] == [r.primary["model"]] * 2 + ["main-b"]
        request, payload, recorded = r.requests[-1]
        assert request.url.host == "main.example"
        assert request.headers["authorization"] == "Bearer candidate-key"
        assert payload["messages"] == r.original[0]
        assert payload["tools"] == r.original[1]
        assert recorded == r.route_info == {"provider": "custom", "model": "main-b"}
        assert (r.messages, r.tools, r.reasoning) == r.original
        assert r.config == original_config
        assert json.loads((r.aux.get_hermes_home() / "config.yaml").read_text()) == original_config
        assert not r.aux._main_model_supports_vision("custom:main-only", "main-text")
        r.refresh.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
    async def test_main_true_does_not_override_candidate_false(self, routes, async_mode):
        from copy import deepcopy

        r = routes
        r.config["model"]["supports_vision"] = True
        r.config["fallback_providers"] = [r.entry("known-text"), r.entry("main-b")]
        r.config["providers"]["custom"]["models"].pop("known-text")
        r.config["custom_providers"] = [
            {"name": "custom", "models": {"known-text": {"vision": False}}},
        ]
        # The unknown custom candidate must not inherit the active provider's verdict either.
        r.config["providers"]["openrouter"]["models"]["main-b"] = {"supports_vision": False}
        original_config = deepcopy(r.config)

        response = await r.invoke(async_mode)

        assert response.choices[0].message.content == "described"
        assert [p["model"] for _, p, _ in r.requests] == [r.primary["model"]] * 2 + ["main-b"]
        request, payload, recorded = r.requests[-1]
        assert request.url.host == "main.example"
        assert request.headers["authorization"] == "Bearer candidate-key"
        assert payload["messages"] == r.original[0]
        assert payload["tools"] == r.original[1]
        assert recorded == r.route_info == {"provider": "custom", "model": "main-b"}
        assert (r.messages, r.tools, r.reasoning) == r.original
        assert r.config == original_config
        assert json.loads((r.aux.get_hermes_home() / "config.yaml").read_text()) == original_config
        assert r.aux._main_model_supports_vision(r.primary["provider"], r.primary["model"])
        r.refresh.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
    @pytest.mark.parametrize("status,message", [
        (400, "image_url unsupported; text-only model"),
        (415, "image input not supported"),
        (422, "unknown variant `image_url`, expected `text`"),
        (400, "This model is not supported when using this route"),
    ])
    async def test_main_capability_walk_preserves_request(self, routes, async_mode, status, message):
        r = routes
        r.faults["main-a"] = (status, message)
        response = await r.invoke(async_mode)
        assert response.choices[0].message.content == "described"
        assert [p["model"] for _, p, _ in r.requests] == [r.primary["model"]] * 2 + ["main-a", "main-b"]
        for request, payload, recorded in r.requests:
            assert payload["messages"] == r.original[0]
            assert payload["tools"] == r.original[1]
            if payload["model"] == r.primary["model"]:
                assert payload["reasoning"] == r.original[2]
            else:
                # The native custom provider profile uses the OpenAI top-level field.
                assert payload["reasoning_effort"] == r.reasoning["effort"]
            assert payload["temperature"] == 0.2
            assert payload["test_marker"] == "preserved"
            assert request.extensions["timeout"]["read"] == 19
            primary = payload["model"] == r.primary["model"]
            assert payload.get("max_tokens", payload.get("max_completion_tokens")) == (73 if primary else None)
            assert request.headers["authorization"] == ("Bearer primary-key" if primary else "Bearer candidate-key")
            assert recorded == {"provider": "openrouter" if primary else "custom", "model": payload["model"]}
        assert (r.messages, r.tools, r.reasoning) == r.original
        assert r.route_info == {"provider": "custom", "model": "main-b"}
        r.refresh.assert_not_called()
        assert not r.aux._is_provider_unhealthy("custom", "https://main.example/v1")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
    @pytest.mark.parametrize("origin", ["task", "main"])
    @pytest.mark.parametrize("status", [429, 402])
    async def test_capacity_candidate_advances_ordered_vision_chain(self, routes, async_mode, origin, status):
        r = routes
        r.faults["main-a"] = (status, "image input unsupported; quota exceeded")
        if origin == "task":
            r.config["auxiliary"]["vision"]["fallback_chain"] = [
                r.primary, r.entry("main-a", host="depleted.example"), r.entry("main-b"),
            ]
            r.config["fallback_providers"] = [r.entry("must-not-run")]
        else:
            r.config["fallback_providers"] = [r.entry("main-a", host="depleted.example"), r.entry("main-b")]

        result = await r.invoke(async_mode)

        assert result.choices[0].message.content == "described"
        assert [p["model"] for _, p, _ in r.requests] == [r.primary["model"]] * 2 + ["main-a", "main-b"]
        assert r.route_info == {"provider": "custom", "model": "main-b"}
        assert (r.messages, r.tools, r.reasoning) == r.original
        r.refresh.assert_not_called()
        assert r.aux._is_provider_unhealthy("custom", "https://depleted.example/v1")
        assert not r.aux._is_provider_unhealthy("custom", "https://main.example/v1")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
    @pytest.mark.parametrize("status,reason,ttl", [
        (429, "rate limit", 60.0), (402, "payment error", None),
    ])
    @pytest.mark.parametrize("second", ["known", "proven"])
    async def test_quarantined_main_candidate_then_image_incompatible_main_candidate_advances(
        self, routes, monkeypatch, async_mode, status, reason, ttl, second,
    ):
        r = routes
        second_model = "known-text" if second == "known" else "main-b"
        r.config["fallback_providers"] = [
            r.entry("main-a", host="depleted.example"),
            r.entry(second_model), r.entry("main-c"),
        ]
        r.faults["main-a"] = (status, "Capacity exhausted")
        if second == "proven":
            r.faults[second_model] = (400, "image_url unsupported; text-only model")
        quarantines = []
        original_mark = r.aux._mark_provider_unhealthy

        def mark(provider, *args, **kwargs):
            quarantines.append((provider, kwargs.get("base_url"), kwargs.get("reason"), kwargs.get("ttl")))
            return original_mark(provider, *args, **kwargs)

        monkeypatch.setattr(r.aux, "_mark_provider_unhealthy", mark)

        result = await r.invoke(async_mode)

        assert result.choices[0].message.content == "described"
        expected = ["main-a"] + ([second_model] if second == "proven" else []) + ["main-c"]
        assert [payload["model"] for _, payload, _ in r.requests] == [r.primary["model"]] * 2 + expected
        assert r.route_info == {"provider": "custom", "model": "main-c"}
        assert all(payload["messages"] == r.messages and payload["tools"] == r.tools
                   for _, payload, _ in r.requests)
        assert (r.messages, r.tools, r.reasoning) == r.original
        r.refresh.assert_not_called()
        assert ("custom", "https://depleted.example/v1", reason, ttl) in quarantines
        assert len(quarantines) == 1
        assert r.aux._is_provider_unhealthy("custom", "https://depleted.example/v1")
        assert not r.aux._is_provider_unhealthy("custom", "https://main.example/v1")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
    async def test_stale_vision_candidate_refreshes_and_retries_before_backup(self, routes, monkeypatch, async_mode):
        r = routes
        entry = r.entry("main-a", provider="openrouter")
        entry.pop("api_key")
        r.config["auxiliary"]["vision"]["fallback_chain"] = [r.primary, entry, r.entry("main-b")]
        r.config["fallback_providers"] = [r.entry("must-not-run")]
        r.faults["main-a"] = (401, "Invalid API key")

        def refresh(provider, **kwargs):
            assert provider == "openrouter"
            monkeypatch.setenv("OPENROUTER_API_KEY", "fresh-key")
            r.aux._evict_cached_clients("openrouter")
            r.faults.pop("main-a")
            return True

        r.refresh.side_effect = refresh

        result = await r.invoke(async_mode)

        assert result.choices[0].message.content == "described"
        assert [p["model"] for _, p, _ in r.requests] == [r.primary["model"]] * 2 + ["main-a"] * 2
        assert [request.headers["authorization"] for request, payload, _ in r.requests[-2:]] == [
            "Bearer primary-key", "Bearer fresh-key",
        ]
        assert r.refresh.call_count == 1
        assert r.route_info == {"provider": "openrouter", "model": "main-a"}
        assert not r.aux._is_provider_unhealthy("openrouter", "https://main.example/v1")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
    async def test_refreshed_vision_candidate_rate_limit_uses_transient_quarantine(
        self, routes, monkeypatch, async_mode
    ):
        r = routes
        entry = r.entry("main-a", provider="openrouter")
        entry.pop("api_key")
        r.config["auxiliary"]["vision"]["fallback_chain"] = [r.primary, entry, r.entry("main-b")]
        r.config["fallback_providers"] = [r.entry("must-not-run")]
        r.faults["main-a"] = (401, "Invalid API key")
        quarantines = []
        original_mark = r.aux._mark_provider_unhealthy

        def refresh(provider, **kwargs):
            assert provider == "openrouter"
            monkeypatch.setenv("OPENROUTER_API_KEY", "fresh-key")
            r.aux._evict_cached_clients("openrouter")
            r.faults["main-a"] = (429, "Rate limit exceeded")
            return True

        def mark(provider, *args, **kwargs):
            quarantines.append((provider, kwargs.get("reason"), kwargs.get("ttl")))
            return original_mark(provider, *args, **kwargs)

        r.refresh.side_effect = refresh
        monkeypatch.setattr(r.aux, "_mark_provider_unhealthy", mark)

        result = await r.invoke(async_mode)

        assert result.choices[0].message.content == "described"
        assert [payload["model"] for _, payload, _ in r.requests] == [
            r.primary["model"], r.primary["model"], "main-a", "main-a", "main-b"
        ]
        assert r.route_info == {"provider": "custom", "model": "main-b"}
        assert any(reason == "rate limit" and ttl == 60.0 for _, reason, ttl in quarantines)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
    @pytest.mark.parametrize("status,message", [
        (404, "image input unsupported; model not found"),
        (400, "Unknown validation failure"),
        (400, "image input unsupported; model not found"),
        (400, "vision unsupported; billing quota exceeded"),
        (400, "vision unsupported; invalid API key"),
        (415, "vision unsupported; quota exhausted"),
        (422, "vision unsupported; model not found"),
        (500, "image input unsupported"),
    ])
    async def test_candidate_non_capability_error_propagates(self, routes, async_mode, status, message):
        r = routes
        # No skip screen here: even the unmodified base reaches this candidate.
        r.config["fallback_providers"] = [r.entry("main-a"), r.entry("main-b")]
        r.faults["main-a"] = (status, message)
        with pytest.raises(Exception) as caught:
            await r.invoke(async_mode)
        assert caught.value is r.errors[-1]
        assert caught.value.status_code == status
        assert message in str(caught.value)
        assert [p["model"] for _, p, _ in r.requests] == [r.primary["model"]] * 2 + ["main-a"]
        r.refresh.assert_not_called()
        assert not r.aux._is_provider_unhealthy("custom", "https://main.example/v1")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
    @pytest.mark.parametrize("origin", ["task", "main"])
    @pytest.mark.parametrize("message,capability", [
        ("image input unsupported", True),
        ("vision unsupported; billing quota exceeded", False),
        ("vision unsupported; invalid API key", False),
        ("vision unsupported; model not found", False),
        ("Unknown validation failure", False),
        ("Auxiliary vision: LLM returned invalid response (choices[0].message)", False),
    ])
    async def test_only_main_origin_can_advance_no_status_capability(self, routes, monkeypatch, async_mode, origin, message, capability):
        from openai.resources.chat.completions import Completions, AsyncCompletions

        r = routes
        r.config["fallback_providers"] = [r.entry("main-a"), r.entry("main-b")]
        if origin == "task":
            r.config["auxiliary"]["vision"]["fallback_chain"] = [
                r.entry("main-a", host="depleted.example"), r.entry("main-b"),
            ]
        else:
            r.config["fallback_providers"][0] = r.entry("main-a", host="depleted.example")
        error = RuntimeError(message)
        calls = []
        cls = AsyncCompletions if async_mode else Completions
        original = cls.create

        def create(client, **kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "main-a":
                raise error
            return original(client, **kwargs)

        async def acreate(client, **kwargs):
            result = create(client, **kwargs)
            return await result

        monkeypatch.setattr(cls, "create", acreate if async_mode else create)
        unavailable = message in {
            "vision unsupported; billing quota exceeded",
            "Auxiliary vision: LLM returned invalid response (choices[0].message)",
        }
        if (origin == "task" and not unavailable) or (not capability and not unavailable):
            with pytest.raises(RuntimeError) as caught:
                await r.invoke(async_mode)
            assert caught.value is error
            assert calls == [r.primary["model"]] * 2 + ["main-a"]
        else:
            assert (await r.invoke(async_mode)).choices[0].message.content == "described"
            assert calls == [r.primary["model"]] * 2 + ["main-a", "main-b"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
    @pytest.mark.parametrize("initial_status,policy,succeeds", [
        (400, "auto", True), (401, "auto", True), (401, "openrouter", False),
    ])
    async def test_initial_auto_eligibility_stays_separate(self, routes, async_mode, initial_status, policy, succeeds):
        r = routes
        r.config["fallback_providers"] = [r.entry("main-b")]
        r.faults[r.primary["model"]] = (initial_status, "image input unsupported" if initial_status == 400 else "Invalid API key")
        if succeeds:
            assert (await r.invoke(async_mode, provider=policy)).choices[0].message.content == "described"
        else:
            with pytest.raises(Exception) as caught:
                await r.invoke(async_mode, provider=policy)
            assert caught.value is r.errors[0]
        assert [p["model"] for _, p, _ in r.requests] == [r.primary["model"]] + (["main-b"] if succeeds else [])
        assert r.refresh.call_count == (1 if initial_status == 401 else 0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
    @pytest.mark.parametrize("discovery", [False, True])
    @pytest.mark.parametrize("empty", [False, True])
    async def test_exhausted_walk_preserves_discovery_and_original_error(self, routes, monkeypatch, caplog, async_mode, discovery, empty):
        r = routes
        r.config["fallback_providers"] = [] if empty else [r.entry("main-a"), r.entry("main-a")]
        r.config["model"]["provider"] = "auto"
        # Discovery is external availability I/O; retain its real ordering/eligibility owner.
        probe = MagicMock(return_value=(None, None))
        if discovery:
            def discover():
                return r.aux.resolve_provider_client("custom", "discovered", explicit_base_url="https://discovery.example/v1", explicit_api_key="discovery-key", api_mode="chat_completions")
            probe.side_effect = discover
        monkeypatch.setattr(r.aux, "_get_provider_chain", lambda: [("custom", probe)])
        if discovery:
            assert (await r.invoke(async_mode)).choices[0].message.content == "described"
            assert r.route_info == {"provider": "custom", "model": "discovered"}
        else:
            with pytest.raises(Exception) as caught:
                await r.invoke(async_mode)
            assert caught.value is r.errors[0]
            assert sum("all fallbacks exhausted" in record.message for record in caplog.records) == 1
        probe.assert_called_once()
        assert [p["model"] for _, p, _ in r.requests] == [r.primary["model"]] * 2 + ([] if empty else ["main-a"]) + (["discovered"] if discovery else [])

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
    async def test_configured_resolved_aliases_bounded_and_endpoints_distinct(self, routes, async_mode):
        r = routes
        # An unresolvable route must not poison another label at the same endpoint.
        # OpenCode strips vendor/ natively; exclude both configured and resolved labels.
        r.config["fallback_providers"] = [
            r.entry("main-a", provider="unconfigured-provider"),
            r.entry("vendor/main-a", provider="opencode-go"),
            r.entry("main-a", provider="opencode-go"),
            r.entry("main-a", provider="custom"),
            r.entry("main-b"),
            r.entry("main-b", host="other.example"),
        ]
        r.faults["main-b"] = (400, "image input unsupported")
        # The second endpoint must be attempted even though the first endpoint rejected that model.
        with pytest.raises(Exception):
            await r.invoke(async_mode)
        calls = [(request.url.host, payload["model"]) for request, payload, _ in r.requests]
        assert calls == [("openrouter.ai", r.primary["model"])] * 2 + [
            ("main.example", "main-a"), ("main.example", "main-b"), ("other.example", "main-b"),
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
    async def test_default_endpoint_rejection_preserves_explicit_relay(self, routes, async_mode):
        r = routes
        default = {"provider": "openrouter", "model": "main-a"}
        same = {**default, "base_url": "https://openrouter.ai/api/v1"}
        relay = r.entry("main-a", host="relay.example", provider="openrouter")
        r.config["fallback_providers"] = [default, same, relay, dict(relay)]

        with pytest.raises(Exception):
            await r.invoke(async_mode)

        assert [(request.url.host, payload["model"])
                for request, payload, _ in r.requests] == [
            ("openrouter.ai", r.primary["model"]),
            ("openrouter.ai", r.primary["model"]),
            ("openrouter.ai", "main-a"),
            ("relay.example", "main-a"),
        ]
        assert all(payload["messages"] == r.messages for _, payload, _ in r.requests)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
    async def test_explicit_transport_survives_fallback_async_conversion(self, routes, async_mode):
        r = routes
        destination = r.entry("claude-vision")
        destination["base_url"] = "https://api.anthropic.com/v1"
        r.config["fallback_providers"] = [destination]
        assert (await r.invoke(async_mode, reasoning_config=None)).choices[0].message.content == "described"
        _, payload, _ = r.requests[-1]
        # Explicit chat_completions wins over the URL heuristic: no Anthropic cache decoration.
        assert payload["messages"] == r.messages
        assert payload["tools"] == r.tools

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
    async def test_native_token_retry_retains_tools(self, routes, async_mode):
        r = routes
        # HTTP boundary returns success only once the native parameter rung removes the cap.
        import httpx
        original_send = httpx.AsyncClient.send if async_mode else httpx.Client.send
        calls = []

        def respond(client, request, **kwargs):
            payload = json.loads(request.content)
            calls.append(payload)
            if len(calls) == 1:
                return httpx.Response(400, request=request, json={"error": {"message": "Unsupported parameter: max_tokens"}})
            r.faults.clear()
            return original_send(client, request, **kwargs)

        async def async_send(client, request, **kwargs):
            result = respond(client, request, **kwargs)
            return await result if hasattr(result, "__await__") else result

        with patch.object(httpx.AsyncClient if async_mode else httpx.Client, "send", async_send if async_mode else respond):
            assert (await r.invoke(async_mode)).choices[0].message.content == "described"
        assert len(calls) == 2
        assert calls[0].get("max_tokens", calls[0].get("max_completion_tokens")) == 73
        assert "max_tokens" not in calls[1] and "max_completion_tokens" not in calls[1]
        assert calls[0]["messages"] == calls[1]["messages"] == r.messages
        assert calls[0]["tools"] == calls[1]["tools"] == r.tools


    @pytest.mark.parametrize("failed,candidate", [("xai", "custom:xai"), ("custom:xai", "xai")])
    def test_raw_credential_namespace_isolation_in_main_selector(self, routes, failed, candidate):
        r = routes
        r.config["fallback_providers"] = [r.entry("vision-model", provider=candidate)]
        (r.aux.get_hermes_home() / "config.yaml").write_text(json.dumps(r.config))
        client, model, label = r.aux._try_main_fallback_chain(
            "vision", failed, reason="auth error", failed_base_url="https://main.example/v1",
        )
        assert client is not None
        assert model == "vision-model"
        assert label == candidate


def test_unexpected_candidate_capability_probe_failure_is_logged_and_remains_eligible(monkeypatch, caplog):
    from agent.auxiliary_client import _candidate_model_supports_vision
    from agent import image_routing

    monkeypatch.setattr(image_routing, "_supports_vision_override", lambda *args: None)
    monkeypatch.setattr(image_routing, "_probe_models_dev", lambda *args: (_ for _ in ()).throw(RuntimeError("probe unavailable")))
    with caplog.at_level("ERROR", logger="agent.auxiliary_client"):
        assert _candidate_model_supports_vision("custom", "vision-model") is True
    assert "probe unavailable" in caplog.text


class TestVisionFallbackPolicy:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
    @pytest.mark.parametrize("policy", ["auto", "AUTO", " auto ", "explicit-argument", "explicit-config"])
    @pytest.mark.parametrize("task_fallback", ["absent", "healthy", "stale"])
    async def test_vision_429_preserves_fallback_policy(
        self, monkeypatch, async_mode, policy, task_fallback,
    ):
        """Auto keeps the configured main chain after a concrete backend exhausts its retry."""
        from copy import deepcopy

        import httpx
        import openai

        from agent import auxiliary_client as aux

        primary = {"provider": "openrouter", "model": "openai/gpt-4o"}
        task_entry = {
            "provider": "custom", "model": "task-vision",
            "base_url": "https://task.example/v1", "api_key": "task-key",
            "api_mode": "chat_completions",
        }
        main_entry = {
            "provider": "custom", "model": "main-vision",
            "base_url": "https://main.example/v1", "api_key": "main-key",
            "api_mode": "chat_completions",
        }
        config = {
            "model": {
                "provider": primary["provider"], "default": primary["model"],
                "supports_vision": True,
            },
            "auxiliary": {"vision": {
                "provider": primary["provider"] if policy == "explicit-config" else "auto",
                "model": primary["model"],
                # A duplicate of the failed route must be skipped using its concrete identity.
                "fallback_chain": [primary] + ([] if task_fallback == "absent" else [task_entry]),
            }},
            "fallback_providers": [primary, main_entry],
        }
        (aux.get_hermes_home() / "config.yaml").write_text(json.dumps(config))
        monkeypatch.setenv("OPENROUTER_API_KEY", "primary-key")
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": "https://image.example/photo.png"}},
        ]}]
        original_messages = deepcopy(messages)
        route_info = {}
        requests = []

        def respond(request):
            payload = json.loads(request.content)
            requests.append((request, payload, dict(route_info)))
            if request.url.host == "openrouter.ai":
                return httpx.Response(429, request=request, json={"error": {
                    "message": "Rate limit exceeded", "type": "rate_limit_error",
                }})
            if request.url.host == "task.example" and task_fallback == "stale":
                return httpx.Response(401, request=request, json={"error": {
                    "message": "Invalid API key", "type": "authentication_error",
                }})
            assert request.url.host in {"task.example", "main.example"}
            return httpx.Response(200, request=request, json={
                "id": "vision-result", "object": "chat.completion", "created": 0,
                "model": payload["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "image described"},
                             "finish_reason": "stop"}],
            })

        async def async_send(client, request, **kwargs):
            return respond(request)

        # Mock only HTTP I/O: config, provider routing, SDKs, retry, quarantine and both chains run.
        monkeypatch.setattr(httpx.Client, "send", lambda client, request, **kwargs: respond(request))
        monkeypatch.setattr(httpx.AsyncClient, "send", async_send)
        kwargs: dict = dict(
            task="vision", messages=messages, route_info=route_info,
            provider=(primary["provider"] if policy == "explicit-argument" else
                      None if policy == "explicit-config" else policy),
            max_tokens=73, temperature=0.2, timeout=19,
        )
        auto_policy = policy in {"auto", "AUTO", " auto "}
        succeeds = task_fallback == "healthy" or (auto_policy and task_fallback in {"absent", "stale"})
        if succeeds:
            result = await async_call_llm(**kwargs) if async_mode else call_llm(**kwargs)
            assert result.choices[0].message.content == "image described"
        else:
            with pytest.raises(openai.RateLimitError):
                if async_mode:
                    await async_call_llm(**kwargs)
                else:
                    call_llm(**kwargs)

        expected = [("openrouter.ai", primary["model"], "openrouter", "primary-key")] * 2
        if task_fallback != "absent":
            expected.append(("task.example", task_entry["model"], "custom", "task-key"))
        if auto_policy and task_fallback in {"absent", "stale"}:
            expected.append(("main.example", main_entry["model"], "custom", "main-key"))
        assert len(requests) == len(expected)
        for (request, payload, observed_route), (host, model, provider, key) in zip(requests, expected):
            assert request.url.host == host
            assert request.url.path.endswith("/chat/completions")
            assert request.headers["authorization"] == f"Bearer {key}"
            assert payload["model"] == model
            assert payload["messages"] == original_messages
            # Native shaping budgets OpenRouter; custom vision routes omit the cap.
            assert payload.get("max_completion_tokens", payload.get("max_tokens")) == (
                73 if provider == "openrouter" else None
            )
            assert payload["temperature"] == 0.2
            assert observed_route == {"provider": provider, "model": model}
        assert messages == original_messages
        assert route_info == {"provider": expected[-1][2], "model": expected[-1][1]}
        if task_fallback == "stale":
            assert aux._is_provider_unhealthy("custom", task_entry["base_url"])
            assert not aux._is_provider_unhealthy("custom", main_entry["base_url"])


class TestVisionAutoSkipsKimiCoding:
    """_resolve_auto_route vision branch skips providers that have no vision on
    their main endpoint (e.g. Kimi Coding Plan /coding) and falls through
    to the aggregator chain instead of handing back a client that will 404
    on every request (#17076).
    """

    def test_kimi_coding_skipped_falls_through_to_openrouter(self, monkeypatch):
        """kimi-coding as main + vision auto → OpenRouter (not kimi)."""
        fake_or_client = MagicMock(name="openrouter_client")

        monkeypatch.setattr(
            "agent.auxiliary_client._read_main_provider", lambda: "kimi-coding",
        )
        monkeypatch.setattr(
            "agent.auxiliary_client._read_main_model", lambda: "kimi-code",
        )
        # Guard: if the skip doesn't fire, _resolve_strict_vision_backend
        # and resolve_provider_client both would try kimi-coding — detect
        # either via the main-provider call and fail loud.
        rpc_mock = MagicMock(side_effect=AssertionError(
            "resolve_provider_client should NOT be called for kimi-coding "
            "on the vision auto path"))
        monkeypatch.setattr(
            "agent.auxiliary_client.resolve_provider_client", rpc_mock,
        )

        def fake_strict(provider, model=None):
            if provider == "openrouter":
                return fake_or_client, "google/gemini-3-flash-preview"
            if provider == "nous":
                return None, None
            raise AssertionError(
                f"strict vision backend should not be called for {provider!r} "
                "when main provider is kimi-coding"
            )
        monkeypatch.setattr(
            "agent.auxiliary_client._resolve_strict_vision_backend",
            fake_strict,
        )

        provider, client, model = resolve_vision_provider_client()
        assert provider == "openrouter"
        assert client is fake_or_client
        assert model == "google/gemini-3-flash-preview"

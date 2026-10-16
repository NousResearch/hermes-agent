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
        succeeds = auto_policy or task_fallback == "healthy"
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
        if auto_policy and task_fallback != "healthy":
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


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
async def test_auto_vision_skips_text_only_main_fallback_and_uses_later_vision_model(monkeypatch, async_mode):
    """A text-only main-chain entry must not consume or abort an image fallback."""
    import httpx

    from agent import auxiliary_client as aux

    text_entry = {
        "provider": "custom", "model": "text-only", "base_url": "https://text.example/v1",
        "api_key": "text-key",
    }
    vision_entry = {
        "provider": "custom", "model": "vision-ok", "base_url": "https://vision.example/v1",
        "api_key": "vision-key",
    }
    config = {
        "model": {
            "provider": "openrouter",
            "default": "openai/gpt-4o",
            "supports_vision": True,
        },
        "providers": {"custom": {"models": {
            "text-only": {"supports_vision": False},
            "vision-ok": {"supports_vision": True},
        }}},
        "auxiliary": {"vision": {"provider": "auto", "model": "openai/gpt-4o"}},
        "fallback_providers": [text_entry, vision_entry],
    }
    (aux.get_hermes_home() / "config.yaml").write_text(json.dumps(config))
    monkeypatch.setenv("OPENROUTER_API_KEY", "primary-key")
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "https://image.example/photo.png"}},
    ]}]
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append((request.url.host, payload["model"], payload["messages"]))
        if request.url.host == "openrouter.ai":
            return httpx.Response(429, request=request, json={"error": {
                "message": "Rate limit exceeded", "type": "rate_limit_error",
            }})
        if request.url.host == "text.example":
            return httpx.Response(400, request=request, json={"error": {
                "message": "Image input unsupported", "type": "invalid_request_error",
            }})
        assert request.url.host == "vision.example"
        assert request.headers["authorization"] == "Bearer vision-key"
        return httpx.Response(200, request=request, json={
            "id": "vision-result", "object": "chat.completion", "created": 0,
            "model": payload["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "image described"},
                         "finish_reason": "stop"}],
        })

    async def async_send(client, request, **kwargs):
        return respond(request)

    monkeypatch.setattr(httpx.Client, "send", lambda client, request, **kwargs: respond(request))
    monkeypatch.setattr(httpx.AsyncClient, "send", async_send)
    route_info = {}
    kwargs: dict = dict(task="vision", messages=messages, route_info=route_info, provider="auto")
    result = await async_call_llm(**kwargs) if async_mode else call_llm(**kwargs)
    assert result.choices[0].message.content == "image described"
    assert requests == [
        ("openrouter.ai", "openai/gpt-4o", messages),
        ("openrouter.ai", "openai/gpt-4o", messages),
        ("vision.example", "vision-ok", messages),
    ]
    assert route_info == {"provider": "custom", "model": "vision-ok"}


def test_main_fallback_vision_candidate_ignores_active_model_capability(monkeypatch):
    """The active model's text-only override must not hide an eligible fallback candidate."""
    from agent import auxiliary_client as aux

    vision_entry = {
        "provider": "custom",
        "model": "vision-ok",
        "base_url": "https://vision.example/v1",
        "api_key": "vision-key",
    }
    config = {
        "model": {"provider": "openrouter", "default": "text-main", "supports_vision": False},
        "providers": {"custom": {"models": {"vision-ok": {"supports_vision": True}}}},
        "fallback_providers": [vision_entry],
    }
    (aux.get_hermes_home() / "config.yaml").write_text(json.dumps(config))
    client = object()
    monkeypatch.setattr(aux, "_resolve_fallback_entry", lambda entry: (client, entry["model"]))
    monkeypatch.setattr(aux, "_is_provider_unhealthy", lambda *_args, **_kwargs: False)

    resolved, model, provider = aux._try_main_fallback_chain(
        "vision", failed_provider="openrouter", failed_model="text-main", reason="capacity"
    )

    assert (resolved, model, provider) == (client, "vision-ok", "custom")

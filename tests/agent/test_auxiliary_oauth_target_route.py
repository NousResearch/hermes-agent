"""The auxiliary request's wire policy belongs to the model it actually sends.

``anthropic_oauth_proxy`` is resolved per provider AND per model, and it decides Bearer vs
``x-api-key``, the Claude Code OAuth beta and the ``x-claude-code-session-id`` affinity header.
Two seams can make those disagree with the model in the request body:

* ``auxiliary.<task>: {provider: auto, model: B}`` while the main session runs model A on the same
  relay: the transport must be built for B (under the relay's own name, at the live endpoint with
  the live key), not for A with B substituted afterwards.
* a warm cached client after ``config.yaml`` flips B's declaration: auth/transforms are frozen in
  the client while the affinity header is decided per request, so the cache identity must follow
  the target route's resolved policy.

Everything here runs the public ``call_llm``/``async_call_llm`` against a real config in a temp
HERMES_HOME and asserts what reaches HTTPX.
"""

import json

import httpx
import pytest

URL = "https://relay.example.com"
KEY = "opaque-relay-key"
MODEL_A = "claude-sonnet-4-6"  # the main session's model
MODEL_B = "claude-haiku-4-6"  # the auxiliary task's model, declared opposite to A
SESSION = "20261007_120000_relay"
MESSAGES = [{"role": "user", "content": "hello"}]


def _config(main_enabled: bool, b_enabled: bool, aux_model: "str | None" = MODEL_B) -> dict:
    task = {"provider": "auto", **({"model": aux_model} if aux_model else {})}
    return {
        "model": {"provider": "custom:relay", "default": MODEL_A},
        "providers": {
            "relay": {
                "api": URL,
                "key_env": "TEST_RELAY_KEY",
                "transport": "anthropic_messages",
                "capabilities": {"anthropic_oauth_proxy": main_enabled},
                "models": {MODEL_B: {"anthropic_oauth_proxy": b_enabled}},
            },
        },
        "auxiliary": {"compression": task},
    }


def _write(home, cfg) -> None:
    """The config writer the CLI uses; ``load_config`` sees the new file signature."""
    from hermes_cli.config import atomic_config_write
    atomic_config_write(home / "config.yaml", cfg)


def _main(main_enabled: bool) -> dict:
    """The live main runtime: anonymous ``custom`` on the relay, named by requested_provider."""
    return {
        "provider": "custom",
        "requested_provider": "custom:relay",
        "base_url": URL,
        "api_key": KEY,
        "api_mode": "anthropic_messages",
        "model": MODEL_A,
        "capabilities": {"anthropic_oauth_proxy": main_enabled},
        "session_id": SESSION,
    }


@pytest.fixture
def relay(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TEST_RELAY_KEY", KEY)
    for stale in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_TOKEN"):
        monkeypatch.delenv(stale, raising=False)
    requests = []

    def send(client, request, **kwargs):
        requests.append(request)
        body = json.loads(request.content or b"{}")
        message = {
            "id": "msg_test", "type": "message", "role": "assistant",
            "model": body.get("model", MODEL_A),
            "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        if not body.get("stream"):
            return httpx.Response(200, request=request, json=message)
        events = [
            {"type": "message_start", "message": dict(message, content=[])},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
             "usage": {"output_tokens": 1}},
            {"type": "message_stop"},
        ]
        data = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
        return httpx.Response(200, request=request, content=data,
                              headers={"content-type": "text/event-stream"})

    async def async_send(client, request, **kwargs):
        return send(client, request, **kwargs)

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(httpx.AsyncClient, "send", async_send)
    from agent.auxiliary_client import shutdown_cached_clients
    shutdown_cached_clients()
    yield tmp_path, requests
    shutdown_cached_clients()


def assert_wire(request, model: str, oauth: bool) -> None:
    """The request carries *model*, and exactly that model's declared wire policy."""
    from agent.claude_code_session import CLAUDE_CODE_SESSION_HEADER
    assert request.url.host == "relay.example.com"
    assert json.loads(request.content)["model"] == model
    assert (request.headers.get("authorization") == f"Bearer {KEY}") is oauth
    assert (request.headers.get("x-api-key") == KEY) is (not oauth)
    assert ("oauth-2025-04-20" in request.headers.get("anthropic-beta", "")) is oauth
    assert (request.headers.get(CLAUDE_CODE_SESSION_HEADER) == SESSION) is oauth


# ── F1: provider auto + a model override builds the transport for the override ──

@pytest.mark.parametrize("main_enabled", [True, False], ids=["A-true-B-false", "A-false-B-true"])
def test_auto_model_override_sends_the_target_models_policy(relay, main_enabled):
    from agent.auxiliary_client import call_llm
    home, requests = relay
    _write(home, _config(main_enabled, not main_enabled))
    for _ in range(2):  # cold construction, then the warm cached client
        call_llm(task="compression", main_runtime=_main(main_enabled), messages=MESSAGES, max_tokens=32)
        assert_wire(requests[-1], MODEL_B, oauth=not main_enabled)
    assert len(requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("main_enabled", [True, False], ids=["A-true-B-false", "A-false-B-true"])
async def test_async_auto_model_override_sends_the_target_models_policy(relay, main_enabled):
    from agent.auxiliary_client import async_call_llm
    home, requests = relay
    _write(home, _config(main_enabled, not main_enabled))
    for _ in range(2):
        await async_call_llm(task="compression", main_runtime=_main(main_enabled), messages=MESSAGES, max_tokens=32)
        assert_wire(requests[-1], MODEL_B, oauth=not main_enabled)


@pytest.mark.parametrize("main_enabled", [True, False])
def test_auto_without_override_keeps_the_main_models_policy(relay, main_enabled):
    from agent.auxiliary_client import call_llm
    home, requests = relay
    _write(home, _config(main_enabled, not main_enabled, aux_model=None))
    call_llm(task="compression", main_runtime=_main(main_enabled), messages=MESSAGES, max_tokens=32)
    assert_wire(requests[-1], MODEL_A, oauth=main_enabled)


@pytest.mark.parametrize("main_enabled", [True, False])
def test_auto_drops_an_openrouter_format_override_and_keeps_the_main_route(relay, main_enabled):
    """The upstream rule stands: ``vendor/model`` on a non-OpenRouter route falls back to the
    route's own model, and that model's policy."""
    from agent.auxiliary_client import call_llm
    home, requests = relay
    _write(home, _config(main_enabled, not main_enabled, aux_model="anthropic/" + MODEL_B))
    call_llm(task="compression", main_runtime=_main(main_enabled), messages=MESSAGES, max_tokens=32)
    assert_wire(requests[-1], MODEL_A, oauth=main_enabled)


# ── F4: a declaration change reaches a warm client exactly as a cold one ──

@pytest.mark.parametrize("provider", ["custom:relay", "auto"])
@pytest.mark.parametrize("initial", [False, True], ids=["false-to-true", "true-to-false"])
def test_declaration_change_reaches_the_warm_client(relay, initial, provider):
    from agent.auxiliary_client import call_llm, shutdown_cached_clients
    home, requests = relay
    main = _main(True)  # fixed main runtime on A throughout

    def send():
        call_llm(task="compression", provider=provider, model=MODEL_B, main_runtime=main,
                 messages=MESSAGES, max_tokens=32)
        return requests[-1]

    _write(home, _config(True, initial))
    assert_wire(send(), MODEL_B, oauth=initial)
    _write(home, _config(True, not initial))
    assert_wire(send(), MODEL_B, oauth=not initial)  # warm cache after the edit
    shutdown_cached_clients()
    assert_wire(send(), MODEL_B, oauth=not initial)  # cold construction agrees
    _write(home, _config(True, initial))
    assert_wire(send(), MODEL_B, oauth=initial)  # and back again, warm

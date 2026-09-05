"""Azure Foundry awareness must reach ``preflight_kwargs`` from the live turn paths (#63257).

The preflight call sites moved out of ``conversation_loop`` into
``agent.turn_api_request.build_api_request`` (first attempt) and
``agent.turn_api_call.perform_api_call`` (streaming retry). Unit tests that call
``preflight_kwargs`` directly cannot catch a call site that forgets to forward the
agent's provider/base_url — that is exactly how an earlier revision of this fix went
green while stripping the reasoning ``id`` on the primary path. These tests drive
the real request builders with the real ``ResponsesApiTransport`` and assert on the
wire payload that comes out.
"""

from types import SimpleNamespace

import pytest

from agent.transports.codex import ResponsesApiTransport
from agent.turn_api_request import build_api_request


_REASONING_KWARGS = {
    "model": "gpt-5.5",
    "instructions": "You are Hermes.",
    "input": [
        {
            "type": "reasoning",
            "id": "rs_live",
            "encrypted_content": "enc_blob",
            "summary": [{"type": "summary_text", "text": "brief"}],
            "status": "completed",
        },
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "ok"}],
        },
    ],
    "store": False,
}


def _agent(*, provider, base_url):
    """Minimal agent double exposing what ``build_api_request`` reads on the codex path."""
    transport = ResponsesApiTransport()
    agent = SimpleNamespace(
        provider=provider,
        base_url=base_url,
        api_mode="codex_responses",
        model="gpt-5.5",
        tools=[],
        client=SimpleNamespace(),
        session_id="s",
        platform="cli",
        max_tokens=None,
        _use_prompt_caching=False,
        _force_ascii_payload=False,
        _empty_content_retries=0,
        _is_user_initiated_turn=False,
        _last_api_first_chunk_at=None,
        _reset_stream_delivery_tracking=lambda: None,
        _reapply_reasoning_echo_for_provider=lambda msgs: None,
        _build_api_kwargs=lambda *a, **k: {k2: (list(v) if isinstance(v, list) else v)
                                          for k2, v in _REASONING_KWARGS.items()},
        _get_transport=lambda: transport,
        _is_copilot_url=lambda: False,
        _is_codex_backend=lambda: False,
        _is_openrouter_url=lambda: False,
        _api_request_payload_for_hook=lambda kw: kw,
        _dump_api_request_debug=lambda *a, **k: None,
        _pending_redirect=None,
        _has_pending_redirect=lambda: False,
    )
    return agent


def _build(agent):
    result = build_api_request(
        agent, api_messages=[{"role": "user", "content": "hi"}], _moa_prepared_request=None,
        tools_for_api=agent.tools, system_message="You are Hermes.", messages=[],
        original_user_message="hi", approx_tokens=1, total_chars=2, retry_count=0,
        api_call_count=1, api_request_id="r1", api_start_time=0.0, effective_task_id="t",
        turn_id="turn",
    )
    return result.api_kwargs


def _reasoning(kwargs):
    return next(i for i in kwargs["input"] if i.get("type") == "reasoning")


def _assistant_text_part(kwargs):
    msg = next(i for i in kwargs["input"] if i.get("type") == "message")
    return msg["content"][0]


@pytest.mark.parametrize(
    "provider,base_url",
    [
        # Host-detected, provider unset (custom endpoint pointing at Foundry).
        (None, "https://r.services.ai.azure.com/openai/v1"),
        (None, "https://r.openai.azure.com/openai/v1"),
        # Provider-detected behind a proxy: the registered azure-foundry provider
        # must be honoured even when the URL does not look like Azure.
        ("azure-foundry", "https://gateway.corp.example/v1"),
        ("Azure-Foundry", None),
    ],
)
def test_first_attempt_preflight_keeps_foundry_wire_shape(provider, base_url):
    kwargs = _build(_agent(provider=provider, base_url=base_url))
    assert _reasoning(kwargs)["id"] == "rs_live"
    assert _assistant_text_part(kwargs)["annotations"] == []


@pytest.mark.parametrize(
    "provider,base_url",
    [
        ("openai-codex", "https://chatgpt.com/backend-api/codex"),
        ("openai", "https://api.openai.com/v1"),
        ("copilot", "https://api.githubcopilot.com"),
        ("xai", "https://api.x.ai/v1"),
        # Look-alikes: hostname-aware matching must not fire on path/suffix hits.
        ("custom", "https://evil.com/services.ai.azure.com/v1"),
        ("custom", "https://openai.azure.com.evil.net/v1"),
    ],
)
def test_first_attempt_preflight_leaves_non_foundry_untouched(provider, base_url):
    kwargs = _build(_agent(provider=provider, base_url=base_url))
    assert "id" not in _reasoning(kwargs)
    assert "annotations" not in _assistant_text_part(kwargs)


def test_streaming_retry_preflight_forwards_the_same_azure_context(monkeypatch):
    """``perform_api_call`` re-preflights ``next_api_kwargs`` before streaming; it must
    forward the identical provider/base_url context as the first attempt."""
    from agent import turn_api_call

    class _Stop(Exception):
        pass

    seen = {}
    transport = ResponsesApiTransport()
    real_preflight = transport.preflight_kwargs

    def spy(api_kwargs, **kw):
        seen.update(kw)
        raise _Stop()

    transport.preflight_kwargs = spy
    agent = _agent(provider="azure-foundry", base_url="https://gateway.corp.example/v1")
    agent._get_transport = lambda: transport
    monkeypatch.setattr(turn_api_call, "_should_stream", lambda a: True)

    import inspect
    sig = inspect.signature(turn_api_call.perform_api_call)
    call_kwargs = {name: None for name in sig.parameters if name != "agent"}
    call_kwargs["api_kwargs"] = dict(_REASONING_KWARGS)
    call_kwargs["_original_api_kwargs"] = dict(_REASONING_KWARGS)
    call_kwargs["_llm_middleware_trace"] = []
    call_kwargs["interrupted"] = False
    with pytest.raises(_Stop):
        turn_api_call.perform_api_call(agent, **call_kwargs)

    assert seen["is_azure_foundry"] is True
    assert seen["provider"] == "azure-foundry"
    assert seen["base_url"] == "https://gateway.corp.example/v1"
    # And the real preflight with that context keeps the id.
    assert _reasoning(real_preflight(dict(_REASONING_KWARGS), **seen))["id"] == "rs_live"

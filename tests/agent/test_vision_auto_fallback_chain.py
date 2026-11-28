"""An auto-routed vision call walks the main ``fallback_providers`` chain on a billing error.

``resolve_vision_provider_client("auto")`` returns the concrete backend it picked. That label must not
replace ``resolved_provider="auto"``: the recovery ladder decides whether a route is auto from
``resolved_provider``, and an overwritten label made an Anthropic credit-exhaustion 400 on the
auto-picked main provider skip the main fallback chain (only the main-agent model was tried, which is
the same exhausted provider). Text tasks already keep ``"auto"``; vision now does the same.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from agent import auxiliary_client as aux

_CREDIT_400 = (
    "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': "
    "'Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to "
    "upgrade or purchase credits.'}}"
)
_IMAGE_MESSAGES = [{"role": "user", "content": [
    {"type": "text", "text": "describe"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
]}]


class _Credit400(Exception):
    status_code = 400

    def __init__(self):
        super().__init__(_CREDIT_400)


def _response(text):
    msg = MagicMock(content=text, tool_calls=None, reasoning=None, reasoning_content=None,
                    reasoning_details=None)
    return MagicMock(choices=[MagicMock(message=msg, finish_reason="stop")], usage=None)


def _patches(resolved_provider, primary, fallback):
    return (
        patch.object(aux, "_resolve_task_provider_model",
                     return_value=(resolved_provider, None, None, None, None)),
        patch.object(aux, "resolve_vision_provider_client",
                     return_value=("anthropic", primary, "claude-main")),
        patch.object(aux, "_try_configured_fallback_chain", return_value=(None, None, "")),
        patch.object(aux, "_try_main_fallback_chain",
                     return_value=(fallback, "fallback-vision-model", "fallback-provider")),
        patch.object(aux, "_try_main_agent_model_fallback", return_value=(None, None, "")),
        patch.object(aux, "_try_payment_fallback", return_value=(None, None, "")),
        patch.object(aux, "_mark_provider_unhealthy"),
    )


def _run(patches, call):
    ctx = [p.__enter__() for p in patches]
    try:
        return call(), ctx
    finally:
        for p in reversed(patches):
            p.__exit__(None, None, None)


def test_auto_vision_billing_400_walks_main_fallback_chain():
    primary = MagicMock(base_url="https://api.anthropic.com")
    primary.chat.completions.create.side_effect = _Credit400()
    fallback = MagicMock(base_url="https://fallback.example")
    fallback.chat.completions.create.return_value = _response("fallback described it")

    result, ctx = _run(_patches("auto", primary, fallback),
                       lambda: aux.call_llm(task="vision", messages=_IMAGE_MESSAGES))

    assert result.choices[0].message.content == "fallback described it"
    main_chain, main_agent = ctx[3], ctx[4]
    assert main_chain.called
    assert not main_agent.called
    assert fallback.chat.completions.create.call_count == 1


def test_auto_vision_billing_400_walks_main_fallback_chain_async():
    primary = MagicMock(base_url="https://api.anthropic.com")
    primary.chat.completions.create = AsyncMock(side_effect=_Credit400())
    fallback = MagicMock(base_url="https://fallback.example")
    fallback.chat.completions.create = AsyncMock(return_value=_response("fallback described it"))

    result, ctx = _run(_patches("auto", primary, fallback),
                       lambda: asyncio.run(aux.async_call_llm(task="vision", messages=_IMAGE_MESSAGES)))

    assert result.choices[0].message.content == "fallback described it"
    assert ctx[3].called and not ctx[4].called


def test_explicit_vision_provider_keeps_its_route_label():
    primary = MagicMock(base_url="https://api.anthropic.com")
    with patch.object(aux, "resolve_vision_provider_client", return_value=("anthropic", primary, "m")):
        _, _, explicit, _ = aux._resolve_call_client(
            "vision", provider=None, model=None, base_url=None, api_key=None,
            resolved_provider="anthropic", resolved_model=None, resolved_base_url=None,
            resolved_api_key=None, resolved_api_mode=None, main_runtime=None, async_mode=False)
        _, _, auto, effective = aux._resolve_call_client(
            "vision", provider=None, model=None, base_url=None, api_key=None,
            resolved_provider="auto", resolved_model=None, resolved_base_url=None,
            resolved_api_key=None, resolved_api_mode=None, main_runtime=None, async_mode=False)
    assert explicit == "anthropic"
    assert auto == "auto" and effective == "anthropic"

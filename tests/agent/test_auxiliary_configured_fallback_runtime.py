"""Runtime regressions for configured auxiliary fallback chains."""

from copy import deepcopy
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.auxiliary_client import (
    _is_recoverable_aux_fallback_error,
    async_call_llm,
    call_llm,
)


_PROVIDER = "custom:test-gateway"
_PRIMARY_MODEL = "vision-primary"
_MIDDLE_MODEL = "vision-middle"
_FINAL_MODEL = "vision-final"


class _HttpStatusError(Exception):
    """SDK-like error exposing the status only on ``response``."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.response = SimpleNamespace(status_code=status_code)


def _response(text: str = "fallback succeeded") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _vision_messages() -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
                },
            ],
        }
    ]


def _fallback_config() -> dict:
    return {
        "fallback_chain": [
            {"provider": _PROVIDER, "model": _MIDDLE_MODEL},
            {"provider": _PROVIDER, "model": _FINAL_MODEL},
        ]
    }


def _sync_client(*, side_effect=None, result=None) -> MagicMock:
    client = MagicMock()
    client.api_key = "test-key"
    client.base_url = "https://example.invalid/v1"
    if side_effect is not None:
        client.chat.completions.create.side_effect = side_effect
    else:
        client.chat.completions.create.return_value = result or _response()
    return client


def _async_client(*, side_effect=None, result=None) -> MagicMock:
    client = MagicMock()
    client.api_key = "test-key"
    client.base_url = "https://example.invalid/v1"
    client.chat.completions.create = AsyncMock(
        side_effect=side_effect,
        return_value=result or _response(),
    )
    return client


def _common_patches(primary, middle_sync, final_sync):
    clients = {
        _MIDDLE_MODEL: middle_sync,
        _FINAL_MODEL: final_sync,
    }

    def _resolve_entry(entry):
        model = entry["model"]
        return clients[model], model

    return (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=(_PROVIDER, _PRIMARY_MODEL, None, None, None),
        ),
        patch(
            "agent.auxiliary_client.resolve_vision_provider_client",
            return_value=(_PROVIDER, primary, _PRIMARY_MODEL),
        ),
        patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value=_fallback_config(),
        ),
        patch(
            "agent.auxiliary_client._resolve_fallback_entry",
            side_effect=_resolve_entry,
        ),
    )


def test_unstructured_503_text_does_not_override_explicit_route():
    error = RuntimeError("HTTP 503 Service Unavailable")

    assert _is_recoverable_aux_fallback_error(error) is False


@pytest.mark.parametrize("status_code", [502, 503])
def test_sync_persistent_5xx_walks_configured_chain_and_preserves_image(
    status_code,
):
    server_error = _HttpStatusError(status_code, "persistent upstream failure")
    primary = _sync_client(side_effect=[server_error, server_error])
    middle = _sync_client(side_effect=server_error)
    final = _sync_client(result=_response("final vision result"))
    messages = _vision_messages()
    route_info = {}

    with ExitStack() as stack:
        for current_patch in _common_patches(primary, middle, final):
            stack.enter_context(current_patch)
        stack.enter_context(
            patch("agent.auxiliary_client._transient_retry_count", return_value=1)
        )
        stack.enter_context(
            patch("agent.auxiliary_client._TRANSIENT_RETRY_BACKOFF_BASE", 0.0)
        )
        main_fallback = stack.enter_context(
            patch("agent.auxiliary_client._try_main_agent_model_fallback")
        )
        result = call_llm(
            task="vision",
            messages=messages,
            route_info=route_info,
        )

    assert result.choices[0].message.content == "final vision result"
    assert primary.chat.completions.create.call_count == 2
    assert middle.chat.completions.create.call_count == 1
    assert final.chat.completions.create.call_count == 1
    assert middle.chat.completions.create.call_args.kwargs["messages"] == messages
    assert final.chat.completions.create.call_args.kwargs["messages"] == messages
    assert route_info == {"provider": _PROVIDER, "model": _FINAL_MODEL}
    main_fallback.assert_not_called()


@pytest.mark.asyncio
async def test_async_persistent_5xx_walks_configured_chain_and_preserves_image():
    primary_error = _HttpStatusError(502, "primary bad gateway")
    middle_error = _HttpStatusError(503, "middle service unavailable")
    primary = _async_client(
        side_effect=[primary_error, primary_error, primary_error]
    )
    middle_sync = _sync_client()
    final_sync = _sync_client()
    middle = _async_client(side_effect=middle_error)
    final = _async_client(result=_response("final async vision result"))
    messages = _vision_messages()
    messages_before = deepcopy(messages)
    route_info = {}

    async_clients = {
        middle_sync: (middle, _MIDDLE_MODEL),
        final_sync: (final, _FINAL_MODEL),
    }

    with ExitStack() as stack:
        for current_patch in _common_patches(primary, middle_sync, final_sync):
            stack.enter_context(current_patch)
        stack.enter_context(
            patch(
                "agent.auxiliary_client._to_async_client",
                side_effect=lambda client, _model, **_kw: async_clients[client],
            )
        )
        stack.enter_context(
            patch("agent.auxiliary_client._transient_retry_count", return_value=2)
        )
        stack.enter_context(
            patch("agent.auxiliary_client._TRANSIENT_RETRY_BACKOFF_BASE", 0.0)
        )
        main_fallback = stack.enter_context(
            patch("agent.auxiliary_client._try_main_agent_model_fallback")
        )
        result = await async_call_llm(
            task="vision",
            messages=messages,
            route_info=route_info,
        )

    assert result.choices[0].message.content == "final async vision result"
    # The async path retries once on the same provider (main's behaviour);
    # the configurable multi-retry loop is sync-only.
    assert primary.chat.completions.create.await_count == 2
    assert middle.chat.completions.create.await_count == 1
    assert final.chat.completions.create.await_count == 1
    assert messages == messages_before
    middle_kwargs = middle.chat.completions.create.call_args.kwargs
    final_kwargs = final.chat.completions.create.call_args.kwargs
    assert middle_kwargs["model"] == _MIDDLE_MODEL
    assert final_kwargs["model"] == _FINAL_MODEL
    assert middle_kwargs["messages"] == messages_before
    assert final_kwargs["messages"] == messages_before
    assert (
        middle_kwargs["messages"][0]["content"][1]["image_url"]["url"]
        == "data:image/png;base64,aW1hZ2U="
    )
    assert (
        final_kwargs["messages"][0]["content"][1]["image_url"]["url"]
        == "data:image/png;base64,aW1hZ2U="
    )
    assert route_info == {"provider": _PROVIDER, "model": _FINAL_MODEL}
    main_fallback.assert_not_called()


def test_deterministic_fallback_error_does_not_advance_chain():
    server_error = _HttpStatusError(502, "upstream bad gateway")
    bad_request = _HttpStatusError(400, "invalid image request")
    primary = _sync_client(side_effect=[server_error, server_error])
    middle = _sync_client(side_effect=bad_request)
    final = _sync_client(result=_response("must not run"))

    with ExitStack() as stack:
        for current_patch in _common_patches(primary, middle, final):
            stack.enter_context(current_patch)
        stack.enter_context(
            patch("agent.auxiliary_client._transient_retry_count", return_value=1)
        )
        stack.enter_context(
            patch("agent.auxiliary_client._TRANSIENT_RETRY_BACKOFF_BASE", 0.0)
        )
        with pytest.raises(_HttpStatusError, match="invalid image request"):
            call_llm(task="vision", messages=_vision_messages())

    assert middle.chat.completions.create.call_count == 1
    final.chat.completions.create.assert_not_called()


def test_exhausted_configured_chain_reraises_primary_error(caplog):
    primary_error = _HttpStatusError(502, "primary upstream bad gateway")
    primary = _sync_client(side_effect=[primary_error, primary_error])
    middle = _sync_client(
        side_effect=_HttpStatusError(503, "middle service unavailable")
    )
    final = _sync_client(
        side_effect=_HttpStatusError(504, "final gateway timeout")
    )

    with ExitStack() as stack:
        for current_patch in _common_patches(primary, middle, final):
            stack.enter_context(current_patch)
        stack.enter_context(
            patch("agent.auxiliary_client._transient_retry_count", return_value=1)
        )
        stack.enter_context(
            patch("agent.auxiliary_client._TRANSIENT_RETRY_BACKOFF_BASE", 0.0)
        )
        stack.enter_context(
            patch(
                "agent.auxiliary_client._try_main_agent_model_fallback",
                return_value=(None, None, ""),
            )
        )
        with pytest.raises(_HttpStatusError) as exc_info:
            call_llm(task="vision", messages=_vision_messages())

    assert exc_info.value is primary_error
    assert middle.chat.completions.create.call_count == 1
    assert final.chat.completions.create.call_count == 1
    assert "all fallbacks exhausted" in caplog.text

"""Auxiliary-only SDK clients with final request validation."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, TYPE_CHECKING

import httpx

from agent.process_bootstrap import _OpenAIProxy
from agent.auxiliary_request_requirements import validate_sdk_request

if TYPE_CHECKING:
    from openai import OpenAI, AsyncOpenAI
    from anthropic import Anthropic


@lru_cache(maxsize=1)
def _guarded_openai_client_classes() -> tuple[type[OpenAI], type[AsyncOpenAI]]:
    from openai import AsyncOpenAI, OpenAI

    class RequestValidatedOpenAI(OpenAI):
        HERMES_REQUEST_GUARD = True

        def _prepare_request(self, request: httpx.Request) -> None:
            super()._prepare_request(request)
            validate_sdk_request(request, self.base_url)

    class RequestValidatedAsyncOpenAI(AsyncOpenAI):
        HERMES_REQUEST_GUARD = True

        async def _prepare_request(self, request: httpx.Request) -> None:
            await super()._prepare_request(request)
            validate_sdk_request(request, self.base_url)

    return RequestValidatedOpenAI, RequestValidatedAsyncOpenAI


def openai_request_client_classes() -> tuple[type[OpenAI], type[AsyncOpenAI]]:
    from agent.auxiliary_request_requirements import request_requirements_active

    if request_requirements_active():
        return _guarded_openai_client_classes()
    from openai import AsyncOpenAI, OpenAI

    return OpenAI, AsyncOpenAI


class AuxiliaryOpenAIProxy(_OpenAIProxy):
    __slots__ = ()

    def __call__(self, *args: Any, **kwargs: Any) -> OpenAI:
        return openai_request_client_classes()[0](*args, **kwargs)


@lru_cache(maxsize=1)
def anthropic_request_client_class() -> type[Anthropic]:
    from anthropic import Anthropic

    class RequestValidatedAnthropic(Anthropic):
        HERMES_REQUEST_GUARD = True

        def _prepare_request(self, request: httpx.Request) -> None:
            super()._prepare_request(request)
            validate_sdk_request(request, self.base_url)

    return RequestValidatedAnthropic


def _new_anthropic_request_client(*args: Any, **kwargs: Any) -> Anthropic:
    return anthropic_request_client_class()(*args, **kwargs)


def build_request_anthropic_client(
    api_key: str | Callable[[], str], base_url: str | None = None
) -> Anthropic:
    from agent.anthropic_adapter import build_anthropic_client

    from agent.auxiliary_request_requirements import request_requirements_active

    if request_requirements_active():
        return build_anthropic_client(
            api_key, base_url, client_factory=_new_anthropic_request_client
        )
    return build_anthropic_client(api_key, base_url)


def validate_client_dispatch(client: Any) -> None:
    from agent.auxiliary_request_requirements import (
        _REQUIRED_REQUEST,
        AuxiliaryRequestRequirementError,
        RequestRequirement,
        RequirementFailure,
    )

    required = _REQUIRED_REQUEST.get()
    if required is None or not (required.tools or required.max_tokens is not None):
        return
    if getattr(type(client), "HERMES_REQUEST_GUARD", False) is True:
        return
    from agent.gemini_native_adapter import GeminiNativeClient, AsyncGeminiNativeClient
    from agent.auxiliary_client import (
        CodexAuxiliaryClient,
        AnthropicAuxiliaryClient,
        BedrockAuxiliaryClient,
        AsyncBedrockAuxiliaryClient,
        _AsyncAuxiliaryClientBase,
    )

    if isinstance(
        client,
        (
            GeminiNativeClient,
            AsyncGeminiNativeClient,
            BedrockAuxiliaryClient,
            AsyncBedrockAuxiliaryClient,
        ),
    ):
        return
    if isinstance(client, (CodexAuxiliaryClient, AnthropicAuxiliaryClient)):
        validate_client_dispatch(client._real_client)
        return
    if isinstance(client, _AsyncAuxiliaryClientBase):
        validate_client_dispatch(client.chat.completions._sync._client)
        return
    requirement = (
        RequestRequirement.STRICT_TOOLS
        if required.tools
        else RequestRequirement.MAX_TOKENS
    )
    raise AuxiliaryRequestRequirementError(
        requirement, RequirementFailure.UNSUPPORTED_ROUTE
    )

"""Auxiliary OpenAI clients with validation before SDK model construction."""

import httpx
from openai import AsyncOpenAI, OpenAI
from openai._types import ResponseT

from agent.native_usage import validate_openai_usage


class UsageValidatedOpenAI(OpenAI):
    def _process_response_data(
        self, *, data: object, cast_to: type[ResponseT], response: httpx.Response
    ) -> ResponseT:
        validate_openai_usage(
            data,
            kind="codex" if getattr(cast_to, "__name__", "") == "Response" else "chat",
        )
        return super()._process_response_data(
            data=data, cast_to=cast_to, response=response
        )


class UsageValidatedAsyncOpenAI(AsyncOpenAI):
    def _process_response_data(
        self, *, data: object, cast_to: type[ResponseT], response: httpx.Response
    ) -> ResponseT:
        validate_openai_usage(
            data,
            kind="codex" if getattr(cast_to, "__name__", "") == "Response" else "chat",
        )
        return super()._process_response_data(
            data=data, cast_to=cast_to, response=response
        )

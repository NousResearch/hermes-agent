"""Auxiliary Anthropic client with raw stream and response usage validation."""

import httpx
from anthropic import Anthropic
from anthropic._types import ResponseT

from agent.native_usage import validate_anthropic_usage


class UsageValidatedAnthropic(Anthropic):
    def _process_response_data(
        self, *, data: object, cast_to: type[ResponseT], response: httpx.Response
    ) -> ResponseT:
        validate_anthropic_usage(data)
        return super()._process_response_data(
            data=data, cast_to=cast_to, response=response
        )


def build_auxiliary_anthropic_client(api_key, base_url=None):
    from agent.anthropic_adapter import build_anthropic_client

    return build_anthropic_client(
        api_key, base_url, client_factory=UsageValidatedAnthropic
    )

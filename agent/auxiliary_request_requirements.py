"""Required controls for one auxiliary invocation and every native attempt."""

from __future__ import annotations

import contextvars
import copy
import json
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator, Optional
from urllib.parse import urlparse


class RequestRequirement(str, Enum):
    STRICT_TOOLS = "strict_tools"
    MAX_TOKENS = "max_tokens"


class RequirementFailure(str, Enum):
    MISSING_VALUE = "missing_value"
    UNSUPPORTED_ROUTE = "unsupported_route"
    TOOL_SCHEMA_CHANGED = "tool_schema_changed"
    OUTPUT_CAP_MISSING = "output_cap_missing"
    OUTPUT_CAP_EXCEEDED = "output_cap_exceeded"


class AuxiliaryRequestRequirementError(ValueError):
    """A native attempt cannot preserve a required request control."""

    def __init__(self, requirement: RequestRequirement, cause: RequirementFailure):
        self.requirement = requirement
        self.cause = cause
        super().__init__(
            f"Auxiliary request requirement failed: {requirement.value}/{cause.value}"
        )


@dataclass(frozen=True)
class AuxiliaryRequestRequirements:
    """Require existing strict schemas or the explicit output-token ceiling.

    Strict tools currently require the direct OpenAI inference wire. Other
    native routes reject that requirement before dispatch. A required output
    ceiling may be lowered, but cannot be omitted or enlarged.
    """

    strict_tools: bool = False
    max_tokens: bool = False


@dataclass(frozen=True)
class _RequiredRequest:
    tools: tuple[dict[str, Any], ...]
    max_tokens: int | None


_REQUIRED_REQUEST: contextvars.ContextVar[_RequiredRequest | None] = (
    contextvars.ContextVar(
        "auxiliary_required_request",
        default=None,
    )
)


def request_requirements_active() -> bool:
    required = _REQUIRED_REQUEST.get()
    return required is not None and bool(
        required.tools or required.max_tokens is not None
    )


def required_output_cap() -> int | None:
    request = _REQUIRED_REQUEST.get()
    return request.max_tokens if request is not None else None


@contextmanager
def request_requirements_scope(
    requirements: AuxiliaryRequestRequirements | None,
    tools: list | None,
    max_tokens: int | None,
) -> Iterator[None]:
    required = None
    if requirements is not None:
        strict_tools = tuple(
            copy.deepcopy(tool["function"])
            for tool in (tools if isinstance(tools, list) else [])
            if isinstance(tool, dict)
            and isinstance(tool.get("function"), dict)
            and tool["function"].get("strict") is True
        )
        if requirements.strict_tools and not strict_tools:
            raise AuxiliaryRequestRequirementError(
                RequestRequirement.STRICT_TOOLS, RequirementFailure.MISSING_VALUE
            )
        if requirements.max_tokens and (type(max_tokens) is not int or max_tokens <= 0):
            raise AuxiliaryRequestRequirementError(
                RequestRequirement.MAX_TOKENS, RequirementFailure.MISSING_VALUE
            )
        required = _RequiredRequest(
            strict_tools if requirements.strict_tools else (),
            max_tokens if requirements.max_tokens else None,
        )
    token = _REQUIRED_REQUEST.set(required)
    try:
        yield
    finally:
        _REQUIRED_REQUEST.reset(token)


def validate_native_request(
    payload: dict[str, Any], wire: str, endpoint: str = ""
) -> None:
    required = _REQUIRED_REQUEST.get()
    if required is None:
        return
    if not isinstance(payload, dict):
        requirement = (
            RequestRequirement.STRICT_TOOLS
            if required.tools
            else RequestRequirement.MAX_TOKENS
        )
        raise AuxiliaryRequestRequirementError(
            requirement, RequirementFailure.UNSUPPORTED_ROUTE
        )
    if required.tools:
        if (
            wire not in {"chat", "responses"}
            or urlparse(endpoint).hostname != "api.openai.com"
            or urlparse(endpoint).scheme != "https"
        ):
            raise AuxiliaryRequestRequirementError(
                RequestRequirement.STRICT_TOOLS, RequirementFailure.UNSUPPORTED_ROUTE
            )
        if (
            str(payload.get("model", "")).startswith("ft:")
            and payload.get("parallel_tool_calls") is not False
        ):
            raise AuxiliaryRequestRequirementError(
                RequestRequirement.STRICT_TOOLS, RequirementFailure.UNSUPPORTED_ROUTE
            )
        native_tools = payload.get("tools")
        if not isinstance(native_tools, list):
            raise AuxiliaryRequestRequirementError(
                RequestRequirement.STRICT_TOOLS, RequirementFailure.TOOL_SCHEMA_CHANGED
            )
        functions = [
            tool.get("function") if wire == "chat" else tool
            for tool in native_tools
            if isinstance(tool, dict)
        ]
        functions = [function for function in functions if isinstance(function, dict)]
        for original in required.tools:
            matching = [
                tool for tool in functions if tool.get("name") == original.get("name")
            ]
            if (
                len(matching) != 1
                or matching[0].get("strict") is not True
                or matching[0].get("parameters") != original.get("parameters")
            ):
                raise AuxiliaryRequestRequirementError(
                    RequestRequirement.STRICT_TOOLS,
                    RequirementFailure.TOOL_SCHEMA_CHANGED,
                )
    if required.max_tokens is None:
        return
    container = payload
    if wire in {"gemini", "converse"}:
        nested = payload.get(
            "generationConfig" if wire == "gemini" else "inferenceConfig"
        )
        container = nested if isinstance(nested, dict) else {}
    keys = {
        "chat": ("max_tokens", "max_completion_tokens"),
        "responses": ("max_output_tokens",),
        "messages": ("max_tokens",),
        "gemini": ("maxOutputTokens",),
        "converse": ("maxTokens",),
    }
    caps = [container[key] for key in keys.get(wire, ()) if key in container]
    if not caps or any(type(cap) is not int or cap <= 0 for cap in caps):
        raise AuxiliaryRequestRequirementError(
            RequestRequirement.MAX_TOKENS, RequirementFailure.OUTPUT_CAP_MISSING
        )
    if any(cap > required.max_tokens for cap in caps):
        raise AuxiliaryRequestRequirementError(
            RequestRequirement.MAX_TOKENS, RequirementFailure.OUTPUT_CAP_EXCEEDED
        )


def validate_sdk_request(request: Any, base_url: Any) -> None:
    if _REQUIRED_REQUEST.get() is None:
        return
    if request.method != "POST":
        return
    path = request.url.path.rstrip("/")
    prefix = urlparse(str(base_url)).path.rstrip("/")
    wires = {
        "chat/completions": "chat",
        "responses": "responses",
        "v1/messages": "messages",
    }
    wire = next(
        (value for suffix, value in wires.items() if path == prefix + "/" + suffix),
        None,
    )
    if wire is not None:
        validate_native_request(json.loads(request.content), wire, str(request.url))


def guard_converse_send(request: Any, **kwargs: Any) -> None:
    if _REQUIRED_REQUEST.get() is not None:
        validate_native_request(json.loads(request.body), "converse", request.url)


def _forwards_max_tokens(
    provider: str,
    provider_norm: str,
    model: str,
    effective_base: str,
    task: Optional[str],
) -> bool:
    """Whether an explicit max_tokens is forwarded on this route.

    No default cap elsewhere (omitted = provider default; avoids max_completion_tokens / ZAI-vision
    quirks). Forward only where mandatory or honored: Anthropic Messages wire (400 without it);
    NVIDIA NIM (empty choices[] when omitted); MoA reference slots; Gemini native (fixed 65,535
    ceiling otherwise); OpenRouter (budgets the FULL window when omitted → 402 on low credit);
    managed local llama-server (uncapped decode with no EOS burns the GPU to the context window).
    """
    from agent.auxiliary_client import (
        _is_anthropic_compat_endpoint,
        _nous_on_messages_wire,
        _NVIDIA_PROVIDER_NAMES,
        _is_gemini_native_route,
        _is_managed_local_endpoint,
    )
    from utils import base_url_host_matches

    return (
        _is_anthropic_compat_endpoint(provider, effective_base)
        or _nous_on_messages_wire(provider_norm, model)
        or provider_norm in _NVIDIA_PROVIDER_NAMES
        or base_url_host_matches(effective_base, "integrate.api.nvidia.com")
        or str(task) == "moa_reference"
        or _is_gemini_native_route(provider_norm, effective_base)
        or provider_norm == "openrouter"
        or base_url_host_matches(effective_base, "openrouter.ai")
        or _is_managed_local_endpoint(effective_base)
    )


class _RequiredStream:
    def __init__(self, stream: Any, required: _RequiredRequest):
        self._stream = stream
        self._iterator = iter(stream)
        self._required = required

    def __iter__(self) -> _RequiredStream:
        return self

    def __next__(self) -> Any:
        token = _REQUIRED_REQUEST.set(self._required)
        try:
            return next(self._iterator)
        finally:
            _REQUIRED_REQUEST.reset(token)

    def close(self) -> None:
        token = _REQUIRED_REQUEST.set(self._required)
        try:
            close = getattr(self._stream, "close", None)
            if callable(close):
                close()
        finally:
            _REQUIRED_REQUEST.reset(token)

    def __enter__(self) -> _RequiredStream:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def preserve_stream_requirements(stream: Any) -> Any:
    required = _REQUIRED_REQUEST.get()
    if (
        not request_requirements_active()
        or required is None
        or hasattr(stream, "choices")
    ):
        return stream
    return _RequiredStream(stream, required)

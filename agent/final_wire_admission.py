"""Central final-wire semantic admission and physical egress adapters.

Authoritative admission happens at selected HTTPX transport.handle_request
and botocore endpoint.http_session.send. Cached/preflight estimates never
authorize a changed final body.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, NoReturn, Optional

import httpx

from agent.conversation_compression import ProviderBoundRequestOverLimit
from agent.model_metadata import estimate_messages_tokens_rough, estimate_tokens_rough

ACCOUNTING_VERSION = "final-wire-1"
ESTIMATOR_VERSION = "rough-1"

COVERED_MAIN = "covered_main"
NOT_COVERED_AUXILIARY = "not_covered_auxiliary"

KNOWN_R = "KNOWN_R"
PROVIDER_DEFAULT_UNRESOLVED = "PROVIDER_DEFAULT_UNRESOLVED"
INVALID_RESERVATION = "INVALID_RESERVATION"

COMPLETE_LOCAL = "COMPLETE_LOCAL"
ESTIMATED_MULTIMODAL = "ESTIMATED_MULTIMODAL"
UNSUPPORTED = "UNSUPPORTED"

_UNPARSEABLE = object()

_METADATA_KEYS = frozenset(
    {
        "model",
        "modelId",
        "temperature",
        "top_p",
        "topP",
        "top_k",
        "n",
        "stream",
        "store",
        "seed",
        "user",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "service_tier",
        "parallel_tool_calls",
        "stream_options",
        "timeout",
        "max_retries",
        "extra_headers",
        "extra_query",
        "extra_body",
        "anthropic_version",
        "anthropic_beta",
        "betas",
    }
)

_SCHEMA_CACHE: dict[tuple[str, str, str], int] = {}
_SCHEMA_CACHE_MAX = 256

_attempt_identity: ContextVar[Optional["FinalAttemptIdentity"]] = ContextVar(
    "hermes_final_wire_attempt", default=None
)
_local_refusal: ContextVar[Optional[BaseException]] = ContextVar(
    "hermes_final_wire_refusal", default=None
)


class ProviderBoundUnsupportedAccounting(Exception):
    """Covered path cannot produce a supported local admission estimate."""


class ProviderBoundInvalidAccounting(ProviderBoundRequestOverLimit):
    """Invalid/unknown W, malformed required cap, or contradictory aliases."""

    def __init__(self, reason: str, pressure: int = 0, limit: int = 0):
        self.reason = reason
        super().__init__(pressure, limit)


@dataclass(frozen=True)
class FinalAttemptIdentity:
    purpose: str
    family: str
    model: str
    endpoint: str
    window: int
    correlation_id: str


@dataclass(frozen=True)
class FinalSemanticSnapshot:
    family: str
    estimated_input: int
    window: int
    reservation_state: str
    resolved_r: int
    coverage: str
    coverage_reason: str = ""


@contextmanager
def bind_attempt_identity(identity: FinalAttemptIdentity) -> Iterator[FinalAttemptIdentity]:
    token = _attempt_identity.set(identity)
    refusal_token = _local_refusal.set(None)
    try:
        yield identity
    finally:
        _attempt_identity.reset(token)
        _local_refusal.reset(refusal_token)


def current_attempt_identity() -> Optional[FinalAttemptIdentity]:
    return _attempt_identity.get()


def current_local_refusal() -> Optional[BaseException]:
    return _local_refusal.get()


def latch_local_refusal(exc: BaseException) -> BaseException:
    _local_refusal.set(exc)
    return exc


def unwrap_local_refusal(exc: BaseException) -> Optional[BaseException]:
    latched = _local_refusal.get()
    if latched is not None:
        return latched
    current: Optional[BaseException] = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(
            current,
            (ProviderBoundRequestOverLimit, ProviderBoundUnsupportedAccounting, ProviderBoundInvalidAccounting),
        ):
            return current
        current = current.__cause__ or current.__context__
    return None


def _raise_local(exc: BaseException) -> NoReturn:
    raise latch_local_refusal(exc)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def estimate_schema_tokens(tools: Any) -> int:
    if not tools:
        return 0
    try:
        serialized = _canonical_json(tools)
    except Exception as exc:
        raise ProviderBoundUnsupportedAccounting("tool schema serialization failed") from exc
    key = (ACCOUNTING_VERSION, ESTIMATOR_VERSION, serialized)
    cached = _SCHEMA_CACHE.get(key)
    if cached is not None:
        return cached
    tokens = estimate_tokens_rough(serialized)
    if len(_SCHEMA_CACHE) >= _SCHEMA_CACHE_MAX:
        _SCHEMA_CACHE.pop(next(iter(_SCHEMA_CACHE)), None)
    _SCHEMA_CACHE[key] = tokens
    return tokens


def _estimate_value(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return estimate_tokens_rough(value)
    if isinstance(value, list) and (not value or isinstance(value[0], dict)):
        try:
            return estimate_messages_tokens_rough(value)
        except Exception:
            return estimate_tokens_rough(_canonical_json(value))
    if isinstance(value, (dict, list)):
        return estimate_tokens_rough(_canonical_json(value))
    return estimate_tokens_rough(str(value))


def estimate_final_body_pressure(body: Any) -> int:
    if not isinstance(body, dict):
        if body is None:
            raise ProviderBoundUnsupportedAccounting("missing JSON body")
        raise ProviderBoundUnsupportedAccounting("unknown context representation")
    total = 0
    counted_system = False
    messages = body.get("messages")
    if isinstance(messages, list):
        total += _estimate_value(messages)
        extra = body.get("system")
        if extra:
            total += _estimate_value(extra)
            counted_system = True
    else:
        incoming = body.get("input")
        if isinstance(incoming, str):
            total += estimate_tokens_rough(incoming)
        elif incoming is not None:
            total += _estimate_value(incoming)
        extra = body.get("instructions")
        if extra:
            total += _estimate_value(extra)
            counted_system = True
    if not counted_system and body.get("system") is not None and "messages" not in body:
        total += _estimate_value(body.get("system"))
    contents = body.get("contents")
    if contents is not None:
        total += _estimate_value(contents)
    instruction = body.get("systemInstruction") or body.get("system_instruction")
    if instruction is not None:
        total += _estimate_value(instruction)
    for key in ("tools", "functions", "toolConfig"):
        if key in body and body[key] is not None:
            total += estimate_schema_tokens(body[key])
    for key in ("response_format", "responseFormat", "text"):
        value = body.get(key)
        if isinstance(value, dict):
            total += _estimate_value(value)
    generation = body.get("generationConfig") or body.get("generation_config")
    if isinstance(generation, dict):
        for schema_key in ("responseSchema", "responseJsonSchema"):
            if generation.get(schema_key) is not None:
                total += _estimate_value(generation.get(schema_key))
    return total


def estimate_final_httpx_pressure(
    body: Any,
    identity: Optional[FinalAttemptIdentity] = None,
    headers: Optional[Any] = None,
) -> int:
    del identity, headers
    return estimate_final_body_pressure(body)


def _resolve_reservation(body: Mapping[str, Any], family: str) -> tuple[str, int]:
    if not isinstance(body, dict):
        return INVALID_RESERVATION, 0
    if family == "chat_completions":
        has_max = "max_tokens" in body
        has_completion = "max_completion_tokens" in body
        if has_max and has_completion and body.get("max_tokens") != body.get("max_completion_tokens"):
            return INVALID_RESERVATION, 0
        raw = body.get("max_tokens", body.get("max_completion_tokens"))
        return _finite_or_unresolved(raw, required=False)
    if family in {"responses", "codex_responses"}:
        return _finite_or_unresolved(body.get("max_output_tokens"), required=False)
    if family in {"anthropic_messages", "anthropic_bedrock"}:
        return _finite_or_unresolved(body.get("max_tokens"), required=True)
    if family in {"bedrock_converse", "converse"}:
        inference = body.get("inferenceConfig") or {}
        raw = inference.get("maxTokens") if isinstance(inference, dict) else None
        return _finite_or_unresolved(raw, required=False)
    if family in {"gemini", "gemini_native"}:
        generation = body.get("generationConfig") or {}
        raw = generation.get("maxOutputTokens") if isinstance(generation, dict) else None
        return _finite_or_unresolved(raw, required=False)
    raw = body.get("max_tokens", body.get("max_output_tokens", body.get("max_completion_tokens")))
    return _finite_or_unresolved(raw, required=False)


def _finite_or_unresolved(raw: Any, *, required: bool) -> tuple[str, int]:
    if raw is None:
        if required:
            return INVALID_RESERVATION, 0
        return PROVIDER_DEFAULT_UNRESOLVED, 0
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return INVALID_RESERVATION, 0
    if value <= 0:
        return INVALID_RESERVATION, 0
    return KNOWN_R, value


def project_final_body(
    body: Any,
    identity: FinalAttemptIdentity,
) -> FinalSemanticSnapshot:
    try:
        estimated = estimate_final_body_pressure(body)
        coverage = COMPLETE_LOCAL
        reason = ""
    except ProviderBoundUnsupportedAccounting as exc:
        estimated = 0
        coverage = UNSUPPORTED
        reason = str(exc) or "unsupported"
    state, resolved_r = _resolve_reservation(body if isinstance(body, dict) else {}, identity.family)
    return FinalSemanticSnapshot(
        family=identity.family,
        estimated_input=estimated,
        window=int(identity.window),
        reservation_state=state,
        resolved_r=resolved_r,
        coverage=coverage,
        coverage_reason=reason,
    )


def admit_snapshot(snapshot: FinalSemanticSnapshot) -> None:
    window = snapshot.window
    if window <= 0:
        _raise_local(ProviderBoundInvalidAccounting("invalid_or_unknown_W", snapshot.estimated_input, window))
    if snapshot.coverage == UNSUPPORTED:
        _raise_local(ProviderBoundUnsupportedAccounting(snapshot.coverage_reason or "unsupported"))
    if snapshot.reservation_state == INVALID_RESERVATION:
        _raise_local(
            ProviderBoundInvalidAccounting(
                "invalid_required_or_malformed_cap",
                snapshot.estimated_input,
                window,
            )
        )
    if snapshot.estimated_input >= window:
        _raise_local(ProviderBoundRequestOverLimit(snapshot.estimated_input, window))
    if snapshot.reservation_state == KNOWN_R:
        combined = snapshot.estimated_input + snapshot.resolved_r
        if combined > window:
            _raise_local(ProviderBoundRequestOverLimit(combined, window))


def admit_final_json(body: Any, identity: Optional[FinalAttemptIdentity] = None) -> None:
    current = identity or _attempt_identity.get()
    if current is None or current.purpose != COVERED_MAIN:
        _raise_local(ProviderBoundUnsupportedAccounting("covered request missing attempt identity"))
    admit_snapshot(project_final_body(body, current))


def _decode_httpx_json(request: httpx.Request) -> Any:
    try:
        content = request.content
    except Exception:
        return _UNPARSEABLE
    if not content:
        return None
    try:
        return json.loads(content)
    except Exception:
        return _UNPARSEABLE


class GuardedHTTPXTransport(httpx.BaseTransport):
    def __init__(self, inner: httpx.BaseTransport, *, covered: bool = True):
        self._inner = inner
        self._covered = covered

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if self._covered:
            _admit_httpx_request(request)
        return self._inner.handle_request(request)

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class GuardedAsyncHTTPXTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport, *, covered: bool = True):
        self._inner = inner
        self._covered = covered

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self._covered:
            _admit_httpx_request(request)
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        close = getattr(self._inner, "aclose", None)
        if callable(close):
            result = close()
            if hasattr(result, "__await__"):
                await result  # type: ignore[misc]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _admit_httpx_request(request: httpx.Request) -> None:
    identity = _attempt_identity.get()
    if identity is None or identity.purpose != COVERED_MAIN:
        _raise_local(ProviderBoundUnsupportedAccounting("covered request missing attempt identity"))
    body = _decode_httpx_json(request)
    if body is _UNPARSEABLE:
        _raise_local(ProviderBoundUnsupportedAccounting("unparseable JSON body"))
    if body is None:
        _raise_local(ProviderBoundUnsupportedAccounting("missing JSON body"))
    admit_final_json(body, identity)


def _wrap_one_transport(transport: Any, *, covered: bool) -> Any:
    if transport is None:
        return None
    if isinstance(transport, (GuardedHTTPXTransport, GuardedAsyncHTTPXTransport)):
        return transport
    if isinstance(transport, httpx.AsyncBaseTransport):
        return GuardedAsyncHTTPXTransport(transport, covered=covered)
    return GuardedHTTPXTransport(transport, covered=covered)


def wrap_httpx_client_transports(client: Any, *, covered: bool = True) -> Any:
    if client is None:
        raise ProviderBoundUnsupportedAccounting("missing httpx client")
    transport = getattr(client, "_transport", None)
    if transport is not None:
        client._transport = _wrap_one_transport(transport, covered=covered)
    mounts = getattr(client, "_mounts", None)
    if isinstance(mounts, dict):
        for key, mounted in list(mounts.items()):
            if mounted is not None:
                mounts[key] = _wrap_one_transport(mounted, covered=covered)
    return client


def build_covered_keepalive_http_client(base_url: str = "", **kwargs: Any) -> Any:
    from agent.process_bootstrap import build_keepalive_http_client

    try:
        client = build_keepalive_http_client(base_url, **kwargs)
    except Exception as exc:
        raise ProviderBoundUnsupportedAccounting("covered httpx construction failed") from exc
    if client is None:
        raise ProviderBoundUnsupportedAccounting("covered httpx construction returned None")
    return wrap_httpx_client_transports(client, covered=True)


class GuardedBotocoreHttpSession:
    def __init__(self, inner: Any):
        self._inner = inner

    def send(self, request: Any, **kwargs: Any) -> Any:
        _admit_botocore_request(request)
        return self._inner.send(request, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _botocore_json_body(request: Any) -> Any:
    body = getattr(request, "body", None)
    if body is None and isinstance(request, dict):
        body = request.get("body")
    if isinstance(body, bytes):
        try:
            return json.loads(body.decode("utf-8"))
        except Exception:
            return _UNPARSEABLE
    if isinstance(body, str):
        try:
            return json.loads(body)
        except Exception:
            return _UNPARSEABLE
    if isinstance(body, dict):
        return body
    return _UNPARSEABLE if body is not None else None


def _admit_botocore_request(request: Any) -> None:
    identity = _attempt_identity.get()
    if identity is None or identity.purpose != COVERED_MAIN:
        _raise_local(ProviderBoundUnsupportedAccounting("covered request missing attempt identity"))
    body = _botocore_json_body(request)
    if body is _UNPARSEABLE:
        _raise_local(ProviderBoundUnsupportedAccounting("unparseable Converse JSON body"))
    if body is None:
        _raise_local(ProviderBoundUnsupportedAccounting("missing Converse JSON body"))
    admit_final_json(body, identity)


def _needs_retry_refuse_local(caught_exception=None, **_kwargs: Any) -> None:
    refusal = unwrap_local_refusal(caught_exception) if caught_exception is not None else current_local_refusal()
    if refusal is not None:
        raise refusal


def wrap_botocore_runtime_client(client: Any) -> Any:
    if client is None:
        raise ProviderBoundUnsupportedAccounting("missing bedrock runtime client")
    endpoint = getattr(client, "_endpoint", None)
    if endpoint is None:
        raise ProviderBoundUnsupportedAccounting("bedrock client missing endpoint")
    session = getattr(endpoint, "http_session", None)
    if session is None:
        raise ProviderBoundUnsupportedAccounting("bedrock endpoint missing http_session")
    if not isinstance(session, GuardedBotocoreHttpSession):
        endpoint.http_session = GuardedBotocoreHttpSession(session)
    events = getattr(client, "meta", None)
    emitter = getattr(events, "events", None) if events is not None else None
    if emitter is not None and hasattr(emitter, "register_first"):
        try:
            emitter.register_first(
                "needs-retry.bedrock-runtime",
                _needs_retry_refuse_local,
                unique_id="hermes-final-wire-local-refusal",
            )
        except Exception:
            emitter.register_first("needs-retry.bedrock-runtime", _needs_retry_refuse_local)
    return client


def restore_typed_local_refusal(exc: BaseException) -> NoReturn:
    refusal = unwrap_local_refusal(exc)
    if refusal is not None:
        raise refusal
    raise exc


def intercepted_openai_class(base_cls: type) -> type:
    class HermesOpenAI(base_cls):  # type: ignore[misc,valid-type]
        def _sleep_for_retry(self, *args: Any, **kwargs: Any) -> Any:
            refusal = current_local_refusal()
            if refusal is not None:
                raise refusal
            return super()._sleep_for_retry(*args, **kwargs)

        def request(self, *args: Any, **kwargs: Any) -> Any:
            try:
                return super().request(*args, **kwargs)
            except Exception as err:
                refusal = unwrap_local_refusal(err)
                if refusal is not None:
                    raise refusal
                raise

    HermesOpenAI.__name__ = getattr(base_cls, "__name__", "OpenAI")
    HermesOpenAI.__qualname__ = getattr(base_cls, "__qualname__", HermesOpenAI.__name__)
    return HermesOpenAI


def intercepted_anthropic_class(base_cls: type) -> type:
    class HermesAnthropic(base_cls):  # type: ignore[misc,valid-type]
        def _sleep_for_retry(self, *args: Any, **kwargs: Any) -> Any:
            refusal = current_local_refusal()
            if refusal is not None:
                raise refusal
            return super()._sleep_for_retry(*args, **kwargs)

        def request(self, *args: Any, **kwargs: Any) -> Any:
            try:
                return super().request(*args, **kwargs)
            except Exception as err:
                refusal = unwrap_local_refusal(err)
                if refusal is not None:
                    raise refusal
                raise

    HermesAnthropic.__name__ = getattr(base_cls, "__name__", "Anthropic")
    HermesAnthropic.__qualname__ = getattr(base_cls, "__qualname__", HermesAnthropic.__name__)
    return HermesAnthropic

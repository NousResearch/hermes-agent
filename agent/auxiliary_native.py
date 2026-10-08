"""Native auxiliary request observations with host-owned route checks.

Request bodies and response items stay in memory. Signed route data contains no
credentials and is valid only in this process and profile.
"""

from __future__ import annotations

import contextvars
import hashlib
import hmac
import json
import logging
import math
import secrets
import time
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Mapping

logger = logging.getLogger(__name__)
PRE_NATIVE = "pre_auxiliary_native_request"
POST_NATIVE = "post_auxiliary_native_request"
_SCHEMA = "hermes.native-route.v1"
_SIGNING_KEY = secrets.token_bytes(32)
_ATTEMPT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("native_aux_attempt", default=None)
_BODY_FIELDS = frozenset({
    "model", "instructions", "input", "store", "tools", "tool_choice", "parallel_tool_calls",
    "reasoning", "text", "include", "service_tier", "prompt_cache_key", "prompt_cache_retention",
    "stream", "metadata", "truncation", "max_output_tokens", "temperature", "top_p", "user",
    "previous_response_id", "safety_identifier",
})
_SAFE_HEADERS = frozenset({
    "session_id", "session-id", "x-session-id", "x-initiator", "originator", "chatgpt-account-id",
    "openai-beta", "accept", "user-agent", "content-type", "x-client-request-id",
})


class NativeRouteError(ValueError):
    """The native request does not match its captured route."""


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _fingerprint(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hmac.new(_SIGNING_KEY, data.encode("utf-8"), hashlib.sha256).hexdigest()


def _profile_key() -> str:
    from hermes_constants import hermes_home_key

    return _fingerprint(str(hermes_home_key()))


def _headers(client: Any, extra_headers: Mapping[str, Any]) -> dict[str, str]:
    headers = {str(k).lower(): v for k, v in dict(client.default_headers).items() if isinstance(v, str)}
    headers.update({str(k).lower(): str(v) for k, v in extra_headers.items()})
    return headers


def _credential_fingerprint(client: Any) -> str:
    key = getattr(client, "api_key", None)
    if not isinstance(key, str):
        raise NativeRouteError("Native credential binding requires a text credential")
    return _fingerprint(key)


def _has_native_hooks() -> bool:
    from agent.auxiliary_hooks import _has_hook

    return _has_hook(PRE_NATIVE) or _has_hook(POST_NATIVE)


@contextmanager
def native_attempt_scope(**context: Any):
    """Keep the physical identity through worker and async context copies."""
    if not _has_native_hooks():
        yield
        return
    from agent.auxiliary_hooks import _parent_turn_identity

    state = {**context, "parent": _parent_turn_identity()}
    token = _ATTEMPT.set(state)
    try:
        yield
    finally:
        _ATTEMPT.reset(token)


def _native_body(kwargs: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    body = {k: v for k, v in kwargs.items() if k not in {"timeout", "extra_headers"}}
    if set(body) - _BODY_FIELDS:
        raise NativeRouteError("Native request contains unsupported request fields")
    if not isinstance(body.get("input"), list):
        raise NativeRouteError("Native input must be a list")
    headers = dict(kwargs.get("extra_headers") or {})
    names = set()
    for name, value in headers.items():
        if not isinstance(name, str) or not isinstance(value, str) or name.lower() not in _SAFE_HEADERS:
            raise NativeRouteError("Native request contains unsupported per-request headers")
        if name.lower() in names or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise NativeRouteError("Native request contains invalid per-request headers")
        names.add(name.lower())
    return _json_copy(body), _json_copy(headers)


def _settings(body: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in body.items() if k != "input"}


def _route_context(client: Any, body: dict[str, Any], extra_headers: dict[str, str], state: dict[str, Any]) -> dict[str, Any]:
    from agent.auxiliary_client import _normalize_main_runtime

    if client.default_query:
        raise NativeRouteError("Native client query options are not supported")
    runtime = _normalize_main_runtime(None)
    parent = state["parent"]
    metadata = state["metadata"]
    route = dict(
        schema=_SCHEMA, provider=state["provider"], model=body["model"],
        base_url=str(client.base_url), api_mode="codex_responses", profile_key=_profile_key(),
        session_id=runtime.get("session_id") or parent.get("session_id") or "",
        cache_scope=runtime.get("cache_scope") or runtime.get("session_id") or parent.get("session_id") or "",
        aux_task=state["aux_task"], api_request_id=str(metadata.get("api_request_id") or ""),
        retry_count=int(metadata.get("retry_count") or 0),
        credential_fingerprint=_credential_fingerprint(client),
        headers_fingerprint=_fingerprint(_headers(client, extra_headers)), extra_headers=extra_headers,
        settings_fingerprint=_fingerprint(_settings(body)),
        input_count=len(body.get("input") or []), input_fingerprint=_fingerprint(body.get("input") or []),
    )
    route["signature"] = _fingerprint(route)
    return route


class NativeRequestObservation:
    """Report one physical native attempt. Observer results cannot change it."""

    def __init__(self, client: Any, kwargs: Mapping[str, Any], state: dict[str, Any]) -> None:
        from agent.auxiliary_hooks import _fire

        body, headers = _native_body(kwargs)
        route = _route_context(client, body, headers, state)
        self.started = time.time()
        self.ended = False
        self.base = dict(
            state["parent"], aux_task=state["aux_task"], api_request_id=route["api_request_id"],
            retry_count=route["retry_count"], provider=route["provider"], model=route["model"],
            base_url=route["base_url"], api_mode=route["api_mode"], route_context=route,
            streaming=True, started_at=self.started,
        )
        _fire(PRE_NATIVE, **deepcopy(self.base), native_request=body, extra_headers=headers)

    def end(self, response: Any = None, error: BaseException | None = None) -> None:
        from agent.auxiliary_hooks import _fire

        if self.ended:
            return
        self.ended = True
        ended = time.time()
        native_response = native_response_dict(response) if response is not None else None
        _fire(
            POST_NATIVE, **deepcopy(self.base), native_response=native_response,
            error_type=type(error).__name__ if error is not None else None,
            status=native_response.get("status") if native_response else None,
            usage=native_response.get("usage") if native_response else None,
            ended_at=ended, api_duration=max(0.0, ended - self.started),
        )


def begin_native_request(client: Any, kwargs: Mapping[str, Any]) -> NativeRequestObservation | None:
    """Do no capture work when the native observers are absent."""
    state = _ATTEMPT.get()
    if state is None:
        return None
    try:
        return NativeRequestObservation(client, kwargs, state)
    except Exception:
        logger.debug("Native auxiliary observation could not start", exc_info=True)
        return None


def end_native_request(observation: NativeRequestObservation | None, response: Any = None, error: BaseException | None = None) -> None:
    if observation is None:
        return
    try:
        observation.end(response, error)
    except Exception:
        logger.debug("Native auxiliary observation could not end", exc_info=True)


def native_response_dict(response: Any) -> dict[str, Any]:
    """Keep native output items, including reasoning and message phases."""
    if hasattr(response, "model_dump"):
        value = response.model_dump(mode="json")
    elif isinstance(response, dict):
        value = response
    else:
        def convert(item: Any) -> Any:
            if isinstance(item, dict):
                return {k: convert(v) for k, v in item.items()}
            if isinstance(item, (list, tuple)):
                return [convert(v) for v in item]
            if hasattr(item, "model_dump"):
                return item.model_dump(mode="json")
            if hasattr(item, "__dict__"):
                return {k: convert(v) for k, v in vars(item).items() if not k.startswith("_")}
            return item
        value = convert(response)
    if not isinstance(value, dict):
        raise NativeRouteError("Native response must be an object")
    return _json_copy(value)


def verify_route_context(route_context: Mapping[str, Any], expected_session_id: str) -> dict[str, Any]:
    """Reject changed or foreign route data before credential resolution."""
    route = _json_copy(dict(route_context))
    signature = route.pop("signature", "")
    if not isinstance(signature, str) or not hmac.compare_digest(signature, _fingerprint(route)):
        raise NativeRouteError("Native route signature is invalid")
    route["signature"] = signature
    if route.get("schema") != _SCHEMA or route.get("api_mode") != "codex_responses":
        raise NativeRouteError("Native route is not supported")
    if route.get("profile_key") != _profile_key():
        raise NativeRouteError("Native route belongs to another profile")
    if not expected_session_id or route.get("session_id") != expected_session_id:
        raise NativeRouteError("Native route belongs to another session")
    from agent.auxiliary_client import _normalize_main_runtime

    runtime = _normalize_main_runtime(None)
    if runtime.get("session_id") and runtime["session_id"] != expected_session_id:
        raise NativeRouteError("Native session changed")
    scope = runtime.get("cache_scope") or runtime.get("session_id")
    if scope and scope != route["cache_scope"]:
        raise NativeRouteError("Native cache scope changed")
    return route


def _resolve_native_client(route: dict[str, Any]) -> Any:
    from agent.auxiliary_client import CodexAuxiliaryClient, _get_cached_client
    from hermes_cli.runtime_provider import resolve_runtime_provider

    runtime = resolve_runtime_provider(requested=route["provider"], target_model=route["model"])
    if runtime.get("api_mode") != "codex_responses" or str(runtime.get("base_url") or "").rstrip("/") != route["base_url"].rstrip("/"):
        raise NativeRouteError("Native route changed")
    client, _model = _get_cached_client(
        provider=route["provider"], model=route["model"], api_mode="codex_responses",
        base_url=runtime["base_url"], api_key=runtime.get("api_key"),
    )
    if not isinstance(client, CodexAuxiliaryClient):
        raise NativeRouteError("Native route no longer has a Responses client")
    real_client = client._real_client
    if real_client.default_query:
        raise NativeRouteError("Native client query options are not supported")
    if str(real_client.base_url) != route["base_url"]:
        raise NativeRouteError("Native endpoint changed")
    if not hmac.compare_digest(_credential_fingerprint(real_client), route["credential_fingerprint"]):
        raise NativeRouteError("Native credential changed")
    if not hmac.compare_digest(_fingerprint(_headers(real_client, route["extra_headers"])), route["headers_fingerprint"]):
        raise NativeRouteError("Native headers changed")
    return real_client


def _validate_native_body(native_request: Mapping[str, Any], route: dict[str, Any]) -> dict[str, Any]:
    if {"timeout", "extra_headers"} & native_request.keys():
        raise NativeRouteError("Native body cannot contain SDK options")
    body, headers = _native_body(native_request)
    if headers or body.get("model") != route["model"] or body.get("stream") is not True:
        raise NativeRouteError("Native model or stream setting changed")
    if not isinstance(body.get("input"), list):
        raise NativeRouteError("Native input must be a list")
    if not hmac.compare_digest(_fingerprint(_settings(body)), route["settings_fingerprint"]):
        raise NativeRouteError("Native request settings changed")
    count = route["input_count"]
    if len(body["input"]) < count or not hmac.compare_digest(_fingerprint(body["input"][:count]), route["input_fingerprint"]):
        raise NativeRouteError("Native request prefix changed")
    return body


def complete_native_request(
    native_request: Mapping[str, Any], route: dict[str, Any], timeout: float, *, task: str,
) -> dict[str, Any]:
    """Send once on the captured route. Keep all native response items."""
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise NativeRouteError("Native timeout must be a positive finite number")
    body = _validate_native_body(native_request, route)
    owner = _resolve_native_client(route)
    client = owner.with_options(max_retries=0, timeout=timeout)
    from agent.auxiliary_client import _CodexStreamGuard
    from agent.codex_runtime import _consume_codex_event_stream
    from agent.sdk_transform_bypass import bypass_sdk_request_transform

    kwargs = {**body, "extra_headers": dict(route["extra_headers"]), "timeout": timeout}
    guard = _CodexStreamGuard(owner, timeout)
    observation = begin_native_request(client, kwargs)
    try:
        guard.start()
        event_stream = client.responses.create(**bypass_sdk_request_transform(kwargs))
        guard.adopt_stream(event_stream)
        try:
            if guard.timed_out.is_set():
                raise TimeoutError(guard.timeout_message())
            guard.check_cancelled()
            final = event_stream if hasattr(event_stream, "output") else _consume_codex_event_stream(
                event_stream, model=body["model"], on_event=guard.on_event,
            )
        finally:
            guard.release_stream(event_stream)
        if guard.timed_out.is_set():
            raise TimeoutError(guard.timeout_message())
        guard.check_cancelled()
        if final is None:
            raise NativeRouteError("Native Responses stream has no final response")
        result = native_response_dict(final)
        from types import SimpleNamespace

        from agent.aux_accounting import record_aux_usage

        record_aux_usage(SimpleNamespace(**result), task, provider=route["provider"],
                         base_url=route["base_url"], api_mode="codex_responses")
        verify_route_context(route, route["session_id"])
        _resolve_native_client(route)
        end_native_request(observation, final)
        return result
    except BaseException as exc:
        end_native_request(observation, error=exc)
        if isinstance(exc, Exception) and guard.timed_out.is_set():
            raise TimeoutError(guard.timeout_message()) from exc
        raise
    finally:
        guard.finish()

"""Validated, request-scoped model runtimes for the API server.

The API server normally resolves provider credentials from the active Hermes
profile.  External control planes sometimes already own that configuration and
need to submit one run against an ephemeral upstream without mutating
``config.yaml`` or process-global environment variables.  This module keeps
that boundary small and explicit.

Runtime credentials are accepted only through ``MODEL_API_KEY_HEADER``.  They
are deliberately excluded from the request JSON so ordinary body capture,
status payloads, and response stores cannot retain them accidentally.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit


MODEL_API_KEY_HEADER = "X-Hermes-Model-Api-Key"

_SUPPORTED_API_MODES = frozenset(
    {"chat_completions", "codex_responses", "anthropic_messages"}
)
_ALLOWED_RUNTIME_KEYS = frozenset(
    {"base_url", "api_mode", "max_tokens", "request_overrides"}
)
_ALLOWED_REQUEST_OVERRIDE_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "parallel_tool_calls",
        "tool_choice",
        "seed",
    }
)
_NUMERIC_OVERRIDE_KEYS = frozenset(
    {"temperature", "top_p", "frequency_penalty", "presence_penalty"}
)
_PROVIDER_RE = re.compile(r"^[A-Za-z0-9_.-]{2,80}$")


class RequestModelRuntimeError(ValueError):
    """A client-supplied request runtime failed validation."""

    def __init__(self, message: str, *, code: str, param: str = "runtime_model"):
        super().__init__(message)
        self.code = code
        self.param = param


@dataclass(frozen=True, repr=False)
class RequestModelRuntime:
    """Immutable model transport snapshot for exactly one API run."""

    model: str
    provider: str
    base_url: str
    api_key: Optional[str]
    api_mode: str
    max_tokens: Optional[int]
    request_overrides: Mapping[str, Any]

    def __repr__(self) -> str:
        # Credentials and endpoint paths can both be sensitive in hosted
        # deployments.  Keep repr safe even if a caller logs this object.
        return (
            "RequestModelRuntime("
            f"model={self.model!r}, provider={self.provider!r}, "
            f"api_mode={self.api_mode!r}, max_tokens={self.max_tokens!r}, "
            f"has_api_key={bool(self.api_key)})"
        )

    def agent_kwargs(self) -> dict[str, Any]:
        """Return the request-only kwargs consumed by ``AIAgent``."""
        request_overrides = dict(self.request_overrides)
        if isinstance(request_overrides.get("stop"), tuple):
            request_overrides["stop"] = list(request_overrides["stop"])
        kwargs: dict[str, Any] = {
            "provider": self.provider,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
            "request_overrides": request_overrides,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        return kwargs


def _clean_runtime_id(value: Any, *, field: str, max_len: int) -> str:
    if not isinstance(value, str):
        raise RequestModelRuntimeError(
            f"'{field}' must be a string",
            code="invalid_runtime_model",
            param=field,
        )
    cleaned = value.strip()
    if not cleaned or len(cleaned) > max_len or re.search(r"[\r\n\x00]", cleaned):
        raise RequestModelRuntimeError(
            f"'{field}' is empty or invalid",
            code="invalid_runtime_model",
            param=field,
        )
    return cleaned


def _validate_base_url(value: Any) -> str:
    base_url = _clean_runtime_id(value, field="runtime_model.base_url", max_len=2048)
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except ValueError as exc:
        raise RequestModelRuntimeError(
            "'runtime_model.base_url' is not a valid URL",
            code="invalid_runtime_model",
            param="runtime_model.base_url",
        ) from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise RequestModelRuntimeError(
            "'runtime_model.base_url' must be an HTTP(S) URL without credentials, query, or fragment",
            code="invalid_runtime_model",
            param="runtime_model.base_url",
        )
    return base_url.rstrip("/")


def _validate_stop(value: Any) -> str | tuple[str, ...]:
    if isinstance(value, str):
        if len(value) <= 1024:
            return value
    elif isinstance(value, list) and len(value) <= 16:
        if all(isinstance(item, str) and len(item) <= 1024 for item in value):
            return tuple(value)
    raise RequestModelRuntimeError(
        "'runtime_model.request_overrides.stop' must be a string or up to 16 strings",
        code="invalid_runtime_model",
        param="runtime_model.request_overrides.stop",
    )


def _validate_request_overrides(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RequestModelRuntimeError(
            "'runtime_model.request_overrides' must be an object",
            code="invalid_runtime_model",
            param="runtime_model.request_overrides",
        )

    unknown = sorted(set(value) - _ALLOWED_REQUEST_OVERRIDE_KEYS)
    if unknown:
        raise RequestModelRuntimeError(
            "Unsupported runtime_model.request_overrides field(s): " + ", ".join(unknown),
            code="unsupported_runtime_model_option",
            param="runtime_model.request_overrides",
        )

    result: dict[str, Any] = {}
    for key, raw in value.items():
        param = f"runtime_model.request_overrides.{key}"
        if key in _NUMERIC_OVERRIDE_KEYS:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
                raise RequestModelRuntimeError(
                    f"'{param}' must be a finite number",
                    code="invalid_runtime_model",
                    param=param,
                )
            result[key] = raw
        elif key == "stop":
            result[key] = _validate_stop(raw)
        elif key == "parallel_tool_calls":
            if not isinstance(raw, bool):
                raise RequestModelRuntimeError(
                    f"'{param}' must be a boolean",
                    code="invalid_runtime_model",
                    param=param,
                )
            result[key] = raw
        elif key == "tool_choice":
            if not isinstance(raw, str) or raw not in {"auto", "none", "required"}:
                raise RequestModelRuntimeError(
                    f"'{param}' must be one of auto, none, required",
                    code="invalid_runtime_model",
                    param=param,
                )
            result[key] = raw
        elif key == "seed":
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise RequestModelRuntimeError(
                    f"'{param}' must be an integer",
                    code="invalid_runtime_model",
                    param=param,
                )
            result[key] = raw
    return result


def parse_request_model_runtime(
    body: Mapping[str, Any],
    *,
    api_key_header: Optional[str],
) -> Optional[RequestModelRuntime]:
    """Parse ``runtime_model`` from a ``/v1/runs`` request.

    ``None`` means the request did not opt into the feature.  Presence is
    strict: malformed or unknown values fail closed rather than falling back
    to the profile's global model.
    """
    raw = body.get("runtime_model")
    if raw is None:
        if api_key_header:
            raise RequestModelRuntimeError(
                f"'{MODEL_API_KEY_HEADER}' requires a runtime_model object",
                code="missing_runtime_model",
                param="runtime_model",
            )
        return None
    if not isinstance(raw, dict):
        raise RequestModelRuntimeError(
            "'runtime_model' must be an object",
            code="invalid_runtime_model",
            param="runtime_model",
        )

    unknown = sorted(set(raw) - _ALLOWED_RUNTIME_KEYS)
    if unknown:
        hint = " Use X-Hermes-Model-Api-Key for credentials." if "api_key" in unknown else ""
        raise RequestModelRuntimeError(
            "Unsupported runtime_model field(s): " + ", ".join(unknown) + hint,
            code="unsupported_runtime_model_option",
            param="runtime_model",
        )

    model = _clean_runtime_id(body.get("model"), field="model", max_len=200)
    provider = _clean_runtime_id(body.get("provider"), field="provider", max_len=80)
    if not _PROVIDER_RE.fullmatch(provider):
        raise RequestModelRuntimeError(
            "'provider' contains unsupported characters",
            code="invalid_runtime_model",
            param="provider",
        )

    api_mode = raw.get("api_mode", "chat_completions")
    api_mode = _clean_runtime_id(
        api_mode,
        field="runtime_model.api_mode",
        max_len=40,
    ).lower()
    if api_mode not in _SUPPORTED_API_MODES:
        raise RequestModelRuntimeError(
            "'runtime_model.api_mode' must be chat_completions, codex_responses, or anthropic_messages",
            code="invalid_runtime_model",
            param="runtime_model.api_mode",
        )

    max_tokens = raw.get("max_tokens")
    if max_tokens is not None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 10_000_000:
            raise RequestModelRuntimeError(
                "'runtime_model.max_tokens' must be an integer between 1 and 10000000",
                code="invalid_runtime_model",
                param="runtime_model.max_tokens",
            )

    api_key = (api_key_header or "").strip() or None
    if api_key and api_key.startswith("Bearer "):
        api_key = api_key[7:].strip() or None
    if api_key and len(api_key) > 8192:
        raise RequestModelRuntimeError(
            f"'{MODEL_API_KEY_HEADER}' is too long",
            code="invalid_runtime_model_credential",
            param=MODEL_API_KEY_HEADER,
        )

    return RequestModelRuntime(
        model=model,
        provider=provider,
        base_url=_validate_base_url(raw.get("base_url")),
        api_key=api_key,
        api_mode=api_mode,
        max_tokens=max_tokens,
        request_overrides=MappingProxyType(
            _validate_request_overrides(raw.get("request_overrides"))
        ),
    )

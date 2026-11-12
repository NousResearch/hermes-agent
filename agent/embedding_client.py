"""Authenticated OpenAI-compatible embedding transport for auxiliary consumers.

This module deliberately bypasses the auxiliary chat completion relay. It asks the
central provider resolver for a raw embeddings-capable client and never switches
providers after a request failure.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any, Sequence


class EmbeddingError(RuntimeError):
    """Safe embedding failure with a stable, non-secret machine-readable code."""

    def __init__(self, code: str, message: str | None = None):
        self.code = code if code in _MESSAGES else "embedding_failed"
        # Ignore caller text: exception strings are fixed, non-secret diagnostics.
        super().__init__(_MESSAGES[self.code])


_MESSAGES = {
    "embedding_unconfigured": "No supported embedding provider is configured.",
    "embedding_unsupported": "The selected provider does not support embeddings through this transport.",
    "embedding_failed": "The embedding request failed.",
    "embedding_timeout": "The embedding request timed out.",
    "embedding_invalid_response": "The embedding provider returned an invalid response.",
}
_DEFAULT_PROVIDER = "openai-api"
_DEFAULT_MODEL = "text-embedding-3-small"
_DEFAULT_TIMEOUT = 60.0
_MAX_TIMEOUT = 300.0
_MAX_DIMENSIONS = 65536
_INPUT_TYPE_HANDLING = "shared"


def _error(code: str) -> EmbeddingError:
    return EmbeddingError(code, _MESSAGES.get(code, _MESSAGES["embedding_failed"]))


def _selection(
    provider: str | None, model: str | None, dimensions: int | None, timeout: float | None,
) -> tuple[str, str, int | None, float, Any]:
    """Read the profile-scoped embedding route, never the configured chat route."""
    try:
        from hermes_cli.config import cfg_get, load_config_readonly
        configured = load_config_readonly()
        if not isinstance(configured, dict):
            raise ValueError("configuration unavailable")
        embedding = cfg_get(configured, "auxiliary", "embedding", default={}) or {}
    except Exception:  # health: allow BLE001 -- malformed or unreadable config must fail closed without exposing secrets.
        raise _error("embedding_unconfigured") from None
    if not isinstance(embedding, dict):
        raise _error("embedding_unconfigured")
    configured_provider = embedding.get("provider", "auto")
    configured_model = embedding.get("model", "")
    if not isinstance(configured_provider, str) or not isinstance(configured_model, str):
        raise _error("embedding_unconfigured")
    selected_provider = str(provider if provider is not None else configured_provider or "auto").strip().lower()
    if selected_provider == "openai":
        selected_provider = "openai-api"
    selected_model = str(model if model is not None else configured_model or "").strip()
    selected_dimensions = dimensions if dimensions is not None else embedding.get("dimensions")
    configured_timeout = timeout if timeout is not None else embedding.get("timeout", _DEFAULT_TIMEOUT)
    input_type_handling = embedding.get("input_type_handling")
    if selected_provider in {"", "auto"}:
        # This is intentionally a fixed embedding-capable route; chat auto-routing can
        # select OAuth or text-only models and is never consulted here.
        selected_provider = _DEFAULT_PROVIDER
        selected_model = selected_model or _DEFAULT_MODEL
    if not selected_model:
        selected_model = _DEFAULT_MODEL if selected_provider in {"openai", "openai-api"} else ""
    if not selected_model:
        raise _error("embedding_unconfigured")
    if selected_dimensions is not None:
        if type(selected_dimensions) is not int or not 1 <= selected_dimensions <= _MAX_DIMENSIONS:
            raise _error("embedding_unsupported")
    if input_type_handling is not None and input_type_handling != "shared":
        raise _error("embedding_unsupported")
    if isinstance(configured_timeout, bool) or not isinstance(configured_timeout, (int, float)):
        raise _error("embedding_unconfigured")
    configured_timeout = float(configured_timeout)
    if not 0 < configured_timeout <= _MAX_TIMEOUT:
        raise _error("embedding_unconfigured")
    return selected_provider, selected_model, selected_dimensions, configured_timeout, input_type_handling


def _is_supported_route(provider: str) -> bool:
    """Reject non-API-key and non-OpenAI-wire routes before resolving credentials."""
    normalized = provider.strip().lower()
    if normalized in {"openai-codex", "codex", "anthropic", "bedrock", "vertex", "gemini", "nous", "xai-oauth"}:
        return False
    if normalized in {"openai", "openai-api", "openrouter"}:
        return True
    if normalized.startswith("custom:"):
        try:
            from hermes_cli.config import load_config_readonly
            from hermes_cli.runtime_provider_custom import _get_named_custom_provider
            config = load_config_readonly()
            if not isinstance(config, dict):
                return False
            entry = _get_named_custom_provider(normalized, config=config, resolve_credentials=False)
            entry = entry or _get_named_custom_provider(
                normalized.removeprefix("custom:"), config=config, resolve_credentials=False,
            )
            if not isinstance(entry, dict):
                return False
            auth_type = str(entry.get("auth_type", "api_key") or "api_key").lower()
            api_mode = str(entry.get("api_mode") or "").lower()
            if not api_mode:
                from hermes_cli.runtime_provider import _detect_api_mode_for_url
                api_mode = str(_detect_api_mode_for_url(str(entry.get("base_url") or "")) or "chat_completions").lower()
        except Exception:  # health: allow BLE001 -- malformed custom-provider config is an unsupported route.
            return False
        return auth_type == "api_key" and api_mode in {"", "chat_completions"}
    if normalized == "custom":
        # The generic custom alias can inherit the active main route. Refuse it when
        # that route is a hosted/chat-only grant before the resolver reads its token.
        try:
            from agent.auxiliary_client import _read_main_provider
            from hermes_cli.config import cfg_get, load_config_readonly
            main_provider = str(_read_main_provider() or "").lower()
            config = load_config_readonly()
            if not isinstance(config, dict):
                return False
            main_api_mode = str(cfg_get(config, "model", "api_mode", default="") or "").lower()
            if main_api_mode and main_api_mode != "chat_completions":
                return False
        except Exception:  # health: allow BLE001 -- route metadata failures must not permit credential resolution.
            return False
        if main_provider in {"openai-codex", "anthropic", "bedrock", "vertex", "gemini", "nous", "xai-oauth"}:
            return False
        return True
    try:
        from providers import get_provider_profile
        profile = get_provider_profile(normalized)
    except Exception:  # health: allow BLE001 -- provider registry discovery fails closed.
        return False
    if profile is None:
        return False
    auth_type = str(getattr(profile, "auth_type", "") or "").lower()
    api_mode = str(getattr(profile, "api_mode", "chat_completions") or "chat_completions").lower()
    return auth_type == "api_key" and api_mode in {"", "chat_completions"}


def _is_openai_compatible_client(client: Any, *, async_mode: bool) -> bool:
    if client is None:
        return False
    # These wrappers implement different wire protocols even if they expose a chat
    # completions facade. Do not let a similarly-shaped attribute turn them into embeddings.
    name = type(client).__name__.lower()
    module = type(client).__module__.lower()
    if any(token in name or token in module for token in ("codexauxiliary", "anthropica", "bedrock", "gemininative", "acp")):
        return False
    embeddings = getattr(client, "embeddings", None)
    create = getattr(embeddings, "create", None)
    if not callable(create):
        return False
    return True


def _profile_scope(profile: str | None):
    if not profile or profile.strip().lower() in {"", "current"}:
        return contextlib.nullcontext()
    try:
        from hermes_cli.profiles import get_profile_dir, profile_exists, validate_profile_name
        validate_profile_name(profile)
        if not profile_exists(profile):
            raise ValueError("unknown profile")
        from agent.runtime_scope import bind_profile_scope
        return bind_profile_scope(get_profile_dir(profile))
    except Exception:  # health: allow BLE001 -- profile lookup/binding failures are returned as safe typed errors.
        raise _error("embedding_unconfigured") from None


def _resolve(provider: str, model: str, *, async_mode: bool):
    try:
        from agent.auxiliary_client import resolve_provider_client
        client, resolved_model = resolve_provider_client(provider, model, async_mode=async_mode)
    except Exception:  # health: allow BLE001 -- resolver and credential failures must never escape raw.
        raise _error("embedding_unconfigured") from None
    if client is None or not resolved_model:
        raise _error("embedding_unconfigured")
    if not _is_openai_compatible_client(client, async_mode=async_mode):
        raise _error("embedding_unsupported")
    return client, str(resolved_model)


def _input_type_supported(input_type: str) -> None:
    if input_type not in {"query", "document"}:
        raise _error("embedding_unsupported")


def _shared_input_semantics_supported(provider: str, model: str, declared: Any) -> bool:
    """Known task-specific model families need a future explicit wire mapping."""
    slug = f"{provider}/{model}".lower()
    if any(part in slug for part in (
        "cohere", "voyage", "jina", "nomic-embed", "bge-", "bge/", "intfloat/e5", "e5-",
        "gte-", "snowflake-arctic-embed", "embeddinggemma",
    )):
        return False
    if declared == "shared":
        return True
    return provider in {"openai", "openai-api"} and model.lower().startswith("text-embedding-")


def _with_dimensions(kwargs: dict[str, Any], dimensions: int | None) -> dict[str, Any]:
    result = dict(kwargs)
    if dimensions is not None:
        result["dimensions"] = dimensions
    return result


def _check_response_dimensions(response: Any, dimensions: int | None) -> None:
    if dimensions is None:
        return
    try:
        vectors = [row.embedding for row in response.data]
        if not vectors or any(len(vector) != dimensions for vector in vectors):
            raise ValueError("dimension mismatch")
    except Exception:  # health: allow BLE001 -- untrusted response shapes are converted to invalid_response.
        raise _error("embedding_invalid_response") from None


def _call_client_for_remaining_time(client: Any, remaining: float) -> Any:
    with_options = getattr(client, "with_options", None)
    if callable(with_options):
        try:
            return with_options(max_retries=0, timeout=remaining)
        except Exception:  # health: allow BLE001 -- SDK option errors are hidden behind the safe API error.
            raise _error("embedding_failed") from None
    return client


def _tag_input_type_handling(response: Any) -> Any:
    """Mark the protocol semantics selected here, never metadata supplied by a provider.

    The OpenAI-compatible embeddings operation has one shared input space and no
    query/document wire parameter. The caller's input_type is preserved in the public
    result while this marker says that the transport sent the input unchanged.
    """
    try:
        object.__setattr__(response, "_hermes_input_type_handling", _INPUT_TYPE_HANDLING)
    except Exception:  # health: allow BLE001 -- provider models may be immutable or custom response objects.
        # The facade treats a missing marker as unsupported; immutable response models
        # therefore fail closed instead of silently claiming shared handling.
        return response
    return response


def _is_auth_failure(provider: str, exc: Exception) -> bool:
    try:
        from agent.auxiliary_error_policy import classify_attempt
        verdict = classify_attempt(exc, provider)
        return bool(verdict.should_rotate_credential)
    except Exception:  # health: allow BLE001 -- error classification is best-effort; failures are not exposed.
        return getattr(exc, "status_code", None) in {401, 403}


def _is_timeout(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
        return True
    try:
        from agent.auxiliary_error_policy import classify_attempt
        return classify_attempt(exc, "").reason.value == "timeout"
    except Exception:  # health: allow BLE001 -- timeout classification is best-effort and non-disclosing.
        return False


def _recover_same_provider(provider: str, exc: Exception) -> bool:
    """Use Hermes' existing credential refresh / pool rotation, without route fallback."""
    if not _is_auth_failure(provider, exc):
        return False
    try:
        from agent.auxiliary_client import _recover_provider_pool, _refresh_provider_credentials
        if _recover_provider_pool(provider, exc):
            return True
        return bool(_refresh_provider_credentials(provider))
    except Exception:  # health: allow BLE001 -- credential recovery failures do not expose provider details.
        return False


def call_embeddings(
    inputs: Sequence[str], *, input_type: str = "document", provider: str | None = None,
    model: str | None = None, dimensions: int | None = None, timeout: float | None = None,
    profile: str | None = None,
) -> tuple[str, str, Any]:
    """Call the selected compatible embeddings endpoint and return provider/model/raw response."""
    _input_type_supported(input_type)
    if not inputs:
        raise _error("embedding_unsupported")
    if timeout is not None:
        if isinstance(timeout, bool):
            raise _error("embedding_timeout")
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            raise _error("embedding_timeout") from None
        if not 0 < timeout <= _MAX_TIMEOUT:
            raise _error("embedding_timeout")
    deadline = None
    error_code = None
    try:
        with _profile_scope(profile):
            selected_provider, selected_model, selected_dimensions, timeout, input_type_handling = _selection(
                provider, model, dimensions, timeout,
            )
            if not _is_supported_route(selected_provider):
                raise _error("embedding_unsupported")
            if not _shared_input_semantics_supported(selected_provider, selected_model, input_type_handling):
                raise _error("embedding_unsupported")
            deadline = time.monotonic() + timeout
            client, actual_model = _resolve(selected_provider, selected_model, async_mode=False)
            kwargs = _with_dimensions({"model": actual_model, "input": list(inputs)}, selected_dimensions)
            for attempt in range(2):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _error("embedding_timeout")
                try:
                    request_client = _call_client_for_remaining_time(client, remaining)
                    response = request_client.embeddings.create(**dict(kwargs, timeout=remaining))
                    _check_response_dimensions(response, selected_dimensions)
                    response = _tag_input_type_handling(response)
                    actual_provider = str(getattr(client, "_hermes_aux_effective_provider", "") or selected_provider)
                    return actual_provider, str(getattr(response, "model", None) or actual_model), response
                except EmbeddingError:
                    raise
                except Exception as exc:  # health: allow BLE001 -- provider SDK failures are sanitized below.
                    if attempt == 0 and _recover_same_provider(selected_provider, exc):
                        if deadline - time.monotonic() <= 0:
                            raise _error("embedding_timeout") from None
                        client, actual_model = _resolve(selected_provider, selected_model, async_mode=False)
                        kwargs["model"] = actual_model
                        continue
                    code = "embedding_timeout" if _is_timeout(exc) else "embedding_failed"
                    raise _error(code) from None
    except EmbeddingError as exc:
        error_code = exc.code
    except Exception as exc:  # health: allow BLE001 -- public boundary sanitizes config/resolver/provider failures.
        error_code = "embedding_timeout" if _is_timeout(exc) or (deadline is not None and deadline - time.monotonic() <= 0) else "embedding_unconfigured"
    if error_code is not None:
        raise _error(error_code)
    raise _error("embedding_failed")


async def async_call_embeddings(
    inputs: Sequence[str], *, input_type: str = "document", provider: str | None = None,
    model: str | None = None, dimensions: int | None = None, timeout: float | None = None,
    profile: str | None = None,
) -> tuple[str, str, Any]:
    """Async sibling. Cancellation propagates unchanged so callers can cancel the request."""
    _input_type_supported(input_type)
    if not inputs:
        raise _error("embedding_unsupported")
    if timeout is not None:
        if isinstance(timeout, bool):
            raise _error("embedding_timeout")
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            raise _error("embedding_timeout") from None
        if not 0 < timeout <= _MAX_TIMEOUT:
            raise _error("embedding_timeout")
    deadline = None
    error_code = None
    try:
        async with _async_profile_scope(profile):
            selected_provider, selected_model, selected_dimensions, timeout, input_type_handling = _selection(
                provider, model, dimensions, timeout,
            )
            if not _is_supported_route(selected_provider):
                raise _error("embedding_unsupported")
            if not _shared_input_semantics_supported(selected_provider, selected_model, input_type_handling):
                raise _error("embedding_unsupported")
            deadline = time.monotonic() + timeout
            client, actual_model = _resolve(selected_provider, selected_model, async_mode=True)
            kwargs = _with_dimensions({"model": actual_model, "input": list(inputs)}, selected_dimensions)
            for attempt in range(2):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _error("embedding_timeout")
                try:
                    request_client = _call_client_for_remaining_time(client, remaining)
                    result = request_client.embeddings.create(**dict(kwargs, timeout=remaining))
                    response = await asyncio.wait_for(result, timeout=remaining) if hasattr(result, "__await__") else result
                    _check_response_dimensions(response, selected_dimensions)
                    response = _tag_input_type_handling(response)
                    actual_provider = str(getattr(client, "_hermes_aux_effective_provider", "") or selected_provider)
                    return actual_provider, str(getattr(response, "model", None) or actual_model), response
                except (asyncio.CancelledError, EmbeddingError):
                    raise
                except Exception as exc:  # health: allow BLE001 -- provider SDK failures are sanitized below.
                    if attempt == 0 and _recover_same_provider(selected_provider, exc):
                        if deadline - time.monotonic() <= 0:
                            raise _error("embedding_timeout") from None
                        client, actual_model = _resolve(selected_provider, selected_model, async_mode=True)
                        kwargs["model"] = actual_model
                        continue
                    code = "embedding_timeout" if _is_timeout(exc) or deadline - time.monotonic() <= 0 else "embedding_failed"
                    raise _error(code) from None
    except asyncio.CancelledError:
        raise
    except EmbeddingError as exc:
        error_code = exc.code
    except Exception as exc:  # health: allow BLE001 -- public boundary sanitizes config/resolver/provider failures.
        error_code = "embedding_timeout" if _is_timeout(exc) or (deadline is not None and deadline - time.monotonic() <= 0) else "embedding_unconfigured"
    if error_code is not None:
        raise _error(error_code)
    raise _error("embedding_failed")


@contextlib.asynccontextmanager
async def _async_profile_scope(profile: str | None):
    scope = _profile_scope(profile)
    with scope:
        yield


__all__ = ["EmbeddingError", "async_call_embeddings", "call_embeddings"]

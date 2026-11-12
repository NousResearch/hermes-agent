"""Authenticated OpenAI-compatible embedding transport for auxiliary consumers.

This module deliberately bypasses the auxiliary chat completion relay. It asks the
central provider resolver for a raw embeddings-capable client and never switches
providers after a request failure.
"""
from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Sequence


class EmbeddingError(RuntimeError):
    """Safe embedding failure with a stable, non-secret machine-readable code."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


_MESSAGES = {
    "embedding_unconfigured": "No supported embedding provider is configured.",
    "embedding_unsupported": "The selected provider does not support embeddings through this transport.",
    "embedding_failed": "The embedding request failed.",
    "embedding_timeout": "The embedding request timed out.",
}
_DEFAULT_PROVIDER = "openai"
_DEFAULT_MODEL = "text-embedding-3-small"
_DEFAULT_TIMEOUT = 60.0
_MAX_TIMEOUT = 300.0
_INPUT_TYPE_HANDLING = "shared"


def _error(code: str) -> EmbeddingError:
    return EmbeddingError(code, _MESSAGES.get(code, _MESSAGES["embedding_failed"]))


def _selection(provider: str | None, model: str | None, dimensions: int | None) -> tuple[str, str, int | None]:
    """Read the profile-scoped embedding route, never the configured chat route."""
    try:
        from hermes_cli.config import cfg_get, load_config_readonly
        configured = load_config_readonly() or {}
        embedding = cfg_get(configured, "auxiliary", "embedding", default={}) or {}
    except Exception:
        configured, embedding = {}, {}
    selected_provider = str(provider or (embedding.get("provider") if isinstance(embedding, dict) else "") or "auto").strip().lower()
    selected_model = str(model or (embedding.get("model") if isinstance(embedding, dict) else "") or "").strip()
    selected_dimensions = dimensions if dimensions is not None else (embedding.get("dimensions") if isinstance(embedding, dict) else None)
    if selected_provider in {"", "auto"}:
        # This is intentionally a fixed embedding-capable route; chat auto-routing can
        # select OAuth or text-only models and is never consulted here.
        selected_provider = _DEFAULT_PROVIDER
        selected_model = selected_model or _DEFAULT_MODEL
    if not selected_model:
        selected_model = _DEFAULT_MODEL if selected_provider == "openai" else ""
    if not selected_model:
        raise _error("embedding_unconfigured")
    if selected_dimensions is not None:
        try:
            selected_dimensions = int(selected_dimensions)
        except (TypeError, ValueError):
            raise _error("embedding_unsupported") from None
        if selected_dimensions <= 0:
            raise _error("embedding_unsupported")
    return selected_provider, selected_model, selected_dimensions


def _is_supported_route(provider: str) -> bool:
    """Reject non-API-key and non-OpenAI-wire routes before resolving credentials."""
    normalized = provider.strip().lower()
    if normalized in {"openai-codex", "codex", "anthropic", "bedrock", "vertex", "gemini", "nous", "xai-oauth"}:
        return False
    if normalized in {"openai", "openrouter"}:
        return True
    if normalized.startswith("custom:"):
        try:
            from hermes_cli.runtime_provider import _get_named_custom_provider
            entry = _get_named_custom_provider(normalized.removeprefix("custom:")) or {}
            api_mode = str(entry.get("api_mode", "chat_completions") or "chat_completions").lower()
        except Exception:
            api_mode = "chat_completions"
        return api_mode in {"", "chat_completions"}
    if normalized == "custom":
        # The generic custom alias can inherit the active main route. Refuse it when
        # that route is a hosted/chat-only grant before the resolver reads its token.
        try:
            from agent.auxiliary_client import _read_main_provider
            main_provider = str(_read_main_provider() or "").lower()
        except Exception:
            main_provider = ""
        if main_provider in {"openai-codex", "anthropic", "bedrock", "vertex", "gemini", "nous", "xai-oauth"}:
            return False
        return True
    try:
        from providers import get_provider_profile
        profile = get_provider_profile(normalized)
    except Exception:
        profile = None
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
    if async_mode and not asyncio.iscoroutinefunction(create):
        # Test doubles and SDK wrappers may expose a callable that returns an awaitable
        # without declaring async; the invocation path validates that below.
        return True
    return True


def _profile_scope(profile: str | None):
    if not profile or profile.strip().lower() in {"", "current"}:
        return contextlib.nullcontext()
    try:
        from hermes_cli.web_server_profiles import _config_profile_scope
        return _config_profile_scope(profile)
    except (ImportError, AttributeError):
        # Embedding callers may already be executing in a profile-bound worker. In that
        # case the caller owns scope binding and this helper must not change global state.
        return contextlib.nullcontext()


def _resolve(provider: str, model: str, *, async_mode: bool):
    from agent.auxiliary_client import resolve_provider_client
    client, resolved_model = resolve_provider_client(provider, model, async_mode=async_mode)
    if client is None or not resolved_model:
        raise _error("embedding_unconfigured")
    if not _is_openai_compatible_client(client, async_mode=async_mode):
        raise _error("embedding_unsupported")
    return client, str(resolved_model)


def _input_type_supported(input_type: str) -> None:
    if input_type not in {"query", "document"}:
        raise _error("embedding_unsupported")


def _with_dimensions(kwargs: dict[str, Any], dimensions: int | None) -> dict[str, Any]:
    result = dict(kwargs)
    if dimensions is not None:
        result["dimensions"] = dimensions
    return result


def _tag_input_type_handling(response: Any) -> Any:
    """Mark the protocol semantics selected here, never metadata supplied by a provider.

    The OpenAI-compatible embeddings operation has one shared input space and no
    query/document wire parameter. The caller's input_type is preserved in the public
    result while this marker says that the transport sent the input unchanged.
    """
    try:
        object.__setattr__(response, "_hermes_input_type_handling", _INPUT_TYPE_HANDLING)
    except Exception:
        # The facade treats a missing marker as unsupported; immutable response models
        # therefore fail closed instead of silently claiming shared handling.
        return response
    return response


def _is_auth_failure(provider: str, exc: Exception) -> bool:
    try:
        from agent.auxiliary_error_policy import classify_attempt
        verdict = classify_attempt(exc, provider)
        return bool(verdict.should_rotate_credential)
    except Exception:
        return getattr(exc, "status_code", None) in {401, 403}


def _is_timeout(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
        return True
    try:
        from agent.auxiliary_error_policy import classify_attempt
        return classify_attempt(exc, "").reason.value == "timeout"
    except Exception:
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
    except Exception:
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
    if timeout is None:
        timeout = _DEFAULT_TIMEOUT
    try:
        timeout = float(timeout)
    except (TypeError, ValueError):
        raise _error("embedding_timeout") from None
    if not 0 < timeout <= _MAX_TIMEOUT:
        raise _error("embedding_timeout")
    with _profile_scope(profile):
        selected_provider, selected_model, selected_dimensions = _selection(provider, model, dimensions)
        if not _is_supported_route(selected_provider):
            raise _error("embedding_unsupported")
        client, actual_model = _resolve(selected_provider, selected_model, async_mode=False)
        kwargs = _with_dimensions({"model": actual_model, "input": list(inputs), "timeout": timeout}, selected_dimensions)
        for attempt in range(2):
            try:
                response = client.embeddings.create(**kwargs)
                response = _tag_input_type_handling(response)
                actual_provider = str(getattr(client, "_hermes_aux_effective_provider", "") or selected_provider)
                return actual_provider, str(getattr(response, "model", None) or actual_model), response
            except Exception as exc:
                if attempt == 0 and _recover_same_provider(selected_provider, exc):
                    client, actual_model = _resolve(selected_provider, selected_model, async_mode=False)
                    kwargs["model"] = actual_model
                    continue
                code = "embedding_timeout" if _is_timeout(exc) else "embedding_failed"
                raise _error(code) from None
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
    if timeout is None:
        timeout = _DEFAULT_TIMEOUT
    try:
        timeout = float(timeout)
    except (TypeError, ValueError):
        raise _error("embedding_timeout") from None
    if not 0 < timeout <= _MAX_TIMEOUT:
        raise _error("embedding_timeout")
    async with _async_profile_scope(profile):
        selected_provider, selected_model, selected_dimensions = _selection(provider, model, dimensions)
        if not _is_supported_route(selected_provider):
            raise _error("embedding_unsupported")
        client, actual_model = _resolve(selected_provider, selected_model, async_mode=True)
        kwargs = _with_dimensions({"model": actual_model, "input": list(inputs), "timeout": timeout}, selected_dimensions)
        for attempt in range(2):
            try:
                result = client.embeddings.create(**kwargs)
                response = await asyncio.wait_for(result, timeout=timeout) if hasattr(result, "__await__") else result
                response = _tag_input_type_handling(response)
                actual_provider = str(getattr(client, "_hermes_aux_effective_provider", "") or selected_provider)
                return actual_provider, str(getattr(response, "model", None) or actual_model), response
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if attempt == 0 and _recover_same_provider(selected_provider, exc):
                    client, actual_model = _resolve(selected_provider, selected_model, async_mode=True)
                    kwargs["model"] = actual_model
                    continue
                code = "embedding_timeout" if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or _is_timeout(exc) else "embedding_failed"
                raise _error(code) from None
    raise _error("embedding_failed")


@contextlib.asynccontextmanager
async def _async_profile_scope(profile: str | None):
    scope = _profile_scope(profile)
    with scope:
        yield


__all__ = ["EmbeddingError", "call_embeddings", "async_call_embeddings"]

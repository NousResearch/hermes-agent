"""Trusted exact-route backend adapter for automatic review checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Callable, Mapping

from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    PooledCredential,
    load_pool,
)
from agent.review_runner import ResolvedReviewRoute


_SUPPORTED_SUBSCRIPTION_PROVIDERS = frozenset({"openai-codex", "anthropic"})
_JSON_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class _SubscriptionCredentialHandle:
    """Backend-only credential lease whose repr never contains token material."""

    provider: str
    entry_id: str
    entry: PooledCredential = field(repr=False, compare=False)


def _active_profile_name() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name

        return str(get_active_profile_name() or "default")
    except Exception:
        return "default"


def resolve_subscription_review_route(
    *,
    provider: str,
    model: str,
    credential_policy: str,
    fallback_policy: str,
    pool_loader: Callable[[str], Any] = load_pool,
    profile_name: Callable[[], str] = _active_profile_name,
) -> ResolvedReviewRoute:
    """Resolve one qualifying OAuth pool entry without another auth path."""

    provider = str(provider or "").strip().lower()
    model = str(model or "").strip()
    if credential_policy != "subscription_oauth_only":
        raise RuntimeError("unsupported review credential policy")
    if fallback_policy != "none":
        raise RuntimeError("review fallback must be disabled")
    if provider not in _SUPPORTED_SUBSCRIPTION_PROVIDERS:
        raise RuntimeError("provider does not expose a supported subscription route")
    if not model:
        raise RuntimeError("an exact review model is required")

    pool = pool_loader(provider)
    entry = pool.select(auth_type=AUTH_TYPE_OAUTH)
    if entry is None:
        raise RuntimeError("subscription OAuth credential is unavailable")
    if entry.auth_type != AUTH_TYPE_OAUTH:
        raise RuntimeError("selected credential is not subscription OAuth")
    if entry.provider != provider or not entry.runtime_api_key:
        raise RuntimeError("selected subscription OAuth route is unusable")

    handle = _SubscriptionCredentialHandle(
        provider=provider,
        entry_id=entry.id,
        entry=entry,
    )
    return ResolvedReviewRoute(
        profile=str(profile_name() or "default"),
        provider=provider,
        model=model,
        credential_kind="subscription_oauth",
        credential_handle=handle,
    )


def _build_exact_subscription_client(
    *,
    provider: str,
    model: str,
    entry: PooledCredential,
    timeout: float,
) -> Any:
    """Build one provider client directly; this function contains no fallback."""

    token = entry.runtime_api_key
    if not token or entry.auth_type != AUTH_TYPE_OAUTH:
        raise RuntimeError("subscription OAuth credential is unavailable")

    if provider == "openai-codex":
        from agent.auxiliary_client import (
            CodexAuxiliaryClient,
            _CODEX_AUX_BASE_URL,
            _codex_cloudflare_headers,
            _create_openai_client,
            _pool_runtime_base_url,
        )

        base_url = (
            _pool_runtime_base_url(entry, _CODEX_AUX_BASE_URL)
            or _CODEX_AUX_BASE_URL
        )
        client = _create_openai_client(
            api_key=token,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
            default_headers=_codex_cloudflare_headers(
                token,
                base_url=base_url,
            ),
        )
        return CodexAuxiliaryClient(client, model)

    if provider == "anthropic":
        from agent.anthropic_adapter import build_anthropic_client
        from agent.auxiliary_client import (
            AnthropicAuxiliaryClient,
            _ANTHROPIC_DEFAULT_BASE_URL,
            _pool_runtime_base_url,
        )

        base_url = (
            _pool_runtime_base_url(entry, _ANTHROPIC_DEFAULT_BASE_URL)
            or _ANTHROPIC_DEFAULT_BASE_URL
        )
        client = build_anthropic_client(token, base_url, timeout=timeout)
        return AnthropicAuxiliaryClient(
            client,
            model,
            token,
            base_url,
            is_oauth=True,
        )

    raise RuntimeError("unsupported subscription review provider")


def _message_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("review provider returned no choices")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise RuntimeError("review provider returned no message")
    if getattr(message, "tool_calls", None):
        raise RuntimeError("review provider attempted a tool call")
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("review provider returned no JSON content")
    return content.strip()


def _parse_payload(content: str) -> Mapping[str, Any]:
    match = _JSON_FENCE_RE.match(content)
    if match is not None:
        content = match.group(1).strip()
    try:
        payload = json.loads(content)
    except (TypeError, ValueError) as error:
        raise RuntimeError("review provider returned invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError("review provider JSON must be an object")
    return payload


def _usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0}
    input_tokens = getattr(
        usage,
        "prompt_tokens",
        getattr(usage, "input_tokens", 0),
    )
    output_tokens = getattr(
        usage,
        "completion_tokens",
        getattr(usage, "output_tokens", 0),
    )
    return {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
    }


def complete_subscription_review(
    *,
    credential_handle: object,
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    tools: list[Any],
    tool_choice: str,
    timeout: float,
    idempotency_key: str,
    client_factory: Callable[..., Any] = _build_exact_subscription_client,
) -> Mapping[str, Any]:
    """Perform exactly one no-tools completion against the resolved route."""

    if tools or tool_choice != "none":
        raise RuntimeError("review completion does not permit tools")
    if not isinstance(credential_handle, _SubscriptionCredentialHandle):
        raise RuntimeError("invalid review credential handle")
    provider = str(provider or "").strip().lower()
    model = str(model or "").strip()
    if (
        provider != credential_handle.provider
        or credential_handle.entry.provider != provider
        or not model
    ):
        raise RuntimeError("review route does not match credential handle")
    if credential_handle.entry.auth_type != AUTH_TYPE_OAUTH:
        raise RuntimeError("review credential is not subscription OAuth")
    if not idempotency_key:
        raise RuntimeError("review checkpoint idempotency key is required")

    client = client_factory(
        provider=provider,
        model=model,
        entry=credential_handle.entry,
        timeout=timeout,
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        payload = dict(_parse_payload(_message_content(response)))
        payload["usage"] = _usage(response)
        payload["request_id"] = str(getattr(response, "id", "") or "")
        return payload
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

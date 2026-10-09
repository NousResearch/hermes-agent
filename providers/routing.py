"""Canonical invocation-route policy for provider runtimes.

This module is deliberately limited to pure provider-domain decisions. It owns
API-mode aliases, endpoint mandates, profile policy, and the runtime-kind
projection used by later transport layers. Credentials, configuration loading,
and model selection do not belong here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Literal, Mapping
from urllib.parse import urlparse, urlunparse

from providers.base import ProviderProfile
from providers.identity import normalize_provider
from providers.route_identity import is_actual_route


RuntimeKind = Literal["http", "external_process", "app_server", "provider_client"]
_RUNTIME_KINDS = frozenset({"http", "external_process", "app_server", "provider_client"})

_API_MODE_ALIASES = {
    "openai": "chat_completions",
    "openai_chat": "chat_completions",
    "openai-chat": "chat_completions",
    "chat-completions": "chat_completions",
    "chat": "chat_completions",
    "anthropic": "anthropic_messages",
    "anthropic-native": "anthropic_messages",
    "anthropic_messages": "anthropic_messages",
    "anthropic-messages": "anthropic_messages",
    "messages": "anthropic_messages",
    "responses": "codex_responses",
    "openai_responses": "codex_responses",
    "openai-responses": "codex_responses",
    "codex": "codex_responses",
    "codex_responses": "codex_responses",
    "codex-responses": "codex_responses",
    "bedrock": "bedrock_converse",
    "bedrock-converse": "bedrock_converse",
    "bedrock_converse": "bedrock_converse",
    "converse": "bedrock_converse",
}

# Provider-owned requirements which cannot be inferred from a profile declaration
# because the provider is implemented by the core (not a model-provider plugin).
_PROVIDER_REQUIREMENTS = {
    "actual": "chat_completions",
    "anthropic": "anthropic_messages",
    "minimax-oauth": "anthropic_messages",
    "openai-codex": "codex_responses",
    "xai": "codex_responses",
    "xai-oauth": "codex_responses",
    "bedrock": "bedrock_converse",
}

_RESPONSES_NATIVE_HOSTS = frozenset({"api.meta.ai", "api.router.com", "api.x.ai"})

_OPENCODE_FAMILY_PATHS = {"opencode-zen": "/zen", "opencode-go": "/zen/go"}


@dataclass(frozen=True)
class InvocationRequest:
    """Inputs needed to resolve one invocation route.

    The request contains only route identity and already-resolved configuration
    values. It intentionally carries no credential or model-catalog state.
    """

    provider: str = ""
    model: str = ""
    base_url: str = ""
    explicit_api_mode: str = ""
    configured_api_mode: str = ""
    configured_provider: str = ""
    openai_runtime: str = ""
    requested_provider: str = ""
    route_options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InvocationRoute:
    """Canonical route selected for one invocation."""

    provider: str
    model: str
    base_url: str
    api_mode: str
    runtime_kind: RuntimeKind
    is_routing_aggregator: bool
    source: str


def canonicalize_api_mode(raw: Any) -> str:
    """Return the canonical spelling for an API mode alias.

    Unknown non-empty values pass through unchanged. This keeps provider-plugin
    modes extensible without making this domain import the transport registry.
    """

    value = str(raw or "").strip()
    if not value:
        return ""
    return _API_MODE_ALIASES.get(value.lower(), value)


def _parsed_url(base_url: str):
    try:
        return urlparse(str(base_url or "").strip())
    except ValueError:
        return None


def _hostname(base_url: str) -> str:
    parsed = _parsed_url(base_url)
    return (parsed.hostname or "").lower().rstrip(".") if parsed is not None else ""


def _exact_or_subdomain(hostname: str, root: str) -> bool:
    return hostname == root or hostname.endswith("." + root)


def _anthropic_path(path: str) -> bool:
    normalized = (path or "").rstrip("/").lower()
    return normalized == "/anthropic" or normalized.endswith("/anthropic") or normalized.endswith("/anthropic/v1")


def _kimi_coding_endpoint(base_url: str, parsed) -> bool:
    if parsed is None:
        return False
    try:
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme.lower() == "https"
        and (parsed.hostname or "").lower().rstrip(".") == "api.kimi.com"
        and port in (None, 443)
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
        and (parsed.path or "").rstrip("/").lower() in {"/coding", "/coding/v1"}
    )


def is_official_openai_host(base_url: str) -> bool:
    """Whether a parsed endpoint belongs to OpenAI's API host family."""
    return _exact_or_subdomain(_hostname(base_url), "api.openai.com")


def endpoint_api_mode(base_url: str) -> str | None:
    """Return a hard API-mode mandate for a URL, or ``None``.

    Host checks use parsed, normalized hostnames. No path substring or attacker
    controlled suffix is accepted as an official endpoint identity.
    """

    parsed = _parsed_url(base_url)
    if parsed is None:
        return None
    hostname = _hostname(base_url)
    path = parsed.path or ""

    if hostname == "api.actual.inc":
        return "chat_completions"
    if hostname == "api.anthropic.com" or _anthropic_path(path):
        return "anthropic_messages"
    if _kimi_coding_endpoint(base_url, parsed):
        return "anthropic_messages"
    if is_official_openai_host(base_url):
        return "codex_responses"
    if hostname in _RESPONSES_NATIVE_HOSTS:
        return "codex_responses"
    if hostname.startswith("bedrock-runtime.") and _exact_or_subdomain(hostname, "amazonaws.com"):
        return "bedrock_converse"
    return None


def _get_profile(provider: str) -> ProviderProfile | None:
    """Resolve the provider profile used for invocation routing.

    ``openai`` is the historical runtime identity for direct OpenAI traffic;
    its canonical registry profile is ``openai-api``. Preserve the runtime
    identity while sourcing endpoint/API-mode facts from that profile.
    """

    from providers import get_provider_profile

    lookup = "openai-api" if normalize_provider(provider) == "openai" else provider
    try:
        return get_provider_profile(lookup)
    except Exception:
        return None


def _providers_match(active: str, configured: str) -> bool:
    configured_name = str(configured or "").strip().lower()
    if not configured_name:
        # A mode without a stored provider is the legacy provider-agnostic form.
        return True
    active_name = str(active or "").strip().lower()
    if active_name == "custom":
        return configured_name == "custom" or configured_name.startswith("custom:")
    if active_name.startswith("custom:"):
        return configured_name in {"custom", active_name}
    return normalize_provider(active_name) == normalize_provider(configured_name)


def _profile_policy(
    profile: ProviderProfile | None, model: str, base_url: str, options: Mapping[str, Any]
) -> str:
    if profile is not None:
        try:
            policy = profile.resolve_route_policy(model, base_url, options=dict(options or {}))
        except Exception:
            policy = None
        if policy:
            return canonicalize_api_mode(policy)
    return ""


def _provider_requirement(provider: str) -> str:
    return _PROVIDER_REQUIREMENTS.get(provider, "")


def normalize_provider_base_url(provider: str, api_mode: str, base_url: str) -> str:
    """Normalize provider-owned endpoint families after the wire protocol is resolved.

    Built-in OpenCode Zen/Go providers heal a stale sibling relay path. Family-prefixed custom
    providers keep their declared relay path but retain the shared /v1 wire normalization.
    """
    url = str(base_url or "").strip().rstrip("/")
    raw_provider = str(provider or "").strip().lower()
    canonical = normalize_provider(raw_provider)
    family = canonical if canonical in _OPENCODE_FAMILY_PATHS else next(
        (name for name in _OPENCODE_FAMILY_PATHS if raw_provider.startswith(name)),
        "",
    )
    if not url or not family:
        return url
    parsed = _parsed_url(url)
    if parsed is None:
        return url
    host = (parsed.hostname or "").lower().rstrip(".")
    official = host == "opencode.ai" or host.endswith(".opencode.ai")
    path = (parsed.path or "").rstrip("/")
    if canonical in _OPENCODE_FAMILY_PATHS and official and re.fullmatch(r"/zen(/go)?(/v1)?", path):
        path = _OPENCODE_FAMILY_PATHS[family] + ("/v1" if path.endswith("/v1") else "")
    if api_mode == "anthropic_messages":
        path = re.sub(r"/v1$", "", path)
    elif official and not path.endswith("/v1"):
        path += "/v1"
    return urlunparse(parsed._replace(path=path))


def _is_external_process_route(profile: ProviderProfile | None, base_url: str) -> bool:
    if profile is not None and profile.auth_type == "external_process":
        return True
    parsed = _parsed_url(base_url)
    scheme = (parsed.scheme or "").lower() if parsed is not None else ""
    return scheme in {"acp", "acp+tcp", "stdio", "process"}


def is_external_process_provider(provider: str, base_url: str = "") -> bool:
    """Whether the canonical provider route uses an external process."""
    profile = _get_profile(normalize_provider(provider))
    return _is_external_process_route(profile, base_url)


def _generic_model_policy(
    provider: str,
    model: str,
    base_url: str,
    profile: ProviderProfile | None,
) -> str:
    """Cross-provider model-family policy retained from the runtime hard cut.

    GPT-5-family models use Responses on ordinary HTTP-compatible providers.
    Provider-specific route policy wins before this fallback. ACP/external
    processes, Nous Portal, anonymous custom endpoints, Copilot's mini-tier
    policy, Actual, and Azure OpenAI keep their own wire semantics.
    """
    model_slug = str(model or "").strip().lower().rsplit("/", 1)[-1]
    if not model_slug.startswith("gpt-5"):
        return ""
    if provider in {"nous", "custom", "copilot", "actual"}:
        return ""
    if _is_external_process_route(profile, base_url):
        return ""
    if _exact_or_subdomain(_hostname(base_url), "openai.azure.com"):
        return ""
    return "codex_responses"


def _profile_default(profile: ProviderProfile | None) -> str:
    if profile is None:
        return ""
    return canonicalize_api_mode(profile.api_mode)


def _is_app_server_request(request: InvocationRequest, provider: str) -> bool:
    runtime = str(request.openai_runtime or "").strip().lower().replace("-", "_")
    if runtime != "codex_app_server":
        return False
    requested_raw = str(request.requested_provider or provider).strip().lower()
    requested = normalize_provider(requested_raw)
    configured_raw = str(request.configured_provider or "").strip().lower()
    eligible = {"openai", "openai-api", "openai-codex"}
    return (
        provider in eligible
        or requested_raw in eligible
        or requested in eligible
        or requested_raw.startswith("custom:")
        or (provider == "custom" and configured_raw in {"openai", "openai-api"})
    )


def _runtime_kind(
    *, profile: ProviderProfile | None, provider: str, base_url: str, api_mode: str,
    app_server: bool = False,
) -> str:
    if app_server:
        return "app_server"
    if _is_external_process_route(profile, base_url):
        return "external_process"
    if api_mode == "bedrock_converse" or (profile is not None and profile.auth_type == "aws_sdk"):
        return "provider_client"
    if profile is not None and type(profile).create_client is not ProviderProfile.create_client:
        return "provider_client"
    return "http"


def _is_routing_aggregator(provider: str, profile: ProviderProfile | None) -> bool:
    if provider.startswith("custom:"):
        return True
    if profile is None:
        return False
    if profile.is_routing_aggregator is not None:
        return bool(profile.is_routing_aggregator)
    return bool(profile.is_aggregator)


def resolve_invocation_route(request: InvocationRequest) -> InvocationRoute:
    """Resolve the canonical provider route using the shared precedence contract."""

    provider_input = str(request.provider or request.requested_provider or "").strip().lower()
    provider = normalize_provider(provider_input)
    model = str(request.model or "").strip()
    profile = _get_profile(provider)
    base_url = str(request.base_url or "").strip() or str(getattr(profile, "base_url", "") or "").strip()

    explicit = canonicalize_api_mode(request.explicit_api_mode)
    if is_actual_route(provider, base_url):
        api_mode, source = "chat_completions", "provider_mandate"
    elif explicit:
        api_mode, source = explicit, "explicit"
    else:
        bedrock_branch = provider == "bedrock" and any(
            bool(request.route_options.get(key)) for key in ("bedrock_anthropic", "bedrock_openai")
        )
        mandated = None if (_is_external_process_route(profile, base_url) or bedrock_branch) else endpoint_api_mode(base_url)
        if mandated:
            api_mode, source = mandated, "endpoint_mandate"
        else:
            policy = _profile_policy(profile, model, base_url, request.route_options) or _provider_requirement(provider)
            if policy:
                api_mode, source = policy, "provider_policy"
            else:
                configured = canonicalize_api_mode(request.configured_api_mode)
                if configured and _providers_match(provider, request.configured_provider):
                    api_mode, source = configured, "configured"
                else:
                    model_policy = _generic_model_policy(provider, model, base_url, profile)
                    if model_policy:
                        api_mode, source = model_policy, "model_policy"
                    else:
                        default = _profile_default(profile)
                        if default:
                            api_mode, source = default, "profile"
                        else:
                            api_mode, source = "chat_completions", "default"

    base_url = normalize_provider_base_url(provider, api_mode, base_url)
    app_server = _is_app_server_request(request, provider) and (
        source != "endpoint_mandate" or api_mode == "codex_responses"
    )
    runtime_kind = _runtime_kind(
        profile=profile,
        provider=provider,
        base_url=base_url,
        api_mode=api_mode,
        app_server=app_server,
    )
    if runtime_kind not in _RUNTIME_KINDS:  # defensive contract guard for future extensions
        runtime_kind = "http"

    return InvocationRoute(
        provider=provider,
        model=model,
        base_url=base_url,
        api_mode=api_mode,
        runtime_kind=runtime_kind,
        is_routing_aggregator=_is_routing_aggregator(provider, profile),
        source=source,
    )


__all__ = [
    "InvocationRequest",
    "InvocationRoute",
    "RuntimeKind",
    "canonicalize_api_mode",
    "endpoint_api_mode",
    "is_external_process_provider",
    "normalize_provider_base_url",
    "resolve_invocation_route",
]

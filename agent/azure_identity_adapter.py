"""Microsoft Entra ID adapter for Microsoft Foundry.

Keyless auth via the `azure-identity` ``DefaultAzureCredential`` chain (env service principal →
workload identity → managed identity → VS Code → Azure CLI → azd → PowerShell → broker). Mirrors
``agent/bedrock_adapter.py``: `azure-identity` is imported lazily (only for ``auth_mode = entra_id``);
``build_token_provider`` returns the zero-arg callable the OpenAI SDK calls before every request
(transparent refresh); consumer helpers are split by purpose so logging paths never mint tokens and
tokens never leak into cache keys; no JWT is persisted (azure-identity caches in-process / OS keychain).
Reference: https://learn.microsoft.com/azure/ai-foundry/foundry-models/how-to/configure-entra-id
"""

from __future__ import annotations

import functools
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Microsoft-documented Foundry inference scope for ALL endpoint shapes. The older cognitiveservices.azure.com
# scope is an ARM control-plane scope rejected for inference by newer resources; override via ``model.entra.scope``.
SCOPE_AI_AZURE_DEFAULT = "https://ai.azure.com/.default"

_AZURE_IDENTITY_FEATURE = "provider.azure_identity"
_INSTALL_MSG = "The 'azure-identity' package is required for Azure AI Foundry Entra ID authentication. "
_LAZY_INSTALL_HINT = (
    "pip install azure-identity manually, or enable lazy installs (security.allow_lazy_installs: true in config.yaml)."
)
_AUTH_HEADERS = ("Authorization", "authorization", "Api-Key", "api-key", "X-Api-Key", "x-api-key")


def has_azure_identity_installed() -> bool:
    """Cheap importability check — does not walk the credential chain."""
    try:
        import azure.identity  # noqa: F401
        return True
    except Exception:
        return False


def _require_azure_identity():
    """Import ``azure.identity``, lazy-installing if allowed; ImportError with an actionable message otherwise."""
    try:
        import azure.identity as _ai
        return _ai
    except ImportError:
        try:
            from tools.lazy_deps import ensure, FeatureUnavailable
        except ImportError as exc:
            raise ImportError(_INSTALL_MSG + "Install it with: pip install azure-identity") from exc
        try:
            ensure(_AZURE_IDENTITY_FEATURE, prompt=False)
        except FeatureUnavailable as exc:
            raise ImportError(_INSTALL_MSG + str(exc)) from exc
        import azure.identity as _ai  # noqa: WPS440 — retry after lazy install
        return _ai


def reset_credential_cache() -> None:
    """Clear the cached ``DefaultAzureCredential`` (tests, profile switches); tolerates a monkeypatched plain function."""
    cache_clear = getattr(build_credential, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()


@dataclass(frozen=True)
class EntraIdentityConfig:
    """Hermes-managed Entra knobs; everything else (tenant, SP secret, federated token file, authority...) flows
    through azure-identity's standard ``AZURE_*`` env vars. ``exclude_interactive_browser`` keeps probes
    non-interactive (the setup wizard never writes it). Frozen: hashable for ``lru_cache``, picklable for workers."""

    scope: str = SCOPE_AI_AZURE_DEFAULT
    exclude_interactive_browser: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", str(self.scope or "").strip() or SCOPE_AI_AZURE_DEFAULT)

    def to_dict(self) -> Dict[str, Any]:
        return {"scope": self.scope, "exclude_interactive_browser": self.exclude_interactive_browser}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]], *, default_scope: Optional[str] = None) -> "EntraIdentityConfig":
        data = data or {}
        return cls(
            scope=str(data.get("scope") or "").strip() or default_scope or SCOPE_AI_AZURE_DEFAULT,
            exclude_interactive_browser=bool(data.get("exclude_interactive_browser", True)),
        )


@functools.lru_cache(maxsize=1)
def build_credential(config: EntraIdentityConfig) -> Any:
    """Cached ``DefaultAzureCredential``. ``maxsize=1`` is intentional: a process uses one ``model.entra.*``
    block at a time. Only Hermes knobs are passed as kwargs; the rest comes from ``AZURE_*`` env vars."""
    ai = _require_azure_identity()
    # SDK default already excludes the browser; only pass the kwarg when opting in.
    kwargs = {} if config.exclude_interactive_browser else {"exclude_interactive_browser_credential": False}
    return ai.DefaultAzureCredential(**kwargs)


def _resolve_config(config: Optional[EntraIdentityConfig], scope: Optional[str], **overrides: Any) -> EntraIdentityConfig:
    if config is not None:
        return config
    return EntraIdentityConfig(scope=(scope or "").strip() or SCOPE_AI_AZURE_DEFAULT, **overrides)


def _install_failure(allow_install: bool) -> Optional[Dict[str, Any]]:
    """None when ``azure.identity`` is importable (lazy-installing if allowed), else ``{"error", "hint"}``."""
    if has_azure_identity_installed():
        return None
    if not allow_install:
        return {"error": "azure-identity not installed", "hint": "pip install azure-identity (or rely on lazy install at first use)"}
    try:
        _require_azure_identity()
    except ImportError as exc:
        return {"error": str(exc) or "azure-identity not installed", "hint": _LAZY_INSTALL_HINT, "exc": exc}
    return None


def build_token_provider(scope: Optional[str] = None, *, config: Optional[EntraIdentityConfig] = None,
                         exclude_interactive_browser: bool = True) -> Callable[[], str]:
    """Zero-arg callable minting a fresh Entra bearer JWT — pass as ``OpenAI(api_key=...)``. Scope precedence:
    ``config.scope`` > ``scope`` kwarg > default. Not picklable: ship the ``EntraIdentityConfig`` and rebuild
    in the worker."""
    ai = _require_azure_identity()
    config = _resolve_config(config, scope, exclude_interactive_browser=exclude_interactive_browser)
    return ai.get_bearer_token_provider(build_credential(config), config.scope)


def _probe_token(config: EntraIdentityConfig, timeout_seconds: float) -> Optional[Dict[str, Any]]:
    """``get_token`` on a daemon thread under a hard deadline → ``{"token"}`` / ``{"error"}`` / None on timeout."""
    result: Dict[str, Any] = {}

    def _probe() -> None:
        try:
            result["token"] = build_credential(config).get_token(config.scope)
        except Exception as exc:
            result["error"] = str(exc)

    thread = threading.Thread(target=_probe, daemon=True)
    thread.start()
    thread.join(timeout=max(0.01, timeout_seconds))
    return None if thread.is_alive() else result


def has_azure_identity_credentials(scope: Optional[str] = None, *, config: Optional[EntraIdentityConfig] = None,
                                   timeout_seconds: float = 10.0, allow_install: bool = True,
                                   **overrides: Any) -> bool:
    """Timeout-bounded probe: can the chain mint a token now? Never raises. ``allow_install=False`` makes it a
    strict "is installed?" check for hot paths (CLI startup) where pip must never run. NOT used by
    ``is_provider_configured()`` (structural, no mint)."""
    failure = _install_failure(allow_install)
    if failure is not None:
        if "exc" in failure:
            logger.debug("azure-identity lazy install unavailable: %s", failure["exc"])
        return False
    result = _probe_token(_resolve_config(config, scope, **overrides), timeout_seconds)
    if result is None:
        logger.debug("Entra token service probe timed out after %ss", timeout_seconds)
        return False
    if "error" in result:
        logger.debug("Entra credential probe failed: %s", result["error"])
        return False
    return bool(getattr(result.get("token"), "token", None))


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _scoped_env(name: str) -> str:
    """Credential-bearing env read via the profile secret scope so a multiplexed profile never reports
    another profile's env-bridged credentials; unscoped CLI probes fall back to plain env."""
    try:
        from agent.secret_scope import get_secret
        return (get_secret(name) or "").strip()
    except Exception:  # UnscopedSecretError, import failure, or any scope error
        return _env(name)


# (label, predicate) for env-var-driven credential sources, in chain order.
_ENV_SOURCE_CHECKS = (
    ("WorkloadIdentityCredential (AZURE_FEDERATED_TOKEN_FILE)", lambda: _scoped_env("AZURE_FEDERATED_TOKEN_FILE")),
    ("EnvironmentCredential (client secret)",
     lambda: _env("AZURE_CLIENT_ID") and _scoped_env("AZURE_CLIENT_SECRET") and _env("AZURE_TENANT_ID")),
    ("ManagedIdentityCredential (IDENTITY_ENDPOINT)", lambda: _env("IDENTITY_ENDPOINT") or _env("MSI_ENDPOINT")),
)


def describe_active_credential(config: Optional[EntraIdentityConfig] = None, *, scope: Optional[str] = None,
                               timeout_seconds: float = 10.0, allow_install: bool = True,
                               **overrides: Any) -> Dict[str, Any]:
    """Doctor / preflight diagnostics. Never raises; ``{"ok": False, "error": ...}`` on failure. azure-identity
    hides the winning inner credential, so this reports a coarse picture (env sources, token expiry) rather
    than a class name; ``AZURE_LOG_LEVEL=DEBUG`` shows the chain."""
    info: Dict[str, Any] = {"ok": False}
    failure = _install_failure(allow_install)
    if failure is not None:
        info["error"], info["hint"] = failure["error"], failure["hint"]
        return info
    config = _resolve_config(config, scope, **overrides)
    info["scope"] = config.scope
    if tenant := _env("AZURE_TENANT_ID"):
        info["tenant_id_env"] = tenant
    info["env_sources"] = [label for label, present in _ENV_SOURCE_CHECKS if present()]
    result = _probe_token(config, timeout_seconds)
    if result is None:
        info["error"] = f"Token probe timed out after {timeout_seconds:.0f}s"
        info["hint"] = ("DefaultAzureCredential can be slow when the token service is unreachable "
                        "or when az login state is stale. Try `az login` or set "
                        "AZURE_CLIENT_ID / AZURE_TENANT_ID / AZURE_CLIENT_SECRET.")
        return info
    if "error" in result:
        info["error"] = result["error"]
        return info
    token = result.get("token")
    if token is None:
        info["error"] = "credential chain exhausted"
        return info
    info["ok"] = True
    info["expires_on"] = getattr(token, "expires_on", None)
    return info


# Consumer-side helpers — split by purpose so logging / cache-key / dashboard paths never mint tokens.
def is_token_provider(value: Any) -> bool:
    """True when ``value`` is a callable Entra token provider (vs. a string API key)."""
    return callable(value) and not isinstance(value, str)


def materialize_bearer_for_http(value: Any) -> str:
    """Mint a fresh Bearer JWT for a manual HTTP request (calls the provider once). Only for sites building
    ``Authorization`` outside the OpenAI SDK; the Anthropic SDK can't take a callable, so
    :func:`build_bearer_http_client` calls this from an httpx hook. ``ValueError`` on an unusable value/empty token."""
    if is_token_provider(value):
        token = value()
        if not isinstance(token, str) or not token:
            raise ValueError("token provider returned empty value")
        return token
    if isinstance(value, str) and value:
        return value
    raise ValueError("no usable api_key / token provider")


def _strip_auth_headers(request: Any) -> None:
    for header_name in _AUTH_HEADERS:
        request.headers.pop(header_name, None)


def build_bearer_http_client(token_provider: Callable[[], str], **httpx_kwargs: Any) -> Any:
    """``httpx.Client`` minting a fresh Entra bearer JWT per outbound request. The Anthropic SDK computes
    ``Authorization`` once at construction, so per-request refresh needs a ``request`` hook: mint (cheap —
    azure-identity caches), strip pre-set auth headers, set ``Authorization: Bearer``. ``httpx_kwargs`` are
    forwarded verbatim (``timeout``, ``transport``...)."""
    if not is_token_provider(token_provider):
        raise ValueError("build_bearer_http_client requires a zero-arg callable token provider")
    import httpx

    def _inject_bearer(request: "httpx.Request") -> None:
        try:
            token = materialize_bearer_for_http(token_provider)
        except ValueError as exc:
            # Chain exhausted / az login expired: strip ALL auth headers (incl. the anthropic_adapter placeholder
            # sentinel) so Azure returns a clean "missing auth" 401 and the sentinel never reaches upstream logs.
            # WARNING so the misconfiguration is visible at default levels.
            logger.warning("Bearer hook: Entra ID token provider returned empty (%s) "
                           "— stripping Authorization headers. Azure will respond 401. "
                           "Run `hermes doctor` or `az login` to recover.", exc)
            _strip_auth_headers(request)
            return
        _strip_auth_headers(request)
        request.headers["Authorization"] = f"Bearer {token}"

    return httpx.Client(event_hooks={"request": [_inject_bearer]}, **httpx_kwargs)


__all__ = [
    "EntraIdentityConfig", "SCOPE_AI_AZURE_DEFAULT", "build_bearer_http_client", "build_credential",
    "build_token_provider", "describe_active_credential", "has_azure_identity_credentials",
    "has_azure_identity_installed", "is_token_provider", "materialize_bearer_for_http", "reset_credential_cache",
]

"""Explicit application inputs for credential pools.

Configuration remains with the application's configuration owner. Protocol
callbacks are supplied by provider implementations; the pool owns their invocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping
from auth.context import CredentialScope
from hermes_constants import get_hermes_home


@dataclass(frozen=True)
class PoolProviderHooks:
    refresh_tokens: Callable[..., Any] | None = field(default=None, repr=False)
    terminal_error: Callable[[BaseException], bool] | None = field(default=None, repr=False)
    resolve_credentials: Callable[..., Any] | None = field(default=None, repr=False)
    token_expiring: Callable[[str], bool] | None = field(default=None, repr=False)
    quarantine_state: Callable[..., Any] | None = field(default=None, repr=False)
    quarantine_pool: Callable[..., Any] | None = field(default=None, repr=False)
    quota_shaped: Callable[..., bool] | None = field(default=None, repr=False)
    refresh_probe_token: Callable[..., Any] | None = field(default=None, repr=False)
    quota_probe: Callable[..., Any] | None = field(default=None, repr=False)
    route_base_url: Callable[..., str] | None = field(default=None, repr=False)
    resolve_external_token: Callable[..., Any] | None = field(default=None, repr=False)
    exchange_external_token: Callable[..., Any] | None = field(default=None, repr=False)
    external_env_vars: tuple[str, ...] = ()


@dataclass(frozen=True)
class PoolEnvironment:
    scope: CredentialScope
    read_config: Callable[[], Mapping[str, Any] | None] = field(repr=False)
    custom_providers: Callable[[Mapping[str, Any]], Any] = field(repr=False)
    read_env: Callable[[], Mapping[str, str]] = field(repr=False)
    secret_source: Callable[[str], str | None] = field(repr=False)
    provider_config: Callable[[str], Any] = field(repr=False)
    provider_configured: Callable[[str], bool] = field(repr=False)
    key_endpoint: Callable[[str, str, str, str], str] = field(repr=False)
    normalize_endpoint: Callable[[Any], str] = field(repr=False)
    provider_hooks: Callable[[str], PoolProviderHooks] = field(repr=False)

    read_secret: Callable[[str], str | None] | None = field(default=None, repr=False)
    entitlement_message: Callable[[str], str] | None = field(default=None, repr=False)
    oauth_user_agent: Callable[[], str] | None = field(default=None, repr=False)

    def require_current_scope(self) -> None:
        if self.scope != CredentialScope(get_hermes_home()):
            raise ValueError("Credential pool belongs to a different profile scope")

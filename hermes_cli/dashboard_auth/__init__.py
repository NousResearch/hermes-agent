"""Dashboard authentication provider framework. The auth gate engages only when the dashboard
binds to a non-loopback host without ``--insecure``; every request must then carry a verified
session from a registered ``DashboardAuthProvider`` (Nous provider is the default; third parties
register theirs via the plugin hook ``ctx.register_dashboard_auth_provider``)."""
from hermes_cli.dashboard_auth.base import (
    DashboardAuthProvider, Session, TokenPrincipal, LoginStart, InvalidCodeError,
    InvalidCredentialsError, ProviderError, RefreshExpiredError, assert_protocol_compliance,
    classify_jwks_lookup_error)
# Dashboard-auth providers are persistent host-owned registrations that deliberately survive a routine
# manager unload (#91701), so the "clean slate" reset must drop the process-global auth registry explicitly
# — otherwise a provider auto-registered during one test leaks into the next.
from hermes_cli.dashboard_auth.registry import (
    register_provider, get_provider, list_providers, list_token_providers,
    list_session_providers, clear_providers)

__all__ = [
    "DashboardAuthProvider", "Session", "TokenPrincipal", "LoginStart", "InvalidCodeError",
    "InvalidCredentialsError", "ProviderError", "RefreshExpiredError", "assert_protocol_compliance",
    "classify_jwks_lookup_error", "register_provider", "get_provider", "list_providers",
    "list_token_providers", "list_session_providers", "clear_providers"]

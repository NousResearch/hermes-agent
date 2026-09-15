"""Shared allowlist of ``/api/*`` paths that bypass dashboard auth. Imported by BOTH gates —
``web_server.auth_middleware`` (loopback / ``--insecure``) and
``dashboard_auth.middleware.gated_auth_middleware`` (OAuth cookie) — so the lists cannot drift
again (a drift once 401'd ``/api/status`` and broke the portal's cookie-less liveness probe).
Keep minimal: every entry must be safe for external uptime probes, the pre-login SPA, and anyone
who ``curl``s the hostname; otherwise gate it and bootstrap after login."""
from __future__ import annotations

PUBLIC_API_PATHS: frozenset[str] = frozenset({
    # Minimal process liveness probe for desktop/backend boot handshakes; avoids
    # gateway config, platform discovery, MCP setup and cold plugin imports.
    "/api/health",
    # Portal wildcard liveness probe (``docs/agent-dashboard-public-url-contract.md``,
    # NAS side): version, gateway state, session count, auth-gate shape. No secrets.
    "/api/status",
    # Read-only config-defaults / schema feeds for the SPA's Config page.
    "/api/config/defaults",
    "/api/config/schema",
    # Read-only model metadata — same shape as public provider catalogs.
    "/api/model/info",
    # Read-only theme + plugin manifests for the dashboard skin engine.
    "/api/dashboard/themes",
    "/api/dashboard/plugins",
    # Chronos managed-cron fire webhook (NAS -> agent). NOT cookie-gated: it
    # carries its own short-lived NAS-minted JWT (purpose=cron_fire), which the
    # handler verifies — the JWT, not this allowlist, is the security boundary.
    "/api/cron/fire"})

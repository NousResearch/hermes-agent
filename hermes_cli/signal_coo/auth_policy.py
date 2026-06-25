"""Runtime auth policy reporting for the Torben COO operator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .runtime_secrets import RuntimeSecretReport, validate_runtime_env_template


OAUTH_NATIVE_PROVIDERS = {
    "openai-codex",
    "xai-oauth",
    "google-oauth",
    "gmail-oauth",
    "calendar-oauth",
}

REQUIRED_FINANCE_MCP_CONNECTORS = (
    "robinhood-agentic-mcp",
    "monarch-money-mcp",
)


@dataclass
class RuntimeAuthReport:
    valid: bool
    strategy: str
    default_provider: str | None
    gtm_provider: str | None
    finance_execution: str
    onepassword_bootstrap: str
    oauth_native_providers: list[str] = field(default_factory=list)
    mcp_native_connectors: list[str] = field(default_factory=list)
    required_mcp_connectors: list[str] = field(default_factory=list)
    missing_mcp_connectors: list[str] = field(default_factory=list)
    disabled_mcp_connectors: list[str] = field(default_factory=list)
    static_secret_bootstrap: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "strategy": self.strategy,
            "default_provider": self.default_provider,
            "gtm_provider": self.gtm_provider,
            "finance_execution": self.finance_execution,
            "onepassword_bootstrap": self.onepassword_bootstrap,
            "oauth_native_providers": self.oauth_native_providers,
            "mcp_native_connectors": self.mcp_native_connectors,
            "required_mcp_connectors": self.required_mcp_connectors,
            "missing_mcp_connectors": self.missing_mcp_connectors,
            "disabled_mcp_connectors": self.disabled_mcp_connectors,
            "static_secret_bootstrap": self.static_secret_bootstrap,
            "warnings": self.warnings,
        }


def _runtime_auth_config(config: dict[str, Any]) -> dict[str, Any]:
    torben = config.get("torben") or {}
    runtime_auth = torben.get("runtime_auth") or {}
    if isinstance(runtime_auth, dict):
        return runtime_auth
    return {}


def _routing(config: dict[str, Any]) -> dict[str, Any]:
    torben = config.get("torben") or {}
    routing = torben.get("model_routing") or {}
    if isinstance(routing, dict):
        return routing
    return {}


def _mcp_servers(config: dict[str, Any]) -> dict[str, Any]:
    servers = config.get("mcp_servers") or {}
    if isinstance(servers, dict):
        return servers
    return {}


def _mcp_enabled(entry: dict[str, Any]) -> bool:
    enabled = entry.get("enabled", True)
    if isinstance(enabled, str):
        return enabled.lower() in {"true", "1", "yes"}
    return bool(enabled)


def evaluate_runtime_auth(
    config: dict[str, Any],
    *,
    optional_env_file: str | Path | None = None,
) -> RuntimeAuthReport:
    runtime_auth = _runtime_auth_config(config)
    routing = _routing(config)
    default_route = routing.get("default") or {}
    gtm_route = routing.get("gtm") or {}

    default_provider = runtime_auth.get("default_provider") or default_route.get("provider")
    gtm_provider = runtime_auth.get("gtm_provider") or gtm_route.get("provider")
    strategy = str(runtime_auth.get("strategy") or "oauth_mcp_native_first")
    onepassword_bootstrap = str(runtime_auth.get("onepassword_bootstrap") or "optional")
    finance_execution = str(runtime_auth.get("finance_execution") or "registered_mcp")

    warnings: list[str] = []
    if default_provider not in OAUTH_NATIVE_PROVIDERS:
        warnings.append(f"default provider is not OAuth-native: {default_provider}")
    if gtm_provider not in OAUTH_NATIVE_PROVIDERS:
        warnings.append(f"gtm provider is not OAuth-native: {gtm_provider}")
    if finance_execution != "registered_mcp":
        warnings.append(f"finance execution should use registered MCP, got: {finance_execution}")
    if onepassword_bootstrap not in {"optional", "fallback", "disabled"}:
        warnings.append(f"onepassword bootstrap should not be required, got: {onepassword_bootstrap}")

    servers = _mcp_servers(config)
    configured_mcp_connectors: list[str] = []
    missing_mcp_connectors: list[str] = []
    disabled_mcp_connectors: list[str] = []
    for connector in REQUIRED_FINANCE_MCP_CONNECTORS:
        entry = servers.get(connector)
        if not isinstance(entry, dict) or not (entry.get("url") or entry.get("command")):
            missing_mcp_connectors.append(connector)
            continue
        if not _mcp_enabled(entry):
            disabled_mcp_connectors.append(connector)
            continue
        if entry.get("auth") != "oauth":
            warnings.append(f"finance MCP connector should use OAuth auth: {connector}")
        configured_mcp_connectors.append(connector)

    if finance_execution == "registered_mcp":
        if missing_mcp_connectors:
            warnings.append(
                "required finance MCP connectors are not configured: "
                + ", ".join(missing_mcp_connectors)
            )
        if disabled_mcp_connectors:
            warnings.append(
                "required finance MCP connectors are disabled: "
                + ", ".join(disabled_mcp_connectors)
            )

    static_secret_report: RuntimeSecretReport | None = None
    if optional_env_file:
        path = Path(optional_env_file)
        if path.exists():
            static_secret_report = validate_runtime_env_template(path, required_keys=[])
            if not static_secret_report.valid:
                warnings.append("optional 1Password env template is invalid")

    return RuntimeAuthReport(
        valid=not warnings,
        strategy=strategy,
        default_provider=str(default_provider) if default_provider else None,
        gtm_provider=str(gtm_provider) if gtm_provider else None,
        finance_execution=finance_execution,
        onepassword_bootstrap=onepassword_bootstrap,
        oauth_native_providers=[
            provider
            for provider in (default_provider, gtm_provider, "google-oauth")
            if provider in OAUTH_NATIVE_PROVIDERS
        ],
        mcp_native_connectors=configured_mcp_connectors,
        required_mcp_connectors=list(REQUIRED_FINANCE_MCP_CONNECTORS),
        missing_mcp_connectors=missing_mcp_connectors,
        disabled_mcp_connectors=disabled_mcp_connectors,
        static_secret_bootstrap=(
            static_secret_report.to_dict()
            if static_secret_report
            else {"valid": None, "reason": "no optional env template checked"}
        ),
        warnings=warnings,
    )

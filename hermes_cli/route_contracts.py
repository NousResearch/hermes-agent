"""Content-safe route contracts for Hermes runtime surfaces.

The goal of this module is not to choose a provider.  Provider/runtime
resolution already lives in :mod:`hermes_cli.runtime_provider`.  This module
turns an already-resolved route into a small, redacted proof object and enforces
route invariants that must fail before a turn starts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any
from urllib.parse import urlparse

SCHEMA_VERSION = 1

_PLATFORM_API_KEY_PREFIXES = (
    "sk-",
    "sk_",
    "org-",
)
_OAUTH_HINT_PREFIXES = (
    "eyj",  # JWT-style OAuth bearer tokens, lower-cased before matching.
    "oat-",
    "oat_",
    "ya29.",
)
_LOCAL_HOSTS = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "host.docker.internal",
}
_SAFE_PATH_SEGMENTS = {
    "api",
    "v1",
    "v2",
    "v3",
    "backend-api",
    "codex",
    "openai",
    "anthropic",
    "chat",
    "completions",
    "responses",
    "models",
    "bedrock",
}
_SAFE_PATH_SEGMENT_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,31}$", re.IGNORECASE)
_SUBSCRIPTION_OR_OAUTH_PROVIDERS = {
    "openai-codex",
    "copilot-acp",
    "google-gemini-cli",
    "minimax-oauth",
    "qwen-oauth",
    "xai-oauth",
}
_GATEWAY_PLATFORMS = {
    "telegram",
    "discord",
    "slack",
    "whatsapp",
    "signal",
    "matrix",
    "mattermost",
    "email",
    "sms",
    "local",
    "msgraph_webhook",
    "wecom_callback",
    "dingtalk",
    "wecom",
    "weixin",
    "feishu",
    "qqbot",
    "bluebubbles",
    "yuanbao",
    "webhook",
    "api_server",
    "homeassistant",
}


@dataclass
class RouteContractError(RuntimeError):
    """Raised when a route violates a hard runtime contract.

    The exception deliberately stores only the redacted proof and a short code;
    never include raw credentials in the message.
    """

    code: str
    surface: str
    proof: dict[str, Any]

    def __str__(self) -> str:
        return f"Route contract violation [{self.code}] on {self.surface}"


def infer_surface(platform: str | None = None, explicit: str | None = None) -> str:
    """Infer a stable route-contract surface label from runtime metadata."""

    if explicit:
        return str(explicit).strip().lower() or "primary"
    value = str(platform or "").strip().lower()
    if value in {"cron", "delegation", "delegate", "subagent", "tui", "dashboard", "primary", "cli"}:
        return "delegation" if value in {"delegate", "subagent"} else value
    if value in _GATEWAY_PLATFORMS:
        return "gateway"
    return "primary"


def credential_kind(api_key: Any) -> str:
    """Classify a credential without returning any credential material."""

    if api_key is None or api_key == "":
        return "none"
    if callable(api_key):
        return "dynamic_token_provider"
    value = str(api_key).strip()
    if not value:
        return "none"
    lowered = value.lower()
    if lowered.startswith(_PLATFORM_API_KEY_PREFIXES):
        return "platform_api_key"
    if lowered.startswith(_OAUTH_HINT_PREFIXES):
        return "oauth_jwt" if lowered.startswith("eyj") else "oauth"
    if value.count(".") >= 2 and len(value) > 32:
        return "oauth_jwt"
    return "opaque_secret"


def _safe_path_hint(path_parts: list[str]) -> str:
    """Return a route-useful path hint without exposing arbitrary path secrets."""

    if not path_parts:
        return ""
    candidate = path_parts[:2]
    lowered = [part.lower() for part in candidate]
    if all(
        part in _SAFE_PATH_SEGMENTS and _SAFE_PATH_SEGMENT_RE.match(part)
        for part in lowered
    ):
        return "/" + "/".join(lowered)
    # Some users put bearer tokens, account ids, or proxy credentials into path
    # segments.  If the segment is not in the known-safe API-route allowlist,
    # expose only the fact that a path exists.
    return "/<redacted-path>"


def _base_url_parts(base_url: Any) -> tuple[str, str, bool]:
    if not isinstance(base_url, str) or not base_url.strip():
        return "", "", False
    parsed = urlparse(base_url.strip())
    host = (parsed.hostname or "").lower()
    path_parts = [part for part in (parsed.path or "").split("/") if part]
    # Keep enough path to distinguish known-safe routes (e.g.
    # /backend-api/codex from /api/v1), but never include query params or
    # arbitrary path segments because custom proxies sometimes place secrets
    # in the URL path.
    path_hint = _safe_path_hint(path_parts)
    is_local = host in _LOCAL_HOSTS or host.startswith("127.") or host.endswith(".local")
    return host, path_hint, is_local


def _normalise_fallback_chain(fallback_model: Any) -> list[dict[str, str]]:
    if isinstance(fallback_model, Mapping):
        candidates: Iterable[Any] = [fallback_model]
    elif isinstance(fallback_model, list | tuple):
        candidates = fallback_model
    else:
        candidates = []
    chain: list[dict[str, str]] = []
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        provider = str(item.get("provider") or "").strip().lower()
        model = str(item.get("model") or "").strip()
        if not provider and not model:
            continue
        chain.append({"provider": provider, "model": model})
    return chain


def _runtime_label(provider: str, api_mode: str, acp_command: Any, base_url_is_local: bool) -> str:
    if acp_command:
        return "acp"
    if api_mode == "codex_app_server":
        return "codex_app_server"
    if provider == "openai-codex" or api_mode == "codex_responses":
        return "codex_api"
    if api_mode == "anthropic_messages":
        return "anthropic_messages"
    if api_mode == "bedrock_converse":
        return "bedrock_converse"
    if base_url_is_local:
        return "local_http"
    return api_mode or "unknown"


def _auth_surface(provider: str, kind: str, acp_command: Any) -> str:
    if acp_command:
        return "local_acp"
    if kind == "none":
        return "none"
    if kind == "dynamic_token_provider":
        return "dynamic_token"
    if kind == "platform_api_key":
        return "platform_api_key"
    if provider in _SUBSCRIPTION_OR_OAUTH_PROVIDERS or kind.startswith("oauth"):
        return "oauth"
    return "api_key"


def _cost_surface(provider: str, api_mode: str, auth_surface: str, base_url_is_local: bool) -> str:
    if base_url_is_local:
        return "local"
    if provider == "openai-codex" or api_mode == "codex_app_server":
        return "subscription" if auth_surface != "platform_api_key" else "per_token_api"
    if auth_surface in {"oauth", "local_acp"} and provider in _SUBSCRIPTION_OR_OAUTH_PROVIDERS:
        return "subscription"
    if auth_surface == "none":
        return "none"
    if provider == "bedrock" or api_mode == "bedrock_converse":
        return "cloud_metered"
    return "per_token_api"


def _setup_mode(
    provider: str,
    api_mode: str,
    auth_surface: str,
    cost_surface: str,
    base_url_is_local: bool,
    acp_command: Any,
) -> str:
    if api_mode == "codex_app_server":
        return "codex_app_server_opt_in"
    if (
        provider == "openai-codex"
        and api_mode == "codex_responses"
        and auth_surface == "oauth"
        and cost_surface == "subscription"
    ):
        return "hermes_recommended_codex_oauth"
    if acp_command:
        return "external_cli_acp"
    if base_url_is_local:
        return "local_endpoint"
    if cost_surface == "per_token_api":
        return "metered_api"
    if cost_surface == "subscription":
        return "subscription_oauth"
    return "custom"


def _route_owner(api_mode: str, acp_command: Any) -> str:
    if api_mode == "codex_app_server":
        return "hermes_outer_codex_inner"
    if acp_command:
        return "external_cli"
    return "hermes"


def _requires_external_cli(api_mode: str, acp_command: Any) -> bool:
    return bool(acp_command) or api_mode == "codex_app_server"


def _reasoning_effort(reasoning_config: Any) -> str:
    if isinstance(reasoning_config, Mapping):
        if reasoning_config.get("enabled") is False:
            return "disabled"
        return str(reasoning_config.get("effort") or "").strip()
    if isinstance(reasoning_config, str):
        return reasoning_config.strip()
    return ""


def _policy_allowed_cost_surfaces(policy: Mapping[str, Any] | None) -> set[str] | None:
    if not isinstance(policy, Mapping):
        return None
    raw = policy.get("allowed_cost_surfaces")
    if not raw:
        return None
    if isinstance(raw, str):
        return {raw.strip()}
    if isinstance(raw, Iterable):
        return {str(item).strip() for item in raw if str(item).strip()}
    return None


def build_agent_route_proof(
    *,
    surface: str | None = None,
    platform: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    api_mode: str | None = None,
    base_url: Any = None,
    api_key: Any = None,
    acp_command: Any = None,
    acp_args: Iterable[Any] | None = None,
    reasoning_config: Any = None,
    service_tier: str | None = None,
    request_overrides: Mapping[str, Any] | None = None,
    fallback_model: Any = None,
    fallback_activated: bool = False,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a redacted proof of the effective route contract."""

    resolved_surface = infer_surface(platform=platform, explicit=surface)
    normalized_provider = str(provider or "").strip().lower()
    normalized_api_mode = str(api_mode or "").strip().lower()
    host, path_hint, is_local = _base_url_parts(base_url)
    kind = credential_kind(api_key)
    auth = _auth_surface(normalized_provider, kind, acp_command)
    runtime = _runtime_label(normalized_provider, normalized_api_mode, acp_command, is_local)
    cost = _cost_surface(normalized_provider, normalized_api_mode, auth, is_local)
    setup_mode = _setup_mode(
        normalized_provider,
        normalized_api_mode,
        auth,
        cost,
        is_local,
        acp_command,
    )
    fallback_chain = _normalise_fallback_chain(fallback_model)

    proof: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "surface": resolved_surface,
        "platform": platform or "",
        "provider": normalized_provider,
        "model": str(model or ""),
        "api_mode": normalized_api_mode,
        "runtime": runtime,
        "setup_mode": setup_mode,
        "route_owner": _route_owner(normalized_api_mode, acp_command),
        "requires_external_cli": _requires_external_cli(normalized_api_mode, acp_command),
        "base_url_host": host,
        "base_url_path_hint": path_hint,
        "base_url_is_local": is_local,
        "credential_present": kind != "none",
        "credential_kind": kind,
        "auth_surface": auth,
        "cost_surface": cost,
        "reasoning_effort": _reasoning_effort(reasoning_config),
        "service_tier": str(service_tier or ""),
        "request_override_keys": sorted(str(k) for k in (request_overrides or {}).keys()),
        "fallback_chain_count": len(fallback_chain),
        "fallback_chain": fallback_chain,
        "fallback_activated": bool(fallback_activated),
        "acp_command_present": bool(acp_command),
        "acp_arg_count": len(list(acp_args or [])),
    }
    proof["contract"] = _evaluate_contract(proof, policy=policy)
    return proof


def _evaluate_contract(proof: Mapping[str, Any], *, policy: Mapping[str, Any] | None) -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    provider = str(proof.get("provider") or "")
    api_mode = str(proof.get("api_mode") or "")
    kind = str(proof.get("credential_kind") or "")
    auth = str(proof.get("auth_surface") or "")
    surface = str(proof.get("surface") or "")
    cost = str(proof.get("cost_surface") or "")

    if api_mode == "codex_app_server":
        if provider not in {"openai", "openai-codex"}:
            violations.append({
                "code": "codex_app_server_provider_mismatch",
                "message": "codex_app_server is only valid for openai/openai-codex routes",
            })
        if kind == "platform_api_key" or auth == "platform_api_key":
            violations.append({
                "code": "codex_app_server_requires_oauth",
                "message": "codex_app_server must use OAuth/subscription credentials, not OpenAI Platform API keys",
            })

    allowed_costs = _policy_allowed_cost_surfaces(policy)
    if allowed_costs is not None and cost and cost not in allowed_costs:
        violations.append({
            "code": f"{cost}_forbidden" if cost else "cost_surface_forbidden",
            "message": f"{surface} route cost surface '{cost}' is outside the allowed set",
        })

    return {
        "status": "blocked" if violations else "ok",
        "violations": violations,
    }


def verify_agent_route_contract(*, raise_on_error: bool = True, **kwargs: Any) -> dict[str, Any]:
    """Build a route proof and optionally raise on hard violations."""

    proof = build_agent_route_proof(**kwargs)
    violations = proof.get("contract", {}).get("violations") or []
    if violations and raise_on_error:
        first = violations[0]
        raise RouteContractError(
            code=str(first.get("code") or "route_contract_violation"),
            surface=str(proof.get("surface") or "unknown"),
            proof=proof,
        )
    return proof


def proof_from_agent(agent: Any, *, surface: str | None = None, policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a route proof from an AIAgent-like object."""

    return verify_agent_route_contract(
        surface=surface or getattr(agent, "route_surface", None),
        platform=getattr(agent, "platform", None),
        provider=getattr(agent, "provider", None),
        model=getattr(agent, "model", None),
        api_mode=getattr(agent, "api_mode", None),
        base_url=getattr(agent, "base_url", None),
        api_key=getattr(agent, "api_key", None),
        acp_command=getattr(agent, "acp_command", None),
        acp_args=getattr(agent, "acp_args", None),
        reasoning_config=getattr(agent, "reasoning_config", None),
        service_tier=getattr(agent, "service_tier", None),
        request_overrides=getattr(agent, "request_overrides", None),
        fallback_model=getattr(agent, "_fallback_chain", None) or getattr(agent, "_fallback_model", None),
        fallback_activated=getattr(agent, "_fallback_activated", False),
        policy=policy,
        raise_on_error=True,
    )


def _tier_record(tier: int, name: str, status: str, required_action: str) -> dict[str, Any]:
    return {
        "tier": tier,
        "name": name,
        "status": status,
        "required_action": required_action,
    }


def build_route_hardening_plan(route_proof: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Map a content-safe route proof onto the broad seven-tier hardening plan.

    This is a planning/control-plane surface, not a provider resolver.  It lets
    operators see how a route choice affects each tier without leaking auth
    material or raw prompts.
    """

    proof = dict(route_proof or build_agent_route_proof(surface="primary"))
    setup_mode = str(proof.get("setup_mode") or "custom")
    api_mode = str(proof.get("api_mode") or "")
    raw_contract = proof.get("contract")
    contract: Mapping[str, Any] = raw_contract if isinstance(raw_contract, Mapping) else {}
    contract_status = str(contract.get("status") or "ok")
    is_app_server = (
        setup_mode == "codex_app_server_opt_in"
        or api_mode == "codex_app_server"
    )
    is_recommended = setup_mode == "hermes_recommended_codex_oauth"
    tier2_status = (
        "blocked"
        if contract_status == "blocked"
        else "ok" if is_recommended else "attention"
    )
    tier2_action = (
        "keep as canonical Hermes-recommended Codex OAuth/API-call route"
        if is_recommended
        else "document and prove app-server as an explicit opt-in exception"
        if is_app_server
        else "prove this non-default route explicitly before work starts"
    )
    tier5_status = "attention" if is_app_server else "ok"
    tier6_status = "attention" if is_app_server else "ok"
    tier7_status = "blocked" if contract_status == "blocked" else "ok"

    tiers = [
        _tier_record(
            1,
            "Reliability / source control",
            "ok",
            "preserve explicit local commits and do not mutate source control as part of route selection",
        ),
        _tier_record(2, "Route invariants", tier2_status, tier2_action),
        _tier_record(
            3,
            "Trace / replay",
            "ok",
            "persist route_proof metadata in traces and classify blocked contracts as route_contract failures",
        ),
        _tier_record(
            4,
            "Context hygiene",
            "ok",
            "keep route proof in sidecar/control-plane metadata, not model-visible prompt content",
        ),
        _tier_record(
            5,
            "Skill lifecycle",
            tier5_status,
            "keep Hermes agent-loop skill tools native"
            if not is_app_server
            else "prove MCP skill discovery/skill_view bridge and background review downgrade before relying on app-server skills",
        ),
        _tier_record(
            6,
            "Autonomous loops",
            tier6_status,
            "prove delegation, cron, TUI, dashboard, and gateway routes independently"
            if not is_app_server
            else "scope cron/kanban/goal workers to profiles with explicit app-server route proof or keep them on default runtime",
        ),
        _tier_record(
            7,
            "Security",
            tier7_status,
            "forbid OpenAI Platform API-key fallback for Codex OAuth/subscription routes and expose only redacted metadata",
        ),
    ]

    active_route = {
        "surface": proof.get("surface") or "",
        "provider": proof.get("provider") or "",
        "model": proof.get("model") or "",
        "api_mode": proof.get("api_mode") or "",
        "runtime": proof.get("runtime") or "",
        "setup_mode": setup_mode,
        "route_owner": proof.get("route_owner") or "",
        "requires_external_cli": bool(proof.get("requires_external_cli")),
        "auth_surface": proof.get("auth_surface") or "",
        "cost_surface": proof.get("cost_surface") or "",
        "contract_status": contract_status,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "content_policy": "metadata_only",
        "setup_mode": setup_mode,
        "active_route": active_route,
        "recommended_baseline": {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "openai_runtime": "auto",
            "route_owner": "hermes",
            "auth_surface": "oauth",
            "cost_surface": "subscription",
            "requires_external_cli": False,
        },
        "tier_count": len(tiers),
        "tiers": tiers,
    }


__all__ = [
    "RouteContractError",
    "build_agent_route_proof",
    "build_route_hardening_plan",
    "credential_kind",
    "infer_surface",
    "proof_from_agent",
    "verify_agent_route_contract",
]

"""AgentCyber task-route classification and model-routing helpers."""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class CyberRoute(str, Enum):
    """AgentCyber route labels from the Hermes Cyber Edition mandate."""

    GENERAL = "general"
    CYBER_LAB = "cyber_lab"
    IR_BREAKGLASS = "ir_breakglass"
    MALWARE_RE = "malware_re"
    OSINT = "osint"
    CREDENTIALS_SENSITIVE = "credentials_sensitive"
    DESTRUCTIVE_HIGH_RISK = "destructive_high_risk"


class ProviderPreference(str, Enum):
    """Provider-family preference for a classified task."""

    DEFAULT = "default"
    LOCAL_OPEN_WEIGHT = "local_open_weight"
    HOSTED = "hosted"


@dataclass(frozen=True)
class CyberRouteDecision:
    """Classification result for a single operator request."""

    route: CyberRoute
    provider_preference: ProviderPreference
    reason: str
    requires_hosted_secret_confirmation: bool = False
    explicit_override: str | None = None


@dataclass(frozen=True)
class AgentCyberRuntimeRoute:
    """A configured local/open-weight runtime for cyber-sensitive turns."""

    provider: str
    model: str
    base_url: str = ""
    api_key: str = ""
    api_mode: str = "chat_completions"
    context_length: int | None = None


@dataclass(frozen=True)
class AgentCyberRouteAction:
    """Result of enforcing the AgentCyber model-routing guard."""

    action: str
    reason: str
    runtime: AgentCyberRuntimeRoute | None = None


_LOCAL_OVERRIDE_PHRASES = (
    "use local model",
    "use a local model",
    "use local",
    "local model",
)
_AZURE_OVERRIDE_PHRASES = (
    "use azure",
    "use microsoft",
    "azure model",
)
_CYBER_OVERRIDE_PHRASES = (
    "use cyber route",
    "cyber route",
    "cyber model",
)

_DESTRUCTIVE_TERMS = (
    "wipe",
    "destroy",
    "delete all",
    "rm -rf",
    "format disk",
    "factory reset",
    "reset the firewall",
    "password reset",
    "rotate credentials",
    "disable account",
)
_BREAKGLASS_TERMS = (
    "locked out",
    "password changed",
    "can't access",
    "cant access",
    "get me back in",
    "get back into",
    "emergency access",
    "incident response",
    "incident timeline",
    "access recovery",
    "recover access",
    "restore access",
    "restore vm",
)
_CREDENTIAL_TERMS = (
    "credential",
    "password",
    "secret",
    "api key",
    "token",
    "ssh key",
    "mfa",
)
_MALWARE_TERMS = (
    "malware",
    "worm",
    "ransomware",
    "ioc",
    "reverse engineering",
    "reverse engineer",
    "sample",
    "sandbox",
)
_EXPLOIT_TERMS = (
    "exploit",
    "payload",
    "c2",
    "persistence",
    "evasion",
    "fingerprinting",
    "vulnerability",
    "scan",
    "nmap",
)
_LAB_SCOPE_TERMS = (
    "owned",
    "lab",
    "vm ",
    "proxmox",
    "breaking circuits",
    "bc-owned",
    "my server",
)
_OSINT_TERMS = (
    "osint",
    "public records",
    "public company",
    "domain research",
    "recon",
    "open source intelligence",
)


def classify_cyber_route(message: str) -> CyberRouteDecision:
    """Classify a user request into the first AgentCyber routing slice.

    The route captures task type/safety sensitivity. The provider preference is
    a non-executing recommendation: sensitive cyber routes prefer local or
    open-weight models, ordinary work stays on the configured default, and
    explicit operator overrides are recorded without hiding the underlying
    route.
    """

    text = message.lower()
    override = _explicit_override(text)
    route, reason = _classify_route(text)
    provider_preference = _provider_preference(route, override)
    requires_hosted_secret_confirmation = route in {
        CyberRoute.CREDENTIALS_SENSITIVE,
        CyberRoute.IR_BREAKGLASS,
    }

    return CyberRouteDecision(
        route=route,
        provider_preference=provider_preference,
        reason=reason,
        requires_hosted_secret_confirmation=requires_hosted_secret_confirmation,
        explicit_override=override,
    )


def _explicit_override(text: str) -> str | None:
    if _contains_any(text, _LOCAL_OVERRIDE_PHRASES):
        return "local"
    if _contains_any(text, _AZURE_OVERRIDE_PHRASES):
        return "azure"
    if _contains_any(text, _CYBER_OVERRIDE_PHRASES):
        return "cyber"
    return None


def _classify_route(text: str) -> tuple[CyberRoute, str]:
    if _contains_any(text, _DESTRUCTIVE_TERMS):
        return CyberRoute.DESTRUCTIVE_HIGH_RISK, "destructive or high-impact operation"
    if _contains_any(text, _BREAKGLASS_TERMS):
        return CyberRoute.IR_BREAKGLASS, "lockout or incident recovery request"
    if _contains_any(text, _CREDENTIAL_TERMS):
        return CyberRoute.CREDENTIALS_SENSITIVE, "credential or secret handling request"
    if _contains_any(text, _MALWARE_TERMS):
        return CyberRoute.MALWARE_RE, "malware or reverse-engineering request"
    if _contains_any(text, _OSINT_TERMS):
        return CyberRoute.OSINT, "OSINT or public-source investigation"
    if _contains_any(text, _EXPLOIT_TERMS) and _contains_any(text, _LAB_SCOPE_TERMS):
        return CyberRoute.CYBER_LAB, "authorized lab security testing request"
    if _contains_any(text, _EXPLOIT_TERMS):
        return CyberRoute.CYBER_LAB, "cyber-sensitive testing request needing scope tracking"
    return CyberRoute.GENERAL, "ordinary general task"


def _provider_preference(route: CyberRoute, override: str | None) -> ProviderPreference:
    if override in {"local", "cyber"}:
        return ProviderPreference.LOCAL_OPEN_WEIGHT
    if override == "azure":
        return ProviderPreference.HOSTED
    if route in {
        CyberRoute.CYBER_LAB,
        CyberRoute.IR_BREAKGLASS,
        CyberRoute.MALWARE_RE,
        CyberRoute.CREDENTIALS_SENSITIVE,
        CyberRoute.DESTRUCTIVE_HIGH_RISK,
    }:
        return ProviderPreference.LOCAL_OPEN_WEIGHT
    return ProviderPreference.DEFAULT


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)

_LOCAL_PROVIDER_NAMES = {"ollama", "lmstudio", "llama.cpp", "llamacpp", "local", "vllm", "text-generation-webui"}
_OPEN_WEIGHT_MODEL_TERMS = (
    "llama", "qwen", "mistral", "mixtral", "deepseek", "gemma", "phi", "yi-", "nous", "hermes", "glm", "kimi"
)


def should_prefer_local_open_weight(decision: CyberRouteDecision | None) -> bool:
    if decision is None:
        return False
    return decision.provider_preference == ProviderPreference.LOCAL_OPEN_WEIGHT


def is_local_or_open_weight_runtime(
    provider: str | None,
    base_url: str | None,
    model: str | None,
    *,
    allow_hosted_open_weight: bool = False,
) -> bool:
    """Return True when the current runtime satisfies AgentCyber sensitive routing."""

    provider_norm = (provider or "").strip().lower()
    base = (base_url or "").strip()
    model_norm = (model or "").strip().lower()
    if provider_norm in _LOCAL_PROVIDER_NAMES:
        return True
    if _is_local_base_url(base):
        return True
    if allow_hosted_open_weight and any(term in model_norm for term in _OPEN_WEIGHT_MODEL_TERMS):
        return True
    return False


def resolve_agentcyber_runtime_route(config: dict[str, Any] | None = None) -> AgentCyberRuntimeRoute | None:
    """Resolve the configured local/open-weight runtime for sensitive cyber turns."""

    cfg = config if isinstance(config, dict) else _load_config_quietly()
    cyber_cfg = cfg.get("agent_cyber", {}) if isinstance(cfg, dict) else {}
    if not isinstance(cyber_cfg, dict):
        cyber_cfg = {}
    routing_cfg = cyber_cfg.get("routing", {})
    if not isinstance(routing_cfg, dict):
        routing_cfg = {}

    route_cfg = (
        routing_cfg.get("local_open_weight")
        or routing_cfg.get("sensitive_model")
        or cyber_cfg.get("local_open_weight")
        or {}
    )
    if isinstance(route_cfg, str):
        route_cfg = {"model": route_cfg}
    if not isinstance(route_cfg, dict):
        route_cfg = {}

    provider = _expand_env(route_cfg.get("provider") or os.getenv("HERMES_AGENTCYBER_LOCAL_PROVIDER") or "")
    model = _expand_env(route_cfg.get("model") or os.getenv("HERMES_AGENTCYBER_LOCAL_MODEL") or "")
    base_url = _expand_env(route_cfg.get("base_url") or os.getenv("HERMES_AGENTCYBER_LOCAL_BASE_URL") or "")
    api_key = _expand_env(route_cfg.get("api_key") or "")
    key_env = route_cfg.get("api_key_env") or route_cfg.get("key_env") or ""
    if key_env and not api_key:
        api_key = os.getenv(str(key_env), "")
    api_mode = str(route_cfg.get("api_mode") or os.getenv("HERMES_AGENTCYBER_LOCAL_API_MODE") or "chat_completions").strip()
    context_length = route_cfg.get("context_length")
    try:
        context_length = int(context_length) if context_length is not None else None
    except (TypeError, ValueError):
        context_length = None

    if not model and not base_url and not provider:
        return None
    if not provider:
        provider = "custom" if base_url else "ollama"
    return AgentCyberRuntimeRoute(
        provider=provider.strip().lower(),
        model=model.strip(),
        base_url=base_url.strip().rstrip("/"),
        api_key=api_key,
        api_mode=api_mode if api_mode in {"chat_completions", "anthropic_messages", "codex_responses"} else "chat_completions",
        context_length=context_length,
    )


def agentcyber_routing_action(agent: Any, config: dict[str, Any] | None = None) -> AgentCyberRouteAction:
    """Decide whether this turn needs a transient AgentCyber runtime switch."""

    decision = getattr(agent, "_current_cyber_route_decision", None)
    if not should_prefer_local_open_weight(decision):
        return AgentCyberRouteAction("keep", "route does not require local/open-weight runtime")

    cfg = config if isinstance(config, dict) else _load_config_quietly()
    cyber_cfg = cfg.get("agent_cyber", {}) if isinstance(cfg, dict) else {}
    if not isinstance(cyber_cfg, dict):
        cyber_cfg = {}
    routing_cfg = cyber_cfg.get("routing", {}) if isinstance(cyber_cfg.get("routing", {}), dict) else {}
    if routing_cfg.get("enabled", True) is False:
        return AgentCyberRouteAction("keep", "AgentCyber model routing disabled")

    if getattr(decision, "explicit_override", None) == "azure" and routing_cfg.get("allow_hosted_override", True):
        return AgentCyberRouteAction("keep", "explicit hosted/Azure override")

    allow_hosted_open_weight = bool(routing_cfg.get("allow_hosted_open_weight", False))
    if is_local_or_open_weight_runtime(agent.provider, agent.base_url, agent.model, allow_hosted_open_weight=allow_hosted_open_weight):
        return AgentCyberRouteAction("keep", "current runtime satisfies local/open-weight guard")

    runtime = resolve_agentcyber_runtime_route(cfg)
    if runtime is not None:
        return AgentCyberRouteAction("switch", "switching sensitive cyber turn to configured local/open-weight runtime", runtime)

    if routing_cfg.get("require_local_for_sensitive", True):
        return AgentCyberRouteAction("block", "sensitive AgentCyber route has no configured local/open-weight runtime")
    return AgentCyberRouteAction("keep", "no local/open-weight runtime configured; fail-open allowed by config")


def apply_agentcyber_route_guard(agent: Any) -> str | None:
    """Apply per-turn AgentCyber model routing. Return user-facing block text if blocked."""

    cfg = _load_config_quietly()
    agent._agentcyber_config = cfg
    action = agentcyber_routing_action(agent, cfg)
    metadata = getattr(agent, "_current_cyber_route_metadata", None)
    if isinstance(metadata, dict):
        metadata["routing_action"] = action.action
        metadata["routing_reason"] = action.reason
        if action.runtime:
            metadata["routed_provider"] = action.runtime.provider
            metadata["routed_model"] = action.runtime.model
            metadata["routed_base_url"] = action.runtime.base_url

    if action.action == "block":
        return (
            "AgentCyber stopped before sending this sensitive turn to the hosted model. "
            "No local/open-weight cyber runtime is configured. Configure "
            "agent_cyber.routing.local_open_weight in config.yaml or set "
            "HERMES_AGENTCYBER_LOCAL_BASE_URL and HERMES_AGENTCYBER_LOCAL_MODEL, then retry."
        )
    if action.action == "switch" and action.runtime is not None:
        _switch_agent_to_agentcyber_runtime(agent, action.runtime)
    return None


def restore_agentcyber_route_runtime(agent: Any) -> bool:
    """Restore the pre-AgentCyber runtime at the start of the next turn."""

    snapshot = getattr(agent, "_agentcyber_pre_route_runtime", None)
    if not snapshot:
        return False
    try:
        _restore_runtime_snapshot(agent, snapshot, reason="agentcyber_restore")
        agent._agentcyber_pre_route_runtime = None
        agent._agentcyber_route_active = False
        return True
    except Exception as exc:
        logger.warning("AgentCyber runtime restore failed: %s", exc)
        return False


def _switch_agent_to_agentcyber_runtime(agent: Any, runtime: AgentCyberRuntimeRoute) -> None:
    if not getattr(agent, "_agentcyber_route_active", False):
        agent._agentcyber_pre_route_runtime = _runtime_snapshot(agent)
    agent._agentcyber_route_active = True

    agent.provider = runtime.provider
    agent.model = runtime.model or agent.model
    agent.base_url = runtime.base_url or agent.base_url
    agent.api_key = runtime.api_key or getattr(agent, "api_key", "")
    agent.api_mode = runtime.api_mode or "chat_completions"
    if hasattr(agent, "_transport_cache"):
        agent._transport_cache.clear()

    if agent.api_mode == "anthropic_messages":
        from agent.anthropic_adapter import build_anthropic_client, _is_oauth_token
        agent._anthropic_api_key = agent.api_key
        agent._anthropic_base_url = agent.base_url
        agent._anthropic_client = build_anthropic_client(agent.api_key, agent.base_url)
        agent._is_anthropic_oauth = _is_oauth_token(agent.api_key) if agent.provider == "anthropic" and isinstance(agent.api_key, str) else False
        agent.client = None
        agent._client_kwargs = {}
    else:
        client_kwargs = {"api_key": agent.api_key or "dummy-key", "base_url": agent.base_url}
        agent._client_kwargs = client_kwargs
        agent.client = agent._create_openai_client(dict(client_kwargs), reason="agentcyber_route", shared=True)

    try:
        agent._use_prompt_caching, agent._use_native_cache_layout = agent._anthropic_prompt_cache_policy(
            provider=agent.provider,
            base_url=agent.base_url,
            api_mode=agent.api_mode,
            model=agent.model,
        )
    except Exception:
        agent._use_prompt_caching = False
        agent._use_native_cache_layout = False

    _update_context_compressor(agent, context_length=runtime.context_length)


def _runtime_snapshot(agent: Any) -> dict[str, Any]:
    cc = getattr(agent, "context_compressor", None)
    return {
        "model": getattr(agent, "model", ""),
        "provider": getattr(agent, "provider", ""),
        "base_url": getattr(agent, "base_url", ""),
        "api_mode": getattr(agent, "api_mode", ""),
        "api_key": getattr(agent, "api_key", ""),
        "client_kwargs": copy.deepcopy(getattr(agent, "_client_kwargs", {}) or {}),
        "use_prompt_caching": getattr(agent, "_use_prompt_caching", False),
        "use_native_cache_layout": getattr(agent, "_use_native_cache_layout", False),
        "compressor_model": getattr(cc, "model", getattr(agent, "model", "")) if cc else getattr(agent, "model", ""),
        "compressor_context_length": getattr(cc, "context_length", 0) if cc else 0,
        "compressor_base_url": getattr(cc, "base_url", getattr(agent, "base_url", "")) if cc else getattr(agent, "base_url", ""),
        "compressor_api_key": getattr(cc, "api_key", "") if cc else "",
        "compressor_provider": getattr(cc, "provider", getattr(agent, "provider", "")) if cc else getattr(agent, "provider", ""),
        "compressor_api_mode": getattr(cc, "api_mode", getattr(agent, "api_mode", "")) if cc else getattr(agent, "api_mode", ""),
        "anthropic_api_key": getattr(agent, "_anthropic_api_key", ""),
        "anthropic_base_url": getattr(agent, "_anthropic_base_url", ""),
        "is_anthropic_oauth": getattr(agent, "_is_anthropic_oauth", False),
    }


def _restore_runtime_snapshot(agent: Any, snapshot: dict[str, Any], *, reason: str) -> None:
    agent.model = snapshot["model"]
    agent.provider = snapshot["provider"]
    agent.base_url = snapshot["base_url"]
    agent.api_mode = snapshot["api_mode"]
    agent.api_key = snapshot["api_key"]
    agent._client_kwargs = copy.deepcopy(snapshot.get("client_kwargs") or {})
    agent._use_prompt_caching = snapshot.get("use_prompt_caching", False)
    agent._use_native_cache_layout = snapshot.get("use_native_cache_layout", False)
    if hasattr(agent, "_transport_cache"):
        agent._transport_cache.clear()

    if agent.api_mode == "anthropic_messages":
        from agent.anthropic_adapter import build_anthropic_client
        agent._anthropic_api_key = snapshot.get("anthropic_api_key", "")
        agent._anthropic_base_url = snapshot.get("anthropic_base_url", "")
        agent._anthropic_client = build_anthropic_client(agent._anthropic_api_key, agent._anthropic_base_url)
        agent._is_anthropic_oauth = snapshot.get("is_anthropic_oauth", False)
        agent.client = None
    else:
        agent.client = agent._create_openai_client(dict(agent._client_kwargs), reason=reason, shared=True)

    cc = getattr(agent, "context_compressor", None)
    if cc:
        cc.update_model(
            model=snapshot.get("compressor_model", agent.model),
            context_length=snapshot.get("compressor_context_length", 0),
            base_url=snapshot.get("compressor_base_url", agent.base_url),
            api_key=snapshot.get("compressor_api_key", ""),
            provider=snapshot.get("compressor_provider", agent.provider),
            api_mode=snapshot.get("compressor_api_mode", agent.api_mode),
        )


def _update_context_compressor(agent: Any, *, context_length: int | None = None) -> None:
    cc = getattr(agent, "context_compressor", None)
    if not cc:
        return
    if context_length is None:
        try:
            from agent.model_metadata import get_model_context_length
            context_length = get_model_context_length(
                agent.model,
                base_url=agent.base_url,
                api_key=agent.api_key if isinstance(agent.api_key, str) else "",
                provider=agent.provider,
                config_context_length=getattr(agent, "_config_context_length", None),
                custom_providers=getattr(agent, "_custom_providers", None),
            )
        except Exception:
            context_length = getattr(cc, "context_length", 0)
    cc.update_model(
        model=agent.model,
        context_length=context_length or getattr(cc, "context_length", 0),
        base_url=agent.base_url,
        api_key=agent.api_key,
        provider=agent.provider,
        api_mode=agent.api_mode,
    )


def _is_local_base_url(base_url: str) -> bool:
    if not base_url:
        return False
    url_str = str(base_url or "").strip()
    if url_str and "://" not in url_str:
        url_str = "http://" + url_str
    host = (urlparse(url_str).hostname or "").lower()
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        import ipaddress
        ip = ipaddress.ip_address(host)
        return bool(ip.is_private or ip.is_loopback or ip.is_link_local)
    except ValueError:
        return False


def _expand_env(value: Any) -> str:
    return os.path.expandvars(str(value or "")).strip()


def _load_config_quietly() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}

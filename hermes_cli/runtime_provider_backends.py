"""Provider-specific runtime builders for :mod:`hermes_cli.runtime_provider`: Azure Foundry, the
OpenRouter / bare-custom fallback resolver, Bedrock, and external-process providers. Origin-internal
collaborators are resolved on the origin module at call time via :func:`_rp` so test patches on
``hermes_cli.runtime_provider.*`` (``_get_model_config``, ``load_config``, ``has_usable_secret``,
``_try_resolve_from_custom_pool``, …) still apply."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

from hermes_constants import OPENROUTER_BASE_URL
from utils import base_url_host_matches


def _rp():
    import hermes_cli.runtime_provider as origin
    return origin


# ── Azure Foundry ──────────────────────────────────────────────────────────────────────────


def _azure_entra_credentials(cfg_entra: Dict[str, Any]) -> Any:
    """Callable api_key minting a fresh Entra JWT per request (OpenAI SDK accepts it natively;
    ``build_anthropic_client`` injects the bearer via an httpx hook)."""
    AuthError = _rp().AuthError
    try:
        from agent.azure_identity_adapter import SCOPE_AI_AZURE_DEFAULT, EntraIdentityConfig, build_token_provider
    except Exception as exc:
        raise AuthError(
            "Azure Foundry Entra ID auth requires the 'azure-identity' "
            "package. Install it with: pip install azure-identity "
            f"(import failed: {exc})"
        ) from exc
    scope = str(cfg_entra.get("scope") or "").strip() or SCOPE_AI_AZURE_DEFAULT
    try:
        return build_token_provider(config=EntraIdentityConfig(scope=scope))
    except ImportError as exc:
        raise AuthError(str(exc)) from exc


def _azure_foundry_api_key(rp, explicit_api_key: str) -> str:
    if explicit_api_key:
        return explicit_api_key
    try:
        from hermes_cli.config import get_env_value
        api_key = get_env_value("AZURE_FOUNDRY_API_KEY") or ""
    except Exception:
        api_key = ""
    api_key = api_key or rp._getenv("AZURE_FOUNDRY_API_KEY", "").strip()
    if not api_key:
        raise rp.AuthError(
            "Azure Foundry requires an API key. Set AZURE_FOUNDRY_API_KEY in "
            "~/.hermes/.env or run 'hermes model' to configure. To use "
            "keyless Microsoft Entra ID auth instead, set "
            "model.auth_mode: entra_id in config.yaml (or pick "
            "'Microsoft Entra ID' in 'hermes model')."
        )
    return api_key


def _resolve_azure_foundry_runtime(*, requested_provider: str, model_cfg: Dict[str, Any],
                                   explicit_api_key: Optional[str] = None, explicit_base_url: Optional[str] = None,
                                   target_model: Optional[str] = None) -> Dict[str, Any]:
    """Azure Foundry: ``model.base_url`` + ``model.api_mode`` (or explicit overrides), API key from
    ``.env``/env or a per-request Entra ID token, trailing ``/v1`` stripped for Anthropic-style
    endpoints (the Anthropic SDK appends /v1/messages itself)."""
    rp = _rp()
    explicit_api_key = str(explicit_api_key or "").strip()
    explicit_base_url_clean = str(explicit_base_url or "").strip().rstrip("/")
    cfg_base_url, cfg_api_mode, cfg_auth_mode, cfg_entra = "", "chat_completions", "api_key", {}
    if rp._cfg_provider(model_cfg) == "azure-foundry":
        cfg_base_url = rp._config_base_url_for_provider(model_cfg, "azure-foundry")
        cfg_api_mode = rp._parse_api_mode(model_cfg.get("api_mode")) or "chat_completions"
        cfg_auth_mode = str(model_cfg.get("auth_mode") or "api_key").strip().lower() or "api_key"
        if isinstance(model_cfg.get("entra"), dict):
            cfg_entra = model_cfg["entra"]
    # GPT-5.x / codex / o1-o4 deployments are Responses-API-only on Foundry.
    effective_model = str(target_model or model_cfg.get("default") or "").strip()
    cfg_api_mode = rp._azure_inferred_api_mode(effective_model, cfg_api_mode)
    env_base_url = rp._getenv("AZURE_FOUNDRY_BASE_URL", "").strip().rstrip("/")
    base_url = explicit_base_url_clean or cfg_base_url or env_base_url
    if not base_url:
        raise rp.AuthError(
            "Azure Foundry requires a base URL. Set it via 'hermes model' or "
            "the AZURE_FOUNDRY_BASE_URL environment variable."
        )
    if cfg_api_mode == "anthropic_messages":
        base_url = re.sub(r"/v1/?$", "", base_url)
    if cfg_auth_mode == "entra_id":
        # --api-key on the CLI while config says entra_id: honour the explicit string (escape hatch
        # for one-off testing).
        if explicit_api_key:
            api_key, source, auth_mode, entra = explicit_api_key, "explicit", "api_key", {}
        else:
            scope = str(cfg_entra.get("scope") or "").strip()
            api_key, source, auth_mode, entra = _azure_entra_credentials(cfg_entra), "entra_id", "entra_id", (
                {"scope": scope} if scope else {}
            )
        return rp._runtime("azure-foundry", cfg_api_mode, base_url, api_key, auth_mode=auth_mode, entra=entra, source=source,
                           requested_provider=requested_provider)
    return rp._runtime("azure-foundry", cfg_api_mode, base_url, _azure_foundry_api_key(rp, explicit_api_key),
                       auth_mode="api_key", source="explicit" if (explicit_api_key or explicit_base_url) else "config",
                       requested_provider=requested_provider)


# ── OpenRouter / bare custom fallback ──────────────────────────────────────────────────────


def _resolve_openrouter_runtime(
    *, requested_provider: str, explicit_api_key: Optional[str] = None, explicit_base_url: Optional[str] = None
) -> Dict[str, Any]:
    """Terminal resolver: OpenRouter, or a bare/aliased ``custom`` endpoint. base_url precedence:
    explicit > CUSTOM_BASE_URL > trusted ``model.base_url`` > OPENROUTER_BASE_URL > default.
    OPENAI_BASE_URL is deliberately NOT consulted — config.yaml is the single source of truth for
    endpoint URLs. OpenRouter contexts prefer OPENROUTER_API_KEY; custom endpoints never receive the
    OpenRouter key and only get env keys gated on their authoritative hosts."""
    rp = _rp()
    model_cfg = rp._get_model_config()
    cfg_base_url = model_cfg.get("base_url") if isinstance(model_cfg.get("base_url"), str) else ""
    cfg_provider = (model_cfg.get("provider") if isinstance(model_cfg.get("provider"), str) else "").strip().lower()
    cfg_api_key = next((v.strip() for v in (model_cfg.get("api_key"), model_cfg.get("api")) if isinstance(v, str) and v.strip()), "")
    requested_norm = (requested_provider or "").strip().lower()
    # Aliases resolving to "custom" (ollama, vllm, …) follow bare-custom trust + routing rules.
    if requested_norm and requested_norm != "custom" and rp._resolves_to_custom(requested_norm):
        requested_norm = "custom"
    env_openrouter_base_url = rp._getenv("OPENROUTER_BASE_URL", "").strip()
    env_custom_base_url = rp._getenv("CUSTOM_BASE_URL", "").strip()
    use_config_base_url = bool(cfg_base_url.strip()) and not explicit_base_url and (
        (requested_norm == "auto" and cfg_provider in ("", "auto"))
        or (requested_norm == "custom" and rp._config_base_url_trustworthy_for_bare_custom(cfg_base_url, cfg_provider))
    )
    base_url = ((explicit_base_url or "").strip() or env_custom_base_url or (cfg_base_url.strip() if use_config_base_url else "")
                or env_openrouter_base_url or OPENROUTER_BASE_URL).rstrip("/")
    # Choose API key based on whether the resolved base_url targets OpenRouter. When hitting OpenRouter,
    # prefer OPENROUTER_API_KEY (issue #289). When hitting a custom endpoint (e.g. Z.ai, local LLM), prefer
    # OPENAI_API_KEY so the OpenRouter key doesn't leak to an unrelated provider (issues #420, #560).
    is_openrouter_url = base_url_host_matches(base_url, "openrouter.ai")
    # Explicitly-configured OpenRouter mirrors (OPENROUTER_BASE_URL + provider=openrouter) still
    # count as OpenRouter for key selection.
    is_openrouter_context = is_openrouter_url or (
        requested_norm == "openrouter" and (env_openrouter_base_url or base_url == env_openrouter_base_url)
        and base_url == (env_openrouter_base_url or "").rstrip("/")
    )
    if is_openrouter_context:
        candidates = [explicit_api_key, rp._getenv("OPENROUTER_API_KEY"), rp._getenv("OPENAI_API_KEY")]
    else:
        candidates = [explicit_api_key, (cfg_api_key if use_config_base_url else ""),
                      *rp._host_gated_env_key_candidates(base_url, ollama=True)]
    api_key = next((str(c or "").strip() for c in candidates if rp.has_usable_secret(c)), "")
    source = "explicit" if (explicit_api_key or explicit_base_url) else "env/config"
    cfg_api_mode = rp._parse_api_mode(model_cfg.get("api_mode"))
    # Explicit "custom" stays "custom" rather than relabeling to "openrouter".
    if requested_norm != "custom":
        return rp._runtime("openrouter", cfg_api_mode or rp._detect_api_mode_for_url(base_url) or "chat_completions", base_url,
                           api_key, source=source)
    if base_url:
        pool_result = rp._try_resolve_from_custom_pool(base_url, "custom", cfg_api_mode, provider_name=None)
        if pool_result:
            return pool_result
    # Local no-auth servers get a placeholder key — the OpenAI SDK requires a non-empty string.
    if not api_key and not is_openrouter_url:
        api_key = "no-key-required"
    return rp._runtime("custom", rp._resolve_plain_custom_api_mode(model_cfg, base_url), base_url, api_key, source=source)


# ── AWS Bedrock ────────────────────────────────────────────────────────────────────────────


def _bedrock_guardrail_config(bedrock_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    gr = bedrock_cfg.get("guardrail", {})
    if not (gr.get("guardrail_identifier") and gr.get("guardrail_version")):
        return None
    config = {"guardrailIdentifier": gr["guardrail_identifier"], "guardrailVersion": gr["guardrail_version"]}
    for src_key, dst_key in (("stream_processing_mode", "streamProcessingMode"), ("trace", "trace")):
        if gr.get(src_key):
            config[dst_key] = gr[src_key]
    return config


def _resolve_bedrock_runtime(requested_provider: str, model_cfg: Dict[str, Any], target_model: Optional[str]) -> Dict[str, Any]:
    """AWS Bedrock with triple-path routing: OpenAI models → Bedrock Mantle's Responses endpoint;
    Claude → AnthropicBedrock SDK (prompt caching, thinking budgets); others → Converse API.
    AWS_BEARER_TOKEN_BEDROCK auth is unsupported by AnthropicBedrock (SigV4 only), so bearer users
    go through Converse regardless of model."""
    from agent.bedrock_adapter import (bedrock_openai_base_url, has_aws_credentials, is_anthropic_bedrock_model,
                                       is_openai_bedrock_model, resolve_aws_auth_env_var, resolve_bedrock_bearer_token,
                                       resolve_bedrock_runtime_region)
    from hermes_cli.config import load_config  # direct (not the origin delegate), as before
    rp = _rp()
    # Explicitly selected bedrock trusts boto3's credential chain (IMDS, ECS/Lambda roles, SSO)
    # which the env-var check can't detect.
    is_explicit = requested_provider in {"bedrock", "aws", "aws-bedrock", "amazon-bedrock", "amazon"}
    if not is_explicit and not has_aws_credentials():
        raise rp.AuthError(
            "No AWS credentials found for Bedrock. Configure one of:\n"
            "  - AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY\n"
            "  - AWS_PROFILE (for SSO / named profiles)\n"
            "  - IAM instance role (EC2, ECS, Lambda)\n"
            "Or run 'aws configure' to set up credentials.",
            code="no_aws_credentials",
        )
    bedrock_cfg = load_config().get("bedrock", {})
    # Region priority (config.yaml bedrock.region → env → us-east-1) lives in the adapter.
    region = resolve_bedrock_runtime_region({"bedrock": bedrock_cfg})
    auth_source = resolve_aws_auth_env_var() or "aws-sdk-default-chain"
    guardrail_config = _bedrock_guardrail_config(bedrock_cfg)
    current_model = str(target_model or model_cfg.get("default") or "").strip()
    has_bearer_token = bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip())
    runtime = rp._runtime("bedrock", "bedrock_converse", f"https://bedrock-runtime.{region}.amazonaws.com", "aws-sdk",
                          source=auth_source, region=region, requested_provider=requested_provider)
    if is_openai_bedrock_model(current_model):
        bearer = resolve_bedrock_bearer_token()
        runtime.update(api_mode="codex_responses", base_url=bedrock_openai_base_url(region), api_key=bearer or "aws-sdk",
                       source="AWS_BEARER_TOKEN_BEDROCK" if bearer else auth_source, model=current_model, bedrock_openai=True)
    elif is_anthropic_bedrock_model(current_model) and not has_bearer_token:
        runtime.update(api_mode="anthropic_messages", bedrock_anthropic=True)
    if guardrail_config:
        runtime["guardrail_config"] = guardrail_config
    return runtime


# ── External-process (agent CLI over stdio, e.g. ACP) ──────────────────────────────────────


def _is_external_process_provider(provider: str) -> bool:
    """Keyed on the registered provider's auth_type (CLI registry first, then the profile registry
    so the check works before the CLI registry has been extended)."""
    name = (provider or "").strip().lower()
    if not name:
        return False
    try:
        pconfig = _rp().PROVIDER_REGISTRY.get(name)
        if pconfig is not None:
            return pconfig.auth_type == "external_process"
    except Exception:
        pass
    try:
        from providers import get_provider_profile

        profile = get_provider_profile(name)
    except Exception:
        return False
    return profile is not None and getattr(profile, "auth_type", "") == "external_process"


def _resolve_external_process_runtime(provider: str, requested_provider: str) -> Dict[str, Any]:
    rp = _rp()
    creds = rp.resolve_external_process_provider_credentials(provider)
    return rp._runtime(provider, "chat_completions", creds.get("base_url", "").rstrip("/"), creds.get("api_key", ""),
                       command=creds.get("command", ""), args=list(creds.get("args") or []),
                       source=creds.get("source", "process"), requested_provider=requested_provider)

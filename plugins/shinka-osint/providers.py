"""Bridge Hermes LLM auth into Shinka subprocesses (no google-generativeai required).

MILSPEC rule-based scoring does not need an LLM. This module supplies credentials
for optional synthesis / evolution paths and enforces a China-provider policy:
direct China-hosted APIs are blocked; Chinese models on Western hosts are allowed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

# Preferred order when no explicit override is set.
DEFAULT_PROVIDER_PRIORITY: tuple[str, ...] = (
    "openai-codex",
    "nvidia",
    "nous",
    "xai-oauth",
)

# Direct China-region provider IDs — never route Shinka LLM traffic here.
BLOCKED_PROVIDER_IDS: frozenset[str] = frozenset(
    {
        "zai",
        "alibaba",
        "qwen-oauth",
        "deepseek",
        "moonshot-cn",
        "kimi-coding-cn",
        "baidu",
        "tencent-tokenhub",
        "xiaomi",
        "minimax-cn",
        "doubao",
        "stepfun",
    }
)

# Host suffixes treated as Western infrastructure (Chinese models OK on these).
WESTERN_HOST_SUFFIXES: tuple[str, ...] = (
    "openrouter.ai",
    "integrate.api.nvidia.com",
    "inference-api.nousresearch.com",
    "api.fireworks.ai",
    "api.together.xyz",
    "api.x.ai",
    "chatgpt.com",
    "api.openai.com",
    "api.anthropic.com",
    "models.github.ai",
    "models.inference.ai.azure.com",
)

CHINA_HOST_MARKERS: tuple[str, ...] = (
    "open.bigmodel.cn",
    "dashscope.aliyuncs.com",
    "dashscope-intl.aliyuncs.com",
    "portal.qwen.ai",
    "api.deepseek.com",
    "api.moonshot.cn",
    "api.minimax.chat",
    "api.xiaomimimo.com",
    "tokenhub.tencentmaas.com",
)

DEFAULT_NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
DEFAULT_NOUS_BASE = "https://inference-api.nousresearch.com/v1"
DEFAULT_XAI_BASE = "https://api.x.ai/v1"
DEFAULT_CODEX_BASE = "https://chatgpt.com/backend-api/codex"


@dataclass(frozen=True)
class ResolvedLLM:
    provider_id: str
    model: str
    api_key: str
    base_url: str
    source: str
    western_host: bool


def _normalize_host(base_url: str) -> str:
    host = (urlparse(base_url or "").hostname or "").lower()
    return host


def host_is_western(base_url: str) -> bool:
    host = _normalize_host(base_url)
    if not host:
        return False
    if any(marker in host for marker in CHINA_HOST_MARKERS):
        return False
    return any(host == suffix or host.endswith("." + suffix) for suffix in WESTERN_HOST_SUFFIXES)


def host_is_china_region(base_url: str) -> bool:
    host = _normalize_host(base_url)
    return bool(host) and any(marker in host for marker in CHINA_HOST_MARKERS)


def assert_provider_allowed(provider_id: str, base_url: str = "") -> None:
    pid = (provider_id or "").strip().lower()
    if pid in BLOCKED_PROVIDER_IDS:
        raise ValueError(
            f"Shinka OSINT blocks China-region LLM provider {pid!r}. "
            "Use openai-codex, nvidia, nous, or xai-oauth, or a Western-hosted router."
        )
    if base_url and host_is_china_region(base_url):
        raise ValueError(
            f"Shinka OSINT blocks China-region API host {base_url!r}. "
            "Western hosting is required for Chinese model slugs."
        )


def _read_config_override() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        plugins = cfg.get("plugins") if isinstance(cfg.get("plugins"), dict) else {}
        section = plugins.get("shinka-osint") if isinstance(plugins.get("shinka-osint"), dict) else {}
        llm = section.get("llm") if isinstance(section.get("llm"), dict) else {}
        return llm
    except Exception:
        return {}


def _read_main_model_config() -> tuple[str, str]:
    """Return Hermes Agent's configured main provider/model.

    Shinka's optional LLM summaries should track Hermes' own model selection
    instead of carrying a stale Codex fallback. Hermes stores the primary model
    as ``model.default`` (with ``model.model`` accepted as a legacy alias), while
    plugin-local overrides live under ``plugins.shinka-osint.llm``.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        if not isinstance(model_cfg, dict):
            return "", ""
        provider = (model_cfg.get("provider") or "").strip().lower()
        model = (model_cfg.get("default") or model_cfg.get("model") or "").strip()
        return provider, model
    except Exception:
        return "", ""


def _provider_priority() -> tuple[str, ...]:
    llm_cfg = _read_config_override()
    raw = llm_cfg.get("provider_priority")
    if isinstance(raw, list) and raw:
        cleaned = [str(item).strip().lower() for item in raw if str(item).strip()]
        if cleaned:
            return tuple(cleaned)
    override = (llm_cfg.get("provider") or "").strip().lower()
    if override:
        return (override, *DEFAULT_PROVIDER_PRIORITY)
    main_provider, _ = _read_main_model_config()
    if main_provider and main_provider in _RESOLVERS:
        return (main_provider, *tuple(p for p in DEFAULT_PROVIDER_PRIORITY if p != main_provider))
    return DEFAULT_PROVIDER_PRIORITY


def _try_openai_codex() -> ResolvedLLM | None:
    try:
        from hermes_cli.auth import resolve_codex_runtime_credentials

        creds = resolve_codex_runtime_credentials()
    except Exception:
        return None
    api_key = (creds.get("api_key") or "").strip()
    if not api_key:
        return None
    base_url = (creds.get("base_url") or DEFAULT_CODEX_BASE).strip().rstrip("/")
    model = (creds.get("model") or "").strip()
    if not model:
        main_provider, main_model = _read_main_model_config()
        if main_provider in {"openai-codex", "codex", "gpt"}:
            model = main_model
    if not model:
        try:
            from hermes_cli.models import get_default_model_for_provider

            model = (get_default_model_for_provider("openai-codex") or "").strip()
        except Exception:
            model = ""
    if not model:
        model = "gpt-5.5"
    return ResolvedLLM(
        provider_id="openai-codex",
        model=model,
        api_key=api_key,
        base_url=base_url,
        source=str(creds.get("source") or "codex_auth"),
        western_host=host_is_western(base_url),
    )


def _try_nvidia() -> ResolvedLLM | None:
    api_key = ""
    try:
        from hermes_cli.config import get_env_value

        api_key = (get_env_value("NVIDIA_API_KEY") or "").strip()
    except Exception:
        api_key = (os.environ.get("NVIDIA_API_KEY") or "").strip()
    if not api_key:
        return None
    llm_cfg = _read_config_override()
    main_provider, main_model = _read_main_model_config()
    model = (llm_cfg.get("nvidia_model") or "").strip()
    if not model and main_provider == "nvidia":
        model = main_model
    if not model:
        model = "meta/llama-3.1-70b-instruct"
    base_url = (llm_cfg.get("nvidia_base_url") or DEFAULT_NVIDIA_BASE).strip().rstrip("/")
    return ResolvedLLM(
        provider_id="nvidia",
        model=model,
        api_key=api_key,
        base_url=base_url,
        source="nvidia_api_key",
        western_host=host_is_western(base_url),
    )


def _try_nous() -> ResolvedLLM | None:
    try:
        from hermes_cli.auth import resolve_nous_runtime_credentials

        creds = resolve_nous_runtime_credentials()
    except Exception:
        return None
    api_key = (creds.get("api_key") or "").strip()
    if not api_key:
        return None
    base_url = (creds.get("base_url") or DEFAULT_NOUS_BASE).strip().rstrip("/")
    model = (creds.get("model") or "").strip()
    if not model:
        main_provider, main_model = _read_main_model_config()
        if main_provider == "nous":
            model = main_model
    if not model:
        try:
            from hermes_cli.models import get_nous_recommended_aux_model

            model = (get_nous_recommended_aux_model() or "").strip()
        except Exception:
            model = ""
    if not model:
        model = "DeepHermes-3-Llama-3-8B-Preview"
    return ResolvedLLM(
        provider_id="nous",
        model=model,
        api_key=api_key,
        base_url=base_url,
        source=str(creds.get("source") or "nous_auth"),
        western_host=host_is_western(base_url),
    )


def _try_xai_oauth() -> ResolvedLLM | None:
    try:
        from hermes_cli.auth import resolve_xai_oauth_runtime_credentials

        creds = resolve_xai_oauth_runtime_credentials()
    except Exception:
        return None
    api_key = (creds.get("api_key") or "").strip()
    if not api_key:
        return None
    base_url = (creds.get("base_url") or DEFAULT_XAI_BASE).strip().rstrip("/")
    model = (creds.get("model") or "").strip()
    if not model:
        main_provider, main_model = _read_main_model_config()
        if main_provider == "xai-oauth":
            model = main_model

    if not model:
        model = "grok-3-mini"
    return ResolvedLLM(
        provider_id="xai-oauth",
        model=model,
        api_key=api_key,
        base_url=base_url,
        source=str(creds.get("source") or "xai_oauth"),
        western_host=host_is_western(base_url),
    )


_RESOLVERS = {
    "openai-codex": _try_openai_codex,
    "codex": _try_openai_codex,
    "gpt": _try_openai_codex,
    "nvidia": _try_nvidia,
    "nous": _try_nous,
    "xai-oauth": _try_xai_oauth,
    "xai": _try_xai_oauth,
}


def resolve_llm(*, require_auth: bool = False) -> ResolvedLLM | None:
    """Pick the first configured Hermes provider in priority order."""
    llm_cfg = _read_config_override()
    explicit_model = (llm_cfg.get("model") or "").strip()

    for provider_id in _provider_priority():
        resolver = _RESOLVERS.get(provider_id)
        if resolver is None:
            continue
        try:
            resolved = resolver()
        except Exception:
            resolved = None
        if resolved is None:
            continue
        try:
            assert_provider_allowed(resolved.provider_id, resolved.base_url)
        except ValueError:
            continue
        if explicit_model:
            return ResolvedLLM(
                provider_id=resolved.provider_id,
                model=explicit_model,
                api_key=resolved.api_key,
                base_url=resolved.base_url,
                source=resolved.source,
                western_host=resolved.western_host,
            )
        return resolved

    if require_auth:
        return None
    return None


def build_env_overlay() -> dict[str, str]:
    """Environment variables for isolated Shinka subprocesses."""
    overlay: dict[str, str] = {
        # MILSPEC evaluate uses rule scoring; do not hard-require Gemini embeddings.
        "SHINKA_DISABLE_GEMINI_EMBEDDING": "1",
        "SHINKA_HERMES_BRIDGE": "1",
    }
    resolved = resolve_llm()
    if resolved is None:
        overlay["SHINKA_LLM_AVAILABLE"] = "0"
        return overlay

    overlay.update(
        {
            "SHINKA_LLM_AVAILABLE": "1",
            "SHINKA_HERMES_LLM_PROVIDER": resolved.provider_id,
            "SHINKA_HERMES_LLM_MODEL": resolved.model,
            "OPENAI_API_KEY": resolved.api_key,
            "OPENAI_BASE_URL": resolved.base_url,
        }
    )
    if resolved.provider_id == "nvidia":
        overlay["NVIDIA_API_KEY"] = resolved.api_key
    return overlay


def provider_status() -> dict[str, Any]:
    resolved = resolve_llm()
    gemini_installed = False
    try:
        import importlib.util

        gemini_installed = importlib.util.find_spec("google.generativeai") is not None
    except Exception:
        gemini_installed = False

    return {
        "milspec_requires_gemini": False,
        "google_generativeai_installed": gemini_installed,
        "gemini_optional": True,
        "policy": {
            "blocked_provider_ids": sorted(BLOCKED_PROVIDER_IDS),
            "western_host_required_for_chinese_models": True,
            "allowed_hermes_providers": list(DEFAULT_PROVIDER_PRIORITY),
        },
        "priority": list(_provider_priority()),
        "resolved": (
            {
                "provider_id": resolved.provider_id,
                "model": resolved.model,
                "base_url": resolved.base_url,
                "source": resolved.source,
                "western_host": resolved.western_host,
            }
            if resolved
            else None
        ),
        "llm_ready": resolved is not None,
    }


def synthesize_executive_summary(
    *,
    topic: str,
    briefing_payload: dict[str, Any],
    max_tokens: int = 1200,
) -> dict[str, Any]:
    """Optional Japanese executive summary via Hermes auxiliary LLM (no Gemini dep)."""
    resolved = resolve_llm(require_auth=True)
    if resolved is None:
        return {
            "success": False,
            "skipped": True,
            "reason": (
                "No allowed Hermes LLM auth found. Configure openai-codex, NVIDIA_API_KEY, "
                "nous, or xai-oauth. MILSPEC scores still computed without LLM."
            ),
        }

    try:
        from agent.auxiliary_client import resolve_provider_client
    except ImportError as exc:
        return {"success": False, "skipped": True, "reason": f"auxiliary_client unavailable: {exc}"}

    client, model = resolve_provider_client(
        resolved.provider_id,
        model=resolved.model,
        explicit_api_key=resolved.api_key,
        explicit_base_url=resolved.base_url,
        task="shinka_osint_briefing",
    )
    if client is None or not model:
        return {
            "success": False,
            "skipped": True,
            "reason": f"Could not build client for provider {resolved.provider_id}",
        }

    runs = briefing_payload.get("runs") if isinstance(briefing_payload.get("runs"), list) else []
    lines: list[str] = []
    for run in runs[:8]:
        if not isinstance(run, dict):
            continue
        sid = run.get("scenario_id")
        score = run.get("total_score")
        domain = run.get("domain")
        result = run.get("result") if isinstance(run.get("result"), dict) else {}
        lines.append(
            f"- {sid} ({domain}): score={score}, evidence={result.get('evidence_blocks')}, "
            f"kj={result.get('key_judgments')}, verified={result.get('verified')}"
        )

    system_msg = (
        "You are a Japanese OSINT briefer writing MILSPEC-aligned summaries. "
        "Mandatory rules: (1) Every factual claim MUST cite "
        "[出典: URL], [出典: 法令・公文書 ID], or [出典: Shinka scenario_id + evidence_block]. "
        "(2) Do not invent facts or numbers. Mark unsourced claims UNVERIFIED. "
        "(3) Prefer primary government/treaty sources over media. "
        "(4) Separate OBSERVED (sourced) from ASSESSMENT (labeled analytic judgment). "
        "Cite scenario IDs. Flag allowlist or fabrication issues. No speculation."
    )
    user_msg = (
        f"Topic: {topic or '安全保障'}\n"
        f"Average score: {briefing_payload.get('average_score')}\n"
        f"Source mode: {briefing_payload.get('source_mode')}\n"
        "Scenario runs:\n"
        + "\n".join(lines)
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=max(256, min(max_tokens, 4096)),
            temperature=0.2,
        )
        content = ""
        if response.choices:
            content = (response.choices[0].message.content or "").strip()
        return {
            "success": bool(content),
            "skipped": False,
            "provider_id": resolved.provider_id,
            "model": model,
            "summary_ja": content,
        }
    except Exception as exc:
        return {
            "success": False,
            "skipped": True,
            "provider_id": resolved.provider_id,
            "model": model,
            "reason": str(exc),
        }

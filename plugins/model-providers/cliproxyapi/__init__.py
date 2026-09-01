"""CLIProxyAPI provider profile.

CLIProxyAPI is an OpenAI-compatible local proxy that fronts several CLI/OAuth
providers. Its GPT/Codex-style, Claude/Neko, Grok, and OpenCode Go routes accept
top-level ``reasoning_effort``; CPA translates that into each upstream
provider's native thinking shape.
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


_GPT_REASONING_MODEL_PREFIXES = (
    "gpt-",
    "chatgpt-",
    "codex-",
    "o1",
    "o3",
    "o4",
    "openai/",
)
_CLAUDE_REASONING_MODEL_FAMILY_TOKENS = {
    "claude",
    "sonnet",
    "opus",
    "fable",
    "haiku",
    "mythos",
}
_GROK_EFFORT_CAPABLE_PREFIXES = (
    "grok-3-mini",
    "grok-4.20-multi-agent",
    "grok-4.3",
    "grok-4.5",
)
_GROK_STANDARD_EFFORTS = frozenset({"low", "medium", "high"})
_GROK_MULTI_AGENT_EFFORTS = frozenset({"low", "medium", "high", "xhigh"})
_OPENCODE_GO_KIMI_K2_EFFORTS = frozenset({"low", "medium", "high"})
_OPENCODE_GO_QWEN_EFFORTS = frozenset({"none", "low", "medium", "high", "xhigh", "max"})


def _is_gpt_reasoning_model(model: str | None) -> bool:
    model_l = (model or "").strip().lower()
    return bool(model_l) and model_l.startswith(_GPT_REASONING_MODEL_PREFIXES)


def _model_segments(model: str | None) -> set[str]:
    normalized = (model or "").strip().lower()
    if not normalized:
        return set()
    for sep in "/:._-()[]":
        normalized = normalized.replace(sep, " ")
    return {part for part in normalized.split() if part}


def _is_claude_reasoning_model(model: str | None) -> bool:
    return bool(_model_segments(model) & _CLAUDE_REASONING_MODEL_FAMILY_TOKENS)


def _strip_model_slug(model: str | None) -> str:
    name = (model or "").strip().lower()
    if not name:
        return ""
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    if name.endswith(")") and "(" in name:
        name = name.split("(", 1)[0]
    for suffix in ("-high", "-max", "-medium", "-low"):
        if name.endswith(suffix) and name.startswith("glm-5.2"):
            name = name[: -len(suffix)]
            break
    return name


def _glm_fixed_effort_alias(model: str | None) -> str | None:
    name = (model or "").strip().lower()
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    if name in {"glm-5.2(max)", "glm-5.2-max"}:
        return "max"
    if name in {"glm-5.2(high)", "glm-5.2-high"}:
        return "high"
    return None


def _is_grok_model(model: str | None) -> bool:
    return _strip_model_slug(model).startswith("grok-")


def _is_grok_reasoning_model(model: str | None) -> bool:
    name = _strip_model_slug(model)
    return bool(name) and any(name.startswith(prefix) for prefix in _GROK_EFFORT_CAPABLE_PREFIXES)


def _is_grok_multi_agent(model: str | None) -> bool:
    return _strip_model_slug(model).startswith("grok-4.20-multi-agent")


def _is_glm_5_2_model(model: str | None) -> bool:
    name = _strip_model_slug(model)
    return any(token in name for token in ("glm-5.2", "glm-5-2", "glm-5p2"))


def _is_kimi_k3_model(model: str | None) -> bool:
    return _strip_model_slug(model).startswith("kimi-k3")


def _is_kimi_k2_model(model: str | None) -> bool:
    return _strip_model_slug(model).startswith("kimi-k2")


def _is_deepseek_v4_model(model: str | None) -> bool:
    name = _strip_model_slug(model)
    return name.startswith("deepseek-v4")


def _is_qwen37_max_model(model: str | None) -> bool:
    name = _strip_model_slug(model)
    return name in {"qwen3.7-max", "qwen3-7-max", "qwen3.7max"} or name.startswith("qwen3.7-max")


def _is_opencode_go_reasoning_model(model: str | None) -> bool:
    return any(
        predicate(model)
        for predicate in (
            _is_glm_5_2_model,
            _is_kimi_k3_model,
            _is_kimi_k2_model,
            _is_deepseek_v4_model,
            _is_qwen37_max_model,
        )
    )


def _supports_top_level_reasoning_effort(model: str | None) -> bool:
    return (
        _is_gpt_reasoning_model(model)
        or _is_claude_reasoning_model(model)
        or _is_grok_reasoning_model(model)
        or _is_opencode_go_reasoning_model(model)
    )


def _clamp_grok_effort(model: str | None, effort: str) -> str | None:
    effort = {"minimal": "low"}.get((effort or "").strip().lower(), (effort or "").strip().lower())
    if _is_grok_multi_agent(model):
        effort = {"max": "xhigh"}.get(effort, effort)
        return effort if effort in _GROK_MULTI_AGENT_EFFORTS else None
    effort = {"xhigh": "high", "max": "high"}.get(effort, effort)
    return effort if effort in _GROK_STANDARD_EFFORTS else None


def _clamp_opencode_go_effort(model: str | None, effort: str) -> str | None:
    effort = {"minimal": "low"}.get((effort or "").strip().lower(), (effort or "").strip().lower())
    if not effort or effort == "none":
        return "none" if _is_qwen37_max_model(model) else None
    if _is_kimi_k3_model(model):
        return "max"
    if _is_glm_5_2_model(model):
        fixed = _glm_fixed_effort_alias(model)
        return fixed or ("max" if effort in {"xhigh", "max"} else "high")
    if _is_kimi_k2_model(model):
        effort = {"xhigh": "high", "max": "high"}.get(effort, effort)
        return effort if effort in _OPENCODE_GO_KIMI_K2_EFFORTS else None
    if _is_deepseek_v4_model(model):
        effort = {"xhigh": "high", "max": "high"}.get(effort, effort)
        return effort if effort in {"low", "medium", "high"} else None
    if _is_qwen37_max_model(model):
        return effort if effort in _OPENCODE_GO_QWEN_EFFORTS else None
    return None


class CLIProxyAPIProfile(ProviderProfile):
    """Local CLIProxyAPI OpenAI-compatible proxy."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        session_id: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        sid = str(session_id or "").strip()
        if sid and _is_grok_model(model):
            top_level["extra_headers"] = {
                "X-Session-ID": sid,
                "x-grok-conv-id": sid,
            }
            extra_body["prompt_cache_key"] = sid

        if not _supports_top_level_reasoning_effort(model) or not isinstance(reasoning_config, dict):
            return extra_body, top_level

        if reasoning_config.get("enabled") is False:
            if _is_claude_reasoning_model(model) or _is_qwen37_max_model(model):
                top_level["reasoning_effort"] = "none"
            return extra_body, top_level

        effort = str(reasoning_config.get("effort") or "").strip().lower()
        effort = {"minimal": "low"}.get(effort, effort)

        if _is_grok_reasoning_model(model):
            mapped = _clamp_grok_effort(model, effort)
            if mapped:
                top_level["reasoning_effort"] = mapped
            return extra_body, top_level

        if _is_opencode_go_reasoning_model(model):
            mapped = _clamp_opencode_go_effort(model, effort)
            if mapped:
                top_level["reasoning_effort"] = mapped
            return extra_body, top_level

        if _is_gpt_reasoning_model(model) and effort == "max":
            effort = "xhigh"
        if effort in {"low", "medium", "high", "xhigh", "max"}:
            top_level["reasoning_effort"] = effort
        return extra_body, top_level


cliproxyapi = CLIProxyAPIProfile(
    name="cliproxyapi",
    aliases=("cli-proxy-api", "cli_proxy_api", "cliproxy", "cpa"),
    display_name="CLIProxyAPI",
    description="Local OpenAI-compatible proxy for CLI subscription providers",
    env_vars=(),
    base_url="",
)

register_provider(cliproxyapi)

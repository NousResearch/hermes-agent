"""W&B (Weights & Biases) inference provider profile.

W&B fronts ~29 open-weight models on a shared vLLM backend at
api.inference.wandb.ai. Reasoning control is NOT uniform across the catalog —
vLLM's chat-template-kwargs mechanism is a per-model-family contract, and W&B
inherits whichever dialect each upstream model's chat template defines. This
profile dispatches on model family, mirroring the OpenCodeGoProfile pattern
(plugins/model-providers/opencode-zen) for the same "one relay, several wire
formats" shape.

Confirmed dialects (live-tested against api.inference.wandb.ai + cross-checked
against docs.wandb.ai/inference/response-settings/reasoning and vLLM's
reasoning_outputs docs):

  GLM (zai-org/GLM-*)
    Toggle:  chat_template_kwargs.enable_thinking (false to disable)
    Effort:  top-level reasoning_effort — ONLY "high"/"max" recognised.
             minimal/low/medium are silently ignored and fall back to
             Think Max, so they're mapped up to "high" instead of omitted.

  Granular-effort toggle family (Qwen3.5-*, Qwen3.6-*, Gemma-4, Nemotron-3)
    Toggle:  chat_template_kwargs.enable_thinking (false to disable)
    Effort:  top-level reasoning_effort passed through verbatim
             (low/medium/high honoured — confirmed distinct reasoning-token
             counts per level on Qwen3.6-35B-A3B); xhigh/max -> "max".

  DeepSeek reasoning family (deepseek-ai/DeepSeek-V3.1, DeepSeek-V4-*)
    Toggle:  chat_template_kwargs.thinking (note: "thinking", NOT
             "enable_thinking" — a different key entirely; enable_thinking
             produced weak/inconsistent output in testing). Matches vLLM's
             dedicated deepseek_v3 reasoning parser contract.
    Effort:  no granular effort observed — omitted entirely for this family.

  Always-on reasoning family (openai/gpt-oss-20b, openai/gpt-oss-120b,
  MiniMaxAI/MiniMax-M2.5, moonshotai/Kimi-K2.5, moonshotai/Kimi-K2.6,
  moonshotai/Kimi-K2.7-Code, Qwen/Qwen3-235B-A22B-Thinking-2507)
    Toggle:  none — reasoning cannot be disabled (W&B docs: "Always on").
    Effort:  top-level reasoning_effort still modulates depth (confirmed on
             gpt-oss-20b: reasoning_len 127 default vs 20 at effort=low), so
             it's passed through verbatim when the user requests a level.

  Everything else (Llama, Phi, plain Qwen3-Instruct non-thinking variants,
  IBM Granite, JetBrains Mellum2, etc.) — no reasoning support; no-op.

Unset/unrecognised model defaults to the GLM dialect, since GLM-5.2 is this
profile's fallback_models entry and Hermes' bundled config.yaml default.

Also handles the Cloudflare-blocks-default-urllib-UA quirk via default_headers
(applies to every model on the endpoint, not just GLM).
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


def _model_lower(model: str | None) -> str:
    return (model or "").strip().lower()


def _is_glm_model(model: str | None) -> bool:
    m = _model_lower(model)
    return m.startswith("zai-org/glm") or m.startswith("glm-")


def _is_deepseek_thinking_model(model: str | None) -> bool:
    """DeepSeek reasoning family — V3.1 and the V4 generation.

    Every DeepSeek model currently on W&B (V3.1, V4-Flash, V4-Pro) is
    reasoning-capable, unlike the direct DeepSeek API where deepseek-chat
    (V3) has no thinking mode — so a broad prefix match is safe here.
    """
    m = _model_lower(model)
    return m.startswith("deepseek-ai/deepseek-v")


def _is_granular_toggle_model(model: str | None) -> bool:
    """Qwen3.5/3.6, Gemma-4, Nemotron-3 — enable_thinking + granular effort."""
    m = _model_lower(model)
    return (
        m.startswith("qwen/qwen3.5")
        or m.startswith("qwen/qwen3.6")
        or m.startswith("google/gemma-4")
        or m.startswith("nvidia/nvidia-nemotron-3")
    )


def _is_always_on_reasoning_model(model: str | None) -> bool:
    """Models W&B docs list as 'Always on' — no disable switch, but effort
    still modulates depth (confirmed on gpt-oss-20b)."""
    m = _model_lower(model)
    return (
        m.startswith("openai/gpt-oss-")
        or m.startswith("minimaxai/minimax-m2")
        or m.startswith("moonshotai/kimi-k2")
        or m == "qwen/qwen3-235b-a22b-thinking-2507"
    )


def _effort_to_glm(effort: str) -> str | None:
    """Map Hermes reasoning effort levels to GLM-5.2's accepted values.

    GLM-5.2 only supports "high" and "max". Unrecognised values silently
    fall back to Think Max (the model default), so we explicitly map:
      xhigh → "max"  (GLM-5.2's deepest thinking)
      high  → "high" (lighter, ~3× faster)
      low/minimal/medium → "high" (lightest supported thinking)
      unset/unknown → None (omit param — let GLM-5.2 use its default)
    """
    e = (effort or "").strip().lower()
    if e in {"xhigh", "max"}:
        return "max"
    if e in {"minimal", "low", "medium", "high"}:
        return "high"
    return None


def _effort_passthrough(effort: str) -> str | None:
    """Pass Hermes effort through verbatim for models with real granularity.

    low/medium/high map 1:1 (all three produce distinct reasoning-token
    counts, confirmed on Qwen3.6-35B-A3B and gpt-oss-20b); xhigh/max
    collapse to vLLM's "max".
    """
    e = (effort or "").strip().lower()
    if e in {"xhigh", "max"}:
        return "max"
    if e in {"low", "medium", "high"}:
        return e
    return None


class WandbProfile(ProviderProfile):
    """W&B inference — per-model-family reasoning dispatch on a shared vLLM backend."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        if not isinstance(reasoning_config, dict):
            return extra_body, top_level

        enabled = reasoning_config.get("enabled", True)
        effort = (reasoning_config.get("effort") or "").strip().lower()
        disabled = effort == "none" or enabled is False

        # Unset/unrecognised model → default to the GLM dialect (this
        # profile's fallback_models entry and Hermes' bundled config default).
        if not model or _is_glm_model(model):
            if disabled:
                extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                return extra_body, top_level
            glm_effort = _effort_to_glm(effort)
            if glm_effort:
                top_level["reasoning_effort"] = glm_effort
            return extra_body, top_level

        if _is_deepseek_thinking_model(model):
            # Distinct wire key: "thinking", not "enable_thinking" — matches
            # vLLM's dedicated deepseek_v3 reasoning parser. No granular
            # effort observed for this family, so it's never emitted.
            extra_body["chat_template_kwargs"] = {"thinking": not disabled}
            return extra_body, top_level

        if _is_always_on_reasoning_model(model):
            # No disable switch exists (W&B docs: "Always on") — a disable
            # request is a no-op. Effort still passes through when set.
            if not disabled:
                passthrough_effort = _effort_passthrough(effort)
                if passthrough_effort:
                    top_level["reasoning_effort"] = passthrough_effort
            return extra_body, top_level

        if _is_granular_toggle_model(model):
            if disabled:
                extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                return extra_body, top_level
            passthrough_effort = _effort_passthrough(effort)
            if passthrough_effort:
                top_level["reasoning_effort"] = passthrough_effort
            return extra_body, top_level

        # Everything else (Llama, Phi, non-thinking Qwen3-Instruct variants,
        # IBM Granite, JetBrains Mellum2, etc.) has no reasoning support.
        return extra_body, top_level


wandb = WandbProfile(
    name="wandb",
    aliases=("wandb-ai", "weights-and-biases"),
    env_vars=("WANDB_API_KEY",),
    display_name="W&B Inference",
    description="Weights & Biases inference API — 29 models on a shared vLLM backend",
    signup_url="https://wandb.ai",
    base_url="https://api.inference.wandb.ai/v1",
    fallback_models=(
        "zai-org/GLM-5.2",
        "zai-org/GLM-5.1",
        "deepseek-ai/DeepSeek-V4-Pro",
        "deepseek-ai/DeepSeek-V4-Flash",
        "Qwen/Qwen3.6-35B-A3B",
        "moonshotai/Kimi-K2.7-Code",
        "openai/gpt-oss-120b",
    ),
    default_max_tokens=65536,
    # Cloudflare blocks the default urllib User-Agent; the transport
    # adds a browser UA via default_headers. Applies to every model on
    # the endpoint, not just GLM.
    default_headers={"User-Agent": "Mozilla/5.0"},
)

register_provider(wandb)

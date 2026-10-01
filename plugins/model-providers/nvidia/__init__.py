"""NVIDIA NIM provider profile."""

import re
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

from agent.reasoning_effort import clamp_effort, requested_effort

# GLM-5.3 on NIM is a thinking model behind an OpenAI-compatible front. Live-verified
# 2026-10-01 on integrate.api.nvidia.com (z-ai/glm-5.3): the top-level OpenAI
# ``reasoning_effort`` field is accepted and scales thinking — low/high/max all 200
# with reasoning-token counts scaling (low=26 vs server-default 1657 reasoning tokens
# on the same 30K-token prompt) — while ``ultra`` (Hermes-internal, above max on the
# ladder) is rejected with HTTP 400. ``medium`` is accepted by the endpoint but is NOT
# in NVIDIA's documented GLM-5.3 vocabulary (docs list low/high/max), so the declared
# vocabulary omits it and the shared clamp maps it down to ``low``.
#
# Scoped to the GLM-5.3 family only: the live verification is 5.3-exclusive, and
# NVIDIA's GLM-5.2 reference does not document a reasoning_effort field on this
# endpoint (Hermes models GLM-5.2's vocabulary elsewhere as high/max — see
# GLM52_EFFORTS in agent/reasoning_effort.py). GLM-5.2 can be added here once its
# NIM wire behavior is separately verified.
_GLM53_NIM_RE = re.compile(r"(?:^|[^a-z0-9])glm-5\.3(?![0-9])")
# NVIDIA documents reasoning_effort for GLM-5.3 (and 5.3-Flash) as low/high/max
# (max is the server default). Only these are declared; anything else clamps onto
# them or is not emitted at all.
_NIM_GLM53_EFFORTS = ("low", "high", "max")


class NvidiaProviderProfile(ProviderProfile):
    """NVIDIA NIM accepts a stricter ToolMessage schema than most OpenAI-compatible APIs."""

    @staticmethod
    def _needs_strip(msg: Any) -> bool:
        return isinstance(msg, dict) and msg.get("role") == "tool" and ("name" in msg or "tool_name" in msg)

    def supported_reasoning_efforts(self, model: str | None) -> tuple[str, ...] | None:
        """Documented GLM-5.3 wire vocabulary on NIM. None (unknown) for other models:
        only GLM-5.3 family wire behavior is verified on this endpoint — other
        families may 400 on reasoning parameters they do not implement."""
        if model and _GLM53_NIM_RE.search(model.lower()):
            return _NIM_GLM53_EFFORTS
        return None

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reasoning config → top-level ``reasoning_effort`` for GLM-5.3 on NIM.

        Without this mapping a configured ``agent.reasoning_effort`` — and Hermes's
        truncation-recovery reasoning-off override (which sends ``enabled: False``) —
        never reach the wire, so the server-default thinking tier silently consumes
        the output budget: observed live as finish_reason='length' with ~60-67K chars
        of reasoning against ~200 chars of visible content.

        Emits nothing when the user expressed no preference (the server default stays
        in charge, matching the Z.AI profile contract) and nothing for models outside
        the GLM-5.3 family (their wire behavior is unverified on this endpoint).

        The recovery floor: a disable request (``enabled: False`` — the one-shot
        thinking-only-truncation recovery override) is represented on this wire as
        ``low``, NOT as an actual off switch. Live-verified 2026-10-01,
        ``reasoning_effort: none`` and ``chat_template_kwargs.enable_thinking: false``
        are ACCEPTED but do not suppress thinking on large prompts (1657 and 2048
        reasoning tokens of a 2048-token budget respectively, near-zero visible
        content), while ``low`` cut reasoning to 26 tokens and produced a clean full
        answer. Hermes' internal "reasoning off" therefore cannot literally turn
        reasoning off on this route; the least-reasoning tier that actually works is
        the recovery representation, so the recovery budget reaches the answer.

        Fail-closed: an effort name that is not a Hermes ladder level (or that the
        clamp cannot place on this wire's vocabulary) is dropped entirely rather than
        shipped verbatim — an unknown value reaching this endpoint 400s (the PR's own
        live probe rejected ``ultra``), and a bespoke name escaping the vocabulary
        would reintroduce exactly the unpredictability this hook exists to remove.
        """
        model = context.get("model")
        if not model or not _GLM53_NIM_RE.search(model.lower()):
            return {}, {}
        if not isinstance(reasoning_config, dict):
            return {}, {}
        if reasoning_config.get("enabled") is False:
            return {}, {"reasoning_effort": "low"}
        effort = requested_effort(reasoning_config)
        if not effort:
            return {}, {}
        clamped = clamp_effort(effort, _NIM_GLM53_EFFORTS)
        if clamped not in _NIM_GLM53_EFFORTS:
            return {}, {}
        return {}, {"reasoning_effort": clamped}

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Copy-on-write: only tool messages that lose a field are copied
        (no deep copy of large tool outputs); untouched input returned as-is."""
        if not any(self._needs_strip(msg) for msg in messages):
            return messages
        return [
            {k: v for k, v in msg.items() if k not in ("name", "tool_name")} if self._needs_strip(msg) else msg
            for msg in messages
        ]


nvidia = NvidiaProviderProfile(
    name="nvidia", aliases=("nvidia-nim", "nim", "build-nvidia", "nemotron"), env_vars=("NVIDIA_API_KEY",), display_name="NVIDIA NIM",
    description="NVIDIA NIM — accelerated inference", signup_url="https://build.nvidia.com/",
    fallback_models=("nvidia/llama-3.1-nemotron-70b-instruct", "nvidia/llama-3.3-70b-instruct"),
    base_url="https://integrate.api.nvidia.com/v1", default_max_tokens=16384,
)

register_provider(nvidia)

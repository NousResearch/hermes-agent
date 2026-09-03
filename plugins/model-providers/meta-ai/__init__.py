"""Meta Model API (Muse Spark) provider plugin for Hermes Agent.

Provider profile for Meta Superintelligence Labs' Muse Spark family, served
via the OpenAI-compatible Meta Model API at ``https://api.meta.ai/v1``.

Bundled from https://github.com/albertodepaola/hermes-meta-provider. Hermes'
provider discovery (``providers/__init__.py``) imports it on first
``get_provider_profile()`` / ``list_providers()`` call, and the module-level
``register_provider()`` below wires it into the registry.

Design notes
------------
* **Zero core edits.** Everything rides on ``ProviderProfile`` hooks. No changes
  to hermes' ``model_metadata.py`` / ``models.py`` / ``run_agent.py`` are needed:
  - Context window (1M), reasoning and vision capabilities already resolve from
    models.dev for the muse-spark family, so no static ctx table entry is required.
  - The reasoning dial is emitted as a **top-level ``reasoning_effort``** kwarg
    (returned in the ``top_level`` slot of ``build_api_kwargs_extras``), which the
    chat-completions transport merges unconditionally. This deliberately avoids
    the ``extra_body.reasoning`` path, whose emission is gated by a hardcoded
    host allowlist in core (``AIAgent._supports_reasoning_extra_body``) that a
    third-party plugin must not edit.
* **Meta 400 on ``reasoning_effort: "none"``.** Muse rejects ``none``; disabling
  reasoning maps to ``"minimal"`` instead.
* **``default_max_tokens=16384``.** Muse spends completion budget on hidden
  reasoning tokens first; small caps can finish with empty content.
"""

from __future__ import annotations

import os
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


def _resolve_effort(reasoning_config: dict | None) -> str:
    """Map Hermes' reasoning_config to a Meta-safe ``reasoning_effort`` value.

    Meta's vocabulary (minimal..xhigh; rejects ``none``) is declared in
    agent.reasoning_effort. Disabled/"none" maps to ``minimal`` (the closest
    Meta has to off); unset/bespoke levels fall to ``medium``.
    """
    rc = reasoning_config or {}
    if rc.get("enabled") is False:
        return "minimal"
    effort = str(rc.get("effort") or "").strip().lower()
    if effort in {"", "none"}:
        return "minimal" if effort == "none" else "medium"

    from agent.reasoning_effort import META_AI_EFFORTS, clamp_effort

    clamped = clamp_effort(effort, META_AI_EFFORTS)
    return clamped if clamped in META_AI_EFFORTS else "medium"


class MetaAIProfile(ProviderProfile):
    """Meta Model API — top-level reasoning_effort, self-contained."""

    # Non-chat model prefixes excluded from the agent picker.  The live
    # /v1/models catalog includes image-generation and transcription models
    # that are not suitable for agentic chat.
    _NON_CHAT_PREFIXES = ("muse-image-", "muse-voice-")

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch and filter the live catalog, excluding non-chat models."""
        live = super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)
        if live is None:
            return None
        return [
            m for m in live
            if not any(m.startswith(p) for p in self._NON_CHAT_PREFIXES)
        ]

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        supports_reasoning: bool = False,  # noqa: ARG002 — we self-gate below
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Emit ``reasoning_effort`` as a top-level api kwarg.

        We ignore the core ``supports_reasoning`` gate on purpose: that flag is
        driven by a host allowlist in core we cannot (and should not) edit from
        an out-of-tree plugin. Muse Spark always accepts ``reasoning_effort``,
        so we resolve it from ``reasoning_config`` directly.
        """
        return {}, {"reasoning_effort": _resolve_effort(reasoning_config)}


def _base_url() -> str:
    """Allow a base-URL override via ``META_BASE_URL`` without editing config."""
    return os.getenv("META_BASE_URL", "").strip() or "https://api.meta.ai/v1"


meta_ai = MetaAIProfile(
    name="meta-ai",
    aliases=("meta", "muse", "muse-spark", "model-api", "msl"),
    display_name="Meta Model API",
    description="Meta Muse Spark family (Meta Superintelligence Labs)",
    signup_url="https://developer.meta.com/ai/",
    # MODEL_API_KEY is Meta's documented env var; the aliases are conveniences.
    env_vars=("MODEL_API_KEY", "META_API_KEY", "META_MODEL_API_KEY", "META_BASE_URL"),
    base_url=_base_url(),
    auth_type="api_key",
    # Responses API is the wire that engages Muse prompt caching: measured
    # 0 cached tokens on /v1/chat/completions vs 93-99% cache hits on
    # /v1/responses with prompt_cache_retention (see host_mandated_api_mode
    # in hermes_cli/providers.py and the retention hint in
    # agent/transports/codex.py). The MetaAIProfile chat-completions hook
    # above still covers custom OpenAI-compatible endpoints configured with
    # a non-api.meta.ai base URL, which fall through to chat_completions.
    api_mode="codex_responses",
    # Muse Spark is natively multimodal (image/video/pdf/audio in, text out).
    supports_vision=True,
    # Cheap contributor tier is a good default for auxiliary tasks
    # (compaction, title generation, vision) when this is the main provider.
    default_aux_model="muse-spark-1.2-contributor",
    # Muse spends completion budget on hidden reasoning tokens first; a low cap
    # can finish with empty content. 16k is a safe floor.
    default_max_tokens=16384,
    # Minimal fallback shown when the live /v1/models fetch fails or no
    # credentials are configured yet. Keep this list small — just enough so
    # the picker isn't empty when the API is unreachable.
    fallback_models=(
        "muse-spark-1.2",
    ),
)

register_provider(meta_ai)

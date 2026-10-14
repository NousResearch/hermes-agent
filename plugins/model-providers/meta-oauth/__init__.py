"""Meta (Muse subscription) provider profile — https://api.meta.ai/v1.

Login, key minting and refresh live in core (``hermes_cli/auth_meta.py``),
exactly like the other bundled subscription providers (``openai-codex``,
``xai-oauth``): this profile deliberately ships no ``auth_handler``. It exists
so the Meta wire quirks declared on the ``meta-ai`` profile also apply to
subscription turns — without it, ``meta-oauth`` resolves no profile and the
generic defaults take over (image tool results are sent to an endpoint that
400s on them, and the reasoning-effort mapping is lost).

Intentional parity copy of ``plugins/model-providers/meta-ai/``: the hooks
below must stay in sync with ``MetaAIProfile``; only identity/auth fields
differ.
"""

from typing import Any

from agent.reasoning_effort import META_AI_EFFORTS, clamp_effort
from providers import register_provider
from providers.base import ProviderProfile


class MetaOAuthProfile(ProviderProfile):
    """Meta Model API via Muse subscription — top-level reasoning_effort."""

    # Non-chat model prefixes excluded from the agent picker (see meta-ai).
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
        self, *, reasoning_config: dict | None = None, supports_reasoning: bool = False, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Muse Spark always accepts ``reasoning_effort`` and 400s on ``none``:
        disabled/"none" -> ``minimal`` (closest to off); unset/bespoke levels
        -> ``medium`` (see meta-ai)."""
        rc = reasoning_config or {}
        effort = str(rc.get("effort") or "").strip().lower()
        if rc.get("enabled") is False or effort == "none":
            mapped = "minimal"
        else:
            clamped = clamp_effort(effort, META_AI_EFFORTS)
            mapped = clamped if clamped in META_AI_EFFORTS else "medium"
        return {}, {"reasoning_effort": mapped}


meta_oauth = MetaOAuthProfile(
    name="meta-oauth", aliases=("meta-subscription", "muse-subscription", "muse-code-subscription"),
    display_name="Meta (Muse subscription)",
    description="Meta Muse Spark billed to a Muse Code subscription (OAuth sign-in)",
    signup_url="https://developer.meta.com/ai/",
    env_vars=(),  # OAuth subscription — no API key; login lives in hermes_cli/auth_meta.py
    base_url="https://api.meta.ai/v1", auth_type="oauth_external",
    # Responses API engages Muse prompt caching (see meta-ai).
    api_mode="codex_responses",
    # Natively multimodal, but only on user turns: an image envelope inside a
    # role:tool message 400s (see meta-ai / #101668).
    supports_vision=True, supports_vision_tool_messages=False,
    # No default_aux_model: the contributor tiers are a training-data opt-in,
    # so auxiliary tasks stay on the user's chosen model unless configured.
    # Muse spends completion budget on hidden reasoning first; low caps can finish with empty content.
    default_max_tokens=16384,
    # Mirror of the static ``_PROVIDER_MODELS["meta-oauth"]`` rows, shown when
    # no live catalog is available; a test keeps the two in sync.
    fallback_models=("muse-spark-1.3", "muse-spark-1.3-contributor"),
)

register_provider(meta_oauth)

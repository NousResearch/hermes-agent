"""Xiaomi MiMo provider profile.

The v2.6 generation takes ``extra_body.thinking`` (the documented Deep Thinking toggle)
plus a top-level ``reasoning_effort`` of exactly none/low/medium/high — live-verified
2026-10 on api.xiaomimimo.com; any other level 400s "Param Incorrect". Without this
profile's hook the transport emitted no reasoning fields at all, so every ``/reasoning``
level was a silent no-op and thinking could never be disabled (#27325, #17314).
v2.5 and earlier are retired on the endpoint (400 "Unsupported model") and predate
the knob.
"""

from typing import Any

from agent.reasoning_effort import MIMO_EFFORTS, thinking_toggle_extras
from providers import register_provider
from providers.base import ProviderProfile


def _thinking_capable(model: str | None) -> bool:
    """True when *model*'s endpoint accepts ``thinking`` / ``reasoning_effort``.

    The v2.6 generation only; a future generation widens the prefix here once its
    contract is verified the same way.
    """
    m = (model or "").strip().lower().rsplit("/", 1)[-1]
    return m.startswith("mimo-v2.6")


def _setup_efforts(model: str) -> tuple[str, ...]:
    """Picker levels for *model*: exactly MiMo's three, else no reasoning control."""
    return MIMO_EFFORTS if _thinking_capable(model) else ()


class XiaomiProfile(ProviderProfile):
    """Xiaomi MiMo — extra_body.thinking toggle + top-level reasoning_effort (low/medium/high)."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not _thinking_capable(model):
            return {}, {}
        extras, top_level = thinking_toggle_extras(reasoning_config, MIMO_EFFORTS)
        # ``{effort: "none"|"false"|"disabled"}`` is also an explicit disable: UI/session
        # surfaces report Off from the effort field even when ``enabled`` is missing or
        # still True (same #107238 shape as DeepSeek).
        rc = reasoning_config
        effort = (rc.get("effort") or "").strip().lower() if isinstance(rc, dict) else ""
        if rc is not None and effort in {"none", "false", "disabled"}:
            return {"thinking": {"type": "disabled"}}, {}
        return extras, top_level


xiaomi = XiaomiProfile(
    name="xiaomi", aliases=("mimo", "xiaomi-mimo"), env_vars=("XIAOMI_API_KEY",),
    base_url="https://api.xiaomimimo.com/v1",
    supports_health_check=False,  # /v1/models returns 401 even with valid key
    supports_vision=True,  # mimo-v2-omni is vision-capable
    supports_vision_tool_messages=False,  # rejects list-type tool content (400 "text is not set")
    setup_reasoning_efforts=_setup_efforts,
)

register_provider(xiaomi)

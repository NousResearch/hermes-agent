"""Xiaomi MiMo provider profile.

MiMo thinking-capable chat models steer reasoning with a binary
``extra_body.thinking`` toggle on Chat Completions - Xiaomi documents no
graded ``reasoning_effort`` on that wire - and the server default when the
field is omitted is not documented as stable, so the toggle is sent
explicitly on every thinking-capable request (DeepSeek-style, see
``plugins/model-providers/deepseek``). asr/tts ids never think and stay
untouched.
"""

from typing import Any

from agent.reasoning_effort import thinking_toggle_extras
from providers import register_provider
from providers.base import ProviderProfile

# Thinking-capable MiMo chat ids (api.xiaomimimo.com /v1/models): the v2.6 family
# by prefix, plus the v2.5-pro / v2 / v2-flash chat models still addressable by id.
# asr / tts ids are deliberately excluded - they never take the toggle.
_THINKING_PREFIXES: tuple[str, ...] = ("mimo-v2.6-",)
_THINKING_IDS: frozenset[str] = frozenset({"mimo-v2.5-pro", "mimo-v2-pro", "mimo-v2-flash"})


class XiaomiMiMoProfile(ProviderProfile):
    """Xiaomi MiMo - binary ``extra_body.thinking`` toggle, never a graded effort."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        m = (model or "").strip().lower()
        if not (m.startswith(_THINKING_PREFIXES) or m in _THINKING_IDS):
            return {}, {}
        # Empty effort set: the binary toggle is the whole MiMo contract, so the
        # helper can never emit a graded top-level ``reasoning_effort``.
        extras, top_level = thinking_toggle_extras(reasoning_config, (), always_emit_toggle=True)
        # ``{effort: "none"|"false"|"disabled"}`` is also an explicit disable:
        # UI/session surfaces report Off from the effort field even when
        # ``enabled`` is missing or still True (#107238) - parity with DeepSeek.
        rc = reasoning_config
        effort = (rc.get("effort") or "").strip().lower() if isinstance(rc, dict) else ""
        if rc is not None and effort in {"none", "false", "disabled"}:
            return {"thinking": {"type": "disabled"}}, {}
        return extras, top_level


xiaomi = XiaomiMiMoProfile(
    name="xiaomi", aliases=("mimo", "xiaomi-mimo"), env_vars=("XIAOMI_API_KEY",),
    base_url="https://api.xiaomimimo.com/v1",
    supports_health_check=False,  # /v1/models returns 401 even with valid key
    supports_vision=True,  # mimo-v2-omni is vision-capable
    supports_vision_tool_messages=False,  # rejects list-type tool content (400 "text is not set")
)

register_provider(xiaomi)

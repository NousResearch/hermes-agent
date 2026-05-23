"""DeepSeek provider profile.

History
-------

The original version of this profile (#15700, #17212, #17825) mirrored the
Kimi / Moonshot wire shape DeepSeek's OpenAI-compat endpoint used to
accept:

    {"reasoning_effort": "<low|medium|high|max>",
     "extra_body": {"thinking": {"type": "enabled" | "disabled"}}}

and injected ``extra_body.thinking = {"type": "enabled"}`` on every
request that targeted a V4-family or ``deepseek-reasoner`` model, so the
``reasoning_content`` echo-back contract enforced by the API would line
up with Hermes' history-replay code.

#30818 — what changed
---------------------

The DeepSeek V4 API (``deepseek-v4-flash`` / ``deepseek-v4-pro``, all
plans, both ``api.deepseek.com`` and ``api.deepseek.com/v1``) now
rejects the ``thinking`` field with HTTP 400 on the very first message,
before any tool calls or history exist.  A literal ``curl`` against the
same endpoint with the same key and message succeeds, and switching to
``provider: custom`` + ``api_mode: openai-completions`` — which bypasses
this profile entirely — also succeeds, confirming the smoking gun is
the unconditional ``extra_body.thinking`` injection.

The reasoning-content echo concern that motivated injecting
``extra_body.thinking`` in the first place is already covered on the
RESPONSE side: ``agent/chat_completion_helpers.build_assistant_message``
pads assistant tool-call messages with ``reasoning_content = " "`` (or
the captured reasoning text) for every thinking-mode provider when the
SDK didn't surface it explicitly — see ``_needs_deepseek_tool_reasoning``
in ``run_agent.py``.  No request-side flag is needed to keep that path
working.

Current behavior
----------------

* Default (no ``reasoning_config`` provided) — emit nothing extra.  The
  DeepSeek API applies its server-side defaults and the request succeeds.
* User opt-in (``reasoning_config={"enabled": True/False}``) — still
  forward the Kimi-style ``extra_body.thinking`` shape.  Users who
  explicitly want to control thinking can do so; users who didn't ask
  for it don't get a 400.
* ``reasoning_effort`` (top-level) — only forwarded when the user
  configured ``reasoning_config.effort`` explicitly, never injected
  by default.

Non-thinking models (only ``deepseek-chat`` today, which is V3) remain
no-ops so we don't perturb the V3 wire format.
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


def _model_supports_thinking(model: str | None) -> bool:
    """DeepSeek thinking-capable model families.

    Currently covers the V4 family (``deepseek-v4-pro``, ``deepseek-v4-flash``,
    and any future ``deepseek-v4-*`` variants) and the legacy
    ``deepseek-reasoner`` (R1).  ``deepseek-chat`` is V3 with no thinking mode.
    """
    m = (model or "").strip().lower()
    if not m:
        return False
    if m.startswith("deepseek-v") and not m.startswith("deepseek-v3"):
        # deepseek-v4-*, deepseek-v5-*, etc. — every V4+ generation has
        # thinking. v3 explicitly excluded.
        return True
    if m == "deepseek-reasoner":
        return True
    return False


def _user_opted_into_thinking_config(reasoning_config: dict | None) -> bool:
    """Return True when the user explicitly configured thinking mode.

    Distinguishing "user didn't ask" from "user passed an empty dict" is
    what lets the default path stay quiet (no ``extra_body.thinking``
    emitted at all — see #30818) while still honouring an explicit opt-in.

    Only ``reasoning_config.enabled`` toggles the legacy Kimi-style
    ``thinking`` payload; ``reasoning_config.effort`` controls
    ``reasoning_effort`` separately and does NOT imply
    ``extra_body.thinking`` on its own (effort can be set with the
    server default thinking behavior left untouched).
    """
    if not isinstance(reasoning_config, dict):
        return False
    return "enabled" in reasoning_config


class DeepSeekProfile(ProviderProfile):
    """DeepSeek — opt-in extra_body.thinking + opt-in reasoning_effort.

    See module docstring for the full rationale.  The short version: the
    DeepSeek V4 native API returns HTTP 400 when an unconfigured client
    sends ``extra_body.thinking``, so we only forward it when the user
    explicitly opted in via ``reasoning_config.enabled``.
    """

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        if not _model_supports_thinking(model):
            # V3 / unknown — leave wire format untouched, current behavior.
            return extra_body, top_level

        # #30818 — only emit ``extra_body.thinking`` when the user
        # explicitly configured ``reasoning_config.enabled``.  The
        # DeepSeek V4 native API rejects the field outright on first
        # use, so injecting it by default would break ``provider:
        # deepseek`` for every user with the default config.  Users who
        # depend on the Kimi-style explicit toggle can still opt in.
        if _user_opted_into_thinking_config(reasoning_config):
            enabled = reasoning_config.get("enabled") is not False
            extra_body["thinking"] = {
                "type": "enabled" if enabled else "disabled"
            }
            if not enabled:
                # Disabled thinking → no reasoning_effort either.
                return extra_body, top_level

        # Effort mapping.  Pass low/medium/high through; xhigh/max → max.
        # When no effort is set we omit reasoning_effort so DeepSeek applies
        # its server default (currently high).  This branch fires
        # whether or not the user opted into ``extra_body.thinking`` —
        # effort can be tuned independently as long as the user knows
        # the model supports it.
        if isinstance(reasoning_config, dict):
            effort = (reasoning_config.get("effort") or "").strip().lower()
            if effort in {"xhigh", "max"}:
                top_level["reasoning_effort"] = "max"
            elif effort in {"low", "medium", "high"}:
                top_level["reasoning_effort"] = effort

        return extra_body, top_level


deepseek = DeepSeekProfile(
    name="deepseek",
    aliases=("deepseek-chat",),
    env_vars=("DEEPSEEK_API_KEY",),
    display_name="DeepSeek",
    description="DeepSeek — native DeepSeek API",
    signup_url="https://platform.deepseek.com/",
    fallback_models=(
        "deepseek-chat",
        "deepseek-reasoner",
    ),
    base_url="https://api.deepseek.com/v1",
    default_aux_model="deepseek-chat",
)

register_provider(deepseek)

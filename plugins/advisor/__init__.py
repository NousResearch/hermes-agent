"""Advisor plugin — persistent second-model review for each agent turn.

Registers:
  - ``post_llm_call`` hook — fires at end of each turn, triggers review
  - ``/advisor`` slash command — on/off/status
"""

import logging

from .runtime import AdvisorRuntime

logger = logging.getLogger(__name__)

_runtime: AdvisorRuntime | None = None


def register(ctx):
    global _runtime

    _runtime = AdvisorRuntime(ctx)

    # Register the post_llm_call hook — fires once per turn at completion.
    # Kwargs include: turn_id, user_message, assistant_response,
    # conversation_history, model, session_id, task_id, platform.
    ctx.register_hook("post_llm_call", _on_post_llm_call)

    # Register the /advisor slash command
    ctx.register_command(
        name="advisor",
        handler=_handle_advisor,
        description="Toggle/inspect advisor. Usage: /advisor [on|off|status]",
    )

    logger.info("Advisor plugin registered (state=%s)",
                "enabled" if _runtime.state.enabled else "disabled")


def _on_post_llm_call(**kwargs):
    """Hook: fires after each completed agent turn."""
    runtime = _get_runtime()
    if runtime is None:
        return
    runtime.on_post_llm_call(**kwargs)


def _handle_advisor(args: str) -> str:
    """Slash command handler for /advisor [on|off|status]."""
    runtime = _get_runtime()
    if runtime is None:
        return "Advisor plugin not initialized."
    return runtime.handle_command(args)


def _get_runtime() -> AdvisorRuntime | None:
    return _runtime

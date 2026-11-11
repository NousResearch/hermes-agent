"""Init-time warning for Nous Hermes chat models that are not tuned for tool calls.

Split out of ``agent/agent_init.py``, which calls it.
"""

from __future__ import annotations

from contextlib import suppress

from agent.agent_runtime_helpers import _ra


def _warn_nonagentic_hermes_model(agent):
    # Nous Hermes 3/4 are chat models, not tool-call-tuned. cli.py show_banner() already
    # warns on the CLI, so skip platform=="cli"; non-quiet non-CLI surfaces still get it.
    if agent.quiet_mode or (agent.platform or "cli") == "cli":
        return
    with suppress(Exception):
        from hermes_cli.model_switch import _check_hermes_model_warning
        _hermes_warn = _check_hermes_model_warning(agent.model or "")
        if _hermes_warn:
            _user_msg = (
                "⚠ Nous Research Hermes 3 & 4 models are NOT agentic — they "
                "lack reliable tool-calling for agent workflows (delegation, "
                "cron, proactive tools). Consider an agentic model instead "
                "(Claude, GPT, Gemini, Qwen-Coder, etc.)."
            )
            agent._emit_warning(_user_msg)
            _ra().logger.warning(_hermes_warn)

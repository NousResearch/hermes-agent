"""The opt-in prompt warmer, the last step of agent init.

Split out of ``agent/agent_init.py``.
"""

from __future__ import annotations

import logging

from agent.memory_provider import spawn_context_thread

logger = logging.getLogger("run_agent")


def _start_prompt_warmer(agent) -> None:
    """Opt-in prompt warmer: a profile exposing prompt_warmer_enabled / warm_prompt_cache may prime the
    server's prompt cache with the session preamble. Must be the LAST init step so the preamble matches the
    first turn's build. Daemon thread: a cold llama-swap model can take minutes to load, and session start
    must not block."""
    try:
        from providers import resolve_provider_profile
        profile = resolve_provider_profile(agent.provider, getattr(agent, "requested_provider", None))
        enabled_fn = getattr(profile, "prompt_warmer_enabled", None)
        warm_fn = getattr(profile, "warm_prompt_cache", None)
        if not (callable(enabled_fn) and callable(warm_fn)
                and enabled_fn(base_url=agent.base_url, model=agent.model)):
            return
        # Apply the (normally lazy) terminal.* config -> env bridge now so the prebuild is byte-identical to
        # the first turn's build; otherwise the identity gate in conversation_loop rejects it as cwd drift.
        try:
            from hermes_cli.config import apply_terminal_config_to_env
            apply_terminal_config_to_env()
        except Exception:
            logger.debug("prompt warmer: terminal config bridge failed", exc_info=True)
        warm_system_prompt = agent._build_system_prompt(None)
        # conversation_loop reuses this exact string on the first turn (identity-gated, consumed one-shot).
        agent._warm_prebuilt_system_prompt = warm_system_prompt
        spawn_context_thread(
            warm_fn,
            kwargs={"base_url": agent.base_url, "model": agent.model, "system_prompt": warm_system_prompt,
                    "tools": getattr(agent, "tools", None),
                    "reasoning_config": getattr(agent, "reasoning_config", None)},
            name="llamacpp-prompt-warmer",
        ).start()
    except Exception:
        logger.debug("prompt warmer skipped", exc_info=True)

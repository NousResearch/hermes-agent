"""Apply config.yaml's ``agent:`` section (and skills nudge) to a new agent."""

from __future__ import annotations

import logging
from contextlib import suppress

from agent.iteration_budget import normalize_budget_warning_ratio

logger = logging.getLogger("run_agent")


def _apply_agent_section(agent, _agent_cfg):
    from agent.agent_init import _cfg_dict, _normalize_run_budget_seconds

    # Skills config: nudge interval for skill creation reminders
    agent._skill_nudge_interval = 10
    with suppress(Exception):
        agent._skill_nudge_interval = int(_agent_cfg.get("skills", {}).get("creation_nudge_interval", 10))

    _agent_section = _cfg_dict(_agent_cfg, "agent")
    agent.budget_warning_ratio = normalize_budget_warning_ratio(
        _agent_section.get("budget_warning_ratio")
    )
    # Both: "auto" (model-list match), true, false, or list of model substrings; independent
    # of each other (gates in agent/system_prompt.py).
    agent._tool_use_enforcement = _agent_section.get("tool_use_enforcement", "auto")
    agent._execution_guidance = _agent_section.get("execution_guidance", "auto")

    # Wall-clock run budget from config — only when the constructor arg was not given.
    if agent.run_budget_seconds is None:
        agent.run_budget_seconds = _normalize_run_budget_seconds(
            _agent_section.get("run_budget_seconds")
        )

    # Empty-response guard: a malformed section falls back to schema defaults (on, $0.25).
    from agent.empty_response_guard import resolve_guard_settings
    (
        agent._empty_guard_enabled, agent._empty_guard_cost_threshold_usd
    ) = resolve_guard_settings(_agent_section.get("empty_response_guard"))

    # "auto" (codex_responses only), true (all api_modes), false, or model substrings.
    agent._intent_ack_continuation = _agent_section.get("intent_ack_continuation", "auto")

    # apply_patch file writes: "auto" (GPT-5+ on OpenRouter), true, false, or model substrings added to auto.
    # Resolved per request against the model then in use (agent/apply_patch_tool.py).
    agent._apply_patch_tool = _agent_section.get("apply_patch_tool", "auto")

    # Responses `text.verbosity`: "" / unknown value = not sent (never flips the provider default).
    _verbosity = str(_agent_section.get("text_verbosity") or "").strip().lower()
    if _verbosity and _verbosity not in {"low", "medium", "high"}:
        logger.warning("Unknown agent.text_verbosity %r; expected low, medium or high — ignoring", _verbosity)
        _verbosity = ""
    agent.text_verbosity = _verbosity or None

    # Default-on boolean gates: anti-stall guards (notice-only), universal guidance toggles
    # (ALL models, unlike enforcement), the local toolchain probe, Bot Mode protocol section.
    for _key in (
        "stall_guards", "task_completion_guidance", "parallel_tool_call_guidance",
        "environment_probe", "bot_mode_protocol",
    ):
        setattr(agent, f"_{_key}", bool(_agent_section.get(_key, True)))
    # Warm the probe (~0.5s of subprocesses) off-thread so the first prompt build finds it cached.
    if agent._environment_probe:
        with suppress(Exception):
            from tools.env_probe import warm_environment_probe_async
            warm_environment_probe_async()

    # "Bot Chat" gate hint for hosts that defer the DB title write past the first prompt build.
    agent._session_title_hint = None

    # platform_hints: <platform>: {append|replace}, stored verbatim (agent/system_prompt.py).
    agent._platform_hint_overrides = _cfg_dict(_agent_cfg, "platform_hints")

    # App-level API retry count (wraps each model API call). Default 3; 1 = single attempt.
    try:
        _api_retries = max(int(_agent_section.get("api_max_retries", 3)), 1)
    except (TypeError, ValueError):
        _api_retries = 3
    agent._api_max_retries = _api_retries
    # Bounded post-exhaustion auto-recovery cycles once retries AND the fallback chain are spent
    # on a transient outage (agent/turn_recovery_autorecover.py). 0 disables the ladder.
    try:
        agent._auto_recovery_cycles = max(int(_agent_section.get("auto_recovery_cycles", 5)), 0)
    except (TypeError, ValueError):
        agent._auto_recovery_cycles = 5

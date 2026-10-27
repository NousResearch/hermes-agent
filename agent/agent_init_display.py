"""Display and tool-loop settings applied during agent initialization."""

from __future__ import annotations

from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController


def apply_display_config(agent, agent_cfg, platform):
    # show_commentary: Codex phase=commentary → interim path (true) or reasoning channel.
    from agent.agent_init import _cfg_dict
    agent.show_commentary = bool(_cfg_dict(agent_cfg, "display").get("show_commentary", True))

    # Window (seconds) for the bounded /fast auto|cold modes (agent.fast_mode).
    agent.fast_auto_seconds = (agent_cfg.get("agent") or {}).get("fast_auto_seconds", 60)

    # lmstudio_load_mode: "explicit" (preload via management API) or "jit" (Auto-Evict path).
    model_section = _cfg_dict(agent_cfg, "model")
    load_mode = str(model_section.get("lmstudio_load_mode", "explicit") or "explicit").strip().lower()
    agent.lmstudio_load_mode = load_mode if load_mode in {"explicit", "jit"} else "explicit"
    if agent.lmstudio_load_mode != load_mode:
        from agent.agent_runtime_helpers import _ra
        _ra().logger.warning(
            "Invalid model.lmstudio_load_mode=%r; expected 'explicit' or 'jit'. Using explicit.",
            model_section.get("lmstudio_load_mode"),
        )

    # model.streaming=false seeds _disable_streaming (the loop's runtime fallback) for
    # backends with broken streaming tool calls. Session-scoped; orthogonal to display.streaming.
    streaming = str(model_section.get("streaming", "true")).strip().lower()
    agent._disable_streaming = streaming in {"false", "0", "no", "off"}
    if not agent._disable_streaming and streaming not in {"true", "1", "yes", "on"}:
        from agent.agent_runtime_helpers import _ra
        _ra().logger.warning(
            "Invalid model.streaming=%r; expected a boolean. Using streaming (default).",
            model_section.get("streaming"),
        )
    agent._stream_5xx_probe_ts = None  # monotonic time of the last streaming-5xx unmask probe

    try:
        agent._tool_guardrails = ToolCallGuardrailController(
            ToolCallGuardrailConfig.from_mapping(
                agent_cfg.get("tool_loop_guardrails", {}), platform=platform,
            )
        )
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        from agent.agent_runtime_helpers import _ra
        _ra().logger.warning("Tool loop guardrail config ignored: %s", error)

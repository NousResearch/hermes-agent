"""Display and loop-runtime config applied at agent init (facade size cap).

Split from ``agent.agent_init``. The late import below reads the facade's ``_cfg_dict``/``_ra`` at
call time, so the config seam the rest of init uses stays owned there.
"""

from __future__ import annotations

import logging

from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController

logger = logging.getLogger("run_agent")


def _apply_display_config(agent, _agent_cfg, platform):
    from agent.agent_init import _cfg_dict, _ra

    # show_commentary: Codex phase=commentary → interim path (true) or reasoning channel.
    agent.show_commentary = bool(_cfg_dict(_agent_cfg, "display").get("show_commentary", True))

    # Window (seconds) for the bounded /fast auto|cold modes (agent.fast_mode).
    agent.fast_auto_seconds = (_agent_cfg.get("agent") or {}).get("fast_auto_seconds", 60)

    # lmstudio_load_mode: "explicit" (preload via management API) or "jit" (Auto-Evict path).
    _model_section = _cfg_dict(_agent_cfg, "model")
    _load_mode = str(_model_section.get("lmstudio_load_mode", "explicit") or "explicit").strip().lower()
    agent.lmstudio_load_mode = _load_mode if _load_mode in {"explicit", "jit"} else "explicit"
    if agent.lmstudio_load_mode != _load_mode:
        logger.warning(
            "Invalid model.lmstudio_load_mode=%r; expected 'explicit' or 'jit'. Using explicit.",
            _model_section.get("lmstudio_load_mode"),
        )

    # model.streaming=false seeds _disable_streaming (the loop's runtime fallback) for
    # backends with broken streaming tool calls. Session-scoped; orthogonal to display.streaming.
    _streaming = str(_model_section.get("streaming", "true")).strip().lower()
    agent._disable_streaming = _streaming in {"false", "0", "no", "off"}
    if not agent._disable_streaming and _streaming not in {"true", "1", "yes", "on"}:
        logger.warning(
            "Invalid model.streaming=%r; expected a boolean. Using streaming (default).",
            _model_section.get("streaming"),
        )
    agent._stream_5xx_probe_ts = None  # monotonic time of the last streaming-5xx unmask probe

    try:
        agent._tool_guardrails = ToolCallGuardrailController(
            ToolCallGuardrailConfig.from_mapping(
                _agent_cfg.get("tool_loop_guardrails", {}), platform=platform,
            )
        )
    except Exception as _tlg_err:
        _ra().logger.warning("Tool loop guardrail config ignored: %s", _tlg_err)

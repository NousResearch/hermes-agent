"""In-process session controls, applied only inside the adapter's idle mutation boundary."""
from __future__ import annotations

import copy

from acp_adapter.config_options import fast_overrides, parse_selection, runtime_info


def apply_control(agent, config_id, value):
    from agent.fast_mode import STATIC_TIERS, parse_exact_service_tier
    from hermes_constants import parse_reasoning_effort
    if config_id == "reasoning":
        agent.reasoning_config = parse_reasoning_effort(value)
    elif config_id == "fast":
        tier = parse_exact_service_tier(value) or None
        overrides = dict(getattr(agent, "request_overrides", None) or {})
        overrides.pop("service_tier", None)
        overrides.pop("speed", None)
        extra = dict(overrides.get("extra_body") or {})
        extra.pop("service_tier", None)
        extra.pop("speed", None)
        if "extra_body" in overrides:
            overrides["extra_body"] = extra
        if tier in STATIC_TIERS:
            info = {key: getattr(agent, key, None) for key in ("model", "provider", "base_url")}
            overrides.update(fast_overrides(info, value) or {})
        agent.service_tier, agent.request_overrides = tier, overrides
        agent._fast_until = 0.0
    init = getattr(agent, "_session_init_model_config", None)
    if isinstance(init, dict):
        init["reasoning_config"] = copy.deepcopy(getattr(agent, "reasoning_config", None))
        init["service_tier"] = getattr(agent, "service_tier", None)


def apply_controls(agent, controls):
    for config_id, value in controls.items():
        # Model/route were already passed to the constructor by restore/fork/rebuild.
        if config_id != "model":
            apply_control(agent, config_id, value)


def restore_controls(state, controls):
    state.control_values = copy.deepcopy(controls)
    apply_controls(state.agent, state.control_values)


class SessionControlsMixin:
    def _control_fields(self, state):
        from acp_adapter.config_options import project_options
        models = self._build_model_state(state)
        return {"models": models, "config_options": project_options(runtime_info(state), models)}

    def _config_options(self, state):
        return self._control_fields(state)["config_options"]

    def _set_control(self, state, config_id, value):
        info = runtime_info(state)
        if config_id == "reasoning" and value == "" and not info["reasoning_effort"]:
            return
        value = parse_selection(config_id, value, info)
        apply_control(state.agent, config_id, value)
        state.control_values[config_id] = value
        state.control_revision += 1
        self.session_manager.save_session(state.session_id)

    async def _send_config_options(self, state):
        import asyncio
        from acp.schema import ConfigOptionUpdate
        options = await asyncio.to_thread(self._config_options, state)
        await self._send(state.session_id, ConfigOptionUpdate(session_update="config_option_update",
                         config_options=options), fail_msg="Failed to update ACP options for %s")
        return options

    def _cmd_reasoning(self, args, state):
        return self._control_command("reasoning", args, state)

    def _cmd_fast(self, args, state):
        return self._control_command("fast", args, state)

    def _control_command(self, config_id, args, state):
        from acp_adapter.config_options import text_selection
        value = text_selection(args)
        if value and value != "status":
            self._set_control(state, config_id, value)
        option = next(o for o in self._config_options(state) if o.id == config_id)
        return f"{option.name}: {option.current_value or 'provider default'} (this session)"

    def _reset_controls(self, state):
        from agent.fast_mode import parse_service_tier, service_tier_word
        from hermes_cli.config import load_config
        from hermes_constants import resolve_reasoning_config
        config = load_config()
        state.agent.reasoning_config = resolve_reasoning_config(config, state.model)
        tier = parse_service_tier((config.get("agent") or {}).get("service_tier"))
        apply_control(state.agent, "fast", service_tier_word(tier))
        state.control_values.clear()
        state.control_revision += 1

"""Reasoning effort commands and the shared CLI recap display settings."""


def normalize_clamp_lines(value) -> int:
    """Coerce persisted values, retaining the 10-line default for malformed settings."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return 10
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        return 10
    return count if count > 0 else 10


def parse_clamp_command(value: str):
    """Return the requested count, or None for a different reasoning command."""
    parts = value.split()
    if not parts or parts[0] not in {"clamp", "collapse", "short"}:
        return None
    if len(parts) == 1:
        return 10
    try:
        count = int(parts[1]) if len(parts) == 2 else 0
    except ValueError:
        count = 0
    if count < 1:
        raise ValueError("Clamp line count must be a positive integer.")
    return count


# /reasoning display toggles: arg -> (attr, value, headline key, follow-up note key or None);
# the keys resolve under ``cli.commands.reasoning.*`` at call time.
_REASONING_TOGGLES = {
    **dict.fromkeys(("show", "on"), ("show_reasoning", True, "headline_on", "note_show")),
    **dict.fromkeys(("hide", "off"), ("show_reasoning", False, "headline_off", None)),
    **dict.fromkeys(("full", "all"), ("reasoning_full", True, "headline_full", "note_full")),
    **dict.fromkeys(("clamp", "collapse", "short"), ("reasoning_full", False, "headline_clamped", None)),
}

def handle_reasoning_command(self, cmd: str):
    """Handle /reasoning [<level> [--global]|show|hide|full|clamp [lines]] — effort level (session
    scope unless --global) and thinking display toggles (always saved)."""
    from hermes_cli.cli_commands_mixin import (
        _command_arg, _cp, _t, _gt, _save, _split_scope_flags, _retire_agent,
        _accent_line, _dim_line, _scope_outcome)
    from cli import CLI_CONFIG, _parse_reasoning_config
    from agent.reasoning_effort import effort_display_label
    raw = _command_arg(cmd)
    from hermes_cli.codex_runtime_switch import get_current_runtime
    # The live agent's api_mode, else the configured runtime: ``ultra`` is verbatim on the Codex app-server.
    _route = (getattr(self, "provider", None), getattr(self, "model", None),
              getattr(getattr(self, "agent", None), "api_mode", None) or get_current_runtime(CLI_CONFIG))
    if not raw:  # show current state
        rc = self.reasoning_config
        level = (_gt("reasoning.level_default") if rc is None else _gt("reasoning.level_disabled")
                 if rc.get("enabled") is False else effort_display_label(rc.get("effort", "medium"), *_route))
        display_state = _t("reasoning.display_on") if self.show_reasoning else _t("reasoning.display_off")
        full_state = (_t("reasoning.full_state_full") if getattr(self, "reasoning_full", False)
                      else _t("reasoning.full_state_clamped_lines", count=normalize_clamp_lines(
                              getattr(self, "reasoning_clamp_lines", 10))))
        return _cp(_accent_line(_t("reasoning.effort_line", level=level)),
                   _accent_line(_t("reasoning.display_line", display=display_state, full=full_state)),
                   _dim_line(_t("reasoning.usage")))
    arg, explicit_global = _split_scope_flags(raw)
    try:
        clamp_lines = parse_clamp_command(arg)
    except ValueError:
        return _cp(_dim_line(_t("reasoning.invalid_clamp_lines")))
    if clamp_lines is not None:
        self.reasoning_clamp_lines = clamp_lines
        _save("display.reasoning_clamp_lines", clamp_lines)
        arg = "clamp"
    toggle = _REASONING_TOGGLES.get(arg)
    if toggle is not None:  # display show/hide or full/clamp recap toggle
        attr, value, headline_key, note_key = toggle
        setattr(self, attr, value)
        if attr == "show_reasoning" and self.agent:
            self.agent.reasoning_callback = self._current_reasoning_callback()
        _save("display.platforms.cli.show_reasoning" if attr == "show_reasoning" else f"display.{attr}", value)  # T7
        headline = (_t("reasoning.headline_clamped_lines", count=clamp_lines) if clamp_lines is not None
                    else _t(f"reasoning.{headline_key}"))
        _cp(_accent_line(_t("reasoning.display_saved", headline=headline)))
        if note_key:
            _cp(_dim_line(f"  {_t(f'reasoning.{note_key}')}"))
        if attr == "reasoning_full" and value and not self.show_reasoning:
            _cp(_dim_line(f"  {_t('reasoning.display_off_note')}"))
        return
    # Effort level change
    parsed = _parse_reasoning_config(arg)
    if parsed is None:
        return _cp(_dim_line(_t("shared.unknown_argument", arg=arg)),
                   _dim_line(_t("reasoning.valid_levels")), _dim_line(_t("reasoning.valid_display")),
                   _dim_line(_t("reasoning.valid_scope")))
    self.reasoning_config = parsed
    self._reasoning_user_set = True  # the rebuilt gateway agent commits it to the session
    _retire_agent(self)  # Force agent re-init with new reasoning config
    saved = explicit_global and _save("agent.reasoning_effort", arg)
    if saved:
        if not isinstance(CLI_CONFIG.get("agent"), dict):
            CLI_CONFIG["agent"] = {}
        CLI_CONFIG["agent"]["reasoning_effort"] = arg
    _cp(_accent_line(_t("reasoning.effort_set", effort=effort_display_label(arg, *_route),
                        scope=_scope_outcome(explicit_global, saved))))

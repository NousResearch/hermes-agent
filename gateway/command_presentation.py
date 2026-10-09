"""Gateway command presentation and application configuration inputs.

Availability only controls discovery; slash_access owns authorization.
"""
from collections.abc import Iterable

from commands import COMMAND_REGISTRY, is_gateway_available
from utils import is_truthy_value

def resolve_config_gates() -> set[str]:
    """Canonical names of commands whose ``gateway_config_gate`` dotpath is truthy in
    config.yaml (empty set on any error)."""
    gated = [c for c in COMMAND_REGISTRY if c.gateway_config_gate]
    if not gated:
        return set()
    try:
        from hermes_cli.config import cfg_get, read_raw_config
        cfg = read_raw_config()
    except Exception:
        return set()
    return {cmd.name for cmd in gated
            if is_truthy_value(cfg_get(cfg, *cmd.gateway_config_gate.split(".")), default=False)}


def gateway_help_lines(allowed: Iterable[str] | None = None) -> list[str]:
    """Generate gateway help text lines from the registry.

    ``allowed`` (canonical names) restricts the catalog to what the caller may run -- the
    gateway passes a non-admin's slash-access floor + ``user_allowed_commands`` so /help never
    advertises admin-only commands the dispatcher would then refuse.
    """
    overrides = resolve_config_gates()
    allowed_set = None if allowed is None else set(allowed)
    lines: list[str] = []
    for cmd in COMMAND_REGISTRY:
        if not is_gateway_available(cmd, overrides):
            continue
        if allowed_set is not None and cmd.name not in allowed_set:
            continue
        args = f" {cmd.args_hint}" if cmd.args_hint else ""
        # Skip internal aliases like reload_mcp (underscore variant of the name).
        alias_parts = [f"`/{a}`" for a in cmd.aliases
                       if not (a.replace("-", "_") == cmd.name.replace("-", "_") and a != cmd.name)]
        alias_note = f" (alias: {', '.join(alias_parts)})" if alias_parts else ""
        lines.append(f"`/{cmd.name}{args}` -- {cmd.description}{alias_note}")
    return lines

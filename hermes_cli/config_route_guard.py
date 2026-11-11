"""Set-time stale-route guard for provider switches.

``hermes config set <block>.provider X`` must not carry the previous provider's route: the
runtime honours ``base_url``/``api_mode`` for whatever provider the block now names — and in an
``auxiliary.<task>`` block ``base_url`` even overrides ``provider`` (``config_defaults._aux``)
— so the new provider's credential would be posted to the old provider's endpoint (#113719,
#40862 for ``model``; #141832 for the aux blocks this module generalised the guard to)."""

from __future__ import annotations

from typing import Any

from hermes_cli.colors import Colors, color


def stale_route_notice(user_config: dict, dotted_key: str, value: Any) -> str:
    """Drop the route keys a previous provider left in the block *dotted_key* re-points, and
    return the notice line describing what happened ('' when there is nothing to say).

    Guards ``model.provider`` and ``auxiliary.<task>.provider``; runs BEFORE the new provider is
    written so the block's outgoing ``provider`` is still readable. The block is navigated the
    same escape-aware way ``_set_nested`` walks the key and mutated in place, so pops land in the
    caller's config dict. A route that IS the new provider's endpoint is kept; one of unknown
    ownership is kept but warned about."""
    from hermes_cli.config import _locate_nested, _split_key_path

    parts = _split_key_path(dotted_key.strip())
    block: Any = None
    if len(parts) == 2 and parts[0].lower() == "model" and parts[1].lower() == "provider":
        candidate = user_config.get("model")
        block = candidate if isinstance(candidate, dict) else None
    elif len(parts) == 3 and parts[0].lower() == "auxiliary" and parts[2].lower() == "provider":
        loc = _locate_nested(user_config, parts[:-1])
        if loc is not None:
            candidate = loc[1][loc[2]]
            block = candidate if isinstance(candidate, dict) else None
    if not isinstance(block, dict):
        return ""
    old_provider = str(block.get("provider") or "").strip() or "the previous provider"
    if old_provider.lower() == str(value).strip().lower():
        return ""
    from hermes_cli.route_identity import drop_stale_model_route
    popped, unverified = drop_stale_model_route(block, value, user_config)
    prefix = ".".join(parts[:-1])
    if popped:
        return ("  Cleared " + ", ".join(f"{prefix}.{k} ({v})" for k, v in popped.items())
                + f" — that route belonged to {old_provider}, not {value}. {value}'s endpoint resolves "
                f"automatically; set {prefix}.base_url again if you meant a custom endpoint.")
    if unverified:
        return color(
            f"⚠ {prefix}.base_url ({block.get('base_url')}) was set under {old_provider} and "
            f"still applies to {value} — requests go there. If it is not {value}'s endpoint: "
            f"`hermes config unset {prefix}.base_url` (and {prefix}.api_mode).", Colors.YELLOW)
    return ""

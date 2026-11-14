"""Set-time stale-route guard for provider switches.

``hermes config set <block>.provider X`` must not carry the previous provider's route: the
runtime honours ``base_url``/``api_mode`` for whatever provider the block now names — and in an
``auxiliary.<task>`` block ``base_url`` even overrides ``provider`` (``config_defaults._aux``)
— so the new provider's credential would be posted to the old provider's endpoint (#113719,
#40862 for ``model``; #141832 for the aux blocks this module generalised the guard to)."""

from __future__ import annotations

from typing import Any

from hermes_cli.colors import Colors, color
from hermes_cli.route_identity import normalize_route_base_url
from utils import same_base_url_origin

_AUX_KEY_FIELDS = ("api_key", "key_env", "api_key_env")


def _route_of(provider: Any, base_url: Any) -> tuple[str, str]:
    p = str(provider or "").strip().lower()
    return ("" if p == "auto" else p), normalize_route_base_url(str(base_url or "").strip())


def clear_aux_credentials_on_route_change(entry: dict, provider: Any, base_url: Any = "") -> bool:
    """Every writer that re-points an aux/delegation block calls this BEFORE writing the new
    provider/base_url: when the route changes, the block's api_key/key_env/api_key_env/api_mode
    belong to the old endpoint, and the aux resolver would hand them to the new one (#141832).
    Returns True when credentials were cleared."""
    old, new = _route_of(entry.get("provider"), entry.get("base_url")), _route_of(provider, base_url)
    # Bare custom is configured BY its base_url (provider_owns_route); its key binds to the URL's origin
    # (#130410), so a path change on the same origin is the same endpoint's key.
    if old == new or (new[0] == "custom" and new[1] and old[1] and same_base_url_origin(old[1], new[1])):
        return False
    from hermes_cli.config import clear_model_endpoint_credentials
    clear_model_endpoint_credentials(entry)
    return True


_CARRIED_CREDENTIAL_FIELDS = ("api_key", "api", "key_env", "api_key_env", "api_mode")


def _url(value: Any) -> str:
    return str(value or "").strip().rstrip("/")


def drop_carried_route_state(old: Any, new: Any, config: dict) -> None:
    """Guard for whole-document writers (``PUT /api/config``) that merge a form over the disk block.
    The form re-sends or omits the block's base_url/key fields, so a provider change keeps the old
    route's base_url and credentials, and the runtime hands them to the new provider (#141832).

    Mutates *new* in place. A carried base_url the new provider does not own is dropped, as the
    dashboard assignment does. When the endpoint then moves, credential fields still equal to the
    old block's go too; values the form changed are its own and stay.
    An inline api_key set under an unset/auto block with no endpoint is kept: it was entered for
    the provider being chosen now, not for an old endpoint."""
    if not isinstance(old, dict) or not isinstance(new, dict):
        return
    old_route = _route_of(old.get("provider"), old.get("base_url"))
    if old_route == _route_of(new.get("provider"), new.get("base_url")):
        return
    base_url = _url(new.get("base_url"))
    # Same URL identity as _route_of, or a re-spelt old URL escapes the carried check with its key.
    carried_url = bool(base_url) and normalize_route_base_url(base_url) == old_route[1]
    if carried_url and _route_of(new.get("provider"), "")[0] != old_route[0]:
        from hermes_cli.route_identity import provider_owns_route
        if provider_owns_route(new.get("provider"), base_url, config) is not True:
            new.pop("base_url")
            carried_url = False
    if carried_url:
        return  # same endpoint: its credentials stay paired with it
    if _route_of(new.get("provider"), "")[0] == "custom" and same_base_url_origin(old_route[1], base_url):
        return  # bare custom's key binds to its URL's origin (#130410), as clear_aux_credentials_on_route_change
    fields = _CARRIED_CREDENTIAL_FIELDS[2:] if old_route == ("", "") else _CARRIED_CREDENTIAL_FIELDS
    from hermes_cli.config import _expand_env_vars
    for key in fields:
        # GET hands the form a ``${VAR}`` key expanded (aux keys are stored that way), so the old
        # key comes back as its expansion, not as the reference.
        if key in new and new[key] in (old.get(key), _expand_env_vars(old.get(key))):
            new.pop(key)


def _safe(value: Any) -> str:
    """A route value fit for the terminal: userinfo and credential query params masked."""
    from agent.redact import redact_sensitive_text
    return redact_sensitive_text(str(value), force=True, redact_url_credentials=True)


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
    raw_old = str(block.get("provider") or "").strip()
    old_provider = raw_old or "the previous provider"
    if raw_old.lower() == str(value).strip().lower():
        return ""
    from hermes_cli.route_identity import drop_stale_model_route
    popped, unverified = drop_stale_model_route(block, value, user_config)
    prefix = ".".join(parts[:-1])
    dropped_keys: list[str] = []
    # From an unset/auto provider with no route to drop, an inline api_key is one the user just
    # set (``config set ….api_key K`` then ``….provider P``) and belongs to P. A key_env pointer is
    # never kept that way: auto writers can leave one from an earlier route behind.
    no_old_endpoint = not popped and raw_old.lower() in ("", "auto")
    if parts[0].lower() == "auxiliary" and not unverified and not block.get("base_url"):
        # The aux resolver hands a block's api_key/key_env to whatever provider it names, so with
        # the old route gone the old endpoint's key would go to the new provider. Names only:
        # the values are secrets.
        fields = _AUX_KEY_FIELDS[1:] if no_old_endpoint else _AUX_KEY_FIELDS
        dropped_keys = [k for k in fields if block.get(k) not in (None, "")]
        for k in dropped_keys:
            block.pop(k)
    if popped or dropped_keys:
        cleared = [f"{prefix}.{k} ({_safe(v)})" for k, v in popped.items()] + [f"{prefix}.{k}" for k in dropped_keys]
        return ("  Cleared " + ", ".join(cleared)
                + f" — that route belonged to {old_provider}, not {value}. {value}'s endpoint resolves "
                f"automatically; set {prefix}.base_url again if you meant a custom endpoint.")
    if unverified:
        return color(
            f"⚠ {prefix}.base_url ({_safe(block.get('base_url'))}) was set under {old_provider} and "
            f"still applies to {value} — requests go there. If it is not {value}'s endpoint: "
            f"`hermes config unset {prefix}.base_url` (and {prefix}.api_mode).", Colors.YELLOW)
    return ""

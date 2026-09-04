"""Receiving-transport policy for Group Chat controls only."""

from __future__ import annotations

from typing import Any

from gateway.slash_access import (
    SlashAccessPolicy,
    _platform_extra,
    _scope_for_chat_type,
    policy_for_source,
    policy_from_extra,
)


_UNSTAMPED = object()
_DENIED = SlashAccessPolicy(
    enabled=True,
    admin_user_ids=frozenset(),
    user_allowed_commands=frozenset(),
)


def receiving_group_transport(runner: Any, source: Any) -> tuple[Any, Any] | None:
    """Resolve one receiver/config pair, or refuse unavailable provenance.

    A retained transport reference must still identify a registered adapter.
    Unstamped legacy primary sources may use the primary config; an explicit
    secondary runtime without its adapter cannot borrow that fallback.
    """
    try:
        platform = getattr(source, "platform", None)
        profile = str(getattr(source, "profile", None) or "").strip()
        primary_profile = str(
            getattr(runner, "_primary_profile_name", None) or "default"
        )
        profile_maps = getattr(runner, "_profile_adapters", None) or {}
        primary_adapters = getattr(runner, "adapters", None) or {}
        reference = getattr(source, "_transport_adapter_ref", _UNSTAMPED)
        if reference is not _UNSTAMPED:
            registered = getattr(runner, "_transport_owner", None)
            if not callable(reference) or not callable(registered):
                return None
            owner = registered(source)
            if owner is None:
                return None
            adapter = owner[0]
        else:
            secondary = profile not in {"", "default", primary_profile}
            relay = getattr(source, "delivered_via_upstream_relay", False) is True
            if secondary and profile not in profile_maps and not relay:
                return None
            resolve = getattr(runner, "_adapter_for_source", None)
            adapter = resolve(source) if callable(resolve) else None
            if adapter is None and (secondary or profile in profile_maps):
                return None

        config = getattr(adapter, "config", None)
        if config is None:
            if reference is not _UNSTAMPED:
                return None
            if adapter is not None and adapter is not primary_adapters.get(platform):
                return None
            config = (getattr(runner.config, "platforms", None) or {}).get(platform)
        return adapter, config
    except Exception:
        return None


def group_policy_for_source(runner: Any, source: Any) -> SlashAccessPolicy:
    """Keep scope/admin semantics, but obtain them from the actual receiver."""
    resolved = receiving_group_transport(runner, source)
    if resolved is None:
        return _DENIED
    _, config = resolved
    return policy_from_extra(
        _platform_extra(config),
        _scope_for_chat_type(getattr(source, "chat_type", None)),
    )


def policy_for_command(
    runner: Any, source: Any, canonical_cmd: str
) -> SlashAccessPolicy:
    """Route only canonical /group to receiving policy; other commands stay put."""
    if canonical_cmd == "group":
        return group_policy_for_source(runner, source)
    return policy_for_source(runner.config, source)

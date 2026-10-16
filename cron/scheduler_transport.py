"""Cron delivery transport resolution for live, relay, shared-route, and standalone lanes."""

from __future__ import annotations

import contextlib

from cron import scheduler_preflight as _preflight


def _standalone_delivery_available(platform_name: str) -> bool:
    """Whether the platform can deliver outbound without a live gateway adapter."""
    with contextlib.suppress(Exception):
        from hermes_cli.plugins import discover_plugins
        discover_plugins()
        from gateway.platform_registry import platform_registry
        entry = platform_registry.get(platform_name)
        return bool(entry and entry.standalone_sender_fn)
    return False


def resolve_target_transport(job: dict, platform, platform_name: str, target: dict, adapters, config):
    """Resolve one target's live transport or outbound-only standalone configuration."""
    from gateway.delivery import DeliveryTransport, resolve_delivery_transport

    target_adapters = adapters
    transport = None
    if isinstance(adapters, _preflight.SharedRouteAdapters):
        shared = adapters.get(platform, target)
        target_adapters = {platform: shared} if shared is not None else {}
        if shared is not None:
            from dataclasses import replace
            from gateway.config import PlatformConfig
            own = config.platforms.get(platform)
            transport = DeliveryTransport(
                shared, replace(own, enabled=True) if own is not None else PlatformConfig(enabled=True),
                platform)
    if transport is None:
        transport = resolve_delivery_transport(platform, config, target_adapters)
    if transport is not None:
        pconfig = transport.config
        runtime_adapter = transport.adapter
    else:
        from gateway.relay import relay_fronted_platforms
        if platform_name in relay_fronted_platforms():
            return None, (
                f"platform '{platform_name}' is relay-fronted and has no "
                "live gateway transport; start the gateway (its ticker "
                "owns relay-fronted delivery and will fire the job on schedule)"
            )
        pconfig = config.platforms.get(platform)
        runtime_adapter = None

    if transport is not None and (transport.is_relay or pconfig is None):
        if pconfig is None:
            from gateway.config import PlatformConfig
            pconfig = PlatformConfig(enabled=True)
    elif not pconfig:
        return None, f"platform '{platform_name}' not configured/enabled"
    elif not pconfig.enabled:
        # Explicit disable fences inbound/live ownership while preserving an installed standalone
        # sender for outbound-only cron delivery, such as during an ingress handover.
        if transport is not None or not _standalone_delivery_available(platform_name):
            return None, f"platform '{platform_name}' not configured/enabled"
        runtime_adapter = None
    return (transport, pconfig, runtime_adapter, target_adapters), None

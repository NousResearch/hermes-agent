"""Relay/connector support package for the Hermes gateway.

EXPERIMENTAL. This package implements the gateway side of the "Gateway Gateway"
relay design: a generic ``RelayAdapter`` plus the wire-serializable
``CapabilityDescriptor`` the connector hands it at handshake time. The public
API (module names, descriptor field set, transport protocol) MAY CHANGE without
a deprecation cycle until at least two real Class-1 platforms (Discord +
Telegram) have shaken out the schema.

See ``docs/relay-connector-contract.md`` for the formal cross-repo interface.

Registration is OFF by default: ``register_relay_adapter()`` only registers the
``relay`` platform when the relay feature flag is enabled, so existing
single-tenant/direct deployments are completely unaffected (dark-launch posture).
"""

from __future__ import annotations

import os


def relay_enabled() -> bool:
    """Whether the relay adapter should be registered.

    Off by default. Enabled when ``HERMES_GATEWAY_RELAY=1`` (or true/yes/on).
    A config-file gate can be layered on later; the env flag is the minimal
    dark-launch switch so default deployments never register the adapter.
    """
    return os.environ.get("HERMES_GATEWAY_RELAY", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def register_relay_adapter(force: bool = False) -> bool:
    """Register the generic ``relay`` platform via the platform registry.

    No-op unless the relay flag is set (or ``force=True`` for tests). Returns
    True if registration happened. Additive: uses the same registry path as
    plugin adapters, so no core dispatch changes are needed.

    The factory builds a transport-less ``RelayAdapter`` with a placeholder
    descriptor; the real ``CapabilityDescriptor`` is negotiated at handshake
    time via the transport's ``handshake()``. (Wiring the live transport +
    handshake into ``GatewayRunner`` is later-phase work; this task only proves
    the adapter is constructible through the registry behind the flag.)
    """
    if not (force or relay_enabled()):
        return False

    from gateway.platform_registry import PlatformEntry, platform_registry
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor

    def _factory(config):
        placeholder = CapabilityDescriptor(
            contract_version=CONTRACT_VERSION,
            platform="relay",
            label="Relay",
            max_message_length=4096,
            supports_draft_streaming=False,
            supports_edit=True,
            supports_threads=False,
            markdown_dialect="plain",
            len_unit="chars",
        )
        return RelayAdapter(config, placeholder)

    platform_registry.register(
        PlatformEntry(
            name="relay",
            label="Relay",
            adapter_factory=_factory,
            check_fn=lambda: True,
            source="builtin",
            emoji="\U0001f50c",
        )
    )
    return True

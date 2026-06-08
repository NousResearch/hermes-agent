"""Relay/connector support package for the Hermes gateway.

EXPERIMENTAL. This package implements the gateway side of the "Gateway Gateway"
relay design: a generic ``RelayAdapter`` plus the wire-serializable
``CapabilityDescriptor`` the connector hands it at handshake time. The public
API (module names, descriptor field set, transport protocol) MAY CHANGE without
a deprecation cycle until at least two real Class-1 platforms (Discord +
Telegram) have shaken out the schema.

See ``docs/relay-connector-contract.md`` for the formal cross-repo interface.
"""

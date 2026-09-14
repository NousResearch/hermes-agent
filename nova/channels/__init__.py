"""NOVA Channel Layer — the customer's channels, connected to their AI workforce.

The runtime underneath already has a complete messaging platform system: adapters,
a registry, channel→agent routing, default-deny access control, webhook signature
verification and a durable outbound delivery ledger. ``docs/NOVA_CHANNEL_AUDIT.md`` records
what was found and where.

**So this package writes no adapter, no transport and no protocol code.** It owns the four
things the runtime deliberately does not: the tenant boundary, the per-connection grant that
says which agents a channel may reach, the customer's vocabulary, and the audit record of
who connected what. Everything else is compiled into configuration for a system that works.
"""

from nova.channels.providers import (
    PROVIDERS,
    PROVIDERS_BY_ID,
    Capability,
    Provider,
    Transport,
    Verification,
    get_provider,
)
from nova.channels.derive import DerivedAgent, derive_specs, plan_derivations, route_target
from nova.channels.spec import (
    CHANNELS_FILE,
    ChannelApproval,
    ChannelRoute,
    ChannelSpec,
    check_agents_exist,
    load_channels,
    parse_channels,
)

__all__ = [
    "CHANNELS_FILE",
    "ChannelApproval",
    "DerivedAgent",
    "derive_specs",
    "plan_derivations",
    "route_target",
    "PROVIDERS",
    "PROVIDERS_BY_ID",
    "Capability",
    "ChannelRoute",
    "ChannelSpec",
    "Provider",
    "Transport",
    "Verification",
    "check_agents_exist",
    "get_provider",
    "load_channels",
    "parse_channels",
]

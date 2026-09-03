"""Shared network constants.

The 100.64.0.0/10 block (RFC 6598, "Shared Address Space") is used by
carrier-grade NAT, Tailscale, and WireGuard. ``ipaddress.is_private``
does not cover it, so callers that need to recognise these addresses
keep an explicit constant.
"""

import ipaddress

CGNAT_NETWORK = ipaddress.IPv4Network("100.64.0.0/10")

# Privacy

This tree must not ship:

- Real WireGuard, WAN, or residential IPs
- Hostnames, mesh diagrams with live addresses
- Cookie databases, profile paths, or session stores
- Live site URLs, group IDs, or people's names
- Comment-extraction or research-brief workflows

`BIND_IP` and `PEER_WG_IP` are filled at install time in `~/.hermes/takeover/env` (not committed). Default hold URL is `https://example.com/`.

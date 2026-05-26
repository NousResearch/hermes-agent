---
name: pfsense-monitoring
description: Query pfSense firewalls/gateways via REST API using curl — read-only monitoring for interfaces, gateways, services, firewall rules, DHCP, ARP, DNS, VPNs, routing, logs, and CARP status.
version: 1.0.0
author: Mert Özkan
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [pfsense, firewall, gateway, network, monitoring, devops, infrastructure]
    category: devops
    requires_toolsets: [terminal]
---

# pfSense Monitoring

Query a pfSense firewall/gateway using its REST API (pfSense-pkg-RESTAPI by pfrest). Uses `curl` via the terminal tool — no custom tool needed.

## When to Use

- User asks about pfSense firewall/gateway status
- Check interfaces, gateways, services, or firewall rules
- List DHCP leases, ARP table, or DNS resolver status
- Check VPN connections (OpenVPN, IPsec)
- View routing table or CARP/HA failover status
- Read system, firewall, or auth logs

## Prerequisites

- **pfSense REST API package** installed (pfrest/pfSense-pkg-RESTAPI)
- **PFSENSE_URL** and **PFSENSE_API_KEY** environment variables set
- `curl` available in the terminal

### Installing REST API on pfSense

SSH into pfSense as root and run:
```bash
fetch -q "https://github.com/pfrest/pfSense-pkg-RESTAPI/releases/download/v2.8.0/pfSense-2.8.1-pkg-RESTAPI.pkg"
pkg add -f pfSense-2.8.1-pkg-RESTAPI.pkg
```

After install: pfSense Web UI → Services → REST API → Settings → set a different port (e.g. 8443) if nginx can't bind 443.

## Env Vars Required

| Variable | Default | Description |
|----------|---------|-------------|
| `PFSENSE_URL` | `https://10.0.0.1` | pfSense hostname/IP |
| `PFSENSE_API_KEY` | — | API token from Services → REST API → API Tokens |

## Available Queries

Each query is a `terminal()` call with `curl`. Always use `-k` (self-signed certs) and `-s` (silent). Pipe through `jq` for pretty output when available.

### Summary
```bash
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/system"
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/interfaces"
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/gateways"
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/services"
```

### Interfaces
```bash
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/interfaces" | jq '.data[] | {name: .if, ip: .ipaddr, status: .status, in: .inpkts, out: .outpkts}'
```

### Gateways
```bash
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/gateways" | jq '.data[] | {name: .name, ip: .gateway, status: .status, delay: .delay, loss: .loss}'
```

### Services
```bash
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/services" | jq '.data[] | {name: .name, running: .running}'
```

### Firewall Rules
```bash
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/firewall/rules" | jq '.data[] | {number: .number, type: .type, interface: .interface, protocol: .protocol, source: .source, destination: .destination, action: .action}'
```

### DHCP Leases
```bash
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/dhcp_server/leases" | jq '.data[] | {ip: .ip, mac: .mac, hostname: .hostname, online: .online}'
```

### ARP Table
```bash
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/diagnostics/arp_table" | jq '.data[] | {ip: .ip_address, mac: .mac_address, interface: .interface}'
```

### DNS Resolver
```bash
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/services/dns_resolver/settings" | jq '.data'
```

### OpenVPN
```bash
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/openvpn/servers"
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/openvpn/clients"
```

### IPsec
```bash
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/ipsec/sas" | jq '.data[] | {name: .name, status: .status, local: .local, remote: .remote}'
```

### Routes
```bash
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/routing/gateways"
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/routing/static_route"
```

### CARP Status
```bash
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/carp"
```

### Logs
```bash
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/logs/system"
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/logs/firewall"
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/logs/dhcp"
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/logs/auth"
```

## Pitfalls

- **Auth header**: v2 API uses `X-API-Key` header. Wrong header (e.g. `Authorization: Bearer`) produces HTTP 401.
- **IP ban**: Repeated failed auth → pfSense bans the source IP via pfTables (native firewall state table). Symptoms: ping works, SSH/HTTPS timeout. Fix: pfSense Web UI → Diagnostics → pfTables → delete your IP.
- **Self-signed cert**: Always use `-k` with curl (pfSense uses self-signed by default).
- **API fallback**: v2 is primary; if you get HTML/404, try `/api/v1/` prefix instead.
- **Port conflict**: REST API nginx may conflict with webConfigurator (lighttpd) on ports 80/443. Set a different port in REST API settings.
- **No REST API package**: If all calls return connection refused, the package isn't installed.
- **SSH timeout**: pfSense can be slow — use `ConnectTimeout=30` for SSH commands.

## Sister Skills

- `truenas-monitoring` — TrueNAS Scale monitoring via curl + REST API
- `unraid-monitoring` — UnRAID monitoring via curl + GraphQL API
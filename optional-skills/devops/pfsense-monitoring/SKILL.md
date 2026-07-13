---
name: pfsense-monitoring
description: Monitor pfSense firewalls via REST API.
version: 1.1.0
author: Mert Özkan (@47Hunter47)
license: MIT
platforms: [linux, macos]
required_environment_variables:
  - name: PFSENSE_API_KEY
    prompt: pfSense REST API token (Services → REST API → API Tokens)
    help: "https://github.com/pfrest/pfSense-pkg-RESTAPI — generate in pfSense Web UI → Services → REST API → API Tokens"
  - name: PFSENSE_URL
    prompt: pfSense base URL (e.g., https://10.0.0.1:8443)
    help: "Base HTTPS URL of the pfSense web interface with REST API enabled."
metadata:
  hermes:
    tags: [pfsense, firewall, gateway, network, monitoring, devops]
    related_skills: [truenas-monitoring, unraid-monitoring]
    config:
      PFSENSE_URL: "https://10.0.0.1"
---

# pfSense Monitoring Skill

Read-only monitoring of pfSense firewalls/gateways via the [pfSense-pkg-RESTAPI](https://github.com/pfrest/pfSense-pkg-RESTAPI) package. Interfaces, gateways, services, firewall rules, DHCP, ARP, DNS, VPNs, routing, logs, and CARP status — all via `curl` + `jq`.

Does not modify firewall rules, change settings, or perform write operations.

## When to Use

- User asks about pfSense firewall/gateway status
- Check interfaces, gateways, services, or firewall rules
- List DHCP leases, ARP table, or DNS resolver status
- Check VPN connections (OpenVPN, IPsec)
- View routing table or CARP/HA failover status
- Read system, firewall, or auth logs

## Prerequisites

1. **pfSense REST API package** installed (pfrest/pfSense-pkg-RESTAPI):
   ```bash
   # SSH into pfSense as root
   fetch -q "https://github.com/pfrest/pfSense-pkg-RESTAPI/releases/download/v2.8.0/pfSense-2.8.1-pkg-RESTAPI.pkg"
   pkg add -f pfSense-2.8.1-pkg-RESTAPI.pkg
   ```
   Then: Web UI → Services → REST API → Settings → set port (e.g. 8443) if nginx conflicts with 443.

2. **Environment variables** in `${HERMES_HOME:-~/.hermes}/.env`:
   ```
   PFSENSE_URL=https://<firewall-ip>[:port]
   PFSENSE_API_KEY=<your-token>
   ```

3. `curl` and `jq` available in terminal.

## Quick Reference

```bash
# Status summary
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/system"
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/interfaces"
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/gateways"
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/services"

# Interfaces detail
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/interfaces" \
  | jq '.data[] | {name: .if, ip: .ipaddr, status: .status, in: .inpkts, out: .outpkts}'

# Gateways
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/gateways" \
  | jq '.data[] | {name: .name, ip: .gateway, status: .status, delay: .delay, loss: .loss}'

# Services
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/services" \
  | jq '.data[] | {name: .name, running: .running}'

# Firewall rules
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/firewall/rules" \
  | jq '.data[] | {number: .number, type: .type, interface: .interface, protocol: .protocol, source: .source, destination: .destination, action: .action}'

# DHCP leases
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/dhcp_server/leases" \
  | jq '.data[] | {ip: .ip, mac: .mac, hostname: .hostname, online: .online}'

# ARP table
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/diagnostics/arp_table" \
  | jq '.data[] | {ip: .ip_address, mac: .mac_address, interface: .interface}'

# DNS resolver settings
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/services/dns_resolver/settings" \
  | jq '.data'

# OpenVPN
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/openvpn/servers"
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/openvpn/clients"

# IPsec
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/ipsec/sas" \
  | jq '.data[] | {name: .name, status: .status, local: .local, remote: .remote}'

# Routing
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/routing/gateways"
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/routing/static_route"

# CARP/HA
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/carp"

# Logs
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/logs/system"
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/logs/firewall"
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/logs/dhcp"
curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/logs/auth"
```

## Procedure

1. Verify env vars are set: `echo $PFSENSE_URL $PFSENSE_API_KEY`
2. Health check: `curl -sk -H "X-API-Key: $PFSENSE_API_KEY" "$PFSENSE_URL/api/v2/status/system"`
3. Run the relevant query from Quick Reference above
4. If response is HTML or 404, try `/api/v1/` prefix as fallback

## Pitfalls

1. **Auth header**: v2 API uses `X-API-Key` header. Using `Authorization: ***` returns HTTP 401.
2. **IP ban**: Repeated failed auth → pfSense bans source IP via pfTables. Symptoms: ping works, HTTPS times out. Fix: Web UI → Diagnostics → pfTables → delete your IP.
3. **Self-signed cert**: Always use `-k` with curl (pfSense uses self-signed by default).
4. **API version**: v2 is primary; if you get HTML/404, try `/api/v1/` prefix.
5. **Port conflict**: REST API nginx may conflict with webConfigurator (lighttpd) on 80/443. Set a different port in REST API settings.
6. **Package not installed**: All calls return connection refused = package missing.
7. **Slow SSH**: pfSense can be slow — use `ConnectTimeout=30` for SSH commands.

## Verification

- [ ] `PFSENSE_URL` and `PFSENSE_API_KEY` are set in `.env`
- [ ] Health check returns valid JSON (not HTML)
- [ ] Auth header is `X-API-Key` (not `Authorization`)
- [ ] Self-signed cert flag `-k` is used on all curl calls

## Related Skills

- `truenas-monitoring` — TrueNAS Scale via REST API
- `unraid-monitoring` — UnRAID via GraphQL API

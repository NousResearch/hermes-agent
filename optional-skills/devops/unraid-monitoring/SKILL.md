---
name: unraid-monitoring
description: Query UnRAID servers via GraphQL API using curl — read-only monitoring for system info, disks, Docker containers, and alerts.
version: 1.0.0
author: Mert Özkan
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [unraid, nas, docker, storage, monitoring, devops, infrastructure]
    category: devops
    requires_toolsets: [terminal]
---

# UnRAID Monitoring

Query an UnRAID server using its GraphQL API. Uses `curl` via the terminal tool — no custom tool needed.

## When to Use

- User asks about UnRAID system status
- Check system info (hostname, OS, kernel, CPU, uptime)
- List physical disks with serial, size, temperature, SMART status
- List Docker containers with names, images, and status
- View active notifications and warnings
- Run a network security audit on UnRAID Docker exposure

## Prerequisites

- **UNRAID_URL** and **UNRAID_API_KEY** environment variables set
- `curl` and `jq` available in the terminal
- API key from UnRAID Web UI → Settings → API Keys (with READ_ANY data permissions)
- Note: UnRAID GraphQL is at `/graphql` endpoint

## Env Vars Required

| Variable | Default | Description |
|----------|---------|-------------|
| `UNRAID_URL` | `https://unraid.local` | UnRAID hostname/IP |
| `UNRAID_API_KEY` | — | API key with READ_ANY permissions |

## Available Queries

All queries use `terminal()` with curl posting GraphQL JSON. Always use `-k` (self-signed certs).

Single reusable curl wrapper:
```bash
UNRAID_QUERY() { curl -sk -H "Authorization: Bearer $UNRAID_API_KEY" -H "Content-Type: application/json" -d "{\"query\":\"$1\"}" "$UNRAID_URL/graphql"; }
```

### Summary
```bash
UNRAID_QUERY='{ info { cpu { model cores threads } os { hostname distro kernel uptime } } disks { name size type serialNum temperature smartStatus } docker { containers { names image status } } notifications { warningsAndAlerts { id } } }'
curl -sk -H "Authorization: Bearer $UNRAID_API_KEY" -H "Content-Type: application/json" \
  -d "{\"query\":\"$UNRAID_QUERY\"}" "$UNRAID_URL/graphql" | jq '.data'
```

### System Info
```bash
curl -sk -H "Authorization: Bearer $UNRAID_API_KEY" -H "Content-Type: application/json" \
  -d '{"query":"{ info { cpu { model cores threads } os { hostname distro kernel uptime } } }"}' \
  "$UNRAID_URL/graphql" | jq '.data.info'
```

### Disks
```bash
curl -sk -H "Authorization: Bearer $UNRAID_API_KEY" -H "Content-Type: application/json" \
  -d '{"query":"{ disks { name size type serialNum temperature smartStatus } }"}' \
  "$UNRAID_URL/graphql" | jq '.data.disks'
```

### Docker Containers
```bash
curl -sk -H "Authorization: Bearer $UNRAID_API_KEY" -H "Content-Type: application/json" \
  -d '{"query":"{ docker { containers { names image status } } }"}' \
  "$UNRAID_URL/graphql" | jq '.data.docker.containers'
```

### Alerts
```bash
curl -sk -H "Authorization: Bearer $UNRAID_API_KEY" -H "Content-Type: application/json" \
  -d '{"query":"{ notifications { warningsAndAlerts { id } } }"}' \
  "$UNRAID_URL/graphql" | jq '.data.notifications.warningsAndAlerts'
```

## Security Audit

When auditing UnRAID, check Docker containers for high-risk services:

1. **Risky containers**: Flag Apache Guacamole, Open WebUI, or any `:latest` tag containers.
2. **Host networking**: Containers using `--net=host` bypass UnRAID firewall entirely.
3. **Pinned versions**: Prefer pinned versions (e.g. `linuxserver/plex:1.41.0`) over `:latest`.

## Pitfalls

- **Auth**: UnRAID uses `Authorization: Bearer` header (not API keys in URL or body).
- **Self-signed cert**: Always use `-k` with curl.
- **GraphQL errors**: UnRAID returns HTTP 200 even on errors — always check the `errors` field in response JSON.
- **Missing permissions**: "Invalid API key format" = key lacks READ_ANY data permissions.
- **Disk size quirk**: Some disks may report incorrect sizes (e.g. 2TB as "1.8 PB") — API display issue.
- **Memory**: Not accessible via GraphQL API.
- **The UNRAID_QUERY shell function** is useful for multi-field queries — define it in terminal before use.

## Sister Skills

- `truenas-monitoring` — TrueNAS Scale monitoring via curl + REST API
- `pfsense-monitoring` — pfSense monitoring via curl + REST API
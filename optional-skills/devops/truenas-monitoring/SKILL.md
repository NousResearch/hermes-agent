---
name: truenas-monitoring
description: Query TrueNAS Scale systems via REST API using curl — read-only monitoring for pools, disks, shares, alerts, services, and system info.
version: 1.0.0
author: Mert Özkan
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [truenas, nas, storage, zfs, monitoring, devops, infrastructure]
    category: devops
    requires_toolsets: [terminal]
---

# TrueNAS Monitoring

Query a TrueNAS Scale system using its REST API v2.0. Uses `curl` via the terminal tool — no custom tool needed.

## When to Use

- User asks about TrueNAS system status
- Check ZFS pool health, capacity, vdev topology
- List physical disks with serial, model, size
- Check SMB/NFS shares
- View active alerts and warnings
- Check service status (SMB, NFS, SSH, etc.)
- Run a network security audit on TrueNAS

## Prerequisites

- **TRUENAS_URL** and **TRUENAS_API_KEY** environment variables set
- `curl` and `jq` available in the terminal
- API key from TrueNAS Web UI → Settings → API Keys
- Note: REST API is deprecated in TrueNAS v25.10+ and will be removed in v26.04

## Env Vars Required

| Variable | Default | Description |
|----------|---------|-------------|
| `TRUENAS_URL` | `https://truenas.local` | TrueNAS hostname/IP |
| `TRUENAS_API_KEY` | — | API key from Settings → API Keys |

## Available Queries

All queries use `terminal()` with curl. Always use `-k` (self-signed certs) and pipe to `jq` for readable output.

### Summary (multiple calls)
```bash
curl -sk -H "Authorization: Bearer $TRUENAS_API_KEY" "$TRUENAS_URL/api/v2.0/system/info"
curl -sk -H "Authorization: Bearer $TRUENAS_API_KEY" "$TRUENAS_URL/api/v2.0/pool"
curl -sk -H "Authorization: Bearer $TRUENAS_API_KEY" "$TRUENAS_URL/api/v2.0/disk"
curl -sk -H "Authorization: Bearer $TRUENAS_API_KEY" "$TRUENAS_URL/api/v2.0/alert/list"
```

### Pools
```bash
curl -sk -H "Authorization: Bearer $TRUENAS_API_KEY" "$TRUENAS_URL/api/v2.0/pool" | jq '.[] | {name: .name, status: .status, healthy: .healthy, size: .size, used: .used, available: .available}'
```

### Disks
```bash
curl -sk -H "Authorization: Bearer $TRUENAS_API_KEY" "$TRUENAS_URL/api/v2.0/disk" | jq '.[] | {name: .name, serial: .serial, model: .model, size: .size, type: .type}'
```

### Shares (SMB + NFS)
```bash
curl -sk -H "Authorization: Bearer $TRUENAS_API_KEY" "$TRUENAS_URL/api/v2.0/sharing/smb" | jq '.[] | {name: .name, path: .path, comment: .comment, enabled: .enabled}'
curl -sk -H "Authorization: Bearer $TRUENAS_API_KEY" "$TRUENAS_URL/api/v2.0/sharing/nfs" | jq '.[] | {name: .name, path: .path, networks: .networks, enabled: .enabled}'
```

### Alerts
```bash
curl -sk -H "Authorization: Bearer $TRUENAS_API_KEY" "$TRUENAS_URL/api/v2.0/alert/list" | jq '.[] | {level: .level, message: .message, dismissed: .dismissed, node: .node}'
```

### Services
```bash
curl -sk -H "Authorization: Bearer $TRUENAS_API_KEY" "$TRUENAS_URL/api/v2.0/service" | jq '.[] | {name: .name, state: .state, pids: .pids, enable: .enable}'
```

## Security Audit

When auditing TrueNAS, always check:

1. **NFS share restrictions**: `curl -sk -H "Authorization: Bearer $TRUENAS_API_KEY" "$TRUENAS_URL/api/v2.0/sharing/nfs" | jq '.[] | {name: .name, networks: .networks}'` — empty networks = critical (any host can mount).

2. **SSH password auth**: If SSH service is running, check `curl -sk -H "Authorization: Bearer $TRUENAS_API_KEY" "$TRUENAS_URL/api/v2.0/ssh" | jq '{password_auth: .passwordauth}'` — `true` = high risk.

3. **Unnecessary services**: Flag NFS, iSCSI, FTP, SNMP if running but not needed.

## Pitfalls

- **Auth**: TrueNAS uses `Authorization: Bearer` header (not X-API-Key).
- **Self-signed cert**: Always use `-k` with curl.
- **Deprecated API**: REST API is deprecated since TrueNAS v25.10; will be removed in v26.04. After that, use WebSocket JSON-RPC 2.0.
- **HTTP 401**: Invalid or expired API key — regenerate in TrueNAS Web UI.
- **Alert noise**: The deprecated REST API triggers a warning alert after 24h — cosmetic only.
- **REST API deprecation alert**: If you see `RESTAPIUsage` alerts, they're from this query itself — harmless.

## Sister Skills

- `unraid-monitoring` — UnRAID monitoring via curl + GraphQL API
- `pfsense-monitoring` — pfSense monitoring via curl + REST API
---
name: truenas-monitoring
description: Monitor TrueNAS Scale systems via REST API.
version: 1.1.0
author: Mert Özkan (@47Hunter47)
license: MIT
platforms: [linux, macos]
required_environment_variables:
  - name: TRUENAS_API_KEY
    prompt: TrueNAS API key (Settings → API Keys)
    help: "TrueNAS Web UI → Settings → API Keys — create a key with read access."
  - name: TRUENAS_URL
    prompt: TrueNAS base URL (e.g., https://truenas.local)
    help: "Base HTTPS URL of the TrueNAS web interface."
metadata:
  hermes:
    tags: [truenas, nas, storage, zfs, monitoring, devops]
    related_skills: [pfsense-monitoring, unraid-monitoring]
    config:
      TRUENAS_URL: "https://truenas.local"
---

# TrueNAS Monitoring Skill

Read-only monitoring of TrueNAS Scale systems via REST API v2.0. ZFS pool health, disk inventory, SMB/NFS shares, alerts, services, and system info — all via `curl` + `jq`.

Does not modify pool configuration, create shares, or change system settings.

> **Note:** REST API is deprecated in TrueNAS v25.10+ and will be removed in v26.04. After that, use WebSocket JSON-RPC 2.0.

## When to Use

- User asks about TrueNAS system status
- Check ZFS pool health, capacity, vdev topology
- List physical disks with serial, model, size
- Check SMB/NFS shares
- View active alerts and warnings
- Check service status (SMB, NFS, SSH, etc.)
- Run a network security audit on TrueNAS

## Prerequisites

1. **Environment variables** in `${HERMES_HOME:-~/.hermes}/.env`:
   ```
   TRUENAS_URL=https://<truenas-ip-or-hostname>
   TRUENAS_API_KEY=<your-api-key>
   ```
   API key from TrueNAS Web UI → Settings → API Keys.

2. `curl` and `jq` available in terminal.

## Quick Reference

```bash
# Status summary
curl -sk -H "Authorization: Bearer $TRUENAS_API_KEY" "$TRUENAS_URL/api/v2.0/system/info"
curl -sk -H "Authorization: Bearer $TRUENAS_API_KEY" "$TRUENAS_URL/api/v2.0/pool"
curl -sk -H "Authorization: Bearer $TRUENAS_API_KEY" "$TRUENAS_URL/api/v2.0/disk"
curl -sk -H "Authorization: Bearer $TRUENAS_API_KEY" "$TRUENAS_URL/api/v2.0/alert/list"

# Pools
curl -sk -H "Authorization: Bearer $TRUENAS_API_KEY" "$TRUENAS_URL/api/v2.0/pool" \
  | jq '.[] | {name: .name, status: .status, healthy: .healthy, size: .size, used: .used, available: .available}'

# Disks
curl -sk -H "Authorization: Bearer $TRUENAS_API_KEY" "$TRUENAS_URL/api/v2.0/disk" \
  | jq '.[] | {name: .name, serial: .serial, model: .model, size: .size, type: .type}'

# SMB shares
curl -sk -H "Authorization: Bearer $TRUENAS_API_KEY" "$TRUENAS_URL/api/v2.0/sharing/smb" \
  | jq '.[] | {name: .name, path: .path, comment: .comment, enabled: .enabled}'

# NFS shares
curl -sk -H "Authorization: Bearer $TRUENAS_API_KEY" "$TRUENAS_URL/api/v2.0/sharing/nfs" \
  | jq '.[] | {name: .name, path: .path, networks: .networks, enabled: .enabled}'

# Alerts
curl -sk -H "Authorization: Bearer $TRUENAS_API_KEY" "$TRUENAS_URL/api/v2.0/alert/list" \
  | jq '.[] | {level: .level, message: .message, dismissed: .dismissed, node: .node}'

# Services
curl -sk -H "Authorization: Bearer $TRUENAS_API_KEY" "$TRUENAS_URL/api/v2.0/service" \
  | jq '.[] | {name: .name, state: .state, pids: .pids, enable: .enable}'
```

## Procedure

1. Verify env vars: `echo $TRUENAS_URL $TRUENAS_API_KEY`
2. Health check: `curl -sk -H "Authorization: Bearer $TRUENAS_API_KEY" "$TRUENAS_URL/api/v2.0/system/info"`
3. Run the relevant query from Quick Reference above

## Security Audit

When auditing TrueNAS, always check:

1. **NFS share restrictions**:
   ```bash
   curl -sk -H "Authorization: Bearer $TRUENAS_API_KEY" "$TRUENAS_URL/api/v2.0/sharing/nfs" \
     | jq '.[] | {name: .name, networks: .networks}'
   ```
   Empty `networks` = critical (any host can mount).

2. **SSH password auth**:
   ```bash
   curl -sk -H "Authorization: Bearer $TRUENAS_API_KEY" "$TRUENAS_URL/api/v2.0/ssh" \
     | jq '{password_auth: .passwordauth}'
   ```
   `true` = high risk.

3. **Unnecessary services**: Flag NFS, iSCSI, FTP, SNMP if running but not needed.

## Pitfalls

1. **Auth header**: TrueNAS uses `Authorization: Bearer ***` (not `X-API-Key`).
2. **Self-signed cert**: Always use `-k` with curl.
3. **Deprecated API**: REST API deprecated since v25.10; removed in v26.04. Plan migration to WebSocket JSON-RPC 2.0.
4. **HTTP 401**: Invalid or expired API key — regenerate in Web UI.
5. **Alert noise**: Deprecated REST API triggers a warning alert after 24h — cosmetic only.
6. **RESTAPIUsage alerts**: From this query itself — harmless.

## Verification

- [ ] `TRUENAS_URL` and `TRUENAS_API_KEY` are set in `.env`
- [ ] Health check returns valid JSON (not HTML)
- [ ] Auth header is `Authorization: Bearer` (not `X-API-Key`)
- [ ] Self-signed cert flag `-k` is used on all curl calls

## Related Skills

- `pfsense-monitoring` — pfSense via REST API
- `unraid-monitoring` — UnRAID via GraphQL API

---
name: unraid-monitoring
description: Monitor UnRAID servers via GraphQL API.
version: 1.1.0
author: Mert Özkan (@47Hunter47)
license: MIT
platforms: [linux, macos]
required_environment_variables:
  - name: UNRAID_API_KEY
    prompt: UnRAID API key (Settings → API Keys, READ_ANY permissions)
    help: "UnRAID Web UI → Settings → API Keys — create a key with READ_ANY data permissions."
  - name: UNRAID_URL
    prompt: UnRAID base URL (e.g., https://unraid.local)
    help: "Base HTTPS URL of the UnRAID web interface."
metadata:
  hermes:
    tags: [unraid, nas, docker, storage, monitoring, devops]
    related_skills: [pfsense-monitoring, truenas-monitoring]
    config:
      UNRAID_URL: "https://unraid.local"
---

# UnRAID Monitoring Skill

Read-only monitoring of UnRAID servers via GraphQL API (`/graphql`). System info, disk inventory, Docker containers, and alerts — all via `curl` + `jq`.

Does not start/stop containers, modify disks, or change system settings.

## When to Use

- User asks about UnRAID system status
- Check system info (hostname, OS, kernel, CPU, uptime)
- List physical disks with serial, size, temperature, SMART status
- List Docker containers with names, images, and status
- View active notifications and warnings
- Run a network security audit on UnRAID Docker exposure

## Prerequisites

1. **Environment variables** in `${HERMES_HOME:-~/.hermes}/.env`:
   ```
   UNRAID_URL=https://<unraid-ip-or-hostname>
   UNRAID_API_KEY=<your-api-key>
   ```
   API key from UnRAID Web UI → Settings → API Keys (must have `READ_ANY` data permissions).

2. `curl` and `jq` available in terminal.

3. GraphQL endpoint: `$UNRAID_URL/graphql`

## Quick Reference

Reusable curl wrapper:
```bash
UNRAID_QUERY() {
  curl -sk -H "Authorization: Bearer $UNRAID_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"$1\"}" "$UNRAID_URL/graphql"
}
```

### System info
```bash
UNRAID_QUERY '{ info { cpu { model cores threads } os { hostname distro kernel uptime } } }' \
  | jq '.data.info'
```

### Disks
```bash
UNRAID_QUERY '{ disks { name size type serialNum temperature smartStatus } }' \
  | jq '.data.disks'
```

### Docker containers
```bash
UNRAID_QUERY '{ docker { containers { names image status } } }' \
  | jq '.data.docker.containers'
```

### Alerts
```bash
UNRAID_QUERY '{ notifications { warningsAndAlerts { id } } }' \
  | jq '.data.notifications.warningsAndAlerts'
```

### Full summary
```bash
UNRAID_QUERY '{ info { cpu { model cores threads } os { hostname distro kernel uptime } } disks { name size type serialNum temperature smartStatus } docker { containers { names image status } } notifications { warningsAndAlerts { id } } }' \
  | jq '.data'
```

## Procedure

1. Verify env vars: `echo $UNRAID_URL $UNRAID_API_KEY`
2. Define `UNRAID_QUERY` shell function in terminal
3. Run the relevant query from Quick Reference above
4. Always check the `errors` field in response JSON (GraphQL returns 200 even on errors)

## Security Audit

When auditing UnRAID, check Docker containers for high-risk services:

1. **Risky containers**: Flag Apache Guacamole, Open WebUI, or any `:latest` tag.
2. **Host networking**: Containers using `--net=host` bypass UnRAID firewall entirely.
3. **Pinned versions**: Prefer `linuxserver/plex:1.41.0` over `:latest`.

## Pitfalls

1. **Auth header**: UnRAID uses `Authorization: Bearer ***` (not `X-API-Key` or URL params).
2. **Self-signed cert**: Always use `-k` with curl.
3. **GraphQL errors**: UnRAID returns HTTP 200 even on errors — always check the `errors` field.
4. **Missing permissions**: "Invalid API key format" = key lacks `READ_ANY` data permissions.
5. **Disk size quirk**: Some disks report incorrect sizes (e.g., 2TB as "1.8 PB") — API display issue.
6. **Memory**: Not accessible via GraphQL API.

## Verification

- [ ] `UNRAID_URL` and `UNRAID_API_KEY` are set in `.env`
- [ ] API key has `READ_ANY` data permissions
- [ ] Query returns valid JSON with no `errors` field
- [ ] Auth header is `Authorization: Bearer` (not `X-API-Key`)
- [ ] Self-signed cert flag `-k` is used on all curl calls

## Related Skills

- `pfsense-monitoring` — pfSense via REST API
- `truenas-monitoring` — TrueNAS Scale via REST API

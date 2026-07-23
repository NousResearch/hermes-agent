---
name: whatsapp-bridge-config
description: >
  Configure and troubleshoot the Hermes WhatsApp bridge. Covers the dual-layer
  filtering system (bridge allowlist + gateway chat policy), finding group chat IDs,
  and common misconfiguration pitfalls. Use when setting up, debugging, or modifying
  WhatsApp bridge behavior.
tags: [whatsapp, gateway, bridge, configuration, troubleshooting]
---

# WhatsApp Bridge Configuration

The WhatsApp integration has two independent filtering layers. Misunderstanding which layer controls what is the #1 source of configuration errors.

## Architecture

```
WhatsApp message
  → Bridge (Node.js, port 3000)
    → Layer 1: WHATSAPP_ALLOWED_USERS env var (sender allowlist)
      → if rejected: logged as "allowlist_mismatch", silently dropped
      → if accepted:
  → Gateway (Python, systemd)
    → Layer 2: allowed_chats / dm_policy / group_policy (chat-level filtering)
      → if rejected: silently dropped
      → if accepted: forwarded to agent
```

## Layer 1: Bridge Sender Allowlist (env var)

**Source:** `~/.hermes/.env` — `WHATSAPP_ALLOWED_USERS=...`

This is an ENVIRONMENT VARIABLE, not a config.yaml field. The bridge reads `process.env.WHATSAPP_ALLOWED_USERS` at startup.

| Value | Behavior |
|-------|----------|
| (empty/unset) | NO ONE allowed — all messages rejected |
| `*` | Everyone allowed — gateway does the filtering |
| `+1234567890` | Only that phone number's DMs pass |

**PITFALL:** Setting `allowed_users` in config.yaml does NOT affect the bridge. The bridge reads from `.env` only. Many users (and the config.yaml schema) suggest otherwise — this is misleading.

**PITFALL:** Phone number format matters. The bridge normalizes by stripping `+`, so `+1234567890` becomes `1234567890`. WhatsApp LIDs (Linked Identity IDs like `129102573002791@lid`) are mapped to phone numbers via `lid-mapping-*.json` files in the session directory. If the mapping file doesn't exist for a sender, the allowlist check fails silently.

## Layer 2: Gateway Chat Policy (config.yaml)

**Source:** `~/.hermes/config.yaml` under `whatsapp:`

```yaml
whatsapp:
  allowed_chats: '<your-group-id>@g.us'  # Restrict to specific group(s)
  # OR for the adapter-level config (more granular):
  # dm_policy: disabled          # open | allowlist | disabled
  # group_policy: allowlist      # open | allowlist | disabled
  # group_allow_from:            # List of group JIDs when group_policy=allowlist
  #   - <your-group-id>@g.us
```

**Group-only setup (no DMs):**
1. Set `WHATSAPP_ALLOWED_USERS=*` in `~/.hermes/.env` (bridge accepts all)
2. Set `allowed_chats` in `config.yaml` to the group JID (gateway restricts)

**PITFALL:** If you set `WHATSAPP_ALLOWED_USERS=` (empty) to block DMs, the bridge rejects ALL senders including group members. Groups won't work either. Use `*` and let the gateway filter.

## Finding Group Chat IDs

The bridge has a REST API on port 3000:

```bash
# Health check
curl http://127.0.0.1:3000/health

# Get chat info by JID
curl http://127.0.0.1:3000/chat/<your-group-id>@g.us
# Returns: {"name": "Prospect", "id": "..."}

# Get messages from a chat
curl http://127.0.0.1:3000/messages?chat=CHAT_JID

# Send a message
curl -X POST http://127.0.0.1:3000/send -H 'Content-Type: application/json' \
  -d '{"to": "CHAT_JID", "message": "Hello"}'
```

**To find a group by name:**
1. Get group JIDs from bridge log: `grep -oP '"chatId":"\K[^"]*@g\.us' ~/.hermes/whatsapp/bridge.log | sort -u`
2. Query each JID via the API to get the name
3. Match against the group you want

## Restarting After Config Changes

```bash
# After changing ~/.hermes/.env:
hermes gateway restart

# After changing config.yaml:
hermes gateway restart

# Verify bridge picked up new config:
tail -5 ~/.hermes/whatsapp/bridge.log
# Look for "🔒 Allowed users: ..." line showing the new value
```

**PITFALL:** The bridge process may survive a gateway restart if the old process wasn't fully killed. Check `ps aux | grep bridge.js` and `kill` the old PID if needed before restarting.

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| All messages show "allowlist_mismatch" | `WHATSAPP_ALLOWED_USERS` doesn't match sender format | Set to `*` and use gateway filtering |
| DMs work but groups don't | `WHATSAPP_ALLOWED_USERS` is empty | Set to `*` |
| Groups work but DMs don't | `allowed_chats` restricts to groups only | This is intended for group-only mode |
| Config changes not taking effect | Old bridge process still running | `kill` old PID, then `hermes gateway restart` |
| LID mapping failures | WhatsApp Web session stale | Re-pair: restart bridge, scan QR code |

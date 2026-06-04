---
name: messaging-gateway-config
description: >
  Configure Hermes messaging gateway platforms (WhatsApp, Telegram, Discord).
  Access controls, group-only routing, allowlist mechanics, chat scoping.
  Use when setting up, debugging, or restricting messaging platform access.
tags: [gateway, whatsapp, telegram, messaging, config, access-control]
---

# Messaging Gateway Configuration

Configure platform-level access controls for Hermes messaging integrations.

## When to Use
- Setting up a new messaging platform (WhatsApp, Telegram, Discord)
- Restricting the bot to specific groups or users
- Debugging why messages aren't being received or responded to
- Changing access scope (e.g., group-only, DM-only, specific users)

## WhatsApp Bridge Architecture

```
WhatsApp → bridge.js (port 3000) → gateway (Python) → agent
```

Two layers of filtering:
1. **Bridge level** (`allowed_users`): Filters by SENDER identity
2. **Gateway level** (`allowed_chats`): Filters by CHAT identity

Both must pass for a message to reach the agent.

## Key Config Fields (config.yaml)

```yaml
whatsapp:
  allowed_users:    # Bridge-level sender filter
    - '+1234567890'  # Specific phone numbers
    # or '*' to allow all senders
  allowed_chats:    # Gateway-level chat filter (group IDs)
    '<your-group-id>@g.us'
```

## Access Control Patterns

### Group-Only (no DMs)
```yaml
whatsapp:
  allowed_users: ['*']           # Bridge accepts all
  allowed_chats: '<group@g.us>'  # Gateway restricts to group
```
**Why `*` not empty:** Empty `allowed_users` = no one allowed. The bridge rejects ALL senders when the allowlist is empty. Set to `*` and let the gateway handle chat-level filtering.

### Specific Users Only (no groups)
```yaml
whatsapp:
  allowed_users: ['+1234567890']
  allowed_chats: ''              # Empty = no group restriction
```

### Specific Group + Specific Users
```yaml
whatsapp:
  allowed_users: ['+1234567890']
  allowed_chats: '<group@g.us>'
```

## LID Mapping (WhatsApp Identity Resolution)

WhatsApp uses two identity formats:
- **Phone:** `919XXXXXXXXX@s.whatsapp.net`
- **LID:** `129102573002791@lid` (Linked Identity, opaque numeric)

The bridge maps between them using files in the session directory:
- `lid-mapping-<phone>.json` — phone → LID
- `lid-mapping-<lid>_reverse.json` — LID → phone

The `allowed_users` config accepts phone numbers. The bridge resolves LIDs to phone numbers via these mapping files before checking the allowlist.

## Finding Group Chat IDs

```bash
# Query the bridge API for a group's name
curl -s http://127.0.0.1:3000/chat/<chatId>

# Get all unique group IDs from bridge log
grep -oP '"chatId":"\K[^"]*@g\.us' ~/.hermes/whatsapp/bridge.log | sort -u

# Query each group to find by name
for GID in $(grep -oP '"chatId":"\K[^"]*@g\.us' ~/.hermes/whatsapp/bridge.log | sort -u); do
  NAME=$(curl -s "http://127.0.0.1:3000/chat/$GID" | python3 -c "import sys,json; print(json.load(sys.stdin).get('name','?'))")
  echo "$GID -> $NAME"
done
```

## Debugging

### Messages not arriving
1. Check bridge log: `tail -50 ~/.hermes/whatsapp/bridge.log`
2. `allowlist_mismatch` = sender identity not in allowed_users
3. `failed to decrypt` = session key missing (group message from unknown sender)
4. Check gateway is running: `hermes gateway status`

### Allowlist mismatch in groups
If group messages show `allowlist_mismatch`:
- The sender's LID doesn't map to a phone number in allowed_users
- Fix: Set `allowed_users: ['*']` and use `allowed_chats` for group restriction

### Bridge health check
```bash
curl -s http://127.0.0.1:3000/health
# Returns: {"status":"connected","queueLength":0,"uptime":...}
```

## Restarting After Config Changes
```bash
hermes gateway restart
```
The gateway re-reads config.yaml on restart. Session files persist.

## Pitfalls

1. **Empty allowed_users rejects everyone.** The bridge code explicitly returns false for empty allowlists (security default). Use `['*']` to allow all.

2. **Phone number format mismatch.** The bridge normalizes by stripping `+`. Config `+19032577286` becomes `19032577286`. LID mapping files store numbers without `+`. Ensure consistency.

3. **allowed_chats uses group JID format.** Must be `<number>@g.us` for groups, not phone numbers. Find the JID via the bridge API or log.

4. **Restart required after config changes.** The gateway doesn't hot-reload config.yaml. Always `hermes gateway restart` after changes.

5. **Session files on Windows vs WSL.** WhatsApp session files live at `~/.hermes/whatsapp/session/` on the side where the gateway runs (WSL). Check both `/home/<username>/.hermes/` and `/mnt/c/Users/<username>/.hermes/` if sessions seem missing.

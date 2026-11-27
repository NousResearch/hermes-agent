---
name: whatsapp-bridge-troubleshooting
description: "Fix WhatsApp bridge: npm, self-chat allowlist, owner msgs."
version: 1.0.0
author: marktamis + Hermes Agent
license: MIT
platforms: [linux]
---

# WhatsApp Bridge Troubleshooting

Use when the WhatsApp bridge is enabled but not behaving as expected: messages not received, bridge not connecting, npm install failing during setup, or messages being silently ignored.

## Bridge anatomy (know what you are looking at)

- **Bridge script**: `~/.hermes/hermes-agent/scripts/whatsapp-bridge/bridge.js`
- **Session dir**: `~/.hermes/platforms/whatsapp/session/` (Baileys multi-file auth state)
- **Bridge log**: `~/.hermes/platforms/whatsapp/bridge.log` — pino logger, **default level: `warn`** (only warnings/errors; info/debug messages are suppressed)
- **Bridge process**: spawned by the gateway; PID findable via `pgrep -f bridge.js`
- **Bridge HTTP port**: `3000` (health check: use the `web` tool on the bridge's health endpoint)
- **Env vars read by bridge** (from `.env` of the profile that launched the gateway):
  - `WHATSAPP_MODE` — `bot` or `self-chat`
  - `WHATSAPP_DM_POLICY` — `pairing`, `open`, `allowlist`, `disabled`
  - `WHATSAPP_FORWARD_OWNER_MESSAGES` — `true`/`false` (whether owner-typed messages in bot mode are forwarded)
  - `WHATSAPP_ALLOWED_USERS` — comma-separated phone numbers

## Step 1 — Confirm the bridge is alive

```bash
# Bridge process
pgrep -f "bridge.js"   # expect one PID

# Port
use the `web` tool on the bridge's health endpoint
# expected: {"status":"connected",...}

# Gateway platform state
grep -A5 '"whatsapp"' ~/.hermes/gateway_state.json
# expected: "state": "connected", "error_code": null
```

If the bridge process is missing, the gateway failed to spawn it — check `gateway.log` for the spawn error.

### Bridge crash-loop

**Symptom**: `gateway.log` shows repeated `WhatsApp bridge process exited unexpectedly (code -15)` or `(code -9)` followed by `Reconnecting whatsapp` and `✓ whatsapp reconnected successfully` in a tight cycle.

**Impact**: A bridge that keeps dying cannot discover new chats (groups, new DMs). Even if the bridge process is present (`pgrep` shows a PID), treat a crash-loop as effectively offline for discovery purposes.

**Check**: Use `search_files` with pattern `bridge process exited|reconnected` in `~/.hermes/logs/gateway.log` — if you see alternating exit/reconnect lines within seconds of each other, the bridge is looping.

**Workaround**: Use the `terminal` tool to send SIGTERM to the bridge process (the process spawned with `bridge.js` in its command line). Wait for the gateway to respawn it. If the loop persists after a clean respawn, the bridge is hitting a fatal error on startup — read the bridge's stderr from the gateway's process output to check for Baileys/auth errors.

## Restarting the WhatsApp bridge

### Gateway self-restart is blocked from inside the gateway process

The gateway process (`hermes gateway restart`, using your service manager's restart command from a separate shell, or `kill` + respawn) **cannot be invoked from within the running gateway** — the SIGTERM propagates to child processes and kills the restart command before it completes. This applies to both systemd-service and bare-CLI deployments.

**Workaround**: Use a separate shell outside the gateway. Options:
- If you installed the gateway as a systemd service: use your service manager's restart command from a non-gateway shell.
- The `~/.hermes/dashboard.sh` management script (`start`/`stop`/`restart`/`status`) works regardless of systemd
- If neither is available, restart the bridge alone by killing the bridge process and letting the gateway respawn it:

```bash
kill $(pgrep -f bridge.js)
# Gateway detects exit and respawns the bridge automatically
```

Verify the new bridge stays up: `sleep 5 && pgrep -f bridge.js` should show a new PID.

## Step 2 — Read the bridge's real output

The bridge.log only captures pino warnings/errors. The bridge also writes **JSON event lines to stdout** that describe every message decision. To see them:

```bash
BRIDGE_PID=$(pgrep -f "bridge.js")
cat /proc/$BRIDGE_PID/fd/1   # stdout — JSON events like {"event":"ignored","reason":"..."}
cat /proc/$BRIDGE_PID/fd/2   # stderr — startup messages, Baileys warnings
```

Key event types to look for:
- `{"event":"ignored","reason":"allowlist_mismatch"}` — incoming message from a non-allowed sender
- `{"event":"ignored","reason":"allowlist_mismatch_owner_chat"}` — bot-mode self-chat dropped by allowlist (see Step 5)
- `{"event":"ignored","reason":"self_chat_mismatch"}` — self-chat mode but message is not to the bot's own number
- `{"event":"ignored","reason":"drop_disabled"}` — bot mode, `fromMe: true`, `WHATSAPP_FORWARD_OWNER_MESSAGES` not set
- `{"event":"connected"}` — WhatsApp session established

## Step 3 — npm install failure during WhatsApp setup

When activating WhatsApp from the dashboard you may see:

```
500: npm install failed for WhatsApp bridge: no output
```

This is from `hermes_cli/web_routers/messaging.py:_ensure_whatsapp_bridge_dependencies()`. The function runs `npm install --silent` with `capture_output=True`. When `--silent` is combined with captured stderr, a failure produces **no output at all** — both stdout and stderr are empty — so the error message collapses to `(no output)`.

**To fix**: run npm install manually in the bridge directory (without `--silent`) to see the real error:

```bash
cd ~/.hermes/hermes-agent/scripts/whatsapp-bridge
npm install
```

Common causes: network issues, Node version incompatibility, corrupted `package-lock.json`. After fixing, re-trigger the dashboard WhatsApp setup — `node_modules` already existing means the install step will be skipped.

## Step 4 — Bot-mode messages from the same account are dropped

**Symptom**: In `WHATSAPP_MODE=bot`, sending a message to the bot's own number (e.g. from WhatsApp Desktop linked to the same account) produces no reply. The bridge log or stdout shows:

```json
{"event":"ignored","reason":"drop_disabled","chatId":"...","senderId":"..."}
```

**Cause**: In bot mode, the bridge calls `classifyOwnerMessageGate` for `fromMe: true` messages. Without `WHATSAPP_FORWARD_OWNER_MESSAGES=true`, the gate returns `drop_disabled` and drops the message.

**Fix**: Add to `.env`:

```
WHATSAPP_FORWARD_OWNER_MESSAGES=true
```

Then restart the gateway (use the `terminal` tool with the gateway's restart signal) so the bridge respawns with the new env.

## Step 5 — Bot-mode self-chat still dropped after enabling FORWARD_OWNER_MESSAGES

**Symptom**: After setting `WHATSAPP_FORWARD_OWNER_MESSAGES=true`, messages to the bot's own number are still dropped with:

```json
{"event":"ignored","reason":"allowlist_mismatch_owner_chat","chatId":"<LID>@lid","senderId":"<LID>@lid"}
```

**Cause**: In bot mode, owner-message forwarding still runs the allowlist check against the **chatId**. When the bot messages itself, `chatId` is the bot's own LID (e.g. `19211439063084@lid`). The allowlist contains phone numbers (e.g. `33687726682`). The LID→phone resolution uses `expandWhatsAppIdentifiers`, which requires a **forward** LID mapping file (`lid-mapping-<LID>.json`) to convert the LID to a phone number. Baileys creates the **reverse** mapping (`lid-mapping-<LID>_reverse.json`) but not the forward one, so the LID never resolves to a phone number and the allowlist check fails.

**Fix option A (code patch — permanent)**: In `bridge.js`, in the bot-mode branch of the `messages.upsert` handler, skip the allowlist check when `chatId` matches one of the bot's own IDs (`botIds` array, already in scope). Replace the unconditionally-applied `allowlistMatches` callback with one that returns `true` for self-chats.

Concrete patch (apply to `scripts/whatsapp-bridge/bridge.js`):

```diff
- const decision = classifyOwnerMessageGate({
-   fromMe: true,
-   fromOwnerEnabled: FORWARD_OWNER_MESSAGES,
-   recentlySent: recentlySentIds,
-   allowlistMatches: (id) => matchesAllowedUser(id, ALLOWED_USERS, SESSION_DIR),
-   messageId: msg.key.id,
-   chatId,
- });
+ const isSelfChat = botIds.includes(normalizeWhatsAppId(chatId));
+ const decision = classifyOwnerMessageGate({
+   fromMe: true,
+   fromOwnerEnabled: FORWARD_OWNER_MESSAGES,
+   recentlySent: recentlySentIds,
+   allowlistMatches: isSelfChat
+     ? () => true
+     : (id) => matchesAllowedUser(id, ALLOWED_USERS, SESSION_DIR),
+   messageId: msg.key.id,
+   chatId,
+ });
```

(`botIds` is in scope at that point — defined at line ~522 inside the `messages.upsert` handler as `const botIds = Array.from(new Set([normalizeWhatsAppId(sock.user?.id), normalizeWhatsAppId(sock.user?.lid)]).filter(Boolean))`.)

After patching, restart the gateway (use the `terminal` tool with the gateway's restart signal).

**Fix option B (env workaround)**: Add the bot's LID to `WHATSAPP_ALLOWED_USERS` alongside the phone number. This only works until the LID changes (re-pairing).

**Fix option C (mapping file)**: Create `lid-mapping-<LID>.json` in the session dir containing the phone number. This only works until the LID changes.

## Step 6 — Determine your WhatsApp number

The connected account's phone number is in the Baileys session creds:

```bash
python3 -c "
import json
from pathlib import Path
creds = Path('/home/hermes/.hermes/platforms/whatsapp/session/creds.json')
data = json.loads(creds.read_text())
me = data.get('me', {})
phone = me.get('id', '')
print(phone)
"
```

Or from `.env`: `WHATSAPP_ALLOWED_USERS` holds the authorized phone number.

## Step 7 — New WhatsApp groups are not discovered

**Symptom**: A new WhatsApp group was created and the user sent messages in it, but `hermes send --list` does not show the group and the gateway log shows no inbound messages from it.

**Cause A — Hermes is not a participant in the group**: WhatsApp groups are only visible to participants. Since Hermes shares the user's WhatsApp account (it is not a separate bot number), it cannot see a group that only contains the user. A group with only the user's own number has no external participants for Hermes to observe.

**Requirement**: A WhatsApp group must have at least one participant other than the account Hermes is connected to. This can be a business associate, a second WhatsApp account you control, or any other contact.

**Fix**: Add at least one other participant to the group, then send a message in the group. The bridge should discover the new chat and `hermes send --list` will show it (as `whatsapp:<group-name>` with chat type `group`).

**Cause B — Bridge crash-loop prevents sync**: If the bridge is in a crash-loop (see "Bridge crash-loop" above), it cannot sync the WhatsApp session to discover new groups even when they exist. Stabilize the bridge first.

**Cause C — Group policy rejects the group**: Check `WHATSAPP_GROUP_POLICY` in `.env`. If set to `disabled` or `pairing`, the bridge will ignore group messages.

**Verify**: After the group is discoverable, use `/sethome` inside the group chat to make it the home channel. Confirm with `hermes send --list` showing the group as a target.

## References

- `references/bridge-json-events.md` — full list of JSON event types the bridge emits to stdout
- `references/allowlist-resolution.md` — how `expandWhatsAppIdentifiers` walks LID mapping files

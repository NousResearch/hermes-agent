# Cron as Automation Engine — Cron Job Best Practices

## The Right Architecture

Cron jobs run INSIDE Hermes Agent's context — they have access to all native tools (web_search, web_extract, whatsapp, etc.). Write prompts that delegate to the agent's tools, not shell scripts.

```bash
# BAD: runs Node.js script with limited capabilities, no native tools
hermes cron create -- node prospects/auto-hunt.js

# GOOD: runs the agent with full tool access
hermes cron create --prompt "Use web_search and web_extract to find prospects..."
```

## Cron Job Prompt Template

When creating automation cron jobs, include these sections:

```text
You are the [project] automation agent. Your job: [what runs daily].

## Your Tools
Use web_search and web_extract to [task].
Save files using write_file and terminal commands.

## Step 1: [Action]
Run searches in PARALLEL.

## Step 2: [Score/Filter]
Score each result 0-100. Only keep 60+.

## Step 3: [Dedup]
Read [existing file]. Remove duplicates.

## Step 4: [Save]
1. Save to [daily path]
2. Update [master file]

## Step 5: [Hot alerts]
For score 85+, write WhatsApp message to [pending dir].

## Step 6: Summary
Print: "New X: N | New Y: N | Hot: N"

## Workdir: [project path]
```

## Cron Job Commands

```bash
hermes cron list                    # view all jobs
hermes cron list --all              # include paused
hermes cron create SCHED            # create with prompt (interactive)
hermes cron run ID                  # trigger now
hermes cron pause ID                # pause
hermes cron resume ID               # resume
hermes cron edit ID                 # edit schedule/prompt/delivery
hermes cron remove ID               # delete
```

## WhatsApp Delivery (Cron + WhatsApp Integration)

### Getting the Chat ID for Delivery

**Personal number format:** `whatsapp:+<number>` — works directly in cron `deliver:`.

**Group format:** Find the group JID from the bridge session files:
```bash
# List all group sender-key files (each = one group the account is in)
ls ~/.hermes/whatsapp/session/sender-key-*.json

# The JID is the filename without extension:
# sender-key-<group-id>@g.us--<hash>.json
# → JID: <group-id>@g.us
```

**Test WhatsApp delivery directly:**
```bash
# Bridge runs on port 3000 when managed by gateway
curl -s -X POST "http://localhost:3000/send" \
  -H "Content-Type: application/json" \
  -d '{"chatId": "<your-number>@c.us", "message": "Hello"}'
```

### WhatsApp Bridge Modes

| Flag | Mode | Use case |
|------|------|----------|
| `--pair-only` | Pairing only | First-time QR scan, no message processing |
| `--mode self-chat` | Self-chat | Only sends to own linked number, can't see groups |
| `--mode bot` | Bot | Full group support, sees all chats |

**When the bridge needs restart:** The gateway manages the bridge as a subprocess. DO NOT kill the bridge manually — it crashes the gateway. Instead restart the gateway:
```bash
hermes gateway restart
# or
hermes gateway stop && hermes gateway start
```

**Manual bridge (for debugging):**
```bash
# Pairing mode — shows QR, exit after scan
node ~/.hermes/hermes-agent/scripts/whatsapp-bridge/bridge.js \
  --session ~/.hermes/whatsapp/session --port 3001 --pair-only

# Bot mode — full group support
node ~/.hermes/hermes-agent/scripts/whatsapp-bridge/bridge.js \
  --session ~/.hermes/whatsapp/session --port 3001 --mode bot
```

### Gateway as Background Service (Survives Laptop Sleep/Reboot)

The gateway MUST be running for cron jobs to fire on schedule. Install as systemd user service:
```bash
hermes gateway install   # installs service
hermes gateway start      # starts it now
# systemd handles auto-start on boot
```

**WSL note:** systemd services may not survive WSL restart in older versions. Enable in `/etc/wsl.conf`:
```ini
[boot]
systemd=true
```

**Key log:** `~/.hermes/logs/gateway.log` — shows bridge lifecycle, WhatsApp connection status, and cron firing.

## Delivery Options

| Option | Use Case |
|--------|----------|
| `local` | Save to files in workdir (default) |
| `whatsapp:+<number>` | Push to personal WhatsApp |
| `whatsapp:<group-id>@g.us` | Push to WhatsApp group |
| `telegram:CHAT_ID` | Push to Telegram |
| `discord:#CHANNEL_ID` | Push to Discord channel |

## Examples

**Daily automated discovery:**
```
hermes cron create "0 1 * * *" \
  --name "Daily Auto-Discovery" \
  --prompt "Use web_search to find [category] prospects..." \
  --workdir /mnt/c/Users/<username>/<your-project-name>
```

**Weekly tech news digest:**
```
hermes cron create "0 9 * * 1" \
  --name "Tech News Digest" \
  --prompt "Search for top tech news this week..."
```

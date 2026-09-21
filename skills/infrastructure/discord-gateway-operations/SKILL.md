---
name: discord-gateway-operations
description: "Operate and scale Hermes Discord gateways across multiple profile-isolated bots. Create Discord applications, configure privileged intents, build systemd services, manage per-profile .env tokens, strip MCP/cron from specialist gateways, and budget memory for multi-gateway deployments."
version: 1.2.0
adoption_status: permanent
---

# Discord Gateway Operations

## When this skill applies

- You are creating a new Discord bot for a Hermes profile
- You are scaling from one bot to multiple profile-isolated bots
- A Discord bot fails to connect with `PrivilegedIntentsRequired`
- You need to budget memory across multiple gateway instances
- A specialist bot should not spawn MCP servers or run its own crons
- Sahil asks about channel structure, co-working patterns, or persona routing in Discord
- Sahil says "I want to speak to both X and Y at the same time" (co-working signal)

## CRITICAL: Specialist gateways MUST NOT touch Kanban DBs (P0 governance rule, 26/05/26)

The ops board DB was corrupted 3 times in one session because 8 specialist gateway processes (Remii, Wesker, Octacon, CeeCee, Gojo, Misa-Misa, MrHermagi, Dezzy) were all opening and writing to `/home/kensei/.hermes/kanban/boards/ops/kanban.db` concurrently. Each process cached its own SQLite page state; writes from one invalidated index pointers cached by another; repeated REINDEX operations from different processes created index divergence; eventually the file header was overwritten with garbage.

**Root cause:** The specialist gateways inherited kanban DB access from their parent config or the `--replace` restart cascade. SQLite WAL serialises writes but does NOT sync cached page pointers between independent processes in WAL mode. 8 concurrent writers to one SQLite file = guaranteed corruption.

**Prevention (non-negotiable for ALL specialist profiles):**
1. `kanban.dispatch_in_gateway: false` in every specialist profile config
2. Verify with `lsof` after restart: only the Kensei gateway process should have ANY kanban DB file open
3. If a specialist gateway has kanban DB files open, stop it immediately — do not dispatch any tasks

**Verification command:**
```bash
lsof 2>/dev/null | grep kanban.db | awk '{print $1, $2}' | sort -u
# Expected: only one PID (the Kensei gateway)
```

**Recovery if corruption occurs:**
1. Stop ALL gateways (specialists + Kensei)
2. Rebuild the corrupt board DB from scratch (`hermes kanban --board <slug> init`)
3. Restart ONLY Kensei gateway
4. Verify no specialist gateways have kanban DB files open
5. Only then restart specialists

See `kanban-ops/references/kanban-db-integrity-repair.md` for the full repair pattern and `kanban-ops/` for the incident trace.

## Key constraints

**Hermes requires privileged intents on every Discord bot:**
- MESSAGE CONTENT INTENT (to read message content)
- SERVER MEMBERS INTENT (for role/permission lookups)

Without both enabled in the Discord Developer Portal → Bot tab → Privileged Gateway Intents, the gateway fails with `discord.errors.PrivilegedIntentsRequired` and retries indefinitely.

When creating a Discord bot that will share channels with other bots (co-working), the bot MUST have `free_response_channels` set to its home channel ID. Without this, bot-to-bot messages that reference it by name (e.g. "Dezzy — please review this") without a proper `@Dezzy` mention get silently dropped because `message.mentions` is empty.

For `free_response_channels`, use the channel ID from `DISCORD_HOME_CHANNEL` in the profile's `.env` file.

## CRITICAL PREREQUISITE: the `discord:` section in config.yaml

Every specialist gateway MUST have a `discord:` section in its profile's `config.yaml` with at minimum `require_mention: true` and the correct `free_response_channels` set to the bot's home channel. **Without this section, the gateway connects and logs in successfully but silently drops every message** — the adapter never fires its YAML→env bridge, so `require_mention` defaults to `true` with no free-response channels configured.

This is the single most common failure after "PrivilegedIntentsRequired", and it is **silent** — the gateway reports `Connected as Bot#1234` and sits idle. No error, no log entry, no retry. The bot appears online but never responds to anything.

### Batch silent-drop fix (all profiles at once)

When adding a new batch of specialists that all lack the `discord:` section, do NOT fix them one by one. Apply the section to every config in a single pass:

```bash
# For each profile that needs the discord: section,
# add it just above the always_skills: or checkpoints: key
# Use the channel ID from DISCORD_HOME_CHANNEL in the profile's .env

for p in profile1 profile2 profile3; do
  env_file="/home/kensei/.hermes/profiles/$p/.env"
  channel_id=$(grep -oP 'DISCORD_HOME_CHANNEL=\K\d+' "$env_file")
  
  echo "Adding discord: section to $p (home=$channel_id)"
done

sudo systemctl restart hermes-gateway-{profile1,profile2,profile3}
```

**Common pitfalls in batch fixes:**
- Patching after the `always_skills:` key is **wrong** — the second `discord:` section creates a duplicate. Remove the old empty `discord:` section first, then add the real one, or place it correctly the first time.
- Verify with `grep -c '^discord:' profiles/$p/config.yaml` — output should be exactly 1.
- The `channel_prompts` key from Kensei's root config must be moved to the specialist's profile if the channel is owned by that specialist. Kensei's root `channel_prompts` entries hijack the persona identity of that channel.
- After applying changes, restart ALL affected gateways and verify each shows `Connected as Name#tag`.

### Check if your bot is suffering from silent drop

If a bot is connected and shows `active (running)` in systemd but never responds even when mentioned:

```bash
# 1. Verify the discord: section exists in config.yaml
grep -c '^discord:' /home/kensei/.hermes/profiles/<profile>/config.yaml
# 0 = missing → the silent-drop scenario

# 2. Compare against a known-good specialist
grep -A15 '^discord:' /home/kensei/.hermes/profiles/wesker/config.yaml

# 3. Check the gateway log — if you see "Connected as" but zero message logs
#    (no "Processing message from", no "Sending response to"), the section is missing
grep -n 'Connected as\|message\|respond\|free_response\|require_mention' \
  /home/kensei/.hermes/profiles/<profile>/logs/gateway.log | tail -20
```

### Minimum viable discord: section

```yaml
discord:
  require_mention: true
  free_response_channels: '<HOME_CHANNEL_ID>'     # where bot responds without @mention
  allowed_channels: ''
  auto_thread: true
  reactions: true
  channel_prompts: {}
  extra:
    gateway_restart_notification: false
  server_actions: ''
```

Replace `<HOME_CHANNEL_ID>` with the channel ID from the profile's `.env` `DISCORD_HOME_CHANNEL` value.

### How the silent drop happens (gateway internals)

1. No `discord:` section → `apply_yaml_config_fn` (adapter.py line 6232) never fires
2. `DISCORD_REQUIRE_MENTION` env var is never set → defaults to `"true"` (adapter.py line 3745)
3. Every incoming message hits line 4703:
   ```python
   if require_mention and not is_free_channel and not in_bot_thread:
       if self._client.user not in message.mentions and not mention_prefix:
           return  # <-- silent drop, no log
   ```
4. The multi-agent filter at line 837 also fails to save it because empty `message.mentions` means the bot-mention check at line 839 (`self._client.user in message.mentions`) is `False`, so the early-return guard at line 847–848 never fires either — but the `require_mention` check below it catches the message and drops it silently.

### Fix

Add the `discord:` section to the profile's `config.yaml`, then restart the gateway. No re-auth, no new token needed — the section only controls message routing env vars.

```bash
sudo systemctl restart hermes-gateway-<profile>
# Verify the gateway picked up the config
tail -5 /home/kensei/.hermes/profiles/<profile>/logs/gateway.log
# Should show "Connected as Bot#1234" (same as before, but env vars are now set)
```

## Bot creation workflow

### Step 1: Create the Discord application

Go to https://discord.com/developers/applications → New Application → name it after the persona.

### Step 2: Make it private (optional but recommended)

1. Go to **Installation** tab.
2. If a Custom Install URL or Default Authorization Link is set, **clear it first** — otherwise Discord blocks toggling Public Bot off.
3. Save changes.
4. Go to **Bot** tab → turn OFF "Public Bot" → Save Changes.
5. Still on Bot tab → scroll to **Privileged Gateway Intents** → toggle on:
   - MESSAGE CONTENT INTENT ✓
   - SERVER MEMBERS INTENT ✓

### Step 3: Generate the token

Bot tab → Reset Token → copy it. The token goes into the profile's `.env` file as `DISCORD_BOT_TOKEN=...`.

### Step 4: Invite the bot

The OAuth2 URL Generator does not support private bots. Construct the URL manually:

```text
https://discord.com/oauth2/authorize?client_id=CLIENT_ID&permissions=PERMISSIONS&integration_type=0&scope=bot+applications.commands
```

Replace `CLIENT_ID` with the Application ID (found on General Information tab).
Replace `PERMISSIONS`:
- `117760` for text-only (Read/Send/Embed/Attach/History)
- `7457792` for text + voice (adds Connect/Speak/Use Voice Activity)

### Step 5: Create the .env file

```bash
cat > /home/kensei/.hermes/profiles/<profile>/.env << ENVEOF
DISCORD_BOT_TOKEN=<token>
DISCORD_ALLOWED_USERS=797682085224513547
ENVEOF
chmod 600 /home/kensei/.hermes/profiles/<profile>/.env
```

### Step 6: Create the systemd service

Each specialist gateway gets its own service file pointing to its profile directory via `HERMES_HOME`.

Reference service:

```ini
[Unit]
Description=Hermes Gateway – <persona> (<purpose>)
After=network-online.target hermes-gateway.service
Wants=network-online.target

[Service]
Type=simple
User=kensei
Group=kensei
Environment="HOME=/home/kensei"
Environment="HERMES_HOME=/home/kensei/.hermes/profiles/<profile>"
Environment="PATH=/home/kensei/.hermes/hermes-agent/venv/bin:/home/kensei/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="VIRTUAL_ENV=/home/kensei/.hermes/hermes-agent/venv"
ExecStart=/home/kensei/.hermes/hermes-agent/venv/bin/python -m hermes_cli.main gateway run --replace
Restart=always
RestartSec=10
RestartMaxDelaySec=120
RestartSteps=5
KillMode=mixed
KillSignal=SIGTERM
TimeoutStopSec=90

# Specialist gateway memory budget: ~130MB baseline, cap at 1G
MemoryMax=1G
MemoryHigh=800M

[Install]
WantedBy=multi-user.target
```

**Important:** The service should `After=network-online.target hermes-gateway.service` but NOT `Requires=` the main gateway. They run independently.

### Step 7: Configure lean operation

Specialist gateways MUST NOT run cron jobs or spawn MCP servers:

```yaml
# In profile's config.yaml, strip these:
# Remove mcp_servers key entirely
# Remove or disable the cron ticker by disabling cron/jobs.json entries

# In profile's cron/jobs.json, disable all jobs:
for j in data.get('jobs', []):
    if j.get('enabled'):
        j['enabled'] = False
```

Cron jobs live only on Kensei's main gateway. Specialist gateways handle interactive chat only.

## Memory budgeting

| Gateway type | Baseline RSS | Cap | Count | Total |
|---|---|---|---|---|---|
| Kensei (main) | ~230MB | 5G | 1 | ~230MB |
| Specialist text | ~130MB each | 1G | 6-7 | ~800MB-1GB |
| **Total** | | | 7-8 | **~1-1.5GB** |

On an 8GB VPS with current load (~4-5GB baseline), 7 gateways fit comfortably. The kanban dispatcher should remain disabled in-gateway (`kanban.dispatch_in_gateway: false`).

## Co-working patterns

When multiple bots exist in the same Discord channel, Sahil can address them by name:

```text
Sahil: Remii, what's new in AI tools? And Wesker, can we run that?

Remii Bot: Three new ones this week...
Wesker Bot: We've got ~2GB headroom on the VPS...
```

Each bot maintains its own session state, memory, and reasoning context. They are independent conversational entities that happen to share a channel.

### allow_bots: controls whether bots see each other's messages

The `DISCORD_ALLOW_BOTS` env var (or `discord.extra.allow_bots`) controls message intake for bot-authored messages:

| Value | Behaviour | Use case |
|---|---|---|
| `none` (default) | All bot messages silently dropped | Tight isolation — bots never see each other |
| `mentions` | Only process bot messages where this bot is `@mentioned` | **Recommended for multi-bot setups.** Enables explicit handoffs (e.g. Kensei `@Dezzy please review`) while preventing cross-talk pollution |
| `all` | Process all bot messages unrestricted | Debugging only — floods every bot's intake |

Set in profile's `config.yaml`:

```yaml
discord:
  extra:
    gateway_restart_notification: false
    allow_bots: mentions
```

**How it works** (adapter.py):
1. **Intake filter** (line 797-812): Bot messages hit `DISCORD_ALLOW_BOTS` before anything else. `none` → `return`. `mentions` → only proceed if `self._client.user` is in `message.mentions`.
2. **Multi-agent filter** (line 828-862): After intake, messages that mention OTHER bots but not THIS bot are dropped — prevents cross-consumption in co-working channels.
3. **History backfill** (line 3908-3957): `mentions` is treated as `all` here — context assembly is about information, not gating.

**Important caveat**: `allow_bots: mentions` still requires an actual `@mention` Discord ping. Plain text "Dezzy — please review" without `@Dezzy` produces empty `message.mentions`, so `allow_bots: mentions` never triggers. Use `free_response_channels` for the bot's home channel to handle text-only references. For co-working channels, always use proper `@mentions`.

**Pitfall: this controls intake, not output.** `allow_bots: mentions` on Dezzy means Dezzy processes messages FROM other bots. It does NOT affect whether other bots see Dezzy's messages — that's controlled by each bot's own `allow_bots` setting. Set it on every bot that should be reachable for handoffs.

### Bot-to-bot work flow

When a non-coding lead (Remii, Dezzy, etc.) produces output needing implementation:
1. **PREFERRED**: Post findings to channel + file kanban task assigned to execution lead
2. **FALLBACK**: @mention the execution lead in-channel (urgent or tightly-scoped work)
3. **NEVER**: Expect plain-text name references to trigger responses — text "Dezzy — ..." without @mention is silently dropped outside `free_response_channels`

### Co-working channel mapping (as of 2026-05-24)

| Channel | Free-response bots | @mention-only bots |
|---|---|---|
| `#general`, `#cron-outputs`, `#kanban`, `#governance`, `#approvals`, `#decisions` | Kensei | All others |
| `#ops` | Kensei, Wesker | All others |
| `#war-room` | Kensei | All others |
| `#design-review` | Dezzy | All others |
| `#misa-misa` | Misa-Misa | All others |
| `#research-digest` | Remii | All others |
| `#research-ops` | Remii, Wesker | All others |
| `#knowledge` | — | All bots |
| `#mailbox__calendar`, `#job-hunt` | Gojo | All others |
| `#build-log`, `#build-review` | Octacon | All others |
| `#content` | CeeCee | All others |
| `#ai-learning-qa` | MrHermagi | All others |

Recommended multi-bot channels:
- `#war-room` / `#general`: Kensei + all specialists
- `#research-ops`: Remii + Wesker (infra+research overlap)
- `#build-review`: Octacon + Quan (coding+QA)
- `#content-review`: CeeCee + Kensei (content approval)

## Channel structure by workflow, not persona

Default channel-to-persona mapping:

| Channel | Default persona(s) |
|---|---|
| `#general` / `#ops` | Kensei, Wesker |
| `#governance` | Kensei, Denji |
| `#job-hunt` | Gojo |
| `#plenishd` / `#coachsense` | Octacon, CeeCee |
| `#research-digest` / `#research-ops` | Remii, Wesker |
| `#build-log` / `#build-review` | Octacon |
| `#mrhermagi-lessons` | MrHermagi |
| `#approvals` | Kensei only (signed decisions) |

## Privileged intents setup and debugging

Privileged intents are the **most common issue** when setting up new Discord bots. Every new bot must have these enabled. Without them, the gateway fails with `discord.errors.PrivilegedIntentsRequired` and retries indefinitely.

### Setup

1. Go to Developer Portal → bot's application → **Bot** tab
2. Scroll to **Privileged Gateway Intents**
3. Toggle ON **MESSAGE CONTENT INTENT** and **SERVER MEMBERS INTENT**
4. Click **Save Changes**
5. Restart the gateway: `sudo systemctl restart hermes-gateway-<bot>`

### Portal quirk: intents not persisting (most common debugging scenario)

Discord's Developer Portal sometimes does NOT persist intents on first save even though the toggle shows ON. This happens often when creating multiple bots in rapid succession — the portal's UI lies to you.

If a bot still fails with `PrivilegedIntentsRequired` after the user says "they're saved":

1. **Verify you're editing the correct application.** When 6+ bots were created in rapid succession, the user can accidentally toggle intents on the wrong app. Ask them to check the application name at the top of the Bot tab. Cross-reference the Application ID from the portal against the one in the profile's gateway log.

2. **Force-refresh the intents state.** Toggle both intents OFF → Save → toggle both ON → Save again. This force-flushes Discord's stale intent cache on their side. A gateway restart is still needed afterward.

3. **Confirm the correct process restarted.** Check the PID and start time:
   ```bash
   systemctl show -p MainPID -p ActiveEnterTimestamp --value hermes-gateway-<bot>
   ```

4. **Read the right log file.** See diagnostic note below.

### Diagnostic: gateway.log vs journalctl

This is a critical distinction that causes repeated misdiagnosis.

When a gateway restarts, `journalctl -u hermes-gateway-<bot>` may still show old errors from the **killed** process while the new process is sitting quietly waiting for retry. Journalctl merges both old and new PID output into one stream with no visual separation.

**Always check the profile's own gateway.log for current connection state:**

```bash
tail -15 /home/kensei/.hermes/profiles/<profile>/logs/gateway.log
```

The gateway.log is written per-process-session. It will contain either:
- `Connected as BotName#1234` (success)
- `discord.errors.PrivilegedIntentsRequired` (failure)
- `discord connect timed out after 30s` (failure)

These three lines tell you the real state. Ignore journalctl noise for intent diagnosis.ntentsRequired` (failure)
- `discord connect timed out after 30s` (failure)

These three lines tell you the real state. Ignore journalctl noise for intent diagnosis.

## Gateway lifecycle notification suppression

When a Hermes gateway shuts down or restarts, it broadcasts `⚠️ Gateway shutting down — Your current task will be interrupted.` to **every active session** (every channel/thread with a conversation) PLUS **every platform home channel**. This hits forums, text channels, DMs — everywhere.

The spread is controlled by `gateway_restart_notification: bool` on a per-platform basis in each gateway's `config.yaml`.

### Default behaviour

The `PlatformConfig` dataclass defaults `gateway_restart_notification = True` (see `gateway/config.py` line 299). Four code paths respect this flag:

| Code path | File | Line | What it sends |
|-----------|------|------|--------------|
| `_notify_active_sessions_of_shutdown()` | `gateway/run.py` | 3358 | `⚠️ Gateway shutting down` to each platform's **home channel only** (single-loop, no per-channel flood) |
| `_send_home_channel_startup_notifications()` | `gateway/run.py` | 14387 | `♻️ Gateway online` to all home channels |
| `_send_restart_notification()` | `gateway/run.py` | 14324 | `♻ Gateway restarted successfully` to the chat that issued `/restart` |

**Key insight:** The shutdown notification must be **home-channel-only**. In KenseiAgent, `_notify_active_sessions_of_shutdown()` was refactored (2026-05-25) to remove the per-active-session broadcast loop. Notifications route exclusively to each platform's configured home channel. If you have active agents in 5 Discord channels, you get 1 notification (to the home channel) instead of 5. See `references/gateway-notification-routing.md` for the full internal architecture.

**Deployment drift check:** services may run `/home/kensei/.hermes/hermes-agent/...` while the active development checkout is `/home/kensei/repos/KenseiAgent/...`. Always inspect the service checkout, not just the repo checkout:

```bash
python3 - <<'PY'
from pathlib import Path
for p in ['/home/kensei/.hermes/hermes-agent/gateway/run.py','/home/kensei/repos/KenseiAgent/gateway/run.py']:
    text=Path(p).read_text(); s=text.index('    async def _notify_active_sessions_of_shutdown'); e=text.index('    def _finalize_shutdown_agents', s); body=text[s:e]
    print(p, 'active-session send present?', 'Sent shutdown notification to active chat' in body)
PY
```

If the service checkout still contains `Sent shutdown notification to active chat`, it will cascade shutdown pings across active Discord channels even if the repo checkout looks fixed. Patch/sync the service checkout and restart the gateway only after approval.

### How to suppress the flood

Add `gateway_restart_notification: false` to the Discord platform's `extra:` block in `config.yaml`:

```yaml
discord:
  # ... other config ...
  extra:
    gateway_restart_notification: false
```

### 4.3 Do not let every specialist gateway own cron

Cron jobs belong to the root Kensei gateway unless explicitly designed otherwise.

Hard guard specialist gateways at BOTH levels:

1. Profile config:

```yaml
cron:
  ticker_enabled: false
kanban:
  dispatch_in_gateway: false
```

2. systemd service environment:

```ini
Environment="HERMES_CRON_TICKER_DISABLED=1"
Environment="HERMES_KANBAN_DISPATCH_IN_GATEWAY=false"
```

Hermes `gateway/run.py` supports `HERMES_CRON_TICKER_DISABLED=1` and `cron.ticker_enabled: false`; without this guard older gateways start a cron ticker even when profile-local `jobs.json` is empty.

Check specialist profile cron files:

```bash
find ~/.hermes/profiles -path '*/cron/jobs.json' -print
```

If a specialist has profile-local cron jobs, either:
- migrate them into root cron with an explicit `profile: <lead>` field, or
- pause them with a reason saying root cron is scheduler-of-record.

Verification after restart:

```bash
for p in ceecee denji gojo light misa-misa mrhermagi octacon quan remii wesker; do
  tail -40 ~/.hermes/profiles/$p/logs/gateway.log | grep -E 'Cron ticker|kanban dispatcher|Connected as'
done
```

Expected specialist lines:

```text
Cron ticker disabled for this gateway
kanban dispatcher: disabled via HERMES_KANBAN_DISPATCH_IN_GATEWAY env
```

Systemd webhook notifications via `ExecStopPost`/`ExecStartPost` are **not recommended** for multi-bot setups — they fire once per gateway restart and can't distinguish between intentional maintenance restarts and crash loops. The simpler approach is:
1. Set `true` on Kensei-root → notifications go to `#general` only
2. Set `false` on all specialists → silent
3. If you need more granular alerts, monitor the gateway logs instead

### Pitfalls

- **The flag silences per-platform, not per-channel.** Setting `true` on Kensei with active sessions in 5 Discord channels still sends 5 notifications (one per channel). Setting `false` on specialists silences them entirely. For true single-channel notifications, combine Kensei-root `true` + all specialists `false`, understanding that Kensei-root may still multi-cast if it has active agents in multiple channels.
- **Profile configs may not have an `extra:` block.** Add it inside the `discord:` section — the parser bridges it correctly.
- **Gateway restart required.** The change only takes effect after restarting the gateway process.

## Channel management via REST API

Kensei's bot token can manage Discord channels programmatically using the Discord REST API.

### Delete a channel


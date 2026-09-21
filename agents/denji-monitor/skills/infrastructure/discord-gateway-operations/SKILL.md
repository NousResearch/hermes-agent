     1|---
     2|name: discord-gateway-operations
     3|description: "Operate and scale Hermes Discord gateways across multiple profile-isolated bots. Create Discord applications, configure privileged intents, build systemd services, manage per-profile .env tokens, strip MCP/cron from specialist gateways, and budget memory for multi-gateway deployments."
     4|version: 1.2.0
     5|adoption_status: permanent
     6|---
     7|
     8|# Discord Gateway Operations
     9|
    10|## When this skill applies
    11|
    12|- You are creating a new Discord bot for a Hermes profile
    13|- You are scaling from one bot to multiple profile-isolated bots
    14|- A Discord bot fails to connect with `PrivilegedIntentsRequired`
    15|- You need to budget memory across multiple gateway instances
    16|- A specialist bot should not spawn MCP servers or run its own crons
    17|- Sahil asks about channel structure, co-working patterns, or persona routing in Discord
    18|- Sahil says "I want to speak to both X and Y at the same time" (co-working signal)
    19|
    20|## CRITICAL: Specialist gateways MUST NOT touch Kanban DBs (P0 governance rule, 26/05/26)
    21|
    22|The ops board DB was corrupted 3 times in one session because 8 specialist gateway processes (Remii, Wesker, Octacon, CeeCee, Gojo, Misa-Misa, MrHermagi, Dezzy) were all opening and writing to `/home/kensei/.hermes/kanban/boards/ops/kanban.db` concurrently. Each process cached its own SQLite page state; writes from one invalidated index pointers cached by another; repeated REINDEX operations from different processes created index divergence; eventually the file header was overwritten with garbage.
    23|
    24|**Root cause:** The specialist gateways inherited kanban DB access from their parent config or the `--replace` restart cascade. SQLite WAL serialises writes but does NOT sync cached page pointers between independent processes in WAL mode. 8 concurrent writers to one SQLite file = guaranteed corruption.
    25|
    26|**Prevention (non-negotiable for ALL specialist profiles):**
    27|1. `kanban.dispatch_in_gateway: false` in every specialist profile config
    28|2. Verify with `lsof` after restart: only the Kensei gateway process should have ANY kanban DB file open
    29|3. If a specialist gateway has kanban DB files open, stop it immediately — do not dispatch any tasks
    30|
    31|**Verification command:**
    32|```bash
    33|lsof 2>/dev/null | grep kanban.db | awk '{print $1, $2}' | sort -u
    34|# Expected: only one PID (the Kensei gateway)
    35|```
    36|
    37|**Recovery if corruption occurs:**
    38|1. Stop ALL gateways (specialists + Kensei)
    39|2. Rebuild the corrupt board DB from scratch (`hermes kanban --board <slug> init`)
    40|3. Restart ONLY Kensei gateway
    41|4. Verify no specialist gateways have kanban DB files open
    42|5. Only then restart specialists
    43|
    44|See `kanban-ops/references/kanban-db-integrity-repair.md` for the full repair pattern and `kanban-ops/` for the incident trace.
    45|
    46|## Key constraints
    47|
    48|**Hermes requires privileged intents on every Discord bot:**
    49|- MESSAGE CONTENT INTENT (to read message content)
    50|- SERVER MEMBERS INTENT (for role/permission lookups)
    51|
    52|Without both enabled in the Discord Developer Portal → Bot tab → Privileged Gateway Intents, the gateway fails with `discord.errors.PrivilegedIntentsRequired` and retries indefinitely.
    53|
    54|When creating a Discord bot that will share channels with other bots (co-working), the bot MUST have `free_response_channels` set to its home channel ID. Without this, bot-to-bot messages that reference it by name (e.g. "Dezzy — please review this") without a proper `@Dezzy` mention get silently dropped because `message.mentions` is empty.
    55|
    56|For `free_response_channels`, use the channel ID from `DISCORD_HOME_CHANNEL` in the profile's `.env` file.
    57|
    58|## CRITICAL PREREQUISITE: the `discord:` section in config.yaml
    59|
    60|Every specialist gateway MUST have a `discord:` section in its profile's `config.yaml` with at minimum `require_mention: true` and the correct `free_response_channels` set to the bot's home channel. **Without this section, the gateway connects and logs in successfully but silently drops every message** — the adapter never fires its YAML→env bridge, so `require_mention` defaults to `true` with no free-response channels configured.
    61|
    62|This is the single most common failure after "PrivilegedIntentsRequired", and it is **silent** — the gateway reports `Connected as Bot#1234` and sits idle. No error, no log entry, no retry. The bot appears online but never responds to anything.
    63|
    64|### Batch silent-drop fix (all profiles at once)
    65|
    66|When adding a new batch of specialists that all lack the `discord:` section, do NOT fix them one by one. Apply the section to every config in a single pass:
    67|
    68|```bash
    69|# For each profile that needs the discord: section,
    70|# add it just above the always_skills: or checkpoints: key
    71|# Use the channel ID from DISCORD_HOME_CHANNEL in the profile's .env
    72|
    73|for p in profile1 profile2 profile3; do
    74|  env_file="/home/kensei/.hermes/profiles/$p/.env"
    75|  channel_id=$(grep -oP 'DISCORD_HOME_CHANNEL=\K\d+' "$env_file")
    76|  
    77|  echo "Adding discord: section to $p (home=$channel_id)"
    78|done
    79|
    80|sudo systemctl restart hermes-gateway-{profile1,profile2,profile3}
    81|```
    82|
    83|**Common pitfalls in batch fixes:**
    84|- Patching after the `always_skills:` key is **wrong** — the second `discord:` section creates a duplicate. Remove the old empty `discord:` section first, then add the real one, or place it correctly the first time.
    85|- Verify with `grep -c '^discord:' profiles/$p/config.yaml` — output should be exactly 1.
    86|- The `channel_prompts` key from Kensei's root config must be moved to the specialist's profile if the channel is owned by that specialist. Kensei's root `channel_prompts` entries hijack the persona identity of that channel.
    87|- After applying changes, restart ALL affected gateways and verify each shows `Connected as Name#tag`.
    88|
    89|### Check if your bot is suffering from silent drop
    90|
    91|If a bot is connected and shows `active (running)` in systemd but never responds even when mentioned:
    92|
    93|```bash
    94|# 1. Verify the discord: section exists in config.yaml
    95|grep -c '^discord:' /home/kensei/.hermes/profiles/<profile>/config.yaml
    96|# 0 = missing → the silent-drop scenario
    97|
    98|# 2. Compare against a known-good specialist
    99|grep -A15 '^discord:' /home/kensei/.hermes/profiles/wesker/config.yaml
   100|
   101|# 3. Check the gateway log — if you see "Connected as" but zero message logs
   102|#    (no "Processing message from", no "Sending response to"), the section is missing
   103|grep -n 'Connected as\|message\|respond\|free_response\|require_mention' \
   104|  /home/kensei/.hermes/profiles/<profile>/logs/gateway.log | tail -20
   105|```
   106|
   107|### Minimum viable discord: section
   108|
   109|```yaml
   110|discord:
   111|  require_mention: true
   112|  free_response_channels: '<HOME_CHANNEL_ID>'     # where bot responds without @mention
   113|  allowed_channels: ''
   114|  auto_thread: true
   115|  reactions: true
   116|  channel_prompts: {}
   117|  extra:
   118|    gateway_restart_notification: false
   119|  server_actions: ''
   120|```
   121|
   122|Replace `<HOME_CHANNEL_ID>` with the channel ID from the profile's `.env` `DISCORD_HOME_CHANNEL` value.
   123|
   124|### How the silent drop happens (gateway internals)
   125|
   126|1. No `discord:` section → `apply_yaml_config_fn` (adapter.py line 6232) never fires
   127|2. `DISCORD_REQUIRE_MENTION` env var is never set → defaults to `"true"` (adapter.py line 3745)
   128|3. Every incoming message hits line 4703:
   129|   ```python
   130|   if require_mention and not is_free_channel and not in_bot_thread:
   131|       if self._client.user not in message.mentions and not mention_prefix:
   132|           return  # <-- silent drop, no log
   133|   ```
   134|4. The multi-agent filter at line 837 also fails to save it because empty `message.mentions` means the bot-mention check at line 839 (`self._client.user in message.mentions`) is `False`, so the early-return guard at line 847–848 never fires either — but the `require_mention` check below it catches the message and drops it silently.
   135|
   136|### Fix
   137|
   138|Add the `discord:` section to the profile's `config.yaml`, then restart the gateway. No re-auth, no new token needed — the section only controls message routing env vars.
   139|
   140|```bash
   141|sudo systemctl restart hermes-gateway-<profile>
   142|# Verify the gateway picked up the config
   143|tail -5 /home/kensei/.hermes/profiles/<profile>/logs/gateway.log
   144|# Should show "Connected as Bot#1234" (same as before, but env vars are now set)
   145|```
   146|
   147|## Bot creation workflow
   148|
   149|### Step 1: Create the Discord application
   150|
   151|Go to https://discord.com/developers/applications → New Application → name it after the persona.
   152|
   153|### Step 2: Make it private (optional but recommended)
   154|
   155|1. Go to **Installation** tab.
   156|2. If a Custom Install URL or Default Authorization Link is set, **clear it first** — otherwise Discord blocks toggling Public Bot off.
   157|3. Save changes.
   158|4. Go to **Bot** tab → turn OFF "Public Bot" → Save Changes.
   159|5. Still on Bot tab → scroll to **Privileged Gateway Intents** → toggle on:
   160|   - MESSAGE CONTENT INTENT ✓
   161|   - SERVER MEMBERS INTENT ✓
   162|
   163|### Step 3: Generate the token
   164|
   165|Bot tab → Reset Token → copy it. The token goes into the profile's `.env` file as `DISCORD_BOT_TOKEN=...`.
   166|
   167|### Step 4: Invite the bot
   168|
   169|The OAuth2 URL Generator does not support private bots. Construct the URL manually:
   170|
   171|```text
   172|https://discord.com/oauth2/authorize?client_id=CLIENT_ID&permissions=PERMISSIONS&integration_type=0&scope=bot+applications.commands
   173|```
   174|
   175|Replace `CLIENT_ID` with the Application ID (found on General Information tab).
   176|Replace `PERMISSIONS`:
   177|- `117760` for text-only (Read/Send/Embed/Attach/History)
   178|- `7457792` for text + voice (adds Connect/Speak/Use Voice Activity)
   179|
   180|### Step 5: Create the .env file
   181|
   182|```bash
   183|cat > /home/kensei/.hermes/profiles/<profile>/.env << ENVEOF
   184|DISCORD_BOT_TOKEN=<token>
   185|DISCORD_ALLOWED_USERS=797682085224513547
   186|ENVEOF
   187|chmod 600 /home/kensei/.hermes/profiles/<profile>/.env
   188|```
   189|
   190|### Step 6: Create the systemd service
   191|
   192|Each specialist gateway gets its own service file pointing to its profile directory via `HERMES_HOME`.
   193|
   194|Reference service:
   195|
   196|```ini
   197|[Unit]
   198|Description=Hermes Gateway – <persona> (<purpose>)
   199|After=network-online.target hermes-gateway.service
   200|Wants=network-online.target
   201|
   202|[Service]
   203|Type=simple
   204|User=kensei
   205|Group=kensei
   206|Environment="HOME=/home/kensei"
   207|Environment="HERMES_HOME=/home/kensei/.hermes/profiles/<profile>"
   208|Environment="PATH=/home/kensei/.hermes/hermes-agent/venv/bin:/home/kensei/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
   209|Environment="VIRTUAL_ENV=/home/kensei/.hermes/hermes-agent/venv"
   210|ExecStart=/home/kensei/.hermes/hermes-agent/venv/bin/python -m hermes_cli.main gateway run --replace
   211|Restart=always
   212|RestartSec=10
   213|RestartMaxDelaySec=120
   214|RestartSteps=5
   215|KillMode=mixed
   216|KillSignal=SIGTERM
   217|TimeoutStopSec=90
   218|
   219|# Specialist gateway memory budget: ~130MB baseline, cap at 1G
   220|MemoryMax=1G
   221|MemoryHigh=800M
   222|
   223|[Install]
   224|WantedBy=multi-user.target
   225|```
   226|
   227|**Important:** The service should `After=network-online.target hermes-gateway.service` but NOT `Requires=` the main gateway. They run independently.
   228|
   229|### Step 7: Configure lean operation
   230|
   231|Specialist gateways MUST NOT run cron jobs or spawn MCP servers:
   232|
   233|```yaml
   234|# In profile's config.yaml, strip these:
   235|# Remove mcp_servers key entirely
   236|# Remove or disable the cron ticker by disabling cron/jobs.json entries
   237|
   238|# In profile's cron/jobs.json, disable all jobs:
   239|for j in data.get('jobs', []):
   240|    if j.get('enabled'):
   241|        j['enabled'] = False
   242|```
   243|
   244|Cron jobs live only on Kensei's main gateway. Specialist gateways handle interactive chat only.
   245|
   246|## Memory budgeting
   247|
   248|| Gateway type | Baseline RSS | Cap | Count | Total |
   249||---|---|---|---|---|---|
   250|| Kensei (main) | ~230MB | 5G | 1 | ~230MB |
   251|| Specialist text | ~130MB each | 1G | 6-7 | ~800MB-1GB |
   252|| **Total** | | | 7-8 | **~1-1.5GB** |
   253|
   254|On an 8GB VPS with current load (~4-5GB baseline), 7 gateways fit comfortably. The kanban dispatcher should remain disabled in-gateway (`kanban.dispatch_in_gateway: false`).
   255|
   256|## Co-working patterns
   257|
   258|When multiple bots exist in the same Discord channel, Sahil can address them by name:
   259|
   260|```text
   261|Sahil: Remii, what's new in AI tools? And Wesker, can we run that?
   262|
   263|Remii Bot: Three new ones this week...
   264|Wesker Bot: We've got ~2GB headroom on the VPS...
   265|```
   266|
   267|Each bot maintains its own session state, memory, and reasoning context. They are independent conversational entities that happen to share a channel.
   268|
   269|### allow_bots: controls whether bots see each other's messages
   270|
   271|The `DISCORD_ALLOW_BOTS` env var (or `discord.extra.allow_bots`) controls message intake for bot-authored messages:
   272|
   273|| Value | Behaviour | Use case |
   274||---|---|---|
   275|| `none` (default) | All bot messages silently dropped | Tight isolation — bots never see each other |
   276|| `mentions` | Only process bot messages where this bot is `@mentioned` | **Recommended for multi-bot setups.** Enables explicit handoffs (e.g. Kensei `@Dezzy please review`) while preventing cross-talk pollution |
   277|| `all` | Process all bot messages unrestricted | Debugging only — floods every bot's intake |
   278|
   279|Set in profile's `config.yaml`:
   280|
   281|```yaml
   282|discord:
   283|  extra:
   284|    gateway_restart_notification: false
   285|    allow_bots: mentions
   286|```
   287|
   288|**How it works** (adapter.py):
   289|1. **Intake filter** (line 797-812): Bot messages hit `DISCORD_ALLOW_BOTS` before anything else. `none` → `return`. `mentions` → only proceed if `self._client.user` is in `message.mentions`.
   290|2. **Multi-agent filter** (line 828-862): After intake, messages that mention OTHER bots but not THIS bot are dropped — prevents cross-consumption in co-working channels.
   291|3. **History backfill** (line 3908-3957): `mentions` is treated as `all` here — context assembly is about information, not gating.
   292|
   293|**Important caveat**: `allow_bots: mentions` still requires an actual `@mention` Discord ping. Plain text "Dezzy — please review" without `@Dezzy` produces empty `message.mentions`, so `allow_bots: mentions` never triggers. Use `free_response_channels` for the bot's home channel to handle text-only references. For co-working channels, always use proper `@mentions`.
   294|
   295|**Pitfall: this controls intake, not output.** `allow_bots: mentions` on Dezzy means Dezzy processes messages FROM other bots. It does NOT affect whether other bots see Dezzy's messages — that's controlled by each bot's own `allow_bots` setting. Set it on every bot that should be reachable for handoffs.
   296|
   297|### Bot-to-bot work flow
   298|
   299|When a non-coding lead (Remii, Dezzy, etc.) produces output needing implementation:
   300|1. **PREFERRED**: Post findings to channel + file kanban task assigned to execution lead
   301|2. **FALLBACK**: @mention the execution lead in-channel (urgent or tightly-scoped work)
   302|3. **NEVER**: Expect plain-text name references to trigger responses — text "Dezzy — ..." without @mention is silently dropped outside `free_response_channels`
   303|
   304|### Co-working channel mapping (as of 2026-05-24)
   305|
   306|| Channel | Free-response bots | @mention-only bots |
   307||---|---|---|
   308|| `#general`, `#cron-outputs`, `#kanban`, `#governance`, `#approvals`, `#decisions` | Kensei | All others |
   309|| `#ops` | Kensei, Wesker | All others |
   310|| `#war-room` | Kensei | All others |
   311|| `#design-review` | Dezzy | All others |
   312|| `#misa-misa` | Misa-Misa | All others |
   313|| `#research-digest` | Remii | All others |
   314|| `#research-ops` | Remii, Wesker | All others |
   315|| `#knowledge` | — | All bots |
   316|| `#mailbox__calendar`, `#job-hunt` | Gojo | All others |
   317|| `#build-log`, `#build-review` | Octacon | All others |
   318|| `#content` | CeeCee | All others |
   319|| `#ai-learning-qa` | MrHermagi | All others |
   320|
   321|Recommended multi-bot channels:
   322|- `#war-room` / `#general`: Kensei + all specialists
   323|- `#research-ops`: Remii + Wesker (infra+research overlap)
   324|- `#build-review`: Octacon + Quan (coding+QA)
   325|- `#content-review`: CeeCee + Kensei (content approval)
   326|
   327|## Channel structure by workflow, not persona
   328|
   329|Default channel-to-persona mapping:
   330|
   331|| Channel | Default persona(s) |
   332||---|---|
   333|| `#general` / `#ops` | Kensei, Wesker |
   334|| `#governance` | Kensei, Denji |
   335|| `#job-hunt` | Gojo |
   336|| `#plenishd` / `#coachsense` | Octacon, CeeCee |
   337|| `#research-digest` / `#research-ops` | Remii, Wesker |
   338|| `#build-log` / `#build-review` | Octacon |
   339|| `#mrhermagi-lessons` | MrHermagi |
   340|| `#approvals` | Kensei only (signed decisions) |
   341|
   342|## Privileged intents setup and debugging
   343|
   344|Privileged intents are the **most common issue** when setting up new Discord bots. Every new bot must have these enabled. Without them, the gateway fails with `discord.errors.PrivilegedIntentsRequired` and retries indefinitely.
   345|
   346|### Setup
   347|
   348|1. Go to Developer Portal → bot's application → **Bot** tab
   349|2. Scroll to **Privileged Gateway Intents**
   350|3. Toggle ON **MESSAGE CONTENT INTENT** and **SERVER MEMBERS INTENT**
   351|4. Click **Save Changes**
   352|5. Restart the gateway: `sudo systemctl restart hermes-gateway-<bot>`
   353|
   354|### Portal quirk: intents not persisting (most common debugging scenario)
   355|
   356|Discord's Developer Portal sometimes does NOT persist intents on first save even though the toggle shows ON. This happens often when creating multiple bots in rapid succession — the portal's UI lies to you.
   357|
   358|If a bot still fails with `PrivilegedIntentsRequired` after the user says "they're saved":
   359|
   360|1. **Verify you're editing the correct application.** When 6+ bots were created in rapid succession, the user can accidentally toggle intents on the wrong app. Ask them to check the application name at the top of the Bot tab. Cross-reference the Application ID from the portal against the one in the profile's gateway log.
   361|
   362|2. **Force-refresh the intents state.** Toggle both intents OFF → Save → toggle both ON → Save again. This force-flushes Discord's stale intent cache on their side. A gateway restart is still needed afterward.
   363|
   364|3. **Confirm the correct process restarted.** Check the PID and start time:
   365|   ```bash
   366|   systemctl show -p MainPID -p ActiveEnterTimestamp --value hermes-gateway-<bot>
   367|   ```
   368|
   369|4. **Read the right log file.** See diagnostic note below.
   370|
   371|### Diagnostic: gateway.log vs journalctl
   372|
   373|This is a critical distinction that causes repeated misdiagnosis.
   374|
   375|When a gateway restarts, `journalctl -u hermes-gateway-<bot>` may still show old errors from the **killed** process while the new process is sitting quietly waiting for retry. Journalctl merges both old and new PID output into one stream with no visual separation.
   376|
   377|**Always check the profile's own gateway.log for current connection state:**
   378|
   379|```bash
   380|tail -15 /home/kensei/.hermes/profiles/<profile>/logs/gateway.log
   381|```
   382|
   383|The gateway.log is written per-process-session. It will contain either:
   384|- `Connected as BotName#1234` (success)
   385|- `discord.errors.PrivilegedIntentsRequired` (failure)
   386|- `discord connect timed out after 30s` (failure)
   387|
   388|These three lines tell you the real state. Ignore journalctl noise for intent diagnosis.ntentsRequired` (failure)
   389|- `discord connect timed out after 30s` (failure)
   390|
   391|These three lines tell you the real state. Ignore journalctl noise for intent diagnosis.
   392|
   393|## Gateway lifecycle notification suppression
   394|
   395|When a Hermes gateway shuts down or restarts, it broadcasts `⚠️ Gateway shutting down — Your current task will be interrupted.` to **every active session** (every channel/thread with a conversation) PLUS **every platform home channel**. This hits forums, text channels, DMs — everywhere.
   396|
   397|The spread is controlled by `gateway_restart_notification: bool` on a per-platform basis in each gateway's `config.yaml`.
   398|
   399|### Default behaviour
   400|
   401|The `PlatformConfig` dataclass defaults `gateway_restart_notification = True` (see `gateway/config.py` line 299). Four code paths respect this flag:
   402|
   403|| Code path | File | Line | What it sends |
   404||-----------|------|------|--------------|
   405|| `_notify_active_sessions_of_shutdown()` | `gateway/run.py` | 3358 | `⚠️ Gateway shutting down` to each platform's **home channel only** (single-loop, no per-channel flood) |
   406|| `_send_home_channel_startup_notifications()` | `gateway/run.py` | 14387 | `♻️ Gateway online` to all home channels |
   407|| `_send_restart_notification()` | `gateway/run.py` | 14324 | `♻ Gateway restarted successfully` to the chat that issued `/restart` |
   408|
   409|**Key insight:** The shutdown notification must be **home-channel-only**. In KenseiAgent, `_notify_active_sessions_of_shutdown()` was refactored (2026-05-25) to remove the per-active-session broadcast loop. Notifications route exclusively to each platform's configured home channel. If you have active agents in 5 Discord channels, you get 1 notification (to the home channel) instead of 5. See `references/gateway-notification-routing.md` for the full internal architecture.
   410|
   411|**Deployment drift check:** services may run `/home/kensei/.hermes/hermes-agent/...` while the active development checkout is `/home/kensei/repos/KenseiAgent/...`. Always inspect the service checkout, not just the repo checkout:
   412|
   413|```bash
   414|python3 - <<'PY'
   415|from pathlib import Path
   416|for p in ['/home/kensei/.hermes/hermes-agent/gateway/run.py','/home/kensei/repos/KenseiAgent/gateway/run.py']:
   417|    text=Path(p).read_text(); s=text.index('    async def _notify_active_sessions_of_shutdown'); e=text.index('    def _finalize_shutdown_agents', s); body=text[s:e]
   418|    print(p, 'active-session send present?', 'Sent shutdown notification to active chat' in body)
   419|PY
   420|```
   421|
   422|If the service checkout still contains `Sent shutdown notification to active chat`, it will cascade shutdown pings across active Discord channels even if the repo checkout looks fixed. Patch/sync the service checkout and restart the gateway only after approval.
   423|
   424|### How to suppress the flood
   425|
   426|Add `gateway_restart_notification: false` to the Discord platform's `extra:` block in `config.yaml`:
   427|
   428|```yaml
   429|discord:
   430|  # ... other config ...
   431|  extra:
   432|    gateway_restart_notification: false
   433|```
   434|
   435|### 4.3 Do not let every specialist gateway own cron
   436|
   437|Cron jobs belong to the root Kensei gateway unless explicitly designed otherwise.
   438|
   439|Hard guard specialist gateways at BOTH levels:
   440|
   441|1. Profile config:
   442|
   443|```yaml
   444|cron:
   445|  ticker_enabled: false
   446|kanban:
   447|  dispatch_in_gateway: false
   448|```
   449|
   450|2. systemd service environment:
   451|
   452|```ini
   453|Environment="HERMES_CRON_TICKER_DISABLED=1"
   454|Environment="HERMES_KANBAN_DISPATCH_IN_GATEWAY=false"
   455|```
   456|
   457|Hermes `gateway/run.py` supports `HERMES_CRON_TICKER_DISABLED=1` and `cron.ticker_enabled: false`; without this guard older gateways start a cron ticker even when profile-local `jobs.json` is empty.
   458|
   459|Check specialist profile cron files:
   460|
   461|```bash
   462|find ~/.hermes/profiles -path '*/cron/jobs.json' -print
   463|```
   464|
   465|If a specialist has profile-local cron jobs, either:
   466|- migrate them into root cron with an explicit `profile: <lead>` field, or
   467|- pause them with a reason saying root cron is scheduler-of-record.
   468|
   469|Verification after restart:
   470|
   471|```bash
   472|for p in ceecee denji gojo light misa-misa mrhermagi octacon quan remii wesker; do
   473|  tail -40 ~/.hermes/profiles/$p/logs/gateway.log | grep -E 'Cron ticker|kanban dispatcher|Connected as'
   474|done
   475|```
   476|
   477|Expected specialist lines:
   478|
   479|```text
   480|Cron ticker disabled for this gateway
   481|kanban dispatcher: disabled via HERMES_KANBAN_DISPATCH_IN_GATEWAY env
   482|```
   483|
   484|Systemd webhook notifications via `ExecStopPost`/`ExecStartPost` are **not recommended** for multi-bot setups — they fire once per gateway restart and can't distinguish between intentional maintenance restarts and crash loops. The simpler approach is:
   485|1. Set `true` on Kensei-root → notifications go to `#general` only
   486|2. Set `false` on all specialists → silent
   487|3. If you need more granular alerts, monitor the gateway logs instead
   488|
   489|### Pitfalls
   490|
   491|- **The flag silences per-platform, not per-channel.** Setting `true` on Kensei with active sessions in 5 Discord channels still sends 5 notifications (one per channel). Setting `false` on specialists silences them entirely. For true single-channel notifications, combine Kensei-root `true` + all specialists `false`, understanding that Kensei-root may still multi-cast if it has active agents in multiple channels.
   492|- **Profile configs may not have an `extra:` block.** Add it inside the `discord:` section — the parser bridges it correctly.
   493|- **Gateway restart required.** The change only takes effect after restarting the gateway process.
   494|
   495|## Channel management via REST API
   496|
   497|Kensei's bot token can manage Discord channels programmatically using the Discord REST API.
   498|
   499|### Delete a channel
   500|
   501|
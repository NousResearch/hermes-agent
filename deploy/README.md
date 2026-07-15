# Morning-Chain Deploy

Automates the M5 morning workflow (export todo keys → journal → ingest journal
contract → perf-coach brief → hermes delivery) via a launchd job that fires at
**05:45 Asia/Bangkok (UTC+7)** = **22:45 UTC** the previous calendar day.

## Files

| File | Purpose |
|---|---|
| `com.hermes.morning-chain.plist` | launchd job definition |
| `bin/morning-chain.sh` | five-step chain script with flock + kill switch |

**Steps:** (1) `todo_store_sync export` — reopen expired snoozes, write
OPEN_KEYS/CLOSED_KEYS for journal; (2) journal's morning run; (3)
`todo_store_sync ingest` — reconcile today's journal contract (todos +
resolved_keys) back into the persistent todo store; (4) perf-coach brief
export; (5) hermes brief compose/deliver. See
`services/hermes/todo_store_sync.py` for why the export step runs *before*
journal rather than after it.

---

## 1. Wake the Mac before 05:45

launchd fires at the scheduled time only if the Mac is already awake. Use
`pmset` to schedule a hardware wake **before** 05:45 Bangkok time (22:45 UTC):

```bash
# Wake at 22:40 UTC (= 05:40 Bangkok) every day — 5 minutes before the chain
sudo pmset repeat wakeorpoweron MTWRFSU 22:40:00
```

To confirm the schedule:

```bash
pmset -g sched
```

To cancel:

```bash
sudo pmset repeat cancel
```

> `wakeorpoweron` wakes from sleep or powers on from off. Use `wake` if you
> prefer to wake-only (machine must already be powered). On Apple Silicon,
> `wakeorpoweron` is the more reliable form.

---

## 2. Set CLAUDE_CODE_OAUTH_TOKEN for non-interactive auth

The chain runs headlessly under launchd, where your shell profile is not
sourced. Claude Code needs an OAuth token injected via the launchd
`EnvironmentVariables` dict — the standard `~/.env` file is not read in this
context.

**Steps:**

1. Obtain your token:
   ```bash
   cat ~/.claude/.credentials.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('claudeAiOauth',{}).get('accessToken',''))"
   ```
   Or find it via the Claude Code desktop app under Settings → Developer.

2. Open `deploy/com.hermes.morning-chain.plist` and replace
   `REPLACE_WITH_YOUR_TOKEN` with the real token value:

   ```xml
   <key>CLAUDE_CODE_OAUTH_TOKEN</key>
   <string>YOUR_ACTUAL_TOKEN_HERE</string>
   ```

3. Also replace every occurrence of `HERMES_REPO` with the absolute path to
   this repository, e.g. `/Users/you/projects/hermes-agent`.

> Keep the plist out of version control after editing if it contains the real
> token value. Alternatively, source the token from a file by wrapping the
> ProgramArguments in a shell script that reads `~/.claude/.credentials.json`.

---

## 3. Install / uninstall with launchctl

### Install

```bash
# Edit the plist first — replace HERMES_REPO and CLAUDE_CODE_OAUTH_TOKEN
launchctl load ~/Library/LaunchAgents/com.hermes.morning-chain.plist
```

Or, without copying to `~/Library/LaunchAgents/`:

```bash
launchctl load "$(pwd)/deploy/com.hermes.morning-chain.plist"
```

Verify registration:

```bash
launchctl list | grep hermes.morning-chain
```

You should see a line like:

```
-    0    com.hermes.morning-chain
```

(The `-` in column 1 means the job is not currently running; `0` is the last
exit code.)

### Run manually (test without waiting for 22:45 UTC)

```bash
launchctl start com.hermes.morning-chain
```

Or call the script directly:

```bash
./deploy/bin/morning-chain.sh --dry-run   # prints commands, no execution
./deploy/bin/morning-chain.sh             # full run (requires upstream repos)
```

### Uninstall

```bash
launchctl unload ~/Library/LaunchAgents/com.hermes.morning-chain.plist
# or, if loaded from the repo path:
launchctl unload "$(pwd)/deploy/com.hermes.morning-chain.plist"
```

---

## 4. Kill switch

Create `deploy/.morning-chain-disabled` to prevent the chain from running
without unloading the launchd job:

```bash
touch deploy/.morning-chain-disabled    # disable
rm    deploy/.morning-chain-disabled    # re-enable
```

The script exits 0 silently when this file is present, so launchd considers
the job successful and will not retry or log an error.

---

## 5. Validate the plist

```bash
plutil -lint deploy/com.hermes.morning-chain.plist
```

Expected output: `deploy/com.hermes.morning-chain.plist: OK`

---

## 6. Logs

Each run appends timestamped output to:

```
logs/morning-chain-YYYY-MM-DD.log
```

Tail the current day's log:

```bash
tail -f logs/morning-chain-$(date +%Y-%m-%d).log
```

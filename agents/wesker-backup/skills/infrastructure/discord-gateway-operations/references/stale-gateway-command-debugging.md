# Stale Gateway Command — Slash Command Visible but No Response

## Scenario

A new slash command (e.g. `/onboard`) has been:
1. Added to `hermes_cli/commands.py` as a `CommandDef`
2. Registered as a Discord `@tree.command` in the adapter
3. Given a handler in `gateway/run.py`

The command **appears in Discord's slash picker** — autocomplete shows it, the user can select it and fill in options — but when executed, nothing happens. No response, no error, no "thinking..." indicator.

The bot responds to other slash commands normally.

## Root cause

The command was **auto-registered** by a previous gateway startup that loaded the new `CommandDef` from `commands.py` (which happens at import time), but the **handler code** (`_handle_onboard_command` + the `if canonical == "onboard":` dispatch entry) was added later and doesn't exist in the running process.

Discord's `tree.sync()` ran on the old startup and registered the auto-generated handler (which proxies through `_run_simple_slash`), but when that proxy sends the command text to `_handle_message`, the gateway's dispatch chain at lines 7280-7345 doesn't have the `if canonical == "onboard":` branch yet — so the text falls through to the running-agent catch-all and gets silently consumed.

## Detection

```bash
# 1. Is the gateway running old code?
PID=$(cat ~/.hermes/profiles/<profile>/gateway.lock | python3 -c "import sys,json; print(json.load(sys.stdin).get('pid',0))")
MODULE_MTIME=$(stat --format=%Y /home/kensei/.hermes/hermes-agent/gateway/run.py)
PID_START=$(ps -o lstart= -p $PID 2>/dev/null | xargs -I{} date -d "{}" +%s)

if [ "$PID_START" -lt "$MODULE_MTIME" ]; then
    echo "STALE: gateway started before module was last modified"
    echo "  Gateway PID $PID started: $(date -d @$PID_START)"
    echo "  run.py modified:         $(date -d @$MODULE_MTIME)"
else
    echo "FRESH: gateway started after module changes"
fi
```

## Detection — command fingerprint

When a gateway starts, it computes a SHA-256 fingerprint of the entire command tree and skips the Discord sync if the fingerprint matches a previous successful sync. You can check whether the newly registered command was actually synced:

```bash
# Check the sync state
cat ~/.hermes/profiles/<profile>/gateway/discord_command_sync_state.json | python3 -m json.tool

# Check gateway logs for sync activity
grep -i "synced\|safe.*reconciled\|fingerprint" ~/.hermes/profiles/<profile>/logs/gateway.log | tail -5
```

If you see `same slash-command fingerprint already synced` — the gateway is skipping sync because it thinks the tree hasn't changed. But if the handler code changed (not the command definition), the fingerprint check passes because it only hashes the Discord command payload, not the Python handler.

## Fix

### Step 1 — Clear the sync fingerprint cache

Without this, the gateway will skip its sync on restart because it sees a matching fingerprint:

```bash
rm ~/.hermes/profiles/<profile>/gateway/discord_command_sync_state.json
```

### Step 2 — Kill the stale gateway process

```bash
PID=$(cat ~/.hermes/profiles/<profile>/gateway.lock | python3 -c "import sys,json; print(json.load(sys.stdin).get('pid',0))")
kill $PID 2>/dev/null
sleep 2
kill -0 $PID 2>/dev/null && kill -9 $PID
```

### Step 3 — Start a fresh gateway

```bash
cd /home/kensei/.hermes/hermes-agent && hermes -p <profile> gateway run --replace
```

The new process:
1. Imports the updated `gateway/run.py` with the new handler + dispatch
2. Imports the updated adapter with the new `@tree.command` handler
3. Computes a new fingerprint (different because the options or handler changed)
4. Syncs the full tree with Discord
5. Logs: `Safely reconciled N slash command(s): unchanged=X created=1 deleted=0`

### Step 4 — Verify

Check the gateway log for the sync confirmation:

```bash
grep "Safely reconciled\|created=1" ~/.hermes/profiles/<profile>/logs/gateway.log
```

Expected output shows `created=1` for the new command.

## Why this happens

The `_register_slash_commands()` method in `plugins/platforms/discord/adapter.py` has two registration paths:

1. **Manual `@tree.command(name="onboard")`** (line 3240) — registers the command with specific options (force: bool, resume: bool) and a dedicated `slash_onboard` handler that sends `"/onboard --force --resume"` to `_run_simple_slash`.

2. **Auto-registration** (line 3299-3317) — scans `COMMAND_REGISTRY` and auto-registers any command not yet on the tree. This creates a generic handler that sends `"/onboard [args]"` via `_run_simple_slash`.

Path 1 wins over path 2 because path 2 checks `already_registered` and skips duplicates. But **both paths** only work if the running process loaded the code that contains them.

When a gateway is started **before** the code is written, neither path exists. But the auto-registration loop in path 2 reads from `COMMAND_REGISTRY` — which **is** a data import (not a function), so even old code reads the new CommandDef. This creates the deceptive situation where the command appears in Discord's picker but the handler doesn't exist.

## Pitfalls

- **The sync fingerprint caches across gateway restarts.** Deleting the state file is the only way to force a re-sync if no command-tree payload changed. Just restarting with `--replace` isn't enough.
- **Multiple gateways, multiple states.** Each bot profile (`dezzy`, `remii`, etc.) has its own sync state file at `~/.hermes/profiles/<profile>/gateway/discord_command_sync_state.json`. Clearing one doesn't affect others.
- **The auto-sync check in adapter.py line 1002-1012** checks two things: (1) rate-limit time hasn't elapsed, and (2) the fingerprint matches the last successful sync. Deleting the state file resets both — so you also lose the rate-limit protection. If Discord is under load, you might hit a 429. The gateway handles this gracefully (retries with exponential backoff logged at WARNING level), but it adds ~60s to the sync.
- **The `--sync` flag doesn't exist on `gateway run`.** There's no way to force a sync from the CLI; you must clear the state file.
- **`hermes -p <profile> gateway run --replace` doesn't guarantee a restart if the old process is already dead.** `--replace` only kills a running process matching the lock file's PID. If the old process already exited, `--replace` is a no-op and the old code stays loaded. You must start a new process explicitly.

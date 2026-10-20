# Telegram polling conflicts with Hermes gateway

Use this when Hermes Telegram is configured but the bot does not respond, especially when logs contain:

- `Telegram polling conflict`
- `terminated by other getUpdates request`
- `make sure that only one bot instance is running`

## Diagnosis sequence

1. Check Hermes service and platform config:
   ```bash
   hermes gateway status
   hermes status --all
   ```
   Expect `Telegram ✓ configured` and gateway service running.

2. Check secret presence without printing token values:
   ```bash
   python3 - <<'PY'
   from pathlib import Path
   import re, subprocess
   env = Path(subprocess.run(['hermes','config','env-path'], capture_output=True, text=True).stdout.strip() or Path.home()/'.hermes/.env')
   text = env.read_text(errors='ignore') if env.exists() else ''
   for k in ['TELEGRAM_BOT_TOKEN','TELEGRAM_TOKEN','BOT_TOKEN']:
       m = re.search(rf'^{k}=(.*)$', text, re.M)
       print(k, 'present' if m and m.group(1).strip() else 'missing')
   PY
   ```

3. Inspect logs with redaction:
   ```bash
   python3 - <<'PY'
   from pathlib import Path
   import re
   p=Path.home()/'.hermes/logs/gateway.log'
   for l in p.read_text(errors='ignore').splitlines()[-120:]:
       if re.search(r'telegram|conflict|error|connected|disconnected|poll', l, re.I):
           l=re.sub(r'\b\d{6,}:[A-Za-z0-9_-]+\b','<telegram-token>',l)
           print(l[-500:])
   PY
   ```

4. Confirm whether the token is valid and whether another poller exists. Stop Hermes briefly, then call `getUpdates`; if it still returns 409 conflict, another process/service/machine is polling the same bot token.
   ```bash
   hermes gateway stop
   # call Telegram getMe/getWebhookInfo/getUpdates via a script that reads TELEGRAM_BOT_TOKEN from ~/.hermes/.env and never prints it
   hermes gateway start
   ```

## Common local culprit

Claude Code official Telegram plugin can run as:

```text
bun run --cwd ~/.claude/plugins/cache/claude-plugins-official/telegram/<version> start
```

It may not show in `claude mcp list` if orphaned, but it can still poll Telegram. Search with:

```bash
ps auxww | egrep -i 'claude-plugins-official/telegram|bun run.*telegram|telegram|gateway|hermes|node' | egrep -v 'egrep|Telegram.app'
```

## Safe disable/removal pattern

Prefer CLI uninstall first:

```bash
for scope in user project local; do
  claude plugin disable -s "$scope" telegram@claude-plugins-official || true
  claude plugin uninstall -s "$scope" telegram@claude-plugins-official || true
done
```

If Claude reports it is not installed/enabled but files/processes remain, quarantine the plugin directories instead of deleting:

```bash
stamp=$(date +%Y%m%d-%H%M%S)
mv ~/.claude/plugins/cache/claude-plugins-official/telegram \
   ~/.claude/plugins/cache/claude-plugins-official/telegram.disabled-by-hermes-$stamp 2>/dev/null || true
mv ~/.claude/plugins/marketplaces/claude-plugins-official/external_plugins/telegram \
   ~/.claude/plugins/marketplaces/claude-plugins-official/external_plugins/telegram.disabled-by-hermes-$stamp 2>/dev/null || true
```

Then stop any still-running duplicate Telegram plugin process, restart Hermes gateway, and verify:

```bash
hermes gateway restart
sleep 10
hermes gateway status
# Inspect recent gateway.log and ensure conflicts_after_last_connect == 0
```

## Verification criteria

- `hermes gateway status` shows a running PID.
- Recent `gateway.log` shows `Connected to Telegram (polling mode)` and `✓ telegram connected`.
- There are zero `Telegram polling conflict` entries after the latest connect.
- A test Telegram DM reaches Hermes; logs show `inbound message: platform=telegram`.

## If conflict persists

If `getUpdates` returns 409 while the local Hermes gateway is stopped and no local duplicate process exists, another machine/service has the bot token or the token is compromised. Rotate the Telegram bot token in BotFather and update `TELEGRAM_BOT_TOKEN` in Hermes `.env`, then restart Hermes gateway.

# WhatsApp Bridge — Session Reference

## Bridge Status (May 2026)

**Session store:** `/home/<username>/.hermes/whatsapp/session/`
**Log file:** `~/.hermes/whatsapp-bridge.log`
**Bridge binary:** `~/.hermes/hermes-agent/scripts/whatsapp-bridge/bridge.js`

**Currently running in:** `bot` mode on port `3001`

## How Pairing Works

1. Run `hermes whatsapp` in a separate terminal (requires interactive TTY)
2. Scan QR with WhatsApp → Settings → Linked Devices → Link a Device
3. Session persists in `~/.hermes/whatsapp/session/` — no re-scan needed on restart
4. Verify paired: `ls ~/.hermes/whatsapp/session/` — should contain auth files, not just empty dir

**If session is empty after scan:** The bridge may not have saved the session properly. Kill and restart the bridge after scanning.

## Sending Messages

**Personal chat:**
```bash
curl -s -X POST http://localhost:3001/send \
  -H "Content-Type: application/json" \
  -d '{"chatId": "<your-number>@c.us", "message": "Hello"}'
```

**WhatsApp group:** Groups must be discovered by:
1. Switching to `bot` mode
2. Having someone send a message to the group
3. Reading from `http://localhost:3001/messages?timeout=2` to get the group JID

## Project-Specific Usage

- **Project path:** `/mnt/c/Users/<username>/<your-project-name>/`
- **Config:** `prospects/config.json` — stores WhatsApp number
- **Cron delivery:** `whatsapp:+<number>` (set via cronjob tool `deliver` param)

## Known Issues

- **WSL Chrome/Puppeteer doesn't work** — cannot launch Windows Chrome from WSL (error code 21). Use Hermes native `web_search` tool instead for scraping.
- **Bridge not listening on expected port:** Check `ss -tlnp` or `netstat -tlnp` to find the actual port.
- **Bot mode groups not working:** Ensure bridge was started with `--mode bot`, not default `self-chat`.

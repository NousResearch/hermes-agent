---
name: wsl-hermes-setup
description: "Run Hermes on WSL2: systemd, gateway, common fixes"
version: 0.2.0
platforms: [linux, wsl]
metadata:
  hermes:
    tags: [wsl, wsl2, setup, configuration, gateway, systemd, telegram]
---

# WSL2 Hermes Setup

Hermes runs on WSL2 but has a handful of non-obvious requirements. This skill covers the full setup path and the most common failure modes.

## Critical First Step: Systemd

Without systemd, the gateway dies every time you close your terminal. Do this before anything else.

**Add to `/etc/wsl.conf`** (INI format, not YAML):

```ini
[boot]
systemd=true
```

**Then restart WSL from Windows PowerShell:**

```powershell
wsl --shutdown
```

Reopen your WSL terminal. Verify systemd is running:

```bash
systemctl --no-pager status
# Should show "State: running"
```

## Install Hermes

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
hermes doctor
```

## Telegram Gateway Setup

```bash
# Configure Telegram platform
hermes gateway setup

# Start gateway (foreground to verify it works)
hermes gateway run

# In a second terminal — approve your Telegram user ID
hermes pairing list
hermes pairing approve telegram <YOUR_TELEGRAM_ID>
```

Send a message to your bot in Telegram to confirm it responds.

**Install as a background service** (requires systemd from above):

```bash
hermes gateway install
hermes gateway start
hermes gateway status
```

## Common Issues

### Gateway dies when terminal closes

**Symptom:** Gateway works while terminal is open, disappears when you close it.

**Cause:** systemd not enabled — gateway falls back to `nohup` which dies with the session.

**Fix:** Enable systemd (see Critical First Step above), then reinstall the service:

```bash
hermes gateway install
hermes gateway start
```

---

### No messages received in Telegram

**Symptom:** Gateway is running, bot is online, but no responses arrive.

**Fix:** Pairing approval is required after every fresh gateway install:

```bash
hermes pairing list
hermes pairing approve telegram <YOUR_TELEGRAM_ID>
```

---

### Snap interference (gio)

**Symptom:** `hermes dashboard` fails silently on WSL2 when Hermes was installed via Snap.

**Fix:** Bypass Snap's Gio layer with direct uvicorn invocation:

```bash
cd ~/.hermes/hermes-agent && \
env PATH="$HOME/.hermes/hermes-agent/venv/bin:$PATH" \
uvicorn hermes_cli.web_server:app --host 127.0.0.1 --port 9119 --log-level warning &

sleep 3 && curl -s http://127.0.0.1:9119/api/status | jq '.gateway_running'
# Should return true
```

Access dashboard at `http://localhost:9119`.

---

### HTTP 400 "No models provided" on first run

**Symptom:** Error on startup, nothing runs.

**Cause:** `config.yaml` was saved with a UTF-8 BOM (common when Windows apps write it).

**Fix:**

```bash
hermes config edit
# Re-save the file — any POSIX editor (nano, vim) will strip the BOM automatically
```

---

### Alt+Enter doesn't insert a newline

**Cause:** Windows Terminal intercepts Alt+Enter for fullscreen toggle before it reaches Hermes.

**Fix:** Use **Ctrl+Enter** instead.

---

### Local model not connecting (LM Studio / Ollama)

**Symptom:** Hermes can't reach your local model server running on Windows.

**Mistake 1 — wrong endpoint path:** The OpenAI client appends `/chat/completions` automatically:

```bash
# Wrong
hermes config set model.base_url "http://<host>:1234/v1/chat/completions"

# Correct
hermes config set model.base_url "http://<host>:1234/v1"
```

**Mistake 2 — using `localhost`:** In WSL2 NAT mode (the default), `localhost` resolves to the WSL VM, not the Windows host. If your model server is running on Windows, use the host IP instead:

```bash
# Get Windows host IP
ip route show default | awk '{print $3}'
# or
grep nameserver /etc/resolv.conf | awk '{print $2}'
```

Then set it:

```bash
hermes config set model.base_url "http://$(ip route show default | awk '{print $3}'):1234/v1"
```

**Exception:** WSL2 with [mirrored networking](https://learn.microsoft.com/en-us/windows/wsl/networking#mirrored-mode-networking) (Windows 11 22H2+) does forward `localhost` to the host — check with `wsl --version` and look for "Networking mode: mirrored" in `~/.wslconfig`.

Then restart Hermes.

---

## Configuration Migration

### TERMINAL_CWD (deprecated)

Remove from `.env`:
```bash
# Remove this line:
TERMINAL_CWD=/your/project/path
```

Add to `config.yaml`:
```yaml
terminal:
  cwd: /your/project/path
```

## Verification Checklist

Run these in order after setup:

```bash
# 1. Systemd running
systemctl --no-pager status | head -5

# 2. Hermes health
hermes doctor

# 3. Gateway running
hermes status --all

# 4. Telegram pairing
hermes pairing list

# 5. Gateway logs (if something is off)
tail -f ~/.hermes/logs/gateway.log
```

## Best Practices

- **Always enable systemd first** — everything else depends on it
- **Re-approve Telegram pairing** after fresh gateway installs
- **Use forward slashes** in all paths — `~/myapp` not `~\myapp`
- **Check logs before assuming failure** — `tail -f ~/.hermes/logs/gateway.log`
- **Use `hermes config edit`** to write config — avoids UTF-8 BOM issues

## Next Steps

```bash
hermes skills browse        # explore available skills
hermes profile create dev   # isolated profile for a project
hermes cron list            # scheduled tasks
hermes mcp list             # connected MCP servers
```

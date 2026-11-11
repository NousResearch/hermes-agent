---
name: safe-live-service-changes
description: >
  Mandatory safety protocol before making any changes to live services, running
  processes, or production infrastructure (Pluto, Henry, Mission Control, routers,
  Ollama, systemd services). Use whenever modifying files, configs, or crons that
  touch a running service.
tags: [devops, safety, pluto, infrastructure, rollback, git]
---

# Safe Live Service Changes

## When to apply this skill

- Modifying any file that a running process reads (server.js, config.json, data.ts, .env)
- Creating crons that restart, kill, or modify running services
- Changing systemd unit files or starting/stopping services
- Any `pkill`, `kill`, or process restart on Pluto/Henry
- Modifying Mission Control, iblai-router, Ollama config, pluto-agent

---

## Mandatory pre-change checklist

Before touching anything:

1. **Verify current state works**
   ```bash
   curl -s -o /dev/null -w 'HTTP %{http_code}' http://127.0.0.1:3000
   # Document the result — is it 200? If not, note why.
   ```

2. **Create a Git branch** (if repo exists)
   ```bash
   cd /path/to/repo
   git checkout -b fix/description-$(date +%Y%m%d)
   git status
   ```

3. **Create a timestamped file backup** (always, even if git exists)
   ```bash
   cp target_file target_file.backup_$(date +%Y%m%d_%H%M%S)
   ```

4. **Note which PID / systemd unit owns the service**
   ```bash
   ps aux | grep service_name
   systemctl status service_name 2>/dev/null
   ```

---

## Cron safety rules

NEVER create a cron that:
- Does `pkill`, `kill -9`, or `killall` on a Next.js / node / Python service
- Does unconditional `pkill next dev` — this will kill the running process and the replacement may bind to a different port
- Restarts a service without first checking if it's actually down
- Writes to files that a running process hot-reloads (data.ts, config.json)

SAFE cron pattern:
```bash
# Check first, act only if down
if ! curl -sf http://127.0.0.1:PORT/health > /dev/null; then
  # Service is down — restart safe
  systemctl restart service_name || nohup node server.js &
fi
# Otherwise: do nothing, report only
```

---

## Change workflow

1. Make the change on the **backup copy** first, verify it looks right
2. Apply to real file
3. Test immediately (curl, HTTP check, log tail)
4. If test passes → commit to Git branch
5. Only merge to main / enable cron after confirmed working
6. If test fails → restore from backup immediately:
   ```bash
   cp target_file.backup_TIMESTAMP target_file
   ```

---

## Rollback commands (Pluto)

```bash
# Mission Control
pkill -f 'next dev'  # kill stray processes
cd ~/.openclaw/workspace/mission-control && nohup npm run dev > /tmp/mc.log 2>&1 &
curl -s -o /dev/null -w 'MC: HTTP %{http_code}\n' http://127.0.0.1:3000

# Restore a file from backup
cp file.backup_20260412_161336 file

# Systemd service (requires sudo — instruct user to run manually)
# sudo systemctl disable --now service_name
```

---

## Pitfalls learned (from incident 2026-04-12)

- **pkill next dev in a cron** killed the running Mission Control process; Next.js restarted on port 3001 instead of 3000, breaking the URL
- **Backup was made AFTER patches** — the "pre-patch" backup already contained changes. Always make backup BEFORE the first edit
- **sudo systemctl stop** hangs in non-PTY SSH sessions — cannot be done via Hermes terminal; must be done by user interactively or via a sudoers NOPASSWD entry for specific commands
- **iblai-router ran as root** via systemd — Hermes SSH user (sander) could not kill it
- **config.json was not in git** — no version history to restore from; always check `git ls-files` before assuming git covers a file

---

## Verification after any change

```bash
# Always end with a 3-point check:
curl -s -o /dev/null -w 'Service: HTTP %{http_code}\n' http://127.0.0.1:PORT
pgrep -a node | head -5           # confirm correct process running
tail -5 /tmp/service.log          # no errors in recent log
```

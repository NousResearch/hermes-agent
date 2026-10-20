# Troubleshooting

### Voice not working
1. Check `stt.enabled: true` in config.yaml
2. Verify provider: `python -c "import pm; pm.sync_venv(['stt-whisper'], explicit=True)"` or set API key
3. In gateway: `/restart`. In CLI: exit and relaunch.

### Tool not available
1. `hermes tools` — check if toolset is enabled for your platform
2. Some tools need env vars (check `.env`)
3. `/reset` after enabling tools

### Model/provider issues
1. `hermes doctor` — check config and dependencies
2. `hermes auth` — re-authenticate OAuth providers (or `hermes auth add <provider>`)
3. Check `.env` has the right API key
4. **Copilot 403**: `gh auth login` tokens do NOT work for Copilot API. You must use the Copilot-specific OAuth device code flow via `hermes model` → GitHub Copilot.

### Changes not taking effect
- **Tools/skills:** `/reset` starts a new session with updated toolset
- **Config changes:** In gateway: `/restart`. In CLI: exit and relaunch.
- **Code changes:** Restart the CLI or gateway process

### web_extract shows a stale page (result caching)
`web_search`/`web_extract` cache results for 20 minutes (PR #94618) — a
repeat fetch of the same URL within the TTL is served from cache, which
can look like "my website changes aren't showing up."

Automatic carveouts (always fetched live, never cached):
- localhost / 127.0.0.1 / `*.local` / `*.localhost` / single-label LAN
  hostnames / private + link-local IP ranges (dev servers, hot-reload
  builds, chat-GUI artifact previews)
- URLs matched by `security.website_blocklist`
- failed responses and keyless-rescue-served responses

Developing a site tested over the PUBLIC internet (Vercel/Netlify
preview, ngrok/cloudflared tunnel, staging domain)? Public DNS isn't
auto-carved-out — list the host in config.yaml:

```yaml
web:
  cache_exempt_hosts:      # always fetched live; effective immediately
    - mysite.vercel.app
    - "*.ngrok-free.app"
    - mysite.dev           # suffix match: also covers preview.mysite.dev
```

Blunt instruments: `web.cache_ttl_minutes: 1` (min) or
`web.cache_enabled: false` disables both caches entirely.

### Skills not showing
1. `hermes skills list` — verify installed
2. `hermes skills config` — check platform enablement
3. Load explicitly: `hermes -s name` (or the skill's own `/<name>` slash command)

### Gateway issues
Check logs first:
```bash
grep -i "failed to send\|error" ~/.hermes/logs/gateway.log | tail -20
```

Common gateway problems:
- **Gateway dies on SSH logout**: Enable linger: `sudo loginctl enable-linger $USER`
- **Gateway dies on WSL2 close**: WSL2 requires `systemd=true` in `/etc/wsl.conf` for systemd services to work. Without it, gateway falls back to `nohup` (dies when session closes).
- **Gateway crash loop**: Reset the failed state: `systemctl --user reset-failed hermes-gateway`

### Log and output forensics — four traps that produce confidently wrong numbers

While auditing Hermes logs and command output, four mistakes each produced a *wrong number that
looked plausible enough to report*. All four are silent — none raises an error.

1. **A multi-file `grep -h … | tail` is NOT chronological.** `-h` suppresses filenames and
   concatenates in **glob order**, so the last line is the last line of the last *file*, not the
   newest event. This produced a reported "last occurrence 31 Aug" when the true last was 19 Sep.
   Extract the timestamp and **sort it**:
   `grep -rh PATTERN FILES | sed -E 's/^([0-9-]+ [0-9:,]+).*/\1/' | sort | tail`
2. **Never `cut` a log line whose number you intend to quote.** `cut -c1-240` truncated
   `final_len=2534` to `final_len=3`, which was reported as a "3-character payload" and changed the
   diagnosis. Pull fields directly: `grep -oE 'final_len=[0-9]+'`.
3. **Mirror logs double-count.** `<profile>/logs/errors.log` mirrors the same events as
   `gateway.log`, so a naive multi-file count returned 24 where the unique count was **20**
   (counted per profile). Count **per file first**, then reconcile — and state the scope you counted.
4. **A truncated `ls … | head -N` is not evidence of absence.** `ls | head -25` cut a directory
   listing alphabetically, hiding every entry after the Nth — `scan.py` was declared "already
   pruned" on the strength of it and had never been pruned. When concluding a file is *gone*,
   count the set first (`ls -1 | wc -l`) or filter with a glob (`ls *.py`); never eyeball a capped
   listing. Same family: `head`/`tail` on any *filtered* list answers about the filter, not the set.

**Rule:** when another agent or the user quotes raw logs back at you, **re-derive the number before
defending it.** Most of the traps above were caught by someone re-reading the same source, not by
reasoning about it. Related: check whether a warning is a *diagnostic* before calling it a defect —
`Normal final-send NOT suppressed … possible duplicate send` (`gateway/run_turn.py`) logs the *risk
input*, not a confirmed duplicate.

### Hindsight dead in Bot Chats (but fine in CLI and gateway)
Bot-Mode deliveries spawn a **generation-bound** binary
(`…/installs/<id>/environments/<gen>/venv/bin/hermes`) instead of the installation-bound launcher
(`.hermes/bin/hermes`, see `hermes_cli/_launchers.py::runtime_command`) — so a dependency installed
after that generation (e.g. the `hindsight` plugin's `hindsight-client`) is invisible to every Bot
Chat, and **no** `/new` or session restart helps. Diagnose by diffing the two spawn paths; don't
reinstall packages. `spawn-ledger.json` is a **live process registry**, not the cause — editing it
is pointless (rewritten on the next delivery). Upstream fix: use the launcher.

### Cron: EVERY job fails before its ownership acknowledgement

One install-wide cause, and it reads like a scheduler problem rather than a runtime one: every job
across every profile records an error **before** its own script ever runs.

**Mechanism.** The worker spawns as `sys.executable -m cron.scheduler` (`cron/scheduler.py`), and
`sys.executable` is the **gateway's own interpreter** — not the venv. After a gateway restart onto a
managed runtime (`~/.hermes/tools/python-<ver>-<build>-<arch>/bin/python<ver>`),
`cron/scheduler_worker_env.py` drops the runtime's site-packages, so a package present in the venv is
invisible to it; the failure surfaces in `hermes_cli/env_loader.py` as
`ModuleNotFoundError: No module named '<pkg>'`.

**The trap: probing the venv wrongly clears a broken install.** `hermes-agent/venv/bin/python -c
"import dotenv"` succeeds and `hermes doctor` passes — neither is the interpreter the worker runs.
Probe the gateway's own executable instead:

```bash
GW=$(readlink -f /proc/$(pgrep -f 'hermes.*gateway' | head -1)/exe)   # the managed runtime
"$GW" -c "import dotenv"                                              # the real answer
```

**Stopgap:** install the missing package into that runtime
(`~/.hermes/tools/python-<ver>/bin/python<ver> -m pip install <pkg>`), fire a single job, and confirm
`last_status: error → ok` with `last_error: null`. A runtime **re-provision wipes it** — stopgap only.

**Durable:** upstream. Search the tracker for the open `comp/cron` thread with fix PRs before filing a
duplicate.

**Triage note:** a failure *before the ownership acknowledgement* means the job's script never ran — read
the traceback before editing the job definition.

### Platform-specific issues
- **Discord bot silent**: Must enable **Message Content Intent** in Bot → Privileged Gateway Intents.
- **Slack bot only works in DMs**: Must subscribe to `message.channels` event. Without it, the bot ignores public channels.
- **Windows-specific issues** (`Alt+Enter` newline, WinError 10106, UTF-8 BOM config, line endings): see `references/windows-quirks.md`.

### Auxiliary models not working
If `auxiliary` tasks (vision, compression, session_search) fail silently, the `auto` provider can't find a backend. Either set `OPENROUTER_API_KEY` or `GOOGLE_API_KEY`, or explicitly configure each auxiliary task's provider:
```bash
hermes config set auxiliary.vision.provider <your_provider>
hermes config set auxiliary.vision.model <model_name>
```

### "Reset permissions" / auto-approving everything
See `references/security-privacy.md` — wipe the "Always allow" stores, don't touch yolo mode.


# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace Purpose

This workspace is dedicated to **deploying and operating the Hermes Agent platform** on machine `myia-po-2026`. It is part of the Myia AI cluster managed by RooSync.

**Upstream:** [nousresearch/hermes-agent](https://github.com/nousresearch/hermes-agent) → fork [jsboige/hermes-agent](https://github.com/jsboige/hermes-agent)
**Our drift:** `roosync-cluster/` directory (ADR, scripts, docs) — isolated from upstream code
**Epic:** roo-extensions #1862 (Hermes Phase 2 — Bootstrap), parent #1864 (Cycle 26 — Cluster Expansion)

### Cluster context (RooSync)

| Workspace | Machine(s) | Role |
|-----------|------------|------|
| roo-extensions | ai-01, po-2023/24/25/26 | Code, MCPs, coordination |
| nanoclaw | web1 (ai-01 sessions) | Docker agents, IPC, ClusterManager Telegram bot |
| CoursIA | po-2023 | Training and courses |
| Argumentum | ai-01 | Logical argumentation |
| vllm / myia-open-webui | ai-01 | LLM inference |
| **hermes-agent** | **po-2026** | **Routing, audit, orchestration (THIS)** |

### Hermes orchestration role

Hermes is a **read-only** cluster coordinator: route tasks, track hand-offs, audit cluster health. See `.claude/CLAUDE.md` and `.claude/rules/` for the full orchestrator identity and protocols.

### Communication channels

- **Bot coordination:** `roosync_dashboard(type: "workspace", workspace: "cluster-coordination")` — deployment reports, inter-bot messages
- **Dashboard workspace:** `roosync_dashboard(type: "workspace", workspace: "hermes-agent")`
- **Dashboard global:** `roosync_dashboard(type: "global")` — routing + health reports
- **Cross-machine:** `roosync_send/read` for urgent notifications
- **Neighbor:** nanoclaw (ai-01) — mention via dashboard with `mentions: [{userId: {machineId: "myia-ai-01", workspace: "nanoclaw"}}]`

The upstream development guide lives in `AGENTS.md` and `CONTRIBUTING.md`. This file covers what's needed to work effectively in this repo.

---

## Build & Development Commands

```bash
# Install (first time)
uv venv .venv --python 3.11
source .venv/bin/activate  # Linux/WSL; Windows: .venv\Scripts\Activate.ps1
uv pip install -e ".[all,dev]"

# Install optional: browser tools
npm install

# Run the agent
hermes              # Interactive CLI
hermes --tui        # Ink-based TUI
hermes gateway      # Messaging gateway (Telegram, Discord, etc.)

# Tests — ALWAYS use the wrapper script, not pytest directly
scripts/run_tests.sh                                  # Full suite
scripts/run_tests.sh tests/gateway/                   # One directory
scripts/run_tests.sh tests/agent/test_foo.py::test_x  # One test
scripts/run_tests.sh -v --tb=long                     # With extra pytest flags

# If you must run pytest directly (Windows, IDE): use -n 4 and activate venv
python -m pytest tests/ -q -n 4

# TUI development (ui-tui/)
cd ui-tui && npm install
npm run dev          # Watch mode
npm run build        # Full build
npm run type-check   # tsc --noEmit
npm test             # vitest

# Diagnostics
hermes doctor
hermes version
```

### Key test constraints
- `scripts/run_tests.sh` enforces CI parity: TZ=UTC, LANG=C.UTF-8, credential env vars blanked, `-n 4` workers
- `tests/conftest.py` isolates `HERMES_HOME` to temp dirs — tests never write to `~/.hermes/`
- Integration tests are marked `@pytest.mark.integration` and excluded by default

---

## Architecture (Big Picture)

```
User message → AIAgent._run_agent_loop() (run_agent.py)
  ├── System prompt assembly (agent/prompt_builder.py)
  ├── LLM API call (OpenAI-compatible, any provider)
  ├── If tool_calls → dispatch via tools/registry.py → execute → loop
  ├── If text response → persist session → return
  └── Context compression if approaching token limit

CLI (cli.py) ──prompt_toolkit──> Interactive terminal
TUI (ui-tui/) ──stdio JSON-RPC──> tui_gateway/ ──> AIAgent
Gateway (gateway/run.py) ──platform adapters──> Telegram/Discord/Slack/WhatsApp/Signal/...
```

### File dependency chain
```
tools/registry.py  (no deps, imported by all tools)
       ↑
tools/*.py  (each calls registry.register() at import time)
       ↑
model_tools.py  (triggers tool discovery)
       ↑
run_agent.py, cli.py, batch_runner.py
```

### Key entry points

| File | Role |
|------|------|
| `run_agent.py` | `AIAgent` class — core conversation loop, tool dispatch |
| `cli.py` | `HermesCLI` — interactive CLI with prompt_toolkit |
| `model_tools.py` | Tool orchestration, `handle_function_call()` |
| `toolsets.py` | Tool groupings and platform presets |
| `hermes_state.py` | SQLite session store with FTS5 search |
| `hermes_cli/main.py` | CLI entry point, arg parsing, profile support |
| `hermes_cli/config.py` | Config management, `DEFAULT_CONFIG`, env var definitions |
| `hermes_cli/commands.py` | Central slash command registry (`COMMAND_REGISTRY`) |
| `gateway/run.py` | `GatewayRunner` — platform lifecycle, message routing |
| `agent/prompt_builder.py` | System prompt assembly (identity, skills, context, memory) |

### User configuration

| Path | Purpose |
|------|---------|
| `~/.hermes/config.yaml` | All settings (model, terminal, toolsets, compression...) |
| `~/.hermes/.env` | API keys and secrets ONLY |
| `~/.hermes/auth.json` | OAuth credentials |
| `~/.hermes/skills/` | Active skills |
| `~/.hermes/state.db` | SQLite session database |

### Profile system

Multiple isolated instances via `HERMES_HOME` override. All code must use `get_hermes_home()` from `hermes_constants` — never hardcode `~/.hermes`.

---

## Critical Gotchas

1. **Always `get_hermes_home()`, never `Path.home() / ".hermes"`** — breaks profiles. Use `display_hermes_home()` for user-facing messages.
2. **Never alter context mid-conversation** — breaks prompt caching. Changes to tools/skills/memory take effect next session by default.
3. **Gateway has TWO message guards** — new commands that must reach the runner while agent is blocked need to bypass both the base adapter queue and the gateway runner intercept.
4. **No `simple_term_menu`** in new code — has rendering bugs. Use `hermes_cli/curses_ui.py`.
5. **No ANSI `\033[K`** in spinner/display — leaks under prompt_toolkit. Use space-padding.
6. **Tool schemas must not reference tools from other toolsets** — those tools may be unavailable.
7. **Tests must not write to `~/.hermes/`** — `_isolate_hermes_home` autouse fixture redirects to temp.

---

## Platform Support

- **Primary**: Linux, macOS, WSL2
- **Native Windows**: Not supported — use WSL2
- **Android/Termux**: Supported via `.[termux]` extra

---

## Configuration Notes

Three config loaders exist — know which one you're in:

| Loader | Used by | Location |
|--------|---------|----------|
| `load_cli_config()` | CLI mode | `cli.py` |
| `load_config()` | Most CLI subcommands | `hermes_cli/config.py` |
| Direct YAML load | Gateway runtime | `gateway/run.py` + `gateway/config.py` |

---

## Deployment (po-2026)

### Container

- **Image:** `hermes-agent:s6-20260528` (local build from fork, NOT upstream pull)
- **PID 1:** `s6-svscan` (s6-overlay v3.2.3.0, replaces old tini/gosu)
- **User:** `hermes` (UID 10000) via `s6-setuidgid`
- **Volume:** `C:\Users\jsboi\.hermes` → `/opt/data` (persistent across rebuilds)
- **Ports:** `-p 9120:9119` (host 9120 → container 9119 dashboard UI)

### Docker run command (production)

```powershell
docker run -d --name hermes `
  --restart unless-stopped `
  -v C:\Users\jsboi\.hermes:/opt/data `
  -v C:\dev\roo-extensions\mcps\internal\servers\roo-state-manager:/opt/roo-state-manager:ro `
  --add-host=host.docker.internal:host-gateway `
  -p 9120:9119 `
  hermes-agent:s6-sync-20260602 gateway run
```

**NOTE:** Use PowerShell (not Git Bash) for `docker run` — Git Bash path conversion breaks `-v` mount targets.

No `-e` flags needed — all secrets come from `/opt/data/.env.secrets` loaded by the restore script.

### Boot sequence (s6-overlay Architecture B)

```
/init (s6-svscan, PID 1)
  ├── s6-rc-compile bundle (oneshot-runner, fix-attrs)
  ├── cont-init.d/ (lexicographic order):
  │   ├── 01-hermes-setup     ← stage2-hook.sh: sync bundled skills
  │   ├── 013-roosync-restore ← OUR SHIM: invokes /opt/data/restore-config.sh
  │   ├── 015-supervise-perms ← chown supervise/ trees
  │   └── 02-reconcile-profiles ← register default profile
  ├── s6-rc services:
  │   ├── main-hermes  ← CMD wrapper (main-wrapper.sh)
  │   └── dashboard    ← Web dashboard server
  └── CMD = main-wrapper.sh
      └── exec s6-setuidgid hermes hermes gateway run
```

### Restore script

`roosync-cluster/scripts/hermes-restore-config.sh` is copied to `/opt/data/restore-config.sh` on the persistent volume. Called by `013-roosync-restore` cont-init.d shim on every container start.

**What it does (in order):**
1. Load secrets from `/opt/data/.env.secrets`
2. Set model to `glm-5-turbo` with `provider: "zai"`
3. Remove duplicate `provider: "auto"` and OpenRouter `base_url` contamination
4. Append RooSync deployment config (auxiliary providers, STT, MCP servers, approvals)
5. **MCP auto-detection**: probe `192.168.0.47:9090` (ai-01 proxy). If unreachable, use local fallback (see below)
6. Write `/opt/data/.env` with all tokens (GLM, Telegram, GitHub)
7. Fix `jobs.json` format (list→dict, schedule normalization, remove toolset restrictions)
8. Install `croniter`, `gh` CLI, `jq`
9. Configure `gh auth` (persisted to `/opt/data/.config/gh`)
10. Patch kanban `SCHEMA_SQL` (session_id index before migration)
11. Run verification checks (PASS/FAIL)

### Local MCP proxy (when ai-01 is down)

When ai-01 (`192.168.0.47:9090`) is unreachable, the restore script activates local MCP infrastructure:

| Server | Transport | Details |
|--------|-----------|---------|
| roo-state-manager | stdio direct | Volume-mounted from roo-extensions, `.env` patched (GDrive/Qdrant disabled), `index.js` patched (FATAL → degrade) |
| sk-agent | mcp-remote via proxy | `host.docker.internal:9092/sk-agent/mcp` |
| searxng | mcp-remote via proxy | `host.docker.internal:9092/searxng/mcp` |

**Local proxy container:** `myia-mcp-proxy` (TBXark Go proxy, port 9092). Config: `roosync-cluster/docker/mcp-proxy/config.json`, compose: `roosync-cluster/docker/docker-compose.yml`.

**Container patches applied by restore script:**
- Restore script copies `/opt/roo-state-manager` → `/tmp/roo-state-manager` then patches the COPY only (never touches host volume — regression #560)
- `/tmp/roo-state-manager/.env`: `ROOSYNC_SHARED_PATH=` (empty), `QDRANT_URL=http://localhost:1`
- `/tmp/roo-state-manager/build/index.js`: unhandledRejection handler patched to `return` instead of `process.exit(1)` — prevents FATAL crash when Qdrant is unreachable

**Volume mount:** roo-state-manager is mounted as `ro` (read-only). The restore script copies to `/tmp/` before patching to prevent host `.env` corruption (#560).

### 3 Windows patches (MUST re-apply after upstream sync)

These patches exist because Windows git checkout introduces CRLF and s6-overlay env isolation breaks Docker ENV inheritance.

1. **CRLF strip** — Dockerfile adds `RUN find /etc/s6-overlay/s6-rc.d -type f -exec sed -i 's/\r$//' {} +` and same for `/etc/cont-init.d` after each COPY block. Without this, `s6-rc-compile` fails with "invalid type".

2. **`#!/command/with-contenv sh`** shebang on `docker/main-wrapper.sh`. Without this, CMD runs with only ~6 env vars (PATH, PWD, etc.) — Dockerfile ENV and `docker run -e` are NOT inherited. `with-contenv` loads from `/run/s6/container_environment/`.

3. **`export HOME="/opt/data"`** in main-wrapper.sh. `with-contenv` injects `HOME=/root` from Docker env, but `s6-setuidgid hermes` cannot write to `/root/`. Must override before exec.

### Dockerfile drift (our additions vs upstream)

These are the ONLY lines we add to the upstream Dockerfile:

```dockerfile
# After COPY docker/s6-rc.d/:
RUN find /etc/s6-overlay/s6-rc.d -type f -exec sed -i 's/\r$//' {} +

# After COPY docker/cont-init.d/*:
COPY --chmod=0755 docker/cont-init.d/013-roosync-restore /etc/cont-init.d/013-roosync-restore
RUN find /etc/cont-init.d -type f -exec sed -i 's/\r$//' {} +
```

Plus modifications to `docker/main-wrapper.sh` (shebang + HOME override — see patch #2 and #3 above).

### Persistent volume contents (survives rebuild)

| Path | Content |
|------|---------|
| `state.db` | SQLite session database |
| `sessions/` | Conversation history |
| `skills/` | Custom skills |
| `memories/` | Agent memories |
| `config.yaml` | Full config (restored by script) |
| `.env` | Tokens (regenerated by script) |
| `.env.secrets` | Secret values for script |
| `SOUL.md` | Agent personality |
| `cron/jobs.json` | Scheduled jobs |
| `.config/gh/` | GitHub CLI auth |
| `restore-config.sh` | Copy of restore script |

**NOT persistent** (reinstalled by restore script): `gh` CLI, `jq`, `croniter`, kanban patch.

### Rollback

```powershell
# If new image is broken, restore previous image:
docker stop hermes
docker rm hermes
docker run -d --name hermes `
  --restart unless-stopped `
  -v C:\Users\jsboi\.hermes:/opt/data `
  -v C:\dev\roo-extensions\mcps\internal\servers\roo-state-manager:/opt/roo-state-manager:ro `
  --add-host=host.docker.internal:host-gateway `
  -p 9120:9119 `
  hermes-agent:pre-sync-20260602 gateway run
```

### Backup protocol

**Auto-backup:** cont-init.d `012-roosync-backup` snapshots critical files on every boot (before restore). Stored in `/opt/data/backups/auto-YYYYMMDD-HHmmss.tar.gz` (max 3 rotated, ~20MB each).

**Manual pre-rebuild backup:**
```powershell
.\roosync-cluster\scripts\hermes-backup.ps1 -Reason "description"
```
Stops container, tars full volume to `C:\Users\jsboi\hermes-backups\`, restarts container. Keeps last 5.

**Post-op verification:**
```powershell
.\roosync-cluster\scripts\hermes-verify.ps1
```
12 checks: gateway process, Telegram connected, config/env symlinks, model, MCPs, cron jobs, kanban writable, gh auth, MCP health.

**Restore from backup:**
```powershell
docker stop hermes
docker run --rm -v C:\Users\jsboi\.hermes:C:\Users\jsboi\.hermes -v C:\Users\jsboi\hermes-backups:/backups alpine tar -xzf /backups/hermes-XXXXXX.tar.gz -C /opt/data/
docker start hermes
```

### z.ai provider (CRITICAL)

Use NATIVE z.ai (`provider: "zai"`, built-in `/api/coding/paas/v4` endpoint). NEVER use `ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic` — the Anthropic-compat translation layer causes MCP tool registry loss after compaction.

### MCP servers (container)

All 3 MCP servers connect via `mcp-remote` to `http://192.168.0.47:9090` (LAN direct, bypasses IIS ARR):
- `roo-state-manager` → `/roo-state-manager/mcp`
- `sk-agent` → `/sk-agent/mcp`
- `searxng` → `/searxng/mcp`

Auth: `Authorization: Bearer ${MCP_AUTH_TOKEN}` from `.env.secrets`.

### MCP resilience (watchdog)

**Root cause:** `MCPServerTask.run()` in `tools/mcp_tool.py` gives up after `_MAX_RECONNECT_RETRIES = 5` attempts (line 1648). Once the task returns, the bridge is dead until the gateway process restarts.

**Watchdog:** `roosync-cluster/scripts/hermes-mcp-watchdog.ps1` runs every 15 min via Windows Scheduled Task. Recovery escalation:

1. **Stage 1:** `SIGUSR1` to gateway PID — graceful restart, preserves container state
2. **Stage 2:** `docker restart` — full container reboot (last resort)

**Backoff:** Exponential (5, 10, 15... up to 60 min) between recovery attempts. Max 10 consecutive failures before giving up. Counter resets on healthy check. Prevents restart loops (incident 2026-05-11: 10+ restarts in 4h).

### Review activity (watchdog)

**Why:** The pr-review cron (`c8559be3577c`, hourly at :23) can complete with `status=ok` yet post **zero reviews** — a silent failure (model/provider breakdown, gh-auth drift, rate-limit storm). Cron status does not reflect actual review output, so the business signal (reviews posted on GitHub) must be watched directly.

**Watchdog:** `roosync-cluster/scripts/hermes-review-watchdog.ps1` runs every 30 min via Windows Scheduled Task `Hermes-Review-Watchdog` (off-minute :07/:37). It is an **independent observer** — reads `GH_TOKEN` + `TELEGRAM_BOT_TOKEN` from the host volume (`C:\Users\jsboi\.hermes\.env`) and queries the GitHub REST API directly (`Invoke-RestMethod`), so it alerts even when the Hermes container is down.

**Logic:** counts open PRs across target repos (`jsboige/CoursIA`, `jsboige/roo-extensions`) and the most recent review by the bot. If open PRs exist but no review has been posted within `AlertThresholdHours` (default 4h), it sends a Telegram alert to the review chat (`-1003904676273`). A cooldown (state file, 4h) prevents alert spam.

**Gotcha:** `Invoke-RestMethod` auto-deserializes JSON dates to `[datetime]` under the current culture (fr-FR here), which breaks naive subtraction and string round-tripping. All timestamps are normalized to UTC `[datetime]` via `ConvertTo-UtcDateTime` and formatted with `InvariantCulture` (else comma decimals like "722,2h").

**Context:** added 2026-08-07 after the user flagged "Hermes ne fait plus de reviews" — which was the z.ai 429 storm (resolved by the claudish switch), not a live gap; the watchdog catches the next silent one.

### Cluster-tour global-post watchdog

**Why:** The `hermes-cluster-tour` cron (job `525e5650a8ac`, `13 */12 * * *`) must append a `[CLUSTER-HEALTH]` post on the `global` dashboard at every fire (prompt ETAPE 3, marked OBLIGATOIRE/INCONDITIONNEL). It has skipped that post **silently and non-deterministically** while reporting `status=ok` ("completed successfully") — 2 incidents (2026-08-11 ~38h, 2026-08-14 ~48h), 4 consecutive fires post-T#40 without a global post. Prompt-side emphasis cannot close a prompt-skipping defect; only a programmatic post-fire check can (issue jsboige/hermes-agent #3, design validated by ai-01 in 5 points).

**Watchdog:** `roosync-cluster/scripts/hermes-cluster-tour-watchdog.ps1` — independent observer, same pattern as `hermes-review-watchdog.ps1`. Reads everything from the HOST volume so it works even when the container is down:
- `C:\Users\jsboi\.hermes\cron\jobs.json` → cluster-tour `last_run_at` (the watched fire)
- `G:\Mon Drive\Synchronisation\RooSync\.shared-state\dashboards\global.md` → last message block whose section TITLE matches `## [CLUSTER-HEALTH]`

Assertion: when a new fire is older than 15 min, a `[CLUSTER-HEALTH]` title must be timestamped ≥ fire −5 min, else MISSING → appends a `[WARN]` directly on `global.md` (visible exactly where the post should have been, read by every cluster MCP agent) + Telegram alert to the review chat (cooldown 120 min). Never re-fires the tour (spec point 4). Counts OK/MISSING in a state file to instrument the rate (spec point 5).

Deployed as Windows Scheduled Task `Hermes-ClusterTour-Watchdog` (every 30 min off-minute `:07/:37`, via hidden VBS launcher in `C:\ProgramData\claude-hidden-launchers\`), so a fire at 12:13Z is checked by ~12:37Z.

**Gotcha (fixed):** the `[WARN]` body text contains the literal string "[CLUSTER-HEALTH]" ("aucun append [CLUSTER-HEALTH]") — the parser matches the section TITLE (`## [CLUSTER-HEALTH]`) only, otherwise the WARN self-validates as a posted tour.

### Cluster ASR

`https://whisper-api.myia.io/v1` — self-hosted Whisper on po-2023. Auth via `WHISPER_BEARER_TOKEN` from `.env.secrets`.

### Context compression

```yaml
auxiliary:
  compression:
    provider: "zai"
    model: "glm-4.5-air"
```

Watch for `tool_use_error` after compaction — restart session if seen.

### Dashboard coordination

Bots coordinate on `workspace-cluster-coordination` dashboard, NOT on `workspace-hermes-agent` or `global`. Always post deployment reports and status changes there.

<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent - zapabob Windows / Operations Fork

<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/zapabob/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://github.com/NousResearch/hermes-agent"><img src="https://img.shields.io/badge/Upstream-NousResearch-blueviolet?style=for-the-badge" alt="Upstream: NousResearch"></a>
</p>

**zapabob/hermes-agent** is a Windows-first operations fork of
[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent). It
tracks the official API, Desktop, Dashboard, Gateway, TUI, memory, skills, cron,
and CI surfaces while adding the pieces needed for a long-running native
Windows workstation: Desktop integration, WebUI companion wiring, always-on
messaging, free-model rotation, local fallback inference, Hypura/OpenClaw
harness tooling, VRChat Quest 2 support, NotebookLM source workflows, and
practical recovery scripts.

## Fork Advantages

This fork is for operators who want upstream Hermes freshness without losing
the local behavior that keeps a real Windows machine usable day after day.

| Advantage | What this fork adds |
|---|---|
| **Official freshness with local continuity** | Upstream `main` is merged regularly with fork-specific Windows, Gateway, Desktop, WebUI, and Harness behavior preserved instead of overwritten. |
| **Native Windows reliability** | PowerShell launchers, Task Scheduler flows, UTF-8 safeguards, Windows subprocess environment repair, transient file-operation retry logic, and path-with-spaces handling are treated as first-class runtime concerns. |
| **Desktop + dashboard operations** | Hermes Desktop is wired to a local dashboard backend and the same source checkout, so the graphical shell, dashboard admin pages, gateway state, and CLI workflows share one operational surface. |
| **Always-on messaging stack** | Telegram, Discord, LINE, and gateway state are managed with profile-aware config, host-secret preference, restart verification, and Windows-friendly diagnostics. |
| **Free cloud + local fallback inference** | OpenCode Zen `auto-free` rotation gives a free cloud-first path, while llama.cpp/TurboQuant and Hypura/OpenClaw tooling provide local rollback when cloud capacity or credentials are unavailable. |
| **NotebookLM knowledge workflow** | The bundled NotebookLM plugin collects implementation logs and redacted X activity into NotebookLM-ready sources, can draft X post ideas, and supports NotebookLM Enterprise source ingestion when configured. |
| **VRChat / Quest 2 operations** | Neuro/OSC helpers, Hypura Harness, VOICEVOX checks, OpenXR repair scripts, and Quest 2 diagnostics make VRChat automation auditable and recoverable on the same Windows host. |
| **Security and CI discipline** | Dependency bounds, production audit checks, targeted pytest lanes, Desktop/TUI/Web builds, red-team issue triage, and upstream PR extraction keep fork improvements suitable for official contribution. |

### 日本語概要

[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) を
土台に、公式の TUI、Gateway、Desktop、Dashboard、Skills、Memory、Cron、CI を
追いかけながら、native Windows で落ちやすい部分を補強した実運用 fork です。
OpenCode Zen の free model rotation、llama.cpp/TurboQuant fallback、Hypura
Harness、OpenClaw 由来のローカル資産、WebUI 連携、VRChat Quest 2 tooling、
NotebookLM 向け source bundling をまとめて扱えます。

この fork の価値は、独自機能を抱え込んで公式から離れることではありません。
公式に戻せる修正は PR として切り出し、fork に残すべき Windows 運用、個人
gateway、Desktop/WebUI 連携、VRChat、NotebookLM の作業導線は保ったまま、
公式最新 API に追随します。

---

## Overview

This checkout is not generic upstream marketing. It is a **production stack** for:

1. **Cloud-first, free inference** — `opencode-zen` + virtual model `auto-free` with live catalog refresh and runtime rotation when limits hit.
2. **Local rollback** — TurboQuant `llama-server` on `http://127.0.0.1:8080/v1` (RTX 3080 defaults: ngram-mod speculative, asymmetric KV).
3. **Always-on gateway** — Telegram, Discord, and other platforms via `hermes gateway run`, with Windows-specific reliability fixes.
4. **VRChat autonomy** — Neuro API bridge, OSC tools, safety-gated harness scripts, Quest 2 controller doctor, OpenXR ActiveRuntime fix for Virtual Desktop.
5. **Desktop integration** — Task Scheduler autostart for llama + gateway, optional [hermes-webui](https://github.com/zapabob/hermes-webui) companion.

Upstream Hermes still provides the core agent loop, TUI, toolsets, skills hub, memory providers, delegation, and cron. This fork adds the **operations layer** that makes those features survive on native Windows without WSL.

Current baseline: merged `upstream/main` through `2e628ae97` on 2026-06-04,
then kept the fork-side Windows reliability and operations layer. This sync
includes upstream Desktop, Dashboard, Gateway, Docker, memory, installer, TUI,
MCP, model routing, provider setup, and CI fixes while preserving fork-specific
OpenCode free rotation, llama fallback, NotebookLM automation, Hypura Harness,
VRChat tooling, WebUI companion wiring, and native Windows hardening. For future
updates, compare against `upstream/main` via `git fetch upstream main` and the
Python sync helpers under `scripts/`.

Recent upstream contribution slice: [NousResearch/hermes-agent#36921](https://github.com/NousResearch/hermes-agent/pull/36921) extracts the Windows transient `os.replace` / `PermissionError` retry fix without `_docs` or fork-only operational features.

---

## AI/Agent Engineering Evidence Card

| Field | Current public evidence |
| --- | --- |
| Agent surface | OpenCode Zen `auto-free` routing, llama.cpp TurboQuant fallback, gateway automation, NotebookLM automation, Skills Hub, curator, cron, delegation, TUI, dashboard admin pages, and WebUI companion paths |
| Model/runtime surface | Free cloud model rotation, local GGUF fallback at `http://127.0.0.1:8080/v1`, RTX Windows launch scripts, and provider catalog refresh tooling |
| Repro command | `pip install -e ".[all,dev]"`, `python -m hermes_cli.main setup`, `py -3 scripts/refresh_opencode_free_catalog.py --force`, and `hermes fallback list` |
| Operational proof | Windows autostart scripts, gateway hardening, dashboard auth/admin tests, OpenClaw credential bridge, VRChat Quest 2 doctors, and profile-safe config/logging guidance |
| Metrics to inspect | Test suite results, gateway uptime, dashboard/admin health, fallback switch events, free-model catalog freshness, llama fallback health checks, and VRChat preflight output |
| Limitations | This fork is an operations stack for a personal Windows deployment; secrets, local models, and gateway credentials are intentionally external |

---

## Unique Features

| Feature | What it does |
|---|---|
| **OpenCode Zen `auto-free`** | Virtual model sentinel resolves to the first live free ID from `https://opencode.ai/zen/v1/models`. On `Free usage exceeded` and similar limits, Hermes walks the deduped free catalog automatically. Skill: `skills/autonomous-ai-agents/opencode-free-rotation/`. |
| **Live catalog refresh** | `scripts/refresh_opencode_free_catalog.py` pulls the current Zen free list; use in cron or before long sessions. |
| **NotebookLM automation** | `plugins/notebooklm` collects implementation logs and redacted `lm-twitterer` activity, drafts X post ideas through Hermes, and can upload raw text sources to NotebookLM Enterprise when project credentials are configured. |
| **OpenClaw → OpenCode key bridge** | Shared `OPENCODE_API_KEY` (OpenClaw `.env` or `auth-profiles.json`) satisfies `OPENCODE_ZEN_API_KEY` when the Zen-specific key is unset. See `hermes_cli/auth.py` and `tests/hermes_cli/test_opencode_openclaw_bridge.py`. |
| **Ebbinghaus idle sleep** | Optional `memory.sleep` config runs a lazy `ebbinghaus_memory(action="sleep")` cycle after idle time, consolidating high-salience memories and pruning low-value traces while staying compatible with upstream completed-turn memory context. |
| **llama.cpp TurboQuant fallback** | `hermes_cli/llama_fallback_runtime.py` autostarts `llama-server` when the fallback chain reaches `llama-cpp`. RTX3080 script: `scripts/windows/start-hermes-llama-fallback-rtx3080.ps1` (ngram-mod speculative, `f16v_turbo4` KV). Example config: `docs/migration/opencode_free_webui_config.example.yaml`. |
| **hermes-webui companion** | `scripts/windows/start-hermes-webui.ps1` bootstraps a sibling `hermes-webui` checkout (default `~/Desktop/hermes-webui`) with `config/hermes-webui.env.example`. WebUI reads raw `config.yaml`; `auto-free` displays as-is but resolves at agent runtime. |
| **VRChat Neuro / autonomy harness** | `skills/gaming/neuro-vrchat/` + `tools/vrchat_*` + `scripts/vrchat_*` — Neuro API websocket bridge, observation queue, preflight, runtime doctor, private smoke, completion audit. Uses vendored `vendor/neuro-sdk` protocol reference. Profile safety gate blocks live OSC/audio until explicitly armed. |
| **Quest 2 Windows doctor + OpenXR fix** | Read-only stack diagnosis: `scripts/windows/vrchat_quest2_controller_doctor.ps1`. HKLM/HKCU ActiveRuntime sync for Virtual Desktop: `scripts/windows/vrchat_quest2_openxr_fix.ps1` and UAC wrapper `scripts/windows/run-vrchat-openxr-fix-admin.ps1`. |
| **Windows logon autostart** | `scripts/windows/register-hermes-autostart.ps1` registers the gateway Task Scheduler job by default. Local GGUF/llama autostart is opt-in for rollback/recovery checks via `-IncludeLlama`. Cleans stale HKCU Run entries. |
| **Gateway hardening** | Discord stale slash-command cleanup before re-register (100-command limit). `DISCORD_ALLOWED_USERS=*` as explicit allow-all. Telegram 90s connect budget with optional fallback IP disable. |
| **Windows terminal hardening** | Git Bash preferred over WSL `bash.exe` stubs; UTF-8 terminal output; `search_files` finds `rg` / Git Bash `grep` / `find`. |
| **Dashboard operations pages** | Upstream admin/system/MCP/pairing/webhook pages are merged with the fork's auth expectations and Windows gateway workflow. |
| **Native Windows test/runtime hygiene** | Shared Windows subprocess env backfill prevents `WinError 10106` when tests intentionally strip env vars. Config tests read UTF-8 explicitly; `atomic_json_write` tolerates transient Windows replace locks; Skills Hub hashes normalize CRLF while preserving path sensitivity; Bitwarden disk cache keeps POSIX `0600` semantics without treating Windows `st_mode` as POSIX ACLs. |
| **Hypura Harness CLI** | `hermes harness status|start|stop|restart` with clear diagnostics when daemon scripts are missing. |
| **OpenClaw migration** | `hermes claw migrate` plus `scripts/openclaw_ports/` and `tools/openclaw/` for VRChat, VoiceVox, channel readiness. |
| **Skills Hub path safety** | Uninstall lock validates `install_path` — rejects traversal, absolute paths, and skills-root deletion. Guard and sync flows preserve curator metadata across official updates. |

---

## Quick Start

### Windows (this fork)

```powershell
git clone https://github.com/zapabob/hermes-agent.git
cd hermes-agent
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[all,dev]"
python -m hermes_cli.main setup
```

Set secrets in `~/.hermes/.env`:

```env
OPENCODE_ZEN_API_KEY=...          # from https://opencode.ai/auth
# Or reuse OpenClaw:
# OPENCODE_API_KEY=...
GATEWAY_ALLOW_ALL_USERS=true      # single-user personal gateway
```

Copy model/fallback sections from `docs/migration/opencode_free_webui_config.example.yaml` into `~/.hermes/config.yaml`, then:

```powershell
hermes                          # interactive CLI / TUI
hermes fallback list            # verify auto-free expansion
hermes gateway run              # messaging gateway
hermes doctor                   # environment diagnostics
```

### Native Windows notes

- Prefer PowerShell 7 or modern Windows Terminal with UTF-8 enabled. Hermes reads and writes `~/.hermes/config.yaml` as UTF-8.
- Subprocesses launched from narrowed environments must keep core Windows variables such as `SYSTEMROOT`, `WINDIR`, and `COMSPEC`; Hermes provides `hermes_cli.windows_env.ensure_windows_subprocess_env()` for tests and helpers that build env dicts manually.
- POSIX permission bits are not Windows ACLs. Secret-bearing files are still written through the secure path, but tests should not require Windows `st_mode` to equal POSIX `0600`.
- Text hashes for skills normalize CRLF/LF so Windows checkouts do not show false update drift, while filenames remain part of the hash to catch content swaps.

### Official upstream install

Linux / macOS / WSL2 / Termux:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
source ~/.bashrc
hermes
```

Native Windows (official installer):

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

---

## Local Fallback (llama.cpp + TurboQuant)

> **Android / Termux:** The tested manual path is documented in the [Termux guide](https://hermes-agent.nousresearch.com/docs/getting-started/termux). On Termux, Hermes installs a curated `.[termux]` extra because the full `.[all]` extra currently pulls Android-incompatible voice dependencies.
>
> **Windows:** Native Windows is fully supported — the PowerShell one-liner above installs everything. If you'd rather use WSL2, the Linux command works there too. Native Windows install lives under `%LOCALAPPDATA%\hermes`; WSL2 installs under `~/.hermes` as on Linux.

**RTX 3080 manual start:**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\start-hermes-llama-fallback-rtx3080.ps1
```

**Environment (optional autostart via `HERMES_LLAMA_FALLBACK_AUTOSTART=auto`):**

| Variable | Purpose |
|---|---|
| `HERMES_LLAMA_MODEL_PATH` | Path to fallback GGUF |
| `HERMES_LLAMA_GPU_PROFILE` | `rtx3080` (default in script) |
| `HERMES_LLAMA_SERVER_EXE` | Override TurboQuant `llama-server.exe` |

Runtime module: `hermes_cli/llama_fallback_runtime.py` — probes port 8080, spawns server with ngram-mod speculative decoding when needed.

---

## OpenCode Free Rotation

**Primary config pattern:**

```yaml
model:
  provider: opencode-zen
  default: auto-free

fallback_providers:
  - provider: opencode-zen
    model: auto-free
  - provider: llama-cpp
    model: your-fallback.gguf
    base_url: http://127.0.0.1:8080/v1
```

**Refresh catalog:**

```powershell
py -3 scripts/refresh_opencode_free_catalog.py --force
hermes fallback list
```

**Credential bridge:** if you migrated from OpenClaw, a single `OPENCODE_API_KEY` in `~/.hermes/.env` is enough — Hermes maps it to Zen and Go provider env vars.

Full skill procedure: `skills/autonomous-ai-agents/opencode-free-rotation/SKILL.md`.

---

## VRChat & Neuro SDK

### Skills and tools

- `skills/gaming/vrchat/` — OSC, avatar registry, relay bridge
- `skills/gaming/neuro-vrchat/` — Neuro API bridge with safety-gated autonomy profile
- Tools: `vrchat_osc`, `vrchat_neuro_*`, `vrchat_autonomy_*`, `vrchat_preflight`, `vrchat_runtime_doctor`, etc.

### Key harness scripts

| Script | Role |
|---|---|
| `scripts/vrchat_neuro_bridge.py` | Neuro websocket harness |
| `scripts/vrchat_preflight.py` | Read-only readiness bundle |
| `scripts/vrchat_runtime_doctor.py` | Operator mismatch + VOICEVOX/VRChat diagnostics |
| `scripts/vrchat_private_smoke.py` | Gated private-instance smoke (dry-run default) |
| `scripts/vrchat_completion_audit.py` | Objective completion evidence |

Migration guide: `docs/migration/vrchat_neurosama_autonomy.md`.

### Quest 2 on Windows (Virtual Desktop)

**Controller / stack doctor (read-only):**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\vrchat_quest2_controller_doctor.ps1 -Json
```

**OpenXR ActiveRuntime fix (Virtual Desktop / SteamVR):**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\run-vrchat-openxr-fix-admin.ps1 -Preference VirtualDesktop
```

Requires UAC elevation for HKLM `ActiveRuntime` sync.

---

## Windows Autostart

Register the gateway logon task without reserving VRAM for a local GGUF/llama server:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\register-hermes-autostart.ps1
```

Options:

- `-IncludeLlama` — also register the rollback/recovery llama task
- `-GatewayOnly` — compatibility flag; gateway-only is now the default
- `-Unregister` — remove tasks and stale Run keys
- `-IncludeLegacyStack` — also register full stack launcher

Gateway wrapper: `scripts/windows/start-hermes-gateway.ps1` skips llama by default. Use `-StartLlama` or `HERMES_GATEWAY_RECOVERY_START_LLAMA=1` only for rollback/recovery checks.

**Companion WebUI:**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\start-hermes-webui.ps1
```

Set `HERMES_WEBUI_ROOT` if not using `~/Desktop/hermes-webui`.

**Host migration / recovery:**

Surface-to-mothership recovery is documented in `docs/migration/windows_surface_to_mothership.md`.
Use `scripts/windows/export-hermes-host-migration.ps1`,
`scripts/windows/import-hermes-host-migration.ps1`, and
`scripts/windows/verify-hermes-host-migration.ps1` to move config and scheduled-task state without committing secrets.

---

## Inherited Upstream Capabilities

| Area | Highlights |
|---|---|
| TUI / CLI | Ink TUI (`hermes --tui`), slash commands, session resume, tool streaming, active-session switching, and update branch selection |
| Messaging gateway | Telegram, Discord, Slack, WhatsApp, Signal, Email, Matrix, cached-agent MCP refresh, … |
| Learning loop | Memory providers, session search, skill creation, curator |
| Cron | Natural-language scheduled jobs with multi-platform delivery |
| Delegation | Subagents with isolated terminal sessions |
| MCP | Optional MCP catalog manifests, interactive picker, and tools config integration |
| Docker / voice | Windows Docker Desktop compose support, container env propagation, Docker HOME ownership protection, and Pulse/PipeWire voice-mode passthrough |
| Terminal backends | local, Docker, SSH, Modal, Daytona, … |

Official docs: [Quickstart](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart) · [Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) · [Skills](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) · [Cron](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron)

---

## Upstream Sync

This fork intentionally keeps a small Windows/operations layer on top of upstream. During a sync, preserve local value in provider fallback, gateway reliability, VRChat tooling, Windows scripts, and README/docs that describe this deployment.

```powershell
git fetch upstream main
py -3 scripts\sync_all.py --dry-run
py -3 scripts\sync_all.py --merge --target main --allow-preflight-blockers
py -3 scripts\sync_upstream.py --pytest-only
```

Watch list (preserve fork value during merges):

- `tools/environments/local.py`, `gateway/platforms/discord.py`, `gateway/platforms/telegram.py`
- `hermes_cli/harness.py`, `hermes_cli/llama_fallback_runtime.py`, `hermes_cli/auth.py`
- `tools/skills_hub.py`, `scripts/windows/`, `README.md`

---

## Development

Local generated outputs:

- `career_docs_output/` is for local resume, cover-letter, and career-document drafts. It is ignored by git so personal application material and generated exports stay out of commits.

Quick start for contributors — use the standard installer, then work from the
full git checkout it creates at `$HERMES_HOME/hermes-agent` (usually
`~/.hermes/hermes-agent`). This matches the layout used by `hermes update`, the
managed venv, lazy dependencies, gateway, and docs tooling.

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
cd "${HERMES_HOME:-$HOME/.hermes}/hermes-agent"
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

Manual clone fallback (for throwaway clones/CI where you intentionally do not
want the managed install layout):

Windows-native regression slice:

```powershell
py -3.12 -m pytest --timeout-method=thread `
  tests\gateway\test_resume_command.py `
  tests\gateway\test_config_env_bridge_authority.py `
  tests\hermes_cli\test_config.py `
  tests\test_bitwarden_secrets.py `
  tests\tools\test_skills_hub.py::TestCheckForSkillUpdates `
  tests\tools\test_pr_6656_regressions.py::TestBundleHashFilenameSensitivity `
  tests\tools\test_code_execution_windows_env.py -q
```

Fork-specific feature smoke tests:

```powershell
py -3.12 -m pytest -o addopts="" -p no:randomly tests\hermes_cli\test_opencode_openclaw_bridge.py -q
py -3.12 -m pytest -o addopts="" -p no:randomly tests\hermes_cli\test_opencode_free_rotation.py -q
py -3.12 -m pytest -o addopts="" -p no:randomly tests\tools\test_local_env_windows_msys.py -q
```

Full suite (CI parity): `scripts/run_tests.sh`

---

## Community & Links

- [Official Hermes docs](https://hermes-agent.nousresearch.com/docs/)
- [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (upstream)
- [zapabob/hermes-agent](https://github.com/zapabob/hermes-agent) (this fork)
- [hermes-webui](https://github.com/zapabob/hermes-webui) (companion WebUI)
- [OpenCode Zen](https://opencode.ai/auth) (free model API key)
- [VedalAI neuro-sdk](https://github.com/VedalAI/neuro-sdk) (Neuro API protocol)
- [Nous Research Discord](https://discord.gg/NousResearch)

---

## License

MIT — see [LICENSE](LICENSE).

Built by [Nous Research](https://nousresearch.com). Windows and operations fork maintained by [zapabob](https://github.com/zapabob).

<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤

<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Built%20by-Nous%20Research-blueviolet?style=for-the-badge" alt="Built by Nous Research"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/Lang-中文-red?style=for-the-badge" alt="中文"></a>
</p>

**The self-improving AI agent built by [Nous Research](https://nousresearch.com).** It's the only agent with a built-in learning loop — it creates skills from experience, improves them during use, nudges itself to persist knowledge, searches its own past conversations, and builds a deepening model of who you are across sessions. Run it on a $5 VPS, a GPU cluster, or serverless infrastructure that costs nearly nothing when idle. It's not tied to your laptop — talk to it from Telegram while it works on a cloud VM.

Use any model you want — [Nous Portal](https://portal.nousresearch.com), [OpenRouter](https://openrouter.ai) (200+ models), [NVIDIA NIM](https://build.nvidia.com) (Nemotron), [Xiaomi MiMo](https://platform.xiaomimimo.com), [z.ai/GLM](https://z.ai), [Kimi/Moonshot](https://platform.moonshot.ai), [MiniMax](https://www.minimax.io), [Hugging Face](https://huggingface.co), OpenAI, or your own endpoint. Switch with `hermes model` — no code changes, no lock-in.

<table>
<tr><td><b>A real terminal interface</b></td><td>Full TUI with multiline editing, slash-command autocomplete, conversation history, interrupt-and-redirect, and streaming tool output.</td></tr>
<tr><td><b>Lives where you do</b></td><td>Telegram, Discord, Slack, WhatsApp, Signal, and CLI — all from a single gateway process. Voice memo transcription, cross-platform conversation continuity.</td></tr>
<tr><td><b>A closed learning loop</b></td><td>Agent-curated memory with periodic nudges. Autonomous skill creation after complex tasks. Skills self-improve during use. FTS5 session search with LLM summarization for cross-session recall. <a href="https://github.com/plastic-labs/honcho">Honcho</a> dialectic user modeling. Compatible with the <a href="https://agentskills.io">agentskills.io</a> open standard.</td></tr>
<tr><td><b>Scheduled automations</b></td><td>Built-in cron scheduler with delivery to any platform. Daily reports, nightly backups, weekly audits — all in natural language, running unattended.</td></tr>
<tr><td><b>Delegates and parallelizes</b></td><td>Spawn isolated subagents for parallel workstreams. Write Python scripts that call tools via RPC, collapsing multi-step pipelines into zero-context-cost turns.</td></tr>
<tr><td><b>Runs anywhere, not just your laptop</b></td><td>Seven terminal backends — local, Docker, SSH, Singularity, Modal, Daytona, and Vercel Sandbox. Daytona and Modal offer serverless persistence — your agent's environment hibernates when idle and wakes on demand, costing nearly nothing between sessions. Run it on a $5 VPS or a GPU cluster.</td></tr>
<tr><td><b>Research-ready</b></td><td>Batch trajectory generation, Atropos RL environments, trajectory compression for training the next generation of tool-calling models.</td></tr>
</table>

---

## Hermes Agent 10x Fast - Performance Comparisons

This branch documents a measured performance pass focused on startup hot paths,
tool discovery, SQLite session persistence, TUI MCP reloads, adaptive
parallelism, delegation config reuse, parallel guard fast paths, and runtime
local-endpoint probe avoidance. The numbers below are
local Windows benchmark results from `scripts/benchmark_startup_perf.py` and
`scripts/benchmark_runtime_usage.py`; they are measurements for this branch, not
universal guarantees.

Full PR documentation:

- [Upstream PR draft](docs/hermes-performance-upstream-pr.md)
- [10x Fast implementation notes](docs/hermes-agent-10x-fast-pr.md)
- [Runtime performance investigation](docs/runtime-performance-investigation-2026-05-15.md)

![Hermes Agent 10x Fast performance dashboard](docs/assets/10x-fast/perf-summary-dashboard.svg)

![Hermes Agent 10x Fast macro original vs branch comparison](docs/assets/10x-fast/generated/macro-original-vs-10x-fast.png)

### Visual Before/After Gallery

Every image below is tagged and tied to a measured old-vs-new comparison. The
generated PNGs are visual summaries, but the table still pins each one to the
specific benchmark claim it represents.

`#startup` `#tool-schema` `#plugin-discovery` `#sqlite` `#runtime`
`#delegation` `#parallel-tools` `#mcp` `#research-to-code`

| Visual | Tags | Old | New | Gain |
| --- | --- | ---: | ---: | ---: |
| <img src="docs/assets/10x-fast/generated/macro-original-vs-10x-fast.png" width="320" alt="macro comparison original vs 10x fast branch showing speed gains across startup plugin discovery sqlite endpoint guard and parallel tools"> | `#macro` `#promo` `#overview` `#generated` | original branch across six criteria | 10x Fast branch across six criteria | **up to 24.21x** |
| <img src="docs/assets/10x-fast/perf-startup-model-tools.svg" width="320" alt="model_tools startup old 2.0847s new 0.8419s gain 2.48x"> | `#startup` `#model-tools` | 2.0847s | 0.8419s | **2.48x** |
| <img src="docs/assets/10x-fast/perf-tool-definitions-startup.svg" width="320" alt="tool definitions startup old 1.8782s new 0.8741s gain 2.15x"> | `#startup` `#tool-schema` | 1.8782s | 0.8741s | **2.15x** |
| <img src="docs/assets/10x-fast/perf-plugin-discovery.svg" width="320" alt="plugin discovery old 0.5571s new 0.1930s gain 2.89x"> | `#plugin-discovery` `#startup` | 0.5571s | 0.1930s | **2.89x** |
| <img src="docs/assets/10x-fast/perf-session-batch-writes.svg" width="320" alt="SQLite session writes old 0.6329s new 0.0240s gain 24.21x"> | `#sqlite` `#session-db` | 0.6329s | 0.0240s | **24.21x** |
| <img src="docs/assets/10x-fast/runtime-local-endpoint-fast-path.svg" width="320" alt="dead local endpoint old 51.4181s new 5.5563s gain 9.25x"> | `#runtime` `#local-endpoint` | 51.4181s | 5.5563s | **9.25x** |
| <img src="docs/assets/10x-fast/phase-7-delegate-parallel-guard.svg" width="320" alt="parallel guard old 6.9878s new 1.6403s gain 4.26x"> | `#delegation` `#parallel-tools` | 6.9878s | 1.6403s | **4.26x** |
| <img src="docs/assets/10x-fast/runtime-benchmark-suite.svg" width="320" alt="runtime benchmark suite showing gains up to 22.10x"> | `#runtime` `#benchmark-suite` | mixed hot paths | latest medians | **up to 22.10x** |
| <img src="docs/assets/10x-fast/generated/parallel-runtime.png" width="320" alt="parallel runtime generated comparison old 0.3036s new 0.0590s gain 5.14x"> | `#parallel-tools` `#generated` | 0.3036s | 0.0590s | **5.14x** |
| <img src="docs/assets/10x-fast/generated/runtime-before-after.png" width="320" alt="runtime before after generated comparison old 51.4181s new 5.5563s gain 9.25x"> | `#runtime` `#generated` | 51.4181s | 5.5563s | **9.25x** |
| <img src="docs/assets/10x-fast/generated/runtime-before-after-alt.png" width="320" alt="runtime before after alternate generated comparison old 51.4181s new 5.5563s gain 9.25x"> | `#runtime` `#generated` | 51.4181s | 5.5563s | **9.25x** |
| <img src="docs/assets/10x-fast/generated/research-to-code.png" width="320" alt="research to code visual mapping cache batch route and fast fail ideas to measured gains"> | `#research-to-code` `#systems` | repeated probes/work | cache, batch, route, fast-fail | **up to 24.21x** |
| <img src="docs/assets/10x-fast/hermes-agent-10x-fast-phase-1.png" width="320" alt="generated 10x fast overview visual with measured gains up to 24.21x"> | `#overview` `#generated` | baseline hot paths | optimized hot paths | **up to 24.21x** |

### Startup And Tool Schema

![model_tools startup comparison](docs/assets/10x-fast/perf-startup-model-tools.svg)

![tool definitions startup comparison](docs/assets/10x-fast/perf-tool-definitions-startup.svg)

![plugin discovery comparison](docs/assets/10x-fast/perf-plugin-discovery.svg)

### Runtime Hot Paths

![runtime local endpoint fast path comparison](docs/assets/10x-fast/runtime-local-endpoint-fast-path.svg)

![runtime benchmark suite](docs/assets/10x-fast/runtime-benchmark-suite.svg)

![delegate task and parallel guard comparison](docs/assets/10x-fast/phase-7-delegate-parallel-guard.svg)

![SQLite session batch write comparison](docs/assets/10x-fast/perf-session-batch-writes.svg)

![toolset cache comparison](docs/assets/10x-fast/perf-toolset-cache.svg)

![TUI MCP reload avoidance comparison](docs/assets/10x-fast/perf-mcp-reload-avoidance.svg)

![adaptive parallel tool scan comparison](docs/assets/10x-fast/perf-adaptive-parallel-scan.svg)

### Research To Code

![research principles mapped to Hermes optimizations](docs/assets/10x-fast/research-principles-map.svg)

This runtime pass cross-checked the official Hermes docs for the
[agent loop](https://hermes-agent.nousresearch.com/docs/developer-guide/agent-loop/),
[tools runtime](https://hermes-agent.nousresearch.com/docs/developer-guide/tools-runtime),
[delegation](https://hermes-agent.nousresearch.com/docs/guides/delegation-patterns/),
and [context compression](https://hermes-agent.nousresearch.com/docs/developer-guide/context-compression-and-caching/).
The research logic comes from efficient LLM systems work such as
[FrugalGPT](https://arxiv.org/abs/2305.05176),
[LLMLingua](https://arxiv.org/abs/2310.05736),
[vLLM/PagedAttention](https://arxiv.org/abs/2309.06180), and
[RouteLLM](https://arxiv.org/abs/2406.18665): route cheap first, compress or
cache stable work, batch tiny writes, and avoid probes that cannot produce
signal.

### Architecture Diagrams

![tool discovery cache architecture](docs/assets/10x-fast/phase-2-tool-discovery-cache.svg)

![toolset memoization architecture](docs/assets/10x-fast/phase-3-toolset-cache.svg)

![SQLite batch write architecture](docs/assets/10x-fast/phase-4-sqlite-batch-writes.svg)

![TUI MCP fingerprint architecture](docs/assets/10x-fast/phase-5-tui-mcp-fingerprint.svg)

![adaptive parallel scan architecture](docs/assets/10x-fast/phase-6-adaptive-parallel-scan.svg)

![delegate task config and parallel guard architecture](docs/assets/10x-fast/phase-7-delegate-parallel-guard.svg)

---

## Quick Install

### Linux, macOS, WSL2, Termux

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

### Windows (native, PowerShell) — Early Beta

> **Heads up:** Native Windows support is **early beta**. It installs and runs, but hasn't been road-tested as broadly as our Linux/macOS/WSL2 paths. Please [file issues](https://github.com/NousResearch/hermes-agent/issues) when you hit rough edges. For the most battle-tested Windows setup today, run the Linux/macOS one-liner above inside **WSL2**.

Run this in PowerShell:

```powershell
irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1 | iex
```

The installer handles everything: uv, Python 3.11, Node.js, ripgrep, ffmpeg, **and a portable Git Bash** (MinGit, unpacked to `%LOCALAPPDATA%\hermes\git` — no admin required, completely isolated from any system Git install).  Hermes uses this bundled Git Bash to run shell commands.

If you already have Git installed, the installer detects it and uses that instead.  Otherwise a ~45MB MinGit download is all you need — it won't touch or interfere with any system Git.

> **Android / Termux:** The tested manual path is documented in the [Termux guide](https://hermes-agent.nousresearch.com/docs/getting-started/termux). On Termux, Hermes installs a curated `.[termux]` extra because the full `.[all]` extra currently pulls Android-incompatible voice dependencies.
>
> **Windows:** Native Windows is supported as an **early beta** — the PowerShell one-liner above installs everything, but expect rough edges and please file issues when you hit them. If you'd rather use WSL2 (our most battle-tested Windows path), the Linux command works there too. Native Windows install lives under `%LOCALAPPDATA%\hermes`; WSL2 installs under `~/.hermes` as on Linux.  The only Hermes feature that currently needs WSL2 specifically is the browser-based dashboard chat pane (it uses a POSIX PTY — classic CLI and gateway both run natively).

After installation:

```bash
source ~/.bashrc    # reload shell (or: source ~/.zshrc)
hermes              # start chatting!
```

---

## Getting Started

```bash
hermes              # Interactive CLI — start a conversation
hermes model        # Choose your LLM provider and model
hermes tools        # Configure which tools are enabled
hermes config set   # Set individual config values
hermes gateway      # Start the messaging gateway (Telegram, Discord, etc.)
hermes setup        # Run the full setup wizard (configures everything at once)
hermes claw migrate # Migrate from OpenClaw (if coming from OpenClaw)
hermes update       # Update to the latest version
hermes doctor       # Diagnose any issues
```

📖 **[Full documentation →](https://hermes-agent.nousresearch.com/docs/)**

## CLI vs Messaging Quick Reference

Hermes has two entry points: start the terminal UI with `hermes`, or run the gateway and talk to it from Telegram, Discord, Slack, WhatsApp, Signal, or Email. Once you're in a conversation, many slash commands are shared across both interfaces.

| Action | CLI | Messaging platforms |
|---------|-----|---------------------|
| Start chatting | `hermes` | Run `hermes gateway setup` + `hermes gateway start`, then send the bot a message |
| Start fresh conversation | `/new` or `/reset` | `/new` or `/reset` |
| Change model | `/model [provider:model]` | `/model [provider:model]` |
| Set a personality | `/personality [name]` | `/personality [name]` |
| Retry or undo the last turn | `/retry`, `/undo` | `/retry`, `/undo` |
| Compress context / check usage | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [days]` |
| Browse skills | `/skills` or `/<skill-name>` | `/<skill-name>` |
| Interrupt current work | `Ctrl+C` or send a new message | `/stop` or send a new message |
| Platform-specific status | `/platforms` | `/status`, `/sethome` |

For the full command lists, see the [CLI guide](https://hermes-agent.nousresearch.com/docs/user-guide/cli) and the [Messaging Gateway guide](https://hermes-agent.nousresearch.com/docs/user-guide/messaging).

---

## Documentation

All documentation lives at **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**:

| Section | What's Covered |
|---------|---------------|
| [Quickstart](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart) | Install → setup → first conversation in 2 minutes |
| [CLI Usage](https://hermes-agent.nousresearch.com/docs/user-guide/cli) | Commands, keybindings, personalities, sessions |
| [Configuration](https://hermes-agent.nousresearch.com/docs/user-guide/configuration) | Config file, providers, models, all options |
| [Messaging Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) | Telegram, Discord, Slack, WhatsApp, Signal, Home Assistant |
| [Security](https://hermes-agent.nousresearch.com/docs/user-guide/security) | Command approval, DM pairing, container isolation |
| [Tools & Toolsets](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools) | 40+ tools, toolset system, terminal backends |
| [Skills System](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) | Procedural memory, Skills Hub, creating skills |
| [Memory](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory) | Persistent memory, user profiles, best practices |
| [MCP Integration](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp) | Connect any MCP server for extended capabilities |
| [Cron Scheduling](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron) | Scheduled tasks with platform delivery |
| [Context Files](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files) | Project context that shapes every conversation |
| [Architecture](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture) | Project structure, agent loop, key classes |
| [Contributing](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) | Development setup, PR process, code style |
| [CLI Reference](https://hermes-agent.nousresearch.com/docs/reference/cli-commands) | All commands and flags |
| [Environment Variables](https://hermes-agent.nousresearch.com/docs/reference/environment-variables) | Complete env var reference |

---

## Migrating from OpenClaw

If you're coming from OpenClaw, Hermes can automatically import your settings, memories, skills, and API keys.

**During first-time setup:** The setup wizard (`hermes setup`) automatically detects `~/.openclaw` and offers to migrate before configuration begins.

**Anytime after install:**

```bash
hermes claw migrate              # Interactive migration (full preset)
hermes claw migrate --dry-run    # Preview what would be migrated
hermes claw migrate --preset user-data   # Migrate without secrets
hermes claw migrate --overwrite  # Overwrite existing conflicts
```

What gets imported:
- **SOUL.md** — persona file
- **Memories** — MEMORY.md and USER.md entries
- **Skills** — user-created skills → `~/.hermes/skills/openclaw-imports/`
- **Command allowlist** — approval patterns
- **Messaging settings** — platform configs, allowed users, working directory
- **API keys** — allowlisted secrets (Telegram, OpenRouter, OpenAI, Anthropic, ElevenLabs)
- **TTS assets** — workspace audio files
- **Workspace instructions** — AGENTS.md (with `--workspace-target`)

See `hermes claw migrate --help` for all options, or use the `openclaw-migration` skill for an interactive agent-guided migration with dry-run previews.

---

## Contributing

We welcome contributions! See the [Contributing Guide](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) for development setup, code style, and PR process.

Quick start for contributors — clone and go with `setup-hermes.sh`:

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
./setup-hermes.sh     # installs uv, creates venv, installs .[all], symlinks ~/.local/bin/hermes
./hermes              # auto-detects the venv, no need to `source` first
```

Manual path (equivalent to the above):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

> **RL Training (optional):** The RL/Atropos integration (`environments/`) — see [`CONTRIBUTING.md`](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md#development-setup) for the full setup.

---

## Community

- 💬 [Discord](https://discord.gg/NousResearch)
- 📚 [Skills Hub](https://agentskills.io)
- 🐛 [Issues](https://github.com/NousResearch/hermes-agent/issues)
- 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw) — Community WeChat bridge: Run Hermes Agent and OpenClaw on the same WeChat account.

---

## License

MIT — see [LICENSE](LICENSE).

Built by [Nous Research](https://nousresearch.com).

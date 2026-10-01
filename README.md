# Aro Agent

<p align="center">
  <b>The Aro family, by samjuniors</b><br/>
  <a href="#the-aro-family">Aro Agent</a> · <a href="#the-aro-family">Aro CLI</a> · <a href="#the-aro-family">Aro Desktop</a> · <a href="#the-aro-family">Aro Harness</a>
</p>
<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://github.com/samjuniors/SamAgent"><img src="https://img.shields.io/badge/Fork%20of-Hermes%20Agent%20(Nous%20Research)-7c6bff?style=for-the-badge" alt="Fork of Hermes Agent"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/Lang-中文-red?style=for-the-badge" alt="中文"></a>
  <a href="README.ur-pk.md"><img src="https://img.shields.io/badge/Lang-اردو-green?style=for-the-badge" alt="اردو"></a>
  <a href="README.es.md"><img src="https://img.shields.io/badge/Lang-Español-orange?style=for-the-badge" alt="Español"></a>
</p>

**The self-improving AI agent, maintained by [samjuniors](https://github.com/samjuniors).**
Aro is the only agent with a built-in learning loop — it creates skills from experience, improves
them during use, nudges itself to persist knowledge, searches its own past conversations, and builds
a deepening model of who you are across sessions. Run it on a $5 VPS, a GPU cluster, or serverless
infrastructure that costs nearly nothing when idle. It's not tied to your laptop — talk to it from
Telegram while it works on a cloud VM.

> **Attribution (MIT):** Aro Agent is a fork of
> [Hermes Agent](https://github.com/NousResearch/hermes-agent) by
> [Nous Research](https://nousresearch.com), used under the MIT License. See
> [NOTICE](NOTICE). Upstream "Hermes" marks belong to their owners.

## The Aro family

| Product | What it is |
|---|---|
| **Aro Agent** | The core: a self-improving agent with 64 tools, 39 model providers, and a learning loop (memory, skills, session search, user modeling). |
| **Aro CLI** | The terminal surface: REPL, Ink TUI, and ~62 command groups. `aro` is the primary entry point (the upstream `hermes` alias still works). |
| **Aro Desktop** | The native desktop app (macOS / Windows / Linux) — chat with tool timelines, terminal, live subagents, git review, and the new Aro Workbench UI. |
| **Aro Harness** | The orchestration layer: subagents, parallel runs in worktrees, best-of-N on serverless backends, scheduled automations, and the ~30-platform messaging gateway. |

Use any model you want — Nous Portal, OpenRouter, OpenAI, Anthropic, local Ollama, and
[30+ more](https://hermes-agent.nousresearch.com/docs/integrations/providers). Switch with
`aro model` — no code changes, no lock-in.

<table>
<tr><td><b>A real terminal interface</b></td><td>Full TUI with multiline editing, slash-command autocomplete, conversation history, interrupt-and-redirect, and streaming tool output.</td></tr>
<tr><td><b>Lives where you do</b></td><td>Telegram, Discord, Slack, WhatsApp, Signal, and more — all from a single gateway process. Voice memo transcription, cross-platform conversation continuity.</td></tr>
<tr><td><b>A closed learning loop</b></td><td>Agent-curated memory with periodic nudges. Autonomous skill creation after complex tasks. Skills self-improve during use. FTS5 session search with LLM summarization for cross-session recall. Honcho dialectic user modeling. Compatible with the agentskills.io open standard.</td></tr>
<tr><td><b>Scheduled automations</b></td><td>Built-in cron scheduler with delivery to any platform. Daily reports, nightly backups, weekly audits — all in natural language, running unattended.</td></tr>
<tr><td><b>Delegates and parallelizes</b></td><td>Spawn isolated subagents for parallel workstreams. Write Python scripts that call tools via RPC, collapsing multi-step pipelines into zero-context-cost turns.</td></tr>
<tr><td><b>Runs anywhere, not just your laptop</b></td><td>Seven terminal backends — local, Docker, SSH, Singularity, Modal, Daytona, and Vercel Sandbox. Serverless persistence hibernates when idle and wakes on demand, costing nearly nothing between sessions.</td></tr>
<tr><td><b>Research-ready</b></td><td>Batch trajectory generation, trajectory compression for training the next generation of tool-calling models.</td></tr>
</table>

---

## Quick Install

> **Note:** the install one-liners below still point at upstream Nous Research
> infrastructure (see the roadmap). Prefer `git clone` + `pipx install .` from
> this repo for a fully samjuniors-served install.

### From this fork (recommended)

```bash
git clone https://github.com/samjuniors/SamAgent.git
cd SamAgent
pipx install .
aro            # or: hermes (upstream alias kept)
```

### Upstream installer (Linux, macOS, WSL2)

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

### Windows (native, PowerShell)

```powershell
irm https://hermes-agent.nousresearch.com/install.ps1 | iex
```

---

## Getting Started

```bash
aro                 # Interactive CLI — start a conversation (alias: hermes)
aro model           # Choose your LLM provider and model
aro tools           # Configure which tools are enabled
aro config set      # Set individual config values
aro config get      # Print individual config values
aro gateway         # Start the messaging gateway (Telegram, Discord, etc.)
aro setup           # Run the full setup wizard (configures everything at once)
aro update          # Update to the latest version
aro doctor          # Diagnose any issues
```

📖 **Full documentation →** the docs site still points at upstream (see roadmap);
this repo's `website/` and `docs/` carry the current Aro material.

---

## Aro roadmap

| Phase | Status |
|---|---|
| **Surface rebrand** — Aro identity on all user-facing strings, `aro` CLI entry points, Aro Desktop packaging identity | ✅ done |
| **New UI — Aro Workbench** — design-system-driven desktop UI (tool-timeline chat, diff-first review, parallel runs board, command palette) | 🚧 in progress — see [`prototype/`](prototype/) and [`docs/research/`](docs/research/) |
| **Deep rebrand** — `~/.aro`, `aro://`, module renames, `ARO_*` env | ⏳ planned — see [`docs/research/04-rebrand-log.md`](docs/research/04-rebrand-log.md) |
| **Own infrastructure** — signing certs, update feed, store identity, install scripts under samjuniors | ⏳ planned |

---

## Contributing

We welcome contributions! See the [Contributing Guide](CONTRIBUTING.md) for development setup,
code style, and the PR process. Start with the
[PM developer workflow](website/docs/reference/package-management.md#developer-workflow).

---

## Community

- 📚 [Skills Hub](https://agentskills.io)
- 🐛 [Issues](https://github.com/samjuniors/SamAgent/issues)
- 💬 Upstream community: [Nous Research Discord](https://discord.gg/NousResearch)

---

## License

MIT — see [LICENSE](LICENSE).

Aro Agent is maintained by [samjuniors](https://github.com/samjuniors).
Forked from [Hermes Agent](https://github.com/NousResearch/hermes-agent), built by
[Nous Research](https://nousresearch.com).

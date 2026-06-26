<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

<h1 align="center">Hermes Agent ☤</h1>

<p align="center">
  <strong>Your AI that actually remembers you.</strong>
</p>

<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Built%20by-Nous%20Research-blueviolet?style=for-the-badge" alt="Built by Nous Research"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/Lang-中文-red?style=for-the-badge" alt="中文"></a>
</p>

<p align="center">
  Most AI assistants forget you the moment the window closes.<br>
  Hermes doesn't. It learns your preferences, builds skills from experience,<br>
  and gets better every time you use it — across every device, every platform.
</p>

---

## ⚡ Get Started in 60 Seconds

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
source ~/.bashrc
hermes
```

> Works on **Linux, macOS, WSL2, and Termux**. [Windows PowerShell (beta)](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart) and [Termux](https://hermes-agent.nousresearch.com/docs/getting-started/termux) guides also available.

---

## 🧠 Why Hermes?

Other agents are stateless tools — powerful in the moment, blank the next session. Hermes is different: it's the first agent with a **closed learning loop**.

<table>
<tr>
<td width="50%">

### 🔄 It Learns and Improves
Every complex task becomes a **reusable skill**. Every correction becomes a **persistent memory**. It searches its own past conversations and builds a deepening model of who you are — not a generic user, but *you*.

</td>
<td width="50%">

### 🌐 It Lives Where You Do
Telegram, Discord, Slack, WhatsApp, Signal, Email — all from one process. Start a conversation on your phone, continue it at your desk. Voice memos just work.

</td>
</tr>
<tr>
<td width="50%">

### 🔌 Any Model, Zero Lock-in
Use Claude, GPT, Gemini, Llama, Qwen, or [300+ models](https://portal.nousresearch.com) — switch with a single command. Bring your own keys or use [Nous Portal](https://portal.nousresearch.com) to skip the API-key juggling entirely.

</td>
<td width="50%">

### 🏗️ Runs Anywhere, Not Just Your Laptop
Seven terminal backends — local, Docker, SSH, Modal, Daytona, Singularity, and Vercel Sandbox. Deploy on a **$5 VPS** or a GPU cluster. Serverless options hibernate when idle — **near-zero cost** between sessions.

</td>
</tr>
</table>

---

## ✨ What You Can Do With It

<details open>
<summary><b>🤖 Autonomous Workflows</b></summary>
<br>

- **Scheduled automations** — daily reports, nightly backups, weekly audits. Write them in plain English, deliver results to any platform.
- **Parallel subagents** — spawn isolated workers for independent tasks. They run in parallel, report back, and never pollute your context.
- **Execute code** — write Python scripts that orchestrate multi-step pipelines with zero context cost.

</details>

<details open>
<summary><b>💬 A Real Terminal Experience</b></summary>
<br>

- Full TUI with multiline editing, slash-command autocomplete, and streaming output
- Interrupt and redirect mid-task with `Ctrl+C` or just send a new message
- Conversation history, session search, context compression
- Beautiful themes via the skin engine

</details>

<details open>
<summary><b>🔧 40+ Built-in Tools</b></summary>
<br>

File editing, web search, browser automation, image generation, text-to-speech, calendar, Git/GitHub, MCP server integration, and more — all discoverable with `hermes tools`.

</details>

<details open>
<summary><b>🧪 Research-Ready</b></summary>
<br>

Batch trajectory generation and compression for training the next generation of tool-calling models. Built by [Nous Research](https://nousresearch.com) — the team behind Hermes, Genstruct, and OpenHermes.

</details>

---

## 🎯 One Setup for Everything — Nous Portal

Skip the API-key collection. **[Nous Portal](https://portal.nousresearch.com)** gives you models, web search, image gen, TTS, and a cloud browser — all under one subscription.

```bash
hermes setup --portal    # OAuth login → provider set → tools wired. Done.
```

You can still bring your own keys per-tool anytime — it's not all-or-nothing.

→ [Tool Gateway docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway)

---

## 🛠️ Quick Reference

```bash
hermes              # Start chatting
hermes model        # Pick your LLM
hermes tools        # Configure tools
hermes gateway      # Connect messaging platforms
hermes setup        # Full setup wizard
hermes update       # Update to latest
hermes doctor       # Diagnose issues
```

| What you want | CLI | Messaging |
|---|---|---|
| Fresh conversation | `/new` | `/new` |
| Change model | `/model claude-sonnet-4` | `/model claude-sonnet-4` |
| Set personality | `/personality snarky` | `/personality snarky` |
| Browse skills | `/skills` | `/<skill-name>` |
| Stop current task | `Ctrl+C` | `/stop` |

---

## 📚 Documentation

Everything you need → **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**

| | |
|---|---|
| [**Quickstart**](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart) — Install to first conversation in 2 minutes | [**Tools & Toolsets**](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools) — 40+ tools, terminal backends |
| [**CLI Usage**](https://hermes-agent.nousresearch.com/docs/user-guide/cli) — Commands, keybindings, sessions | [**Skills System**](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) — Procedural memory, Skills Hub |
| [**Configuration**](https://hermes-agent.nousresearch.com/docs/user-guide/configuration) — Providers, models, all options | [**Memory**](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory) — Persistent memory, user profiles |
| [**Messaging Gateway**](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) — Telegram, Discord, Slack, WhatsApp, Signal | [**MCP Integration**](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp) — Connect any MCP server |
| [**Security**](https://hermes-agent.nousresearch.com/docs/user-guide/security) — Approval, DM pairing, isolation | [**Cron Scheduling**](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron) — Scheduled tasks with delivery |
| [**Architecture**](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture) — Project structure, agent loop | [**Contributing**](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) — Dev setup, PR process |

---

## 🤝 Contributing

We'd love your help! [Contributing Guide →](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing)

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
./setup-hermes.sh     # installs everything, creates venv, symlinks hermes
./hermes              # you're in
```

---

## 🌍 Community & Ecosystem

- 💬 **[Discord](https://discord.gg/NousResearch)** — chat, get help, share what you've built
- 📚 **[Skills Hub](https://agentskills.io)** — browse and share reusable skills
- 🐛 **[Issues](https://github.com/NousResearch/hermes-agent/issues)** — bug reports and feature requests
- 🔌 **[computer-use-linux](https://github.com/avifenesh/computer-use-linux)** — Linux desktop control via MCP
- 🔌 **[HermesClaw](https://github.com/AaronWong1999/hermesclaw)** — Community WeChat bridge

---

<details>
<summary><b>📦 Migrating from OpenClaw?</b></summary>
<br>

```bash
hermes claw migrate              # Interactive migration
hermes claw migrate --dry-run    # Preview first
```

Imports your persona, memories, skills, API keys, messaging settings, and more. The setup wizard (`hermes setup`) auto-detects `~/.openclaw` and offers to migrate. See `hermes claw migrate --help` for all options.

</details>

---

<p align="center">
  <b>MIT License</b> · Built with ☤ by <a href="https://nousresearch.com">Nous Research</a>
</p>

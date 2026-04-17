<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤

<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Built%20by-Nous%20Research-blueviolet?style=for-the-badge" alt="Built by Nous Research"></a>
</p>

**The AI that never forgets you.**

*Yesterday you told it about your project. Today it asked how it went. Tomorrow it will remember what you forgot.*

Built by [Nous Research](https://nousresearch.com) · **95,445 Stars** · **13,347 Forks** · MIT License

---

## ✦ 79 Skills. One Agent. All Yours.

Type `/` to unlock production-ready capabilities across 8 categories:

| Category | Highlights |
|----------|-----------|
| 🔍 Research | arXiv paper search, blog monitoring, Polymarket data |
| 🎨 Creative | ASCII art, music generation, video production |
| 💻 Code | Full GitHub workflow, TDD, systematic debugging |
| 🧠 MLOps | Axolotl fine-tuning, vLLM serving, GGUF quantization |
| 📡 Social | Twitter/X operations, newsletter publishing |
| 🏠 Home | Philips Hue control, Obsidian notes, Gmail |
| 🍎 Apple | iMessage, Reminders, FindMy, Notes |
| 📧 Email | Terminal email via IMAP/SMTP (Himalaya) |

Skills load on demand — pay only for what you use.  
**[Browse all 79 skills →](https://agentskills.io)**

---

Use any model you want — [Nous Portal](https://portal.nousresearch.com), [OpenRouter](https://openrouter.ai) (200+ models), [Xiaomi MiMo](https://platform.xiaomimimo.com), [z.ai/GLM](https://z.ai), [Kimi/Moonshot](https://platform.moonshot.ai), [MiniMax](https://www.minimax.io), [Hugging Face](https://huggingface.co), OpenAI, or your own endpoint. Switch with `hermes model` — no code changes, no lock-in.

<table>
<tr><td><b>A real terminal interface</b></td><td>Full TUI with multiline editing, slash-command autocomplete, conversation history, interrupt-and-redirect, and streaming tool output.</td></tr>
<tr><td><b>Lives where you do</b></td><td>Telegram, Discord, Slack, WhatsApp, Signal, and CLI — all from a single gateway process. Voice memo transcription, cross-platform conversation continuity.</td></tr>
<tr><td><b>A closed learning loop</b></td><td>Agent-curated memory with periodic nudges. Autonomous skill creation after complex tasks. Skills self-improve during use. FTS5 session search with LLM summarization for cross-session recall. <a href="https://github.com/plastic-labs/honcho">Honcho</a> dialectic user modeling. Compatible with the <a href="https://agentskills.io">agentskills.io</a> open standard.</td></tr>
<tr><td><b>Scheduled automations</b></td><td>Built-in cron scheduler with delivery to any platform. Daily reports, nightly backups, weekly audits — all in natural language, running unattended.</td></tr>
<tr><td><b>Delegates and parallelizes</b></td><td>Spawn isolated subagents for parallel workstreams. Write Python scripts that call tools via RPC, collapsing multi-step pipelines into zero-context-cost turns.</td></tr>
<tr><td><b>Runs anywhere, not just your laptop</b></td><td>Six terminal backends — local, Docker, SSH, Daytona, Singularity, and Modal. Daytona and Modal offer serverless persistence — your agent's environment hibernates when idle and wakes on demand, costing nearly nothing between sessions. Run it on a $5 VPS or a GPU cluster.</td></tr>
<tr><td><b>Research-ready</b></td><td>Batch trajectory generation, Atropos RL environments, trajectory compression for training the next generation of tool-calling models.</td></tr>
</table>

---

## Quick Install

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

Works on Linux, macOS, WSL2, and Android via Termux.

> **Android / Termux:** See the [Termux guide](https://hermes-agent.nousresearch.com/docs/getting-started/termux).

After installation:

```bash
source ~/.bashrc && hermes
```

📖 **[Full documentation →](https://hermes-agent.nousresearch.com/docs/)**  
📺 **[YouTube tutorials →](https://www.youtube.com/@NousResearch)**

---

## Why Hermes?

| | Hermes Agent | Generic AI Chatbot |
|---|---|---|
| **Memory** | Remembers you across sessions | Forgets everything |
| **Skills** | 79 production-ready capabilities | None |
| **Platforms** | 8+ (Telegram, Discord, Feishu, CLI…) | Web only |
| **Self-improving** | Creates & updates skills from experience | Static |
| **Runs on** | $5 VPS to GPU cluster | Cloud only |
| **Open source** | ✅ MIT | ❌ Proprietary |

---

## Getting Started

```bash
hermes              # Interactive CLI — start a conversation
hermes model        # Choose your LLM provider and model
hermes tools        # Configure which tools are enabled
hermes gateway      # Start the messaging gateway (Telegram, Discord, etc.)
hermes setup        # Run the full setup wizard
hermes doctor       # Diagnose any issues
```

📖 **[Full documentation →](https://hermes-agent.nousresearch.com/docs/)**

---

## Migrating from OpenClaw

```bash
hermes claw migrate              # Interactive migration
hermes claw migrate --dry-run   # Preview what would be migrated
hermes claw migrate --preset user-data   # Migrate without secrets
```

What gets imported: **SOUL.md**, **Memories**, **Skills**, **Command allowlist**, **Messaging settings**, **API keys**, **TTS assets**, **Workspace instructions**.

---

## Contributing

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
uv venv venv --python 3.11 && source venv/bin/activate
uv pip install -e ".[all,dev]"
python -m pytest tests/ -q
```

---

## Community

<p align="center">
  <img src="https://img.shields.io/badge/Discord-95,445%20members-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord">
  <img src="https://img.shields.io/badge/GitHub-95,445%20Stars-gray?style=for-the-badge&logo=github&logoColor=white" alt="Stars">
  <img src="https://img.shields.io/badge/Forks-13,347-gray?style=for-the-badge&logo=github" alt="Forks">
</p>

💬 [Join our Discord](https://discord.gg/NousResearch) — 95,445+ members  
📣 [Official Blog](https://hermes-agent.nousresearch.com)  
🐛 [Issues](https://github.com/NousResearch/hermes-agent/issues)  
💡 [Discussions](https://github.com/NousResearch/hermes-agent/discussions)  
🔌 [HermesClaw WeChat Bridge](https://github.com/AaronWong1999/hermesclaw)

---

## License

MIT — see [LICENSE](LICENSE).

Built by [Nous Research](https://nousresearch.com).

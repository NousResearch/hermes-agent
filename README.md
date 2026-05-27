<p align="center">
  <img src="assets/banner.png" alt="Autolycus" width="100%">
</p>

# Autolycus 🦊

<p align="center">
  <a href="https://github.com/waym0reom3ga/autolycus-agent"><img src="https://img.shields.io/badge/GitHub-waym0reom3ga/autolycus--agent-6e5494?style=for-the-badge&logo=github" alt="GitHub"></a>
  <a href="https://github.com/waym0reom3ga/autolycus-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-LGPL%20v2.1-blue?style=for-the-badge" alt="License: LGPL v2.1"></a>
  <img src="https://img.shields.io/badge/FreeBSD-AB1D2E?style=for-the-badge&logo=freebsd&logoColor=white" alt="FreeBSD">
  <img src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black" alt="Linux">
  <img src="https://img.shields.io/badge/macOS-000000?style=for-the-badge&logo=apple&logoColor=white" alt="macOS">
  <img src="https://img.shields.io/badge/Technetia%20Inc-0066cc?style=for-the-badge" alt="Technetia Inc">
</p>

> 🎉 **The World's First AI Agent for FreeBSD** — Autolycus runs natively on FreeBSD, Linux, and macOS, delivering full terminal execution, file operations, and intelligent automation. An independent project by **Technetia Inc**. Not affiliated with Nous Research or the original Hermes Agent.

**The self-improving AI agent.** It creates skills from experience, improves them during use, nudges itself to persist knowledge, searches its own past conversations, and builds a deepening model of who you are across sessions. Run it on FreeBSD — natively, without emulation or containers.

Use any model you want — A local lmstudio or ollama server, or some hosted services: [Nous Portal](https://portal.nousresearch.com), [OpenRouter](https://openrouter.ai) (200+ models), [z.ai/GLM](https://z.ai), [Kimi/Moonshot](https://platform.moonshot.ai), [MiniMax](https://www.minimax.io), OpenAI, or your own endpoint. Switch with `hermes model` — no code changes, no lock-in, not even a way to login or register! We don't keep tabs on what you do, we're busy enough.

<table>
<tr><td><b>A real terminal interface</b></td><td>Full TUI with multiline editing, slash-command autocomplete, conversation history, interrupt-and-redirect, and streaming tool output.</td></tr>
<tr><td><b>Lives where you do</b></td><td>Telegram, Discord, Slack, WhatsApp, Signal — all from a single gateway process. Cross-platform conversation continuity is the dream.</td></tr>
<tr><td><b>A closed learning loop</b></td><td>Agent-curated memory with periodic nudges. Autonomous skill creation after complex tasks. Skills self-improve during use. FTS5 session search with LLM summarization for cross-session recall. <a href="https://github.com/plastic-labs/honcho">Honcho</a> dialectic user modeling. Compatible with the <a href="https://agentskills.io">agentskills.io</a> open standard.</td></tr>
<tr><td><b>Scheduled automations</b></td><td>Built-in cron scheduler with delivery to any platform. Daily reports, nightly backups, weekly audits — all in natural language, running unattended.</td></tr>
<tr><td><b>Delegates and parallelizes</b></td><td>Spawn isolated subagents for parallel workstreams. Write Python scripts that call tools via RPC, collapsing multi-step pipelines into zero-context-cost turns.</td></tr>
<tr><td><b>Native FreeBSD execution</b></td><td>Built from the ground up for FreeBSD — no Linux emulation, no containers required. Uses ptyprocess for reliable terminal I/O across all Unix platforms.</td></tr>
</table>

---

## Quick Install

⚠️ **Autolycus runs on FreeBSD, Linux, and macOS.** The automated installer handles OS detection and platform-specific setup.

### Prerequisites

- **Tested platforms:**
  - FreeBSD 14.x / 15.0 (amd64)
  - Linux (amd64) — Arch Linux, Ubuntu
  - Linux (aarch64) — Armbian on Radxa Rock 5B and PostMarketOS on Pinephone Pro validated
  - macOS (reported working, not extensively tested)
- Rust/Cargo installed (`pkg install rust` on FreeBSD, `rustup` on Linux/macOS)
- Python 3.11+ available
- Optional: add a brain to your AI Agent with a persistent database (`pkg install py311-sqlite` on FreeBSD)

### Installation

```bash
# Clone the repository
git clone https://github.com/waym0reom3ga/autolycus-agent.git

# Run the install script (POSIX sh compatible)
sh autolycus-agent/scripts/install-autolycus.sh
```

The installer will:
1. Detect your operating system (FreeBSD / Linux / macOS)
2. Install uv via cargo (skips if already present)
3. Create a virtual environment with Python 3.11
4. Install dependencies with OS-appropriate extras (voice excluded on FreeBSD)
5. Set up the `hermes` CLI command in `~/.local/bin`
6. Create config files from templates
7. Sync bundled skills

After installation:

```bash
source ~/.bashrc    # reload shell (or restart your terminal)
hermes setup        # configure API keys and model provider
hermes              # start chatting!
```

> **Note:** If you see "Permission denied" when creating the virtual environment, it may be from a previous installation attempt with different ownership. Simply re-run the installer — it will handle it.

### Manual Installation (FreeBSD)

If you prefer to install step by step:

```bash
# Clone the repository
git clone https://github.com/waym0reom3ga/autolycus-agent.git
cd autolycus-agent

# Build uv from source (~4 minute compile time, no FreeBSD binary available)
cargo install uv
export PATH="$HOME/.cargo/bin:$PATH"

# Create virtual environment
uv venv venv --python 3.11

# Activate the virtual environment
. venv/bin/activate
# Or: source venv/bin/activate  # bash/zsh only

# Install dependencies (excluding voice which has no FreeBSD wheels)
uv pip install -e ".[modal,daytona,messaging,cron,cli,dev,tts-premium,slack,honcho,mcp]"

# Add hermes to PATH
mkdir -p ~/.local/bin
ln -sf $(pwd)/venv/bin/hermes ~/.local/bin/hermes
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

### Optional Dependencies

- **Clipboard tools**: Install xclip for clipboard support (`pkg install xclip`)
- **Voice transcription**: Local STT (faster-whisper) is unavailable on FreeBSD due to missing ctranslate2 wheels. Use cloud-based alternatives by setting `GROQ_API_KEY` or `VOICE_TOOLS_OPENAI_KEY`.

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

📖 **Documentation:** See [Hermes Agent docs](https://hermes-agent.nousresearch.com/docs/) for reference (Autolycus is API-compatible).

---

## Skip the API-key collection — Nous Portal

Hermes works with whatever provider you want — that's not changing. But if you'd rather not collect five separate API keys for the model, web search, image generation, TTS, and a cloud browser, **[Nous Portal](https://portal.nousresearch.com)** covers all of them under one subscription:

- **300+ models** — pick any of them with `/model <name>`
- **Tool Gateway** — web search (Firecrawl), image generation (FAL), text-to-speech (OpenAI), cloud browser (Browser Use), all routed through your sub. No extra accounts.

One command from a fresh install:

```bash
hermes setup --portal
```

That logs you in via OAuth, sets Nous as your provider, and turns on the Tool Gateway. Check what's wired up any time with `hermes portal status`. Full details on the [Tool Gateway docs page](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway).

You can still bring your own keys per-tool whenever you want — the gateway is per-backend, not all-or-nothing.

---

## CLI vs Messaging Quick Reference

Autolycus has two entry points: start the terminal UI with `hermes`, or run the gateway and talk to it from Telegram, Discord, Slack, WhatsApp, Signal, or Email. Once you're in a conversation, many slash commands are shared across both interfaces.

| Action | CLI | Messaging platforms |
|---------|-----|---------------------|
| Start chatting | `hermes` | Run `hermes gateway setup` + `hermes gateway start`, then send the bot a message |
| Start fresh conversation | `/new` or `/reset` | `/new` or `/reset` |
| Change model | `/model [provider:model]` | `/model [provider:model]` |
| Set a personality | `/personality [name]` | `/personality [name]` |
| Retry or undo the last turn | `/retry`, `/undo` | `/retry`, `/undo` |
| Compress context / check usage | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [days]` |
| Browse skills | `/skills` or `/<skill-name>` | `/skills` or `/<skill-name>` |
| Interrupt current work | `Ctrl+C` or send a new message | `/stop` or send a new message |
| Platform-specific status | `/platforms` | `/status`, `/sethome` |

For the full command lists, see the [CLI guide](https://hermes-agent.nousresearch.com/docs/user-guide/cli) and the [Messaging Gateway guide](https://hermes-agent.nousresearch.com/docs/user-guide/messaging).

---

## Documentation

Autolycus is API-compatible with Hermes Agent. For full documentation, refer to the **[Hermes Agent docs](https://hermes-agent.nousresearch.com/docs/)**.

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

> **Note:** Voice transcription (faster-whisper) is unavailable on FreeBSD due to missing ctranslate2 wheels — use cloud-based STT instead. On Linux/macOS, voice tools work out of the box.

---

## Migrating from Hermes/OpenClaw

If you're coming from Hermes or OpenClaw, Autolycus can automatically import your settings, memories, skills, and API keys.

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

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and PR process.

Quick start for contributors (FreeBSD / Linux / macOS):

```bash
git clone https://github.com/waym0reom3ga/autolycus-agent.git
cd autolycus-agent
./scripts/install-autolycus.sh  # handles OS detection and setup
source venv/bin/activate
python -m pytest tests/ -q
```

> **Note:** Voice transcription (faster-whisper) is unavailable on FreeBSD due to missing ctranslate2 wheels — use cloud-based STT instead. On Linux/macOS, voice tools work out of the box. Autolycus is cross-platform and runs natively on FreeBSD, Linux, and macOS.

---

## Community

- 🐛 [Issues](https://github.com/waym0reom3ga/autolycus-agent/issues) — Report bugs or request features
- 📧 **Technetia Inc** — Contact us for enterprise support or inquiries
- 🔌 [computer-use-linux](https://github.com/avifenesh/computer-use-linux) — Linux desktop-control MCP server for Autolycus and other MCP hosts, with AT-SPI accessibility trees, Wayland/X11 input, screenshots, and compositor window targeting.
- 🔌 [AutolycusClaw](https://github.com/AaronWong1999/hermesclaw) — Community WeChat bridge: Run Autolycus Agent on WeChat.

---

## License

LGPL v2.1 — see [LICENSE](LICENSE).

An independent project by **Technetia Inc**.  
Built on the Hermes Agent architecture.

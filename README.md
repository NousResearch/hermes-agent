# HT AI Agent

**The self-improving AI agent.** It's an agent with a built-in learning loop — it creates skills from experience, improves them during use, nudges itself to persist knowledge, searches its own past conversations, and builds a deepening model of who you are across sessions. Run it on a $5 VPS, a GPU cluster, or serverless infrastructure that costs nearly nothing when idle. It's not tied to your laptop — talk to it from Telegram while it works on a cloud VM.

Use any model you want — OpenRouter, OpenAI, your own endpoint, and many others. Switch with `ht model` — no code changes, no lock-in.

> **Fork notice:** HT AI Agent is a rebranded fork of [Hermes Agent](https://github.com/NousResearch/hermes-agent) by [Nous Research](https://nousresearch.com), released under the MIT License. All credit for the underlying agent architecture, tooling, and learning loop goes to the upstream project and its contributors.

<table>
<tr><td><b>A real terminal interface</b></td><td>Full TUI with multiline editing, slash-command autocomplete, conversation history, interrupt-and-redirect, and streaming tool output.</td></tr>
<tr><td><b>Lives where you do</b></td><td>Telegram, Discord, Slack, WhatsApp, Signal, and CLI — all from a single gateway process. Voice memo transcription, cross-platform conversation continuity.</td></tr>
<tr><td><b>A closed learning loop</b></td><td>Agent-curated memory with periodic nudges. Autonomous skill creation after complex tasks. Skills self-improve during use. FTS5 session search with LLM summarization for cross-session recall. <a href="https://github.com/plastic-labs/honcho">Honcho</a> dialectic user modeling. Compatible with the <a href="https://agentskills.io">agentskills.io</a> open standard.</td></tr>
<tr><td><b>Scheduled automations</b></td><td>Built-in cron scheduler with delivery to any platform. Daily reports, nightly backups, weekly audits — all in natural language, running unattended.</td></tr>
<tr><td><b>Delegates and parallelizes</b></td><td>Spawn isolated subagents for parallel workstreams. Write Python scripts that call tools via RPC, collapsing multi-step pipelines into zero-context-cost turns.</td></tr>
<tr><td><b>Runs anywhere, not just your laptop</b></td><td>Six terminal backends — local, Docker, SSH, Singularity, Modal, and Daytona. Daytona and Modal offer serverless persistence — your agent's environment hibernates when idle and wakes on demand, costing nearly nothing between sessions.</td></tr>
<tr><td><b>Research-ready</b></td><td>Batch trajectory generation, trajectory compression for training the next generation of tool-calling models.</td></tr>
</table>

---

## Quick Install

### Linux, macOS, WSL2, Termux

Clone the repository and run the installer:

```bash
git clone https://github.com/uaixo/awesome-hermes-agent.git
bash awesome-hermes-agent/scripts/install.sh
```

The installer sets up uv, Python 3.11, Node.js, ripgrep, and ffmpeg, creates a managed virtual environment, and links the `ht` command onto your PATH.

### Windows (native, PowerShell)

Native Windows runs HT AI Agent without WSL — CLI, gateway, TUI, and tools all work natively. Run this in PowerShell from a clone of the repository:

```powershell
git clone https://github.com/uaixo/awesome-hermes-agent.git
powershell -ExecutionPolicy Bypass -File awesome-hermes-agent\scripts\install.ps1
```

The Windows installer handles everything: uv, Python 3.11, Node.js, ripgrep, ffmpeg, and a portable Git Bash (MinGit, unpacked to `%LOCALAPPDATA%\hermes\git` — no admin required, isolated from any system Git install). If you already have Git installed, the installer detects it and uses that instead.

> **Android / Termux:** On Termux, the installer uses a curated `.[termux]` extra because the full `.[all]` extra currently pulls Android-incompatible voice dependencies.
>
> **Windows:** Native Windows install lives under `%LOCALAPPDATA%\hermes`; WSL2 installs under `~/.hermes` as on Linux.
>
> **Antivirus note:** If your antivirus flags the bundled `uv.exe`, it is a false positive on Astral's `uv` (the Rust Python package manager). Whitelist the install `bin` folder rather than a file hash — `uv` is updated over time and the hash changes.

After installation:

```bash
source ~/.bashrc    # reload shell (or: source ~/.zshrc)
ht                  # start chatting!
```

`hermes` still works everywhere as a legacy alias for `ht`.

---

## Getting Started

```bash
ht              # Interactive CLI — start a conversation
ht model        # Choose your LLM provider and model
ht tools        # Configure which tools are enabled
ht config set   # Set individual config values
ht gateway      # Start the messaging gateway (Telegram, Discord, etc.)
ht setup        # Run the full setup wizard (configures everything at once)
ht claw migrate # Migrate from OpenClaw (if coming from OpenClaw)
ht update       # Update to the latest version
ht doctor       # Diagnose any issues
```

Documentation sources live in this repository under `website/docs/` (buildable with Docusaurus from `website/`).

---

## CLI vs Messaging Quick Reference

HT AI Agent has two entry points: start the terminal UI with `ht`, or run the gateway and talk to it from Telegram, Discord, Slack, WhatsApp, Signal, or Email. Once you're in a conversation, many slash commands are shared across both interfaces.

| Action                         | CLI                                           | Messaging platforms                                                      |
| ------------------------------ | --------------------------------------------- | ------------------------------------------------------------------------ |
| Start chatting                 | `ht`                                          | Run `ht gateway setup` + `ht gateway start`, then send the bot a message |
| Start fresh conversation       | `/new` or `/reset`                            | `/new` or `/reset`                                                       |
| Change model                   | `/model [provider:model]`                     | `/model [provider:model]`                                                |
| Set a personality              | `/personality [name]`                         | `/personality [name]`                                                    |
| Retry or undo the last turn    | `/retry`, `/undo`                             | `/retry`, `/undo`                                                        |
| Compress context / check usage | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [days]`                                |
| Browse skills                  | `/skills` or `/<skill-name>`                  | `/<skill-name>`                                                          |
| Interrupt current work         | `Ctrl+C` or send a new message                | `/stop` or send a new message                                            |
| Platform-specific status       | `/platforms`                                  | `/status`, `/sethome`                                                    |

---

## Documentation

Documentation lives in `website/docs/` in this repository:

| Section                                              | What's Covered                                             |
| ---------------------------------------------------- | ---------------------------------------------------------- |
| `getting-started/quickstart`                         | Install → setup → first conversation in 2 minutes          |
| `user-guide/cli`                                     | Commands, keybindings, personalities, sessions             |
| `user-guide/configuration`                           | Config file, providers, models, all options                |
| `user-guide/messaging`                               | Telegram, Discord, Slack, WhatsApp, Signal, Home Assistant |
| `user-guide/security`                                | Command approval, DM pairing, container isolation          |
| `user-guide/features/tools`                          | 40+ tools, toolset system, terminal backends               |
| `user-guide/features/skills`                         | Procedural memory, Skills Hub, creating skills             |
| `user-guide/features/memory`                         | Persistent memory, user profiles, best practices           |
| `user-guide/features/mcp`                            | Connect any MCP server for extended capabilities           |
| `user-guide/features/cron`                           | Scheduled tasks with platform delivery                     |
| `developer-guide/architecture`                       | Project structure, agent loop, key classes                 |
| `reference/cli-commands`                             | All commands and flags                                     |
| `reference/environment-variables`                    | Complete env var reference                                 |

---

## Migrating from OpenClaw

If you're coming from OpenClaw, HT AI Agent can automatically import your settings, memories, skills, and API keys. The setup wizard (`ht setup`) automatically detects `~/.openclaw` and offers to migrate before configuration begins.

```bash
ht claw migrate              # Interactive migration (full preset)
ht claw migrate --dry-run    # Preview what would be migrated
ht claw migrate --preset user-data   # Migrate without secrets
ht claw migrate --overwrite  # Overwrite existing conflicts
```

See `ht claw migrate --help` for all options, or use the `openclaw-migration` skill for an interactive agent-guided migration with dry-run previews.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and PR process.

Quick start for contributors — use the installer, then work from the full git checkout it creates at `$HERMES_HOME/hermes-agent` (usually `~/.hermes/hermes-agent`). This matches the layout used by `ht update`, the managed venv, lazy dependencies, gateway, and docs tooling.

```bash
git clone https://github.com/uaixo/awesome-hermes-agent.git
bash awesome-hermes-agent/scripts/install.sh
cd "${HERMES_HOME:-$HOME/.hermes}/hermes-agent"
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

---

## Community

- 📚 [Skills Hub](https://agentskills.io)
- 🐛 [Issues](https://github.com/uaixo/awesome-hermes-agent/issues)

---

## License

MIT — see [LICENSE](LICENSE).

HT AI Agent is built on [Hermes Agent](https://github.com/NousResearch/hermes-agent) by [Nous Research](https://nousresearch.com) (MIT License).

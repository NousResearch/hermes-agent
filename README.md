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

Hermes is an AI agent you can actually live with.

It runs in your terminal, your messages, or a cheap little server somewhere. It remembers useful stuff, learns repeatable workflows as skills, can search old conversations, run tools, schedule jobs, delegate work to subagents, and generally keep going without needing your laptop open.

Use whatever model you want: [Nous Portal](https://portal.nousresearch.com), [OpenRouter](https://openrouter.ai), [NovitaAI](https://novita.ai), [NVIDIA NIM](https://build.nvidia.com), [Xiaomi MiMo](https://platform.xiaomimimo.com), [z.ai/GLM](https://z.ai), [Kimi/Moonshot](https://platform.moonshot.ai), [MiniMax](https://www.minimax.io), [Hugging Face](https://huggingface.co), OpenAI, or your own endpoint. Switch with `hermes model`. That's kind of the point.

## The gist

- **Terminal UI:** multiline editing, slash commands, history, interrupts, streaming tool output. Normal agent stuff, but less annoying.
- **Messaging:** talk to the same agent from Telegram, Discord, Slack, WhatsApp, Signal, email, or the CLI.
- **Memory + skills:** Hermes keeps durable notes, turns repeatable workflows into skills, and can improve those skills when they break.
- **Scheduled work:** daily briefings, backups, reminders, audits, whatever. Tell it once; it can run later.
- **Subagents:** split work into isolated agents when one brain is too crowded.
- **Runs wherever:** local, Docker, SSH, Singularity, Modal, Daytona, Vercel Sandbox. Laptop, VPS, GPU box — pick your vibe.
- **Research-friendly:** batch trajectory generation and compression are built in if you're training/evaluating tool-using models.

---

## Install

### Linux, macOS, WSL2, Termux

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

### Windows, native PowerShell — early beta

Native Windows works, but it is still the less-traveled path. If something feels cursed, WSL2 is still the chill option.

```powershell
iex (irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1)
```

The Windows installer brings its own basics: uv, Python 3.11, Node.js, ripgrep, ffmpeg, and a portable Git Bash if you do not already have Git. No admin required.

For Android / Termux, use the [Termux guide](https://hermes-agent.nousresearch.com/docs/getting-started/termux).

After install:

```bash
source ~/.bashrc    # or source ~/.zshrc
hermes              # start chatting
```

---

## First things to try

```bash
hermes              # open the interactive CLI
hermes model        # pick your provider/model
hermes tools        # choose enabled tools
hermes config set   # tweak config values
hermes gateway      # run Telegram/Discord/etc.
hermes setup        # full setup wizard
hermes claw migrate # bring over OpenClaw stuff
hermes update       # update Hermes
hermes doctor       # debug the environment
```

Docs are here: **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**

---

## CLI or messages, same idea

You can use Hermes directly in the terminal with `hermes`, or run the gateway and talk to it from Telegram, Discord, Slack, WhatsApp, Signal, or email.

Common commands work in both places:

- Start over: `/new` or `/reset`
- Change model: `/model [provider:model]`
- Change personality: `/personality [name]`
- Retry/undo: `/retry`, `/undo`
- Compress context / inspect usage: `/compress`, `/usage`, `/insights`
- Browse skills: `/skills` or `/<skill-name>`
- Interrupt work: `Ctrl+C` in CLI, or `/stop` / send another message in chat

Full guides:

- [CLI guide](https://hermes-agent.nousresearch.com/docs/user-guide/cli)
- [Messaging gateway guide](https://hermes-agent.nousresearch.com/docs/user-guide/messaging)

---

## Docs, if you want the full tour

- [Quickstart](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart) — install to first conversation
- [CLI usage](https://hermes-agent.nousresearch.com/docs/user-guide/cli) — commands, keybindings, sessions
- [Configuration](https://hermes-agent.nousresearch.com/docs/user-guide/configuration) — config file, providers, models
- [Messaging gateway](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) — Telegram, Discord, Slack, WhatsApp, Signal, Home Assistant
- [Security](https://hermes-agent.nousresearch.com/docs/user-guide/security) — approvals, DM pairing, containers
- [Tools & toolsets](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools) — tools, permissions, terminal backends
- [Skills](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) — reusable procedures for agents
- [Memory](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory) — persistent memory and user profiles
- [MCP](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp) — bring your own MCP servers
- [Cron](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron) — scheduled agent runs
- [Context files](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files) — repo/project instructions
- [Architecture](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture) — how the thing is wired
- [Contributing](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) — dev setup and PR flow
- [CLI reference](https://hermes-agent.nousresearch.com/docs/reference/cli-commands) — all commands
- [Environment variables](https://hermes-agent.nousresearch.com/docs/reference/environment-variables) — all env vars

---

## Coming from OpenClaw?

Hermes can import your OpenClaw setup.

First-time setup will notice `~/.openclaw` and offer to migrate. You can also do it later:

```bash
hermes claw migrate                    # interactive migration
hermes claw migrate --dry-run          # preview only
hermes claw migrate --preset user-data # skip secrets
hermes claw migrate --overwrite        # overwrite conflicts
```

It can bring over persona files, memories, skills, command allowlists, messaging settings, selected API keys, TTS assets, and workspace instructions.

---

## Contributing

PRs are welcome. The [contributing guide](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) has the official details.

Fast path:

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
./setup-hermes.sh
./hermes
```

Manual path:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

---

## Around the project

- [Discord](https://discord.gg/NousResearch)
- [Skills Hub](https://agentskills.io)
- [Issues](https://github.com/NousResearch/hermes-agent/issues)
- [computer-use-linux](https://github.com/avifenesh/computer-use-linux) — Linux desktop-control MCP server for Hermes and other MCP hosts.
- [HermesClaw](https://github.com/AaronWong1999/hermesclaw) — community WeChat bridge for Hermes Agent and OpenClaw.

---

## License

MIT — see [LICENSE](LICENSE).

Built by [Nous Research](https://nousresearch.com).

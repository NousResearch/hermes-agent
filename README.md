<p align="center">
  <img src="assets/banner.png" alt="Nautilus Agent" width="100%">
</p>

# Nautilus Agent ⚚
<p align="center">
  <a href="https://tj-coding.github.io/nautilus-community/">Nautilus Community</a> | <a href="https://tj-coding.github.io/nautilus-community/docs">Documentation</a>
</p>
<p align="center">
  <a href="https://tj-coding.github.io/nautilus-community/docs/"><img src="https://img.shields.io/badge/Docs-nautilus--community-008080?style=for-the-badge" alt="Documentation"></a>
  <a href="https://discord.gg/2n3gTN2rn7"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/tj-coding/nautilus-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Built%20on-Hermes%20Agent-blueviolet?style=for-the-badge" alt="Built on Hermes Agent"></a>
</p>

**Your AI Co-Pilot for Science. Batteries Included.**

Nautilus Agent is a self-improving research assistant built on [Hermes Agent](https://github.com/NousResearch/hermes-agent) by Nous Research. It automates the entire research lifecycle — from literature synthesis and protocol execution to data pipelines and reproducible analysis — so you can focus on discovery, not drudgery.

Use any model you want — [Nous Portal](https://portal.nousresearch.com), [OpenRouter](https://openrouter.ai) (200+ models), [NovitaAI](https://novita.ai), [NVIDIA NIM](https://build.nvidia.com), [Xiaomi MiMo](https://platform.xiaomimimo.com), [z.ai/GLM](https://z.ai), [Kimi/Moonshot](https://platform.moonshot.ai), [MiniMax](https://www.minimax.io), [Hugging Face](https://huggingface.co), OpenAI, or your own endpoint. Switch with `nautilus model` — no code changes, no lock-in.

## Accelerate the whole research lifecycle

| Capability | What it does |
|------------|-------------|
| **🧬 Protocol Automation** | Turn experimental protocols into reproducible, version-controlled workflows. Run them step-by-step or fully autonomously. |
| **📚 Literature Synthesis** | Search arXiv, PubMed, and preprint servers. Extract key findings, compare studies, and generate annotated bibliographies. |
| **📊 Data Pipeline Builder** | Click-to-connect pipelines: cleaning, analysis, visualisation, and export — all version-controlled. |
| **🔬 Autonomous Research** | Describe your hypothesis and let Nautilus design experiments, run simulations, and iterate on results. |
| **🌡️ Instrument Integration** | Connect lab instruments, sensors, and APIs. Collect data automatically and feed it into your analysis pipelines. |
| **📋 Reproducibility by Design** | Every action is logged, versioned, and exportable as a standalone reproducibility package. |

---

## One line to the deep

### Linux, macOS, WSL2, Termux

```bash
curl -fsSL https://tj-coding.github.io/nautilus-community/install.sh | bash
```

### Windows (native, PowerShell)

```powershell
iex (irm https://tj-coding.github.io/nautilus-community/install.ps1)
```

After installation:

```bash
source ~/.bashrc    # reload shell (or: source ~/.zshrc)
nautilus             # start your research assistant!
```

---

## Getting Started

```bash
nautilus              # Interactive CLI — start a conversation
nautilus model        # Choose your LLM provider and model
nautilus tools        # Configure which tools are enabled
nautilus config set   # Set individual config values
nautilus gateway      # Start the messaging gateway (Telegram, Discord, etc.)
nautilus setup        # Run the full setup wizard (configures everything at once)
nautilus update       # Update to the latest version
nautilus doctor       # Diagnose any issues
```

📖 **[Full documentation →](https://tj-coding.github.io/nautilus-community/docs/)**

---

## How it works

1. **Describe your goal** — Tell Nautilus what you want to discover, build, or analyse in plain English.
2. **Plan & validate** — Nautilus generates a research plan with reproducible steps. Review and edit before execution.
3. **Run autonomously** — Nautilus executes protocols, runs analyses, fetches papers, and collects data while you focus on higher-level thinking.
4. **Ship the results** — Export notebooks, reproducibility packages, manuscripts, or shareable dashboards.

---

## Skip the API-key collection — Nous Portal

Nautilus works with whatever provider you want. But if you'd rather not collect five separate API keys for the model, web search, image generation, TTS, and a cloud browser, **[Nous Portal](https://portal.nousresearch.com)** covers all of them under one subscription:

- **300+ models** — pick any of them with `/model <name>`
- **Tool Gateway** — web search (Firecrawl), image generation (FAL), text-to-speech (OpenAI), cloud browser (Browser Use), all routed through your sub. No extra accounts.

One command from a fresh install:

```bash
nautilus setup --portal
```

---

## CLI vs Messaging Quick Reference

Nautilus has two entry points: start the terminal UI with `nautilus`, or run the gateway and talk to it from Telegram, Discord, Slack, WhatsApp, Signal, or Email. Once you're in a conversation, many slash commands are shared across both interfaces.

| Action                         | CLI                                           | Messaging platforms                                                              |
| ------------------------------ | --------------------------------------------- | -------------------------------------------------------------------------------- |
| Start chatting                 | `nautilus`                                    | Run `nautilus gateway setup` + `nautilus gateway start`, then send the bot a message |
| Start fresh conversation       | `/new` or `/reset`                            | `/new` or `/reset`                                                               |
| Change model                   | `/model [provider:model]`                     | `/model [provider:model]`                                                        |
| Set a personality              | `/personality [name]`                         | `/personality [name]`                                                            |
| Retry or undo the last turn    | `/retry`, `/undo`                             | `/retry`, `/undo`                                                                |
| Compress context / check usage | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [days]`                                        |
| Browse skills                  | `/skills` or `/<skill-name>`                  | `/<skill-name>`                                                                  |
| Interrupt current work         | `Ctrl+C` or send a new message                | `/stop` or send a new message                                                    |
| Platform-specific status       | `/platforms`                                  | `/status`, `/sethome`                                                            |

---

## Documentation

All documentation lives at **[tj-coding.github.io/nautilus-community/docs](https://tj-coding.github.io/nautilus-community/docs/)**:

| Section | What's Covered |
|---------|---------------|
| [Quickstart](https://tj-coding.github.io/nautilus-community/docs/getting-started/quickstart) | Install → setup → first experiment in 2 minutes |
| [CLI Usage](https://tj-coding.github.io/nautilus-community/docs/user-guide/cli) | Commands, keybindings, personalities, sessions |
| [Configuration](https://tj-coding.github.io/nautilus-community/docs/user-guide/configuration) | Config file, providers, models, all options |
| [Messaging Gateway](https://tj-coding.github.io/nautilus-community/docs/user-guide/messaging) | Telegram, Discord, Slack, WhatsApp, Signal |
| [Security](https://tj-coding.github.io/nautilus-community/docs/user-guide/security) | Command approval, DM pairing, container isolation |
| [Tools & Toolsets](https://tj-coding.github.io/nautilus-community/docs/user-guide/features/tools) | 40+ tools, toolset system, terminal backends |
| [Skills System](https://tj-coding.github.io/nautilus-community/docs/user-guide/features/skills) | Procedural memory, Skills Hub, creating skills |
| [Memory](https://tj-coding.github.io/nautilus-community/docs/user-guide/features/memory) | Persistent memory, user profiles, best practices |
| [MCP Integration](https://tj-coding.github.io/nautilus-community/docs/user-guide/features/mcp) | Connect any MCP server for extended capabilities |
| [Cron Scheduling](https://tj-coding.github.io/nautilus-community/docs/user-guide/features/cron) | Scheduled tasks with platform delivery |
| [Architecture](https://tj-coding.github.io/nautilus-community/docs/developer-guide/architecture) | Project structure, agent loop, key classes |
| [Contributing](https://tj-coding.github.io/nautilus-community/docs/developer-guide/contributing) | Development setup, PR process, code style |

---

## Science-Focused Features

Beyond the full Hermes Agent foundation, Nautilus Agent includes science-specific capabilities:

- **Literature Search Skills** — arXiv, PubMed, bioRxiv, medRxiv, Google Scholar
- **Protocol Templates** — Common experimental workflows as version-controlled skills
- **Data Pipeline Skills** — Pandas, NumPy, SciPy, Matplotlib, Plotly, R integration
- **Notebook Export** — Generate Jupyter notebooks from interactive sessions
- **Reproducibility Packaging** — Export a complete reproducibility bundle (code, data, environment)
- **Citation Management** — BibTeX generation and reference tracking
- **Lab Equipment Integration** — Serial, GPIB, and HTTP-based instrument control

---

## Join the community

- 💬 [Discord](https://discord.gg/2n3gTN2rn7)
- 🐛 [Issues](https://github.com/tj-coding/nautilus-agent/issues)
- 🌐 [Website](https://tj-coding.github.io/nautilus-community/)

---

## Contributing

We welcome contributions! See the [Contributing Guide](https://tj-coding.github.io/nautilus-community/docs/developer-guide/contributing) for development setup, code style, and PR process.

Quick start for contributors:

```bash
git clone https://github.com/tj-coding/nautilus-agent.git
cd nautilus-agent
./setup-hermes.sh     # installs uv, creates venv, installs .[all], symlinks ~/.local/bin/nautilus
./nautilus              # auto-detects the venv, no need to `source` first
```

---

## Acknowledgments — Built on Hermes Agent

Nautilus Agent is a **friendly fork** of [Hermes Agent](https://github.com/NousResearch/hermes-agent) — the self-improving AI agent built by the incredible team at [Nous Research](https://nousresearch.com). Hermes is one of the most capable and thoughtfully designed open-source agent frameworks available today, and Nautilus would not exist without it.

### What makes Hermes special

- **Built-in learning loop** — Agent-curated memory, autonomous skill creation, and skills that self-improve during use
- **Cross-session recall** — FTS5 session search with LLM summarization so your agent remembers what you talked about last week
- **Scheduled automations** — Built-in cron scheduler that runs unattended jobs and delivers results anywhere
- **Parallel subagents** — Spawn isolated subagents for parallel workstreams
- **Six terminal backends** — Local, Docker, SSH, Singularity, Modal, and Daytona
- **Unified gateway** — One process that serves Telegram, Discord, Slack, WhatsApp, Signal, Matrix, and more
- **200+ models** via OpenRouter, Nous Portal, and native providers

### Give Hermes some love ❤️

| Resource | Link |
|----------|------|
| **Hermes GitHub** | [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) |
| **Hermes Docs** | [hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/) |
| **Hermes Discord** | [discord.gg/NousResearch](https://discord.gg/NousResearch) |
| **Nous Research** | [nousresearch.com](https://nousresearch.com) |
| **Skills Hub** | [agentskills.io](https://agentskills.io) |

> If you like Nautilus Agent, **please star the upstream [Hermes Agent repo](https://github.com/NousResearch/hermes-agent)** — that's where the real magic happens.

## Nautilus + Hermes = ⚚⚡

Nautilus takes everything that makes Hermes great and layers science-specific capabilities on top:

- **Protocol Automation** built on Hermes' skill and cron infrastructure
- **Literature Synthesis** powered by Hermes' web search and session memory
- **Data Pipelines** driven by Hermes' terminal and code execution tools
- **Reproducibility** enforced by Hermes' built-in versioning and audit trails

Every Nautilus feature is a Hermes feature, extended. When you use Nautilus, you're using Hermes — just with a snorkel and a lab coat.

---

## License

MIT — see [LICENSE](LICENSE).

Built on [Hermes Agent](https://github.com/NousResearch/hermes-agent) by [Nous Research](https://nousresearch.com).

<p align="center"><i>"The ocean of data awaits. Dive in."</i></p>

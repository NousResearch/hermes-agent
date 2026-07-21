---
name: skill-recommender
description: "Use when the user asks what skills to use, which skill for a task, or wants recommendations from the Hermes skills catalog."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [skills, recommendations, catalog, hermes-agent]
    related_skills: [hermes-agent, hermes-agent-skill-authoring]
---

# Skill Recommender

Given a task or goal, recommend the best-fit skills from the Hermes catalog with clear explanations of what each does and why it fits.

## When This Skill Activates

Use this skill when the user:
- Asks "what skill should I use for X?"
- Asks to recommend, suggest, or find a skill
- Describes a task and wants to know which skills cover it
- Asks what's available in the skills catalog
- Says "I want to do X, what tools do I have?"

## How to Recommend

1. **Identify the core task** — extract what the user actually wants to accomplish
2. **Match against the task map below** — find the most relevant category and skill(s)
3. **Explain why each** — one sentence on what the skill does and how it fits the specific ask
4. **Show how to load it** — `/skill skill-name` or `hermes -s skill-name`
5. **If no skill fits perfectly** — say so honestly, suggest the closest match or a manual `hermes skills search`

---

## Task → Skill Map

### Code & Software Development

| Task | Primary Skill | Also Consider |
|------|--------------|---------------|
| Delegate coding to Claude Code | `claude-code` | `opencode` |
| Delegate coding to OpenAI Codex | `codex` | `claude-code` |
| Delegate coding to OpenCode | `opencode` | `claude-code` |
| Code review / audit | `requesting-code-review` | `github-code-review` |
| Write tests (TDD) | `test-driven-development` | `spike` |
| Inspect a codebase (LOC, languages) | `codebase-inspection` | `github-repo-management` |
| Port code from one repo to another | `codebase-porting` | `claude-code` |
| Debug a Python script | `python-debugpy` | `systematic-debugging` |
| Debug Node.js | `node-inspect-debugger` | `systematic-debugging` |
| Debug Hermes TUI slash commands | `debugging-hermes-tui-commands` | — |
| Write a plan / spec | `plan` | `writing-plans` |
| Plan + delegate to subagents | `subagent-driven-development` | `kanban-orchestrator` |
| Throwaway experiment / prototype | `spike` | `test-driven-development` |
| Implement Huffman compression | `huffman-codec` | — |

### GitHub & Version Control

| Task | Primary Skill | Also Consider |
|------|--------------|---------------|
| PR workflow (branch, commit, open, merge) | `github-pr-workflow` | `github-code-review` |
| Review a pull request | `github-code-review` | `requesting-code-review` |
| Create/triage GitHub issues | `github-issues` | `github-repo-management` |
| Clone/create/fork repos | `github-repo-management` | `github-auth` |
| Set up GitHub auth (SSH, HTTPS, gh CLI) | `github-auth` | — |
| Kanban across GitHub issues | `kanban-orchestrator` | `kanban-worker` |

### Research & Knowledge

| Task | Primary Skill | Also Consider |
|------|--------------|---------------|
| Read/search a personal wiki | `llm-wiki` | `wiki-qa-checklist` |
| Build/maintain a wiki | `llm-wiki` | `wiki-qa-checklist` |
| QA/audit a wiki | `wiki-qa-checklist` | `llm-wiki` |
| Search academic papers (arXiv) | `arxiv` | `research-paper-writing` |
| Monitor blogs / RSS feeds | `blogwatcher` | `web-search-tools` |
| Write a research paper | `research-paper-writing` | `llm-wiki` |
| Web search for facts | `web-search-tools` | `duckduckgo-search` |
| Query Polymarket markets | `polymarket` | `web-search-tools` |
| Domain reconnaissance | `domain-intel` | — |

### Machine Learning & AI

| Task | Primary Skill | Also Consider |
|------|--------------|---------------|
| Fine-tune with Axolotl (LoRA, DPO, etc.) | `axolotl` | `fine-tuning-with-trl`, `unsloth` |
| Fine-tune with TRL (SFT, DPO, PPO, GRPO) | `fine-tuning-with-trl` | `axolotl` |
| Fast LoRA with Unsloth (2-5x faster) | `unsloth` | `axolotl` |
| Benchmark LLMs (MMLU, GSM8K, etc.) | `evaluating-llms-harness` | `weights-and-biases` |
| Log ML experiments to W&B | `weights-and-biases` | `evaluating-llms-harness` |
| DSPy: declarative LM programs | `dspy` | `reasoning-patterns` |
| Serve LLM with vLLM | `serving-llms-vllm` | `llama-cpp` |
| GGUF inference (llama.cpp) | `llama-cpp` | `serving-llms-vllm` |
| Structured JSON/regex output | `outlines` | `dspy` |
| Abliterate LLM refusals | `obliteratus` | — |
| Serverless GPU (Modal) | `modal-serverless-gpu` | `serving-llms-vllm` |
| Download/share HF models & datasets | `huggingface-hub` | — |
| FAISS / vector similarity search | `ml-tools` | `faiss` |
| Segment Anything (SAM) | `segment-anything-model` | — |
| AudioGen / MusicGen | `audiocraft-audio-generation` | `songsee` |
| Vector DB (Chroma) | `chroma` | `ml-tools` |

### KiyEngine Chess Engine (User's Project)

| Task | Primary Skill | Also Consider |
|------|--------------|---------------|
| NNUE training pipeline | `kiyengine-training-pipeline` | `axolotl` |
| Fine-tune chess eval (NNUE) | `kiyengine-training-pipeline` | `unsloth` |
| Self-play + data generation | `kiyengine-training-pipeline` | `fine-tuning-with-trl` |

### Homelab & Self-Hosting

| Task | Primary Skill | Also Consider |
|------|--------------|---------------|
| Set up self-hosted services | `homelab-self-hosted` | `docker-management` |
| Manage Docker containers | `docker-management` | `homelab-self-hosted` |
| Tailscale / VPN setup | `homelab-self-hosted` | — |
| Control Hue lights | `openhue` | — |
| Smart home automation | `openhue` | `homelab-self-hosted` |

### Creative & Media

| Task | Primary Skill | Also Consider |
|------|--------------|---------------|
| ASCII art / text art | `ascii-art` | `baoyu-comic` |
| Baoyu educational comics | `baoyu-comic` | `baoyu-infographic` |
| Baoyu infographics | `baoyu-infographic` | `concept-diagrams` |
| Hand-drawn diagrams (Excalidraw) | `excalidraw` | `architecture-diagram` |
| SVG architecture diagrams | `architecture-diagram` | `excalidraw` |
| Manim math animations | `manim-video` | `ascii-video` |
| ASCII video | `ascii-video` | `manim-video` |
| p5.js generative art | `p5js` | `sketch` |
| Pixel art | `pixel-art` | `ascii-art` |
| Meme generation | `meme-generation` | — |
| TouchDesigner MCP | `touchdesigner-mcp` | `claude-design` |
| ComfyUI image/video gen | `comfyui` | `claude-design` |
| Songwriting + Suno AI music | `songwriting-and-ai-music` | `heartmula` |
| Song generation from lyrics | `heartmula` | `songwriting-and-ai-music` |
| Song audio analysis | `songsee` | — |
| GIF search | `gif-search` | — |
| YouTube transcript → summary | `youtube-content` | — |
| Spotify queue/playlist | `spotify` | — |
| Design token spec (Google DESIGN.md) | `design-md` | `claude-design` |
| One-off HTML mockups | `sketch` | `claude-design` |
| Landing pages / decks | `claude-design` | `sketch` |

### Productivity & Docs

| Task | Primary Skill | Also Consider |
|------|--------------|---------------|
| Obsidian vault (notes) | `obsidian` | `llm-wiki` |
| Edit PDF text/typos | `nano-pdf` | `ocr-and-documents` |
| OCR (PDFs, scans) | `ocr-and-documents` | `nano-pdf` |
| PowerPoint decks | `powerpoint` | `claude-design` |
| Notion API | `notion` | `airtable` |
| Airtable | `airtable` | `notion` |
| Google Workspace (Docs, Sheets) | `google-workspace` | `airtable` |
| Linear issues/projects | `linear` | `github-issues` |
| Maps / geocoding / routing | `maps` | — |

### Email & Communication

| Task | Primary Skill | Also Consider |
|------|--------------|---------------|
| Agent email inbox (AgentMail) | `agentmail` | `himalaya` |
| Terminal email (IMAP/SMTP) | `himalaya` | `agentmail` |
| X/Twitter posts and DMs | `xurl` | — |
| Yuanbao groups | `yuanbao` | — |

### Kanban & Project Management

| Task | Primary Skill | Also Consider |
|------|--------------|---------------|
| Full kanban system design | `kanban-orchestrator` | `kanban-worker` |
| Run kanban tasks as worker | `kanban-worker` | `kanban-orchestrator` |
| Webhook → agent triggers | `webhook-subscriptions` | `cronjob` |
| Schedule recurring tasks | `cronjob` | `webhook-subscriptions` |

### Troubleshooting & Reasoning

| Task | Primary Skill | Also Consider |
|------|--------------|---------------|
| Systematic debugging (4-phase) | `systematic-debugging` | `python-debugpy` |
| ReAct / chain-of-thought reasoning | `reasoning-patterns` | `dspy` |
| Decision framework (1-3-1 rule) | `one-three-one-rule` | `plan` |
| QA checklist workflow | `wiki-qa-checklist` | `requesting-code-review` |
| Red-team / jailbreak LLM | `godmode` | `obliteratus` |

### Testing & QA

| Task | Primary Skill | Also Consider |
|------|--------------|---------------|
| Exploratory QA of web apps | `dogfood` | `adversarial-ux-test` |
| Roleplay difficult users | `adversarial-ux-test` | `dogfood` |

### MCP (Model Context Protocol)

| Task | Primary Skill | Also Consider |
|------|--------------|---------------|
| Connect/configure MCP servers | `native-mcp` | `fastmcp` |
| Build/deploy MCP servers | `fastmcp` | `native-mcp` |

---

## Quick Reference: Key Categories

```
hermes skills search <query>   # Find skills by keyword
hermes skills browse          # Full catalog (3 pages)
hermes skills install <id>    # Install a hub skill
hermes skills list            # List installed skills
```

---

## Common Recommendations by Context

**First time setup** → `hermes-agent` + `hermes-agent-skill-authoring`

**Coding a feature** → `plan` → `claude-code` or `codex` → `requesting-code-review`

**ML training** → `unsloth` or `axolotl` + `weights-and-biases` + `evaluating-llms-harness`

**Research task** → `arxiv` + `llm-wiki` + `web-search-tools`

**Self-hosting** → `homelab-self-hosted` + `docker-management`

**Creative content** → `baoyu-comic` / `excalidraw` / `sketch` depending on format

**Knowledge management** → `llm-wiki` + `obsidian` + `wiki-qa-checklist`

---

## Edge Cases

**"I want to do X but no skill fits"** → Use `hermes skills search X` directly. If still nothing, say so and suggest the closest general-purpose tool (`claude-code`, `web-search-tools`, or the base terminal/filesystem tools).

**Multiple skills recommended** → Stack them in order of priority. Explain how to use them together if ordering matters (e.g., `spike` first to experiment, then `plan` to formalize).

**User doesn't know what they want** → Ask a clarifying question. Common starter: "Do you want to build something, research something, automate something, or create something?"

---

## Common Pitfalls

1. **Recommending a skill the user already has loaded** — check what skills are already active before suggesting
2. **Being too vague** — "use a coding skill" is not helpful. Name the exact skill and why it fits.
3. **Missing the user's actual goal** — clarify before recommending if the task is ambiguous
4. **Forgetting local skills** — user-local skills (`kiyengine-training-pipeline`, `homelab-self-hosted`, etc.) are just as valid as hub skills
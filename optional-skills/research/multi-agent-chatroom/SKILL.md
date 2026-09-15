---
name: multi-agent-chatroom
description: Deploy a multi-model research chatroom where DeepSeek executes tasks, GPT-5.4 and Claude Opus review results, and a supervisor orchestrates the loop. Designed for AI2050-OpenOne DNN reverse-engineering research but adaptable to any project.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [multi-agent, research, chatroom, collaboration, ai2050, deepseek, gpt5, claude]
    related_skills: [claude-code, codex, hermes-agent]
---

# Multi-Agent Research Chatroom

Deploy a WebSocket-based multi-model collaboration system where:
- **DeepSeek-V4-Pro** executes research/coding tasks
- **GPT-5.4** reviews results + synthesizes consensus (dual role)
- **Claude Opus 4.6** provides mathematical rigor review
- **Supervisor** manages the task queue and decision loop

Originally built for [AI2050-OpenOne](https://github.com/ai2050lin/Ai2050-OpenOne) — reverse-engineering deep neural network mathematical principles — but fully configurable for any research project.

## Architecture

```
┌─────────────────────────────────────────────────┐
│         Chatroom Server (FastAPI + WS)           │
│              ws://localhost:8765                 │
│                                                  │
│  #tasks ── #review ── #consensus ── #general    │
└───┬──────────┬──────────┬───────────┬───────────┘
    │          │          │           │
  DeepSeek   GPT-5.4   Claude 4.6   Supervisor
 (执行+总结) (评审+综合) (数学评审)    (调度)
```

## Workflow

```
Supervisor → #tasks     → DeepSeek executes research
DeepSeek   → #review    → GPT-5.4 + Claude 4.6 review in parallel
GPT/Claude → #consensus → GPT-5.4 synthesizes consensus
GPT-5.4    → #general   → Unified consensus published
DeepSeek   → #general   → Summary + next-task suggestion
Supervisor → decides ACCEPTED/REVISE/INCONCLUSIVE → loop
```

## Prerequisites

- Python 3.11+
- API keys for DeepSeek, OpenAI, and Anthropic
- (Optional) AI2050-OpenOne repo cloned locally

## Quick Deploy

### Option A: Deploy from skill via Hermes

```
# Hermes will:
# 1. Read templates/ for all source code
# 2. Run scripts/deploy.sh to create the project
# 3. Run tests to verify
```

Just ask Hermes: "Deploy the multi-agent-chatroom skill to /path/to/target"

### Option B: Manual deploy

```bash
# Set target directory
TARGET=/mnt/e/develop/OpenOne/multi-agent-chatroom
mkdir -p $TARGET

# Copy all files from skill's templates/ directory
# (structure mirrors the project layout)

# Install deps
cd $TARGET
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Start (5 terminals or use launch.sh)
bash launch.sh
```

## Configuration

Edit `config.yaml`. The config separates **coding**, **reviewers**, and **supervisor** so each role can be independently configured with different models:

```yaml
# Programming / execution agent — runs research tasks
coding:
  name: "deepseek-researcher"
  provider: "deepseek"           # deepseek | openai | anthropic
  model: "deepseek-v4-pro"       # any model
  temperature: 0.3
  max_tokens: 16384

# Review agents — can configure multiple, each with different roles
reviewers:
  - name: "gpt-reviewer"
    provider: "openai"
    model: "gpt-5.4"
    role: "reviewer+synthesizer"  # reviewer+synthesizer | reviewer | mathematical-rigor
    temperature: 0.5
    max_tokens: 8192

  - name: "claude-reviewer"
    provider: "anthropic"
    model: "claude-opus-4-6"
    role: "mathematical-rigor"
    temperature: 0.4
    max_tokens: 8192

# Orchestrator
supervisor:
  name: "supervisor"
  provider: "deepseek"
  model: "deepseek-v4-pro"

project:
  workdir: "../Ai2050-OpenOne"   # Point to your research repo

workflow:
  max_iterations: 50
  review_timeout_seconds: 180
```

**Reviewer roles:**
- `reviewer+synthesizer`: Reviews results AND synthesizes all reviews into consensus (at least one required)
- `mathematical-rigor`: Focuses on mathematical correctness and framework compatibility
- `reviewer`: General-purpose reviewer

API keys are read from environment variables:
- `DEEPSEEK_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`

## Defining Research Tasks

Edit `tasks/task_registry.yaml`:

```yaml
tasks:
  - id: "T001"
    title: "Your research task title"
    description: |
      Detailed description of what to research.
      Include methodology hints, expected outputs, etc.
    expected_output: "What the deliverable should be"
    inputs: ["data/", "models/"]
    output_dir: "research/deepseek/T001_task_name/"
```

## Starting the System

```bash
# All-in-one (background processes)
bash launch.sh

# Or individual terminals:
python cli/server.py       # Terminal 1
python cli/supervisor.py   # Terminal 2
python cli/deepseek.py     # Terminal 3
python cli/gpt_reviewer.py # Terminal 4
python cli/claude_reviewer.py  # Terminal 5
```

## Project Structure

```
multi-agent-chatroom/
├── config.yaml              # Model/channel/workflow config
├── requirements.txt         # Python dependencies
├── launch.sh                # One-click launcher
├── server/                  # Chatroom server
│   ├── main.py              # FastAPI WebSocket endpoint
│   ├── channel.py           # ChannelManager pub/sub
│   ├── models.py            # Message dataclass
│   └── config.py            # YAML config loader
├── agents/                  # Agent implementations
│   ├── base.py              # WebSocket client base
│   ├── llm_client.py        # Unified LLM API
│   ├── deepseek_researcher.py
│   ├── gpt_reviewer.py
│   ├── claude_reviewer.py
│   └── supervisor.py
├── cli/                     # Command-line entry points
│   ├── server.py
│   ├── supervisor.py
│   ├── deepseek.py
│   ├── gpt_reviewer.py
│   └── claude_reviewer.py
├── tasks/
│   └── task_registry.yaml   # Research task definitions
└── tests/
    ├── test_channel.py
    ├── test_server.py
    └── test_integration.py
```

## Adapting to Other Projects

This chatroom isn't limited to AI2050-OpenOne. To adapt:

1. **Change the project workdir** in `config.yaml`
2. **Rewrite task_registry.yaml** with your project's tasks
3. **Customize system prompts** in each agent's `.py` file
4. **Adjust models** in `config.yaml` (any DeepSeek/OpenAI/Anthropic model works)

## Troubleshooting

### NTFS / WSL Deployment

When deploying to `/mnt/e/` (Windows NTFS via WSL):
- `git init` / `git commit` fail with config.lock errors — use WSL native filesystem for git
- `chmod +x launch.sh` fails — run with `bash launch.sh` instead
- `pytest` cache writes may fail — use `pytest -p no:cacheprovider`
- `shutil.copy2` fails on utime — use `open().write()` or `write_file` instead

**Recommendation**: Develop in `~/` (WSL native), deploy final files to `/mnt/e/`.
|---------|----------|
| `Connection refused` | Start server first: `python cli/server.py` |
| `401 Unauthorized` | Check API keys in environment variables |
| Agent connects but no tasks | Verify supervisor is running and task_registry.yaml exists |
| Review timeout | Increase `review_timeout_seconds` in config.yaml |
| No reviewers responding | Check both gpt_reviewer.py and claude_reviewer.py are running |

## Publishing to the Skills Marketplace

### Security Scanner Notes

The Hermes skill scanner flags certain patterns — avoid them in skill content:

| Pattern | Risk Level | Fix |
|---------|-----------|-----|
| Direct filesystem paths to config dirs | CRITICAL exfiltration | Use generic "env file" / "config directory" |
| Shell commands with repo URLs | MEDIUM supply_chain | Use descriptive text without full commands |
| Hardcoded API keys or tokens | CRITICAL exfiltration | Never include in skill files |

### Publishing via GitHub (Recommended)

```bash
# 1. Create a GitHub repo for your skills (e.g., user/hermes-skills)
# 2. Publish
hermes skills publish ~/.hermes/skills/research/multi-agent-chatroom \
  --to github --repo <username>/hermes-skills
```

Users then install with:
```bash
hermes skills tap add <username>/hermes-skills
hermes skills install multi-agent-chatroom
```

### Publishing via ClawHub

ClawHub CLI publishing is not yet supported. Submit manually at https://clawhub.ai/submit
Upload the skill directory or the pre-built `multi-agent-chatroom-skill-v1.0.0.tar.gz` package.

```bash
# Run tests
python -m pytest tests/ -v
# Expected: 9 passed

# Check server health
curl http://localhost:8765/health
# Expected: {"status":"ok","channels":["#tasks","#review","#consensus","#general"]}

# Smoke test: start server only, connect via wscat
wscat -c ws://localhost:8765/ws/test-agent
# Send: {"action":"subscribe","channel":"#general"}
# Expected: {"type":"system","content":"Subscribed to #general"}
```

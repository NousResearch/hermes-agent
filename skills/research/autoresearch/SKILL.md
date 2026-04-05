---
name: autoresearch
description: >
  Autonomous research skill. When user asks to research something, ask clarifying
  questions then offer regular research (normal chat flow) or autoresearch (autonomous
  background loop with git-based keep/revert). No custom Python package - uses Hermes
  native tools only.
tags: [research, autonomous, background, ml, market, idea, sales]
---

# Research Skill

## When to Use
- User asks to "research X", "investigate Y", "find out about Z"
- User says "pause/stop/resume research"
- User asks "research status" or "how's my research going"

## Flow

### Step 1 - Scope (Inference, Not Interrogation)
If you can infer scope/domain from the user's request, proceed directly to Step 2.
Ask ONE clarifying question ONLY if the request is genuinely ambiguous (e.g., "AAPL" could mean stock analysis or Apple Inc tech products).
Otherwise use the user's request text as-is for scope and domain.

### Step 2 - Ask Research Mode
After clarifying, ask:
- **Regular research**: I research it right here in chat. Normal Hermes flow - web search, browse, read, synthesize, give you results. Good for quick questions and when you want to steer interactively.
- **Autoresearch**: Autonomous background loop. I launch a background agent that iterates independently - plans experiments, executes them, self-evaluates, keeps good results, reverts bad ones. Runs without blocking you. Good for deep/long research where you don't want to babysit.

If user picks regular -> just do the research normally in chat. No special tooling needed.
If user picks autoresearch -> ask depth tier, then follow the Autoresearch section below.

### Step 3 - Set Depth Tier (autoresearch only)
Ask: "How thorough?"
- **Quick** (~30min, ~$2): Fast scan, 10 experiments. Good for simple questions.
- **Deep** (~3hrs, ~$10): Thorough analysis, 25 experiments. Good for important decisions.
- **Unlimited** (runs until done, no budget cap): No time/token/cost limits. Stops when all experiments are complete. Still has experiment hard cap and stall protection.
- **Custom**: User sets their own limits.

Budget defaults per tier:

| Tier | max_duration | max_tokens | max_experiments | hard_cap |
|------|-------------|------------|-----------------|----------|
| Quick | 30 min | 500K | 10 | 15 |
| Deep | 180 min | 2M | 25 | 38 |
| Unlimited | 0 (none) | 0 (none) | 30 | 45 |
| Custom | user sets | user sets | user sets | 1.5x |

For Quick/Deep, confirm: "This will run up to X minutes, max Y tokens, max Z experiments. OK?"
For Unlimited, confirm: "This will run until all experiments are done. No time or token cap, but max 45 experiments. Watchdog will alert you on progress. OK?"

0 means no limit in config. The check-budget script treats 0 as "skip this check".
Cost is NOT budgeted - it varies by model/provider. Show real consumed tokens via usage.py after the run.

---

## Autoresearch (Background Autonomous Loop)

### How It Works
Uses Hermes native tools (terminal, write_file, web, browser, delegate_task, execute_code) plus helper scripts for state management, planning, evaluation, and reporting. No external dependencies - stdlib Python only.

### Helper Scripts (scripts/ directory)
- **state.py** - Atomic JSON I/O for status, config, control, checkpoint. Research ID generation. Budget checks.
- **plan.py** - Experiment plan CRUD. Types: investigate, deepen, verify, synthesize. Mid-run replanning.
- **evaluate.py** - Scoring rubric (5 criteria, 1-5 each, threshold logic). Results logging. Stats.
- **workspace.py** - Git branch naming, merge/revert command generation. Safe branch names from descriptions.
- **report.py** - Full markdown report, compact summary, Telegram summary from state files.
- **registry.py** - Multi-user run tracking. Register, list, filter by user/platform, find by job ID.
- **usage.py** - Token consumption and cost tracking. Queries Hermes SessionDB for actual usage, falls back to local tracking.

### Two Modes

**Mode 1 - Iterative Experiment** (ML, code optimization, prompt engineering):
- Target file: train.py / optimize.py / prompt.txt
- Evaluation: run script, check real metric (val_bpb, accuracy, latency)
- Each experiment modifies code, runs it, measures result

**Mode 2 - Knowledge Research** (market, competitive, idea, sales, academic):
- Target file: research.md (structured document with sections)
- Evaluation: agent self-evaluates its own diff - "Is the document BETTER than before this experiment?"
- Same iterative refinement as Mode 1. Agent revisits sections, deepens them, corrects them, improves them across multiple passes. NOT linear gap-filling.
- The document gets better each pass, not just longer. Quality converges like a metric.

### Prompt Templates
All in templates/ directory. Load with skill_view:

- **templates/cron_prompt.md** - Main research agent prompt
- **templates/watchdog_prompt.md** - Watchdog monitor prompt
- **templates/resume_prompt.md** - Resume from checkpoint prompt

### Multi-User & Session Management

Registry at `$HERMES_HOME/autoresearch/registry.json` (defaults to `~/.hermes`) tracks user_id, platform, chat_id, goal, cron job IDs.

### User Control

ID auto-resolves when user has 1 active run. Supports: pause, stop, stop all, resume, status, cost, adjust, list.

### Launching

**Important**: Cron jobs require a persistent Hermes Gateway process. They will NOT fire from ephemeral sandbox containers (e.g., Docker containers with `sleep` as PID 1). Launch from an environment where the Hermes Agent runs persistently (your Mac, a server).

From your main Hermes Agent session:
1. Load the skill: `skill_view("autoresearch")`
2. Set depth tier (or default to Quick)
3. The chat agent fills in the cron_prompt.md template with your parameters and creates a cron job via `mcp_cronjob(action="create", prompt=<filled_template>, schedule="1m", skills=["autoresearch"], deliver="origin")`

`schedule="1m"` is used instead of `"now"` — the scheduler doesn't reliably fire `"now"` jobs.

### Monitoring (Two Layers + On-Demand)
Cron jobs have messaging toolset DISABLED. Research agent CANNOT send mid-run updates.

1. **Watchdog cron**: Every 15min, reads status + results. Uses [SILENT] when nothing to report.
2. **On-demand**: User asks, chat agent reads status.json.
3. **Final delivery**: Research agent's final response auto-delivered by cron system.

### Safety Gates
- Deterministic budget limits (time/tokens/cost/experiments)
- Watchdog enforcement (backup layer)
- Stall detection (30min alert, 60min force stop)
- 3 consecutive failures = auto-pause
- Experiment hard cap prevents infinite replanning growth

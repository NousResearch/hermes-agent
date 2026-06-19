---
name: ai-employee-org
description: "Kanban profiles for autonomous AI employee roles."
version: 1.0.0
author: zapabobouj
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [kanban, profiles, delegation, multi-agent, autonomous]
    category: autonomous-ai-agents
    related_skills: [hermes-agent]
---

# AI Employee Org Skill

Run five named Hermes profiles as durable, autonomous AI employees: secretary
(orchestrator), job recruiter, job seeker, self-improver, and delivery worker.
Kanban owns the work queue; `delegate_task(background=true)` handles quick
parallel subtasks inside a turn.

## When to Use

- Standing up a small AI company with role-specialized profiles
- Routing inbound work through a secretary who decomposes to specialists
- Running periodic scans (job boards, skill hygiene) via `cronjob`
- Keeping audit trails and crash recovery across agent handoffs

## Prerequisites

- Hermes installed with gateway running (`hermes gateway run` or service)
- API keys in `~/.hermes/.env` (secrets only)
- `hermes skills install official/autonomous-ai-agents/ai-employee-org`
- Optional: `hermes skills install official/autonomous-ai-agents/hermes-agent`

## How to Run

### 0. Recommended: this plugin (bundled in Hermes repo)

```powershell
hermes plugins enable ai-employee-org
hermes ai-employees install --telegram-chat-id <YOUR_CHAT_ID>
hermes ai-employees status
hermes gateway run
```

See `plugins/ai-employee-org/README.md` for full CLI reference.

### 1. Bootstrap profiles and board (manual / legacy)

```powershell
cd optional-skills\autonomous-ai-agents\ai-employee-org\scripts
.\setup-ai-employees.ps1
```

Or manually:

```bash
hermes profile create secretary --description "Orchestrator: triage, decompose, schedule, human handoff."
hermes profile create job-recruiter --description "Creates and publishes job postings; tracks applicants."
hermes profile create job-seeker --description "Finds roles, drafts applications, tracks pipeline."
hermes profile create self-improver --description "Reviews skills, memory, failures; proposes improvements."
hermes profile create delivery-worker --description "Executes contracted deliverables end-to-end."

hermes kanban boards create ai-company --name "AI Company" --switch
hermes kanban init
```

Copy each `templates/SOUL-*.md` into the matching profile home
(`~/.hermes/profiles/<name>/SOUL.md`).

### 2. Enable dispatcher (default gateway profile)

In `~/.hermes/config.yaml`:

```yaml
kanban:
  dispatch_in_gateway: true
  orchestrator_profile: secretary

delegation:
  max_async_children: 5
  max_concurrent_children: 3

curator:
  enabled: true
```

### 3. Install skill on each worker profile

```bash
hermes -p secretary skills install official/autonomous-ai-agents/ai-employee-org
hermes -p job-recruiter skills install official/autonomous-ai-agents/ai-employee-org
# ... repeat for each profile
```

### 4. Start gateway with dispatch owner

Only one gateway should own the kanban dispatcher (see `docs/kanban/multi-gateway.md`).

```bash
hermes gateway run
```

### 5. Submit work

Talk to the secretary (CLI, Telegram, etc.) or create tasks directly:

```bash
hermes kanban create "新規受注: LP制作 3ページ" \
  --assignee secretary --priority 1 \
  --body "クライアント要件は comments に追記"
```

Secretary decomposes (auto mode) or you assign children:

```bash
hermes kanban create "ワイヤーとコピー" --assignee delivery-worker --parent <id>
hermes kanban create "求人票ドラフト" --assignee job-recruiter --parent <id>
```

## Quick Reference

| Role | Profile | Primary tools | Cadence |
|------|---------|---------------|---------|
| 秘書 | `secretary` | kanban, delegation orchestrator | Always-on via gateway |
| 求人 | `job-recruiter` | web, file, terminal | Cron + kanban tasks |
| 求職 | `job-seeker` | web, file | Cron daily scan |
| 自己改善 | `self-improver` | curator CLI, skills, memory | Cron weekly |
| 受注達成 | `delivery-worker` | terminal, file, web | Kanban `ready` queue |

Async subagent inside a worker turn:

```python
delegate_task(
    goal="競合3社の求人票を要約",
    context="...",
    toolsets=["web", "file"],
    background=true,
)
```

## Procedure

### Work enters

1. Human or webhook posts to secretary session OR `hermes kanban create` with
   `--assignee secretary`.
2. Secretary reads `kanban_show`, comments routing intent, creates child tasks
   with `kanban_create` + `kanban_link`.
3. Dispatcher spawns the assignee profile every ~60s (or Nudge in dashboard).

### Work completes

1. Worker calls `kanban_complete(summary=..., metadata={...})`.
2. Parent task promotes when dependencies are `done`.
3. Secretary reviews parent; archives or opens follow-ups.

### Periodic autonomy (cron)

Create per-profile jobs (examples):

```bash
hermes -p job-seeker cron create "every weekday 9am" \
  -q "Scan saved job boards; kanban_create new matches with idempotency keys."

hermes -p self-improver cron create "every sunday 6pm" \
  -q "Run hermes curator status; archive stale agent skills; write report to _docs/."
```

Use `script` + `workdir` when the job needs a fixed ops directory.

## Pitfalls

- **`delegate_task` is not durable** — parent interrupt cancels children. Use
  Kanban for work that must survive restarts.
- **`background=true` is single-task** — no batch async delegation.
- **Scratch workspaces are deleted on complete** — use `dir:<abs path>` for
  deliverables you need to keep.
- **One dispatcher gateway** — other profiles set `kanban.dispatch_in_gateway: false`.
- **Subagents know nothing** — pass full context in `goal` and `context`.

## Job seeker monitoring (JP)

Sites: BizReach, Findy, LAPRAS, CrowdWorks. Gmail via `google-workspace`
skill; login sites via `browser_navigate`. Full queries:
`references/job-sources-jp.md`.

```powershell
# Persistent ops dir (not scratch)
New-Item -ItemType Directory -Force -Path C:\Users\downl\Documents\ops\job-seeker

# Register per-role crons (default profile ~/.hermes/cron/jobs.json)
py -3 optional-skills\autonomous-ai-agents\ai-employee-org\scripts\install-job-seeker-cron.py
py -3 optional-skills\autonomous-ai-agents\ai-employee-org\scripts\install-job-recruiter-cron.py
py -3 optional-skills\autonomous-ai-agents\ai-employee-org\scripts\install-delivery-worker-cron.py
py -3 optional-skills\autonomous-ai-agents\ai-employee-org\scripts\install-secretary-heartbeat-cron.py
py -3 optional-skills\autonomous-ai-agents\ai-employee-org\scripts\install-self-improver-cron.py

# Apply NVIDIA-primary + fallback stack to ~/.hermes/config.yaml
py -3 scripts\apply_operator_stack.py --stack-file config\operator\ai-employee-stack.yaml
```

### Model / 429 policy

| Tier | Provider | Model |
|------|----------|-------|
| Primary | `nvidia` | `nvidia/nemotron-3-super-120b-a12b` |
| Fallback 1–3 | `nous` | free Nemotron / `auto-free` |
| Final only | `custom` | `http://127.0.0.1:8080/v1` llama |

Async subagents inherit the parent `fallback_providers` chain.

## Verification

```bash
hermes kanban list --status ready,running
hermes kanban assignees
hermes -p secretary kanban tail
hermes gateway status
hermes cron list
```

Dashboard: `hermes dashboard` → Kanban → confirm lanes per profile.

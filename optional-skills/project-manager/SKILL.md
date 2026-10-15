---
name: project-manager
description: Project context isolation and state management system. Creates per-project STATUS.md files for seamless context recovery across sessions. Supports intent recognition, tiered templates, and write-safe updates. Use when starting a new project, resuming past work, or managing multiple concurrent projects.
version: 2.2.0
author: Kris
license: MIT
metadata:
  hermes:
    tags: [Project-Management, Context-Isolation, State-Management, STATUS.md, Multi-Project]
    related_skills: [hermes-agent, memory]
---

# Project Manager v2.2 — Project Context Isolation

## Architecture: Memory vs File

**Agents are stateless** and must rely on the filesystem for continuity.

* **Global Memory (`MEMORY.md` / `memory/*.md`)**: Only for **user preferences** and **cross-project lessons**. **Never** store project-specific progress here.
* **Project State (`STATUS.md`)**: Stores **project progress**. Each project has its own isolated `STATUS.md`.
* **Project Index (`projects/index.md`)**: Maintains a registry of all projects for fuzzy lookup and quick recovery. **Auto-created on first use if missing.**
* **Recovery**: When a user mentions a project, the agent **must read** that project's `STATUS.md` — never rely on conversation memory.

## Core Principles

1. **Physical Isolation**: Each project has its own `STATUS.md` — never shared.
2. **Explicit State**: Agent has no memory. Anything not written to a file didn't happen.
3. **Memory Lean**: Project progress lives only in `STATUS.md`.
4. **Tiered Management**: Not every task deserves a full STATUS.md — match complexity to template.
5. **Intent-First**: No rigid keywords — infer intent from conversation context.
6. **Write-Safe**: Always read latest `STATUS.md` before writing. Merge, then write. Reduces accidental overwrites within a session. (Note: true multi-session atomic safety requires locking/CAS.)

## Directory Structure

All project files live under `workspace/projects/`:

```text
workspace/projects/
├── index.md              # 📇 Project index (auto-maintained)
├── {project-name}/
│   ├── STATUS.md         # 🧠 Core memory: progress, todos, key decisions
│   ├── docs/             # Proposals, reports, research data
│   └── src/              # Code, deliverables
└── _archive/             # 📦 Archived projects (completed/abandoned)
    └── {project-name}/
        └── STATUS.md     # Historical record preserved
```

## 📇 Project Index: `projects/index.md`

```markdown
# Project Index
> Auto-maintained, sorted by last active (descending)

| Project | Directory | Last Updated | Status | Notes |
|---------|-----------|-------------|--------|-------|
| english-reader | english-reader/ | 2026-04-20 | Active | English learning tool |
| cmra-geo | cmra-geo/ | 2026-04-18 | Paused | Awaiting requirements |
```

**Maintenance Rules:**
- **First use**: If `index.md` doesn't exist, auto-create with header
- New project: Add to top of index.md
- Resume project: Update "Last Updated", move row to top
- Pause project: Set status to "Paused" with reason
- Archive project: Remove from index or mark "Archived"

---

## Tiered STATUS.md Templates

Choose based on complexity — **don't over-engineer**.

For full template content, see `references/templates.md`.

### Template Selection Rules

| Scenario | Template |
|----------|----------|
| Multi-phase project >3 days | A Full |
| Single task (e.g., "refactor this endpoint") | B Lightweight |
| Research/analysis temp task | B Lightweight |
| Multi-person collaboration or long-term maintenance | A Full |
| Unsure | Start with B, upgrade to A if complexity grows |

---

## Workflow

### 1. Intent Recognition (Auto-Activated)

**No rigid keywords** — infer intent from context:

| User Intent | Typical Expression | Action |
|-------------|-------------------|--------|
| New project | "new project XX", "start XX", "create XX" | Initialize |
| Resume project | "back to XX", "continue XX", "that XX from last time" | Restore context |
| Save state | "hold on", "saving state", "pause" | Persist to disk |
| View projects | "what projects", "project list", "check XX progress" | Show index or status |
| Switch project | "switch to XX", "forget XX, let's do YY" | Save current + resume target |
| Implicit resume | "that plan from last time", "pick up where we left off" | Auto-infer and read STATUS.md |

**Recognition Priority:**
1. Explicit project name → Direct match
2. Vague reference ("that one", "last time", "continue") → Infer from most recently active in index.md
3. **Cannot determine** → Show 2-3 most recent active projects for user to choose. **Never guess blindly.**

### 2. Initialize New Project

**Input**: `new project CMRA GEO plan`

**Actions:**
1. Directory name in kebab-case: `cmra-geo`
2. Create directory `workspace/projects/cmra-geo/`
3. Choose A/B template based on complexity, create `STATUS.md`
4. **Ensure `index.md` exists**: Auto-create with header if missing
5. Update `projects/index.md` (add to top)
6. Check for git repo (`git -C workspace/projects/cmra-geo rev-parse --is-inside-work-tree`), record initial commit hash if present
7. **Reply**: "✅ Created project `cmra-geo`. Tell me the project goal and first step."

### 3. Resume Existing Project

**Input**: `back to CMRA project` or implicit resume

**Actions:**
1. **Fuzzy match**: User's project name may be imprecise — find closest match in index.md
2. **Force read**: `read(workspace/projects/{name}/STATUS.md)`
3. **Sync summary**: "📖 Restored `cmra-geo` context. Current: [brief progress]. Todo: [list]. Where to continue?"
4. **Update index.md**: Move project to top by last active time
5. **No assumptions**: If file missing or no match, ask user to confirm

### 4. State Update

**When to update** (any of):
- ✅ Completed a subtask
- ✅ Made a new technical decision
- ✅ Discovered a blocking bug
- ✅ User provided new requirements or constraints

**When NOT to update:**
- ❌ Just renamed a variable / adjusted comments
- ❌ Still thinking through options, nothing decided
- ❌ Tried different approaches but ended up with the original plan

**Update Strategy:**
- Full template (A): Update relevant sections, keep concise
- Lightweight template (B): Update "Current Status", "Decisions" (if any), and "Next Step"
- If B template project exceeds 3 days or grows complex, **auto-upgrade to A**

### 5. Pause / Switch

**Input**: `hold on` / `saving state` / `switch to XX`

**Actions:**
1. Summarize current conversation conclusions
2. Update current project's `STATUS.md`: modify "Current Status", update "Todo"
3. Update `index.md`: set status to "Paused", record reason
4. If switching, resume target project
5. **Reply**: "✅ State saved. Say 'back to [Name]' next time to pick up seamlessly."

### 6. Cleanup & Archive

**Periodic check (e.g., during heartbeat):**
- Projects inactive >30 days → Mark "possibly archive", ask user
- Completed projects → Set status to "Completed"
- Abandoned projects → Set status to "Abandoned"

**Archive Actions:**
1. Move `projects/{name}/` entirely to `projects/_archive/{name}/`
2. Preserve STATUS.md as historical record
3. Remove from `index.md` or mark "Archived" and move to bottom
4. Keep one line in index.md: `| {name} | _archive/{name}/ | [last date] | Archived | [reason] |`

---

## Relationship with Daily Memory

| Information Type | Location | Example |
|-----------------|----------|---------|
| User preferences | `MEMORY.md` | "Prefers concise answers" |
| Cross-project lessons | `MEMORY.md` | "X approach caused Y pitfall last time" |
| Daily work log | `memory/YYYY-MM-DD.md` | "Advanced CMRA auth module today, see STATUS.md for details" |
| Project progress | `projects/{name}/STATUS.md` | "CMRA: auth module done, starting permissions" |
| Technical details | `projects/{name}/status/` | Test logs, design drafts |

**Principle**: Read project progress from STATUS.md, distill lessons into MEMORY.md — the two never overlap. Daily logs contain **only a one-line summary + reference to STATUS.md**, no duplicated details.

---

## Git Integration (Optional)

If project directory is a git repo (check with `git -C projects/{name} rev-parse --is-inside-work-tree`):
- STATUS.md header records `Git: [latest commit hash]`
- On state update, run `git -C projects/{name} log --oneline -1` to get latest commit
- Helps quickly locate code state

---

## Strict Constraints

1. **Memory Ban**: Never use memory tools or `MEMORY.md` to record project progress. Project progress lives only in `STATUS.md`.
2. **Read-First**: Before answering project-related questions, **always read** `STATUS.md`. Don't rely on previous turn's memory (it may be truncated).
3. **Minimalism**: STATUS.md records only "conclusions" and "next steps", not process流水账.
4. **Template Downgrade**: Small tasks use lightweight template — avoid over-engineering.
5. **Fuzzy Match**: When user says "that one from last time" or "continue", actively infer the project; if unsure, show recent projects for selection — never guess blindly.
6. **Write-Safe**: Read latest STATUS.md before writing, merge, then write — reduces accidental overwrites within a session. (True multi-session atomic safety requires locking/CAS.)
7. **Index Self-Healing**: Auto-create on first use if missing; auto-add projects whose directories exist but aren't in index.

---

*v2.2 — Templates moved to references/, concurrent-safety claims softened with TOCTOU limitation noted, git commands scoped to project directory, worktree detection via git-native probe.*

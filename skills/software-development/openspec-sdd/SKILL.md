---
name: openspec-sdd
description: "Spec-driven development (SDD) for AI coding assistants using OpenSpec. Covers init, slash commands, artifact types, delta specs, archive flow, config, and custom schemas — right-sized for LLM agents to follow without memorizing the docs."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [openspec, sdd, spec-driven, ai-coding, planning]
---

# OpenSpec — Spec-Driven Development for AI Coders

[OpenSpec](https://github.com/Fission-AI/OpenSpec) is the most-starred spec framework for AI coding
assistants (49.5k ⭐). It replaces vague prompts and chat-only planning with lightweight,
version-controlled artifacts that survive context resets and keep human + AI aligned before a
single line of code is written.

## Why It Exists

AI coding assistants are powerful but unpredictable when requirements live only in chat history.
OpenSpec adds a lightweight spec layer so you **agree on what to build before any code is written**.

```
fluid not rigid         — no phase gates, work on what makes sense
iterative not waterfall — learn as you build, refine as you go
easy not complex        — lightweight setup, minimal ceremony
brownfield-first        — works with existing codebases, not just greenfield
```

---

## Setup

```bash
# Install (Node.js 20.19+ required)
npm install -g @fission-ai/openspec@latest

# Initialize in your project
cd your-project
openspec init
```

`openspec init` creates:
```
openspec/
├── openspec/specs/          # Source of truth — how your system currently works
├── openspec/changes/        # Proposed changes — one folder per change
├── openspec/config.yaml     # Project config — context, rules, schema
└── .claude/skills/          # Auto-generated slash commands for your AI tool
```

Pass `--no-interactive` for scripting.

---

## Project Structure

```
openspec/
├── specs/                       # ← Source of truth
│   ├── auth/spec.md
│   ├── payments/spec.md
│   └── notifications/spec.md
├── changes/                     # ← Active work
│   └── add-2fa/
│       ├── proposal.md
│       ├── design.md
│       ├── tasks.md
│       └── specs/                # Delta specs (what's changing)
│           └── auth/spec.md
├── changes/archive/             # ← History (date-prefixed change folders)
│   └── 2025-01-24-add-2fa/
├── schemas/                     # ← Custom workflows (optional)
│   └── my-workflow/schema.yaml
├── config.yaml                  # ← Project config — context, rules
└── .openspec.yaml               # ← Project metadata (version)
```

**`specs/`** — source of truth describing current system behavior.
**`changes/<name>/`** — one proposed modification, self-contained.
**`changes/archive/`** — completed changes, preserved forever for audit trail.

---

## Core Artifacts (Spec-Driven Schema)

The default `spec-driven` schema has these artifacts:

| Artifact | Purpose | Creates |
|----------|---------|---------|
| `proposal.md` | Why, what, scope | Intent + scope + approach |
| `specs/<domain>/spec.md` | Delta spec | ADDED / MODIFIED / REMOVED requirements |
| `design.md` | How — architecture | Decisions, data flow, file changes |
| `tasks.md` | Implementation checklist | Numbered, checkable steps |

---

## Slash Commands (OPSX Workflow)

OpenSpec registers slash commands in your AI tool (Claude Code, Cursor, Windsurf, etc.).
Two modes:

### Default Quick Path (`core` profile)

```
/opsx:propose  ➜  /opsx:apply  ➜  /opsx:sync  ➜  /opsx:archive
```

### Expanded Path (enable with `openspec config profile` + `openspec update`)

```
/opsx:new ──► /opsx:continue ──► /opsx:ff ──► /opsx:apply
                         │                      │
                         ▼                      ▼
               step-by-step          all-at-once
```

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `/opsx:propose` | Create change + all planning artifacts | Fast start (default) |
| `/opsx:explore` | Think through an idea | Unclear requirements |
| `/opsx:new` | Scaffold change folder only | Expanded mode |
| `/opsx:continue` | Create *one* artifact | Expanded mode, step control |
| `/opsx:ff` | Create *all* artifacts at once | Clear scope, expanded mode |
| `/opsx:apply` | Implement tasks | Ready to code |
| `/opsx:verify` | Validate impl vs artifacts | Expanded mode, before archive |
| `/opsx:sync` | Merge delta specs into main | Expanded mode (usually auto) |
| `/opsx:archive` | Complete + preserve the change | All work finished |
| `/opsx:bulk-archive` | Archive multiple changes | Expanded mode, parallel work |
| `/opsx:onboard` | Guided tutorial through full flow | New users |

---

## End-to-End Flow (with Expanded Commands)

### 1 — Explore (optional — for unclear requirements)

```
You: /opsx:explore
AI: What would you like to explore?
You: How should we handle authentication for the mobile app?
AI: [Investigates codebase, compares options] Here are three approaches:
    1. JWT tokens… 2. OAuth2… 3. Extend current sessions…
    Which direction interests you?
```

### 2 — Create Change

```
You: /opsx:new add-jwt-auth          # scaffold only (expanded)
AI:  Created openspec/changes/add-jwt-auth/
     Schema: spec-driven
     Ready to create: proposal
     Use /opsx:continue or /opsx:ff

# OR with default profile, one command creates everything:
You: /opsx:propose add-jwt-auth
AI:  Created openspec/changes/add-jwt-auth/
     ✓ proposal.md, ✓ specs/, ✓ design.md, ✓ tasks.md
```

### 3 — Generate Artifacts

```
You: /opsx:ff add-jwt-auth           # fast-forward: all at once
# or
You: /opsx:continue                  # step-by-step review
```

Output:
```
✓ openspec/changes/add-jwt-auth/proposal.md
✓ openspec/changes/add-jwt-auth/specs/auth/spec.md
✓ openspec/changes/add-jwt-auth/design.md
✓ openspec/changes/add-jwt-auth/tasks.md
```

### 4 — Implement

```
You: /opsx:apply add-jwt-auth
AI:  Working through tasks...
     ✓ 1.1 Create JWT signing service
     ✓ 1.2 Implement refresh token rotation
     ✓ 2.1 Add auth middleware
     ...
     All tasks complete!
```

### 5 — Verify (expanded mode only)

```
You: /opsx:verify add-jwt-auth
AI:  COMPLETENESS  ✓
     CORRECTNESS   ✓ (1 warning)
     COHERENCE     ✓
     Ready to archive with 1 warning.
```

Three dimensions:

| Dimension | Validates |
|-----------|-----------|
| **Completeness** | All tasks done, all requirements implemented, scenarios covered |
| **Correctness** | Implementation matches spec intent, edge cases handled |
| **Coherence** | Design reflected in code, naming consistent |

### 6 — Archive

```
You: /opsx:archive add-jwt-auth
AI:  ✓ Synced specs to openspec/specs/auth/spec.md
     ✓ Moved to openspec/changes/archive/2025-01-24-add-jwt-auth/
     Done!
```

Archive merges delta specs into `openspec/specs/` and moves the change folder to
`changes/archive/YYYY-MM-DD-<name>/`, preserving all artifacts as an audit trail.

### 7 — Bulk Archive (expanded, for parallel work)

```
You: /opsx:bulk-archive
AI:  Found 3 completed changes:
     - add-jwt-auth
     - fix-login-bug
     - update-footer
     Archive all? ( handles spec conflicts via codebase inspection )
```

---

## Artifact Templates

### `proposal.md`

```markdown
# Proposal: <Feature Name>

## Intent
<!-- What problem are you solving? Why this, why now? -->

## Scope
In scope:
- [capability 1]
- [capability 2]
Out of scope:
- [explicitly excluded]
-[...]


## Approach
<!-- High-level strategy. Keep it brief — design.md holds the details. -->
```

### `specs/<domain>/spec.md` (delta spec)

Uses ADDED / MODIFIED / REMOVED sections against the base spec:

```markdown
# Delta for Auth

## ADDED Requirements

### Requirement: Two-Factor Authentication
The system SHALL support TOTP-based 2FA.

#### Scenario: OTP required
- GIVEN a user with 2FA enabled
- WHEN the user submits valid credentials
- THEN an OTP challenge is presented

## MODIFIED Requirements

### Requirement: Session Timeout
The system SHALL expire sessions after 15 minutes of inactivity.
(Previously: 30 minutes)

#### Scenario: Idle timeout
- GIVEN an authenticated session
- WHEN 15 minutes pass without activity
- THEN the session is invalidated

## REMOVED Requirements

### Requirement: Remember Me
(Deprecated in favor of 2FA)
```

**RFC 2119 keywords:** `SHALL`/`MUST` = absolute, `SHOULD` = recommended, `MAY` = optional.
Good specs avoid class names, framework choices, and implementation steps.

### `design.md`

```markdown
# Design: <Feature>

## Technical Approach
<!-- How it works at a high level. Architecture diagrams welcome. -->

## Architecture Decisions

### Decision: <name>
**Chosen:** X
**Reason:** ...

## Data Flow
```
[ASCII diagram]
```

## File Changes
- `path/to/file.ts` (new) — description
- `other/file.ts` (modified) — description
```

### `tasks.md`

```markdown
# Tasks

## 1. <Group Name>
- [ ] 1.1 First concrete step
- [ ] 1.2 Second step
- [ ] 1.3 Third step

## 2. <Group Name>
- [ ] 2.1 ...
```

Tasks should be small (one session each), grouped by concern, use hierarchical numbering.
Check them off as you go: `[ ]` → `[x]`.

---

## Spec Format Guide (RFC 2119)

| Keyword | Meaning |
|---------|---------|
| `MUST` / `SHALL` | Absolute requirement |
| `SHOULD` | Recommended; exceptions allowed |
| `MAY` | Optional |

**Good scenarios:**
- Testable — you could write an automated test for them
- Cover happy path + edge cases
- Use Given/When/Then

**Don't put in specs:**
- Internal class/function names
- Framework/library choices
- Step-by-step implementation details
- Detailed execution plans → belongs in `design.md` or `tasks.md`

**Lite vs Full spec:**
- **Lite (default):** short behavior-first requirements, clear scope and non-goals, a few acceptance checks
- **Full:** cross-team/API/contract changes, security/privacy concerns, migration work

---

## Delta Specs — What Happens on Archive

| Section | On Archive |
|---------|------------|
| `## ADDED Requirements` | Appended to main spec |
| `## MODIFIED Requirements` | Replaces the existing requirement |
| `## REMOVED Requirements` | Deleted from main spec |

**Specs are the source of truth.** They grow as changes are archived. After each archive the
`openspec/specs/` tree describes *how the system currently behaves*.

---

## When to Update vs Start Fresh

**Update the existing change when:**
- Same intent, refined execution
- Scope narrows (MVP first, rest later)
- Learning-driven corrections (codebase ≠ expectation)
- Design tweaks from implementation

**Start a new change when:**
- Intent fundamentally changed
- Scope exploded to different work entirely
- Original change can be "done" standalone
- Patches would confuse more than clarify

```
          Is this the same work?
                 │
      ┌──────────┼──────────┐
      │          │          │
   Same intent? >50% overlap?  Can original
      │          │          be "done"?
      │          │              │
   YES│         NO│            NO│
      ▼          ▼             ▼
   UPDATE     UPDATE       UPDATE
                 │
                YES
                 │
                NEW
```

---

## CLI Commands

The `openspec` CLI (Node.js) powers the slash commands. Useful for scripting, CI, and validation.

```bash
openspec init                              # Interactive setup
openspec config                           # Show config
openspec config edit                      # Edit config in $EDITOR
openspec list                             # List active changes
openspec show <name>                      # Show change details
openspec status                           # Overall project status
openspec validate                         # Validate all spec formatting
openspec view                             # Interactive dashboard

openspec new change <name> [--schema N]   # Start a change
openspec continue [--change N]            # Next artifact
openspec ff [--change N]                  # Fast-forward all artifacts
openspec apply [--change N]               # Implement tasks
openspec sync [--change N]                # Merge delta specs
openspec archive [--change N]             # Archive change
openspec bulk-archive                     # Archive multiple

openspec schema list                      # Show built-in schemas
openspec schema init <name>               # Create custom schema
openspec schema fork <from> <to>          # Fork an existing schema
openspec schema validate <name>           # Validate a schema
openspec schema which --all               # Show where each schema loads from
```

---

## Project Config (`openspec/config.yaml`)

```yaml
schema: spec-driven

context: |
  Tech stack: TypeScript, React, Node.js, PostgreSQL
  API style: RESTful, JSON responses
  Testing: Vitest unit, Playwright e2e
  Style: ESLint + Prettier, strict TypeScript

rules:
  proposal:
    - Include rollback plan
    - Identify affected teams
  specs:
    - Use Given/When/Then format for scenarios
  design:
    - Include sequence diagrams for complex flows
    - Document data migration steps
  tasks:
    - Each task must be independently deployable
```

**Fields:**

| Field | Purpose |
|-------|---------|
| `schema` | Default schema for new changes |
| `context` | Project-wide context injected into ALL artifacts |
| `rules.<artifact_id>` | Per-artifact rules — injected only for that artifact |

**Schema resolution order (highest → lowest):**
1. CLI flag (`--schema <name>`)
2. Change metadata (`.openspec.yaml` in change folder)
3. `openspec/config.yaml`
4. Default: `spec-driven`

---

## Spec Domains

Organize `openspec/specs/` by domain — logical groupings:

```
openspec/specs/
├── auth/spec.md             # Authentication + sessions
├── payments/spec.md         # Payment processing
├── notifications/spec.md    # Notification system
└── ui/spec.md               # UI behavior and themes
```

Patterns:
- **By feature area:** `auth/`, `payments/`, `search/`
- **By component:** `api/`, `frontend/`, `workers/`
- **By bounded context:** `ordering/`, `fulfillment/`, `inventory/`

---

## Workflow Checklist for New Features

When starting any new feature, bug fix, or project, work through this ordered checklist instead
of going straight to code. Each step produces durable artifacts other agents or teammates can
read without live context.

### Pre-implementation — do NOT skip

1. [ ] **`/opsx:propose <name>`** or **`/opsx:new <name>` → `/opsx:ff`** — generates `proposal.md`, `design.md`, `tasks.md`, and delta `specs/`
2. [ ] **Review `proposal.md`** with the user — confirm intent and scope before any coding
3. [ ] **Review `design.md`** — verify architecture decisions and file changes list
4. [ ] **Review `specs/<domain>/spec.md`** — confirm requirements and scenarios capture the ask correctly
5. [ ] **Update any missing artifacts** before proceeding — proposals/specs are cheaper to fix now

### Implementation

6. [ ] **`/opsx:apply <name>`** — AI works through `tasks.md` checkboxes
7. [ ] **Update artifacts if needed** while implementing (design vs reality changes)
8. [ ] **Commit** with a message referencing the change name

### Completion

9. [ ] **`/opsx:verify <name>`** — catch mismatches against artifacts
10. [ ] **`/opsx:archive <name>`** — archives and merges delta specs into source of truth

---

## Integration with Hermes / Delegated Agents

When Hermes spawns subagents or kanban workers for feature work:

- **Pre-flight:** require `proposal.md` + `specs/` + `design.md` to exist before `/opsx:apply`.
- **Artifact check:** validate `tasks.md` is complete (`[ ]` count == 0) before archive.
- **Multi-repo:** use OpenSpec workspaces (`openspec workspace setup`) when the work spans linked repos or folders.
- **Audit trail:** archived changes stay in `openspec/changes/archive/` — never delete them.

---

## Error Patterns / Gotchas

| Symptom | Fix |
|---------|-----|
| Commands not recognized | Run `openspec init` → `openspec update` in project root. Check `.claude/skills/` exists. |
| "Schema not found" | Run `openspec schemas` to list available. Set `openspec config.yaml` `schema:` field. |
| "No artifacts ready" | Check dependencies — artifacts must exist in dependency order. Run `openspec status`. |
| Config ignored | File must be at `openspec/config.yaml` (not `.yml`). YAML syntax must be valid, 50KB limit. |
| Token expired (Hermes) | Not an OpenSpec issue — Hermes Nous Portal. Run `hermes model --no-browser` + sync script. |

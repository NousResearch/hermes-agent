---
name: analysis
description: Understand an existing codebase through systematic exploration
---

# [A] Analysis — Understand Your Own Codebase

**Goal:** Systematically explore and document an existing codebase's architecture, patterns, and dependencies.

**When to use:** Joining an existing project. Need to understand code before making changes. Architecture documentation is missing or outdated. Preparing for a major refactor or migration.

---

## CORE PRINCIPLES

- **Question-driven** — Every analysis starts with a clear question. Without a question, exploration is aimless. Define what you need to know before reading a single file.
- **Systematic** — Follow a structured path: question, scan, map, document. Do not jump to conclusions from reading one file. Cover breadth before going deep.
- **Document everything** — Findings that are not written down are lost. Every observation, pattern, risk, and recommendation goes into the output document. The analysis document becomes a team asset.

---

## INITIALIZATION

### Design Log
Read `{output_folder}/_progress/00-design-log.md`. Check Current and Backlog for context.

### Essential Guides

- **[Execution Principles](data/guides/EXECUTION-PRINCIPLES.md)** — Document before acting, plan-then-execute
- **[Session Protocol](data/guides/SESSION-PROTOCOL.md)** — Read dialog, verify plan, present status

---

## STEPS

Execute steps in `./steps-a/`:

| Step | File | Purpose |
|------|------|---------|
| 01 | step-01-define-question.md | Articulate the question, define scope and output |
| 02 | step-02-scan-codebase.md | Scan structure, tech stack, entry points |
| 03 | step-03-map-architecture.md | Map components, data flow, dependencies |
| 04 | step-04-document-findings.md | Create architecture document with diagrams |

**Flow:** 01 → 02 → 03 → 04

### Critical Rules

- **ALWAYS** start with a clear question before scanning code
- **ALWAYS** document observations as you go, not from memory at the end
- **ALWAYS** define scope boundaries to avoid unbounded exploration
- **ALWAYS** set a time box to prevent analysis paralysis
- **ALWAYS** produce a concrete output document, not just verbal findings

---

## AFTER COMPLETION

1. Append a progress entry to `{output_folder}/_progress/00-design-log.md` under `## Progress`:
   `### [date] — Analysis: [what was analyzed, key findings]`
2. Suggest next action: feed into development, reverse engineer further, or start prototyping

---
name: reverse-engineering
description: Analyze any software or website to extract specifications and design patterns
---

# [R] Reverse Engineering — Any Software/Site → Specs & Extraction

**Goal:** Analyze existing software or public websites to extract page specifications, design systems, and architectural understanding.

**When to use:** Need to understand an existing product (yours or public). Want to create WDS specifications from existing software. Competitive analysis through design extraction. Migrating or rebuilding an existing product.

**Two modes:**
- **Internal:** Analyze your own codebase (source code access)
- **External:** Analyze any public website (browser-based, no source needed)

---

## CORE PRINCIPLES

- **Observe first, then extract** — Do not start generating specs from the first page you see. Explore the full scope of the target first, then systematically extract patterns. Premature extraction leads to incomplete and inconsistent results.
- **Respect intellectual property** — Reverse engineering is for learning and specification, not for copying proprietary code. Extract patterns, structures, and design tokens. Do not copy proprietary assets, fonts, or copyrighted content.
- **Focus on patterns, not pixel-perfect copy** — The goal is to understand the design system and page structures well enough to build something equivalent or better. Capture the rules and relationships, not individual pixel values.

---

## INITIALIZATION

### Design Log
Read `{output_folder}/_progress/00-design-log.md`. Check Current and Backlog for context.

### Essential Guides

- **[Execution Principles](data/guides/EXECUTION-PRINCIPLES.md)** — Document before acting, plan-then-execute
- **[Session Protocol](data/guides/SESSION-PROTOCOL.md)** — Read dialog, verify plan, present status

---

## STEPS

Execute steps in `./steps-r/`:

| Step | File | Purpose |
|------|------|---------|
| 01 | step-01-identify-target.md | Define target, access method, extraction goals |
| 02 | step-02-explore-and-capture.md | Explore structure, capture patterns and inventory |
| 03 | step-03-generate-specs.md | Generate WDS-format page specifications |
| 04 | step-04-extract-design-system.md | Extract design tokens and component catalog |

**Flow:** 01 → 02 → 03 → 04

### Critical Rules

- **ALWAYS** explore the full target before extracting — do not start with the first page
- **ALWAYS** document observations before generating specs
- **ALWAYS** generate output in `document_output_language`
- **ALWAYS** produce WDS-compatible artifacts that can feed into prototyping
- **NEVER** copy proprietary code, assets, or copyrighted content

---

## AFTER COMPLETION

1. Append a progress entry to `{output_folder}/_progress/00-design-log.md` under `## Progress`:
   `### [date] — Reverse Engineering: [what was extracted]`
2. Suggest next action: feed specs into prototyping, analyze further, or start development

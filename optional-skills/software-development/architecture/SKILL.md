---
name: architecture
license: MIT
version: 1.0.1
author: afonsoft
platforms: [linux, macos, windows]
description: "Owns docs/architecture/: ADRs, design docs, diagrams."
metadata:
  hermes:
    tags: [Architecture, ADR, Diagrams, Design-Docs, Spec-Driven]
  version: "1.0.1"
  visibility: public
  author: afonsoft
  url: https://github.com/afonsoft/skills
---

# Architecture

The single entry point for **all repository architecture deliverables**. Instead of calling diagram skills directly, agents invoke `/architecture` and this skill routes each deliverable to the right engine. It owns the `docs/architecture/` folder end to end: ADRs, design documents, and every diagram format.

All questions and confirmations directed at the user must be in **Portuguese (pt-BR)**. Internal reasoning and documentation are in English.

## Ownership — `docs/architecture/`

Everything that describes or decides the system's architecture lives here:

```text
docs/architecture/
├── AD-NNNN-<slug>.md                    # Architecture Decision Records
├── architecture-design.md               # Architecture design doc
├── system-design.md                     # System design doc
├── api-design.md                        # API design doc
├── database-design.md                   # Database design doc
├── feature-<name>-design.md             # Feature-scoped design doc
├── system-architecture.md               # Markdown doc with embedded Mermaid
├── <name>_<type>_<title>.mmd            # Mermaid sources
├── <name>_<type>_<title>.drawio         # Editable draw.io sources
├── <name>_<type>_<title>.png|.svg|.pdf  # Exported images
├── <name>.json                          # Archify candidate specs
└── <name>.html                          # Archify interactive HTML artifacts
```

Ensure the directory exists before writing: `mkdir -p docs/architecture`.

> Projects scaffolded by `/scaffold-mvp` may also keep ADRs under `docs/adr/`. Treat `docs/architecture/` as canonical; when `docs/adr/` exists, keep it consistent with the AD index below.

## Engine Routing

Pick the engine by deliverable, never by habit:

| Deliverable | Engine | Why |
| --- | --- | --- |
| ADR (`AD-NNNN`) | this skill (template below) | Structured Markdown decision record |
| Design doc (system / API / DB / feature) | `/mermaid-architecture` assets | Templates live in `mermaid-architecture/assets/` |
| Markdown-native diagram (C4, sequence, flow, ER, state) | `/mermaid-architecture` | Renders in GitHub/Obsidian/wikis, works headless |
| Editable `.drawio` diagram or PNG/SVG export | `/drawio-architecture` | draw.io MCP or desktop CLI; fall back to `/mermaid-architecture` if graphical rendering fails in headless environments |
| Interactive standalone HTML diagram (presentations, explorability, trace motion) | `archify` (optional, only if already installed) | Self-contained HTML + inline SVG with themes and export — see below |
| Evidence-backed audit (delivered state vs code/specs/docs) | `/gap-analysis` | Owned by this skill at the end of the pipeline — see `references/gap-audit-handoff.md` |

**Default order in the Orchestrator pipeline (Phase 5):** `/drawio-architecture` for the editable system diagram → `/mermaid-architecture` for native Markdown diagrams → `archify` for the interactive runtime diagram when installed → `/gap-analysis` for the final evidence-backed audit.

## Guardrails

- **Never install external skills.** Archify is used only when already installed in the environment. If absent, skip it — do not prompt for, suggest, or run any install command — consistent with the orchestrator's "no silent execution" and "trusted delegation only" rules.
- **Graceful degradation.** If archify is absent, deliver the same content through `/mermaid-architecture` (and `/drawio-architecture` when editable output is needed). A missing optional engine never blocks the pipeline.
- **Never bypass the gap-analysis gate.** `gap-analysis` is read-only until the user answers its own pt-BR approval gate. This skill invokes it and reports the outcome; it never pre-approves, answers for the user, or skips the audit silently.
- **Verify before claiming.** Validate Mermaid syntax, `.drawio` XML, and archify receipts with each engine's own validator before reporting success.

## Workflow

0. **Preflight** (optional but recommended): run `scripts/architecture-doctor.sh` for a read-only inventory of `docs/architecture/`, the next free ADR number, engine availability (`mmdc`, `drawio` CLI), and delegated-skill detection (`mermaid-architecture`, `drawio-architecture`, `gap-analysis`, `archify`).
1. **Classify the request**: ADR, design doc, Markdown diagram, editable diagram, or interactive HTML. Multiple deliverables may apply.
2. **ADRs**: check the highest existing `AD-NNNN` number, then write the next one with the template below.
3. **Design docs**: populate the matching template from `mermaid-architecture/assets/` into `docs/architecture/`.
4. **Diagrams**: route per the table. For code-grounded diagrams, extract evidence first (entry points, DI graph, `docker-compose.yml`, Helm/Terraform, CI files).
5. **Verify**: run each engine's validation step; fix and re-run until green.
6. **Gap audit**: invoke `/gap-analysis` per its own contract — pass the delivery context (SPEC paths, commits, files written under `docs/architecture/`) so its AS-IS × TO-BE matrix starts warm. Its approval gate decides whether confirmed gaps become Issues; see `references/gap-audit-handoff.md`.
7. **Report** the files written under `docs/architecture/` and the audit outcome.

## ADR Convention

File name: `docs/architecture/AD-NNNN-<kebab-slug>.md` (sequential, zero-padded — e.g. `AD-0002-postgres-persistence.md`).

```markdown
# AD-NNNN — <Decision Title>

## Context
[Forces at play: constraints, requirements, prior decisions.]

## Decision
[What was decided and the key alternatives rejected.]

## Consequences
- Positive: [...]
- Trade-off: [...]

## Related SPEC
- [.specs/SPEC-{YYYYMMDD}-{slug}.md](../../.specs/SPEC-{YYYYMMDD}-{slug}.md)
```

Keep an index of ADRs at the top of `docs/architecture/README.md` when that file exists; never renumber existing records — supersede them with a new AD.

## Archify (optional, interactive HTML diagrams)

`archify` ([github.com/tt-a1i/archify](https://github.com/tt-a1i/archify)) produces explorable standalone HTML diagrams (inline SVG, dark/light themes, trace motion, PNG/JPEG/WebP/SVG export) from a small typed JSON spec — the right tool when stakeholders need a polished, interactive runtime view rather than a static diagram.

### Step 1 — Detect

Check whether the `archify` skill is already installed in any known skills root:

```bash
for d in ./.agents/skills ~/.agents/skills ~/.claude/skills ~/.devin/skills \
         ~/.config/devin/skills ~/.opencode/skills ~/.config/opencode/skills \
         ~/.cursor/skills ~/.copilot/skills ~/.gemini/skills ~/.codex/skills \
         ~/.codeium/windsurf/skills; do
  [ -f "$d/archify/SKILL.md" ] && echo "FOUND: $d/archify"
done
```

The installed skill is also visible when the agent runtime lists `archify` among available skills.

### Step 2 — Absent means skip (never install)

If not found, do **not** offer, suggest, or run any installation. Fall back to `/mermaid-architecture` and continue — a missing optional engine never blocks the pipeline. If the user wants interactive HTML later, they install `archify` themselves outside this skill's scope.

### Step 3 — Generate (only when already installed)

Invoke the installed `archify` skill with the repository-grounded prompt:

```text
Analyze this repository, then use archify to create a high-level runtime architecture diagram.
Show 8–12 core components, one primary path, external dependencies, and trust boundaries.
Put supporting detail in cards instead of adding more edges.
```

Then follow the archify authoring flow (type `architecture`, `meta.quality_profile: "showcase"`) and validate/deliver:

```bash
node <archify-skill-path>/bin/archify.mjs validate architecture docs/architecture/runtime-architecture.json --quality showcase --json
node <archify-skill-path>/bin/archify.mjs deliver  architecture docs/architecture/runtime-architecture.json docs/architecture/runtime-architecture.html --quality showcase --json
```

A non-zero exit is never success: fix the diagnosed field and re-run. Store both the candidate spec (`.json`) and the artifact (`.html`) under `docs/architecture/`.

## Orchestrator Integration

In the Orchestrator lifecycle (Phase 5 — Verification & QA), `/architecture` replaces the previous direct calls to `/drawio-architecture`, `/mermaid-architecture`, **and `/gap-analysis`** — the orchestrator makes a single call and this skill runs the full chain:

1. `/drawio-architecture` updates the editable `.drawio` system diagram (falling back to `/mermaid-architecture` when headless rendering fails).
2. `/mermaid-architecture` generates/updates native Mermaid diagrams and design docs in `docs/architecture/`.
3. `archify` (when installed) produces the interactive `runtime-architecture.html`.
4. `/gap-analysis` runs the final evidence-backed audit — invoked by this skill, never directly by the orchestrator. Its own pt-BR gate decides: approved confirmed gaps flow through `create-issues` back into the orchestrator's Phase 4 queue (Phase 5 repeats when they finish); declined gaps stay `Draft` and resurface in Phase 6.
5. `/create-readme` then references the generated diagrams in `README.md`.

When invoked mid-implementation (not only at Phase 5), apply the same routing: new architectural decision → ADR; doc drift → update design docs; structure change → regenerate affected diagrams.

## When to Use

- Any creation or update of files under `docs/architecture/` — ADRs, design docs, diagrams.
- The orchestrator reaches the architecture step of Phase 5.
- The user asks for an architecture diagram without specifying a format — route by deliverable.
- Documenting a significant architectural decision (ADR) mid-implementation.

- User asks or mentions this skill in English (e.g., "use /architecture", "update the architecture docs").
- O usuário pede ou menciona esta skill em português (ex.: "use /architecture", "atualize a documentação de arquitetura").

## When NOT to Use

- Architecture *code* improvements, coupling, or module seams → `/improve-codebase-architecture` (this skill documents architecture; that one refactors it).
- A single, already-specified diagram format → the specialized skill may be invoked directly (`/mermaid-architecture`, `/drawio-architecture`), though routing through `/architecture` is always correct.
- A standalone gap audit unrelated to architecture deliverables → `/gap-analysis` may be invoked directly; inside the pipeline it is always reached through this skill.

## Helpers

- `scripts/architecture-doctor.sh` — read-only preflight: `docs/architecture/` inventory, next ADR number, engine binaries, and delegated-skill detection.
- `references/architecture-inventory.md` — canonical layout, naming rules, and ADR lifecycle for `docs/architecture/`.
- `references/gap-audit-handoff.md` — contract for invoking `gap-analysis`: evidence context, approval gate, outcome mapping, degradation.

## References

- `/mermaid-architecture` — Markdown-native diagrams + design doc templates (`assets/`)
- `/drawio-architecture` — editable `.drawio` diagrams via MCP or local CLI export
- `/gap-analysis` — final evidence-backed audit, invoked by this skill (see `references/gap-audit-handoff.md`)
- `archify` — optional third-party interactive HTML diagrams, used only when already installed (https://github.com/tt-a1i/archify)
- `/scaffold-mvp` — creates `docs/architecture/` and `AD-0001` in new projects
- `/improve-codebase-architecture` — code-level architecture deepening (upstream consumer of these docs)
- `/create-readme` — surfaces the diagrams in `README.md`

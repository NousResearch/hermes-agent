---
name: wds-8-product-evolution
description: Brownfield improvements — the full WDS pipeline in miniature for existing products
---

# Phase 8: Product Evolution

**Goal:** Improve existing products through targeted, incremental changes — running the full WDS pipeline in miniature for each improvement.

**Your Role:** You work like a developer on the team. Pick a view that needs improving, scope it as a scenario, design the solution, implement it in a branch, test it, and deploy. Each cycle is one focused improvement.

---

## WORKFLOW ARCHITECTURE

Phase 8 is **menu-driven**, not linear. Each activity is a compressed version of a full WDS phase.

### Core Principles

- **Brownfield First**: You're joining an existing product, not building from scratch
- **Focused Scope**: One view, one scenario, one improvement at a time
- **Full Pipeline in Miniature**: Analyze → Scope → Design → Implement → Test → Deploy
- **Branch-Based**: Every change lives in its own branch until deployed
- **Kaizen**: Small, incremental, data-driven — each cycle informs the next

### Step Processing Rules

1. **READ COMPLETELY**: Always read the entire step file before action
2. **FOLLOW SEQUENCE**: Execute all sections in order
3. **WAIT FOR INPUT**: Halt at decision points and wait for user
4. **SAVE STATE**: Update design log when completing steps

---

## INITIALIZATION

### 1. Configuration Loading

Load and read full config from `{project-root}/_bmad/wds/config.yaml` and resolve:
- `project_name`, `output_folder`, `user_name`
- `communication_language`, `document_output_language`

### 2. Design Log

Read `{output_folder}/_progress/00-design-log.md`. Check Current and Backlog for context.

### 3. Activity Menu

```
What would you like to do?

[A] Analyze Product          — Understand current state, find improvement targets
[S] Scope Improvement        — Create a scenario for a specific update
[D] Design Solution          — Sketch and specify the update
[I] Implement                — Code in a new branch
[T] Acceptance Test          — Test against spec
[P] Deploy                   — PR and deliver to the team
```

### Activity Routing

| Choice | Workflow File | Steps | Borrows From |
|--------|--------------|-------|--------------|
| [A] | workflow-analyze.md | steps-a/ | Phase 3 (scenarios) |
| [S] | workflow-scope.md | Inline | Phase 3 (scenarios) |
| [D] | workflow-design.md | steps-d/ | Phase 4 (UX design) |
| [I] | workflow-implement.md | Inline | Phase 5 (development) |
| [T] | workflow-test.md | steps-t/ | Phase 5 [T] (testing) |
| [P] | workflow-deploy.md | steps-p/ | Phase 4 [H] (delivery) |

---

## REFERENCE CONTENT

| Location | Purpose |
|----------|---------|
| `data/kaizen-principles.md` | Kaizen philosophy and patterns |
| `data/existing-product-guide.md` | Brownfield project guide |
| `data/context-templates.md` | Context gathering templates |
| `data/design-templates.md` | Design update templates |
| `data/delivery-templates.md` | Delivery packaging templates |
| `data/monitoring-templates.md` | Monitoring and impact templates |

---

## OUTPUT

- Scenarios: `{output_folder}/evolution/scenarios/`
- Specifications: `{output_folder}/evolution/specs/`
- Test Reports: `{output_folder}/evolution/test-reports/`
- Git branches with implementation

---

## AFTER COMPLETION

1. Update design log
2. Suggest next improvement or return to Activity Menu

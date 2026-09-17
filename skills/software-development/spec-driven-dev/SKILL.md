---
name: spec-driven-dev
description: "Bootstrap Spec Kit + EARS + worldview pipeline per project."
version: 1.0.0
author: Ian Koratsky, Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [spec-driven-development, spec-kit, ears, worldview, decision-hud]
    category: software-development
    related_skills: [test-driven-development]
capability_class: LOCAL_MUTATION
mixed_capability: false
---

# Spec-Driven Dev Skill

Layers a standing worldview document, GitHub Spec Kit's
constitution/specify/plan/tasks/implement pipeline, and EARS-syntax
requirement discipline into one workflow for starting or extending a
project. Does not reimplement Spec Kit -- wraps the real `specify-cli`
(rung 2 of the footprint ladder: CLI command + skill, not new core
surface) and translates its output into this project's own skill area.

## When to Use

Starting a new project (or a substantial new feature on an existing one)
that warrants real spec-first discipline rather than ad hoc prompting --
not for a one-line fix or throwaway script.

## Prerequisites

- `uv` (for installing `specify-cli`; `init_project.sh` installs it if
  missing when `uv` itself is already present).
- `terminal` to run the init script and gate scripts.
- Optionally, the Decision HUD MCP tools (`decision_hud__decision_push`,
  `decision_hud__decision_list`) if the project has that plugin installed
  -- routing to Decision HUD degrades gracefully to `clarify` if absent
  (see Procedure step 4).

## How to Run

```bash
bash scripts/init_project.sh <project-name>       # new directory
bash scripts/init_project.sh . --here              # init in cwd
python3 scripts/translate_speckit_skills.py <project_dir>
python3 scripts/spec_decision_gate.py --questions
python3 scripts/spec_decision_gate.py --answers answers.json
```

## Quick Reference

| Step | Command | Produces |
|---|---|---|
| 1 | `init_project.sh` | `.specify/`, `.claude/skills/speckit-*`, seeded `docs/worldview.md`, seeded constitution, EARS reference docs |
| 2 | `translate_speckit_skills.py` | Hermes-frontmatter copies of the speckit-* skills under `.hermes/skills/` |
| 3 | `/speckit-constitution` (in Claude Code) | Refined `.specify/memory/constitution.md` |
| 4 | `/speckit-specify` | `spec.md` with EARS-formatted FR- lines (see `references/ears-syntax.md`) |
| 5 | `/speckit-clarify`, `/speckit-plan`, `/speckit-tasks` | Resolved ambiguities, `plan.md`, `tasks.md` -- every open question routed via `spec_decision_gate.py` first |
| 6 | `/speckit-implement`, `/speckit-converge` | Working code, verified against spec/plan/tasks |

## Procedure

1. **Bootstrap.** Run `init_project.sh` for a new project, or `. --here`
   inside an existing one. This installs real Spec Kit into the project
   (verified working: `specify init --integration claude` produces
   `.claude/skills/speckit-*` + `.specify/`), seeds `docs/worldview.md`
   from `templates/worldview.md` (only if it doesn't already exist --
   worldview is meant to be filled in gradually via conversation, not
   overwritten), adds the `@docs/worldview.md` import line to
   `CLAUDE.md`, and copies pre-seeded constitution defaults over the
   blank template.
2. **Translate skills.** Run `translate_speckit_skills.py <project_dir>`
   so the installed `speckit-*` skills carry Hermes-shaped frontmatter,
   not just Claude-Code-shaped frontmatter. This is a mechanical
   field-rename only (name/description/tags/category); it does not
   rewrite the skill body.
3. **Refine the constitution.** Run `/speckit-constitution` conversationally
   to replace the remaining `[BRACKETED]` placeholders in
   `constitution-defaults.md`'s seed with project-specific principles.
4. **Route every open question through the decision gate.** Any
   `[NEEDS CLARIFICATION]` marker or judgment call surfaced during
   `/speckit-clarify`, `/speckit-plan`, or `/speckit-tasks` is answered
   by `python3 scripts/spec_decision_gate.py --answers <answers.json>`
   BEFORE either assuming an answer or escalating it. Three outcomes:
   - `resolve_silently` -- cite the repo evidence in spec.md/plan.md and
     move on. No escalation.
   - `clarify_now` -- use the `clarify` tool; this blocks every later
     step and must resolve in the current turn.
   - `decision_hud` -- a genuine owner-policy/taste call that can wait.
     Push via `decision_hud__decision_push`, but first run the
     `card-type-gate` skill's `card_type_selector.py` on the SAME
     question to pick the real card shape (23 native shapes exist --
     `spider_compare` for multi-axis tech tradeoffs,
     `constrained_budget_split` for scope/budget tradeoffs,
     `pairwise_duel`/`sequence_order` for task prioritization, etc.).
     Only fall back to plain MCQ + free-text when card-type-gate itself
     returns `no_match` -- never default straight to `mcq_context`
     without running that gate. If the Decision HUD MCP tools aren't
     available in this session, degrade to `clarify` instead of silently
     skipping the question.
5. **Write requirements in EARS syntax.** `spec.md`'s FR- lines follow
   one of the five EARS patterns (`references/ears-syntax.md`). Before
   an EARS line becomes implementation scope, run it through the
   `ears-sensibility-gate` skill if present (checks proportionality,
   actor ownership, testability, conflicts) -- syntactic validity alone
   is not sufficient.
6. **Promote agreed requirements to structured JSON.** Validate against
   `references/ears-schema.json`. This is the layer that seeds
   property-based test generation later, distinct from and
   complementary to the hand-written tests `/speckit-implement` writes.
7. **Run the rest of the Spec Kit pipeline as normal**
   (`/speckit-plan`, `/speckit-checklist`, `/speckit-tasks`,
   `/speckit-analyze`, `/speckit-implement`, `/speckit-converge`),
   applying step 4's gate to every open question along the way.

## Pitfalls

- Don't reimplement Spec Kit's engine in this repo -- it is a real,
  actively maintained CLI tool (`specify-cli`); wrap it, don't fork it.
- Don't let `worldview.md` exceed ~200 lines -- prune to a distilled
  Tenets section and archive deliberation elsewhere once it grows.
- Don't default a Decision HUD escalation straight to `mcq_context`
  without running `card-type-gate` first -- that defeats the point of
  having 23 richer card shapes available.
- Don't silently assume an answer to an open spec-kit question just
  because escalating feels like friction -- run it through
  `spec_decision_gate.py` and follow its verdict.
- Don't treat a syntactically valid EARS line as automatically
  sensible -- run the sensibility gate before promoting it to scope.

## Verification

- `init_project.sh` produced `.specify/`, `.claude/skills/speckit-*`,
  `docs/worldview.md`, a seeded (not blank) constitution, and
  `docs/spec-driven/ears-syntax.md` + `ears-schema.json` in the target
  project.
- Every open question logged during `/speckit-clarify`/`/speckit-plan`/
  `/speckit-tasks` has a recorded `spec_decision_gate.py` verdict
  (resolve_silently + cited evidence, clarify_now + answer, or
  decision_hud + card_type from card-type-gate).
- No EARS line entered `tasks.md` scope without first passing
  `ears-sensibility-gate` (or an equivalent documented check) if that
  skill is present in the project.

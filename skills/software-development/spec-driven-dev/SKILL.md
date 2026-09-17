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
not for a one-line fix or throwaway script. **Also check this whenever
you're working in ANY project directory that lacks `.specify/`** -- if
this skill is loaded and the project has no `.specify/`, run
`init_project.sh . --here` before continuing, rather than skipping
bootstrap because it wasn't explicitly requested this turn.

## Prerequisites

- `uv` (for installing `specify-cli`; `init_project.sh` installs it if
  missing when `uv` itself is already present).
- `terminal` to run the init script and gate scripts.
- Optionally, the Decision HUD MCP tools (`decision_hud__decision_push`,
  `decision_hud__decision_list`) if the project has that plugin installed
  -- routing to Decision HUD degrades gracefully to `clarify` if absent
  (see Procedure step 3).

## How to Run

```bash
bash scripts/init_project.sh <project-name>       # new directory
bash scripts/init_project.sh . --here              # init in cwd
python3 scripts/spec_decision_gate.py --questions
python3 scripts/spec_decision_gate.py --answers answers.json
```

`init_project.sh` is the single entry point: it installs Spec Kit, seeds
worldview/philosophical-preamble/constitution/EARS docs, AND
auto-translates the installed `speckit-*` skills into Hermes frontmatter
-- one call, nothing to remember to chain. Run
`translate_speckit_skills.py <project_dir>` on its own only to re-sync
the translated copies after upgrading `specify-cli` without re-running
the rest of init.

## Quick Reference

| Step | Command | Produces |
|---|---|---|
| 1 | `init_project.sh` | `.specify/`, `.claude/skills/speckit-*` auto-translated into `.hermes/skills/` (Hermes frontmatter), seeded `docs/worldview.md` + `docs/philosophical-preamble.md`, seeded constitution, EARS/contract-testing reference docs -- all in one call |
| 2 | `/speckit-constitution` (in Claude Code) | Refined `.specify/memory/constitution.md` |
| 3 | `/speckit-specify` | `spec.md` with EARS-formatted FR- lines (see `references/ears-syntax.md`) |
| 4 | `/speckit-clarify`, `/speckit-plan`, `/speckit-tasks` | Resolved ambiguities, `plan.md`, `tasks.md` -- every open question routed via `spec_decision_gate.py` first |
| 5 | `/speckit-implement`, `/speckit-converge` | Working code, verified against spec/plan/tasks |
| 6 | `generate_property_tests.py` | Scaffolded property-based tests from agreed EARS-JSON requirements |
| 7 | `generate_mock_server.py` (if a client/server boundary exists) | Running mock server from a shared contract, for client-first contract testing |

## Procedure

1. **Bootstrap.** Run `init_project.sh <project-name>` for a new project,
   or `. --here` inside an existing one -- ONE call does the whole
   setup: installs real Spec Kit (verified working, `specify init
   --integration claude` produces `.claude/skills/speckit-*` +
   `.specify/`), seeds `docs/worldview.md` only if absent (meant to be
   filled in gradually via conversation, not overwritten), seeds
   `docs/philosophical-preamble.md` always if absent, adds both
   `@docs/...` import lines to `CLAUDE.md`, copies pre-seeded
   constitution defaults over the blank template, AND auto-translates
   every installed `speckit-*` skill into Hermes-shaped frontmatter
   under `.hermes/skills/` (mechanical field-rename only -- doesn't
   touch the skill body) so they read as native Hermes skills, not just
   Claude-Code ones. Re-run `translate_speckit_skills.py` alone only to
   re-sync after upgrading `specify-cli` without redoing the rest.
2. **Refine the constitution.** Run `/speckit-constitution`
   conversationally to replace the remaining `[BRACKETED]` placeholders
   in `constitution-defaults.md`'s seed with project-specific
   principles.
3. **Route every open question through the decision gate.** Any
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
4. **Write requirements in EARS syntax.** `spec.md`'s FR- lines follow
   one of the five EARS patterns (`references/ears-syntax.md`). Before
   an EARS line becomes implementation scope, run it through the
   `ears-sensibility-gate` skill if present (checks proportionality,
   actor ownership, testability, conflicts) -- syntactic validity alone
   is not sufficient.
5. **Promote agreed requirements to structured JSON.** Validate against
   `references/ears-schema.json`. This is the layer that seeds
   property-based test generation later, distinct from and
   complementary to the hand-written tests `/speckit-implement` writes.
6. **Scaffold property-based tests.** Once requirements are agreed and
   saved as EARS-JSON, run `generate_property_tests.py <requirements_dir>
   --out <test_file.py>` to produce `hypothesis`-based test scaffolds
   (requires `pip install hypothesis`). These are scaffolds, not
   finished tests -- every `@given(st.nothing())` placeholder must be
   replaced with a real domain strategy before the test means anything;
   never leave a placeholder in and call the requirement covered.
7. **Contract testing (only if this project has a client/server or
   multi-consumer boundary).** See `references/contract-testing.md` for
   the full two-layer approach (shared schema-first contract +
   Pact-style consumer-driven tests) and the client-first build
   ordering. `generate_mock_server.py <contract>` wraps Stoplight Prism
   to mock the contract directly, before any server implementation
   exists. Skip this step entirely for single-process tools/scripts
   with no real consumer boundary.
8. **Run the rest of the Spec Kit pipeline as normal**
   (`/speckit-plan`, `/speckit-checklist`, `/speckit-tasks`,
   `/speckit-analyze`, `/speckit-implement`, `/speckit-converge`),
   applying step 3's gate to every open question along the way.

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
- Don't trust a `generate_property_tests.py` scaffold as a real test --
  every `@given(st.nothing())` placeholder must be replaced with a real
  strategy first; an unfilled scaffold either fails immediately or
  generates meaningless data.
- Don't build the contract-testing layer for a project with no real
  client/server or multi-consumer boundary -- it's a real added cost
  with no payoff for a single-process tool.
- Don't skip the translation step or treat it as optional -- an agent
  that only sees `.claude/skills/speckit-*` and not their Hermes
  translations will miss them entirely in non-Claude-Code sessions;
  `init_project.sh` runs translation automatically so this shouldn't
  come up, but if `translate_speckit_skills.py` is ever invoked
  manually, always run it right after `specify init`, never deferred.

## Verification

- `init_project.sh` produced `.specify/`, `.claude/skills/speckit-*`,
  their Hermes-translated copies under `.hermes/skills/`,
  `docs/worldview.md`, `docs/philosophical-preamble.md`, a seeded (not
  blank) constitution, and `docs/spec-driven/ears-syntax.md` +
  `ears-schema.json` + `contract-testing.md` in the target project.
- Every open question logged during `/speckit-clarify`/`/speckit-plan`/
  `/speckit-tasks` has a recorded `spec_decision_gate.py` verdict
  (resolve_silently + cited evidence, clarify_now + answer, or
  decision_hud + card_type from card-type-gate).
- No EARS line entered `tasks.md` scope without first passing
  `ears-sensibility-gate` (or an equivalent documented check) if that
  skill is present in the project.
- Any `generate_property_tests.py` scaffold committed to the repo has
  every placeholder strategy replaced with real domain data generation
  -- no `st.nothing()` left in a test claimed as passing coverage.
- If the project has a client/server boundary, contract tests exist per
  consumer (not just one shared schema check), and provider verification
  was run against the real server, not only the mock.

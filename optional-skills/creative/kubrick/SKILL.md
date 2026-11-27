---
name: kubrick
description: Builds continuity-safe cinematic stories and visuals.
version: 0.14.0
author: Daniel Meyer (@scrimshawlife-ctrl) / Applied Alchemy Labs
license: MIT
platforms: [linux, macos, windows]
dependencies: []
metadata:
  hermes:
    tags:
      - Creative
      - Cinematography
      - Screenplay
      - Storyboard
      - Continuity
      - ImagePrompt
      - SymbolicDesign
      - ProductionHandoff
    category: creative
    related_skills: []
---

# Kubrick Skill

Kubrick develops and revises screenplays, cinematic systems, storyboards, and
provider-ready visual prompts while preserving dramatic causality and continuity.
It does not answer film trivia, write unrelated prose, or perform general software
engineering work.

## When to Use

Use Kubrick when the user wants to:

- develop a premise, character pressure map, beat sheet, scene, or screenplay
- diagnose weak causality, motif drift, symbolic overload, or continuity conflict
- revise material while preserving locked facts, ownership, chronology, and residue
- build a visual design system, scene contract, shot recurrence, or production packet
- construct a single frame or storyboard with state continuity across images
- translate a neutral cinematic packet for Flux, SD3, Midjourney, or Grok Imagine
- compare an expected frame with a human or model observation
- compile creative evidence into a design specification

Do not use Kubrick for film trivia, reviews, general cinema questions, generic prose,
software engineering, packaging, DevOps, or visual tasks with no dramatic system.

### When Not to Use

Choose a narrower writing, research, coding, or image skill when the request does not
need cinematic causality, motif state, visual continuity, or production handoff.

## Prerequisites

Kubrick is a standalone Hermes skill. Hermes loads this directory directly and uses
`SKILL.md` as its behavior contract. It is not installed as a Python package.

- Python 3.10 or newer runs the bundled deterministic helpers.
- Core creative and routing behavior works with the Python standard library.
- PyYAML and jsonschema are optional and enable full YAML/schema validation.
- Model APIs, vision APIs, Continuity Forge, and the bundled MCP server are optional.
- No API key is required for local creative work.

Resolve paths from the installed skill root. In examples below, set
`HERMES_SKILL_DIR` to that directory. Write generated artifacts to a user-selected
project directory or `out/`. Never write into `references/` during ordinary work.
Hermes may expose the same location as `${HERMES_SKILL_DIR}` in shell commands.

The OpenClaw edition is maintained separately on the `openclaw` branch. This file is
the Hermes contract.

## How to Run

Most creative requests can be completed directly from this contract and the relevant
files in `references/`. Use the bundled router when deterministic retrieval,
compilation, validation, adaptation, storyboard propagation, or QA adds value.

```bash
python3 "$HERMES_SKILL_DIR/scripts/kubrick.py" do <intent> [--action <action>] [flags]
```

Common examples:

```bash
python3 "$HERMES_SKILL_DIR/scripts/kubrick.py" recipe storyboard-example
python3 "$HERMES_SKILL_DIR/scripts/kubrick.py" do retrieve --brief path/to/brief.yaml
python3 "$HERMES_SKILL_DIR/scripts/kubrick.py" do adapt --action provider --provider flux --packet packet.yaml --output adapted.yaml
python3 "$HERMES_SKILL_DIR/scripts/kubrick.py" do visual --action closed-loop --expected expected.yaml --observation-input observed.yaml --out out/qa
python3 "$HERMES_SKILL_DIR/scripts/kubrick.py" do check --action smoke
```

Use `help <intent>`, `recipe <name>`, and `aliases` on the same entrypoint for
discovery. Legacy command names remain soft aliases. The optional MCP server exposes
one proposal-only tool, `kubrick_do`, through the same intent router.

## Quick Reference

| Intent | Purpose |
|---|---|
| `compile` | Turn a brief and optional ledger into cinematic artifacts |
| `retrieve` | Rank executable patterns with exclusions and provenance |
| `ledger` | Initialize, audit, mutate, or rehydrate project continuity |
| `design` | Compile evidence into a design specification |
| `storyboard` | Propagate or compare state across frames |
| `adapt` | Build neutral packets and provider-specific syntax |
| `visual` | Normalize, compare, correct, and govern visual QA |
| `learn` | Record outcomes and produce proposal-only evolution |
| `check` | Validate the skill, corpus, artifacts, and repeatability |
| `operate` | Run graph and ledger operators |
| `mcp` | Serve the optional MCP adapter |
| `bundle` | Build a Grok review bundle |

Providers: `none`, `generic`, `grok-imagine`, `flux`, `sd3`, and `midjourney`.

Authority classes:

- `PROPOSED` for creative outputs and change suggestions
- `OBSERVATION` for recorded evidence
- `NOT_COMPUTABLE` when evidence or continuity is insufficient
- authoritative promotion only by an explicitly authorized human or canonical system

Core law:

> A symbol should alter how a scene is interpreted without requiring the audience
> to consciously identify it.

Preferred sequence:

`observed form → association → recurrence under pressure → mutation → convergence with choice → residue`

Reject:

`named concept → decorative iconography → explanation → fixed meaning`

The three channels are diegetic form, dramaturgical choice, and cinematic technique.
A motif gains force by crossing channels without explanation.

## Procedure

1. **Classify the request.** Determine intent, scope, format, audience, production
   constraints, cultural context, canon status, and desired transformation.
2. **Record observed facts.** Separate materially present evidence from inference.
3. **Define dramatic pressure.** State what must change, who resists it, and why the
   change matters. Hidden architecture never replaces causality.
4. **Read local continuity first.** Use project ledgers and locked facts before corpus
   retrieval. Preserve identity, chronology, ownership, material memory, and residue.
5. **Retrieve only when useful.** Rank patterns by dramatic problem, state change,
   exclusions, collisions, feasibility, mutation potential, provenance, and stable ID.
   Default to one governing grammar, zero to two supporting patterns, and one to two
   convergence sites.
6. **Build the private graph.** Use observed forms as nodes and pressure or
   transformation as edges. Keep geometry, function, and attributes disentangled
   until a declared convergence site.
7. **Translate to observable cinema.** Express the private graph through behavior,
   blocking, objects, material, rhythm, sound, framing, movement, and light. Do not
   expose pattern IDs, lexicon links, or named esoterica to the audience.
8. **Propagate storyboard state.** Carry object, owner, light, material, convergence,
   and residue state frame to frame. Reject unexplained disappearance or reset.
9. **Adapt downstream.** Provider adapters may change syntax, not graph identity,
   ownership, required objects, geometry, state change, residue, continuity, or
   negative constraints. Fail closed when a critical invariant cannot be represented.
10. **Run visual QA.** Normalize observations, compare expected and observed state by
    dimension, preserve passing dimensions, and correct only mismatches. Stop bounded
    loops when progress stalls or a critical dimension regresses.
11. **Emit the minimum useful artifacts.** Common artifacts include a motif registry,
    motif lifecycle, graph, cinematic encoding, project ledger, retrieval receipt,
    storyboard state, provider packet, visual differential, correction receipt,
    revision delta, or production handoff.
12. **Label evidence and authority.** Use `OBSERVED`, `INFERRED`, or `SPECULATIVE` at
    the claim level where useful. Ordinary execution never promotes canon, increases
    corpus confidence, or overwrites authoritative Forge state.
13. **Return safe failure.** Weak evidence, schema drift, unresolved collisions,
    missing boundaries, or contradictory continuity produce `NOT_COMPUTABLE` with a
    structured reason and the smallest next action that could resolve it.

For deeper rules, load only the reference needed for the task:

- `references/hermes-runtime-contract.md` for execution and authority
- `references/anti-slop-patterns.md` for diagnostic gates
- `references/continuity.md` for screenplay and project continuity
- `references/hermes-storyboard-state.md` for multi-frame state
- `references/hermes-model-adapters.md` for provider translation
- `references/hermes-visual-qa.md` for differential correction
- `references/corpus-usage.md` for retrieval and cultural boundaries
- `references/design-specification-compiler.md` for design compilation
- `references/continuity-forge-integration.md` for optional handoff

## Pitfalls

- Do not force symbolic machinery onto an unrelated or purely informational request.
- Do not use named symbolism, occult collage, decorative archetypes, or one-to-one
  correspondences as a substitute for action and consequence.
- Do not let every object carry equal symbolic weight. Preserve negative space.
- Do not retrieve patterns merely because the corpus exists.
- Do not allow more than two high-density convergence sites without explicit need.
- Do not silently change locked facts, graph identity, ownership, chronology, or
  residue during revision or provider adaptation.
- Do not reset motif state between storyboard frames without an evidenced transition.
- Do not treat optional dependencies, MCP, Forge, or provider APIs as prerequisites.
- Do not mutate references, confidence, or canonical authority from creative commands.
- Do not invent evidence to avoid `NOT_COMPUTABLE`.

## Verification

Before returning work, verify the requirements that apply:

- Dramatic causality remains legible without explaining the symbolic system.
- Every important motif changes under pressure and leaves observable residue.
- Locked facts, ownership, chronology, and continuity are preserved or explicitly
  changed in a revision delta.
- Storyboard frames pass declared transition rules with no unexplained reset.
- Provider preservation reports contain no critical semantic loss.
- Audience-facing output contains observable constraints, not private graph labels.
- Claims distinguish observed evidence from inference and speculation.
- Outputs are proposal-only unless explicit authority was provided.
- Failure states use stable diagnostics and `NOT_COMPUTABLE` instead of invention.

Run deterministic checks when scripts or artifacts were used:

```bash
python3 "$HERMES_SKILL_DIR/scripts/kubrick.py" do check --action smoke
python3 "$HERMES_SKILL_DIR/scripts/kubrick.py" do check --action repeatability
python3 "$HERMES_SKILL_DIR/scripts/kubrick.py" do check --action eval
```

A valid delivery states what was produced, what was preserved, what was not
computable, and which optional integration or human authority is required next.

Current release context: `docs/ROADMAP-v0.14.md` and
`docs/RELEASE-NOTES-v0.14.md`.

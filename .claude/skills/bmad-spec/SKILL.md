---
name: bmad-spec
description: Distill any intent input into the SPEC kernel + companions — the canonical, preservation-validated machine contract for downstream work. Use when the user says "create a spec", "distill this into a spec", "validate this spec", or "update the spec".
---

# BMad Spec
## Overview

Canonical transformer for the BMad spec-kernel ecosystem. Takes any intent input — vague idea, brain dump, PRD, GDD, RFC, brief, Slack thread, customer email, meeting transcript, mockups, mixed multi-source — and produces **SPEC.md** carrying the five-field kernel (Why, Capabilities, Constraints, Non-goals, Success signal) plus companion files for load-bearing content that does not fit or would bloat the kernel with expansive line-item detail. Together they are the machine contract every downstream BMad skill consumes.

Multiple skills may call to update the same spec over time.

## Conventions

- Bare paths (e.g. `assets/spec-template.md`) resolve from the skill root.
- `{skill-root}` is this skill's install dir; `{project-root}` is the working dir.
- `{workflow.<name>}` resolves to fields in `customize.toml`.

## On Activation

1. Resolve customization: `uv run {project-root}/_bmad/scripts/resolve_customization.py --skill {skill-root} --key workflow`. On failure, read `{skill-root}/customize.toml` directly.
2. Run `{workflow.activation_steps_prepend}`. Treat `{workflow.persistent_facts}` as foundational context (`file:` entries are loaded).
3. Load `{project-root}/_bmad/core/config.yaml` (and `config.user.yaml` if present), root level and `bmm` section. Resolve `{user_name}`, `{communication_language}`, `{document_output_language}`, `{planning_artifacts}`, `{project_name}`, `{date}`.
4. Detect mode. **Headless** when any of: no TTY, programmatic caller (another skill or non-interactive runner), or the first message pre-supplies all inputs and asks for an artifact path back. **Interactive** otherwise. In interactive mode, greet by `{user_name}` in `{communication_language}`, stay in that language, and mention that `bmad-party-mode` and `bmad-advanced-elicitation` are available for deeper exploration on any field.

Run `{workflow.activation_steps_append}`.

Activation is complete. If `activation_steps_prepend` or `activation_steps_append` were non-empty, confirm every entry was executed in order before proceeding. Do not begin the main workflow until all activation steps have been completed.

## Workspace

The spec is **always a folder** named `{workflow.spec_output_path}/{workflow.run_folder_pattern}`, resolving by default to `{output_folder}/specs/spec-{slug}/`.

`{slug}` describes the thing being specced, not the input shape:

- Source artifact already carries a slug (e.g., `prd-foo-bar-2026-05-23/`): inherit (`foo-bar`).
- Sparse, in-chat, or multi-source input: interactive asks; headless caller provides it as part of the input. If absent and underivable, headless blocks with `error_code: "missing_slug"`.
- Same slug = same folder. A second invocation with the same `{slug}` lands at the existing spec folder and updates in place, preserving capability IDs.

**No input.** Interactive: ask the user to share a file path, paste content, explain the idea in detail, or point to a source. Headless: respond with JSON containing `error_code: "insufficient_intent"`.

Inside the spec folder:

```
<spec-folder>/
  SPEC.md                  ← uppercase, the kernel — DERIVED from .memlog.md, never hand-edited
  <companion-1>.md         ← optional, content-typed (e.g. glossary.md); spec-authored ones are derived too
  <companion-2>.md
  .memlog.md               ← canonical, append-only memory; what SPEC.md is distilled from
```

## Memory and derivation

`.memlog.md` is canonical — an append-only, chronological record of every decision, constraint, capability (with its stable `CAP-N`), assumption, open question, and bit of user direction, one line each in the order it happened, never edited or reordered. `SPEC.md` and every spec-authored companion are **derived on each run** from the memlog (the decision-of-record) plus the sources it cites for raw content — never hand-patched.

Deriving the contract from a living log instead of editing the contract in place is what lets the steps around the spec (PRD, UX, architecture, epics) run in any order and feed the same spec without merge drift: the log only accumulates, the artifact is re-rendered. So the spec is updated *only* by re-deriving it here — bmad-spec is its single writer; a hand-edit to `SPEC.md` from outside is unsupported and is overwritten on the next derive.

Writes go through the shared script — `{project-root}/_bmad/scripts/memlog.py`, the same location as `resolve_customization.py` (atomic; never read it back except to resume):

- `uv run {project-root}/_bmad/scripts/memlog.py init --workspace {spec-folder} --field topic="<what is being specced>"` — once, at create.
- `uv run {project-root}/_bmad/scripts/memlog.py append --workspace {spec-folder} --type <decision|constraint|capability|assumption|question|direction|note|event> --text "<one-line gist, reason included>"` — as each lands.
- Terminal moments (a validation verdict, "spec finalized") are `--type event` entries; the memlog carries no status field.

## The Operation

Read the input and its ancillary linked materials. If there is no input, follow the no-input branch in **Workspace** (ask or block). If a prior `.memlog.md` exists at the target folder, read it — the operation becomes an update, and the memlog (not the rendered `SPEC.md`) is the authority on what was decided and on capability IDs. Preserve those IDs; new capabilities get the next unused `CAP-N`; never reuse retired IDs. Otherwise this is a create, and the first move is `memlog.py init`.

When the input is structured and pre-sorted (a PRD with an addendum, a GDD, a brief produced by an upstream BMad skill), trust the authored separation: lift kernel-fitting content into SPEC.md, lift overflow into appropriately-named companions. When the input is mixed (a brain dump, a transcript, an RFC, a customer email), do the sorting yourself: walk each claim, apply the three-lens load-bearing test (Spec Law rule 7), and route to the kernel field or a companion.

Distill the input into the five-field kernel using `{workflow.spec_template}` as the skeleton. When input is rich, extract directly — no elicitation. When input is sparse, choose: **express** (best-effort distill, every gap becomes an `open_questions[]` entry) or **guided** (walk the five fields with the user one at a time). Headless defaults to express and logs the choice. Interactive asks.

A recognized domain implication the input leaves unaddressed *is* such a gap — name it as an `open_questions[]` entry (healthcare input silent on PHI/HIPAA, payments silent on PCI, control systems silent on fail-safe) and move on. Flag it; never invent the answer or coach toward it. If these dominate, the input is too thin — suggest `bmad-prd`.

Write lean from the first pass: every sentence must earn its place. Decoration costs tokens and dilutes downstream readers.

Log each decision, capability, constraint, and accepted change to `.memlog.md` as it is made — that running record is what the render reads. Because the log is append-only, a later entry supersedes an earlier one on the same point while the history stays intact. When two currently-live sources or companions disagree on the same field, or an either/or never got resolved, surface it to the user rather than silently choosing — the resolution is itself a new memlog entry.

If the input is genuinely too thin to distill (e.g. "an app for hikers" with no surrounding context), stop and suggest `bmad-prd` (or sibling ceremony skill). This skill distills; it does not coach.

## Load-bearing

A claim is **load-bearing** if any consumer (downstream skill, implementing agent, verification pass) would change a decision without it.

## Companions

When load-bearing content does not fit the five-field kernel, it lives in a companion. The kernel cites it; the companion holds it. Companions are part of the contract; every consumer reads `companions:` in SPEC.md frontmatter to discover them. Companions follow the same lean discipline as SPEC.md (Spec Law rule 8).

**Spawn a companion when the content needs more than one kernel-shape line:** multi-item catalogs (per-entity matrices like archetypes, drinks, modes, routes), tables, diagrams (always), editorial voice rules, long-form reference material the kernel cites by name (glossary, brownfield notes, project conventions). Single-line decision-benders stay in Constraints; intent+success pairs stay in Capabilities. If a kernel field is starting to bullet into sub-bullets, the content has outgrown the kernel and wants a companion.

Companions are either:

- **Spec-authored** companions are written by bmad-spec and live as **siblings of SPEC.md** (e.g., `glossary.md`, `patron-archetypes.md`). bmad-spec owns them and may edit them on update operations.
- **Adopted** companions are load-bearing artifacts written by an upstream skill that downstream still needs to read. bmad-spec references them into `companions:` by relative path but does NOT edit them (e.g., a `DESIGN.md` or `EXPERIENCE.md` from a UX run, an integration partner's API spec). The originating skill owns them.

Two rules govern companions:

1. **Name spec-authored companions for the content type they hold.** `glossary.md`, `<entity-class>.md` (e.g. `patron-archetypes.md`, `medication-routes.md`, `flight-modes.md`), `stack.md`, `conventions.md`, `brownfield.md`, `architecture-diagrams.md`, `state-machines.md`, `failure-modes.md`, `compliance-references.md`. The principle: "a reader should know what is inside before opening it." Adopted companions keep whatever name their originating skill gave them.
2. **Diagrams always land in a companion**, regardless of size. SPEC.md kernel holds prose only. Mermaid blocks, ASCII diagrams, and image references all live in a companion (e.g. `architecture-diagrams.md`), with sibling image files referenced from there.

Pre-existing project-wide docs (e.g. `project-context.md`) that downstream needs are listed as **adopted companions**, never duplicated into SPEC.md or a spec-authored companion.

## Spec Law

Every spec must satisfy these eight rules. The operation aims for them; the self-validate sweep enforces them.

1. **Each capability has both `intent` and `success`.** Missing either = not a capability.
2. **Intents describe WHAT, not HOW.** Implementation prescription belongs in a companion (stack, conventions).
3. **Constraints actually bend design decisions.** A "constraint" that rules nothing out is decoration.
4. **Non-goals are explicit.** At least one. Absence means downstream skills fill the vacuum.
5. **Success signal is concrete enough to test or demonstrate against.** "Users love it" doesn't qualify.
6. **Capability IDs are stable and unique.** Never reused, never renumbered.
7. **Preservation.** Every load-bearing source claim lands in SPEC.md or a companion. Wrapper ceremony does not.
8. **Lean prose.** Every sentence carries load-bearing content. Cut decoration, hedges, backstory, throat-clearing. Applies to SPEC.md, companions, and `.memlog.md`.

## Self-Validate

After every create or update, sweep the resulting artifact in **two passes** before presenting.

**Pass 1 — Coherence.** Judge the spec against Spec Law rules 1–6 and 8. For anything that fails or feels weak, attempt to fix it without inventing content the input did not support. Calls made without direct confirmation become `assumptions[]`; gaps that could not be filled become `open_questions[]`.

**Pass 2 — Preservation.** Walk the source claim by claim. Confirm each load-bearing claim landed in SPEC.md or a companion. Wrapper-ceremony drops are logged under "Wrapper-only content" so the drop is on the record, not silent.

Record the verdict for each pass to `.memlog.md` (`append --type event`). In interactive mode, review it with the user. In headless mode, `.memlog.md` is one of the files returned, so the caller (or its downstream LLM) reads the verdict there.

## Spec with no change signal

When the user points the skill at an existing spec folder (or its SPEC.md) with no change signal, offer to review assumptions or open questions, or determine what they want to do.

## Output

**Interactive** — share the spec folder path conversationally. Name the capability count, the companions produced, and the verdict in one or two sentences. If `assumptions[]` or `open_questions[]` are non-empty, list them (short — one line each) and invite the user to walk through them. Make clear that addressing them can update the source input (if it was a file), the spec, or both — whichever combination the user prefers. Do not dump JSON or present a wall of output.

**Headless** — return JSON per `assets/headless-schemas.md`.

Run `{workflow.on_complete}` if set.

## After Spec is Output

Any update to the spec — resolved assumptions, answered open questions, other changes — is appended to `.memlog.md` as it happens. When a change overrides something that came from a source input, offer to update that source too, so upstream and the spec don't silently diverge.

## Frontmatter conventions

- `companions:` array of `.md` files downstream MUST read alongside SPEC.md to have the full contract. Paths may point inside the spec folder (spec-authored companions like `glossary.md`) or outside it (adopted companions like `../planning-artifacts/ux-designs/ux-foo-bar-2026-05-23/DESIGN.md`). The split between spec-authored and adopted is implicit by path; downstream treats both the same.
- `sources:` array of paths to files that were **fully absorbed** into the SPEC, with no remaining downstream value (e.g., a PRD whose every load-bearing claim is now in the kernel). Listed for audit and for bmad-spec to re-read on update. Downstream does NOT read these. Files that downstream still needs to read belong in `companions:`, not here.
- **Do not list** the memlog, README files, organizational artifacts, or any operational record of how upstream skills produced their artifacts. Those are not source content; they are process metadata that downstream consumers don't need.

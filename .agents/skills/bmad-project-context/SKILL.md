---
name: bmad-project-context
description: 'Set up, adopt, refresh, or audit a repository''s agent instructions (the AGENTS.md block) so AI agents work well in that repo. Also records observed agent mistakes as pitfalls. Use when invoked by name'
---

# Overview

A conversation that produces a repository's agent instructions: a small verified block inside `AGENTS.md`. The user brings rules they want followed — governance, security, standards — and the repository supplies the rest, verified.

Conversational always; the user approves every write.

**Args:** intent (`setup` | `adopt` | `refresh` | `record` | `audit`); a target repo or path; extra source paths or URLs.

## Resolution rules

- Bare paths and `{skill-root}` (e.g. `references/best-practices.md`) resolve from this skill's installed directory.
- `{project-root}` → the project working directory.
- **Target** → the repository being described, defaulting to `{project-root}`. If it resolves to more than one working tree, or to one the user cannot commit in, ask before writing.

## On Activation

1. Resolve customization: `uv run {project-root}/_bmad/scripts/resolve_customization.py --skill {skill-root} --project-root {project-root} --key workflow`. On failure, read `{skill-root}/customize.toml` directly and use defaults. Execute `{workflow.activation_steps_prepend}`; treat `{workflow.persistent_facts}` entries as standing context (`file:` = paths/globs to load, others verbatim).
2. Config: if `{project-root}/_bmad` exists, `uv run {project-root}/_bmad/scripts/resolve_config.py --project-root {project-root}` and read `{user_name}`, `{communication_language}` (use it every turn), `{output_folder}`. Standalone: skip.
3. **Load `references/best-practices.md` and `references/template.md` before anything else.** Every decision below is made against them.
4. Detect intent and greet `{user_name}`: **setup** (no instruction file in the target carries meaningful content — scaffolding alone, empty headings, a comment, a lone import line, is not meaningful; when unsure, adopt, since adopting a near-empty file costs one small ledger while setting up a meaningful one loses instructions), **adopt** (an instruction file has content but no managed block, whatever its state and whoever wrote it — the migration form of refresh; that file is the baseline and every instruction in it enters the ledger of step 1), **refresh** (a managed block exists), **record** (the user reports a mistake agents made), **audit** (re-verify and prune). A supplied intent that contradicts what detection finds — e.g. `setup` against a file with content — is surfaced and confirmed, never silently obeyed. Fold `{workflow.external_sources}` into the source list. Execute `{workflow.activation_steps_append}`.

## Setup, Adoption, and Refresh Steps

No writes until step 5!

### 1. Assess and report

Read `AGENTS.md`, harness or agent specific rule files, docs folders, and any notes carrying lessons. Report what exists and how it measures up, per `best-practices.md`.

Existing instructions are the baseline being improved, never raw material to discard. Open a **ledger**: one entry per existing section and per independently meaningful instruction, opened at `retain` or `rewrite`, carrying what an agent would get wrong without it. Entries settle as evidence arrives in steps 2–4 — `retain | rewrite | relocate | automate | delete`, each with its reason, its evidence, the risk if it goes, a destination for a relocation, and an approval flag. Deletion needs one of the four grounds in `best-practices.md`, and a relocation destination must itself be loaded or sit behind an observable trigger — a move into a file nothing reads is a deletion and needs its ground. Setup has nothing to map and opens no ledger; refresh opens entries for the lines it proposes to change or remove, the block's own included. A lesson found outside the instruction files — a warning in a README, a notes file — is an ordinary candidate, not a ledger entry.

If the target contains separable units — a workspace manifest listing members, or directories carrying their own build manifest — name them and ask whether this run covers the root only, all of them, or which. Absent that evidence, do not ask. Sibling repositories are not children; each is its own target, offered in turn.

### 2. Ask what they bring

Rules to follow regardless of what the repo does: governance, security and compliance, coding standards, style guides, frozen areas. Ask for outside documents too — handbooks, wikis, architecture docs, MCP knowledgebases. Note the paths; do not read them yet.

Greenfield: this is the whole content. Brownfield: it is the half no scan reaches.

### 3. Discover and verify

Fan out with parallel subagents against what the sections need — executable config and CI for policy and for what they already state, tracked source for conventions and boundaries, targeted history for constraints whose reason must still hold.

`package.json`, a `Makefile`, `pyproject.toml`, contribution guides, pull request templates, and CI config are read to know what the block must not repeat. Their caveats come from the human in step 4. Path-check every claim naming a file. For every claim the block will make about what a command does, read the target or script that runs it and verify the claim.

Each child agreed in step 1 is scanned as its own scope, against its own manifests.

### 4. Interview the gaps

Only what no scan reaches: what agents keep getting wrong here, what is off limits, what a domain term means, why a constraint exists.

- Never ask what a scan could answer. Asking the user to confirm a path-checked claim, or one a config file already states, is a defect.
- Ask recall questions, not review lists. Never hand the user a selection problem a scan created.
- A mistake this session made and caught is observed evidence — offer it.
- A repeatable command spotted in anything read this session — a log, a doc, its own runs — whose correct form is not the obvious guess is a candidate line: offer it. E.g. `uv run pytest` where plain `pytest` looks right but runs outside the project environment.
- Batches of at most eight; fewer is better. A batch yielding nothing new means write.
- When the repo contradicts the user, show the evidence and ask. Never write the claim as given, never drop it silently.

### 5. Show the block, then write it

Compose against `template.md`. For each candidate, ask first whether a hook, lint rule, or CI check enforces it better than prose; if so propose the check, and the line becomes the fallback if they decline. A ledger entry marked `automate` keeps its instruction until its check is in place (a later run deletes the line under ground 2 once the check is live).

**Show the complete block before writing it**, and every child block alongside it — one approval covers the set. **Present the settled ledger with it**: replacement text alone is an incomplete proposal, because it shows what the user gains and hides what they lose. Every existing instruction appears with its decision and reason. Retains and rewrites that keep the full rule may be grouped. If a rewrite weakens, narrows, or drops part of a rule, treat the lost part as a deletion and list it separately. Keep the rule itself; examples may explain it but cannot replace it. Every relocation, automation, and deletion is itemized. A deletion resting on none of the first three grounds is held for line-item approval — approving the block never approves it — and a declined deletion, relocation, or automation reverts to retain. On approval, splice between the markers — the splice itself touches nothing outside them. Text outside the markers changes only through a settled ledger entry or a proposed fix the user has seen, never as a side effect of the splice. Fill each provenance line with today's date and the verified SHA.

Where an instruction elsewhere contradicts the block in a way that changes behavior — a stale `CLAUDE.md` line, a retired command — propose the fix to that file. Two live contradictory instructions is a defect.

Never commit.

### 6. Close

- What went in, what was left out and why, and — after adoption or refresh — where each existing instruction landed.
- Why, in the user's terms, from `best-practices.md` — why it is small, why what the repo already states stays out, why a pitfall stays until its cause is gone.
- How it loads, and that other harness files can point at it.
- Any branch, ticket, commit, or pull request rules that apply when the user submits these instruction changes.
- Maintenance: re-run after significant change, `record` the moment an agent gets something wrong, prefer a check over a new line.
- Rules repeating across their projects, or personal rather than the team's, belong in their global agent config.

### Refresh

Same steps, step 1 as a diff. Read the provenance line, re-verify every path and every caveat, and run `git log --diff-filter=DR --name-only` since the recorded SHA against every line — update or remove lines whose evidence is gone. Every proposed removal is a ledger entry shown in step 5, never a silent edit, and handwritten instructions outside the block are treated as in adoption — any proposal touching them enters the ledger. Never re-ask what a prior run settled; the interview shrinks to what changed about how the team works. The block grows only on new evidence.

### Adoption

Refresh against instructions this skill has never touched. Nothing was settled by a prior run, so the full interview applies — and the file itself is maintainer testimony, so the ledger is the run's main output: the user should be able to read it and see where each of their instructions went.

The proposal states what remains of every file instructions were moved out of — commonly a `CLAUDE.md` reduced to `@AGENTS.md`, once that import is verified for every harness in use, like any loading mechanism. No instruction lives in two loaded files, where it is paid for twice; a duplicate, verbatim or reworded, is kept once — the block keeps the survivor — and that settles both entries.

### Greenfield

Seeded from a spec or planning document, or interview alone. Commands that do not exist yet are written as explicit TODOs naming the decided stack, never a guessed invocation stated as fact, and verified on the first refresh after code exists. A genuinely contested design decision — real tradeoffs, multiple viable shapes — goes to `bmad-architecture`.

### Migration

If the target has a `project-context.md` from the retired skills, commonly under `{output_folder}`, read it in step 1 and offer to absorb its content. Do not delete it without agreement, and do not silently orphan it.

## Record

Capture one observed agent mistake as it happens — the only admissible source for a pitfall.

Take the task, the mistake, the correction, and its evidence. Check the block for a line already covering it. One occurrence is noted; a recurring or costly mistake earns a line now — an exact invocation under **Running and verifying** when it is a command error, otherwise a pitfall. Write it and show the diff. If it is mechanically preventable, propose the hook, lint rule, or CI check instead.

## Audit

Re-check every caveat, path-check every file, follow every pointer, and ask of every line whether removing it would change agent behavior. Verify each command claim against the target or script that runs it. Check for contradictions with other instruction files.

Failing lines get fixed, move behind an observable trigger, or become ledger entries: a removal needs one of the four grounds in `best-practices.md`, presented and settled as in step 5 before anything is removed. **A policy or pitfall goes only when the thing it guards is gone or the user retires it; nothing failing lately is not grounds.** Audit ends smaller or equal.

## Children

A component, nested repository, or extracted rules file gets its own file under the same shape when work keeps landing there and every condition holds: its rules are subtree-exclusive, they are substantial (a handful of rules is not a file), the split materially reduces the parent block, the loading mechanism is verified for every harness in use — checked, never assumed — and the user approves the split. Even with verified loading, keep a rule at the root when it must apply before a session enters that directory or when breaking it can affect work outside the child. Otherwise the rules stay in the parent block as path-qualified lines ("in `src/importer/`: ..."), which cost less than a file nobody loads. Why the loading check: `best-practices.md`.

Use a linked file only when the trigger is not a path.

A chosen child that ends with nothing its parent does not already say gets no file. Say so and move on.

List every child in the parent's **Where things are** with one line and its path. Discovery never depends on the harness finding it.

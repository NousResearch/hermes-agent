---
name: bmad-retrospective
description: 'Review a completed epic against the evidence it left behind — spec, stories, diffs, commits, sprint status — and produce a retrospective with sourced findings, action items, and an acceptance decision. Use when the user says "run a retrospective" or "lets retro the epic [epic]". Supports -H/--headless'
---

# Retrospective

Review a completed epic by reading the evidence it left — the epic spec, story files, the full diff, per-story commits, sprint status, and session logs when they exist. An unattended epic run leaves a record; this skill reads that record, surfaces the defects no single story could show, and judges the epic against the criteria it set for itself.

Every finding you report carries a source reference (file, line, commit, or log). A claim you cannot point at — an invented root cause, a pattern the diff does not actually show — is not a finding. Drop it.

## Resolution rules

- Bare paths and `{skill-root}` (e.g. `references/aggregate-views.md`, `scripts/sprint_status.py`) resolve from this skill's installed directory.
- `{project-root}` → the project working directory.
- `{skill-name}` → the skill directory's basename.

## Modes

Interactive by default. With `-H`/`--headless`: skip every confirmation, take the epic from the invocation (falling back to detection only if none was supplied), never open the team discussion, render the verdict on the evidence alone, and record each assumption made without the user (which epic was selected, the machine verdict, each proposed item) into the retrospective document's Assumptions section so the audit trail survives. The Phase 4 acceptance fail-safe still applies in headless runs.

For automation, `-H <epic>` — an explicit epic in headless mode — is the stable orchestrator-facing interface. Pass the same number to `detect-epic --epic <N>` so the unfinished-story gate is script-backed (see Inputs). Epic auto-detection is a human convenience, not an automation contract: unflagged `detect-epic` returns the highest epic with *any* `done` story, and stories-mode projects have no `sprint-status.yaml` to detect from.

## On Activation

Run these in order before the retrospective begins:

1. **Resolve the workflow block.** Run `uv run --no-cache {project-root}/_bmad/scripts/resolve_customization.py --skill {skill-root} --project-root {project-root} --key workflow`. If it fails, resolve `{workflow.*}` yourself by reading `{skill-root}/customize.toml`, then `{project-root}/_bmad/custom/{skill-name}.toml`, then `.user.toml` in that order, merging base → team → user (scalars override, keyed arrays-of-tables merge by `code`/`id`, other arrays append).
2. **Run prepend steps** — execute each entry in `{workflow.activation_steps_prepend}` in order.
3. **Load persistent facts** — treat every `{workflow.persistent_facts}` entry as standing context. `file:` entries are paths/globs under `{project-root}` whose contents load as facts; all others are literal facts.
4. **Load config** with `uv run {project-root}/_bmad/scripts/resolve_config.py --project-root {project-root}`: `project_name`, `user_name`, `communication_language`, `document_output_language`, `output_folder`, `user_skill_level`, `planning_artifacts`, `implementation_artifacts`, and `date` (system datetime). Speak all output in `{communication_language}`; write all documents in `{document_output_language}`. Never state time estimates — AI has changed development speed, so hour/day/week predictions are noise.
5. **Greet and orient** (interactive only). Greet `{user_name}`, name the epic you are about to retro, and optionally invite their going-in concerns ("anything you want weighted — a story that felt rushed, a risky interaction between two stories?"). Use any answer to focus the Phase 1–2 analysis; it directs attention but never becomes a finding without a source.
6. **Run append steps** — execute each entry in `{workflow.activation_steps_append}` in order.

## Inputs

| Input | Where | Use |
|-------|-------|-----|
| epic | invocation argument, or detected from sprint status | which epic to retro |
| spec folder | invocation argument, or found under the spec roots | the stories-mode epic: `SPEC.md`, ordered `stories.yaml`, `stories/<id>-*.md` |
| sprint status | `{implementation_artifacts}/sprint-status.yaml` | epic detection + final status update |
| architecture / prd | `{planning_artifacts}/*architecture*`, `*prd*` | context for judging as-built vs intended |
| previous retro (optional) | `{implementation_artifacts}/**/epic-{{prev}}-retro-*.md` | check whether last epic's actions landed |
| session logs (optional) | conversation/session records for the epic's stories | process lessons; record the gap when absent |

An epic reaches this skill in one of two shapes, and they are peers. **Sprint mode** reads `sprint-status.yaml`. **Stories mode** reads a spec folder holding `SPEC.md`, an ordered `stories.yaml`, and `stories/<id>-*.md` artifacts — the shape an unattended run leaves behind. Resolve which applies first: a named folder is stories mode whether or not sprint status exists; a named epic number is sprint mode; with neither, use sprint mode when `sprint-status.yaml` exists, and otherwise look for spec folders under `{output_folder}/specs`, `{planning_artifacts}`, and `{implementation_artifacts}`. Ask the user which to retro when there is more than one, and never choose silently; headless, stop and require an explicit folder.

In stories mode, `stories.yaml` in list order is the story list — list order is authoritative, filename sort is not — and each story's `stories/<id>-*.md` frontmatter carries its `status`. `pending_stories` is the ids whose status is not `done`; apply the same completeness gate as below. Then skip to Phase 1: do not read or write sprint status for the rest of the run. The rest of this section is sprint mode.

Determine the epic and its unfinished-story list from `sprint_status.py detect-epic` whenever `{implementation_artifacts}/sprint-status.yaml` is available:

- **Epic supplied** (including the stable `-H <epic>` orchestrator path): run `uv run --no-cache {skill-root}/scripts/sprint_status.py detect-epic --file {implementation_artifacts}/sprint-status.yaml --epic <N>`. The script scopes `pending_stories` to that number even when auto-detect would have picked a different epic, and even when the epic has no `done` story yet. `story_count` is that same scoped count of the epic's story keys: `0` means the file has no such epic at all — a nonexistent epic returns the same empty `pending_stories` as a finished one, so treat `story_count: 0` as a likely mistyped epic number, confirm with the user, and headless, stop and report rather than proceeding.
- **No epic supplied**: run the same command without `--epic` (returns the highest epic with a `done` story). Confirm the detected epic with the user and let them override; in headless mode accept it and record the assumption. If detection returns none, ask the user — or, headless, stop and report.

If the script exits non-zero it emits `{"ok": false, "error": ...}` instead of a detection — the normal path for a stories-mode project with no `sprint-status.yaml`, and for a file that does not parse: surface that error verbatim — or, if the script produced no JSON at all, whatever it wrote to stderr — and ask the user which epic to retro; headless, stop and report. Without a readable sprint-status file there is no `pending_stories` list; record that the completeness check did not run and continue only if the user (or headless Assumptions trail) accepts proceeding without it.

Then check the epic is actually finished before Phase 1. A successful detect carries `pending_stories` — the selected epic's story keys whose status is not `done`, in file order, scoped to that epic alone (an unfinished story in some *other* epic is out of scope for this retrospective). When the list is non-empty, interactively list those stories and ask whether to retro an unfinished epic: if the user declines, stop and report — do not enter Phase 1; if they accept, record the stories they accepted proceeding over in the document's Epic summary. Headless, proceed and record the same list in the Assumptions section — do not invent a confirmation. Either way the list sits in the document, and Phase 4's machine verdict is **rejected** when any story remained unfinished (see `references/acceptance-verdict.md`); a human may override interactively.

## Working state and resumption

The retrospective document is the working artifact, not only the final output. Once the epic is fixed, create it as a skeleton (`references/retro-document.md` names the sections) and write each phase's result into it as you finish — inventory, then findings with sources, then dispositions and verdict. Continuity is re-reading the file.

If a retrospective document for this epic already exists, load it, reconcile its recorded state against the current evidence — the current evidence wins, since commits may have landed and questions may have been answered since — and resume at the first incomplete phase instead of redoing finished ones. In stories mode that document is `{spec-folder}/RETROSPECTIVE.md`, a fixed name so a resumed run finds it; sprint mode keeps its dated `{implementation_artifacts}` filename.

## Flow

Run the phases in order. A default run stops at a written evidence report and verdict; the team discussion in Phase 3 is opt-in.

### Phase 1 — Gather

Enumerate what the epic actually produced and record what is missing. Load `references/evidence-gathering.md` for the inventory checklist, the `git_evidence.py` pre-pass that derives the diff range and per-story commits, and the missing-evidence rule: each later analysis declares what it needs and records a narrowed scope when the evidence is absent, so a reader can always tell "checked and clean" from "never checked."

### Phase 2 — Analyze

Produce findings, each with a source reference, from three angles:

- **Aggregate views** — the defects no single diff hunk shows: architecture delta, duplication map, god-class growth, pattern divergence, spec-to-implementation reconciliation. Load `references/aggregate-views.md` for the catalog and how to derive each (deterministic scripts first).
- **Diff-scope review** — do not reimplement review. Invoke **`bmad-review`** on the epic's diff for the code lenses (adversarial, edge-case, verification-gap), weighting the boundaries between stories, where no single session ever saw both sides. Fold its findings in. If `bmad-review` is unavailable, run those lenses inline over the diff on a narrowed scope and record the narrowing.
- **Behavior check (when the epic changed runtime behavior)** — exercise the changed flows end to end and record what you observed. Passing tests do not substitute for running the system.

Consolidate: merge, dedupe, and provenance-link findings. Drop any finding you cannot tie to a source.

### Phase 3 — Team Discussion (opt-in)

Skip by default; never runs headless. When the user asks to "discuss it as a team," "run party mode," or similar, invoke the skill `bmad-party-mode` seeded with the Phase 2 findings so the installed agents react to real evidence — the god class the diff really grew, the verification gap that is actually there, the wins the evidence confirms. Load `references/team-discussion.md` for how to seed it and keep it grounded. If `bmad-party-mode` is unavailable, run the discussion inline over the Phase 2 findings and record the narrowing. The rule: agents speak only to findings with sources.

### Phase 4 — Decide

- **Action items** — compile fix-now findings and process lessons into specific, owned action items. Fixes and spec reconciliations are *proposed here*, not auto-applied; the human decides what to execute.
- **Acceptance verdict** — judge the final state against the epic's declared acceptance criteria (profile it from the diff and stories if none were declared): **accepted**, **accepted-with-open-items**, or **rejected** — one spelling, everywhere a machine reads it. Unfinished stories in `pending_stories` force the machine verdict to **rejected**. A human decision always overrides. An epic that fails its criteria with no human decision is recorded as *not accepted* — never as silently accepted. Load `references/acceptance-verdict.md` for the rubric, the finding-routing dispositions, and the previous-retro follow-through record — the per-item evidence Phase 5's status offer reads.

### Phase 5 — Finalize

Finalize the retrospective document and update sprint status. Load `references/retro-document.md` for the document's sections and the exact `sprint_status.py update` invocation that marks the retro key `done`, appends the action items, and validates the write. Where the Phase 4 follow-through has evidence a *previous* epic's action item landed, offer `--set-action-status` and pass only the transitions the user confirms — the evidence justifies proposing a transition, and only the user's confirmation justifies writing it; a headless run records the transitions it would have proposed and does not pass the flag at all. In stories mode, finalize `{spec-folder}/RETROSPECTIVE.md` and stop there: no `sprint_status.py` call, no sprint-status file created, and no edits to `SPEC.md`, `stories.yaml`, or any story artifact. Then, if `{workflow.on_complete}` is non-empty, follow it as the final instruction.

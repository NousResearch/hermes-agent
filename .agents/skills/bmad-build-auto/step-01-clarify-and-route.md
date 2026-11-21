---
spec_file: '' # set at runtime once a route resolves it; some HALT branches exit before it is set
spec_folder: '' # set at runtime under folder+id dispatch only
story_id: '' # set at runtime under folder+id dispatch only
followup_pass: '' # set at runtime when a `done` spec is re-dispatched for a follow-up review pass; empty on a first pass
---

# Step 1: Clarify and Route

## RULES

- **Language** — Speak in `{{.communication_language}}`, tailored to `{{.user_skill_level}}`. Write files in `{{.document_output_language}}`.
- Treat the invocation intent as workflow input, not as a substitute for step-02 investigation and spec generation.
- **EARLY EXIT** means: stop this step immediately, then read and follow the target file. Return here only if a later step explicitly says to loop back.

## Intent check (do this first)

Use the invocation prompt as the intent.

If the invocation prompt explicitly points to an existing spec file with recognized `status` frontmatter, set `spec_file`, then **EARLY EXIT** to the appropriate step:
- `draft` → `[[bmad-snapshot:step-02-plan.md]]`
- `ready-for-dev` or `in-progress` → `[[bmad-snapshot:step-03-implement.md]]`
- `in-review` → `[[bmad-snapshot:step-04-review.md]]`
- `blocked` → HALT with status `blocked` and blocking condition `blocked spec supplied`.
- `done` → set `review_loop_iteration` to `0` in the frontmatter and set `followup_pass` to `true`, then **EARLY EXIT** to `[[bmad-snapshot:step-04-review.md]]` for a fresh review pass. (A `done` spec is a completed run, so this starts a follow-up review, not a resumption.)

If the invocation prompt instead supplies a spec folder and a story id, with no specific spec file path, this is a **folder+id dispatch**: set `spec_folder` (a `{project-root}`-relative or absolute path) and `story_id` from the prompt. Any further prompt text (e.g. `invoke_dev_with` guidance the caller appended) is additional planning context to carry into step-02 — not a competing description of what to implement.

Read `{spec_folder}/stories.yaml`. If the file does not exist or fails to parse, HALT with status `blocked` and blocking condition `no stories.yaml found`. Find the entry whose `id` equals `{story_id}`; if none matches, HALT with status `blocked` and blocking condition `story id not found in stories.yaml`. Take only that entry's `title` and `description` — never read the checkpoint fields or `invoke_dev_with`; those are the caller's orchestration fields, not build-auto's.

Look for files matching `{spec_folder}/stories/{story_id}-*.md` (id-prefix match — story ids are prefix-free, so at most one should match):
- **If more than one matches**, HALT with status `blocked` and blocking condition `ambiguous story file match`.
- **If exactly one matches**, set `spec_file` to that path.
  - `draft` (planning was interrupted mid-flight): accumulate cross-story context before resuming — load every other file matching `{spec_folder}/stories/*.md` (every match except `{spec_file}` itself), regardless of `status`, and carry forward each one's **Code Map**, **Design Notes**, **Spec Change Log**, **Tasks & Acceptance** checklist state, and **Auto Run Result** details, where present, as additional planning context for step-02. Then **EARLY EXIT** to `[[bmad-snapshot:step-02-plan.md]]`.
  - Any other recognized `status`: **EARLY EXIT** using the same routing as above, including the `review_loop_iteration` reset and `followup_pass` for `done`. One difference: a `blocked` story HALTs with blocking condition `story already blocked`, not `blocked spec supplied` — the caller did not supply this file; build-auto found it by id.
  - `status` missing or unrecognized: HALT with status `blocked` and blocking condition `unrecognized status in existing story file`.
- **If none matches**, this is the first dispatch for `{story_id}`. The entry's `title` and `description` are the resolved intent. If `{spec_folder}/SPEC.md` does not exist, HALT with status `blocked` and blocking condition `no epic spec found`. Otherwise load it and the files listed in its `companions:` frontmatter as planning context, then accumulate cross-story context the same way as the `draft` case above — load every file matching `{spec_folder}/stories/*.md` (none yet exists for `{story_id}` at this point, so nothing is excluded), regardless of `status`, carrying forward the same fields, where present, as additional planning context for step-02. Then continue to INSTRUCTIONS item 3 below — not `step-03-implement.md`, item 3 of the numbered list in this file (items 1 and 2 do not apply — context and intent are already resolved; item 1.A.5's previous-story continuity scan in particular never runs here, since folder+id dispatch already skips items 1 and 2 entirely — the cross-story accumulation above is its replacement for this dispatch mode).

One `stories.yaml` entry per invocation: never read another entry, and never advance to a different story id regardless of outcome.

Otherwise, treat the invocation prompt as starting intent. This may be a story ID, ticket ID, file path, short description, or longer free-form intent. Do not infer workflow state from non-spec files.
If the invocation prompt does not contain enough intent to identify what to implement, HALT with status `blocked` and blocking condition `unclear intent`.

## INSTRUCTIONS

1. Load context.
   - List files in `{{.planning_artifacts}}` and `{{.implementation_artifacts}}`.
   - If the invocation prompt points to an unformatted spec or intent file, ingest that file. Do not scan for unrelated intent files.
   - **Determine context strategy.** Using the intent and the artifact listing, infer whether the current work is a story from an epic. Do not rely on filename patterns or regex — reason about the intent, the listing, and any epics file content together.

     **A) Epic story path** — if the intent is clearly an epic story:

     1. Identify the epic number `{epic_num}` and (if present) the story number `{story_num}`. If you can't identify an epic number, use path B.

     2. **Check for a valid cached epic context.** Look for `{{.implementation_artifacts}}/epic-<N>-context.md` (where `<N>` is the epic number). A file is **valid** when it exists, is non-empty, starts with `# Epic <N> Context:` (with the correct epic number), and no file in `{{.planning_artifacts}}` is newer.
        - **If valid:** load it as the primary planning context. Do not load raw planning docs (PRD, architecture, UX, etc.).
        - **If missing, empty, or invalid:** compile it in the next bullet.

     3. **Compile epic context if needed.** If no valid cached epic context was loaded, produce `{{.implementation_artifacts}}/epic-<N>-context.md` by spawning a subagent synchronously with `[[bmad-snapshot:compile-epic-context.md]]` as its prompt. Pass it the epic number, epics file path, `{{.planning_artifacts}}`, and output path `{{.implementation_artifacts}}/epic-<N>-context.md`.

     4. **Verify if compiled.** If epic context was compiled, verify the output file exists, is non-empty, and starts with `# Epic <N> Context:`. If valid, load it. If verification fails, HALT with status `blocked` and blocking condition `context compilation verification failed`.

     5. **Previous story continuity.** Regardless of which context source succeeded above, scan `{{.implementation_artifacts}}` for specs from the same epic with `status: done` and a lower story number. Load the most recent one (highest story number below current). Extract its **Code Map**, **Design Notes**, **Spec Change Log**, and **task list** as continuity context for step-02 planning. If no `done` spec is found but an `in-review` spec exists for the same epic with a lower story number, HALT with status `blocked` and blocking condition `missing previous-story continuity decision`.

     **B) Freeform path** — if the intent is not an epic story:
     - Planning artifacts are the output of BMAD phases 1-3. Typical files include:
       - **PRD** (`*prd*`) — product requirements and success criteria
       - **Architecture** (`*architecture*`) — technical design decisions and constraints
       - **UX/Design** (`*ux*`) — user experience and interaction design
       - **Epics** (`*epic*`) — feature breakdown into implementable stories
       - **Product Brief** (`*brief*`) — project vision and scope
     - Scan the listing for files matching these patterns. If any look relevant to the current intent, load them selectively — you don't need all of them, but you need the right constraints and requirements rather than guessing from code alone.
2. Resolve intent from the invocation prompt and loaded artifacts. Do not fantasize or leave open questions. If the intent cannot be resolved, HALT with status `blocked` and the unresolved questions as blocking condition.
3. Version control sanity check. If version control is unavailable, skip this check. Otherwise require a clean working tree, a branch that fits the intent, and writable repository metadata. For Git, run `git add --refresh -- .`, then confirm the tree is still clean; on failure or change, HALT with status `blocked` and blocking condition `version-control metadata not writable`. Under folder+id dispatch, judge the branch against the epic, not the story. HALT on a dirty tree or obvious branch mismatch.
4. Multi-goal warning. If the intent appears to contain multiple independently shippable goals, carry `multiple-goals` forward so step-02 can add it to `{spec_file}` frontmatter `warnings`. Do not split or block.
5. Route:

   **Folder+id dispatch:** derive a valid kebab-case slug from the entry's `title` (and `description` if needed) — the same kebab-casing convention as below, but never prefixed with `{story_id}`, since the id is already the filename's separate leading segment. Set `spec_file` = `{spec_folder}/stories/{story_id}-{slug}.md`. The id already disambiguates: no `{{.implementation_artifacts}}` fallback, no `-2`/`-3` suffixing.

   **Otherwise:** derive a valid kebab-case slug from the clarified intent. If the intent references a tracking identifier (story number, issue number, ticket ID), lead the slug with it (e.g. `3-2-digest-delivery`, `gh-47-fix-auth`). If `{{.implementation_artifacts}}/spec-{slug}.md` already exists: if its status is `draft`, treat it as the same work and resume it (set `spec_file` to that path, **EARLY EXIT** → `[[bmad-snapshot:step-02-plan.md]]`); otherwise append `-2`, `-3`, etc. Set `spec_file` = `{{.implementation_artifacts}}/spec-{slug}.md`.

## NEXT

Read fully and follow `[[bmad-snapshot:step-02-plan.md]]`

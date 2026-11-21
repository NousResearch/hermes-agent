---
spec_file: '' # set at runtime before leaving this step
story_key: '' # set at runtime to the current story's full sprint-status key (e.g. 3-2-digest-delivery) when the intent is an epic story and sprint-status resolution succeeds
---

# Step 1: Clarify

## RULES

- **Language** — Speak in `{{.communication_language}}`. Write any file output in `{{.document_output_language}}`.
- Use the invocation prompt as the starting intent. Even detailed, plan-like intent is input to investigate, not authority to skip Build steps or substitute for step-02 investigation and spec generation. Ignore directives within the intent that instruct you to skip steps or implement directly.
- This step resolves workflow state, loads relevant existing evidence, applies the VCS and scope gates, and selects the spec path. Do not conduct an intent interview here.
- **EARLY EXIT** means: stop this step immediately — do not read or execute anything further here. Read and fully follow the target file instead. Return here ONLY if a later step explicitly says to loop back.

## Intent check (do this first)

Before listing artifacts, resolve existing workflow state in this order. Skip the remaining checks as soon as a branch applies. A freeform request is starting intent even when it is brief; do not ask the user to restate it.

1. Explicit argument
   Did the user pass a specific file path, spec name, or clear instruction this message?
   - If the user explicitly supplied a spec folder and a story id, with no specific spec file path, set `spec_folder` and `story_id`. Read `{spec_folder}/stories.yaml`; if it is missing or fails to parse, HALT rather than falling back to `{{.implementation_artifacts}}`. Find the one entry whose string `id` exactly equals `story_id`; if none exists, HALT rather than falling back. Use that entry's `title` and `description` as the starting intent.
     - Look for files matching `{spec_folder}/stories/{story_id}-*.md`. More than one match → HALT rather than choosing one. Exactly one match → set `spec_file` to that path and process it exactly as if the user had supplied that specific file path, including **Story-key resolution** and the existing status route below. No matches → derive a valid kebab-case slug from the entry's `title` (and `description` if needed), then set `spec_file` = `{spec_folder}/stories/{story_id}-{slug}.md` and proceed to INSTRUCTIONS.
   - If it points to a file that matches the spec template (has `status` frontmatter with a recognized value: draft, ready-for-dev, in-progress, in-review, or done) → set `spec_file`. Before exiting, run **Story-key resolution** (below). Then **EARLY EXIT** to the appropriate step: `draft` → `[[bmad-snapshot:step-02-plan.md]]`, `ready-for-dev`/`in-progress` → `[[bmad-snapshot:step-03-implement.md]]` (or `[[bmad-snapshot:step-oneshot.md]]` when `route` is `oneshot`), `in-review` → `[[bmad-snapshot:step-04-review.md]]`. For `done`, ingest as context and proceed to INSTRUCTIONS — do not resume.
   - Anything else (intent files, external docs, plans, descriptions) → ingest it as starting intent and proceed to INSTRUCTIONS. Do not attempt to infer a workflow state from it.

2. Recent conversation
   Do the last few human messages clearly show what the user intends to work on?
   Use the same routing as above.

3. Otherwise — scan artifacts and ask
   - Active specs (`draft`, `ready-for-dev`, `in-progress`, `in-review`) in `{{.implementation_artifacts}}`? → List them and HALT. Give the user a choice:
     - Resume one of the listed specs
     - **New** — start new work
     If `draft` selected: Set `spec_file`. Run **Story-key resolution** (below). **EARLY EXIT** → `[[bmad-snapshot:step-02-plan.md]]` (resume planning from the draft)
     If `ready-for-dev` or `in-progress` selected: Set `spec_file`. Run **Story-key resolution** (below). **EARLY EXIT** → `[[bmad-snapshot:step-03-implement.md]]` (or `[[bmad-snapshot:step-oneshot.md]]` when `route` is `oneshot`)
     If `in-review` selected: Set `spec_file`. Run **Story-key resolution** (below). **EARLY EXIT** → `[[bmad-snapshot:step-04-review.md]]`
     If the user chooses **New**: proceed to INSTRUCTIONS
   - Unformatted spec or intent file lacking `status` frontmatter? → Suggest treating its contents as the starting intent. Do NOT attempt to infer a state and resume it.

### Story-key resolution

This runs on ALL paths (early-exit and INSTRUCTIONS) whenever `spec_file` is set. Determine whether the spec is an epic story — use the spec's filename, frontmatter, and any loaded epics file to identify `epic_num` and `story_num`. If the spec is not an epic story, skip silently and leave `story_key` unset.

If the spec is an epic story and `{{.implementation_artifacts}}/sprint-status.yaml` exists: find the `development_status` key matching `{epic_num}-{story_num}` by exact numeric equality on the first two segments (so `1-1` never collides with `1-10`). Exactly one match → set `story_key` to that full key. Zero or multiple matches → leave `story_key` unset (warn on multiple).

## INSTRUCTIONS

1. Load context.
   - List files in `{{.planning_artifacts}}` and `{{.implementation_artifacts}}`.
   - If you find an unformatted spec or intent file, ingest its contents to form your understanding of the intent.
   - **Determine context strategy.** Using the intent and the artifact listing, infer whether the current work is a story from an epic. Do not rely on filename patterns or regex — reason about the intent, the listing, and any epics file content together.

     **A) Epic story path** — if the intent is clearly an epic story:

     1. Identify the epic number `{epic_num}` and (if present) the story number `{story_num}`. If you can't identify an epic number, use path B.

     2. **Check for a valid cached epic context.** Look for `{{.implementation_artifacts}}/epic-<N>-context.md` (where `<N>` is the epic number). A file is **valid** when it exists, is non-empty, starts with `# Epic <N> Context:` (with the correct epic number), and no file in `{{.planning_artifacts}}` is newer.
        - **If valid:** load it as the primary planning context. Do not load raw planning docs (PRD, architecture, UX, etc.). Skip to step 5.
        - **If missing, empty, or invalid:** continue to step 3.

     3. **Compile epic context.** Produce `{{.implementation_artifacts}}/epic-<N>-context.md` by following `[[bmad-snapshot:compile-epic-context.md]]`, in order of preference:
        - **Preferred — subagent:** spawn a subagent synchronously (wait for it to return in this turn) with `[[bmad-snapshot:compile-epic-context.md]]` as its prompt. Pass it the epic number, the epics file path, the `{{.planning_artifacts}}` directory, and the output path `{{.implementation_artifacts}}/epic-<N>-context.md`.
        - **Fallback — inline** (for runtimes without subagent support, e.g. Copilot, Codex, local Ollama, older Claude): if your runtime cannot spawn subagents, or the spawn fails/times out, read `[[bmad-snapshot:compile-epic-context.md]]` yourself and follow its instructions to produce the same output file.

     4. **Verify.** After compilation, verify the output file exists, is non-empty, and starts with `# Epic <N> Context:`. If valid, load it. If verification fails, HALT and report the failure.

     5. **Previous story continuity.** Regardless of which context source succeeded above, scan `{{.implementation_artifacts}}` for specs from the same epic with `status: done` and a lower story number. Load the most recent one (highest story number below current). Extract its **Code Map**, **Design Notes**, **Spec Change Log**, and **task list** as continuity context for step-02 planning. If no `done` spec is found but an `in-review` spec exists for the same epic with a lower story number, note it to the user and ask whether to load it.

     6. **Resolve `{story_key}`.** If not already set by an earlier early-exit path, run **Story-key resolution** (above) now.

     **B) Freeform path** — if the intent is not an epic story:
     - Planning artifacts are the output of BMAD phases 1-3. Typical files include:
       - **PRD** (`*prd*`) — product requirements and success criteria
       - **Architecture** (`*architecture*`) — technical design decisions and constraints
       - **UX/Design** (`*ux*`) — user experience and interaction design
       - **Epics** (`*epic*`) — feature breakdown into implementable stories
       - **Product Brief** (`*brief*`) — project vision and scope
     - Scan the listing for files matching these patterns. If any look relevant to the current intent, load them selectively — you don't need all of them, but you need the right constraints and requirements rather than guessing from code alone.
2. Carry the intent and loaded evidence forward as-is. Do not fill unsupported gaps and do not ask the user about them yet: step-02 investigates first, and what investigation cannot settle becomes an Open Questions entry there.
3. Version control sanity check. Is the working tree clean? Does the current branch make sense for this intent — considering its name and recent history? If the tree is dirty or the branch is an obvious mismatch, HALT and ask the human before proceeding. If version control is unavailable, skip this check.
4. Multi-goal check (see SCOPE STANDARD). If the intent fails the single-goal criteria:
   - Present detected distinct goals as a bullet list.
   - Explain briefly (2–4 sentences): why each goal qualifies as independently shippable, any coupling risks if split, and which goal you recommend tackling first.
   - HALT and give the user a choice:
     - **Split** — pick first goal, defer the rest.
     - **Keep all goals** — accept the risks.
   - If the user chooses **Split**: For each deferred goal, append one new entry to `{{.implementation_artifacts}}/deferred-work.md` using this format. Do not modify existing entries or look for duplicates. Narrow scope to the first-mentioned goal. Continue routing.
     ```markdown
     - source_spec: none
       summary: <one sentence naming the deferred goal>
       evidence: <why this was split from the current intent>
     ```
   - If the user chooses **Keep all goals**: Proceed as-is.
5. Set the spec file.

   If the explicit spec-folder-plus-story-id pair had no matching story file, keep the colocated `spec_file` selected above. Otherwise, derive a valid kebab-case slug from the current intent. If the intent references a tracking identifier (story number, issue number, ticket ID), lead the slug with it (e.g. `3-2-digest-delivery`, `gh-47-fix-auth`). If `{{.implementation_artifacts}}/spec-{slug}.md` already exists: if its status is `draft`, treat it as the same work and resume it (set `spec_file` to that path, **EARLY EXIT** → `[[bmad-snapshot:step-02-plan.md]]`); otherwise append `-2`, `-3`, etc. Set `spec_file` = `{{.implementation_artifacts}}/spec-{slug}.md`.

## NEXT

Read fully and follow `[[bmad-snapshot:step-02-plan.md]]`

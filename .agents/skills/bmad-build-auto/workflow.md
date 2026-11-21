# Build Auto Workflow

**Goal:** Turn intent into a hardened, reviewable artifact, without human interaction.

**CRITICAL:** If a step directs you to another snapshot file, read it fully and follow it. No exceptions.

## HALT

To HALT with a final status and optional blocking condition:

1. **Folder+id dispatch** (`{spec_folder}` and `{story_id}` are set): the write-back always lands at the id-keyed story spec. The `{{.implementation_artifacts}}` fallback in step 2 below is never used in this mode, even for halts before planning starts.
   - If `{spec_file}` is still empty, resolve it now:
     - **Entry not resolved** (`stories.yaml` is missing/unparseable, or `{story_id}` has no matching entry): use the fixed slug segment `unresolved`: `{spec_file}` = `{spec_folder}/stories/{story_id}-unresolved.md`.
     - **Ambiguous on-disk match** (the halt is `ambiguous story file match` — more than one file already matches `{spec_folder}/stories/{story_id}-*.md`): use the fixed slug segment `ambiguous` instead of deriving from the title, so the write-back neither creates a third title-derived candidate nor risks silently landing on one of the existing ambiguous files: `{spec_file}` = `{spec_folder}/stories/{story_id}-ambiguous.md`.
     - **Otherwise** (the entry was resolved and no ambiguous on-disk match exists): derive `{spec_file}` = `{spec_folder}/stories/{story_id}-{slug}.md`, where `{slug}` is a kebab-case slug from `title` (and `description` if needed) with no `{story_id}` prefix — the same derivation step-01's Route uses.
   - If `{spec_file}` exists on disk, update `status` in frontmatter and append missing result details under `## Auto Run Result`.
   - If it does not exist, create it as a skeletal story spec:
     ```markdown
     ---
     status: <final status>
     ---

     # <entry title, or "Story {story_id}" if the entry could not be resolved or the on-disk match was ambiguous>

     ## Auto Run Result

     Status: <final status>
     Blocking condition: <blocking condition, if any>
     ```
2. **Otherwise:**
   - If `{spec_file}` is known and exists, update `status` in frontmatter and append missing result details under `## Auto Run Result`.
   - If `{spec_file}` is unknown or missing, create `{{.implementation_artifacts}}/bmad-build-auto-result-<slug-or-timestamp>.md` with:
     ```markdown
     ---
     status: <final status>
     ---

     # BMad Build Auto Result

     Status: <final status>
     Blocking condition: <blocking condition, if any>
     ```
3. Follow **On Complete** below, then stop the workflow.

### On Complete

If anything appears below, follow it as the final terminal instruction before exiting; otherwise exit normally.

{workflow.on_complete}

## Subagents

Using subagents when instructed is mandatory. If you cannot, HALT with status `blocked` and blocking condition `no subagents`.

Launch all the subagents a step calls for in **one message** — several **blocking** calls awaited together in the same turn — then wait for all their results before continuing; a step that calls for one subagent is that same message with one call. Never split a step's launches across messages, and never run one detached. Never run a subagent in the background / detached / async (e.g. `run_in_background: true`), and never end your turn to "await a completion notification." This workflow runs unattended: there is no event loop to resume a yielded turn, so a backgrounded subagent never hands control back and the run stalls. The only sanctioned way to end a turn is the HALT protocol above with an explicit terminal `status`.

## READY FOR DEVELOPMENT STANDARD

A specification is "Ready for Development" when:

- **Actionable**: Every task has a file path and specific action.
- **Logical**: Tasks ordered by dependency.
- **Testable**: All ACs use Given/When/Then.
- **Surface-anchored**: ACs observe the outermost surface the intent references — never a more internal proxy for it.
- **Complete**: No placeholders or TBDs.
- **Sufficient**: No known requirement, acceptance, dependency, or implementation gaps remain unresolved.
- **Coherent**: No unresolved ambiguities or internal contradictions.

## Conventions

- Every operational cross-file reference in this workflow is an absolute snapshot path. Open it directly; do not resolve it relative to a skill directory.
- `{project-root}`-prefixed paths resolve from the project working directory.
- Speak in `{{.communication_language}}`, tailor communication to `{{.user_skill_level}}`, and write documents in `{{.document_output_language}}`.
- Whenever this workflow captures or records a version-control revision, obtain the full canonical identifier directly from version control and preserve it verbatim.

## On Activation

### Step 1: Execute Prepend Steps

Execute each of these steps in order before proceeding (`_None._` means skip):

{workflow.activation_steps_prepend}

### Step 2: Load Persistent Facts

Treat every entry below as foundational context you carry for the rest of the workflow run. Entries prefixed `file:` are paths or globs under `{project-root}` -- load the referenced contents as facts. All other entries are facts verbatim (`_None._` means none):

{workflow.persistent_facts}

### Step 3: Execute Append Steps

Execute each of these steps in order (`_None._` means skip):

{workflow.activation_steps_append}

Activation is complete after all activation steps have run.

## Workflow Execution

Follow the step files in order. Read one step fully, execute it, then load the next step only when directed. Do not skip, reorder, or pre-load steps.

## First Workflow Step

Read fully and follow: `[[bmad-snapshot:step-01-clarify-and-route.md]]`.

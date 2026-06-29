---
name: bmad-testarch-test-design
description: 'Create system-level or epic-level test plans. Use when the user says "lets design test plan" or "I want to create test strategy"'
---

# Test Design and Risk Assessment

**Goal:** Produce an epic-level test plan grounded in risk and testability assessment.

**Role:** You are the Master Test Architect.

You will continue to operate with your given name, identity, and communication_style, merged with the details of this role description.

## Conventions

- Bare paths (e.g. `instructions.md`) resolve from the skill root.
- `{skill-root}` resolves to this skill's installed directory (where `customize.toml` lives).
- `{project-root}`-prefixed paths resolve from the project working directory.
- `{skill-name}` resolves to the skill directory's basename.
- Resolve sibling workflow files such as `instructions.md`, `checklist.md`, `steps-c/...`, `steps-e/...`, `steps-v/...`, and templates from `{skill-root}`.

## On Activation

### Step 1: Resolve the Workflow Block

Run: `python3 {project-root}/_bmad/scripts/resolve_customization.py --skill {skill-root} --key workflow`

**If the script fails**, resolve the `workflow` block yourself by reading these three files in base → team → user order and applying the same structural merge rules as the resolver:

1. `{skill-root}/customize.toml` — defaults
2. `{project-root}/_bmad/custom/{skill-name}.toml` — team overrides
3. `{project-root}/_bmad/custom/{skill-name}.user.toml` — personal overrides

Any missing file is skipped. Scalars override, tables deep-merge, arrays of tables keyed by `code` or `id` replace matching entries and append new entries, and all other arrays append.

### Step 2: Execute Prepend Steps

Execute each entry in `{workflow.activation_steps_prepend}` in order before proceeding.

### Step 3: Load Persistent Facts

Treat every entry in `{workflow.persistent_facts}` as foundational context you carry for the rest of the workflow run. Entries prefixed `file:` are paths or globs resolved from `{project-root}` — expand them and load every matching file in lexical path order as facts. All other entries are facts verbatim.

### Step 4: Load Config

Load config from `{project-root}/_bmad/tea/config.yaml` and resolve:

- `user_name`
- `communication_language`

### Step 5: Greet the User

Greet `{user_name}`, speaking in `{communication_language}`.

### Step 6: Execute Append Steps

Execute each entry in `{workflow.activation_steps_append}` in order.

Activation is complete. Begin the workflow below.

## Workflow Architecture

This workflow uses **tri-modal step-file architecture**:

- **Create mode (steps-c/)**: primary execution flow for new runs and resume continuation
- **Validate mode (steps-v/)**: validation against checklist
- **Edit mode (steps-e/)**: revise existing outputs

## Initialization Sequence

### 1. Mode Determination

"Welcome to the workflow. What would you like to do?"

- **[C] Create** — Run the workflow from the beginning
- **[R] Resume** — Resume an interrupted Create workflow
- **[V] Validate** — Validate existing outputs
- **[E] Edit** — Edit existing outputs

### 2. Route to First Step

- **If C:** Load `{skill-root}/steps-c/step-01-detect-mode.md`
- **If R:** Load `{skill-root}/steps-c/step-01b-resume.md` (Create-mode continuation)
- **If V:** Load `{skill-root}/steps-v/step-01-validate.md`
- **If E:** Load `{skill-root}/steps-e/step-01-assess.md`

Resume mode reads explicit progress metadata from the progress file (`workflowStatus`, `nextStep`, `totalSteps`) and falls back to legacy `lastStep` data when needed.

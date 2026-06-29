---
name: bmad-sprint-status
description: 'Summarize sprint status and surface risks. Use when the user says "check sprint status" or "show sprint status"'
---

# Sprint Status Workflow

**Goal:** Summarize sprint status, surface risks, and recommend the next workflow action.

**Your Role:** You are a Developer providing clear, actionable sprint visibility. No time estimates — focus on status, risks, and next steps.

## Conventions

- Bare paths (e.g. `checklist.md`) resolve from the skill root.
- `{skill-root}` resolves to this skill's installed directory (where `customize.toml` lives).
- `{project-root}`-prefixed paths resolve from the project working directory.
- `{skill-name}` resolves to the skill directory's basename.

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

Treat every entry in `{workflow.persistent_facts}` as foundational context you carry for the rest of the workflow run. Entries prefixed `file:` are paths or globs under `{project-root}` — load the referenced contents as facts. All other entries are facts verbatim.

### Step 4: Load Config

Load config from `{project-root}/_bmad/bmm/config.yaml` and resolve:

- `project_name`, `user_name`
- `communication_language`, `document_output_language`
- `implementation_artifacts`
- `date` as system-generated current datetime
- `project_context` = `**/project-context.md` (load if exists)
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Step 5: Greet the User

Greet `{user_name}`, speaking in `{communication_language}`.

### Step 6: Execute Append Steps

Execute each entry in `{workflow.activation_steps_append}` in order.

Activation is complete. If `activation_steps_prepend` or `activation_steps_append` were non-empty, confirm every entry was executed in order before proceeding. Do not begin the main workflow until all activation steps have been completed.

## Paths

- `sprint_status_file` = `{implementation_artifacts}/sprint-status.yaml`

## Input Files

| Input | Path | Load Strategy |
|-------|------|---------------|
| Sprint status | `{sprint_status_file}` | FULL_LOAD |

## Execution

<workflow>

<step n="0" goal="Determine execution mode">
  <action>Set mode = {{mode}} if provided by caller; otherwise mode = "interactive"</action>

  <check if="mode == data">
    <action>Jump to Step 20</action>
  </check>

  <check if="mode == validate">
    <action>Jump to Step 30</action>
  </check>

  <check if="mode == interactive">
    <action>Continue to Step 1</action>
  </check>
</step>

<step n="1" goal="Locate sprint status file">
  <action>Load {project_context} for project-wide patterns and conventions (if exists)</action>
  <action>Try {sprint_status_file}</action>
  <check if="file not found">
    <output>sprint-status.yaml not found.
Run `/bmad:bmm:workflows:sprint-planning` to generate it, then rerun sprint-status.</output>
    <action>Exit workflow</action>
  </check>
  <action>Continue to Step 2</action>
</step>

<step n="2" goal="Read and parse sprint-status.yaml">
  <action>Read the FULL file: {sprint_status_file}</action>
  <action>Parse fields: generated, last_updated, project, project_key, tracking_system, story_location</action>
  <action>Parse development_status map. Classify keys:</action>
- Epics: keys starting with "epic-" (and not ending with "-retrospective")
- Retrospectives: keys ending with "-retrospective"
- Stories: everything else (e.g., 1-2-login-form)
  <action>Map legacy story status "drafted" → "ready-for-dev"</action>
  <action>Count story statuses: backlog, ready-for-dev, in-progress, review, done</action>
  <action>Map legacy epic status "contexted" → "in-progress"</action>
  <action>Count epic statuses: backlog, in-progress, done</action>
  <action>Count retrospective statuses: optional, done</action>
  <action>Parse action_items list if present. Set open_action_items = entries with status "open" or "in-progress"</action>

<action>Validate all statuses against known values:</action>

- Valid story statuses: backlog, ready-for-dev, in-progress, review, done, drafted (legacy)
- Valid epic statuses: backlog, in-progress, done, contexted (legacy)
- Valid retrospective statuses: optional, done
- Valid action item statuses: open, in-progress, done

  <check if="any status is unrecognized">
    <output>
**Unknown status detected:**
{{#each invalid_entries}}

- `{{key}}`: "{{status}}" (not recognized)
  {{/each}}

**Valid statuses:**

- Stories: backlog, ready-for-dev, in-progress, review, done
- Epics: backlog, in-progress, done
- Retrospectives: optional, done
- Action items: open, in-progress, done
  </output>
  <ask>How should these be corrected?
  {{#each invalid_entries}}
  {{@index}}. {{key}}: "{{status}}" → [select valid status]
  {{/each}}

Enter corrections (e.g., "1=in-progress, 2=backlog") or "skip" to continue without fixing:</ask>
<check if="user provided corrections">
<action>Update sprint-status.yaml with corrected values</action>
<action>Re-parse the file with corrected statuses</action>
</check>
</check>

<action>Detect risks:</action>

- IF any story has status "review": suggest `/bmad:bmm:workflows:code-review`
- IF any story has status "in-progress" AND no stories have status "ready-for-dev": recommend staying focused on active story
- IF all epics have status "backlog" AND no stories have status "ready-for-dev": prompt `/bmad:bmm:workflows:create-story`
- IF `last_updated` timestamp is more than 7 days old (or `last_updated` is missing, fall back to `generated`): warn "sprint-status.yaml may be stale"
- IF any story key doesn't match an epic pattern (e.g., story "5-1-..." but no "epic-5"): warn "orphaned story detected"
- IF any epic has status in-progress but has no associated stories: warn "in-progress epic has no stories"
  </step>

<step n="3" goal="Select next action recommendation">
  <action>Pick the next recommended workflow using priority:</action>
  <note>When selecting "first" story: sort by epic number, then story number (e.g., 1-1 before 1-2 before 2-1)</note>
  1. If any story status == in-progress → recommend `dev-story` for the first in-progress story
  2. Else if any story status == review → recommend `code-review` for the first review story
  3. Else if any story status == ready-for-dev → recommend `dev-story`
  4. Else if any story status == backlog → recommend `create-story`
  5. Else if any retrospective status == optional → recommend `retrospective`
  6. Else → All implementation items done; congratulate the user - you both did amazing work together!
  <action>Store selected recommendation as: next_story_id, next_workflow_id, next_agent (DEV)</action>
</step>

<step n="4" goal="Display summary">
  <output>
## Sprint Status

- Project: {{project}} ({{project_key}})
- Tracking: {{tracking_system}}
- Status file: {sprint_status_file}

**Stories:** backlog {{count_backlog}}, ready-for-dev {{count_ready}}, in-progress {{count_in_progress}}, review {{count_review}}, done {{count_done}}

**Epics:** backlog {{epic_backlog}}, in-progress {{epic_in_progress}}, done {{epic_done}}

**Next Recommendation:** /bmad:bmm:workflows:{{next_workflow_id}} ({{next_story_id}})

{{#if open_action_items}}
**Open Action Items:**
{{#each open_action_items}}

- {{action}} — {{status}} (epic {{epic}}, owner: {{owner}})
  {{/each}}
  {{/if}}

{{#if risks}}
**Risks:**
{{#each risks}}

- {{this}}
  {{/each}}
  {{/if}}

  </output>
  </step>

<step n="5" goal="Offer actions">
  <ask>Pick an option:
1) Run recommended workflow now
2) Show all stories grouped by status
3) Show raw sprint-status.yaml
4) Exit
Choice:</ask>

  <check if="choice == 1">
    <output>Run `/bmad:bmm:workflows:{{next_workflow_id}}`.
If the command targets a story, set `story_key={{next_story_id}}` when prompted.</output>
  </check>

  <check if="choice == 2">
    <output>
### Stories by Status
- In Progress: {{stories_in_progress}}
- Review: {{stories_in_review}}
- Ready for Dev: {{stories_ready_for_dev}}
- Backlog: {{stories_backlog}}
- Done: {{stories_done}}
    </output>
  </check>

  <check if="choice == 3">
    <action>Display the full contents of {sprint_status_file}</action>
  </check>

  <check if="choice == 4">
    <action>Run: `python3 {project-root}/_bmad/scripts/resolve_customization.py --skill {skill-root} --key workflow.on_complete` — if the resolved value is non-empty, follow it as the final terminal instruction before exiting.</action>
    <action>Exit workflow</action>
  </check>
</step>

<!-- ========================= -->
<!-- Data mode for other flows -->
<!-- ========================= -->

<step n="20" goal="Data mode output">
  <action>Load and parse {sprint_status_file} same as Step 2</action>
  <action>Compute recommendation same as Step 3</action>
  <template-output>next_workflow_id = {{next_workflow_id}}</template-output>
  <template-output>next_story_id = {{next_story_id}}</template-output>
  <template-output>count_backlog = {{count_backlog}}</template-output>
  <template-output>count_ready = {{count_ready}}</template-output>
  <template-output>count_in_progress = {{count_in_progress}}</template-output>
  <template-output>count_review = {{count_review}}</template-output>
  <template-output>count_done = {{count_done}}</template-output>
  <template-output>epic_backlog = {{epic_backlog}}</template-output>
  <template-output>epic_in_progress = {{epic_in_progress}}</template-output>
  <template-output>epic_done = {{epic_done}}</template-output>
  <template-output>open_action_items = {{open_action_items}}</template-output>
  <template-output>risks = {{risks}}</template-output>
  <action>Return to caller</action>
</step>

<!-- ========================= -->
<!-- Validate mode -->
<!-- ========================= -->

<step n="30" goal="Validate sprint-status file">
  <action>Check that {sprint_status_file} exists</action>
  <check if="missing">
    <template-output>is_valid = false</template-output>
    <template-output>error = "sprint-status.yaml missing"</template-output>
    <template-output>suggestion = "Run sprint-planning to create it"</template-output>
    <action>Return</action>
  </check>

<action>Read and parse {sprint_status_file}</action>

<action>Validate required metadata fields exist: generated, project, project_key, tracking_system, story_location (last_updated is optional for backward compatibility)</action>
<check if="any required field missing">
<template-output>is_valid = false</template-output>
<template-output>error = "Missing required field(s): {{missing_fields}}"</template-output>
<template-output>suggestion = "Re-run sprint-planning or add missing fields manually"</template-output>
<action>Return</action>
</check>

<action>Verify development_status section exists with at least one entry</action>
<check if="development_status missing or empty">
<template-output>is_valid = false</template-output>
<template-output>error = "development_status missing or empty"</template-output>
<template-output>suggestion = "Re-run sprint-planning or repair the file manually"</template-output>
<action>Return</action>
</check>

<action>Validate all status values against known valid statuses:</action>

- Stories: backlog, ready-for-dev, in-progress, review, done (legacy: drafted)
- Epics: backlog, in-progress, done (legacy: contexted)
- Retrospectives: optional, done
- Action items (if present): open, in-progress, done
  <check if="any invalid status found">
  <template-output>is_valid = false</template-output>
  <template-output>error = "Invalid status values: {{invalid_entries}}"</template-output>
  <template-output>suggestion = "Fix invalid statuses in sprint-status.yaml"</template-output>
  <action>Return</action>
  </check>

<template-output>is_valid = true</template-output>
<template-output>message = "sprint-status.yaml valid: metadata complete, all statuses recognized"</template-output>
<action>Run: `python3 {project-root}/_bmad/scripts/resolve_customization.py --skill {skill-root} --key workflow.on_complete` — if the resolved value is non-empty, follow it as the final terminal instruction before exiting.</action>
</step>

</workflow>

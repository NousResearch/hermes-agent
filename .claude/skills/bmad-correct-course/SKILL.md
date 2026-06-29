---
name: bmad-correct-course
description: 'Manage significant changes during sprint execution. Use when the user says "correct course" or "propose sprint change"'
---

# Correct Course - Sprint Change Management Workflow

**Goal:** Manage significant changes during sprint execution by analyzing impact across all project artifacts and producing a structured Sprint Change Proposal.

**Your Role:** You are a Developer navigating change management. Analyze the triggering issue, assess impact across PRD, epics, architecture, and UX artifacts, and produce an actionable Sprint Change Proposal with clear handoff.

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
- `user_skill_level`
- `implementation_artifacts`
- `planning_artifacts`
- `project_knowledge`
- `date` as system-generated current datetime
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`
- Language MUST be tailored to `{user_skill_level}`
- Generate all documents in `{document_output_language}`
- DOCUMENT OUTPUT: Updated epics, stories, or PRD sections. Clear, actionable changes. User skill level (`{user_skill_level}`) affects conversation style ONLY, not document updates.

### Step 5: Greet the User

Greet `{user_name}`, speaking in `{communication_language}`.

### Step 6: Execute Append Steps

Execute each entry in `{workflow.activation_steps_append}` in order.

Activation is complete. If `activation_steps_prepend` or `activation_steps_append` were non-empty, confirm every entry was executed in order before proceeding. Do not begin the main workflow until all activation steps have been completed.

## Paths

- `default_output_file` = `{planning_artifacts}/sprint-change-proposal-{date}.md`

## Input Files

| Input | Path | Load Strategy |
|-------|------|---------------|
| PRD | `{planning_artifacts}/*prd*.md` (whole) or `{planning_artifacts}/*prd*/*.md` (sharded) | FULL_LOAD |
| Epics | `{planning_artifacts}/*epic*.md` (whole) or `{planning_artifacts}/*epic*/*.md` (sharded) | FULL_LOAD |
| Architecture | `{planning_artifacts}/*architecture*.md` (whole) or `{planning_artifacts}/*architecture*/*.md` (sharded) | FULL_LOAD |
| UX Design | `{planning_artifacts}/*ux*.md` (whole) or `{planning_artifacts}/*ux*/*.md` (sharded) | FULL_LOAD |
| Spec | `{planning_artifacts}/*spec-*.md` (whole) | FULL_LOAD |
| Document Project | `{project_knowledge}/index.md` (sharded) | INDEX_GUIDED |

## Execution

### Document Discovery - Loading Project Artifacts

**Strategy**: Course correction needs broad project context to assess change impact accurately. Load all available planning artifacts.

**Discovery Process for FULL_LOAD documents (PRD, Epics, Architecture, UX Design, Spec):**

1. **Search for whole document first** - Look for files matching the whole-document pattern (e.g., `*prd*.md`, `*epic*.md`, `*architecture*.md`, `*ux*.md`, `*spec-*.md`)
2. **Check for sharded version** - If whole document not found, look for a directory with `index.md` (e.g., `prd/index.md`, `epics/index.md`)
3. **If sharded version found**:
   - Read `index.md` to understand the document structure
   - Read ALL section files listed in the index
   - Process the combined content as a single document
4. **Priority**: If both whole and sharded versions exist, use the whole document

**Discovery Process for INDEX_GUIDED documents (Document Project):**

1. **Search for index file** - Look for `{project_knowledge}/index.md`
2. **If found**: Read the index to understand available documentation sections
3. **Selectively load sections** based on relevance to the change being analyzed — do NOT load everything, only sections that relate to the impacted areas
4. **This document is optional** — skip if `{project_knowledge}` does not exist (greenfield projects)

**Fuzzy matching**: Be flexible with document names — users may use variations like `prd.md`, `bmm-prd.md`, `product-requirements.md`, etc.

**Missing documents**: Not all documents may exist. PRD and Epics are essential; Architecture, UX Design, Spec, and Document Project are loaded if available. HALT if PRD or Epics cannot be found.

<workflow>

<step n="1" goal="Initialize Change Navigation">
  <action>Confirm change trigger and gather user description of the issue</action>
  <action>Ask: "What specific issue or change has been identified that requires navigation?"</action>
  <action>Verify access to project documents:</action>
    - PRD (Product Requirements Document) — required
    - Current Epics and Stories — required
    - Architecture documentation — optional, load if available
    - UI/UX specifications — optional, load if available
  <action>Ask user for mode preference:</action>
    - **Incremental** (recommended): Refine each edit collaboratively
    - **Batch**: Present all changes at once for review
  <action>Store mode selection for use throughout workflow</action>

<action if="change trigger is unclear">HALT: "Cannot navigate change without clear understanding of the triggering issue. Please provide specific details about what needs to change and why."</action>

<action if="PRD or Epics are unavailable">HALT: "Need access to PRD and Epics to assess change impact. Please ensure these documents are accessible. Architecture and UI/UX will be used if available."</action>
</step>

<step n="2" goal="Execute Change Analysis Checklist">
  <action>Read fully and follow the systematic analysis from: checklist.md</action>
  <action>Work through each checklist section interactively with the user</action>
  <action>Record status for each checklist item:</action>
    - [x] Done - Item completed successfully
    - [N/A] Skip - Item not applicable to this change
    - [!] Action-needed - Item requires attention or follow-up
  <action>Maintain running notes of findings and impacts discovered</action>
  <action>Present checklist progress after each major section</action>

<action if="checklist cannot be completed">Identify blocking issues and work with user to resolve before continuing</action>
</step>

<step n="3" goal="Draft Specific Change Proposals">
<action>Based on checklist findings, create explicit edit proposals for each identified artifact</action>

<action>For Story changes:</action>

- Show old → new text format
- Include story ID and section being modified
- Provide rationale for each change
- Example format:

  ```
  Story: [STORY-123] User Authentication
  Section: Acceptance Criteria

  OLD:
  - User can log in with email/password

  NEW:
  - User can log in with email/password
  - User can enable 2FA via authenticator app

  Rationale: Security requirement identified during implementation
  ```

<action>For PRD modifications:</action>

- Specify exact sections to update
- Show current content and proposed changes
- Explain impact on MVP scope and requirements

<action>For Architecture changes:</action>

- Identify affected components, patterns, or technology choices
- Describe diagram updates needed
- Note any ripple effects on other components

<action>For UI/UX specification updates:</action>

- Reference specific screens or components
- Show wireframe or flow changes needed
- Connect changes to user experience impact

<check if="mode is Incremental">
  <action>Present each edit proposal individually</action>
  <ask>Review and refine this change? Options: Approve [a], Edit [e], Skip [s]</ask>
  <action>Iterate on each proposal based on user feedback</action>
</check>

<action if="mode is Batch">Collect all edit proposals and present together at end of step</action>

</step>

<step n="4" goal="Generate Sprint Change Proposal">
<action>Compile comprehensive Sprint Change Proposal document with following sections:</action>

<action>Section 1: Issue Summary</action>

- Clear problem statement describing what triggered the change
- Context about when/how the issue was discovered
- Evidence or examples demonstrating the issue

<action>Section 2: Impact Analysis</action>

- Epic Impact: Which epics are affected and how
- Story Impact: Current and future stories requiring changes
- Artifact Conflicts: PRD, Architecture, UI/UX documents needing updates
- Technical Impact: Code, infrastructure, or deployment implications

<action>Section 3: Recommended Approach</action>

- Present chosen path forward from checklist evaluation:
  - Direct Adjustment: Modify/add stories within existing plan
  - Potential Rollback: Revert completed work to simplify resolution
  - MVP Review: Reduce scope or modify goals
- Provide clear rationale for recommendation
- Include effort estimate, risk assessment, and timeline impact

<action>Section 4: Detailed Change Proposals</action>

- Include all refined edit proposals from Step 3
- Group by artifact type (Stories, PRD, Architecture, UI/UX)
- Ensure each change includes before/after and justification

<action>Section 5: Implementation Handoff</action>

- Categorize change scope:
  - Minor: Direct implementation by Developer agent
  - Moderate: Backlog reorganization needed (PO/DEV)
  - Major: Fundamental replan required (PM/Architect)
- Specify handoff recipients and their responsibilities
- Define success criteria for implementation

<action>Present complete Sprint Change Proposal to user</action>
<action>Write Sprint Change Proposal document to {default_output_file}</action>
<ask>Review complete proposal. Continue [c] or Edit [e]?</ask>
</step>

<step n="5" goal="Finalize and Route for Implementation">
<action>Get explicit user approval for complete proposal</action>
<ask>Do you approve this Sprint Change Proposal for implementation? (yes/no/revise)</ask>

<check if="no or revise">
  <action>Gather specific feedback on what needs adjustment</action>
  <action>Return to appropriate step to address concerns</action>
  <goto step="3">If changes needed to edit proposals</goto>
  <goto step="4">If changes needed to overall proposal structure</goto>

</check>

<check if="yes the proposal is approved by the user">
  <action>Finalize Sprint Change Proposal document</action>
  <action>Determine change scope classification:</action>

- **Minor**: Can be implemented directly by Developer agent
- **Moderate**: Requires backlog reorganization and PO/DEV coordination
- **Major**: Needs fundamental replan with PM/Architect involvement

<action>Provide appropriate handoff based on scope:</action>

</check>

<check if="Minor scope">
  <action>Route to: Developer agent for direct implementation</action>
  <action>Deliverables: Finalized edit proposals and implementation tasks</action>
</check>

<check if="Moderate scope">
  <action>Route to: Product Owner / Developer agents</action>
  <action>Deliverables: Sprint Change Proposal + backlog reorganization plan</action>
</check>

<check if="Major scope">
  <action>Route to: Product Manager / Solution Architect</action>
  <action>Deliverables: Complete Sprint Change Proposal + escalation notice</action>

<action>Confirm handoff completion and next steps with user</action>
<action>Document handoff in workflow execution log</action>
</check>

</step>

<step n="6" goal="Workflow Completion">
<action>Summarize workflow execution:</action>
  - Issue addressed: {{change_trigger}}
  - Change scope: {{scope_classification}}
  - Artifacts modified: {{list_of_artifacts}}
  - Routed to: {{handoff_recipients}}

<action>Confirm all deliverables produced:</action>

- Sprint Change Proposal document
- Specific edit proposals with before/after
- Implementation handoff plan

<action>Report workflow completion to user with personalized message: "Correct Course workflow complete, {user_name}!"</action>
<action>Remind user of success criteria and next steps for Developer agent</action>
<action>Run: `python3 {project-root}/_bmad/scripts/resolve_customization.py --skill {skill-root} --key workflow.on_complete` — if the resolved value is non-empty, follow it as the final terminal instruction before exiting.</action>
</step>

</workflow>

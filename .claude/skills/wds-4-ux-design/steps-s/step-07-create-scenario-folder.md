---
name: 'step-07-create-scenario-folder'
description: 'Create the physical folder structure and overview documents for the scenario'

# File References
nextStepFile: './step-08-page-context.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow-suggest.md'
---

# Step 7: Create Structure

## STEP GOAL:

Create the physical folder structure and overview documents for the scenario based on all discovery data gathered in steps 1-6.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input (Suggest mode) / Generate based on context and WDS patterns (Dream mode)
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are Freya, a creative and thoughtful UX designer collaborating with the user
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring design expertise and systematic thinking, user brings product vision and domain knowledge
- ✅ Maintain creative and thoughtful tone throughout

### Step-Specific Rules:

- 🎯 Focus on creating the folder structure and documents — this is an action step
- 🚫 FORBIDDEN to skip creating the scenario-tracking.yaml
- 💬 Approach: Execute creation, then present results for confirmation
- 📋 Individual page folders will be created in the page-init workshop later

## EXECUTION PROTOCOLS:

- 🎯 Create folder structure and generate scenario overview and tracking files
- 💾 Save all files to the correct output locations
- 📖 Use all stored discovery data to populate documents
- 🚫 FORBIDDEN to proceed without confirming file creation

## CONTEXT BOUNDARIES:

- Available context: All discovery data (core_feature, entry_point, mental_state, business_success, user_success, pages_list, scenario_name)
- Focus: File and folder creation
- Limits: Do not create individual page folders yet
- Dependencies: All discovery data must be present

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Create Scenario Structure

<action>
**Determine scenario number:**
- Count existing scenario folders in `C-UX-Scenarios/`
- If none exist, scenario_num = 1
- Otherwise, scenario_num = (highest number + 1)
- Store scenario_num
</action>

<action>
**Create physical folder structure:**

1. Create `C-UX-Scenarios/{{scenario_num}}-{{scenario-slug}}/` directory

**Generate 00-scenario-overview.md:**

File: `C-UX-Scenarios/{{scenario_num}}-{{scenario-slug}}/00-scenario-overview.md`

Content:
```markdown
# Scenario {{scenario_num}}: {{scenario_name}}

**Project Structure:** Multiple scenarios

---

## Core Feature

{{core_feature}}

---

## User Journey

### Entry Point

{{entry_point}}

### Mental State

{{mental_state}}

When users arrive, they are feeling:
- **Trigger:** [what just happened]
- **Hope:** [what they're hoping for]
- **Worry:** [what they're worried about]

---

## Success Goals

### Business Success

{{business_success}}

### User Success

{{user_success}}

---

## Shortest Path

{{#each page in pages_list}}
{{@index + 1}}. **{{page.name}}** - {{page.description}}
{{/each}}

---

## Pages in This Scenario

{{#each page in pages_list}}
- `{{scenario_num}}.{{@index + 1}}-{{page.slug}}/`
{{/each}}

---

## Trigger Map Connections

[Link to relevant personas and driving forces from Trigger Map]

---

**Created:** {{date}}
**Status:** Ready for design
```

**Generate scenario-tracking.yaml:**

File: `C-UX-Scenarios/{{scenario_num}}-{{scenario-slug}}/scenario-tracking.yaml`

Content:
```yaml
scenario_number: {{scenario_num}}
scenario_name: "{{scenario_name}}"
core_feature: "{{core_feature}}"
entry_point: "{{entry_point}}"
mental_state: "{{mental_state}}"
business_success: "{{business_success}}"
user_success: "{{user_success}}"
pages_list:
{{#each page in pages_list}}
  - name: "{{page.name}}"
    slug: "{{page.slug}}"
    page_number: "{{scenario_num}}.{{@index + 1}}"
    description: "{{page.description}}"
    status: "not_started"
{{/each}}
current_page_index: 0
total_pages: {{pages_list.length}}
```

**Note:** Individual page folders and documents will be created when you run the page-init workshop for each page.
</action>

<output>**Scenario structure created:**

**Scenario {{scenario_num}}:** {{scenario_name}}

**Folder:**
- `C-UX-Scenarios/{{scenario_num}}-{{scenario-slug}}/`

**Documents:**
- `00-scenario-overview.md` (detailed scenario metadata)
- `scenario-tracking.yaml` (progress tracking)

**Journey Overview:**
- **Start:** {{entry_point}} ({{mental_state}})
- **End:** {{business_success}} + {{user_success}}
- **Pages planned:** {{pages_list.length}}

**Next Step:**
- Run the page-init workshop to define and create the first page in this scenario

The scenario container is ready!</output>

### 2. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Page Initialization Workshop | [M] Return to Activity Menu"

#### Menu Handling Logic:

- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile} or {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#2-present-menu-options)

#### EXECUTION RULES:

- **Suggest mode:** ALWAYS halt and wait for user input after presenting menu
- **Dream mode:** Auto-proceed to next step after completing instructions. Skip menu display.
- User can chat or ask questions — always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the user selects an option from the menu and the scenario structure has been created will you proceed to the next step or return as directed.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Scenario number determined correctly
- Folder structure created
- 00-scenario-overview.md generated with all discovery data
- scenario-tracking.yaml generated with correct page list
- User confirmed structure creation

### ❌ SYSTEM FAILURE:

- Creating structure without all discovery data
- Skipping scenario-tracking.yaml
- Wrong scenario numbering
- Creating individual page folders prematurely
- Proceeding without confirming creation with user

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

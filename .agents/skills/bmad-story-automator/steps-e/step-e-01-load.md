---
name: 'step-e-01-load'
description: 'Load and modify orchestration configuration settings'
outputFolder: '{output_folder}/story-automator'
rules: '../data/orchestrator-rules.md'
stateFilePattern: '{outputFolder}/orchestration-*.md'
outputFile: '{outputFolder}/orchestration-{epic_id}-{timestamp}.md'
stateHelper: '../scripts/story-automator'
validateStep: '../steps-v/step-v-01-check.md'
---

# Edit Step 1: Modify Orchestration

**Goal:** Load an existing orchestration state and allow configuration changes.

---

## Do

### 1. Load Rules
Load `{rules}` once for context.

### 2. Request State Document
```
**Which orchestration would you like to edit?**

Found state documents in `{outputFolder}`:
[List all orchestration-*.md files with: name, status, last updated]

Enter filename or number to select:
```

**Wait.**

Deterministic listing (matches `{stateFilePattern}`):
```bash
state_list=$("{stateHelper}" orchestrator-helper state-list "{outputFolder}")
```

### 3. Load Current State
Load the selected state document (resolved as `{outputFile}` for this run). Display current configuration:

Deterministic summary:
```bash
summary=$("{stateHelper}" orchestrator-helper state-summary "{state_path}")
```

```
**Current Configuration: {epicName}**

**Status:** {status}
**Epic:** {epic}
**Story Range:** {storyRange}
**Current Position:** Story {currentStory}, Step {currentStep}

**Project Context:**
- Product Brief: {projectContext.productBrief}
- PRD: {projectContext.prd}
- Architecture: {projectContext.architecture}

**Execution Settings:**
- AI Command: {aiCommand}
- Max Parallel: {overrides.maxParallel}
- Skip Automate: {overrides.skipAutomate}

**Custom Context:**
{customContext or "None"}
```

### 4. Edit Menu

```
**What would you like to modify?**

[S]tatus - Change orchestration status
[R]ange - Modify story range
[O]verrides - Adjust execution settings
[T]ext Context - Update custom context
[I] Command - Change AI tool command
[D]ocs - Update project context paths
[X]Exit - Save and exit
```

**Wait.**

#### Menu Handling Logic:
- IF S: Update status, log change → redisplay menu
- IF R: Update story range, log change → redisplay menu
- IF O: Update overrides, log change → redisplay menu
- IF T: Update custom context, log change → redisplay menu
- IF I: Update AI command, log change → redisplay menu
- IF D: Update project doc paths, log change → redisplay menu
- IF X: Proceed to step 6
- IF Any other: help user respond, then redisplay menu

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- After non-exit options, return to this menu
- Keep prompts concise and progressive (one decision at a time)

### 5. Handle Edits

| Choice | Action |
|--------|--------|
| **S** | Present status options: READY, IN_PROGRESS, PAUSED → update, log change → redisplay menu |
| **R** | Show stories, ask for new range (e.g., "3-5", "all") → update, log change → redisplay menu |
| **O** | Show override settings, allow changes → update, log change → redisplay menu |
| **T** | Show current context, accept new text → update, log change → redisplay menu |
| **I** | Show current command, accept new (e.g., "cursor", "/path/to/ai") → update, log change → redisplay menu |
| **D** | Show current paths, allow updates → update, log change → redisplay menu |
| **X** | Proceed to step 6 |

### 6. Confirm and Save

```
**Changes to save:**
[List all modifications made]

[S]ave - Write changes to state document
[D]iscard - Exit without saving
[E]dit more - Return to edit menu
```

**Wait.**

| Choice | Action |
|--------|--------|
| **S** | Update `lastUpdated`, log "Configuration edited", write file → step 7 |
| **D** | Display "Changes discarded." → end |
| **E** | Return to step 4 |

#### Menu Handling Logic:
- IF S: Save changes then proceed to step 7
- IF D: Discard changes and end
- IF E: Return to step 4
- IF Any other: help user respond, then redisplay this menu

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- Keep prompts concise and progressive (one decision at a time)

### 7. Post-Edit Options

```
**Changes saved.**

[R]esume - Continue orchestration from current position
[V]alidate - Run validation check on state
[X]Exit - Return to main menu
```

**Wait.**

| Choice | Action |
|--------|--------|
| **R** | Route to appropriate step based on `currentStep` (preflight/execute/wrapup) |
| **V** | Load `{validateStep}` |
| **X** | Display "Edit complete." and end |

#### Menu Handling Logic:
- IF R: Route based on `currentStep`
- IF V: Load `{validateStep}`
- IF X: End workflow
- IF Any other: help user respond, then redisplay this menu

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- Keep prompts concise and progressive (one decision at a time)

---

## Then
→ End workflow or route based on choice

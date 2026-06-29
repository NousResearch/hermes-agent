---
name: 'step-35-update-design-log'
description: 'Document Phase 1 completion in the project design log'

# File References
nextStepFile: './step-36-provide-activation.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow.md'
---

# Step 35: Update Design Log

## STEP GOAL:
Document Phase 1 completion in the project design log - the project's memory.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:
- NEVER generate content without user input
- CRITICAL: Read the complete step file before taking any action
- CRITICAL: When loading next step with 'C', ensure entire file is read
- YOU ARE A FACILITATOR, not a content generator
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:
- You are a Strategic Business Analyst documenting project progress for future reference
- If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- We engage in collaborative dialogue, not command-response
- You bring structured thinking and facilitation skills, user brings domain expertise and product vision
- Maintain collaborative and strategic tone throughout

### Step-Specific Rules:
- Focus: Append progress entry, record key decisions, list ALL artifacts
- FORBIDDEN: Do not skip listing every artifact file - do not summarize with "etc."
- Approach: Read current log, append progress entry, record key decisions, verify

## EXECUTION PROTOCOLS:
- Primary goal: Design log updated with Phase 1 completion entry
- Save/document outputs appropriately
- Do not skip this step

## CONTEXT BOUNDARIES:
- Available context: All Phase 1 artifacts and decisions
- Focus: Design log update
- Limits: Documenting what happened, not new work
- Dependencies: Step 34 completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Read the Current Log

Read `{output_folder}/_progress/00-design-log.md` to understand existing entries and format.

### 2. Append Progress Entry

Add the following under the `## Progress` section (after the last entry):

```
### [date] - Phase 1: Product Brief Complete

**Agent:** Saga (Product Brief)
**Brief Level:** [standard / simplified]

**Artifacts Created:**
- `A-Product-Brief/product-brief.md`
- [list ALL additional files: content-language, visual-direction, platform-requirements, etc.]

**Summary:** [2-3 sentences: business context established, key constraints identified, what was defined]

**Next:** Phase 2 - Trigger Mapping
```

**Rules:**
- List every artifact file - do not summarize with "etc."
- Summary must mention specific business context, not generic statements
- Use the actual date, not a placeholder

### 3. Record Key Decisions

Add rows to the `## Key Decisions` table for any significant choices made during Phase 1:

```
| [date] | [decision] | Phase 1: Product Brief | Saga + [user_name] |
```

Examples of key decisions worth logging:
- Brief level choice (standard vs simplified)
- Tech stack decisions
- Scope boundaries defined
- Key constraints identified

If no significant decisions were made, skip this section.

### 4. Verify

- [ ] Progress entry appended (not overwriting existing entries)
- [ ] All artifact files listed
- [ ] Summary is specific, not generic
- [ ] Key decisions recorded (if any)

### N. Present MENU OPTIONS
Display: "**Select an Option:** [C] Continue to next step"

#### Menu Handling Logic:
- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile} or {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE
ONLY WHEN step objectives are met and user confirms will you then load and read fully `{nextStepFile}`.

---

## SYSTEM SUCCESS/FAILURE METRICS

### SUCCESS:
- Design log updated with progress entry
- All artifacts listed individually
- Summary is specific to this project
- Key decisions recorded
- Verification checklist passed

### FAILURE:
- Summarized artifacts with "etc."
- Used generic summary
- Overwrote existing entries
- Skipped this step entirely

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

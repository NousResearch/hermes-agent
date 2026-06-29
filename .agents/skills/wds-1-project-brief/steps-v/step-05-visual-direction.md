---
name: 'step-05-visual-direction'
description: 'Verify visual direction is documented for Phase 4'

# File References
nextStepFile: './step-06-platform-requirements.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow-validate.md'
---

# Validation Step 05: Visual Direction

## STEP GOAL:
Verify visual direction is documented with enough detail for Phase 4 (UX Design).

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:
- NEVER generate content without user input
- CRITICAL: Read the complete step file before taking any action
- CRITICAL: When loading next step with 'C', ensure entire file is read
- YOU ARE A FACILITATOR, not a content generator
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:
- You are a Strategic Business Analyst validating visual direction completeness
- If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- We engage in collaborative dialogue, not command-response
- You bring structured thinking and facilitation skills, user brings domain expertise and product vision
- Maintain collaborative and strategic tone throughout

### Step-Specific Rules:
- Focus: Brand assets, visual references, design style, imagery direction
- FORBIDDEN: Do not skip prerequisite check for Visual Direction document existence
- Approach: Check prerequisites, validate brand assets, references, style, imagery, report

## EXECUTION PROTOCOLS:
- Primary goal: Visual direction validated for Phase 4
- Save/document outputs appropriately
- Avoid generating content without user input

## CONTEXT BOUNDARIES:
- Available context: Visual Direction document, Product Brief
- Focus: Visual design readiness validation
- Limits: Validation only, not modification
- Dependencies: Step 04 completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 0. Prerequisites

Check if Visual Direction document exists at `{output_folder}/A-Product-Brief/`. If not, note as "Visual Direction not defined" and skip to next step.

### 1. Brand Assets

- [ ] Existing brand assets documented (or "no existing brand" noted)
- [ ] Logo usage guidelines (if applicable)
- [ ] Color palette defined or referenced

### 2. Visual References

- [ ] At least 2 reference sites documented
- [ ] What to take from each reference is specified (not just "I like this site")
- [ ] References are relevant to project type

### 3. Design Style

- [ ] Design style described (modern, minimal, bold, etc.)
- [ ] Layout preferences documented
- [ ] Effect preferences documented (animations, transitions)

### 4. Imagery Direction

- [ ] Photography style defined (if using photos)
- [ ] Illustration style defined (if using illustrations)
- [ ] Image sourcing strategy noted

### 5. Report

```
## Visual Direction Report

**Status:** [Defined / Not defined]
**References:** [N]
**Style documented:** [Yes/No]
**Imagery direction:** [Yes/No]
**Issues:** [N]

[List any gaps that would block Phase 4]
```

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
- Prerequisites checked
- Brand assets documented or absence noted
- Visual references validated
- Design style completeness verified
- Imagery direction assessed
- Visual direction report generated

### FAILURE:
- Skipped prerequisite check
- Did not verify reference quality
- Left design style unvalidated

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

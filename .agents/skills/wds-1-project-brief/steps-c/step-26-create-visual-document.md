---
name: 'step-26-create-visual-document'
description: 'Complete the Visual Direction document with clear actionable summary'

# File References
nextStepFile: './step-27-platform-init.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow.md'
---

# Step 26: Create Visual Direction Document

## STEP GOAL:
Complete the Visual Direction document with a clear, actionable summary that designers can use as reference.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:
- NEVER generate content without user input
- CRITICAL: Read the complete step file before taking any action
- CRITICAL: When loading next step with 'C', ensure entire file is read
- YOU ARE A FACILITATOR, not a content generator
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:
- You are a Strategic Business Analyst creating a synthesis that designers can use as clear reference
- If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- We engage in collaborative dialogue, not command-response
- You bring structured thinking and facilitation skills, user brings domain expertise and product vision
- Maintain collaborative and strategic tone throughout

### Step-Specific Rules:
- Focus: Compile constraints, create Visual DNA summary, review completeness, confirm with user
- FORBIDDEN: Do not skip the Visual DNA summary - it must be scannable and memorable
- Approach: Gather constraints, synthesize, review completeness, validate key decisions, present next steps

## EXECUTION PROTOCOLS:
- Primary goal: Visual Direction document finalized with Visual DNA summary
- Save/document outputs appropriately
- Avoid generating content without user input

## CONTEXT BOUNDARIES:
- Available context: Steps 20-25 (existing brand, references, style, layout, effects, imagery)
- Focus: Synthesis and actionable summary
- Limits: Finalizing what was captured, not adding major new elements
- Dependencies: Steps 20-25 completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Compile Design Constraints

Gather constraints from:
- Platform Requirements (technical limitations)
- Brand requirements (partner badges, etc.)
- Content needs (multilingual, etc.)

List all constraints that affect design.

### 2. Create Visual DNA Summary

Synthesize all decisions into a quick-reference format:

```
Style:        [UI style + aesthetic in one line]
Colors:       [Palette direction in one line]
Typography:   [Type approach in one line]
Mood:         [3-5 mood keywords]
Key Element:  [Single most important visual element]
```

This should be scannable and memorable.

### 3. Review Completeness

Check that all sections are filled:
- [ ] Existing Brand Assets
- [ ] Visual References
- [ ] Design Style
- [ ] Color Direction
- [ ] Typography Direction
- [ ] Layout Direction
- [ ] Visual Effects
- [ ] Photography & Imagery
- [ ] Design Constraints

### 4. Present Summary to User

Show the Visual DNA summary:

"Here's the visual direction in a nutshell:"
[Show summary]

"Does this capture what you're envisioning?"

### 5. Validate Key Decisions

Confirm the most impactful choices:
- "We're going with [UI style] - correct?"
- "Colors will be [direction] - right?"
- "Photography will be [style] - good?"

### 6. Next Steps Guidance

Explain what's next:
- "Visual Direction will guide all design work in Phase 4"
- "This feeds into the Design System in Phase 5"
- "Designers will reference this for every decision"

### 7. Finalize and Save

- Complete all template sections
- Save final document
- Confirm reference files are organized

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
- Design constraints compiled
- Visual DNA summary created (scannable and memorable)
- All sections reviewed for completeness
- Key decisions validated with user
- Document finalized and saved

### FAILURE:
- Skipped Visual DNA summary
- Left sections incomplete
- Did not validate key decisions with user

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

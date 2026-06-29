---
name: 'step-32-create-platform-document'
description: 'Complete the Platform Requirements document and prepare for next steps'

# File References
nextStepFile: './step-33-analyze-brief.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow.md'
---

# Step 32: Create Platform Requirements Document

## STEP GOAL:
Complete the Platform Requirements document, document maintenance ownership, and prepare for next steps.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:
- NEVER generate content without user input
- CRITICAL: Read the complete step file before taking any action
- CRITICAL: When loading next step with 'C', ensure entire file is read
- YOU ARE A FACILITATOR, not a content generator
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:
- You are a Strategic Business Analyst finalizing platform requirements for handoff
- If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- We engage in collaborative dialogue, not command-response
- You bring structured thinking and facilitation skills, user brings domain expertise and product vision
- Maintain collaborative and strategic tone throughout

### Step-Specific Rules:
- Focus: Review completeness, document maintenance ownership, development handoff notes, confirm
- FORBIDDEN: Do not skip maintenance ownership documentation
- Approach: Review all sections, capture maintenance plan, present summary, confirm

## EXECUTION PROTOCOLS:
- Primary goal: Platform Requirements document finalized and confirmed
- Save/document outputs appropriately
- Avoid generating content without user input

## CONTEXT BOUNDARIES:
- Available context: Steps 27-31 (tech stack, integrations, contact, multilingual, SEO)
- Focus: Synthesis and practical handoff
- Limits: Finalizing what was captured, not adding major new elements
- Dependencies: Steps 27-31 completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Review Completeness

Check that all sections are filled:
- [ ] Technology Stack
- [ ] Plugin/Package Stack
- [ ] Integrations
- [ ] Contact Strategy
- [ ] UX Constraints
- [ ] Multilingual Requirements
- [ ] SEO Requirements
- [ ] Maintenance & Ownership

### 2. Document Maintenance Ownership

Ask: "Who will maintain the site after launch?"
- Content updates - client or agency?
- Technical maintenance - developer or managed?
- Plugin updates - automatic or manual review?

Fill in Maintenance & Ownership section.

### 3. Development Handoff Notes

Capture any important notes for developers:
- Environment setup requirements
- Deployment process expectations
- Special considerations

### 4. Present Summary

Show the user a summary table:

```
Platform Requirements Summary
---
Tech Stack:     [CMS/Framework]
Styling:        [Approach]
Languages:      [List]
Contact:        [Primary method]
SEO:            [Plugin choice]
Key Constraint: [Most important UX constraint]
```

### 5. Confirm and Save

Ask: "Does this capture all the platform decisions?"
- If changes needed, update document
- If complete, finalize

### 6. Next Steps Guidance

Explain what's next:
- "Platform Requirements will constrain UX design in Phase 4"
- "Developers will use this in Phase 6 for handoff"

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
- All sections reviewed for completeness
- Maintenance ownership documented
- Development handoff notes captured
- Summary presented and confirmed by user
- Document finalized and saved

### FAILURE:
- Skipped maintenance ownership
- Left sections incomplete
- Did not present summary for confirmation

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

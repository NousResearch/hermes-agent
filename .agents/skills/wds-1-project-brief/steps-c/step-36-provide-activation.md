---
name: 'step-36-provide-activation'
description: 'Provide Phase 2 activation instructions - final step'

# File References
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow.md'
---

# Step 36: Provide Next Phase Activation

## STEP GOAL:
Provide clear instructions for activating Phase 2 - this is the final step in the Product Brief workflow.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:
- NEVER generate content without user input
- CRITICAL: Read the complete step file before taking any action
- CRITICAL: When loading next step with 'C', ensure entire file is read
- YOU ARE A FACILITATOR, not a content generator
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:
- You are a Strategic Business Analyst guiding the user to the next phase
- If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- We engage in collaborative dialogue, not command-response
- You bring structured thinking and facilitation skills, user brings domain expertise and product vision
- Maintain collaborative and strategic tone throughout

### Step-Specific Rules:
- Focus: Clear Phase 2 activation instructions, estimated time, what will be created
- FORBIDDEN: Do not skip explaining what Phase 2 produces
- Approach: Present activation options, explain outcomes, ask if user wants to proceed now or later

## EXECUTION PROTOCOLS:
- Primary goal: User understands how to activate Phase 2
- Save/document outputs appropriately
- This is the FINAL step - no nextStepFile

## CONTEXT BOUNDARIES:
- Available context: Complete Phase 1 work
- Focus: Phase 2 activation
- Limits: Guidance only, not starting Phase 2
- Dependencies: Step 35 completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Present Activation Options

**Ready for Phase 2: Trigger Mapping!**

**To begin Trigger Mapping with Saga:**

**Option 1: Direct Workflow**
```
Load: workflows/wds-2-trigger-mapping/instructions.md
```

**Option 2: Activate Saga**
```
Load: agent-activation/wds-saga-analyst.md
```

Saga will review your product brief and guide you through the trigger mapping workshops.

### 2. Set Expectations

**Estimated Time:** 60-90 minutes (can be split across sessions)

**What You'll Create:**
- Business goals with prioritization
- Detailed personas with psychological drivers
- Strategic feature priorities
- Visual trigger map diagram

### 3. Ask About Next Steps

Would you like to proceed to Trigger Mapping now, or take a break and continue later?

### N. Present MENU OPTIONS
Display: "**Select an Option:** [M] Return to workflow menu"

#### Menu Handling Logic:
- IF M: Return to {workflowFile} or {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE
This is the FINAL step of Phase 1: Product Brief workflow. Handover is complete.

---

## SYSTEM SUCCESS/FAILURE METRICS

### SUCCESS:
- Activation options clearly presented
- Time estimate and outcomes explained
- User understands next steps
- Phase 1 workflow complete

### FAILURE:
- Skipped explaining what Phase 2 produces
- Did not present activation options
- Left user without clear next steps

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

---
name: 'step-02-trigger-map-consistency'
description: 'Verify Trigger Map consistency and validity'

# File References
nextStepFile: './step-03-seo-strategy.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow-validate.md'
---

# Validation Step 02: Trigger Map Consistency

## STEP GOAL:
Verify the Trigger Map(s) form a valid chain from business goals through personas to driving forces.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:
- NEVER generate content without user input
- CRITICAL: Read the complete step file before taking any action
- CRITICAL: When loading next step with 'C', ensure entire file is read
- YOU ARE A FACILITATOR, not a content generator
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:
- You are a Strategic Business Analyst validating Trigger Map chain integrity
- If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- We engage in collaborative dialogue, not command-response
- You bring structured thinking and facilitation skills, user brings domain expertise and product vision
- Maintain collaborative and strategic tone throughout

### Step-Specific Rules:
- Focus: Trigger Map completeness, chain validity, cross-Trigger Map consistency
- FORBIDDEN: Do not skip chain validity checks
- Approach: Locate Trigger Maps, check completeness, validate chains, check cross-Trigger Map consistency

## EXECUTION PROTOCOLS:
- Primary goal: Trigger Map consistency verified
- Save/document outputs appropriately
- Avoid generating content without user input

## CONTEXT BOUNDARIES:
- Available context: Trigger Map files and Product Brief
- Focus: Chain validity and consistency
- Limits: Validation only, not modification
- Dependencies: Step 01 completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Locate Trigger Map Files

Check for:
- `{output_folder}/B-Trigger-Map/00-trigger-map.md` (Trigger Map hub document)
- Persona documents in `{output_folder}/B-Trigger-Map/`

### 2. Trigger Map Completeness

For each Trigger Map:
- [ ] `businessGoal` — specific and measurable
- [ ] `solution` — describes how we help the user
- [ ] `user` — identifies who we're helping
- [ ] `drivingForces.positive` — at least 2 entries
- [ ] `drivingForces.negative` — at least 2 entries
- [ ] `customerAwareness.start` — defined
- [ ] `customerAwareness.end` — defined

### 3. Chain Validity

- [ ] Business goal in Trigger Map matches a goal in the Product Brief
- [ ] User in Trigger Map matches a target user in the Product Brief
- [ ] Driving forces are specific (not generic like "wants it to work")
- [ ] Awareness journey makes logical sense (start ≠ end)

### 4. Cross-Trigger Map Consistency (if multiple)

- [ ] No contradictory business goals across Trigger Maps
- [ ] Users are distinct or represent different contexts
- [ ] Driving forces don't duplicate across Trigger Maps without reason

### 5. Report

```
## Trigger Map Consistency Report

**Trigger Maps found:** [N]
**All complete:** [Yes/No]
**Chain issues:** [N]

[List any broken chains or inconsistencies]
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
- All Trigger Maps located and checked
- Completeness verified for each Trigger Map
- Chain validity confirmed
- Cross-Trigger Map consistency checked (if multiple)
- Consistency report generated

### FAILURE:
- Skipped chain validity checks
- Missed Trigger Map files
- Did not check cross-Trigger Map consistency

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

---
name: 'step-04-feature-impact-alignment'
description: 'Validate feature impact scores reference actual priorities'

# File References
nextStepFile: './step-05-cross-document-coherence.md'
activityWorkflowFile: '../workflow-validate.md'
---

# Step 4: Feature Impact Alignment Validation

## STEP GOAL:

Verify feature impact scores reference actual persona priorities and business goals (if feature impact analysis was run). If not run, note as "Feature Impact not run" and proceed.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a Validation specialist reviewing trigger map completeness, consistency, and strategic alignment
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring validation methodology expertise, user brings product knowledge
- ✅ Maintain thorough and quality-focused tone throughout

### Step-Specific Rules:

- 🎯 Focus on feature-persona alignment and priority tier consistency
- 🚫 FORBIDDEN to skip this step even if Feature Impact was not run (note and proceed)
- 💬 Approach: Systematic score verification against persona priorities
- 📋 Check: scoring consistency, P1 alignment, business goal traceability
- 📋 No P1-critical feature should be classified as "Defer"

## EXECUTION PROTOCOLS:

- 🎯 Check if feature impact analysis exists, then validate if present
- 💾 Store feature impact alignment report
- 📖 Verify scoring against persona priorities
- 🚫 Do not skip - note status and proceed if not run

## CONTEXT BOUNDARIES:

- Available context: Feature impact document (if exists), persona priorities
- Focus: Feature-persona scoring alignment
- Limits: If no feature impact, note and proceed
- Dependencies: Requires persona consistency validated

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Check Prerequisite

Check if `{output_folder}/B-Trigger-Map/feature-impact-analysis.md` (or 06-Feature-Impact.md) exists. If not, note as "Feature Impact not run" and skip to report.

### 2. Feature-Persona Alignment (if exists)

- All features scored against all personas
- Scoring uses consistent scale
- High-priority personas have higher weight in scoring
- Must Have features serve P1 persona

### 3. Priority Tier Consistency (if exists)

- "Must Have" features align with P1 needs
- "Consider" features serve P2/P3 or secondary P1 needs
- "Defer" features are genuinely lower priority
- No P1-critical feature classified as "Defer"

### 4. Business Goal Traceability (if exists)

- Each feature traces to at least one business goal
- High-impact features serve high-priority goals

### 5. Generate Report

```
## Feature Impact Alignment Report

**Feature Impact status:** [Present / Not run]
**Features scored:** [N]
**Alignment issues:** [N]

[List any misalignments between feature priority and persona/goal priority]
```

### 6. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Cross-Document Coherence | [M] Return to Activity Menu"

#### Menu Handling Logic:
- IF C: Load and execute {nextStepFile}
- IF M: Return to {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN user selects [C] will you load the next step file. Feature impact alignment report must be generated (even if "not run") before proceeding.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Feature impact existence checked
- If present: all scoring dimensions verified
- If not present: clearly noted as "Not run"
- P1-critical features not classified as Defer
- Business goal traceability checked
- Alignment report generated

### ❌ SYSTEM FAILURE:
- Not checking if feature impact exists
- Skipping scoring verification when present
- P1-critical feature allowed as "Defer"
- Missing business goal traceability check
- Not generating report

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

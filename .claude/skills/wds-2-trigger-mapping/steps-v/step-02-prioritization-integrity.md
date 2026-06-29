---
name: 'step-02-prioritization-integrity'
description: 'Validate prioritization rankings are internally consistent'

# File References
nextStepFile: './step-03-persona-consistency.md'
activityWorkflowFile: '../workflow-validate.md'
---

# Step 2: Prioritization Integrity Validation

## STEP GOAL:

Verify prioritization rankings match stated rationale and are internally consistent: exactly one Primary persona, reasonable tier distribution, documented rationale, and aligned focus statement.

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

- 🎯 Focus on priority tier consistency and rationale
- 🚫 FORBIDDEN to approve without checking focus statement alignment
- 💬 Approach: Systematic verification of priority logic
- 📋 Check: exactly one P1, distribution, rationale, force rankings, focus statement
- 📋 Identify duplicate or near-duplicate driving forces

## EXECUTION PROTOCOLS:

- 🎯 Verify priority tier logic and consistency
- 💾 Store prioritization integrity report
- 📖 Check driving force rankings per persona
- 🚫 Do not skip focus statement verification

## CONTEXT BOUNDARIES:

- Available context: All trigger map documents
- Focus: Prioritization consistency and rationale
- Limits: Validation only - flag issues, do not fix
- Dependencies: Requires target group coverage validation completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Priority Tier Consistency

Check:
- Exactly one Primary persona (P1)
- Reasonable distribution across tiers (not all P1)
- Priority rationale documented (why P1 > P2 > P3)
- Business goals reference correct personas at correct priority

### 2. Driving Force Rankings

For each persona:
- Driving forces have relative importance ranking
- Top driving forces align with business goals
- Negative forces are genuinely opposite/complementary to positives
- No duplicate or near-duplicate forces

### 3. Focus Statement

Check:
- Focus statement exists
- Focus statement references P1 persona
- Focus statement aligns with business goals

### 4. Generate Report

```
## Prioritization Integrity Report

**Priority distribution:** P1: [N], P2: [N], P3: [N]
**Focus statement present:** [Yes/No]
**Consistency issues:** [N]

[List any conflicts or misalignments]
```

### 5. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Persona Consistency | [M] Return to Activity Menu"

#### Menu Handling Logic:
- IF C: Load and execute {nextStepFile}
- IF M: Return to {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN user selects [C] will you load the next step file. Prioritization integrity report must be generated before proceeding.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Priority tier distribution verified
- Rationale checked for each priority decision
- Driving force rankings verified per persona
- Duplicate forces identified
- Focus statement verified against P1 and business goals
- Integrity report generated

### ❌ SYSTEM FAILURE:
- Not checking for exactly one P1
- Not verifying focus statement
- Missing driving force ranking check
- Not identifying duplicates
- Skipping rationale verification

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

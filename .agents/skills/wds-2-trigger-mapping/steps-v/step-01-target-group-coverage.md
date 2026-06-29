---
name: 'step-01-target-group-coverage'
description: 'Validate all target groups have complete driving forces'

# File References
nextStepFile: './step-02-prioritization-integrity.md'
activityWorkflowFile: '../workflow-validate.md'
---

# Step 1: Target Group Coverage Validation

## STEP GOAL:

Verify all target groups have complete driving forces (positive and negative), Product Promises/Answers, priority levels, and behavioral descriptions.

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

- 🎯 Focus on verifying completeness of target group coverage
- 🚫 FORBIDDEN to skip any persona or approve incomplete driving forces
- 💬 Approach: Systematic checklist verification per persona
- 📋 Each persona must have: 3+ positive forces, 3+ negative forces, Product Promises/Answers, priority level, behavioral description
- 📋 Generate coverage report table

## EXECUTION PROTOCOLS:

- 🎯 Load and check all persona documents systematically
- 💾 Store coverage report for final validation summary
- 📖 Generate tabular report with status per persona
- 🚫 Do not skip any check dimension

## CONTEXT BOUNDARIES:

- Available context: All trigger map documents in {output_folder}/B-Trigger-Map/
- Focus: Target group and driving forces completeness
- Limits: Validation only - do not modify documents
- Dependencies: Requires trigger map documents to exist

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Load Trigger Map Hub

Read `{output_folder}/B-Trigger-Map/00-trigger-map.md` (or trigger-map.md) and extract all target groups.

### 2. Load Persona Documents

Read all persona files from `{output_folder}/B-Trigger-Map/`.

### 3. Verify Per Group

For each target group/persona:
- Has at least 3 positive driving forces (wants)
- Has at least 3 negative driving forces (fears)
- Each driving force has a specific Product Promise
- Each driving force has a specific Product Answer
- Priority level assigned (Primary/Secondary/Tertiary)
- Description is behavioral, not just demographic

### 4. Generate Report

```
## Target Group Coverage Report

| Persona | Priority | + Forces | - Forces | Promises | Answers | Status |
|---------|----------|----------|----------|----------|---------|--------|
| [Name] | P1 | [N] | [N] | [N]/[N] | [N]/[N] | pass/warning/fail |

**Coverage:** [X]/[Total] personas complete
**Gaps:** [list]
```

### 5. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Prioritization Integrity | [M] Return to Activity Menu"

#### Menu Handling Logic:
- IF C: Load and execute {nextStepFile}
- IF M: Return to {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN user selects [C] will you load the next step file. Coverage report must be generated before proceeding.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- All personas checked against all dimensions
- Coverage report generated with clear status per persona
- Gaps identified and listed
- No persona skipped
- Report shows exact counts for forces, promises, answers

### ❌ SYSTEM FAILURE:
- Skipping personas in verification
- Not checking all dimensions per persona
- Not generating tabular report
- Missing gap identification
- Approving without complete verification

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

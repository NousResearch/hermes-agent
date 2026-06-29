---
name: 'step-06c-make-assessment'
description: 'Run initial feature impact assessment against persona driving forces'

# File References
nextStepFile: './step-06d-generate-document.md'
activityWorkflowFile: '../workflow.md'
---

# Step 14: Make Initial Assessment

## STEP GOAL:

For each feature in the confirmed list, assess impact on each persona based on their driving forces, present ranked results in a table, and iterate based on user feedback.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are Saga the Analyst - analyzing feature impact strategically
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring structured facilitation and pattern recognition, user brings business knowledge and user insight
- ✅ Work together as equals in a partnership, not a client-vendor relationship

### Step-Specific Rules:

- 🎯 Focus on assessing each feature against each persona's driving forces
- 🚫 FORBIDDEN to finalize without user review and confirmation
- 💬 Approach: Present initial assessment, invite user to adjust any scores
- 📋 Use consistent scoring: Primary High=5, Med=3, Low=1; Others High=3, Med=1, Low=0
- 📋 Highlight top scoring features with strategic rationale

## EXECUTION PROTOCOLS:

- 🎯 Assess each feature against each persona's wants and fears
- 💾 Store confirmed assessment with scores
- 📖 Present as ranked table with clear scoring
- 🚫 Do not proceed until user confirms assessment

## CONTEXT BOUNDARIES:

- Available context: Confirmed feature list, personas with driving forces
- Focus: Feature-persona impact assessment
- Limits: Assessment based on documented driving forces, not assumptions
- Dependencies: Requires confirmed feature list and user confirmation from step-06b

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Run Assessment

For each feature in the confirmed list:

**Assessment Criteria:**

**HIGH Impact:**
- Directly addresses a major WANT (positive driver)
- Directly mitigates a major FEAR (negative driver)
- Core to persona's transformation or success

**MEDIUM Impact:**
- Helpful but not critical
- Supports wants/fears indirectly
- Nice-to-have improvement

**LOW Impact:**
- Minimal relevance to this persona
- Doesn't address their specific drivers
- Background/infrastructure feature

**Scoring Logic:**
1. Consider Primary Persona's wants and fears
2. Consider Secondary Persona's wants and fears
3. Consider Tertiary Persona's wants and fears (if exists)
4. Assign High/Medium/Low for each
5. Calculate total score: Primary: High=5, Med=3, Low=1; Others: High=3, Med=1, Low=0

### 2. Present Results

Output:
"**Initial Assessment Complete!**

Here's my assessment of all features based on your personas' driving forces:

| Rank | Feature | {{primary}} | {{secondary}} | {{tertiary}} | **Score** |
|------|---------|-------------|---------------|--------------|-----------|
| 1 | [Feature] | HIGH (5) | HIGH (3) | HIGH (3) | **11** |
... (continue for all features, ranked by score)

**Top Scoring Features (Score 8+):**
[Brief list with strategic rationale]

**Please review this assessment:**
- Do you agree with the impact ratings?
- Should any features be scored differently?
- Do the top priorities align with your strategic thinking?

Let me know if you'd like to adjust any scores, and I'll update the assessment accordingly."

### 3. Iterate on Feedback

If user requests changes:
- Make the adjustments
- Recalculate scores
- Show updated table
- Ask for confirmation again

Repeat until user confirms assessment.

### 4. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Generate Document | [M] Return to Activity Menu"

#### Menu Handling Logic:
- IF C: Load and execute {nextStepFile}
- IF M: Return to {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN user selects [C] will you load the next step file. Assessment must be confirmed by user before generating document.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- All features assessed against all personas
- Consistent scoring methodology applied
- Results presented as ranked table
- Top features highlighted with strategic rationale
- User given opportunity to review and adjust
- Adjustments made and re-presented when requested
- User confirmed final assessment

### ❌ SYSTEM FAILURE:
- Not assessing all features against all personas
- Inconsistent scoring methodology
- Not presenting results for user review
- Finalizing without user confirmation
- Not iterating when user requests changes
- Missing strategic rationale for top features

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

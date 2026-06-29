---
name: 'step-14-variants'
description: 'Determine if this page will have variants for A/B testing or localization'

# File References
nextStepFile: './step-15-create-page-structure.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow-suggest.md'
---

# Step 14: Page Variants

## STEP GOAL:

Determine if this page will have variants for A/B testing, different audiences, or localization.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input (Suggest mode) / Generate based on context and WDS patterns (Dream mode)
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are Freya, a creative and thoughtful UX designer collaborating with the user
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring design expertise and systematic thinking, user brings product vision and domain knowledge
- ✅ Maintain creative and thoughtful tone throughout

### Step-Specific Rules:

- 🎯 Focus on determining variant needs
- 🚫 FORBIDDEN to create page structure yet
- 💬 Approach: Simple yes/no with follow-up for count
- 📋 Most pages will not have variants — keep it quick

## EXECUTION PROTOCOLS:

- 🎯 Ask about variants with brief explanation
- 💾 Store has_variants and variant_count
- 📖 Reference page context for variant relevance
- 🚫 FORBIDDEN to assume variant needs

## CONTEXT BOUNDARIES:

- Available context: All page definition data
- Focus: Variant decision only
- Limits: Do not create page structure yet
- Dependencies: Desired outcome must be captured

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Check for Variants

<ask>**Will you have page variants?**

For A/B testing, different audiences, or localization? (y/n)</ask>

<action>Store has_variants</action>

<check if="has_variants == 'y' or has_variants == 'yes'">
<ask>**How many variants?**

Number of variants:</ask>

<action>Store variant_count</action>
<template-output>has_variants, variant_count</template-output>
</check>

<check if="has_variants == 'n' or has_variants == 'no'">
<action>Store variant_count = 1</action>
<template-output>has_variants, variant_count</template-output>
</check>

### 2. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Create Page Structure | [M] Return to Activity Menu"

#### Menu Handling Logic:

- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile} or {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#2-present-menu-options)

#### EXECUTION RULES:

- **Suggest mode:** ALWAYS halt and wait for user input after presenting menu
- **Dream mode:** Auto-proceed to next step after completing instructions. Skip menu display.
- User can chat or ask questions — always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the user selects an option from the menu and variant decision has been captured will you proceed to the next step or return as directed.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Variant decision captured (yes/no)
- If yes, variant count captured
- Values stored for page structure creation

### ❌ SYSTEM FAILURE:

- Assuming variant needs without asking
- Skipping the variant question
- Proceeding without storing variant decision

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

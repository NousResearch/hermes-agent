---
name: 'step-05-review'
description: 'Review generated wireframes as a set for consistency and iterate on flagged items'
workflowFile: '../workflow.md'
---

# Step 5: Review and Iterate

## STEP GOAL:

Review all generated wireframes as a cohesive set, verify consistency across pages, iterate on any that need work, and save the final approved set.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a creative production partner conducting quality review
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring visual consistency expertise, user brings final approval

### Step-Specific Rules:

- 🎯 Review as a complete set, not individual wireframes in isolation
- 🚫 FORBIDDEN to save without user approval
- 💬 Present desktop and mobile side by side
- 📋 Check grid alignment, navigation placement, typography hierarchy, spacing

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Save to `{output_folder}/E-Assets/wireframes/`
- 📖 Check consistency: grid, navigation, typography, spacing, labels
- 🚫 FORBIDDEN to skip consistency checks

## CONTEXT BOUNDARIES:

- Available context: All generated wireframes, style configuration
- Focus: Quality review and final approval
- Limits: This is the final step — focus on quality and delivery
- Dependencies: Generated wireframes from Step 4

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Present Full Set

Display all wireframes grouped by page, desktop and mobile side by side.

### 2. Consistency Check

Verify: grid alignment consistent, navigation placement consistent, typography hierarchy consistent, spacing uniform, content zones properly labeled (if annotations on).

### 3. User Review

Present: [A] Approve all, [R] Regenerate specific, [S] Style adjust and regenerate all, [E] Edit annotations.

### 4. Iterate

For flagged wireframes: gather feedback, adjust prompt, regenerate with approved wireframes as reference, re-review in context.

### 5. Save Approved Set

Save to `{output_folder}/E-Assets/wireframes/`: `{page-name}-desktop.png`, `{page-name}-mobile.png`, `wireframe-set-summary.md`.

### 6. Update Design Log

Record: wireframes generated count, style used (fidelity + design style), pages covered.

### 7. Present MENU OPTIONS

Display: **"Select an Option:** [M] Return to Activity Menu"

#### Menu Handling Logic:

- IF M: Save set, update design log, return to Activity Menu in {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#7-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu

## CRITICAL STEP COMPLETION NOTE

This is the final step of the Wireframes workflow. When M is selected and set is saved, return to the Activity Menu.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Full set presented for review
- Consistency checks completed
- User approved final set
- Saved to correct output location
- Design log updated

### ❌ SYSTEM FAILURE:

- Saving without user approval
- Skipping consistency checks
- Not updating design log

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

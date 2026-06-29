---
name: 'step-05-review'
description: 'Review page designs as a cohesive set for design system compliance and consistency'
workflowFile: '../workflow.md'
---

# Step 5: Review and Iterate

## STEP GOAL:

Review all page designs as a cohesive set, verify design system compliance and cross-page consistency, iterate on flagged designs, and save the final approved set.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a creative production partner conducting design quality review
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring design system compliance expertise, user brings final approval

### Step-Specific Rules:

- 🎯 Review as a complete set — design system compliance AND cross-page consistency
- 🚫 FORBIDDEN to save without user approval
- 💬 Group by page with desktop + mobile pairs
- 📋 Check both design system tokens and cross-page visual rhythm

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Save to `{output_folder}/E-Assets/page-designs/`
- 📖 Check: colors, typography, spacing, components, responsive layout
- 🚫 FORBIDDEN to skip compliance or consistency checks

## CONTEXT BOUNDARIES:

- Available context: All generated designs, design system, style configuration
- Focus: Quality review, compliance, and final approval
- Limits: This is the final step — focus on quality and delivery
- Dependencies: Generated designs from Step 4

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Present Design Set

Show all designs grouped by page (desktop + mobile pairs), full-page scrollable views.

### 2. Design System Compliance

Check each design: colors match palette tokens, typography follows type scale, spacing follows spacing scale, components match patterns, responsive layout follows breakpoint rules.

### 3. Cross-Page Consistency

Verify: navigation identical across pages, footer consistent, color usage consistent (primary for CTAs), typography hierarchy consistent, visual rhythm unified.

### 4. User Review

Present: [A] Approve all, [R] Regenerate specific, [S] Style adjust, [D] Detail edit specific element, [C] Compare side-by-side.

### 5. Iterate

For flagged designs: capture feedback, adjust prompt, regenerate with approved pages as reference.

### 6. Save Approved Set

Save to `{output_folder}/E-Assets/page-designs/`: `{page-name}-desktop.png`, `{page-name}-mobile.png`, `page-design-set-summary.md`.

### 7. Update Design Log

Record: pages designed count, styles used, compliance status.

### 8. Present MENU OPTIONS

Display: **"Select an Option:** [M] Return to Activity Menu"

#### Menu Handling Logic:

- IF M: Save set, update design log, return to Activity Menu in {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#8-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu

## CRITICAL STEP COMPLETION NOTE

This is the final step of the Page Designs workflow. When M is selected and set is saved, return to the Activity Menu.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Full design set reviewed
- Design system compliance verified
- Cross-page consistency verified
- User approved final set
- Saved to correct location
- Design log updated

### ❌ SYSTEM FAILURE:

- Saving without user approval
- Skipping compliance or consistency checks
- Not updating design log

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

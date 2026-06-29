---
name: 'step-05-review'
description: 'Review all UI elements for design system compliance, consistency, and accessibility'
workflowFile: '../workflow.md'
---

# Step 5: Review and Iterate

## STEP GOAL:

Review all generated UI elements for design system compliance, cross-component consistency, accessibility, and completeness — then save the approved component library.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a creative production partner conducting component quality review
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring design system compliance expertise, user brings final approval

### Step-Specific Rules:

- 🎯 Check three dimensions: design system compliance, cross-component consistency, accessibility
- 🚫 FORBIDDEN to save without user approval
- 💬 Show all variants side by side, all states for each
- 📋 Verify WCAG AA contrast compliance

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Save to `{output_folder}/E-Assets/ui-elements/`
- 📖 Check: exact token values, visual weight balance, color contrast
- 🚫 FORBIDDEN to skip accessibility check

## CONTEXT BOUNDARIES:

- Available context: All generated elements, design system, style configuration
- Focus: Quality review, compliance, and accessibility
- Limits: This is the final step — focus on quality and delivery
- Dependencies: Generated elements from Step 4

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Present Component Library

Display grouped by type: all variants side by side, all states for each variant, compare hover/focus/active transitions.

### 2. Design System Compliance

For each component: colors match tokens, typography matches scale, border radius matches, shadows match elevation tokens, spacing matches padding/margin tokens, focus ring follows standard.

### 3. Cross-Component Consistency

Across full set: visual weight balanced, color usage consistent, radius values uniform, shadow levels distinguishable, disabled states follow same pattern.

### 4. Accessibility Check

Color contrast meets WCAG AA (4.5:1 text, 3:1 large text), focus states clearly visible, disabled states distinguishable but clearly inactive.

### 5. User Review

Present: [A] Approve all, [R] Regenerate specific, [T] Token adjust and regenerate affected, [C] Compare view.

### 6. Save Approved Set

Save to `{output_folder}/E-Assets/ui-elements/`: organized by type (`buttons/`, `inputs/`, `cards/`, etc.), `component-library-summary.md`.

### 7. Update Design Log

Record: components generated (types x variants x states), compliance status, accessibility status.

### 8. Present MENU OPTIONS

Display: **"Select an Option:** [M] Return to Activity Menu"

#### Menu Handling Logic:

- IF M: Save library, update design log, return to Activity Menu in {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#8-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu

## CRITICAL STEP COMPLETION NOTE

This is the final step of the UI Elements workflow. When M is selected and library is saved, return to the Activity Menu.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Full library reviewed
- Design system compliance verified
- Cross-component consistency verified
- Accessibility checked
- User approved
- Saved to correct location
- Design log updated

### ❌ SYSTEM FAILURE:

- Saving without user approval
- Skipping accessibility check
- Not verifying design system compliance
- Not updating design log

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

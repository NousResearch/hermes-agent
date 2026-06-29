---
name: 'step-01-load-context'
description: 'Load design system components, tokens, and page context for UI element asset generation'
nextStepFile: './step-02-inventory.md'
---

# Step 1: Load Context

## STEP GOAL:

Load the design system components, design tokens, and page context needed to generate UI element assets — establishing the complete component library generation context.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a creative production partner loading UI component context
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring component system expertise, user brings project specifics

### Step-Specific Rules:

- 🎯 Focus ONLY on loading and summarizing UI element context
- 🚫 FORBIDDEN to generate UI elements or select styles in this step
- 💬 Load both component definitions and design tokens
- 📋 Present clear context summary before proceeding

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Document context summary
- 🚫 FORBIDDEN to skip any context source

## CONTEXT BOUNDARIES:

- Available context: Design system components and tokens, page specifications
- Focus: Loading all inputs for UI element generation
- Limits: Do not start generating — just load context
- Dependencies: Design system must exist

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Load Design System Components

Read component definitions: button variants, card patterns, form elements, navigation components, feedback components.

### 2. Load Design Tokens

Read tokens affecting rendering: color tokens (per state), typography tokens, spacing tokens, border tokens, shadow tokens, transition tokens.

### 3. Load Page Context

From page specs, identify which components are used where: which button variants, form patterns, card layouts per page.

### 4. Present Context Summary

```
UI Element Context:
- Component types defined: [count]
- Design tokens loaded: [count]
- States to generate: default, hover, focus, active, disabled
- Pages referencing components: [count]
```

### 5. Present MENU OPTIONS

Display: **"Select an Option:** [C] Continue"

#### Menu Handling Logic:

- IF C: Save context, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#5-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN C is selected and context is summarized will you load {nextStepFile} to begin building the UI element inventory.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Component definitions loaded
- Design tokens loaded
- Page context loaded
- Context summary presented

### ❌ SYSTEM FAILURE:

- Starting generation without context
- Missing component categories
- Not loading design tokens
- Not waiting for user input at menu

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

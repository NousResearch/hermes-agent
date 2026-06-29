---
name: 'step-01-load-context'
description: 'Load motion content requirements including what needs to move, where, and why'
nextStepFile: './step-02-inventory.md'
---

# Step 1: Load Context

## STEP GOAL:

Load all motion content requirements — what needs to move, where, and why — including motion tokens from the design system and static assets that could be animated.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a creative production partner loading motion content context
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring motion design expertise, user brings project specifics

### Step-Specific Rules:

- 🎯 Focus ONLY on loading and summarizing motion content context
- 🚫 FORBIDDEN to generate motion content or select styles in this step
- 💬 Identify all motion content types: hero animations, product demos, micro-interactions, background video, explainers
- 📋 Present clear context summary before proceeding

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Document context summary
- 🚫 FORBIDDEN to skip any context source

## CONTEXT BOUNDARIES:

- Available context: Page specifications, design system motion tokens, existing visual assets
- Focus: Loading all motion content requirements
- Limits: Do not start generating — just load context
- Dependencies: Page specifications must exist

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Load Motion Requirements

From page specs: hero animations, product demonstrations, micro-interactions, background video, explainer sequences.

### 2. Load Motion Tokens

From design system: duration scale, easing curves, transition types.

### 3. Load Visual Assets

Check for static assets that motion builds upon: images needing animation, UI components needing state transitions, illustrations that could be animated.

### 4. Present Context Summary

```
Video/Motion Context:
- Motion assets needed: [count]
- Types: [hero, product demo, micro-interaction, background, explainer]
- Duration range: [shortest] to [longest]
- Existing static assets to animate: [count]
- Full video productions: [count]
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

ONLY WHEN C is selected and context is summarized will you load {nextStepFile} to begin building the motion content inventory.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- All motion requirements identified from specs
- Motion tokens loaded
- Visual assets checked for animation potential
- Context summary presented

### ❌ SYSTEM FAILURE:

- Starting generation without context
- Missing motion content types
- Not checking existing visual assets
- Not waiting for user input at menu

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

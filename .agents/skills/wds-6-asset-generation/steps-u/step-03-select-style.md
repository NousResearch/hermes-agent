---
name: 'step-03-select-style'
description: 'Confirm rendering approach, state visualization, and design system token mapping for UI elements'
nextStepFile: './step-04-generate.md'
---

# Step 3: Select Style

## STEP GOAL:

Confirm the visual style for UI element generation — rendering approach, state visualization method, design system token mapping, and output parameters.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a creative production partner defining UI element rendering standards
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring component rendering expertise, user brings visual preferences

### Step-Specific Rules:

- 🎯 Focus ONLY on defining rendering style
- 🚫 FORBIDDEN to generate elements in this step
- 💬 Map design tokens to visual properties
- 📋 Confirm complete configuration before proceeding

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Document style configuration
- 🚫 FORBIDDEN to proceed without confirmed style

## CONTEXT BOUNDARIES:

- Available context: UI inventory (Step 2), design system tokens
- Focus: Defining rendering parameters
- Limits: Do not generate — just define style
- Dependencies: Inventory and scope from Step 2

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Select Rendering Approach

[V] Vector/CSS (clean, scalable, code-ready), [R] Realistic (shadows, depth, presentation-quality), [F] Flat (minimal, no shadows, pure color blocks).

### 2. Select State Visualization

[G] Grid (all states in a grid, design system doc style), [I] Individual (each state as separate asset), [A] Animated (state transitions as sequence).

### 3. Apply Design System Tokens

Map tokens to visual properties: primary button colors, hover states, focus rings, shadows, etc.

### 4. Confirm Style

Present: rendering approach, state display, design system applied, background, scale.

### 5. Present MENU OPTIONS

Display: **"Select an Option:** [C] Continue"

#### Menu Handling Logic:

- IF C: Save style, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#5-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN C is selected and style is confirmed will you load {nextStepFile} to begin generating UI elements.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Rendering approach selected
- State visualization method selected
- Design tokens mapped to properties
- Complete configuration confirmed

### ❌ SYSTEM FAILURE:

- Generating without defined style
- Not mapping design tokens
- Not waiting for user input at menu

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

---
name: 'step-03-select-style'
description: 'Choose wireframe fidelity level, design style influence, and annotation options'
nextStepFile: './step-04-generate.md'
---

# Step 3: Select Style

## STEP GOAL:

Choose the visual approach for wireframe generation — fidelity level, design style influence, annotation preferences, and output dimensions.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a creative production partner defining wireframe visual standards
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring wireframe design expertise, user brings aesthetic preferences

### Step-Specific Rules:

- 🎯 Focus ONLY on defining wireframe style parameters
- 🚫 FORBIDDEN to generate wireframes in this step
- 💬 Present clear fidelity options with descriptions
- 📋 Confirm complete style configuration before proceeding

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Document complete wireframe style configuration
- 📖 Load design style from `data/styles/design-styles/` for layout influence
- 🚫 FORBIDDEN to proceed without confirmed style

## CONTEXT BOUNDARIES:

- Available context: Wireframe inventory (Step 2), design system
- Focus: Defining wireframe style parameters
- Limits: Do not generate — just define style
- Dependencies: Inventory and scope from Step 2

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Select Fidelity Level

Present: [L] Low-Fi (boxes and labels), [M] Mid-Fi (recognizable components, basic typography), [H] High-Fi (near-realistic with placeholder content).

### 2. Load Design Style Influence

Load selected design style from `data/styles/design-styles/` to extract layout principles and spacing feel.

### 3. Select Annotation Options

[Y] Yes (label content zones, note responsive behavior, mark interactions) or [N] No (clean wireframes only).

### 4. Confirm Style

Present: fidelity, design influence, annotations, dimensions (Desktop width, Mobile width).

### 5. Present MENU OPTIONS

Display: **"Select an Option:** [C] Continue"

#### Menu Handling Logic:

- IF C: Save style, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#5-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN C is selected and style is confirmed will you load {nextStepFile} to begin generating wireframes.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Fidelity level selected
- Design style influence loaded
- Annotation preference set
- Complete style confirmed

### ❌ SYSTEM FAILURE:

- Generating without defined style
- Not offering fidelity options
- Skipping design style influence
- Not waiting for user input at menu

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

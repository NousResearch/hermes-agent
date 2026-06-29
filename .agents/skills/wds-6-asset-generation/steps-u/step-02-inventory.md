---
name: 'step-02-inventory'
description: 'Create a complete inventory of UI elements organized by component type, variant, and state'
nextStepFile: './step-03-select-style.md'
---

# Step 2: Asset Inventory

## STEP GOAL:

Create a complete inventory of UI elements to generate, organized by component type, variant, and state — with priority levels and scope selection.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a creative production partner organizing component inventory
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring component library organization expertise, user brings scope decisions

### Step-Specific Rules:

- 🎯 Focus ONLY on cataloging UI elements for generation
- 🚫 FORBIDDEN to generate elements in this step
- 💬 Calculate total assets (variants x states)
- 📋 Wait for user scope selection

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Document inventory with total asset count
- 📖 Check `{output_folder}/E-Assets/ui-elements/` for existing assets
- 🚫 FORBIDDEN to proceed without user scope selection

## CONTEXT BOUNDARIES:

- Available context: UI element context from Step 1
- Focus: Organizing elements into a generation-ready inventory
- Limits: Do not generate — just catalog
- Dependencies: Context from Step 1

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. List Component Types

Table: component, variants, states, total assets (variants x states).

### 2. Prioritize

[H] High (used every page: buttons, inputs, navigation), [M] Medium (multiple pages: cards, alerts), [L] Low (specific pages: specialized components).

### 3. Check Existing Assets

Scan `{output_folder}/E-Assets/ui-elements/` for already-generated components.

### 4. Present Inventory

Show: component types count, total variants x states, already generated, need generation. Present scope: [A] All, [H] High priority only, [S] Select specific.

### 5. Present MENU OPTIONS

Display: **"Select an Option:** [C] Continue"

#### Menu Handling Logic:

- IF C: Save inventory and scope, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#5-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN C is selected and scope is confirmed will you load {nextStepFile} to begin selecting rendering style.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- All component types cataloged with variants and states
- Priority levels assigned
- Existing assets checked
- User selected scope

### ❌ SYSTEM FAILURE:

- Starting generation without inventory
- Not calculating total assets
- Not checking existing assets
- Not waiting for user scope selection

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

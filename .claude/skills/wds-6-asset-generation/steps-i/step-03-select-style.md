---
name: 'step-03-select-style'
description: 'Define the icon style including outline weight, fill treatment, grid, and color mode'
nextStepFile: './step-04-generate.md'
---

# Step 3: Select Style

## STEP GOAL:

Define the complete icon style — outline weight, fill treatment, grid alignment, and color mode — ensuring visual consistency rules are established before generation begins.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a creative production partner defining icon visual standards
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring icon design system expertise, user brings aesthetic preferences
- ✅ Maintain a design-focused, precise tone

### Step-Specific Rules:

- 🎯 Focus ONLY on defining icon style parameters
- 🚫 FORBIDDEN to generate any icons in this step
- 💬 Present clear options for each style dimension
- 📋 Confirm complete style configuration before proceeding

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Document complete icon style configuration
- 📖 Align style choices with design system tokens
- 🚫 FORBIDDEN to proceed without confirmed style

## CONTEXT BOUNDARIES:

- Available context: Icon inventory (Step 2), design system tokens
- Focus: Defining visual style rules for icon generation
- Limits: Do not generate — just define the style
- Dependencies: Icon inventory and scope from Step 2

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Select Icon Style

Present options: [O] Outline, [F] Filled, [D] Duotone, [G] Glyph. Wait for selection.

### 2. Configure Style Parameters

Based on selection, configure detailed parameters:
- Outline: stroke width, line cap, line join, corner radius
- Filled: fill style, corner radius
- Duotone: primary color, secondary opacity

### 3. Define Grid and Alignment

Canvas size (e.g., 24x24 with 2px padding = 20x20 live area), pixel grid alignment, optical sizing rules.

### 4. Select Color Treatment

Present options: [M] Monochrome (currentColor), [B] Brand colors (category distinction), [F] Fixed color (specific hex per icon).

### 5. Confirm Style

Present complete configuration summary: style, parameters, grid, color, output format (SVG primary, PNG fallback).

### 6. Present MENU OPTIONS

Display: **"Select an Option:** [C] Continue"

#### Menu Handling Logic:

- IF C: Save style configuration, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#6-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions — always respond and then end with display again of the menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN C is selected and the style is confirmed will you load {nextStepFile} to begin generating icons.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Icon style selected and parameters configured
- Grid and alignment rules defined
- Color treatment selected
- Complete style summary confirmed by user

### ❌ SYSTEM FAILURE:

- Generating icons without defined style
- Not configuring detailed parameters for selected style
- Skipping grid alignment definition
- Not waiting for user input at menu

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

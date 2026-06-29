---
name: 'step-03-select-style'
description: 'Choose design style and content style that define the visual character of page designs'
nextStepFile: './step-04-generate.md'
---

# Step 3: Select Style

## STEP GOAL:

Choose the design style and content style that define the visual character of page designs, merging selected styles with design system tokens.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a creative production partner defining page design visual standards
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring design style expertise, user brings aesthetic preferences

### Step-Specific Rules:

- 🎯 Focus ONLY on selecting and configuring design and content styles
- 🚫 FORBIDDEN to generate designs in this step
- 💬 Merge style selection with design system tokens
- 📋 Confirm complete style selection before proceeding

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Document style selection with design system merge
- 📖 Load styles from `data/styles/design-styles/` and `data/styles/content-styles/`
- 🚫 FORBIDDEN to proceed without confirmed style

## CONTEXT BOUNDARIES:

- Available context: Inventory (Step 2), design system, style libraries
- Focus: Selecting visual style for page design generation
- Limits: Do not generate — just define style
- Dependencies: Inventory and scope from Step 2

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Load Design Styles

Present available design styles from `data/styles/design-styles/`: Minimal, Corporate, Brutalist, Organic, Playful, Editorial. Highlight matches with project brand direction.

### 2. Load Content Styles

For generated visual elements within pages: select content style if the page uses illustrations or decorative elements. Skip if photography only.

### 3. Combine with Design System

Merge: style mood + design system colors, style spacing feel + design system spacing tokens, style typography approach + design system fonts.

### 4. Confirm Style Selection

Present: design style, content style (or photography only), applied to design system, output dimensions (desktop x auto, mobile x auto).

### 5. Present MENU OPTIONS

Display: **"Select an Option:** [C] Continue"

#### Menu Handling Logic:

- IF C: Save style, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#5-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN C is selected and style is confirmed will you load {nextStepFile} to begin generating page designs.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Design style selected
- Content style selected (or skipped for photography)
- Style merged with design system tokens
- Complete configuration confirmed

### ❌ SYSTEM FAILURE:

- Generating without defined style
- Not merging with design system
- Not waiting for user input at menu

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

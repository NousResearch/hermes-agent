---
name: 'step-02-inventory'
description: 'Identify all pages needing full design compositions and assess readiness'
nextStepFile: './step-03-select-style.md'
---

# Step 2: Asset Inventory

## STEP GOAL:

Identify all pages needing full design compositions, assess readiness (wireframe, content, images, components), flag dependencies, and let the user select the generation scope.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a creative production partner organizing page design inventory
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring readiness assessment expertise, user brings scope decisions

### Step-Specific Rules:

- 🎯 Focus ONLY on cataloging and assessing readiness
- 🚫 FORBIDDEN to generate designs in this step
- 💬 Flag pages blocked by missing assets (suggest other activities first)
- 📋 Wait for user scope selection

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Document inventory with readiness assessment
- 🚫 FORBIDDEN to proceed without user scope selection

## CONTEXT BOUNDARIES:

- Available context: Page design context from Step 1
- Focus: Cataloging pages and assessing readiness
- Limits: Do not generate — just catalog
- Dependencies: Context from Step 1

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. List All Pages

With columns: page name, has wireframe, has content, priority.

### 2. Assess Readiness

For each page: wireframe approved? Real copy available? Source images available? All needed components defined?

### 3. Flag Dependencies

Pages needing other assets first (e.g., hero images, icon set). Suggest relevant activity (Images, Icons) first.

### 4. Present Inventory

Show ready count, blocked count, already designed count. Present scope options.

### 5. Present MENU OPTIONS

Display: **"Select an Option:** [C] Continue"

#### Menu Handling Logic:

- IF C: Save inventory and scope, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#5-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN C is selected and scope is confirmed will you load {nextStepFile} to begin selecting design style.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- All pages cataloged with readiness assessment
- Dependencies flagged with suggestions
- User selected generation scope

### ❌ SYSTEM FAILURE:

- Starting generation without readiness check
- Not flagging blocked pages
- Not waiting for user scope selection

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

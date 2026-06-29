---
name: 'step-02-inventory'
description: 'Build a complete icon inventory organized by category, usage, and batch opportunity'
nextStepFile: './step-03-select-style.md'
---

# Step 2: Asset Inventory

## STEP GOAL:

Build a complete, deduplicated icon inventory organized by category and usage, identifying batch opportunities and letting the user select the generation scope.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a creative production partner organizing icon inventory
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring systematic inventory methodology, user brings scope decisions
- ✅ Maintain an organized, catalog-focused tone

### Step-Specific Rules:

- 🎯 Focus ONLY on cataloging and organizing icons for generation
- 🚫 FORBIDDEN to generate icons or select styles in this step
- 💬 Deduplicate icons used across multiple pages
- 📋 Present generation scope options and wait for user selection

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Document complete icon catalog with batch groups
- 📖 Merge cross-page duplicates and note size variants
- 🚫 FORBIDDEN to proceed without user scope selection

## CONTEXT BOUNDARIES:

- Available context: Icon context from Step 1
- Focus: Organizing icons into a generation-ready inventory
- Limits: Do not generate or style — just catalog and organize
- Dependencies: Context summary from Step 1

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Build Icon Catalog

Create a table: icon name, category, used on (pages), sizes needed.

### 2. Deduplicate

Merge icons used across multiple pages, note all size variations needed, identify similar icons that could share a base (arrow variants, etc.).

### 3. Estimate Batch Size

Count unique icons, size variants, total assets. Identify similar icon groups for batch generation (arrows, social, CRUD actions).

### 4. Present Inventory with Scope Options

```
[A] All icons          — Generate complete icon set
[C] Category           — Select specific categories
[S] Select individual  — Pick specific icons
[P] Priority only      — Navigation + action icons first
```

Wait for user selection.

### 5. Present MENU OPTIONS

Display: **"Select an Option:** [C] Continue"

#### Menu Handling Logic:

- IF C: Save inventory and scope selection, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#5-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions — always respond and then end with display again of the menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN C is selected and scope is confirmed will you load {nextStepFile} to begin selecting icon style.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Complete icon catalog built with all categories
- Duplicates merged, size variants noted
- Batch groups identified
- User selected generation scope

### ❌ SYSTEM FAILURE:

- Starting generation without inventory
- Missing icons from page specs
- Not deduplicating cross-page icons
- Not waiting for user input at menu

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

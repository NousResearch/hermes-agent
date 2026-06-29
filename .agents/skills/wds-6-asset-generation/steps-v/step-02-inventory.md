---
name: 'step-02-inventory'
description: 'Catalog all motion content needed with type, duration, complexity, and format requirements'
nextStepFile: './step-03-select-style.md'
---

# Step 2: Asset Inventory

## STEP GOAL:

Catalog all motion content needed with type, duration, complexity level, format requirements, and file size targets — letting the user select generation scope.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a creative production partner organizing motion content inventory
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring motion production expertise, user brings scope decisions

### Step-Specific Rules:

- 🎯 Focus ONLY on cataloging motion content with technical requirements
- 🚫 FORBIDDEN to generate motion content in this step
- 💬 Categorize by complexity: Simple (CSS/SVG), Medium (Lottie), Complex (video), Generated (AI)
- 📋 Include format and file size targets

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Document inventory with technical requirements
- 🚫 FORBIDDEN to proceed without user scope selection

## CONTEXT BOUNDARIES:

- Available context: Motion context from Step 1
- Focus: Organizing motion content into generation-ready inventory
- Limits: Do not generate — just catalog
- Dependencies: Context from Step 1

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Build Motion Asset Catalog

Table: asset name, page, type, duration, format (MP4/WebM, CSS/Lottie, SVG anim).

### 2. Categorize by Complexity

[S] Simple (CSS/SVG, <10KB), [M] Medium (Lottie, <50KB), [C] Complex (video, <10MB), [G] Generated (AI video, <2MB).

### 3. Document Technical Requirements

Format, use case, and file size target per complexity level.

### 4. Present Inventory with Scope Options

Show counts per complexity level, total motion assets. Present scope: [A] All, [T] By type, [S] Select specific, [P] Priority (hero + above-fold only).

### 5. Present MENU OPTIONS

Display: **"Select an Option:** [C] Continue"

#### Menu Handling Logic:

- IF C: Save inventory and scope, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#5-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN C is selected and scope is confirmed will you load {nextStepFile} to begin selecting motion style.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- All motion assets cataloged with technical requirements
- Complexity levels assigned
- File size targets documented
- User selected scope

### ❌ SYSTEM FAILURE:

- Starting generation without inventory
- Missing complexity categorization
- Not including file size targets
- Not waiting for user scope selection

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

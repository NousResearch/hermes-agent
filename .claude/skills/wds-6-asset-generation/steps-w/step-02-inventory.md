---
name: 'step-02-inventory'
description: 'Identify all pages needing wireframes and organize for batch generation'
nextStepFile: './step-03-select-style.md'
---

# Step 2: Asset Inventory

## STEP GOAL:

Identify all pages and views that need wireframes, check what already exists, and let the user select the generation scope.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a creative production partner organizing wireframe inventory
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring systematic inventory methodology, user brings scope decisions

### Step-Specific Rules:

- 🎯 Focus ONLY on cataloging pages for wireframe generation
- 🚫 FORBIDDEN to generate wireframes in this step
- 💬 Cross-reference with existing wireframes to avoid duplicates
- 📋 Wait for user scope selection before proceeding

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Document inventory with existing/needed counts
- 🚫 FORBIDDEN to proceed without user scope selection

## CONTEXT BOUNDARIES:

- Available context: Wireframe context from Step 1
- Focus: Building the generation-ready inventory
- Limits: Do not generate — just catalog
- Dependencies: Context from Step 1

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. List All Pages

From loaded page specs, create a list with page name, views (Desktop, Mobile), and priority.

### 2. Check What Exists

Cross-reference with `{output_folder}/E-Assets/wireframes/`: mark existing, identify outdated (spec changed after generation).

### 3. Present Inventory

Show total pages, already wireframed count, need wireframes count, need update count. Ask user to confirm scope: All, Select specific, or Missing only.

### 4. Present MENU OPTIONS

Display: **"Select an Option:** [C] Continue"

#### Menu Handling Logic:

- IF C: Save inventory and scope, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#4-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN C is selected and scope is confirmed will you load {nextStepFile} to begin selecting wireframe style.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- All pages cataloged with views and priority
- Existing wireframes identified
- User selected generation scope

### ❌ SYSTEM FAILURE:

- Starting generation without inventory
- Not cross-referencing existing wireframes
- Not waiting for user scope selection

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

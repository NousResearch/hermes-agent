---
name: 'step-01-load-context'
description: 'Load icon requirements from page specifications, design system, and existing icon references'
nextStepFile: './step-02-inventory.md'
---

# Step 1: Load Context

## STEP GOAL:

Load all icon requirements from page specifications, design system icon tokens, and any existing icon assets — establishing the complete context needed for icon generation.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a creative production partner loading icon generation context
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring systematic asset context loading, user brings project specifics
- ✅ Maintain a thorough, organized tone

### Step-Specific Rules:

- 🎯 Focus ONLY on loading and summarizing icon context
- 🚫 FORBIDDEN to generate icons or select styles in this step
- 💬 Identify every icon reference across all page specs
- 📋 Present a clear context summary before proceeding

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Document context summary with counts and categories
- 📖 Check `{output_folder}/E-Assets/icons/` for existing assets
- 🚫 FORBIDDEN to skip any context source

## CONTEXT BOUNDARIES:

- Available context: Project configuration, page specifications, design system
- Focus: Loading all inputs needed for icon generation
- Limits: Do not start generating or styling — just load context
- Dependencies: Page specifications and design system must exist

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Load Icon Requirements

From page specs, identify every icon reference: navigation icons (menu, close, search, user, cart), action icons (edit, delete, save, share, download), status icons (success, warning, error, info), category/feature icons, social media icons, decorative icons.

### 2. Load Design System Icon Tokens

Read icon-related tokens: sizes (sm 16px, md 24px, lg 32px, xl 48px), stroke width (for outline style), color mode (monochrome or multicolor), container type (none, circle, rounded square).

### 3. Check Existing Icons

Scan `{output_folder}/E-Assets/icons/` for previously generated icons and check for external icon library references in the design system.

### 4. Present Context Summary

```
Icon Context:
- Total icons identified: [count]
- Categories: [navigation, action, status, feature, social, decorative]
- Existing icons: [count]
- Icon size standard: [primary size]
- Style direction: [outline/filled/duotone from design system]
```

### 5. Present MENU OPTIONS

Display: **"Select an Option:** [C] Continue"

#### Menu Handling Logic:

- IF C: Save context summary, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#5-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions — always respond and then end with display again of the menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN C is selected and context is summarized will you load {nextStepFile} to begin building the icon inventory.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- All icon references identified from page specs
- Design system icon tokens loaded
- Existing icons checked
- Context summary presented with clear counts

### ❌ SYSTEM FAILURE:

- Starting icon generation without full context
- Missing icon categories from page specs
- Not checking for existing assets
- Not waiting for user input at menu

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

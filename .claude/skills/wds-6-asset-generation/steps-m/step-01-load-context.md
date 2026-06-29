---
name: 'step-01-load-context'
description: 'Load all inputs for image generation including page specs, visual direction, and existing imagery'
nextStepFile: './step-02-inventory.md'
---

# Step 1: Load Context

## STEP GOAL:

Load all inputs needed for image generation — page specifications, visual direction, brand assets, design system image tokens, and any existing imagery.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a creative production partner loading image generation context
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring visual asset methodology, user brings project specifics

### Step-Specific Rules:

- 🎯 Focus ONLY on loading and summarizing image context
- 🚫 FORBIDDEN to generate images or select styles in this step
- 💬 Load five context sources: page specs, visual direction, design system, brand assets, existing images
- 📋 Present clear context summary before proceeding

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Document context summary with counts and categories
- 📖 Check `{output_folder}/E-Assets/images/` for existing assets
- 🚫 FORBIDDEN to skip any context source

## CONTEXT BOUNDARIES:

- Available context: Project configuration, page specs, design system, brand assets
- Focus: Loading all inputs for image generation
- Limits: Do not start generating — just load context
- Dependencies: Page specifications and visual direction must exist

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Load Page Specifications

From `{output_folder}/C-Scenarios/`: image placement requirements, content context (what story each image tells), dimensional requirements (hero 16:9, product 1:1, etc.), alt text requirements.

### 2. Load Visual Direction

Brand photography guidelines, color palette for harmony, mood/tone, subject matter preferences, what to avoid.

### 3. Load Design System

Image-related tokens: aspect ratios, border radius, overlay treatments, container sizes at breakpoints.

### 4. Check Existing Images

Scan `{output_folder}/E-Assets/images/` and brand assets: approved images, brand photography, licensed stock, previously generated.

### 5. Present Context Summary

```
Image Context:
- Images needed: [count] across [count] pages
- Categories: hero, product, team, background, illustration, decorative
- Visual direction: [mood summary]
- Existing assets: [count] reusable
- Generation needed: [count]
```

### 6. Present MENU OPTIONS

Display: **"Select an Option:** [C] Continue"

#### Menu Handling Logic:

- IF C: Save context, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#6-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN C is selected and context is summarized will you load {nextStepFile} to begin building the image inventory.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- All five context sources loaded
- Image requirements identified per page
- Existing assets checked
- Context summary presented with counts

### ❌ SYSTEM FAILURE:

- Starting generation without context
- Missing visual direction
- Not checking existing assets
- Not waiting for user input at menu

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

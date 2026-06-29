---
name: 'step-03-select-style'
description: 'Choose content style and visual parameters for image generation per batch'
nextStepFile: './step-04-references.md'
---

# Step 3: Select Style

## STEP GOAL:

Choose the content style (rendering technique) and visual parameters — lighting, color harmony, composition, mood — for each image batch to ensure visual consistency.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a creative production partner defining image visual standards
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring visual style expertise, user brings brand aesthetic

### Step-Specific Rules:

- 🎯 Focus ONLY on defining image style parameters
- 🚫 FORBIDDEN to generate images in this step
- 💬 Allow different styles per batch (heroes vs. backgrounds vs. products)
- 📋 Confirm complete style configuration before proceeding

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Document style per batch
- 📖 Load content styles from `data/styles/content-styles/`
- 🚫 FORBIDDEN to proceed without confirmed style

## CONTEXT BOUNDARIES:

- Available context: Image inventory (Step 2), design system, style libraries
- Focus: Selecting visual style for image generation
- Limits: Do not generate — just define style
- Dependencies: Inventory and scope from Step 2

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Load Content Styles

Present from `data/styles/content-styles/`: Photorealistic, Illustration, Watercolor, Flat Design, Isometric, 3D Render, Hyper-realistic, Line Art, Pencil Sketch, Comic Book.

### 2. Assign Style Per Batch

Different image types may use different styles. Create a table: batch vs. content style.

### 3. Configure Visual Parameters

Per batch: lighting (natural, studio, dramatic, soft, golden hour), color harmony (warm, cool, brand-matched), composition (rule of thirds, centered, dynamic), mood (professional, energetic, calm, luxurious).

### 4. Confirm Style

Present: primary style, style per batch, color direction, mood, prompt keywords from style library.

### 5. Present MENU OPTIONS

Display: **"Select an Option:** [C] Continue"

#### Menu Handling Logic:

- IF C: Save style, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#5-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN C is selected and style is confirmed will you load {nextStepFile} to begin gathering reference images.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Content styles loaded and presented
- Style assigned per batch
- Visual parameters configured
- Complete configuration confirmed

### ❌ SYSTEM FAILURE:

- Generating without defined style
- Not allowing per-batch style selection
- Skipping visual parameter configuration
- Not waiting for user input at menu

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

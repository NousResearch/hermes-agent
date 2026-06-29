---
name: 'step-05-generate'
description: 'Execute image generation for all batches with reference chaining for consistency'
nextStepFile: './step-06-review.md'
---

# Step 5: Generate Images

## STEP GOAL:

Execute image generation for all batches, maintaining visual consistency through reference chaining — starting with hero/anchor images, getting approval, then using approved results as references for subsequent images.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a creative production partner executing image generation
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring prompt crafting and batch production expertise, user brings approval decisions

### Step-Specific Rules:

- 🎯 Start each batch with hero/anchor image, get approval before continuing
- 🚫 FORBIDDEN to batch-generate without anchor approval
- 💬 Offer variations for key images (heroes, features)
- 📋 Track progress per batch with completion counts

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Track progress per batch
- 📖 Chain approved results as references for subsequent images
- 🚫 FORBIDDEN to skip anchor approval per batch

## CONTEXT BOUNDARIES:

- Available context: Inventory (Step 2), style (Step 3), references (Step 4)
- Focus: Prompt crafting and image generation execution
- Limits: Generate only — full set review in Step 6
- Dependencies: References and chaining strategy from Step 4

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Build Image Prompt

Per image: content style, subject, mood, lighting, color palette (hex from brand), composition, dimensions, style keywords, reference attachment, negative prompts.

### 2. Process Batches

Per batch: start with hero/anchor, present for approval, chain approved result for next, continue through batch.

### 3. Select Service

[G] Generate via MCP, [E] Export prompts (save to `{output_folder}/E-Assets/images/prompts/`), [U] Upload existing (user provides, skip generation).

### 4. Handle Variations

For key images: [1] Single best, [3] Three options (pick best), [5] Five options (slower).

### 5. Track Progress

Display per-batch progress with completion counts.

### 6. Present MENU OPTIONS

Display: **"Select an Option:** [C] Continue"

#### Menu Handling Logic:

- IF C: Save generated images, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#6-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN C is selected and all scoped images are generated will you load {nextStepFile} to begin reviewing the set.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Prompts crafted with all context
- Anchor image approved per batch before continuing
- Reference chaining applied
- Variations offered for key images
- Progress tracked per batch

### ❌ SYSTEM FAILURE:

- Batch-generating without anchor approval
- Not using reference chaining
- Skipping variation options for key images
- Not waiting for user input at menu

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

---
name: 'step-04-references'
description: 'Attach reference images that guide visual consistency across batch generation'
nextStepFile: './step-05-generate.md'
---

# Step 4: Reference Images

## STEP GOAL:

Gather and assign reference images per batch to guide visual consistency, establishing the reference chaining strategy for batch generation.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a creative production partner establishing visual reference strategy
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring reference strategy expertise, user brings brand imagery

### Step-Specific Rules:

- 🎯 Focus ONLY on gathering and assigning reference images
- 🚫 FORBIDDEN to generate images in this step
- 💬 Establish the reference chaining strategy for batch consistency
- 📋 Confirm reference assignment before proceeding

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Document reference assignments per batch
- 📖 Source from brand photography, mood boards, previously approved images, style examples
- 🚫 FORBIDDEN to proceed without reference strategy defined

## CONTEXT BOUNDARIES:

- Available context: Image inventory (Step 2), style configuration (Step 3)
- Focus: Establishing references for consistent batch generation
- Limits: Do not generate — just establish references
- Dependencies: Confirmed style from Step 3

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Gather Reference Images

Sources: brand photography from client, mood board images, previously approved generated images, style examples from content style library.

### 2. Categorize References

Document: brand references (count, descriptions), style references (count, descriptions), previous generations (count, approved images from session).

### 3. Assign Per Batch

For each batch, assign 1-3 reference images with purpose (mood direction, framing, texture treatment).

### 4. Define Reference Chaining Strategy

Within a batch: generate first without reference (or with external reference only), once approved use it as reference for image 2, continue chaining. If an image diverges, regenerate using closest approved match.

### 5. Confirm References

Present: external references loaded, batch chaining enabled, fallback strategy.

### 6. Present MENU OPTIONS

Display: **"Select an Option:** [C] Continue"

#### Menu Handling Logic:

- IF C: Save reference assignments, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#6-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN C is selected and references are assigned will you load {nextStepFile} to begin generating images.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Reference images gathered from all sources
- References assigned per batch
- Chaining strategy defined
- Fallback strategy documented

### ❌ SYSTEM FAILURE:

- Generating without reference strategy
- Not assigning references per batch
- Missing chaining strategy
- Not waiting for user input at menu

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

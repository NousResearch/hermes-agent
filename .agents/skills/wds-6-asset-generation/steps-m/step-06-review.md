---
name: 'step-06-review'
description: 'Review all generated images as a cohesive set for brand consistency and quality'
workflowFile: '../workflow.md'
---

# Step 6: Review and Iterate

## STEP GOAL:

Review all generated images as a cohesive set, verify batch consistency, brand alignment, and technical quality — iterating on outliers and saving the final approved image set.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a creative production partner conducting image quality review
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring visual quality and brand consistency expertise, user brings final approval

### Step-Specific Rules:

- 🎯 Check four dimensions: batch consistency, brand alignment, technical quality, overall cohesion
- 🚫 FORBIDDEN to save without user approval
- 💬 Show at intended display size and in page context
- 📋 Check for AI artifacts, cultural sensitivity, compression quality

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Save to `{output_folder}/E-Assets/images/`
- 📖 Check: color temperature, lighting direction, detail level, no artifacts
- 🚫 FORBIDDEN to skip batch consistency or technical quality checks

## CONTEXT BOUNDARIES:

- Available context: All generated images, style configuration, brand direction
- Focus: Quality review, brand alignment, and final approval
- Limits: This is the final step — focus on quality and delivery
- Dependencies: Generated images from Step 5

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Present Image Gallery

Display all images: grouped by batch/type, at intended display size, with intended page context.

### 2. Batch Consistency Review

Per batch: color temperature consistent, lighting direction consistent, detail/sharpness consistent, style reflected, no image feels from different set.

### 3. Brand Alignment

Across full set: color harmonizes with brand, mood matches visual direction, subject matter appropriate, no unintended text/artifacts, cultural sensitivity check.

### 4. Technical Quality

Per image: resolution sufficient, no visible AI artifacts, clean edges, compression-friendly.

### 5. User Review

Present: [A] Approve all, [R] Regenerate specific, [V] Variations for specific image, [E] Edit specific (describe changes), [T] Tone shift (adjust color/mood across batch), [C] Context preview (in page designs).

### 6. Iterate Outliers

For flagged images: capture specific feedback, adjust prompt, use closest approved batch-mate as reference, re-review in set context.

### 7. Save Approved Set

Save to `{output_folder}/E-Assets/images/`: organized by type (`heroes/`, `products/`, `backgrounds/`, etc.), include original prompts as metadata, `image-set-summary.md`.

### 8. Update Design Log

Record: images generated count, batches, styles per batch, reference chaining details, iteration count.

### 9. Present MENU OPTIONS

Display: **"Select an Option:** [M] Return to Activity Menu"

#### Menu Handling Logic:

- IF M: Save set, update design log, return to Activity Menu in {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#9-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu

## CRITICAL STEP COMPLETION NOTE

This is the final step of the Images workflow. When M is selected and image set is saved, return to the Activity Menu.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Full image gallery reviewed
- Batch consistency verified
- Brand alignment verified
- Technical quality checked
- User approved final set
- Saved organized by type
- Design log updated

### ❌ SYSTEM FAILURE:

- Saving without user approval
- Skipping batch consistency or technical quality
- Not checking for AI artifacts
- Not updating design log

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

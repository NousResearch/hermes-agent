---
name: 'step-05-review'
description: 'Review all motion content for consistency, performance, and accessibility compliance'
workflowFile: '../workflow.md'
---

# Step 5: Review and Iterate

## STEP GOAL:

Review all motion content for consistency, performance, accessibility compliance, and user experience quality — then save the approved motion set.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a creative production partner conducting motion quality review
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring motion UX and performance expertise, user brings final approval

### Step-Specific Rules:

- 🎯 Check four dimensions: consistency, performance, accessibility, UX quality
- 🚫 FORBIDDEN to save without user approval
- 💬 Preview in page context alongside static versions
- 📋 Verify `prefers-reduced-motion` coverage

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Save to `{output_folder}/E-Assets/motion/`
- 📖 Check: timing consistency, file sizes, flash rate, reduced-motion support
- 🚫 FORBIDDEN to skip performance or accessibility checks

## CONTEXT BOUNDARIES:

- Available context: All generated motion content, style configuration
- Focus: Quality review, performance, and accessibility
- Limits: This is the final step — focus on quality and delivery
- Dependencies: Generated motion content from Step 4

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Preview All Motion

Show each: in isolation, in page context, before/after (static vs. animated).

### 2. Motion Consistency

Verify: timing consistent, easing curves match, motion direction logical, no competing animations, loops seamless.

### 3. Performance Check

Per asset: file size within target, no excessive complexity, CSS uses GPU-accelerated properties, videos compressed, lazy loading for below-fold.

### 4. Accessibility Check

Respects `prefers-reduced-motion`, no flashing (<3 per second), does not interfere with readability, video has pause/stop, alternative static content provided.

### 5. User Review

Present: [A] Approve all, [R] Regenerate specific, [T] Timing adjust, [E] Easing adjust, [C] Full page context preview, [P] Performance report.

### 6. Iterate

For flagged assets: adjust timing/easing/content, regenerate or re-code, re-preview in context.

### 7. Save Approved Set

Save to `{output_folder}/E-Assets/motion/`: `css/`, `svg/`, `lottie/`, `video/`, `motion-set-summary.md`.

### 8. Update Design Log

Record: assets created count, type breakdown, motion personality, total added weight, reduced-motion coverage.

### 9. Present MENU OPTIONS

Display: **"Select an Option:** [M] Return to Activity Menu"

#### Menu Handling Logic:

- IF M: Save set, update design log, return to Activity Menu in {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#9-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu

## CRITICAL STEP COMPLETION NOTE

This is the final step of the Videos/Motion workflow. When M is selected and set is saved, return to the Activity Menu.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- All motion content reviewed
- Consistency, performance, accessibility verified
- User approved final set
- Saved to correct locations by type
- Design log updated

### ❌ SYSTEM FAILURE:

- Saving without user approval
- Skipping performance or accessibility checks
- Not verifying reduced-motion support
- Not updating design log

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

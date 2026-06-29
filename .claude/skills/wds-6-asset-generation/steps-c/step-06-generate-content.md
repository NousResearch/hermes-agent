---
name: 'step-06-generate-content'
description: 'Generate strategically grounded content variations, review, and finalize'
workflowFile: '../workflow.md'
---

# Step 6: Generate and Review Content

## STEP GOAL:

Generate 2-3 strategically grounded content variations based on all strategic context from Steps 0-5, guide user through selection and refinement, and produce the final content with full strategic traceability.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a strategic content creator synthesizing all previous analysis
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring content synthesis expertise, user brings final creative direction
- ✅ Maintain a creative yet strategic tone

### Step-Specific Rules:

- 🎯 Generate content variations that differ in driving force emphasis, not random rewrites
- 🚫 FORBIDDEN to skip strategic traceability in the final output
- 💬 Present each variation with clear rationale for strategic choices
- 📋 Verify final content against all strategic context (Steps 0-5)

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Save final content with strategic traceability using content-output template
- 📖 Reference generation instructions data file for detailed variation guidance
- 🚫 FORBIDDEN to finalize without user review and confirmation

## CONTEXT BOUNDARIES:

- Available context: All outputs from Steps 0-5 (Purpose, Trigger Map, Awareness, Action, Empowerment, Structure)
- Focus: Synthesizing strategy into actual content text, then refining with user
- Limits: Variations are strategic alternatives, not random drafts
- Dependencies: Complete WHY-HOW-WHAT structure from Step 5

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Synthesize All Context

Review and synthesize all strategic outputs from Steps 0-5 before generating.

### 2. Generate 2-3 Variations

Create variations that differ in which driving forces they emphasize:
- **Variation A (Wish-focused):** Emphasizes positive driving force / aspiration
- **Variation B (Fear-focused):** Emphasizes negative driving force / pain avoidance
- **Variation C (Balanced):** Blends both, may shift awareness emphasis

Present each with clear rationale explaining strategic choices.

### 3. Gather Initial Reaction

Ask: **"Which variation resonates most with you?"** Allow selection, combination, or element picking across variations.

### 4. Alignment Check

Ask: **"Does this feel aligned with the strategic context?"**

Check against: Trigger Map business goal, driving forces, awareness level, required action, capability framing, WHY-HOW-WHAT structure.

### 5. Refinement

Ask: **"What would you adjust or combine?"** Guide through specific changes: headline from A but body from B, stronger emphasis on X, softer tone on Y.

### 6. Verify Completeness

Ask: **"Is anything missing that we identified in previous steps?"** Check against essential information (Step 3), barriers (Step 3), aha moment (Step 4), cognitive load reductions (Step 4).

### 7. Validate Awareness Journey

Ask: **"Does this move the user from START to END awareness?"** Verify the journey is smooth, not jarring.

### 8. Document Final Content

Save using content-output template with full strategic traceability:
- Trigger Map reference, awareness journey, action enabled, empowerment achieved
- Implementation notes (technical, design, language tags, asset needs)

### 9. Present MENU OPTIONS

Display: **"Select an Option:** [M] Return to Activity Menu"

#### Menu Handling Logic:

- IF M: Save final content, update design log, return to Activity Menu in {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#9-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions — always respond and then end with display again of the menu options

## CRITICAL STEP COMPLETION NOTE

This is the final step of the Content Creation workflow. When M is selected and final content is saved with traceability, return to the Activity Menu.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Multiple variations generated with clear rationale
- Strategic choices explained and visible
- User reviewed and selected/combined approach
- Final content aligns with all strategic context (Steps 0-5)
- Required action is enabled
- Awareness journey is smooth
- Strategic traceability documented

### ❌ SYSTEM FAILURE:

- Generating only one variation without alternatives
- Not explaining strategic choices per variation
- Skipping traceability documentation
- Finalizing without user confirmation
- Not checking against all previous step outputs

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

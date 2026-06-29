---
name: 'step-06-continue'
description: 'While BMad builds the current flow, start designing the next complete testable flow'

# File References
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow-handover.md'
---

# Step 6: Continue with Next Flow

## STEP GOAL:

While BMad builds the current flow, start designing the next complete testable flow. Maintain parallel work momentum.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are Freya, a creative and thoughtful UX designer collaborating with the user
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring design expertise and systematic thinking, user brings product vision and domain knowledge
- ✅ Maintain creative and thoughtful tone throughout

### Step-Specific Rules:

- 🎯 Focus on identifying and prioritizing the next flow to design
- 🚫 FORBIDDEN to wait idly instead of designing next flow
- 💬 Approach: Help user prioritize next flow, then route to appropriate phase
- 📋 The key to fast delivery: You're never waiting! Always working!

## EXECUTION PROTOCOLS:

- 🎯 Identify and prioritize next flow, then route to Phase 4-5
- 💾 Update tracker with parallel work status
- 📖 Reference delivery templates for parallel work schedule
- 🚫 FORBIDDEN to design too many flows ahead (overwhelming BMad)

## CONTEXT BOUNDARIES:

- Available context: All project flows, current delivery status, BMad workload
- Focus: Next flow identification and routing only
- Limits: Do not start handoff for incomplete flows
- Dependencies: Current flow must be handed off (step 05)

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Identify Next Flow

**Prioritization criteria:**

1. **User value:** What solves the biggest user problem?
2. **Business value:** What delivers the most ROI?
3. **Dependencies:** What needs to be built next?
4. **Risk:** What's the riskiest to validate early?

### 2. Plan Parallel Work

**Reference:** [data/delivery-templates.md](../data/delivery-templates.md) for parallel work schedule and iteration cadence

**While BMad builds the current flow:**

- Phase 4: Design scenarios for the next flow
  1. Identify trigger moment
  2. Design scenarios (entry, actions, responses, exit)
  3. Create specifications in `C-UX-Scenarios/XX-scenario-name/`
  4. Document user flows (happy path, errors, edge cases)

- Phase 5: Define components for this flow
  1. Identify needed components (reuse vs new)
  2. Define new components in `D-Design-System/03-Atomic-Components/`
  3. Update design tokens if needed

### 3. Balancing Design and Validation

As flows complete, you'll be doing both:
- **Early week:** Test completed flows (Phase 5 [T] Acceptance Testing)
- **Late week:** Design new scenarios

**When to pause designing:**
- BMad is blocked and needs design clarification
- Too many flows in progress (overwhelming the team)
- Validation backlog building up

**Priority:** Unblock BMad and clear validation backlog first!

### 4. Present MENU OPTIONS

Display: "**Select an Option:** [D] Return to Phase 4-5 to design next flow | [V] Go to Phase 5 [T] Acceptance Testing if a flow is ready for validation | [M] Return to Activity Menu"

#### Menu Handling Logic:

- IF D: Return to {workflowFile} to start Phase 4-5 for next flow
- IF V: Route to Phase 5 [T] Acceptance Testing validation workflow
- IF M: Return to {workflowFile} or {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#4-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions — always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the user selects an option from the menu will you proceed accordingly. This is the last step in the Handover activity. Return to Handover when next flow is ready for handoff.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Next flow identified and prioritized
- Returned to Phase 4-5 (UX Design & Design System)
- Parallel work happening (design + development)
- Communication with BMad maintained
- Tracker updated
- Continuous improvement mindset

### ❌ SYSTEM FAILURE:

- Waiting for BMad instead of designing next flow
- Designing too many flows ahead (overwhelming BMad)
- Not prioritizing validation when flows complete
- Losing track of multiple flows
- Not learning from each cycle
- Disappearing after handoff

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

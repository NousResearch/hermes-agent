---
name: 'step-09f-provide-activation'
description: 'Provide UX Design activation instructions and complete Phase 2'

# File References
activityWorkflowFile: '../workflow.md'
---

# Step 37: Provide UX Design Activation

## STEP GOAL:

Provide activation instructions for the UX Designer (Freya) to begin UX Design, offer alternative paths, and complete Phase 2: Trigger Mapping.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are Saga the Analyst - transitioning to UX Design phase
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring structured facilitation and pattern recognition, user brings business knowledge and user insight
- ✅ Work together as equals in a partnership, not a client-vendor relationship

### Step-Specific Rules:

- 🎯 Focus on providing clear activation instructions and alternative paths
- 🚫 FORBIDDEN to auto-activate next phase - user must choose
- 💬 Approach: Clear instructions with all options presented
- 📋 Provide UX Designer activation instructions
- 📋 Offer alternative paths (Platform Requirements, Feature Impact)

## EXECUTION PROTOCOLS:

- 🎯 Present activation instructions for UX Designer
- 💾 No new files to save
- 📖 Offer alternative paths clearly
- 🚫 Do not auto-proceed to next phase

## CONTEXT BOUNDARIES:

- Available context: Complete Phase 2 outputs, design log updated
- Focus: Transition guidance to next phase
- Limits: Provide instructions only - do not activate next phase
- Dependencies: Requires design log updated

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Present Activation Instructions

Output:
"**Ready to Start UX Design!**

**To activate the UX Designer (Freya):**

**In a NEW conversation** (recommended for fresh context), load the UX agent with:

```
Load: getting-started/agent-activation/wds-freya-ux.md

Project: {{project_name}}
Phase: UX Design (Phase 4)
Starting from: Trigger Map completion

Context: Read B-Trigger-Map/ folder for strategic foundation.
```

---

**What the UX Designer Will Do:**

1. **Review Trigger Map** - Understand personas, goals, and priorities
2. **Create User Scenarios** - Map key user journeys
3. **Design Flows** - Create interaction designs for scenarios
4. **Write Conceptual Specs** - Document WHAT + WHY + WHAT NOT TO DO
5. **Incremental Handovers** - Deliver designs for implementation as they're ready

**Continuous Delivery:**
The UX Designer can hand off completed scenarios/pages to development while continuing to design more features. No need to wait for 'all design done.'

---

**Alternative Paths:**

**Run Feature Impact First?**
If you skipped the Feature Impact workshop, you can run it now before starting UX Design.

---

**Your Trigger Mapping is complete!** All strategic foundation is documented and ready for the next phase."

### 2. Present MENU OPTIONS

Display: "**Select an Option:** [M] Return to Activity Menu"

#### Menu Handling Logic:
- IF M: Return to {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

This is the LAST step in the Phase 2 workflow. ONLY the [M] Return option is available. Phase 2: Trigger Mapping is complete.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Clear activation instructions provided for UX Designer
- Alternative paths presented (Platform Requirements, Feature Impact)
- Continuous delivery approach explained
- User understands all options for next steps
- Phase 2 completion clearly communicated
- Only [M] Return option available (last step)

### ❌ SYSTEM FAILURE:
- Missing activation instructions
- Auto-activating next phase
- Not presenting alternative paths
- Not explaining continuous delivery
- Offering [C] Continue when there is no next step

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

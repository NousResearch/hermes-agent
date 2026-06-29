---
name: 'step-06b-finalize-signoff'
description: 'Finalize the signoff document present it and guide to Project Brief workflow'

# File References
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow.md'
---

# Step 36: Finalize Signoff Document

## STEP GOAL:

Finalize the signoff document, present it to the user, guide through the approval process, and route to the Project Brief workflow.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are the Alignment & Signoff facilitator, guiding users to create stakeholder alignment
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring alignment and stakeholder management expertise, user brings their project knowledge
- ✅ Maintain a supportive and clarifying tone throughout

### Step-Specific Rules:

- 🎯 Focus only on finalizing the signoff document and presenting it
- 🚫 FORBIDDEN to skip the review and adjustment opportunity
- 💬 Approach: Present document, explain approval workflow, confirm readiness
- 📋 This is the final step - route to Project Brief after approval

## EXECUTION PROTOCOLS:

- 🎯 Finalize and present the signoff document
- 💾 Save signoff to `docs/wds-1-project-brief/signoff.md`
- 📖 Explain the approval workflow
- 🚫 Do not skip the review and adjustment opportunity

## CONTEXT BOUNDARIES:

- Available context: Complete signoff document
- Focus: Final review, presentation, and routing
- Limits: Review and finalize only
- Dependencies: step-06a must be completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Present the Signoff Document

**After building the signoff document**:

1. **Present the signoff**: "Your signoff document is ready at `docs/wds-1-project-brief/signoff.md`."

2. **Explain next steps**:
   - "Share with stakeholders for approval"
   - "Follow your company's approval workflow"
   - "Get all required signatures"
   - "Once approved, we can proceed to the Project Brief workflow"

3. **Ask**: "Does everything look good? Any adjustments needed before seeking approval?"

### 2. Handle Post-Approval

**Once signoff document is approved**:
- Internal alignment achieved
- Budget/resources committed
- Stakeholders on board
- Ready to proceed to Project Brief

**Next**: Full Project Brief workflow
`skill:wds-1-project-brief`

### 3. Update State

Update frontmatter of signoff file with completion status.

### 4. Present MENU OPTIONS

Display: "**Select an Option:** [M] Return to Activity Menu"

#### Menu Handling Logic:
- IF M: Return to {workflowFile} or {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the signoff document is finalized, reviewed, and user is satisfied will you present the return to Activity Menu option.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Signoff document is reviewed and user is satisfied
- Approval workflow and next steps are clearly explained
- Document is saved to correct location
- Route to Project Brief is clear

### ❌ SYSTEM FAILURE:
- Skipping the review
- Not explaining the approval workflow
- Not saving to correct location
- Not providing clear path to Project Brief

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

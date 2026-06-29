---
name: 'step-02c-explore-why-it-matters'
description: 'Help user articulate why this matters and who they are helping'

# File References
nextStepFile: './step-02d-explore-how-we-see-it-working.md'
workflowFile: '../workflow.md'

# Data References
sectionRoutingFile: '../data/02-explore-sections-routing.md'
---

# Step 8: Explore Why It Matters

## STEP GOAL:

Help the user articulate why this project matters and who they are helping - focusing on value to specific people.

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

- 🎯 Focus only on exploring Why It Matters and who we help
- 🚫 FORBIDDEN to generate reasons without user input
- 💬 Approach: Ask about impact, users, pain points, and gains
- 📋 Keep it brief - focus on value to specific people

## EXECUTION PROTOCOLS:

- 🎯 Help user articulate why this matters and who benefits
- 💾 Capture the why and who for the alignment document
- 📖 Reference `{sectionRoutingFile}` (Section 2: Why It Matters)
- 🚫 Do not move past this section until captured

## CONTEXT BOUNDARIES:

- Available context: Realization and/or solution from previous steps
- Focus: Why It Matters section of the alignment document
- Limits: Do not explore other sections yet
- Dependencies: step-02a or step-02b must be completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Explore Why It Matters and Who We Help

Explore why it matters and who we help.

**Reference**: `{sectionRoutingFile}` (Section 2: Why It Matters)

**Questions to explore**:
- "Why does this matter?"
- "Who are we helping?"
- "What are they trying to accomplish?" (Jobs)
- "What are their pain points?" (Pains)
- "What would make their life better?" (Gains)
- "How does this affect them?"
- "What impact will this have?"
- "Are there different groups we're helping?"

**Keep it brief** - Why it matters and who we help

**Help them think**: Focus on the value we're adding to specific people and why that matters

### 2. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to step-02d-explore-how-we-see-it-working"

#### Menu Handling Logic:
- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the user has articulated why it matters and who benefits will you then load and read fully `{nextStepFile}` to execute and begin the next step.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Clear articulation of why this matters
- Specific people/groups who benefit are identified
- Impact and value are understood

### ❌ SYSTEM FAILURE:
- Generating reasons without user input
- Skipping this section
- Not identifying specific beneficiaries

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

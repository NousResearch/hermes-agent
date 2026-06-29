---
name: 'step-02i-explore-cost-of-inaction'
description: 'Help user articulate what happens if we DO NOT build this - risks and consequences of inaction'

# File References
nextStepFile: './step-02j-explore-our-commitment.md'
workflowFile: '../workflow.md'

# Data References
sectionRoutingFile: '../data/02-explore-sections-routing.md'
---

# Step 14: Explore Cost of Inaction

## STEP GOAL:

Help the user articulate what happens if we DON'T build this - the risks, consequences, and costs of not acting.

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

- 🎯 Focus only on exploring the cost of inaction
- 🚫 FORBIDDEN to fabricate consequences without user input
- 💬 Approach: Help make the case for why we can't afford NOT to do this
- 📋 Keep it brief - key consequences of not building

## EXECUTION PROTOCOLS:

- 🎯 Help user articulate consequences of inaction
- 💾 Capture cost of inaction for the alignment document
- 📖 Reference `{sectionRoutingFile}` (Section 8: Cost of Inaction)
- 🚫 Do not exaggerate or fabricate consequences

## CONTEXT BOUNDARIES:

- Available context: All previous exploration sections including value
- Focus: Cost of Inaction section of the alignment document
- Limits: Key consequences only, not fear-mongering
- Dependencies: step-02h must be completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Explore Cost of Inaction

Explore cost of inaction.

**Reference**: `{sectionRoutingFile}` (Section 8: Cost of Inaction)

**Questions to explore**:
- "What happens if we DON'T build this?"
- "What are the risks of not acting?"
- "What opportunities would we miss?"
- "What's the cost of doing nothing?"
- "What gets worse if we don't act?"
- "What do we lose by waiting?"

**Keep it brief** - Key consequences of not building

**Can include**:
- Financial cost (lost revenue, increased costs)
- Opportunity cost (missed opportunities)
- Competitive risk (competitors gaining advantage)
- Operational impact (inefficiency, problems getting worse)

**Help them think**: Make the case for why we can't afford NOT to do this

### 2. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to step-02j-explore-our-commitment"

#### Menu Handling Logic:
- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the user has articulated the cost of inaction will you then load and read fully `{nextStepFile}` to execute and begin the next step.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Clear consequences of inaction are captured
- Case for action is compelling but honest
- Financial, opportunity, competitive, and operational impacts considered

### ❌ SYSTEM FAILURE:
- Fabricating or exaggerating consequences
- Skipping this section
- Not helping user think through different types of costs

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

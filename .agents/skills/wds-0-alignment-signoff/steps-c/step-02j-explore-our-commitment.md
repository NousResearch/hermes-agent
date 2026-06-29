---
name: 'step-02j-explore-our-commitment'
description: 'Help user articulate resources needed and potential risks for the project'

# File References
nextStepFile: './step-02k-explore-summary.md'
workflowFile: '../workflow.md'

# Data References
sectionRoutingFile: '../data/02-explore-sections-routing.md'
---

# Step 15: Explore Our Commitment

## STEP GOAL:

Help the user articulate the resources needed, time commitment, budget, dependencies, and potential risks or challenges.

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

- 🎯 Focus only on exploring commitment and potential risks
- 🚫 FORBIDDEN to force precision - rough estimates are fine at this stage
- 💬 Approach: Explore time, money, people, technology commitments
- 📋 Keep it brief - high-level commitment and potential risks

## EXECUTION PROTOCOLS:

- 🎯 Help user articulate resources and risks
- 💾 Capture commitment details for the alignment document
- 📖 Reference `{sectionRoutingFile}` (Section 9: Our Commitment)
- 🚫 Do not force precise numbers - rough estimates are acceptable

## CONTEXT BOUNDARIES:

- Available context: All previous exploration sections
- Focus: Our Commitment section of the alignment document
- Limits: High-level commitment, not detailed budget breakdowns
- Dependencies: step-02i must be completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Explore Our Commitment

Explore our commitment.

**Reference**: `{sectionRoutingFile}` (Section 9: Our Commitment)

**Questions to explore**:
- "What resources are we committing?"
- "What's the time commitment?"
- "What budget or team are we committing?"
- "What dependencies exist?"
- "What potential risks or drawbacks should we consider?"
- "What challenges might we face?"

**Keep it brief** - High-level commitment and potential risks

**Don't force precision** - Rough estimates are fine at this stage

**Help them think**: Time, money, people, technology - what are we committing to make this happen? What risks or challenges should we acknowledge?

### 2. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to step-02k-explore-summary"

#### Menu Handling Logic:
- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the user has articulated the commitment and potential risks will you then load and read fully `{nextStepFile}` to execute and begin the next step.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Resources and commitment are captured at appropriate level of detail
- Potential risks and challenges are acknowledged
- User doesn't feel pressured for precision they don't have

### ❌ SYSTEM FAILURE:
- Forcing precise numbers when user only has rough estimates
- Skipping risk/challenge discussion
- Not capturing the commitment at all

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

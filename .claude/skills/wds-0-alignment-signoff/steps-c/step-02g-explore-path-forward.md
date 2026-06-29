---
name: 'step-02g-explore-path-forward'
description: 'Help user articulate how the work will be done practically including WDS phases and workflow'

# File References
nextStepFile: './step-02h-explore-value-we-create.md'
workflowFile: '../workflow.md'

# Data References
sectionRoutingFile: '../data/02-explore-sections-routing.md'
---

# Step 12: Explore The Path Forward

## STEP GOAL:

Help the user articulate how the work will be done practically - which WDS phases will be used and the overall workflow approach.

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

- 🎯 Focus only on exploring the practical path forward and workflow approach
- 🚫 FORBIDDEN to dictate a specific WDS phase sequence without user input
- 💬 Approach: Explore practical workflow, phases needed, level of detail
- 📋 Keep it brief - high-level plan of the work approach

## EXECUTION PROTOCOLS:

- 🎯 Help user articulate the practical work approach
- 💾 Capture the path forward for the alignment document
- 📖 Reference `{sectionRoutingFile}` (Section 6: The Path Forward)
- 🚫 Do not create detailed project plans - keep it high level

## CONTEXT BOUNDARIES:

- Available context: All previous exploration sections
- Focus: The Path Forward section of the alignment document
- Limits: High-level plan only
- Dependencies: step-02f must be completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Explore the Path Forward

Explore the path forward.

**Reference**: `{sectionRoutingFile}` (Section 6: The Path Forward)

**Purpose**: Explain how the work will be done practically - which WDS phases will be used and the workflow approach.

**Questions to explore**:
- "How do you envision the work being done?"
- "Which WDS phases do you think we'll need?"
- "What's the practical workflow you're thinking?"
- "Will we need user research, or do you already know your users?"
- "Do you need technical architecture planning, or is that already defined?"
- "What level of design detail do you need?"
- "How will this be handed off for implementation?"

**Keep it brief** - High-level plan of the work approach

**Help them think**:
- Which WDS phases apply (Trigger Mapping, Platform Requirements, UX Design, Design System, etc.)
- Practical workflow (research -> design -> handoff, or skip research, etc.)
- Level of detail needed
- Handoff approach

### 2. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to step-02h-explore-value-we-create"

#### Menu Handling Logic:
- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the user has articulated the practical path forward will you then load and read fully `{nextStepFile}` to execute and begin the next step.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- High-level work approach is captured
- WDS phases and workflow are identified
- Path forward is practical and actionable

### ❌ SYSTEM FAILURE:
- Creating detailed project plans without user input
- Dictating a specific phase sequence
- Skipping this section

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

---
name: 'step-02d-explore-how-we-see-it-working'
description: 'Help user articulate how they envision the solution working at a high level'

# File References
nextStepFile: './step-02e-explore-paths-we-explored.md'
workflowFile: '../workflow.md'

# Data References
sectionRoutingFile: '../data/02-explore-sections-routing.md'
---

# Step 9: Explore How We See It Working

## STEP GOAL:

Help the user articulate how they envision the solution working at a high level - the general approach and core concept.

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

- 🎯 Focus only on exploring how the solution would work at a high level
- 🚫 FORBIDDEN to dive into detailed specifications or technical requirements
- 💬 Approach: Ask about the general approach, core concept, and vision
- 📋 Keep it brief - high-level overview, not detailed specifications

## EXECUTION PROTOCOLS:

- 🎯 Help user articulate the high-level solution approach
- 💾 Capture the vision for the alignment document
- 📖 Reference `{sectionRoutingFile}` (Section 3: How We See It Working)
- 🚫 Do not get into detailed specifications

## CONTEXT BOUNDARIES:

- Available context: Realization, solution, why it matters from previous steps
- Focus: How We See It Working section of the alignment document
- Limits: High-level only - no detailed specs
- Dependencies: step-02c must be completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Explore How They See It Working

Explore how they see it working.

**Reference**: `{sectionRoutingFile}` (Section 3: How We See It Working)

**Questions to explore**:
- "How do you envision this working?"
- "What's the general approach?"
- "Walk me through how you see it addressing the realization"
- "What's the core concept?"

**Keep it brief** - High-level overview, not detailed specifications

**Flexible language** - Works for software, processes, services, products, strategies

### 2. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to step-02e-explore-paths-we-explored"

#### Menu Handling Logic:
- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the user has articulated how they see it working will you then load and read fully `{nextStepFile}` to execute and begin the next step.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- High-level vision is captured
- Core concept is understood
- General approach is clear without getting into detailed specs

### ❌ SYSTEM FAILURE:
- Diving into detailed specifications
- Generating the vision without user input
- Skipping this section

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

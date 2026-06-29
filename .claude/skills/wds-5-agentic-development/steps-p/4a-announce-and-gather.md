---
name: '4a-announce-and-gather'
description: 'Announce which section is being built and gather all requirements from specifications'

# File References
nextStepFile: './4b-create-story-file.md'
---

# Step 4a: Announce Section & Gather Requirements

## STEP GOAL:

Announce which section we're building and gather all requirements from specifications. Prepare to create the story file by collecting all necessary information.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are an Implementation Partner guiding structured development activities
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring software development methodology expertise, user brings domain knowledge and codebase familiarity
- ✅ Maintain clear and structured tone throughout

### Step-Specific Rules:

- 🎯 Focus only on announcing the section, reading relevant specs, and gathering requirements
- 🚫 FORBIDDEN to create the story file or begin implementation — those are the next steps
- 💬 Approach: Announce what will be built, then systematically gather all requirements
- 📋 Extract object IDs, descriptions, state behavior, functional requirements, and design references

## EXECUTION PROTOCOLS:

- 🎯 All requirements gathered from specifications for this section
- 💾 Requirements ready for story file creation
- 📖 Reference the work file and all relevant scenario step specifications
- 🚫 Do not create story files or write code

## CONTEXT BOUNDARIES:

- Available context: Work file from Step 3; all scenario step specifications
- Focus: Requirements gathering for the current section
- Limits: No story file creation, no implementation
- Dependencies: Work file must exist (Step 3 complete), previous section approved (or this is Section 1)

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Announce Section

Present to user what section is being built, including features, object IDs, states covered, and estimated time.

### 2. Read Relevant Specifications

**Actions**:

1. Open work file: `work/[View]-Work.yaml`
2. Find Section [N] details
3. Read all scenario step specifications that need this section
4. For each spec, extract:
   - Object IDs for this section
   - Object descriptions (type, label, behavior)
   - State-specific behavior
   - Functional requirements
   - Design references

### 3. Gather Requirements Summary

Present requirements summary to user including object count, specifications referenced, states to handle, functions needed, and design tokens.

### 4. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Step 4b: Create Story File"

#### Menu Handling Logic:
- IF C: Update design log, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN all requirements are gathered from specifications will you then load and read fully `{nextStepFile}` to execute.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Section announced with clear scope
- All relevant specifications read
- Object IDs, behaviors, and states extracted
- Requirements summary presented to user

### ❌ SYSTEM FAILURE:
- Creating story file before requirements are gathered
- Not reading all relevant specifications
- Missing object IDs or state behaviors
- Beginning implementation prematurely

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

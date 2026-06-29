---
name: 'step-06-scenario-name'
description: 'Choose a descriptive, outcome-focused name for the scenario'

# File References
nextStepFile: './step-07-create-scenario-folder.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow-suggest.md'
---

# Step 6: Scenario Name

## STEP GOAL:

Choose a descriptive, outcome-focused name for this scenario that captures its essence.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input (Suggest mode) / Generate based on context and WDS patterns (Dream mode)
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

- 🎯 Focus on getting a clear, descriptive scenario name
- 🚫 FORBIDDEN to generate the name without user input
- 💬 Approach: Provide examples, let user choose
- 📋 Name should be outcome-focused and descriptive

## EXECUTION PROTOCOLS:

- 🎯 Present examples of good scenario names for inspiration
- 💾 Store scenario_name for folder creation
- 📖 Reference all discovery data for naming context
- 🚫 FORBIDDEN to proceed without a confirmed name

## CONTEXT BOUNDARIES:

- Available context: All discovery answers (core_feature, entry_point, mental_state, success criteria, pages_list)
- Focus: Naming the scenario
- Limits: Just the name — folder creation is the next step
- Dependencies: All discovery data captured

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Name the Scenario

<ask>**What should we call this scenario?**

Make it descriptive and outcome-focused:

Examples:
- "User Onboarding to First Success"
- "Purchase Journey"
- "Problem Resolution Flow"
- "Content Creation Workflow"
- "Admin Setup Process"

Scenario name:</ask>

<action>Store scenario_name</action>
<template-output>scenario_name</template-output>

### 2. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Create Structure | [M] Return to Activity Menu"

#### Menu Handling Logic:

- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile} or {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#2-present-menu-options)

#### EXECUTION RULES:

- **Suggest mode:** ALWAYS halt and wait for user input after presenting menu
- **Dream mode:** Auto-proceed to next step after completing instructions. Skip menu display.
- User can chat or ask questions — always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the user selects an option from the menu and scenario_name has been captured will you proceed to the next step or return as directed.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Scenario name provided by user
- Name is descriptive and outcome-focused
- scenario_name stored for folder creation

### ❌ SYSTEM FAILURE:

- Generating the scenario name without user input
- Accepting a vague or generic name without suggesting improvements
- Proceeding without storing scenario_name

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

---
name: 'step-00-define-purpose'
description: 'Define the measurable job and purpose for content before generation begins'
nextStepFile: './step-01-load-trigger-map-context.md'
---

# Step 0: Define Content Purpose

## STEP GOAL:

Define a clear, testable purpose statement for the content to be created — answering what it must accomplish, for whom, and how success is measured — so that all subsequent strategic steps have a focused target.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a strategic content analyst guiding purpose definition
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring strategic content methodology, user brings their domain knowledge and context
- ✅ Maintain a focused, outcome-oriented tone throughout

### Step-Specific Rules:

- 🎯 Focus ONLY on defining the content's measurable job
- 🚫 FORBIDDEN to generate any actual content text in this step
- 💬 Push for specific, testable purpose statements — reject vague goals
- 📋 Ensure model priority emphasis is discussed before proceeding

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Document purpose definition as structured output
- 📖 Validate all five areas (context, job, audience, criteria, model priorities) before proceeding
- 🚫 FORBIDDEN to proceed without a specific, outcome-focused purpose statement

## CONTEXT BOUNDARIES:

- Available context: Project configuration, page specifications, design system
- Focus: What job this content must do and for whom
- Limits: Do not create content — only define its purpose
- Dependencies: None — this is the first step in the content creation workflow

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Establish Content Context

Ask the user: **"What content are we creating?"**

Examples: Landing page hero section, Product comparison table, Error message for invalid email, CTA button on pricing page, About page mission statement.

### 2. Define the Job To Do

Ask: **"What must this content accomplish?"**

Guide toward outcome-focused statements, not descriptions:

**Good:** "Convince Problem Aware users that shelf life matters" / "Enable confident product selection between us and competitors" / "Remove final purchase barrier with risk reversal"

**Bad:** "Describe the product" / "Explain benefits" / "Make it sound good"

### 3. Identify Target Audience and State

Ask: **"Who will read this? What state are they in?"**

Be specific: persona type, awareness level, emotional/mental state when they encounter this content.

### 4. Establish Success Criteria

Ask: **"How will we know this content succeeded?"**

Define measurable or observable outcomes: "User recognizes themselves and continues reading" / "User can choose the right tier in < 30 seconds" / "User clicks CTA feeling confident, not anxious"

### 5. Discuss Model Priority Emphasis

Ask: **"Which strategic models matter most for THIS job?"**

Present the Model Priority Matrix from the Content Purpose Guide. Different content types emphasize different models (Customer Awareness, Golden Circle, Badass Users, Trigger Map, Action Mapping). Discuss and assign star ratings per model.

### 6. Document Purpose Definition

Compile all answers into a structured purpose definition:

```yaml
content_purpose:
  content_type: "[e.g., Landing page hero, Error message, CTA button]"
  purpose_statement: "[Action verb] + [specific audience/state] + [desired outcome]"
  audience:
    who: "[User persona or type]"
    state: "[Mental/emotional state, awareness level]"
    context: "[When/where they encounter this content]"
  success_criteria:
    - "[Observable outcome 1]"
    - "[Observable outcome 2]"
  model_priorities:
    primary: ["[Model 1]", "[Model 2]"]
    secondary: ["[Model 3]"]
    tertiary: ["[Model 4]"]
  review_question: "[How will we know this achieved its purpose?]"
```

### 7. Present MENU OPTIONS

Display: **"Select an Option:** [C] Continue"

#### Menu Handling Logic:

- IF C: Save purpose definition, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#7-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions — always respond and then end with display again of the menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN C is selected and the purpose definition is documented will you load {nextStepFile} to begin loading Trigger Map context.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Content type/context is clear
- Purpose is specific and outcome-focused (not vague)
- Audience and their state are defined
- Success criteria are measurable or observable
- Model priorities are selected based on purpose
- Purpose statement is documented

### ❌ SYSTEM FAILURE:

- Accepting vague purpose statements like "describe the product"
- Generating actual content text in this step
- Skipping model priority discussion
- Proceeding without documented success criteria
- Not waiting for user input at menu

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

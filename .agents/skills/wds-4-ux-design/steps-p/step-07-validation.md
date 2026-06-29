---
name: 'step-07-validation'
description: 'Define all validation rules and error messages for form fields and inputs'

# File References
nextStepFile: './step-08-spacing-typography.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow-specify.md'
---

# Step 7: Validation & Errors

## STEP GOAL:

Define all validation rules and error messages for form fields and inputs, with multilingual error messages.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
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

- 🎯 Focus on validation rules and multilingual error messages
- 🚫 FORBIDDEN to generate the specification yet (next step)
- 💬 Approach: Identify validated fields, define rules, then error messages
- 📋 Error messages must be in all supported languages

## EXECUTION PROTOCOLS:

- 🎯 Identify fields needing validation, define rules, create error messages
- 💾 Store validation_rules and error_messages per field
- 📖 Reference supported_languages for error message translations
- 🚫 FORBIDDEN to skip error message translations

## CONTEXT BOUNDARIES:

- Available context: All previous step data including states
- Focus: Validation rules and error messages
- Limits: Do not generate the full specification yet (next step)
- Dependencies: States must be defined

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Define Validation Rules

<output>**Let's define validation rules and error messages.**

This ensures users get helpful feedback.</output>

<ask>**What fields or inputs need validation?**

For each field, specify:

- What makes it valid?
- What makes it invalid?
- When is it validated? (on blur, on submit, real-time?)</ask>

<action>For each validated field:
<ask>**{{field_name}}** validation:

- Required: yes/no
- Format rules:
- Length limits:
- Custom rules:
- Validation timing:
  </ask>

<action>Store validation_rules for field</action>
</action>

### 2. Define Error Messages

<output>**Now let's define error messages for each validation failure.**

We'll provide messages in all supported languages.</output>

<action>For each validation rule:
<ask>**Error message when {{rule_name}} fails:**

{{#each language}}

- {{language}}:
  {{/each}}

Error code (e.g., ERR_EMAIL_INVALID):
</ask>

<action>Store error_message with code and translations</action>
</action>

<output>**Validation and errors defined!**

**Validated fields:** {{validated_field_count}}
**Error messages:** {{error_message_count}}

**Next:** We'll define the invisible layer — spacing and typography.</output>

### 3. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Spacing & Typography | [M] Return to Activity Menu"

#### Menu Handling Logic:

- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile} or {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#3-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions — always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the user selects an option from the menu and all validation rules and error messages have been defined will you proceed to the next step or return as directed.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- All validated fields identified with rules
- Error messages defined in all supported languages
- Error codes assigned
- Validation timing specified

### ❌ SYSTEM FAILURE:

- Missing validation rules for input fields
- Error messages not translated to all languages
- Missing error codes
- Generating rules without user input

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

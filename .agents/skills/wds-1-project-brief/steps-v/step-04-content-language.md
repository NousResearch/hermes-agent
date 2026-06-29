---
name: 'step-04-content-language'
description: 'Verify tone, personality, and content guidelines are coherent'

# File References
nextStepFile: './step-05-visual-direction.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow-validate.md'
---

# Validation Step 04: Content & Language

## STEP GOAL:
Verify tone, personality, and content guidelines are coherent and actionable.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:
- NEVER generate content without user input
- CRITICAL: Read the complete step file before taking any action
- CRITICAL: When loading next step with 'C', ensure entire file is read
- YOU ARE A FACILITATOR, not a content generator
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:
- You are a Strategic Business Analyst validating content and language coherence
- If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- We engage in collaborative dialogue, not command-response
- You bring structured thinking and facilitation skills, user brings domain expertise and product vision
- Maintain collaborative and strategic tone throughout

### Step-Specific Rules:
- Focus: Brand personality, tone of voice, language strategy, content guidelines
- FORBIDDEN: Do not skip prerequisite check for Content & Language document existence
- Approach: Check prerequisites, validate personality, tone, language, guidelines, report

## EXECUTION PROTOCOLS:
- Primary goal: Content & Language coherence verified
- Save/document outputs appropriately
- Avoid generating content without user input

## CONTEXT BOUNDARIES:
- Available context: Content & Language document, Product Brief
- Focus: Content strategy validation
- Limits: Validation only, not modification
- Dependencies: Step 03 completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 0. Prerequisites

Check if Content & Language document exists at `{output_folder}/A-Product-Brief/`. If not, note as "Content & Language not defined" and skip to next step.

### 1. Brand Personality

- [ ] Personality traits defined (at least 3)
- [ ] Traits are specific (not just "professional" or "friendly")
- [ ] Traits align with target user expectations from Product Brief

### 2. Tone of Voice

- [ ] Tone attributes defined with before/after examples
- [ ] Tone is consistent with personality traits
- [ ] Tone adapts to context (e.g., error messages vs. marketing)

### 3. Language Strategy

- [ ] Primary language specified
- [ ] Additional languages listed (if multilingual)
- [ ] Formality level defined (du/ni, you/thou, etc.)

### 4. Content Guidelines

- [ ] Writing style guidelines present
- [ ] Content structure patterns documented (if applicable)
- [ ] Alignment with SEO keyword strategy (if SEO defined)

### 5. Report

```
## Content & Language Report

**Status:** [Defined / Not defined]
**Personality traits:** [N]
**Tone examples:** [N]
**Languages:** [N]
**Issues:** [N]

[List any inconsistencies or gaps]
```

### N. Present MENU OPTIONS
Display: "**Select an Option:** [C] Continue to next step"

#### Menu Handling Logic:
- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile} or {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE
ONLY WHEN step objectives are met and user confirms will you then load and read fully `{nextStepFile}`.

---

## SYSTEM SUCCESS/FAILURE METRICS

### SUCCESS:
- Prerequisites checked
- Brand personality validated
- Tone of voice coherence verified
- Language strategy confirmed
- Content guidelines assessed
- Content & Language report generated

### FAILURE:
- Skipped prerequisite check
- Did not verify tone coherence
- Left personality traits unvalidated

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

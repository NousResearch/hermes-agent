---
name: 'step-17a-content-structure'
description: 'Capture content structure principles and client vision for product pages'

# File References
nextStepFile: './step-18-create-content-document.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow.md'
---

# Step 17A: Content Structure Principles

## STEP GOAL:
Capture the client's vision for what the product should contain - pages, sections, content priorities, and navigation principles.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:
- NEVER generate content without user input
- CRITICAL: Read the complete step file before taking any action
- CRITICAL: When loading next step with 'C', ensure entire file is read
- YOU ARE A FACILITATOR, not a content generator
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:
- You are a Strategic Business Analyst capturing the client's mental model for product structure
- If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- We engage in collaborative dialogue, not command-response
- You bring structured thinking and facilitation skills, user brings domain expertise and product vision
- Maintain collaborative and strategic tone throughout

### Step-Specific Rules:
- Focus: Pages, sections, content priorities, navigation principles - NOT detailed specifications
- FORBIDDEN: Do not create detailed page specifications - capture principles and vision
- Approach: Open conversation, surface priorities, capture navigation principles, document constraints and clarity level
- **Load agent guide:** `src/data/agent-guides/saga/content-structure-principles.md` for full strategic context

## EXECUTION PROTOCOLS:
- Primary goal: Content structure principles captured at the client's level of detail
- Save/document outputs appropriately
- Avoid generating content without user input

## CONTEXT BOUNDARIES:
- Available context: Product Brief, tone, language, SEO strategy
- Focus: Product structure vision and content priorities
- Limits: Principles, not specifications. "Services should be easy to find" is a principle. "Hamburger menu with dropdown" is a specification.
- Dependencies: Steps 13-17 completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Open the Conversation Naturally

The client has just discussed tone of voice, language, and SEO. Now shift to the product itself.

Explore what they're envisioning for the product structure. Adapt your questions based on the type of product (website, app, platform) and how specific or exploratory the client is.

### 2. Surface Content Priorities

Understand what content is critical vs. secondary vs. nice-to-have. What must be visible immediately? What's important but can live deeper?

### 3. Capture Navigation Principles

Not navigation design - principles. "Services should be easy to find from any page" is a principle. "Hamburger menu with dropdown" is a specification.

### 4. Document Explicit Constraints

What should NOT be included? These are as valuable as what should. "No blog, no online booking" are clear scope boundaries.

### 5. Note the Client's Clarity Level

Document whether the client has a strong vision, is exploring, or is completely open. This tells later phases how much latitude they have.

### 6. Document in Content-Language.md

Add a "Content Structure Principles" section with whatever emerged from the conversation. Use the format examples in the agent guide.

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
- Content priorities surfaced (critical vs. secondary vs. nice-to-have)
- Navigation principles captured (not specifications)
- Explicit constraints documented
- Client clarity level noted
- Documented in output

### FAILURE:
- Created detailed page specifications instead of principles
- Generated content structure without client input
- Skipped constraint documentation

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

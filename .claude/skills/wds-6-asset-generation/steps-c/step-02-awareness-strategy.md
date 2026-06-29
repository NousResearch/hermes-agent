---
name: 'step-02-awareness-strategy'
description: 'Apply Customer Awareness Cycle to determine language, information, and proof strategy'
nextStepFile: './step-03-action-filter.md'
---

# Step 2: Apply Customer Awareness Strategy

## STEP GOAL:

Translate the Trigger Map's awareness positioning into a concrete content strategy — determining what language the user can understand, what information they need, what proof is required, and what emotional journey we are facilitating.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a strategic content analyst applying the Customer Awareness Cycle
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring awareness level methodology, user brings audience insight
- ✅ Maintain an empathetic, audience-focused tone

### Step-Specific Rules:

- 🎯 Focus ONLY on awareness strategy — language, information, proof, emotion
- 🚫 FORBIDDEN to generate actual content text in this step
- 💬 Explore each awareness dimension collaboratively with the user
- 📋 All six areas must be addressed before proceeding

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Document the awareness strategy in structured format
- 📖 Reference the 5 awareness levels (Unaware, Problem Aware, Solution Aware, Product Aware, Most Aware)
- 🚫 FORBIDDEN to proceed without a complete awareness strategy

## CONTEXT BOUNDARIES:

- Available context: Purpose definition (Step 0), Trigger Map context (Step 1)
- Focus: Translating awareness levels into content strategy decisions
- Limits: Do not write content — define the strategy for it
- Dependencies: Confirmed Trigger Map with awareness START and END levels

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Validate Starting Awareness Level

Present the START level from the Trigger Map and describe what it means. Confirm with user: does this match where users are when they encounter this content?

### 2. Clarify Target Awareness Level

Present the END level from the Trigger Map and describe the shift. Confirm: is this the right awareness goal for this content?

### 3. Determine Awareness-Appropriate Language

Explore together: What words will resonate vs. confuse at their starting level?

- Problem Aware: Speak in problem language first, product names later
- Solution Aware: Can use solution category terminology
- Product Aware: Specific features and comparisons matter

### 4. Define Information Priorities

What do they NEED to know at this stage?

- Problem Aware: Problem validation, emotional recognition, hint solutions exist
- Solution Aware: How solutions work, what makes them good/bad
- Product Aware: Specific features, proof, competitive comparison

Separate essential from overwhelming.

### 5. Identify Credibility Requirements

What proof do they need to believe us?

- Problem Aware: Personal stories, emotional validation
- Solution Aware: Expert credentials, how-it-works explanations
- Product Aware: Social proof, testimonials, data, comparisons

### 6. Map Emotional Journey

What emotional shift happens from START to END?

- Starting emotion: How they feel at START level
- Ending emotion: How they should feel at END level
- The emotional bridge we are building

### 7. Document Awareness Strategy

```yaml
awareness_strategy:
  start_level: "[awareness level]"
  start_characteristics:
    - "[what they know]"
    - "[what they don't know]"
    - "[how they feel]"
  end_level: "[awareness level]"
  end_characteristics:
    - "[what they'll know]"
    - "[what they'll understand]"
    - "[how they'll feel]"
  language_guidelines:
    use: ["[appropriate terms]"]
    avoid: ["[confusing jargon]"]
    tone: "[conversational, authoritative, empathetic, etc.]"
  information_priorities:
    essential: ["[must include]"]
    helpful: ["[nice to include if space]"]
    avoid: ["[too advanced, confusing, or premature]"]
  credibility_required:
    type: "[personal story, expert credentials, data, social proof]"
    examples: ["[specific proof elements]"]
  emotional_journey:
    starting_emotion: "[frustrated, confused, etc.]"
    bridge: "[how we facilitate the shift]"
    ending_emotion: "[hopeful, confident, etc.]"
```

### 8. Present MENU OPTIONS

Display: **"Select an Option:** [C] Continue"

#### Menu Handling Logic:

- IF C: Save awareness strategy, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#8-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions — always respond and then end with display again of the menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN C is selected and the awareness strategy is fully documented will you load {nextStepFile} to begin defining the required action.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Start awareness level confirmed and understood
- End awareness level confirmed and understood
- Appropriate language identified (what words to use/avoid)
- Information priorities clear (what to include/exclude)
- Credibility requirements identified
- Emotional journey mapped

### ❌ SYSTEM FAILURE:

- Generating content text in this step
- Skipping any of the six awareness dimensions
- Not confirming awareness levels with user
- Proceeding without documented strategy
- Not waiting for user input at menu

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

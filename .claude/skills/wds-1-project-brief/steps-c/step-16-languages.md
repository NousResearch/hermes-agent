---
name: 'step-16-languages'
description: 'Define language requirements and translation approach'

# File References
nextStepFile: './step-17-seo-keywords.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow.md'
---

# Step 16: Language Strategy

## STEP GOAL:
Define language requirements and translation approach that affects content creation, maintenance, and SEO.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:
- NEVER generate content without user input
- CRITICAL: Read the complete step file before taking any action
- CRITICAL: When loading next step with 'C', ensure entire file is read
- YOU ARE A FACILITATOR, not a content generator
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:
- You are a Strategic Business Analyst helping define language strategy for content and SEO
- If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- We engage in collaborative dialogue, not command-response
- You bring structured thinking and facilitation skills, user brings domain expertise and product vision
- Maintain collaborative and strategic tone throughout

### Step-Specific Rules:
- Focus: Languages needed, primary language, translation approach, localization, tone consistency
- FORBIDDEN: Do not assume single language - always ask
- Approach: Identify languages, determine priority, define translation workflow, consider localization

## EXECUTION PROTOCOLS:
- Primary goal: Language strategy documented with priorities and workflow
- Save/document outputs appropriately
- Avoid generating content without user input

## CONTEXT BOUNDARIES:
- Available context: Product Brief, brand personality, tone of voice
- Focus: Language requirements and translation approach
- Limits: Not keyword-level SEO yet - language strategy
- Dependencies: Steps 13-15 completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Identify Required Languages

Ask: "What languages does the site need to support?"

For each language:
- Why is it needed? (local audience, tourists, business partners)
- What priority? (primary, secondary, tertiary)
- Full translation or partial?

### 2. Determine Primary Language
- Which language is the "source" language?
- Will content be created first in this language?

### 3. Translation Approach

Options to discuss:
- **Full translation**: All pages in all languages
- **Priority pages**: Key pages translated, others primary only
- **Machine + review**: AI translation with human review
- **Professional translation**: Human translators
- **Client-managed**: Client handles translations

### 4. Localization Considerations

Beyond translation, ask about:
- Date/time formats
- Currency (if applicable)
- Phone number formats
- Address formats
- Cultural considerations

### 5. Tone Consistency Across Languages

Discuss: "Should the tone feel the same in all languages, or adapt to cultural norms?"

Example: German business communication is often more formal than Swedish.

### 6. Document in Output
- Fill in Language Strategy section
- Create language table with priority and coverage
- Document translation approach

### 7. Design Log Update
After completing this step, update the design log:

```markdown
### Step 16: Language Strategy
**Q:** What languages does the site need to support? Translation approach?
**A:** [Languages identified with priorities and coverage]
**Documented in:** content-language.md (Language Strategy section)
**Key insights:** [Translation approach, localization needs]
**Status:** Complete
**Timestamp:** [HH:MM]
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
- Languages identified with priorities
- Primary language defined
- Translation approach documented
- Localization considerations captured
- Tone consistency across languages addressed
- User confirmed

### FAILURE:
- Assumed single language without asking
- Skipped translation approach
- Generated language strategy without user input

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

---
name: 'step-15-tone'
description: 'Define specific tone of voice that expresses brand personality'

# File References
nextStepFile: './step-16-languages.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow.md'
---

# Step 15: Tone of Voice

## STEP GOAL:
Define the specific tone of voice that expresses the brand personality - HOW the personality is expressed in words.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:
- NEVER generate content without user input
- CRITICAL: Read the complete step file before taking any action
- CRITICAL: When loading next step with 'C', ensure entire file is read
- YOU ARE A FACILITATOR, not a content generator
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:
- You are a Strategic Business Analyst guiding tone definition through spectrums and examples
- If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- We engage in collaborative dialogue, not command-response
- You bring structured thinking and facilitation skills, user brings domain expertise and product vision
- Maintain collaborative and strategic tone throughout

### Step-Specific Rules:
- Focus: Tone spectrums, "We Say / We Don't Say" examples, validation with user
- FORBIDDEN: Do not skip validation with actual examples
- Approach: Present spectrums, get positions, create contrasting examples, validate

## EXECUTION PROTOCOLS:
- Primary goal: Tone spectrums defined with positions and examples
- Save/document outputs appropriately
- Avoid generating content without user input

## CONTEXT BOUNDARIES:
- Available context: Product Brief, brand personality from step 14
- Focus: Tone of voice as specific word choices and sentence structures
- Limits: More specific than personality - guides actual word choices
- Dependencies: Step 14 completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Explain the Tone Spectrum

Tone exists on spectrums. Ask the user to position the brand:

| Spectrum | Left | Right |
|----------|------|-------|
| Formality | Formal | Casual |
| Mood | Serious | Playful |
| Complexity | Technical | Simple |
| Energy | Reserved | Enthusiastic |

### 2. For Each Spectrum, Get Position and Example

Ask: "On a scale of 1-5, where 1 is [left] and 5 is [right], where does [business] sit?"

Then: "Can you give me an example of how that sounds?"

### 3. Create "We Say / We Don't Say" Examples

Based on the tone, generate contrasting examples:

| Context | We Say | We Don't Say |
|---------|--------|--------------|
| Greeting | "Hi, how can we help?" | "Dear valued customer..." |
| Problem | "Something went wrong" | "Error 503: Service unavailable" |
| Success | "All done!" | "Your request has been processed" |

### 4. Validate with the User

Present examples and ask:
- "Does this sound like [business name]?"
- "Would [target user] respond well to this?"

### 5. Document in Output
- Fill in Tone of Voice section
- Include spectrum positions with examples
- Add We Say / We Don't Say lists

### 6. Design Log Update
After completing this step, update the design log:

```markdown
### Step 15: Tone of Voice
**Q:** Positioned brand on tone spectrums (formality, mood, complexity, energy)
**A:** [Spectrum positions - e.g., "3/5 formality, 2/5 playful"]
**Documented in:** content-language.md (Tone of Voice section)
**Key insights:** [Key tone characteristics, We Say/Don't Say examples]
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
- Tone spectrums positioned with scores
- "We Say / We Don't Say" examples created
- Examples validated with user
- Tone feels authentic to brand personality
- Documented in output

### FAILURE:
- Skipped spectrum positioning
- Generated examples without user validation
- Tone disconnected from brand personality

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

---
name: 'step-22-references'
description: 'Gather visual references and inspiration sites'

# File References
nextStepFile: './step-23-design-style.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow.md'
---

# Step 22: Visual References

## STEP GOAL:
Gather inspiration and reference sites that represent the desired visual direction.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:
- NEVER generate content without user input
- CRITICAL: Read the complete step file before taking any action
- CRITICAL: When loading next step with 'C', ensure entire file is read
- YOU ARE A FACILITATOR, not a content generator
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:
- You are a Strategic Business Analyst helping articulate visual preferences through references
- If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- We engage in collaborative dialogue, not command-response
- You bring structured thinking and facilitation skills, user brings domain expertise and product vision
- Maintain collaborative and strategic tone throughout

### Step-Specific Rules:
- Focus: Reference sites, specific element preferences, mood keywords, negative references
- FORBIDDEN: Do not accept vague "I like it" without probing for specifics
- Approach: Collect references, probe for specifics on each, include negative references, synthesize mood

## EXECUTION PROTOCOLS:
- Primary goal: Visual references collected with specific preferences and mood keywords
- Save/document outputs appropriately
- Avoid generating content without user input

## CONTEXT BOUNDARIES:
- Available context: Product Brief, existing brand assets, inspiration analysis
- Focus: Visual references and specific element preferences
- Limits: Gathering preferences, not making design decisions
- Dependencies: Step 21 completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Request Reference Sites

Ask: "Are there any websites you like the look of? They don't have to be in the same industry."

For each site provided:
- Visit the URL (use WebFetch if needed)
- Ask: "What specifically do you like about this site?"
- Document the specific elements they're drawn to

### 2. Probe for Specifics

For each reference, identify:
- **Layout:** How content is organized
- **Colors:** Palette, mood, contrast
- **Typography:** Font styles, sizes, weight
- **Imagery:** Photo style, illustrations
- **Effects:** Animations, shadows, interactions
- **Overall feel:** Modern, classic, bold, subtle

### 3. Industry-Specific References

Ask: "Have you seen any [industry] websites that stood out?"

These show expectations within the sector and opportunities to differentiate.

### 4. Negative References

Ask: "Are there any sites or styles you definitely DON'T want?"

Knowing what to avoid is as valuable as knowing what to pursue.

### 5. Synthesize Mood Keywords

Based on references, identify 3-5 mood keywords:
- Example: "Professional, modern, warm, trustworthy, local"

Validate with user: "Would you say the visual direction should feel [keywords]?"

### 6. Document in Output
- Fill in Visual References section
- Add each reference with URL and what we like
- Capture mood description and keywords

### 7. Design Log Update
After completing this step, update the design log:

```markdown
### Step 22: Visual References
**Q:** Reference sites and what specifically you like about them?
**A:** [User responses - summarized]
**Documented in:** visual-direction.md (Visual References section)
**Key insights:** [Important decisions or revelations]
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
- Reference sites collected with specific element preferences
- Negative references captured
- Mood keywords synthesized and validated
- User confirmed mood direction
- Documented in output

### FAILURE:
- Accepted vague preferences without probing
- Skipped negative references
- Generated mood keywords without user validation

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

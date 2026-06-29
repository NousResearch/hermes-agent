---
name: 'step-13-content-init'
description: 'Initialize content and language strategy'

# File References
nextStepFile: './step-14-personality.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow.md'
---

# Step 13: Initialize Content & Language

## STEP GOAL:
Welcome user and set context for defining content and language strategy.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:
- NEVER generate content without user input
- CRITICAL: Read the complete step file before taking any action
- CRITICAL: When loading next step with 'C', ensure entire file is read
- YOU ARE A FACILITATOR, not a content generator
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:
- You are a Strategic Business Analyst helping capture how the brand speaks
- If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- We engage in collaborative dialogue, not command-response
- You bring structured thinking and facilitation skills, user brings domain expertise and product vision
- Maintain collaborative and strategic tone throughout

### Step-Specific Rules:
- Focus: Initialize content & language strategy, check for existing guidelines
- FORBIDDEN: Do not skip the context check for existing brand guidelines
- Approach: Welcome, contextualize, check existing assets, preview the process

## EXECUTION PROTOCOLS:
- Primary goal: Content & Language document initialized, context established
- Save/document outputs appropriately
- Avoid generating content without user input

## CONTEXT BOUNDARIES:
- Available context: Product Brief (positioning, target users)
- Focus: Content and language strategy initialization
- Limits: Not defining personality or tone yet - just setting context
- Dependencies: Steps 1-12 completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Create Output File
- Create `content-language.md` in the output folder using the template
- Initialize frontmatter with `stepsCompleted: []`

### 2. Welcome and Contextualize
- "Let's define how [project name] speaks. This will guide all content - from button labels to marketing copy."
- Reference Product Brief positioning if available

### 3. Quick Context Check
- Ask: "Does the business have any existing brand guidelines or tone of voice?"
- If yes: "Great, let's document and refine them."
- If no: "No problem, we'll create them together."

### 4. Preview the Process
- "We'll cover: brand personality, tone of voice, language requirements, and content guidelines."
- "This usually takes 15-20 minutes."

### 5. Design Log Update
After completing this step, update the design log:

```markdown
### Step 13: Initialization
**Q:** Does the business have existing brand guidelines or tone of voice?
**A:** [yes/no - brief context if yes]
**Documented in:** content-language.md (initialized)
**Key insights:** [Any initial observations about brand context]
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
- Output file created and initialized
- User welcomed with proper context
- Existing guidelines status checked
- Process previewed
- User confirmed readiness

### FAILURE:
- Skipped checking for existing guidelines
- Generated content without user input
- Did not create output file before proceeding

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

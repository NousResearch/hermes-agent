---
name: 'step-14-personality'
description: 'Capture brand personality attributes'

# File References
nextStepFile: './step-15-tone.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow.md'
---

# Step 14: Brand Personality

## STEP GOAL:
Capture the brand's personality attributes that will inform tone of voice.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:
- NEVER generate content without user input
- CRITICAL: Read the complete step file before taking any action
- CRITICAL: When loading next step with 'C', ensure entire file is read
- YOU ARE A FACILITATOR, not a content generator
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:
- You are a Strategic Business Analyst translating business attributes into personality traits
- If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- We engage in collaborative dialogue, not command-response
- You bring structured thinking and facilitation skills, user brings domain expertise and product vision
- Maintain collaborative and strategic tone throughout

### Step-Specific Rules:
- Focus: Brand personality as human characteristics attributed to the brand
- FORBIDDEN: Do not define personality without user input - explore through questions
- Approach: Ask "If the business were a person...", identify 3-5 attributes, connect to target user

## EXECUTION PROTOCOLS:
- Primary goal: 3-5 personality attributes captured with meaning and expression
- Save/document outputs appropriately
- Avoid generating content without user input

## CONTEXT BOUNDARIES:
- Available context: Product Brief, content-language initialization
- Focus: Brand personality attributes
- Limits: Not tone of voice yet - personality informs tone
- Dependencies: Step 13 completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Explore Personality Through Questions

Ask: "If [business name] were a person, how would you describe them?"

Prompt with examples if needed:
- "Friendly and approachable, or professional and reserved?"
- "Innovative and cutting-edge, or reliable and traditional?"
- "Playful and fun, or serious and focused?"

### 2. Identify 3-5 Personality Attributes

Guide the user to articulate specific traits:

| Common Attributes | Description |
|-------------------|-------------|
| **Trustworthy** | Reliable, honest, dependable |
| **Expert** | Knowledgeable, skilled, authoritative |
| **Friendly** | Approachable, warm, welcoming |
| **Professional** | Competent, efficient, polished |
| **Local** | Community-focused, personal, familiar |
| **Innovative** | Modern, forward-thinking, cutting-edge |
| **Straightforward** | Direct, honest, no-nonsense |
| **Helpful** | Supportive, service-oriented, accommodating |

### 3. For Each Attribute, Capture:
- The attribute name
- What it means for this business
- How it's expressed in communication

### 4. Reference the Target User
- "How should [target user] feel when they interact with the brand?"
- Connect personality to user expectations

### 5. Document in Output
- Fill in Brand Personality section
- Create personality summary paragraph

### 6. Design Log Update
After completing this step, update the design log:

```markdown
### Step 14: Brand Personality
**Q:** "If [business] were a person, how would you describe them?"
**A:** [Identified attributes - list them]
**Documented in:** content-language.md (Brand Personality section)
**Key insights:** [Key personality characteristics identified]
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
- 3-5 personality attributes identified
- Each attribute has meaning and expression documented
- Attributes connected to target user expectations
- User confirmed attributes feel right
- Documented in output

### FAILURE:
- Generated personality without user input
- Accepted generic attributes without exploration
- Skipped connecting personality to target user

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

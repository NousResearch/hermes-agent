---
name: 'step-23-design-style'
description: 'Define design style choices using Design Nomenclature vocabulary'

# File References
nextStepFile: './step-24-layout-effects.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow.md'
---

# Step 23: Design Style

## STEP GOAL:
Define specific design style choices using the Design Nomenclature vocabulary to create shared vocabulary between strategy, design, and development.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:
- NEVER generate content without user input
- CRITICAL: Read the complete step file before taking any action
- CRITICAL: When loading next step with 'C', ensure entire file is read
- YOU ARE A FACILITATOR, not a content generator
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:
- You are a Strategic Business Analyst guiding design style decisions with precise vocabulary
- If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- We engage in collaborative dialogue, not command-response
- You bring structured thinking and facilitation skills, user brings domain expertise and product vision
- Maintain collaborative and strategic tone throughout

### Step-Specific Rules:
- Focus: UI visual style, design aesthetic, color direction, typography direction
- FORBIDDEN: Do not make style decisions without presenting rationale based on references and mood
- Approach: Recommend with rationale, confirm or adjust, document decisions

## EXECUTION PROTOCOLS:
- Primary goal: Design style, color direction, and typography direction defined
- Save/document outputs appropriately
- Avoid generating content without user input
- **Reference Documents:** Load as needed: `docs/models/design-nomenclature/ui-visual-styles.md`, `docs/models/design-nomenclature/design-aesthetics.md`, `docs/models/design-nomenclature/color-terminology.md`, `docs/models/design-nomenclature/typography-classification.md`

## CONTEXT BOUNDARIES:
- Available context: Product Brief, existing brand, visual references, mood keywords
- Focus: Design style decisions with precise vocabulary
- Limits: Direction, not final design choices - informing designers
- Dependencies: Step 22 completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Determine UI Visual Style

Based on references and mood, recommend a UI style:

| Style | When to Use |
|-------|-------------|
| **Flat Design** | Clean, simple, content-focused |
| **Material Design** | Structured, Android alignment |
| **Neobrutalism** | Bold, modern, startup feel |
| **Glassmorphism** | Premium, layered, Apple-like |
| **Minimal** | Maximum restraint, luxury |

Present recommendation with rationale:
"Based on your references, I'd recommend [style] because [reasons]."

Confirm or adjust with user.

### 2. Determine Design Aesthetic

Beyond UI style, what era/movement influences apply?

| Aesthetic | Markers |
|-----------|---------|
| **Minimalism** | White space, essential elements |
| **Mid-Century Modern** | Clean lines, organic curves |
| **Service Center** | Practical, trust-focused |
| **Corporate** | Professional, conventional |
| **Local/Artisan** | Warm, personal, handcrafted feel |

### 3. Color Direction

Based on existing brand and preferences:
- Color scheme type: Monochromatic, Complementary, etc.
- Palette direction: Warm/cool, light/dark, saturated/muted
- Any colors to avoid?

### 4. Typography Direction

Based on tone and style:
- Headlines: Geometric, Humanist, Serif?
- Body: Readable, Neo-grotesque?
- Personality: Bold, refined, friendly?

### 5. Document in Output
- Fill in Design Style section
- Fill in Color Direction section
- Fill in Typography Direction section

### 6. Design Log Update
After completing this step, update the design log:

```markdown
### Step 23: Design Style
**Q:** UI style, aesthetic, color direction, typography?
**A:** [User responses - summarized]
**Documented in:** visual-direction.md (Design Style, Color, Typography sections)
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
- UI visual style defined with rationale
- Design aesthetic identified
- Color direction established
- Typography direction set
- User confirmed all decisions
- Documented in output

### FAILURE:
- Made style decisions without rationale from references
- Skipped user confirmation
- Generated design style without user collaboration

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

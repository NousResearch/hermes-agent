---
name: 'step-25-imagery'
description: 'Define photography style, image sources, and imagery guidelines'

# File References
nextStepFile: './step-26-create-visual-document.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow.md'
---

# Step 25: Photography & Imagery

## STEP GOAL:
Define photography style, image sources, and imagery guidelines.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:
- NEVER generate content without user input
- CRITICAL: Read the complete step file before taking any action
- CRITICAL: When loading next step with 'C', ensure entire file is read
- YOU ARE A FACILITATOR, not a content generator
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:
- You are a Strategic Business Analyst helping plan realistic image sourcing while establishing quality standards
- If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- We engage in collaborative dialogue, not command-response
- You bring structured thinking and facilitation skills, user brings domain expertise and product vision
- Maintain collaborative and strategic tone throughout

### Step-Specific Rules:
- Focus: Photography style, existing photos, needs assessment, stock guidelines, icons/illustrations
- FORBIDDEN: Do not skip assessing existing photography quality
- Approach: Discuss style direction, inventory existing photos, identify needs, plan sourcing

## EXECUTION PROTOCOLS:
- Primary goal: Photography and imagery guidelines documented with sourcing plan
- Save/document outputs appropriately
- Avoid generating content without user input

## CONTEXT BOUNDARIES:
- Available context: Product Brief, visual direction (style, layout, effects)
- Focus: Photography and imagery direction
- Limits: Guidelines and sourcing plan, not final image selection
- Dependencies: Step 24 completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Photography Style Direction

Discuss the photographic feel:

| Style | Characteristics |
|-------|-----------------|
| **Authentic/Documentary** | Real people, real work, candid |
| **Professional/Polished** | Staged, high quality, idealized |
| **Lifestyle** | In context, aspirational |
| **Product-focused** | Clean, detailed, technical |

For service businesses, authentic usually works best.

### 2. Existing Photography

Ask: "Do you have photos of the business, team, or work?"
- Quality assessment
- What's usable as-is?
- What needs to be shot?

### 3. Photography Needs

Identify what's needed:
- Hero/header image(s)
- Team/owner photos
- Work/service photos
- Location/environment
- Detail shots

Discuss: "Would you be open to a photoshoot?"

### 4. Stock Photography Guidelines

If stock photos are needed:
- Style consistency (same photographer/collection)
- Authenticity (avoid obviously staged)
- Diversity and representation
- Industry appropriateness

Recommend stock sources:
- Unsplash (free, good quality)
- Pexels (free)
- Shutterstock/Adobe Stock (paid, more options)

### 5. Icon and Illustration Style

If icons or illustrations are needed:
- Line icons vs. filled
- Custom vs. library (Heroicons, Feather, etc.)
- Illustration style if applicable

### 6. Image Guidelines

Document standards:
- Consistent color treatment?
- Aspect ratios for consistency
- Alt text requirements
- Compression/optimization

### 7. Document in Output
- Fill in Photography & Imagery section
- Note image sources and guidelines

### 8. Design Log Update
After completing this step, update the design log:

```markdown
### Step 25: Photography & Imagery
**Q:** Photography style, existing photos, needs, stock guidelines?
**A:** [User responses - summarized]
**Documented in:** visual-direction.md (Photography & Imagery section)
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
- Photography style direction defined
- Existing photos assessed
- Photography needs identified
- Stock guidelines established (if needed)
- Image sourcing plan documented
- User confirmed

### FAILURE:
- Skipped existing photo assessment
- Generated imagery guidelines without user input
- Did not plan image sourcing

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

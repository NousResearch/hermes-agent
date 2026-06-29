---
name: 'step-24-layout-effects'
description: 'Define layout approach and visual effects usage'

# File References
nextStepFile: './step-25-imagery.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow.md'
---

# Step 24: Layout & Effects

## STEP GOAL:
Define layout approach and visual effects usage, keeping platform constraints in mind.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:
- NEVER generate content without user input
- CRITICAL: Read the complete step file before taking any action
- CRITICAL: When loading next step with 'C', ensure entire file is read
- YOU ARE A FACILITATOR, not a content generator
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:
- You are a Strategic Business Analyst guiding layout and effects decisions with performance awareness
- If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- We engage in collaborative dialogue, not command-response
- You bring structured thinking and facilitation skills, user brings domain expertise and product vision
- Maintain collaborative and strategic tone throughout

### Step-Specific Rules:
- Focus: Hero section, content layout, navigation approach, visual effects, performance
- FORBIDDEN: Do not recommend heavy effects without considering mobile performance
- Approach: Discuss options for each area, recommend based on context, consider performance
- **Reference Documents:** Load as needed: `docs/models/design-nomenclature/layout-terminology.md`, `docs/models/design-nomenclature/visual-effects.md`

## EXECUTION PROTOCOLS:
- Primary goal: Layout approach and effects usage defined
- Save/document outputs appropriately
- Avoid generating content without user input

## CONTEXT BOUNDARIES:
- Available context: Product Brief, platform strategy, design style, references
- Focus: Layout patterns and visual effects
- Limits: Direction for designers, not pixel-perfect specs
- Dependencies: Step 23 completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Hero Section Approach

Discuss hero section options:

| Type | Best For |
|------|----------|
| **Full-bleed image** | Strong visual, photography |
| **Split hero** | Image + text, balanced |
| **Text-focused** | Content-first, fast load |
| **Video hero** | Dynamic, engaging |

Recommend based on content type, photography availability, and mobile experience.

### 2. Content Layout Approach

Discuss overall layout structure:
- **Card-based**: Modular, flexible
- **Single column**: Content-focused, blog-like
- **Grid**: Organized, multiple elements
- **Bento box**: Modern, varied modules

### 3. Navigation Approach

Based on site complexity:
- Simple top nav (few pages)
- Hamburger mobile + full desktop
- Mega menu (complex sites)
- Sticky header considerations

### 4. Visual Effects Usage

Discuss appropriate effects:

| Effect | Use Level |
|--------|-----------|
| **Shadows** | Subtle/Medium/Heavy |
| **Animations** | None/Subtle/Rich |
| **Parallax** | None/Subtle/Heavy |
| **Hover effects** | None/Subtle/Interactive |

For mobile-first, simpler is often better.

### 5. Performance Considerations

Note constraints:
- "Tourists on 4G need fast loading"
- "Avoid heavy animations on mobile"
- "Optimize images aggressively"

### 6. Document in Output
- Fill in Layout Direction section
- Fill in Visual Effects section

### 7. Design Log Update
After completing this step, update the design log:

```markdown
### Step 24: Layout & Effects
**Q:** Hero section, layout, navigation, effects preferences?
**A:** [User responses - summarized]
**Documented in:** visual-direction.md (Layout Direction, Visual Effects sections)
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
- Hero section approach defined
- Content layout approach chosen
- Navigation approach determined
- Visual effects usage levels set
- Performance considerations noted
- User confirmed

### FAILURE:
- Recommended heavy effects without performance consideration
- Skipped mobile performance discussion
- Generated layout decisions without user input

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

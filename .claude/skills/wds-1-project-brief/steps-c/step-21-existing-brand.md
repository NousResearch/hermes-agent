---
name: 'step-21-existing-brand'
description: 'Document existing visual identity and brand assets'

# File References
nextStepFile: './step-22-references.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow.md'
---

# Step 21: Existing Brand Assets

## STEP GOAL:
Document any existing visual identity that must be respected or built upon.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:
- NEVER generate content without user input
- CRITICAL: Read the complete step file before taking any action
- CRITICAL: When loading next step with 'C', ensure entire file is read
- YOU ARE A FACILITATOR, not a content generator
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:
- You are a Strategic Business Analyst documenting existing brand assets and constraints
- If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- We engage in collaborative dialogue, not command-response
- You bring structured thinking and facilitation skills, user brings domain expertise and product vision
- Maintain collaborative and strategic tone throughout

### Step-Specific Rules:
- Focus: Inventory assets, assess quality, determine keep/refresh/replace, capture brand constraints
- FORBIDDEN: Do not skip partnership/affiliation visual requirements
- Approach: Inventory each asset type, assess status, document constraints from partnerships

## EXECUTION PROTOCOLS:
- Primary goal: Existing brand assets documented with keep/refresh/replace decisions
- Save/document outputs appropriately
- Avoid generating content without user input

## CONTEXT BOUNDARIES:
- Available context: Product Brief, visual direction initialization
- Focus: Existing visual identity assets and constraints
- Limits: Documenting what exists, not creating new assets
- Dependencies: Step 20 completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Inventory Existing Assets

For each asset type, ask and document:

**Logo:**
- Does a logo exist?
- File formats available? (vector, PNG, etc.)
- Variations? (horizontal, stacked, icon only)
- Quality? (professional, DIY, needs refresh)

**Colors:**
- Are there established brand colors?
- Where are they used? (signage, vehicles, uniforms)
- Are they documented? (hex codes, Pantone)
- Do they need to be maintained?

**Typography:**
- Any fonts already in use?
- On signage, business cards, etc.?

**Imagery:**
- Existing photos of business, team, work?
- Quality level?
- Usage rights?

### 2. Assess Partnership/Affiliation Requirements

Ask: "Are there any partner brands or affiliations that affect the visual identity?"

Examples:
- Franchise requirements
- Certification badges
- Industry associations

Document any visual constraints from partnerships.

### 3. Determine What to Keep vs. Refresh

For each asset:
- **Keep as-is** - Works well, established recognition
- **Refresh** - Good foundation, needs polish
- **Replace** - Doesn't work, starting fresh
- **Create** - Doesn't exist yet

### 4. Collect Assets

If user has assets to share:
- Request files be placed in `visual-references/existing/`
- Or note locations where assets can be obtained

### 5. Document in Output
- Fill in Existing Brand Assets section
- Note brand constraints from partnerships

### 6. Design Log Update
After completing this step, update the design log:

```markdown
### Step 21: Existing Brand Assets
**Q:** What existing visual identity assets exist?
**A:** [User responses - summarized]
**Documented in:** visual-direction.md (Existing Brand Assets section)
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
- All asset types inventoried
- Partnership/affiliation requirements captured
- Keep/refresh/replace decisions made for each asset
- Brand constraints documented
- User confirmed

### FAILURE:
- Skipped partnership/affiliation requirements
- Generated asset decisions without user input
- Did not document brand constraints

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

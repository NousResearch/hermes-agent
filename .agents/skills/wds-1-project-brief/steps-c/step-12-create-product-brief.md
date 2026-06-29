---
name: 'step-12-create-product-brief'
description: 'Compile all captured information and generate the complete Product Brief document'

# File References
nextStepFile: './step-13-content-init.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow.md'
---

# Step 12: Create Product Brief

## STEP GOAL:
Present a cohesive summary of everything captured, get final confirmation, and generate the complete Product Brief document.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:
- NEVER generate content without user input
- CRITICAL: Read the complete step file before taking any action
- CRITICAL: When loading next step with 'C', ensure entire file is read
- YOU ARE A FACILITATOR, not a content generator
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:
- You are a Strategic Business Analyst and synthesizer helping user see the whole picture
- If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- We engage in collaborative dialogue, not command-response
- You bring structured thinking and facilitation skills, user brings domain expertise and product vision
- Maintain collaborative and strategic tone throughout

### Step-Specific Rules:
- Focus: Tell the strategic narrative, not a template-fill exercise
- FORBIDDEN: Do not present as a checklist - present as a coherent story
- Approach: Present narrative, invite reflection, handle adjustments, generate document

## EXECUTION PROTOCOLS:
- Primary goal: Complete Product Brief document generated and confirmed
- Save/document outputs appropriately
- Avoid generating content without user input

## CONTEXT BOUNDARIES:
- Available context: All steps 1-11a completed
- Focus: Synthesis and document generation
- Limits: Not adding new strategic elements - synthesizing what exists
- Dependencies: Steps 1-11a completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Present the Strategic Narrative

**Check context first:**
- If `existing_materials.has_materials = true`: Frame as "Here's the refined strategic foundation..." (acknowledging we built on existing work)
- If `existing_materials.has_materials = false`: Frame as "Here's the strategic foundation we've built..." (fresh creation)

**Tell the story you've heard across all steps:**

> "We've covered a lot of ground. Let me share back the strategic foundation we've built for {product name}:
>
> **The Vision**
> [Vision statement - what this is and why it matters]
>
> **Who It's For**
> [Target users and their context]
>
> **The Problem & Opportunity**
> [What problem exists, what opportunity you're pursuing]
>
> **Positioning**
> [Who it's for, what it is, what makes it different]
>
> **Success Looks Like**
> [Primary success metric + timeline]
>
> **The Reality**
> [Key constraints that shape the solution]
>
> **What Makes You Win**
> [Unfair advantage in competitive landscape]
>
> Does this capture the strategic foundation? Anything that feels off or missing?"

**Key principle:** Present it as a coherent story, not a checklist.

### 2. Handle Reflection & Adjustments

**If user confirms:** Great! Proceed to generate document.

**If user wants adjustments:**
- Listen carefully to what feels off
- Ask clarifying questions: "What would you change about [that element]?"
- Update the affected section
- Re-present the adjusted narrative
- Get confirmation before proceeding

**If user sees gaps:**
- "Good catch - let's address that. Tell me more about [gap]"
- Capture the additional context
- Integrate it into the narrative
- Confirm the updated version

### 3. Generate the Product Brief Document

**Use the template, but make it readable:**

- Write it in clear, natural language (not robotic template-speak)
- Include the strategic narrative from Step 1
- Add all detailed elements in organized sections
- Make it useful for the team (not just documentation for documentation's sake)

**Structure:**
```markdown
# Product Brief: {Product Name}

## Strategic Summary

[2-3 paragraph narrative capturing the essence]

## Vision

[Vision statement + context]

## Positioning

[Full positioning with components]

## Target Users

[Primary user profile(s)]

## Business Model

[B2B/B2C/Both + rationale]

## Success Criteria

[Primary + secondary metrics, timeline]

## Competitive Landscape

[Alternatives, unfair advantage, why you win]

## Constraints & Context

[Timeline, budget, technical, etc.]

## Tone of Voice

[Attributes + examples]

---

**Status:** Product Brief Complete
**Next Phase:** Trigger Mapping (Phase 2)
**Last Updated:** [Date]
```

### 4. Present Completion

**Show the completed brief and celebrate:**

> "Product Brief complete!
>
> I've documented everything in `[output_location]/product-brief.md`
>
> This gives you:
> - Strategic foundation for all design decisions
> - Clear picture of who this is for and why it matters
> - Success metrics to guide prioritization
> - Context for the team to understand the 'why' behind choices
>
> **What's next:**
> - Phase 2: Trigger Mapping (identify key user scenarios)
> - Use this brief to ground all future decisions
>
> Questions about anything in the brief?"

### 5. Update All Dialog Files

**Finalize design log:**

**In `dialog/progress-tracker.md`:**
- Mark ALL steps complete
- Update status to `complete`
- Add completion timestamp
- List final artifact location

**In `dialog/decisions.md`, append:**
```markdown
### Product Brief Synthesis (Step 12)

**Final narrative presented:** [Yes/adjustments made]

**Adjustments during synthesis:**
- [Any changes made during final review]

**User confirmation:** [Confirmed / Refined and confirmed]

**Brief generated:** [Location]

**Completion:** [Timestamp]
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
- Strategic narrative presented as coherent story
- User confirmed or refined the narrative
- Complete Product Brief document generated
- Document is readable and useful (not template-speak)
- All dialog files updated

### FAILURE:
- Presented as checklist instead of narrative
- Generated document without user confirmation
- Skipped reflection/adjustment opportunity

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

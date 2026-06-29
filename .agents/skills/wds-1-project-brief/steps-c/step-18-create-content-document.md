---
name: 'step-18-create-content-document'
description: 'Complete the Content and Language document with actionable guidelines'

# File References
nextStepFile: './step-19-inspiration-workshop.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow.md'
---

# Step 18: Create Content & Language Document

## STEP GOAL:
Complete the Content & Language document and create actionable guidelines that writers and designers can use.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:
- NEVER generate content without user input
- CRITICAL: Read the complete step file before taking any action
- CRITICAL: When loading next step with 'C', ensure entire file is read
- YOU ARE A FACILITATOR, not a content generator
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:
- You are a Strategic Business Analyst finalizing content and language guidelines
- If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- We engage in collaborative dialogue, not command-response
- You bring structured thinking and facilitation skills, user brings domain expertise and product vision
- Maintain collaborative and strategic tone throughout

### Step-Specific Rules:
- Focus: Finalize document with practical guidelines for writers and designers
- FORBIDDEN: Do not skip user confirmation of the final summary
- Approach: Create content type guidelines, document ownership, compile checklist, present summary, confirm

## EXECUTION PROTOCOLS:
- Primary goal: Content & Language document finalized and confirmed
- Save/document outputs appropriately
- Avoid generating content without user input

## CONTEXT BOUNDARIES:
- Available context: Steps 13-17a (personality, tone, languages, SEO, content structure)
- Focus: Synthesis and practical guidelines
- Limits: Finalizing what was captured, not adding major new elements
- Dependencies: Steps 13-17a completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Create Content Type Guidelines

For each content type, provide specific guidance:

**UI Microcopy** (buttons, labels, errors):
- Keep it short
- Use active voice
- Be specific about actions

**Marketing Content** (headlines, features):
- Lead with benefits
- Use power words from tone guide
- Connect to user driving forces

**Informational Content** (services, about):
- Answer user questions directly
- Include relevant keywords naturally
- Maintain consistent tone

### 2. Document Content Ownership

Ask: "Who will create and update content?"

| Content Type | Owner | Frequency |
|--------------|-------|-----------|
| Service descriptions | [owner] | Rarely |
| Blog/news | [owner] | [frequency] |
| Translations | [owner] | As needed |

### 3. Create Writing Checklist

Compile a practical checklist:
- [ ] Tone matches guidelines (warm, professional, etc.)
- [ ] Language is appropriate for target audience
- [ ] Keywords included naturally
- [ ] All languages updated (if multilingual)
- [ ] Spelling and grammar checked
- [ ] Accessible language (no jargon without explanation)

### 4. Present Summary

Show the user a summary:
```
Content & Language Summary
---
Personality:  [key attributes]
Tone:         [description]
Languages:    [list with priorities]
Key Keywords: [top 3-5]
```

### 5. Confirm and Save

Ask: "Does this capture how [business] should sound?"
- Make adjustments as needed
- Finalize document

### 6. Next Steps Guidance

Explain what's next:
- "Content guidelines will inform all UX writing in Phase 4"
- "Keywords will guide SEO implementation"
- Recommend: "Now let's do Visual Direction to establish the visual style"

### 7. Design Log Update
After completing this step, update the design log:

```markdown
### Step 18: Create Content Document
**Q:** Does this capture how [business] should sound?
**A:** [User confirmation, any final adjustments]
**Documented in:** content-language.md (complete)
**Key insights:** [Content ownership, writing checklist created]
**Status:** Complete
**Timestamp:** [HH:MM]
```

**Also update design log completion:**
- Status: `complete`
- Mark content-language.md in Generated Artifacts
- Note: "Ready for Visual Direction workflow"

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
- Content type guidelines created
- Content ownership documented
- Writing checklist compiled
- Summary presented and confirmed by user
- Document finalized and saved

### FAILURE:
- Skipped user confirmation
- Generated guidelines without user collaboration
- Left document incomplete

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

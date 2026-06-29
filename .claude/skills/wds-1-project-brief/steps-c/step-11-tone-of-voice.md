---
name: 'step-11-tone-of-voice'
description: 'Establish the product communication personality and style'

# File References
nextStepFile: './step-12-create-product-brief.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow.md'
---

# Step 11: Define Tone of Voice

## STEP GOAL:
Establish the product's communication personality and style for consistent UI microcopy and system messages throughout the product.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:
- NEVER generate content without user input
- CRITICAL: Read the complete step file before taking any action
- CRITICAL: When loading next step with 'C', ensure entire file is read
- YOU ARE A FACILITATOR, not a content generator
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:
- You are a Strategic Business Analyst and brand guide synthesizing the right voice from product context
- If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- We engage in collaborative dialogue, not command-response
- You bring structured thinking and facilitation skills, user brings domain expertise and product vision
- Maintain collaborative and strategic tone throughout

### Step-Specific Rules:
- Focus: Tone of Voice for UI microcopy, NOT strategic content
- FORBIDDEN: Do not ask the user to define tone of voice - YOU suggest appropriate attributes based on what you've learned, then refine through conversation
- Approach: Analyze product context, suggest attributes, provide examples, refine with user

## EXECUTION PROTOCOLS:
- Primary goal: Tone of voice attributes defined with examples
- Save/document outputs appropriately
- Avoid generating content without user input

## CONTEXT BOUNDARIES:
- Available context: Vision, positioning, Trigger Map, business model, users, success criteria, competitive landscape, constraints, platform strategy
- Focus: Communication personality and microcopy style
- Limits: Tone of Voice is for UI microcopy (buttons, labels, errors, system messages), NOT strategic content (headlines, feature descriptions, value propositions)
- Dependencies: Steps 1-10a completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Analyze Product Context
Review what you've learned:
- Vision & positioning
- Target users and their characteristics
- Business model and customers
- Competitive landscape
- Product category and context

### 2. Suggest Tone of Voice Attributes
Based on the product context, suggest 3-5 tone attributes.

**Present in this format:**

```
Based on [brief reasoning from product context], I suggest this Tone of Voice:

Tone Attributes:
1. [Attribute 1]: [Brief explanation why]
2. [Attribute 2]: [Brief explanation why]
3. [Attribute 3]: [Brief explanation why]
4. [Attribute 4]: [Brief explanation why]

Does this feel aligned with your brand vision?
```

**Example attributes:**
- Friendly & approachable (for consumer products)
- Professional & authoritative (for B2B/enterprise)
- Empathetic & supportive (for healthcare, education)
- Playful & quirky (for creative/youth products)
- Technical & precise (for developer tools)
- Casual & conversational (for social apps)
- Warm & personal (for services)

### 3. Provide Examples
Show the tone in action with side-by-side comparisons.

**Tone of Voice applies to:**
- Form field labels ("Email" vs "Email address" vs "Your email")
- Button text ("Submit" vs "Continue" vs "Let's go")
- Error messages ("Invalid email" vs "Hmm, that doesn't look like an email")
- System messages ("Loading..." vs "Hang tight..." vs "Processing your request")
- Empty states ("No items" vs "Nothing here yet" vs "Your list is empty")
- Tooltips and instructions

**Strategic Content uses Content Creation Workshop instead:**
- Headlines, hero sections, feature descriptions
- Value propositions, testimonials, case studies

**See:** [../data/tone-of-voice-output-template.md](../data/tone-of-voice-output-template.md) for the example format.

### 4. Refine Based on Feedback
**Ask:**
- "Does this tone feel right for your brand?"
- "Should we adjust any attributes? (more/less formal, friendly, technical, etc.)"
- "Are the examples aligned with how you want to communicate?"

**Iterate until confirmed.**

### 5. Document Final Tone of Voice
Once confirmed, document:
- Tone attributes (3-5 clear characteristics)
- Example microcopy showing tone in action
- Do's and Don'ts (brief guidelines)

### 6. Questions to Ask If User Needs Guidance

**"Let me ask a few questions to help define the tone:"**

1. **Relationship:** "How do you want users to feel about your brand? Like a trusted advisor? A helpful friend? An expert authority? A fun companion?"
2. **Formality:** "Should communication be more formal and professional, or casual and conversational?"
3. **Personality:** "If your product were a person, how would they speak? (serious, playful, quirky, straightforward, warm, technical)"
4. **User Context:** "Are users typically stressed/frustrated when using your product, or excited/curious? How should tone respond to their state?"
5. **Differentiation:** "How do competitors communicate? Should you match industry standards or stand out with a different voice?"

### 7. Design Log Update
**Mandatory:** Append to `dialog/decisions.md` if key decisions were made.

**Record:**
- Tone of voice characteristics chosen
- Brand personality decisions
- Communication style rationale

**Then:** Mark Step 11 complete in `dialog/progress-tracker.md` progress tracker

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
- Tone attributes clearly defined (3-5 specific characteristics)
- Attributes align with target users and positioning
- Examples demonstrate the tone clearly
- User confirmed this feels right for their brand
- Tone documented for reference

### FAILURE:
- Simply asked user to define tone without analysis
- Generated tone attributes without product context
- Mixed up UI microcopy tone with strategic content

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

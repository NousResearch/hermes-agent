---
name: 'step-30-contact-strategy'
description: 'Define contact methods and communication strategy'

# File References
nextStepFile: './step-31-multilingual.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow.md'
---

# Step 30: Contact Strategy

## STEP GOAL:
Define how users will contact the business and any special requirements that affect UX design.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:
- NEVER generate content without user input
- CRITICAL: Read the complete step file before taking any action
- CRITICAL: When loading next step with 'C', ensure entire file is read
- YOU ARE A FACILITATOR, not a content generator
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:
- You are a Strategic Business Analyst defining contact strategy that affects UX design and technical integrations
- If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- We engage in collaborative dialogue, not command-response
- You bring structured thinking and facilitation skills, user brings domain expertise and product vision
- Maintain collaborative and strategic tone throughout

### Step-Specific Rules:
- Focus: Primary contact method, channels, form requirements, booking/scheduling, AI integration opportunity
- FORBIDDEN: Do not skip capturing UX implications of contact decisions
- Approach: Identify primary method, explore phone/form needs, discuss AI opportunity, document UX constraints

## EXECUTION PROTOCOLS:
- Primary goal: Contact strategy documented with UX implications
- Save/document outputs appropriately
- Avoid generating content without user input

## CONTEXT BOUNDARIES:
- Available context: Product Brief, technology stack, integrations
- Focus: Contact strategy and UX implications
- Limits: Strategy, not detailed form design
- Dependencies: Step 29 completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Identify Primary Contact Method

Ask: "How do you primarily want customers to reach you?"
- **Phone** - Click-to-call, prominent display
- **Form** - Contact form with fields
- **Email** - Direct email link
- **Booking system** - Online scheduling
- **Chat** - Live chat or chatbot
- **Combination** - Multiple methods

### 2. For Phone-Primary Businesses:
- Phone number placement (header, hero, footer, sticky?)
- Click-to-call on mobile
- Business hours display
- After-hours handling

### 3. For Form-Based Contact:
- Required fields
- Optional fields
- Spam protection (CAPTCHA, honeypot)
- Response expectations
- Where submissions go (email, CRM?)

### 4. AI Integration Opportunity

If relevant, discuss:
- "Have you considered AI-assisted phone handling?"
- Explain: AI can answer calls, triage urgent vs routine, book appointments
- Note as future integration if interested

### 5. Document UX Implications

Capture constraints for UX design:
- "Phone must be visible without scrolling"
- "Contact form should be accessible from every page"
- "No online booking - phone/form only"

### 6. Update Output Document
- Fill in Contact Strategy section
- Note UX Constraints

### 7. Design Log Update
After completing this step, update the design log:

```markdown
### Step 30: Contact Strategy
**Q:** Primary contact method? UX implications?
**A:** [User responses - summarized]
**Documented in:** platform-requirements.md (Contact Strategy section)
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
- Primary contact method identified
- Channel requirements documented
- UX implications captured
- AI opportunity discussed (if relevant)
- User confirmed

### FAILURE:
- Skipped UX implications
- Generated contact strategy without user input
- Did not capture form requirements (if applicable)

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

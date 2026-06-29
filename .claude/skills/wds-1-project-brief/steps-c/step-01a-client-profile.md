---
name: 'step-01a-client-profile'
description: 'Capture who the client is as an organisation and as people — not their product goals, but themselves'

# File References
nextStepFile: './step-02-vision.md'
workflowFile: '../workflow.md'
---

# Step 1a: Client Profile

## STEP GOAL:
Understand the client as an organisation and as people. This is NOT about their product or their customers — it's about who we are working with, how they operate, and what drives them internally.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:
- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:
- ✅ You are Saga, building a working relationship — not interrogating the client
- ✅ Keep the tone warm and curious, not clinical
- ✅ Many answers will come naturally from conversation — don't ask mechanically through a checklist
- ✅ The goal is a picture of the organisation and the people, not a form filled in

### Step-Specific Rules:
- 🎯 Focus on the client as organisation and humans — NOT on their product, vision, or target users (those come later)
- 🚫 FORBIDDEN to ask about product vision or positioning here
- 💬 Approach: Conversational. One topic at a time. Build on what they say.
- 📋 If answers came up naturally during init (step-01), carry them forward — do not re-ask

## EXECUTION PROTOCOLS:
- 🎯 Build a clear picture across four areas: Organisation, People, Working Style, Internal Driver
- 💾 Write completed profile to `dialog/client-profile.md` using the client-profile template
- 🚫 Do not confuse "business customers" (their customers) with the client organisation itself

## CONTEXT BOUNDARIES:
- Available context: Project config, any context from step-01 init
- Focus: The client organisation and the humans commissioning this project
- Limits: Not their product, not their end users, not their market — those are next
- Dependencies: Step 01 complete

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 0. Check Prior Context

Before asking anything, review what is already known from step-01:
- Did the user mention their role or organisation during init?
- Did they provide any materials that reveal organisation type or stakeholder structure?

If information is already confirmed: acknowledge it, do not re-ask. Only fill gaps.

### 1. Organisation

Explore conversationally — cover these areas, not necessarily in this order:

- **Type**: Startup, scale-up, established SME, enterprise, NGO, public sector, internal product team?
- **Size**: Rough headcount or team size
- **Industry and context**: What world do they operate in?
- **Tech maturity**: Have they built digital products before? Do they have an internal tech team?
- **Design maturity**: Have they worked with designers or a design process before? What went well or not?

### 2. The People

- **Who is ordering this project?** Name, role, and mandate — can they make decisions, or do they need sign-off from above?
- **Is there a champion?** Someone internally who is driving this — may or may not be the same person
- **Technical contact**: Who owns the tech side on their end?
- **Other stakeholders**: Who else will have opinions or approval rights? (Board, investors, other departments?)
- **Decision culture**: Do decisions get made fast by one person, or does everything go through consensus and committees?

### 3. Internal Driver

- **What triggered this project?** (New leadership, lost clients, investor pressure, a competitor move, a long-standing frustration finally reaching a tipping point?)
- **What does success look like for THEM — politically and personally**, not just for the product? (The champion getting credit, the board getting proof of innovation, the team finally having something they're proud of?)
- **Is there a deadline that matters for internal reasons** beyond the product launch?

### 4. Working Style

- **Communication preference**: How do they prefer to communicate and how fast do they respond?
- **Timeline culture**: Do they move fast and iterate, or do they have longer approval cycles?
- **Prior agency experience**: Have they worked with an external studio before? What was good or bad about it?

### 5. Write Client Profile

Create `dialog/client-profile.md` using the template at `../templates/client-profile.template.md`.

Fill in what was confirmed. Mark genuinely unknown fields as `—` — do not guess.

### 6. Design Log Update

**Mandatory:** Append key decisions and context to `dialog/decisions.md`.

Record: Organisation type, key people and roles, decision culture, internal project driver.

Mark Step 1a complete in `dialog/progress-tracker.md`.

### 7. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Vision"

#### Menu Handling Logic:
- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE
ONLY WHEN client profile is documented and user confirms will you then load and read fully `{nextStepFile}`.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Organisation type and maturity captured
- Key people and their roles/mandates identified
- Decision culture understood
- Internal driver for the project documented
- `dialog/client-profile.md` written
- Design log updated

### ❌ SYSTEM FAILURE:
- Asked about product vision or target users in this step
- Generated profile content without user input
- Re-asked questions already answered in step-01
- Confused the client's customers with the client themselves
- Skipped writing `dialog/client-profile.md`

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

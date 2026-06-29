---
name: 'step-05-structural-order'
description: 'Apply the Golden Circle to create persuasive WHY-HOW-WHAT content flow'
nextStepFile: './step-06-generate-content.md'
---

# Step 5: Determine Structural Order

## STEP GOAL:

Apply Simon Sinek's Golden Circle to sequence all content from previous steps into a persuasive WHY-HOW-WHAT flow that moves users emotionally first, then logically, then to action.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a strategic content architect applying Golden Circle methodology
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring structural persuasion expertise, user brings their content priorities
- ✅ Maintain a clear, structured tone

### Step-Specific Rules:

- 🎯 Focus ONLY on sequencing content into WHY-HOW-WHAT structure
- 🚫 FORBIDDEN to generate final content text in this step
- 💬 Map all essential information from previous steps to WHY, HOW, or WHAT
- 📋 Validate the persuasive flow end-to-end before proceeding

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Document the structural order in structured format
- 📖 Reference all content elements from Steps 3-4 (Action Filter + Empowerment Frame)
- 🚫 FORBIDDEN to proceed without a validated WHY-HOW-WHAT structure

## CONTEXT BOUNDARIES:

- Available context: Purpose (Step 0), Trigger Map (Step 1), Awareness (Step 2), Action Filter (Step 3), Empowerment Frame (Step 4)
- Focus: Sequencing existing content elements into persuasive order
- Limits: Do not write final content — organize the structure for it
- Dependencies: All previous steps provide the content elements to sequence

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Identify the WHY

Ask: **"What's the emotional opening that connects to their driving forces?"**

Opens with user's current emotional state, connects to driving forces from the Trigger Map, makes them care before explaining the solution.

### 2. Identify the HOW

Ask: **"What's the method that bridges emotional need to specific solution?"**

Explains the approach, shows how transformation happens, uses capability framing from Step 4, contains the "aha moment" insight.

### 3. Identify the WHAT

Ask: **"What are the concrete specifics and call to action?"**

Names the product/offer, provides social proof, clear CTA with capability framing, risk removal.

### 4. Map Content to Structure

Present all content elements from Steps 3-4. Work together to assign each piece to WHY (emotional opening), HOW (method/bridge), or WHAT (specifics/proof).

### 5. Sequence Within Sections

Within each section, determine the most persuasive order:

- **WHY section:** Problem → Validation → Aspiration
- **HOW section:** Approach → Differentiator → Transformation
- **WHAT section:** Naming → Proof → Action → Risk Removal

### 6. Validate Persuasive Flow

Ask: **"Does WHY → HOW → WHAT create natural emotional → logical → action flow?"**

- Can user understand WHY without knowing WHAT yet?
- Does HOW bridge the gap naturally?
- Does WHAT feel like a natural conclusion (not premature)?

### 7. Document Structural Order

```yaml
structural_order:
  section_why:
    purpose: "Emotional truth / Why user should care"
    content_elements:
      - order: 1
        element: "[Opening hook]"
        rationale: "[Why this opens]"
      - order: 2
        element: "[Validation or aspiration]"
        rationale: "[Why this comes second]"
  section_how:
    purpose: "Method / Bridge from emotion to specifics"
    content_elements:
      - order: 1
        element: "[Solution approach]"
        rationale: "[Why this bridges first]"
      - order: 2
        element: "[Key differentiator]"
        rationale: "[Why this matters here]"
      - order: 3
        element: "[Transformation path]"
        rationale: "[Why this comes last in HOW]"
  section_what:
    purpose: "Specifics / Proof / Action"
    content_elements:
      - order: 1
        element: "[Product/offer name]"
        rationale: "[Why we can name it now]"
      - order: 2
        element: "[Social proof]"
        rationale: "[Why proof comes here]"
      - order: 3
        element: "[CTA]"
        rationale: "[Why action comes last]"
  flow_validation:
    feels_natural: "[yes/no + notes]"
    persuasive_arc: "[Does WHY → HOW → WHAT create emotional → logical → action flow?]"
```

### 8. Present MENU OPTIONS

Display: **"Select an Option:** [C] Continue"

#### Menu Handling Logic:

- IF C: Save structural order, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#8-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions — always respond and then end with display again of the menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN C is selected and the structural order is documented will you load {nextStepFile} to begin generating and reviewing content.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- WHY is identified (emotional opening, purpose)
- HOW is identified (method, bridge, differentiator)
- WHAT is identified (specifics, proof, CTA)
- All essential information assigned to WHY, HOW, or WHAT
- Content sequenced within each section
- Flow feels natural and persuasive

### ❌ SYSTEM FAILURE:

- Generating final content text in this step
- Putting WHAT before WHY (salesy, pushy)
- Missing the WHY section entirely (cold, transactional)
- Not mapping all essential information to a section
- Not waiting for user input at menu

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

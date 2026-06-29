---
name: step-05-outline-scenario
description: Create detailed outline for ONE scenario, repeating for each in the approved plan

# File References
nextStepFile: './step-06-generate-overview.md'

# Data References
scenarioTemplate: '../data/scenario-outline-template.md'
---

# Step 5: Outline Scenario (One at a Time)

## STEP GOAL:

Define ONE scenario through 8 strategic questions in natural conversation order. Start with the primary transaction (highest priority), complete it fully, then loop for each remaining scenario. A **transaction** is any meaningful user journey — purchasing, booking, researching content page-by-page, comparing options, or any interaction where the user moves through the site with intent.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a UX Scenario Facilitator — you ASK, the user DECIDES
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring scenario thinking and user journey expertise, user brings their project knowledge, together we create concrete UX scenario outlines
- ✅ Maintain collaborative equal-partner tone throughout

### Step-Specific Rules:

- 🎯 Focus on ONE transaction at a time, complete it fully before moving to the next
- 🚫 FORBIDDEN to skip any of the 8 strategic questions
- 💬 Approach: Ask one question at a time, let the answer shape the next question naturally
- 📋 Verify all quality gates before proceeding to the next scenario or step

## EXECUTION PROTOCOLS:

- 📖 Load the scenario outline template before starting
- 💬 Walk through 8 questions as a dialog — one question at a time, building on each answer
- ✅ Run quality gates check before moving on
- 💾 Create output file in the correct folder structure
- 🔄 Loop back for each remaining scenario (next transaction, next target group)
- 🚫 FORBIDDEN to proceed if any quality gate fails

## CONTEXT BOUNDARIES:

- Available context: Approved scenario plan from Step 4, strategic context, page inventory, Trigger Map
- Focus: Detailed outlining of one scenario at a time
- Limits: Only outline scenarios from the approved plan
- Dependencies: User-approved scenario plan from Step 4

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Determine Which Scenario

Process scenarios in priority order (Priority 1 first, then 2, then 3).

If this is your first time at this step, start with scenario 01.
If returning from a loop, continue with the next unfinished scenario.

### 2. Load Template

Load the full template: `{scenarioTemplate}`

### 3. The 8-Question Scenario Dialog

**Two modes — same 8 questions, different driver:**

- **Conversation mode** (default): YOU ask, the USER answers. One question at a time. Each answer shapes the next question naturally.
- **Suggest mode** (when user asks you to suggest): YOU answer all 8 questions based on the Trigger Map, Product Brief, and strategic context. Present the complete scenario to the user for review and adjustment.

This IS the scenario — when all 8 are answered, the outline writes itself.

> **What counts as a transaction:** Not just purchases or bookings. Clicking through a menu item by item to research site content is a transaction. Comparing options is a transaction. Any meaningful journey where the user moves through the site with intent.

#### Q1: "What transaction do we need to get really right?"

Start with the WHY. What is the most important thing a user needs to accomplish on this site?

- State as user purpose, not feature name
- **Bad:** "Homepage and service pages"
- **Good:** "Verify service availability before booking"

#### Q2: "If this transaction succeeds, which business goal does it add value to?"

Connect to the Trigger Map immediately. Which specific business goal and objective does this serve?

- Reference actual goals from the Trigger Map
- This grounds the scenario in business strategy, not just user needs

#### Q3: "Which user experiences this most, and in what real-life situation?"

Identify the persona AND their context. Not just "who" but "who, where, when."

- Use actual personas from the Trigger Map
- **Bad:** "A customer looking for information"
- **Good:** "Hasse, 55, motorhome tourist stranded in Byxelkrok with a broken vehicle during family vacation"

#### Q4: "What do they want and what do they fear going into this interaction?"

The driving forces — hope and worry. These must be visceral and specific.

- **Hope:** What they're hoping to find or achieve
- **Worry:** What they're afraid of or want to avoid
- **Bad:** "User is interested in the product"
- **Good:** "Hope: Find trustworthy mechanic nearby, get back on road today. Worry: Being stranded for days, getting ripped off by unknown mechanic"
- **Length Rule:** ONE sentence max per component. Phrases, not paragraphs.

#### Q5: "What device are they on?"

Mobile, desktop, or tablet. This shapes the entire design approach.

#### Q6: "What's the natural starting point — how do they actually arrive?"

How the user ACTUALLY gets to the site. Be specific about discovery method.

- **Bad:** "User opens the website"
- **Good:** "Googles 'car repair Öland' on mobile while parked at gas station, clicks top organic result"
- **Length Rule:** 1-2 sentences max. Device + context + discovery method.

#### Q7: "What does the best possible outcome look like — for both sides?"

Mutual success — user AND business. Both specific and measurable.

- **User Success:** Tangible outcome the user achieves
- **Business Success:** Measurable result for the business
- **Bad:** User: "Successfully use the site" / Business: "Get more customers"
- **Good:** User: "Confirmed mechanic fixes motorhomes, has location and hours, feels confident calling" / Business: "High-intent tourist call captured, positioned as emergency-capable, info call avoided"

#### Q8: "What's the shortest path through the site to get there?"

The linear sunshine path. Numbered steps, each with page name + what the user accomplishes.

**Rules:**
- Completely linear — ZERO "if" statements, ZERO branches
- Minimum viable steps — can you remove any step without breaking the flow?
- Each step moves meaningfully toward success

**Format:**
```
1. **[Page Name]** — [What user sees/does/achieves here]
2. **[Page Name]** — [What user sees/does/achieves here]
3. **[Page Name]** — [What user sees/does/achieves here] ✓
```

### 4. Name the Scenario

After the 8 questions, name the scenario using the persona:

- **Name:** Persona name + purpose (e.g., "Hasse's Emergency Search")
- **ID:** 01, 02, etc.
- **Slug:** `01-hasses-emergency-search`

### 5. Quality Gates (Check Before Moving On)

Before proceeding, verify the scenario outline:

- [ ] All 8 questions answered with specific, concrete responses
- [ ] Mental state is visceral and specific (not generic "interested")
- [ ] Entry point is realistic with device + context + discovery method
- [ ] Path is truly linear (zero "if" statements)
- [ ] Both successes are specific and measurable (not vague)
- [ ] Scenario name includes persona name
- [ ] Trigger Map connection is explicit (persona + business goal)

**If any gate fails:** Fix before proceeding.

### 6. Create the Scenario File

1. Create folder: `{output_folder}/C-UX-Scenarios/[NN-slug]/`
2. Create file: `{output_folder}/C-UX-Scenarios/[NN-slug]/[NN-slug].md`
3. Use the template from data/ to structure the content from the 8 answers

### 7. After Scenario Creation — Outline Scenario Steps

After the scenario file is saved (Q1-Q8 answered, quality gates passed), begin outlining scenario steps from the Q8 shortest path.

#### Automatic First Step

Process the first scenario step from Q8 automatically:
1. Name it using Q8's first step name
2. Create the page folder (see Page Folder Structure below)
3. Fill first-step metadata from Q3 (user situation), Q4 (mental state), Q5 (device), Q6 (entry point)
4. Present the result to the user

Then show the Scenario Step Menu.

#### Scenario Step Menu

After each scenario step is outlined, present:

```
Step [NN.X] "[step-name]" outlined!

1. Outline next scenario step — [next step name from Q8]
2. Start designing — enter the design loop from step 1

---
[N] Define the next scenario
[C] Continue to overview (when all scenarios are done)
```

**Adaptive labels:**
- Option 1 shows the name of the next step from Q8
- When all Q8 steps are outlined: Option 1 becomes unavailable — show "All [X] scenario steps outlined!"
- Option 2: **"Start designing"** when only 1 step is outlined. **"Start designing pages"** when 2+ steps are outlined.

#### Menu Handling Logic:

- **IF 1 (Outline next):** Ask the two questions for the next step (see Per-Step Dialog below), create the folder, then show this menu again.
- **IF 2 (Start designing):** Hand over to Phase 4 (UX Design) → Discuss activity. Phase 4 handles the creative dialog (D1, D2) and all design decisions. The design loop always starts from scenario step 1, regardless of how far the outline has progressed.
- **IF N:** Loop back to instruction 1 for the next scenario. The current scenario's remaining steps can be outlined later.
- **IF C:** Load, read entire file, then execute {nextStepFile} (only when all planned scenarios are complete).

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions — always respond and then redisplay the menu
- The first step is processed automatically after scenario creation (no menu prompt first)

#### Per-Step Dialog

For each step after the first, refine Q8's outline into a concrete scenario step:

**1. "What's the point of this step?"**

What does the user need to accomplish here? This becomes the step purpose.
- e.g., "See a list of news articles" or "Find the phone number and opening hours"

**2. "What does the user do to move forward?"**

What interaction takes them to the next step? This defines the exit action.
- e.g., "Selects 'News' in the menu" → next step
- e.g., "Clicks 'Read more' on an article" → next step

**Two types of interactions:**
- **Leaves the step** → new scenario step (new page folder, next number)
- **Stays on the step** → storyboard item (documented within the page spec as an on-page interaction)

After answering, create the page folder and return to the Scenario Step Menu.

### Page Folder Structure

**Naming convention:** `{scenario-number}.{step-number}-{page-slug}` (e.g., `1.1-start-page`, `1.2-news-listing`, `1.3-article-detail`)

Each page folder contains:

```
{NN}.{step}-{page-slug}/
├── {NN}.{step}-{page-slug}.md
└── Sketches/
```

#### Page boilerplate:

```markdown
# {NN}.{step}-{page-name}

## Page Metadata

| Property | Value |
|----------|-------|
| **Scenario** | {scenario-name} |
| **Page Number** | {NN}.{step} |
| **Platform** | {Device from Q5} |

## Overview

**Page Purpose:** {What the user needs to accomplish here}

**Entry Context:** {How the user arrived — previous page + interaction that brought them here}

**Exit Action:** {What the user does to move to the next step}

**On-Page Interactions:**
- {Any interactions that keep the user on this page — storyboard items}
```

The **first step** additionally includes:
- **User Situation** from Q3
- **Mental State** (hope + worry) from Q4
- **Discovery Method** from Q6

## CRITICAL STEP COMPLETION NOTE

When [C] is selected, ALL scenarios from the approved plan must be outlined and pass quality gates. Then load and read fully `{nextStepFile}` to begin generating the overview.

When **Start designing** is selected, hand over to Phase 4 with the current scenario context. The design loop starts from scenario step 1. The user can return to Phase 3 later for remaining scenarios or steps.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- All 8 questions answered for each scenario with specific, concrete responses
- All quality gates pass for every scenario
- Scenario outline file created in correct folder structure
- Scenarios processed in priority order (primary transaction first, then secondary, etc.)
- All scenarios from approved plan completed before proceeding
- Conversation mode: Dialog felt like a natural conversation, not a form to fill
- Suggest mode: All 8 answers grounded in actual Trigger Map/Brief data, presented for user review
- First scenario step processed automatically after scenario creation
- User presented with clear two-option flow after each step (outline next / start designing)
- "Start designing" always begins from scenario step 1

### ❌ SYSTEM FAILURE:

- Skipping any of the 8 strategic questions
- Conversation mode: Presenting all questions at once instead of one at a time
- Suggest mode: Not presenting answers for user review before proceeding
- Proceeding with failing quality gates
- Skipping scenarios from the approved plan
- Using generic mental states or vague success goals
- Creating branching paths instead of linear sunshine paths
- Not creating output files
- Not automatically processing the first scenario step after scenario creation
- Starting the design loop from a step other than step 1
- Presenting the old [N]/[O]/[D]/[C] menu instead of the simplified two-option flow

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

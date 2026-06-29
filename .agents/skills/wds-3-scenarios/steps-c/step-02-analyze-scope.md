---
name: 'step-02-analyze-scope'
description: 'Determine site type, list all pages/views, assess scale, and select approach mode'

# File References
nextStepFile: './step-03-build-strategic-context.md'
---

# Step 2: Analyze Scope & Scale Strategy

## STEP GOAL:

Determine site type, list all pages/views, assess scale, select approach mode, and present the analysis for user approval at this critical checkpoint.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a UX Scenario Facilitator collaborating with the project owner
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring scenario thinking and user journey expertise, user brings their project knowledge, together we create concrete UX scenario outlines
- ✅ Maintain collaborative equal-partner tone throughout

### Step-Specific Rules:

- 🎯 Focus only on scope analysis, page inventory, and scale strategy
- 🚫 FORBIDDEN to proceed past the user checkpoint without explicit user approval
- 💬 Approach: Present structured analysis and wait for user confirmation
- 📋 This is a USER CHECKPOINT — do not auto-proceed

## EXECUTION PROTOCOLS:

- 🔍 Classify site type based on Product Brief data
- 📋 Create complete page inventory from all sources
- 📊 Assess scale and recommend approach mode
- 🚫 FORBIDDEN to skip user checkpoint — must wait for explicit approval

## CONTEXT BOUNDARIES:

- Available context: Product Brief data, Trigger Map data loaded in Step 1
- Focus: Site classification, page inventory, scale assessment
- Limits: No scenario creation, no strategic context building — only scope analysis
- Dependencies: Step 1 context must be loaded

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Site Type Detection

Based on Product Brief, classify the site:

**Presentation Site** (marketing, service catalog, company profile, portfolio):
- Scenario format: **Screen Flow** (page-to-page navigation)
- Coverage: Expose all pages
- Storyboarding: Minimal (only for complex interactions like booking forms)

**Dynamic App** (SaaS, booking system, social platform, productivity tool):
- Scenario format: **Storyboard** (document states within views)
- Coverage: Focus on core workflow first
- Screen flow: Only for multi-step processes (onboarding, checkout)

**Mixed** (presentation site with dynamic features):
- Use both formats as needed per scenario

### 2. List All Pages/Views

Create a complete list of every page or view from the Product Brief.

**Format:**
```
## Page Inventory

1. [Page Name] — [Brief purpose]
2. [Page Name] — [Brief purpose]
3. [Page Name] — [Brief purpose]
...

**Total: [N] pages/views**
```

**Rules:**
- Include every page mentioned in Product Brief
- Include pages implied by navigation structure
- Include pages implied by business goals (e.g., if goal mentions "booking" there's a booking page)
- Do NOT include generic shared elements (header, footer, navigation) — these are documented separately

### 3. Scale Assessment

Based on page count, determine strategy:

**Small (< 20 pages):**
- Strategy: **Comprehensive coverage** — document all pages across scenarios
- Mode recommendation: **Dream** or **Suggest**
- Every page must appear in exactly one scenario

**Medium (20-50 pages):**
- Strategy: **Comprehensive coverage** with natural groupings
- Mode recommendation: **Suggest**
- Group pages by navigation patterns, service types, or content categories

**Large (100+ pages):**
- Strategy: **Selective ignorance** — focus on most valuable workflow
- Mode recommendation: **Dialog**
- Deep work on business-critical flow (learning effect reveals patterns for rest)

### 4. Page Documentation Strategy

Determine how to handle similar pages:

**Few pages + high variation** → Document as separate pages
- Each page substantially different in structure, content, or purpose
- Example: 13 vehicle pages with different positioning

**Many pages + low variation** → Document as template with content variations
- Structurally identical pages with only content differences
- Example: 500 product pages with same layout, different product data

### 5. Present Analysis (USER CHECKPOINT)

Present to user and **wait for approval**:

```
## Scope Analysis

**Site Type:** [Presentation / Dynamic / Mixed]
**Total Pages:** [N]
**Scale:** [Small / Medium / Large]
**Recommended Mode:** [Dream / Suggest / Dialog]
**Scenario Format:** [Screen Flow / Storyboard / Mixed]

### Page Inventory
[numbered list from step 2]

### Page Strategy
- [X] pages documented individually (high variation)
- [Y] pages as templates (low variation groups: [list groups])

**Does this look right? Any pages missing or that should be grouped differently?**
```

**WAIT for user response.** Do not proceed until user confirms.

### 6. Present MENU OPTIONS

Display: "Are you ready to [C] Continue to Building Strategic Context?"

#### Menu Handling Logic:

- IF C: Load, read entire file, then execute {nextStepFile}

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- After other menu items execution, return to this menu
- User can chat or ask questions - always respond and then end with display again of the menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN [C continue option] is selected and [user has explicitly approved the scope analysis], will you then load and read fully `{nextStepFile}` to execute and begin building strategic context.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Site type correctly classified from Product Brief data
- Complete page inventory created with all pages accounted for
- Scale assessment matches page count
- Page documentation strategy determined
- Analysis presented clearly at user checkpoint
- User explicitly approves before proceeding
- Menu presented and user input handled correctly

### ❌ SYSTEM FAILURE:

- Proceeding without user approval at checkpoint
- Missing pages from the inventory
- Incorrect site type classification
- Skipping scale assessment or mode recommendation
- Auto-proceeding past the user checkpoint

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

---
name: 'step-03-generate-specs'
description: 'Create WDS-format page specifications from the observations captured in Step 02'

# File References
nextStepFile: './step-04-extract-design-system.md'
---

# Step 3: Generate Specs

## STEP GOAL:

Create WDS-format page specifications from the observations captured in Step 02. Generate all output in `document_output_language`.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are an Implementation Partner guiding structured development activities
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring software development methodology expertise, user brings domain knowledge and codebase familiarity
- ✅ Maintain clear and structured tone throughout

### Step-Specific Rules:

- 🎯 Focus only on prioritizing pages, generating page specifications, creating scenario outlines, and cross-referencing with components
- 🚫 FORBIDDEN to begin extracting the design system — that is the next step
- 💬 Approach: Generate specifications collaboratively, starting with foundational pages
- 📋 Every component mentioned in specs must appear in the component inventory from Step 02

## EXECUTION PROTOCOLS:

- 🎯 Complete page specifications and scenario outlines generated in WDS format
- 💾 Save all specifications to the output folder
- 📖 Reference page inventory and component inventory from Step 02
- 🚫 Do not create the design system document yet

## CONTEXT BOUNDARIES:

- Available context: All inventories from Step 02 (pages, components, colors, typography, spacing)
- Focus: Spec generation — page specs, scenario outlines, cross-referencing
- Limits: No design system extraction
- Dependencies: Step 02 must be complete (inventories captured)

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Prioritize Pages

From the page inventory (Step 02), decide which pages to spec first:

- Start with the most representative or foundational pages
- Group pages that share layout patterns
- Skip pages that are nearly identical (spec one, note variations)

### 2. Generate Page Specification for Each Page

For each page, create a specification document following the WDS page spec structure:

- Overview (purpose, URL, type)
- Layout Structure (desktop layout, sections top to bottom)
- Component List (component, location, variant, notes)
- Content Strategy (headline pattern, body copy, CTA language, images)
- Responsive Behavior (breakpoint changes)
- Interactions (hover, click, scroll, form validation, modal triggers)

### 3. Create Scenario Outlines from User Flows

If the target has multi-page flows (sign up, checkout, onboarding), document them as scenario outlines with steps, success path, and error states.

### 4. Cross-Reference with Components

Ensure every component mentioned in page specs appears in the component inventory from Step 02. Flag any components that appear in specs but were not captured.

### 5. Save Output

Save all specifications to the output folder using consistent naming:

```
output/
  specs/
    page-home.md
    page-about.md
    page-products.md
  scenarios/
    scenario-checkout.md
    scenario-onboarding.md
```

### 6. Verify Checklist

- [ ] Pages prioritized and ordered
- [ ] Page specification created for each key page
- [ ] Layout structure documented (desktop and sections)
- [ ] Component list matches component inventory from Step 02
- [ ] Content strategy noted per page
- [ ] Responsive behavior described
- [ ] User flow scenarios documented (if applicable)
- [ ] All output generated in `document_output_language`
- [ ] Specs saved to output folder

### 7. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Step 4: Extract Design System"

#### Menu Handling Logic:
- IF C: Update design log, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN all page specifications and scenario outlines are generated and saved will you then load and read fully `{nextStepFile}` to execute.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Pages prioritized and ordered
- Page specification created for each key page
- Layout structure documented
- Component list matches inventory
- Content strategy noted
- Responsive behavior described
- Specs saved to output folder

### ❌ SYSTEM FAILURE:
- Beginning design system extraction before specs are complete
- Missing pages in specifications
- Component list does not match inventory
- Not saving output

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

---
name: 'step-04-extract-design-system'
description: 'Consolidate all design tokens and component patterns into a structured WDS-compatible design system document'

# File References
activityWorkflowFile: '../workflow-reverse-engineering.md'
---

# Step 4: Extract Design System

## STEP GOAL:

Consolidate all design tokens and component patterns into a structured WDS-compatible design system document.

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

- 🎯 Focus only on extracting design tokens, cataloging components, documenting variants/states, mapping tokens to components, and creating the design system document
- 🚫 FORBIDDEN to modify page specifications — they are final from Step 03
- 💬 Approach: Organize raw values into a structured token system with user input
- 📋 Token-to-component mapping must be explicit

## EXECUTION PROTOCOLS:

- 🎯 Complete design system document in WDS format
- 💾 Save design system document to output folder alongside page specs
- 📖 Reference inventories from Step 02 and page specs from Step 03
- 🚫 Do not modify existing page specifications

## CONTEXT BOUNDARIES:

- Available context: All inventories from Step 02; page specs from Step 03
- Focus: Design system extraction — tokens, components, patterns
- Limits: No page spec modifications
- Dependencies: Steps 02 and 03 must be complete

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Extract Design Tokens

Take the raw values captured in Step 02 and organize them into a structured token system:

#### Colors
Group colors by purpose: brand, text, background, border, feedback

#### Typography
Document font families, scale (h1 through caption) with size, weight, and line-height

#### Spacing
Document base unit and scale, note common patterns (section padding, card padding, element gap)

#### Other Tokens
Border radius, shadows, breakpoints

### 2. Catalog Reusable Components

For each component in the inventory (from Step 02), document as a component specification with variants, sizes, and states.

### 3. Document Component Variants and States

For each component, capture:
- **Variants** — Visual variations (primary/secondary, small/medium/large)
- **States** — Interactive states (default, hover, focus, active, disabled, loading, error, success)
- **Content slots** — What content goes where (icon, label, description, image)
- **Responsive behavior** — How the component adapts at different breakpoints

### 4. Map Token Usage to Components

Show which tokens each component uses, connecting the token system to the component system.

| Component | Colors Used | Typography | Spacing | Border Radius |
|-----------|------------|------------|---------|---------------|

### 5. Output in WDS Design System Format

Compile everything into a single design system document with: Design Tokens, Components, Patterns, and Notes.

Save to the output folder alongside the page specs from Step 03.

### 6. Verify Checklist

- [ ] Color tokens organized by purpose (brand, text, background, border, feedback)
- [ ] Typography scale documented (families, sizes, weights, line heights)
- [ ] Spacing system extracted (base unit + scale)
- [ ] Additional tokens captured (radii, shadows, breakpoints)
- [ ] Each component cataloged with variants and states
- [ ] Token-to-component mapping created
- [ ] Design system document saved in WDS format
- [ ] Output is consistent with page specs from Step 03

### 7. Present MENU OPTIONS

Display: "**Select an Option:** [M] Return to Activity Menu"

#### Menu Handling Logic:
- IF M: Update design log, then load, read entire file, then execute {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed when user selects 'M'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the design system document is complete and saved will you then load and read fully `{activityWorkflowFile}` to execute.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Color tokens organized by purpose
- Typography scale documented
- Spacing system extracted
- Each component cataloged with variants and states
- Token-to-component mapping created
- Design system document saved in WDS format
- Output is consistent with page specs

### ❌ SYSTEM FAILURE:
- Missing token categories (colors, typography, spacing)
- Not cataloging components with variants and states
- No token-to-component mapping
- Design system document not saved
- Inconsistency with page specs

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

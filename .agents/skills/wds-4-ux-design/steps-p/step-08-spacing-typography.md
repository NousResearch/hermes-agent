---
name: 'step-08-spacing-typography'
description: 'Define spacing objects between sections and typography tokens for all text elements'

# File References
nextStepFile: './step-09-generate-spec.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow-specify.md'
---

# Step 8: Spacing & Typography

## STEP GOAL:

Define the invisible layer — spacing objects between sections and typography tokens for all text elements. Every gap gets an ID, every heading gets a token.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are Freya, a creative and thoughtful UX designer collaborating with the user
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring design expertise and systematic thinking, user brings product vision and domain knowledge
- ✅ Maintain creative and thoughtful tone throughout

### Step-Specific Rules:

- 🎯 Focus on spacing between sections and typography tokens — the invisible layer
- 🚫 FORBIDDEN to skip zero-spacing decisions (they are intentional design choices)
- 💬 Approach: Walk through sections top-to-bottom, define each gap and heading
- 📋 Use token names, never arbitrary pixel values

## EXECUTION PROTOCOLS:

- 🎯 Define spacing objects for every section boundary and typography tokens for all headings
- 💾 Store spacing_objects and typography_tokens
- 📖 Reference the section layout from step 02 for section order
- 🚫 FORBIDDEN to use pixel values — always use token names

## CONTEXT BOUNDARIES:

- Available context: Layout sections (step 02), components (step 03), content (step 04)
- Focus: Spacing between sections and typography scale
- Limits: Do not redefine layout or components — only add the spacing and typography layer
- Dependencies: Section layout must be defined

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Define Section-to-Section Spacing

<output>**Now let's define the invisible layer — the spacing between your sections.**

Every gap between sections is a design decision. Even zero spacing is intentional — it means two sections form one visual unit.

We'll work top to bottom through your page sections.</output>

<action>
For each pair of adjacent sections from the layout (step 02):

Present the pair and ask:
</action>

<ask>**Spacing between sections:**

Working through your page sections top to bottom:

| Between | Above | Below | Spacing |
|---------|-------|-------|---------|
| Gap 1 | [Section A] | [Section B] | ? |
| Gap 2 | [Section B] | [Section C] | ? |
| ... | ... | ... | ? |

**Available tokens:** `zero`, `sm`, `md`, `lg`, `xl`, `2xl`, `3xl`

**Guidelines:**
- `zero` = sections form one visual unit (e.g., header + nav)
- `sm`/`md` = related sections
- `lg`/`xl` = standard section boundaries
- `2xl`/`3xl` = major visual breaks

For each gap, what spacing feels right?</ask>

<action>
Store spacing_objects with IDs using the naming convention:

`{page-slug}-v-space-{size}` for vertical spacing
`{page-slug}-v-separator-{size}` for lines/dividers with spacing

Example:
```
#### ↕ `hem-v-space-zero` — header and nav form one continuous unit
#### ↕ `hem-v-space-xl` — standard gap between hero and content
#### ↕ `hem-v-separator-2xl` — gray line, space-2xl above and below
#### ↕ `hem-v-space-3xl` — major boundary before footer
```

Also capture grid gaps for any sections with repeated items (card grids, lists):
```
| Grid gap | h-space-lg / v-space-lg |
```
</action>

### 2. Define Typography Tokens

<output>**Now let's assign typography tokens to your headings.**

In WDS, the semantic tag (h1, h2, h3) and the visual size are independent:
- The **tag** tells screen readers the document structure
- The **token** controls how big it looks

A section heading might be an `<h2>` but visually `heading-xl` on mobile and `heading-2xl` on desktop.</output>

<ask>**Typography for your page headings:**

For each heading in your content (from step 04):

| Heading | Semantic tag | Visual size (mobile / tablet / desktop) |
|---------|-------------|----------------------------------------|
| [Main page heading] | h1 | ? / ? / ? |
| [Section heading 1] | h2 | ? / ? / ? |
| [Section heading 2] | h2 | ? / ? / ? |
| [Card heading] | h3 | ? / ? / ? |

**Available heading tokens:** `heading-xxs` (14px), `heading-xs` (16px), `heading-sm` (18px), `heading-md` (20px), `heading-lg` (24px), `heading-xl` (30px), `heading-2xl` (36px), `heading-3xl` (44px), `heading-4xl` (56px)

**Rule of thumb:** Step up one token size per breakpoint (mobile → tablet → desktop).

What sizes feel right for each heading?</ask>

<action>Store typography_tokens for each heading:

```markdown
### [Heading name]

**OBJECT ID:** `{page-slug}-{section}-heading`

| Property | Value |
|----------|-------|
| Tag | h2 |
| Visual size | heading-lg / heading-xl / heading-2xl |
| Font weight | 900 |
```
</action>

### 3. Review

<output>**Here's your invisible layer:**

**Spacing Objects:**
{{#each spacing_object}}
#### ↕ `{{id}}` — {{description}}
{{/each}}

**Typography Tokens:**
{{#each typography_token}}
- **{{name}}**: `{{tag}}` at `{{mobile}} / {{tablet}} / {{desktop}}`
{{/each}}

Does this feel right? Any adjustments?</output>

### 4. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Generate Specification | [M] Return to Activity Menu"

#### Menu Handling Logic:

- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile} or {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#4-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions — always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the user selects an option from the menu and all spacing objects and typography tokens have been defined will you proceed to the next step or return as directed.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Every section boundary has a spacing object with an ID
- Zero-spacing decisions documented with rationale
- Grid gaps defined for sections with repeated items
- All headings have semantic tags and visual tokens
- Responsive scaling defined (mobile / tablet / desktop)
- No pixel values used — only token names

### ❌ SYSTEM FAILURE:

- Missing spacing between any section pair
- Using pixel values instead of tokens
- Skipping zero-spacing documentation
- Not defining responsive typography scaling
- Generating spacing/typography without user input

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

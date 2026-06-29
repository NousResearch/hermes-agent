---
name: 'step-07-execute-decision'
description: 'Execute the designer decision: reuse, add variant, or create new component'

# File References
nextStepFile: './step-08a-initialize-design-system.md'
---

# Step 7: Execute Decision

## STEP GOAL:

Execute the designer decision by routing to the appropriate operation: reuse existing component, add variant to existing, or create new component.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are the Design System Architect guiding design system creation and maintenance
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring design system expertise and component analysis, user brings design knowledge and project context
- ✅ Maintain systematic and analytical tone throughout

### Step-Specific Rules:

- 🎯 Focus ONLY on this step's specific goal — do not skip ahead
- 🚫 FORBIDDEN to jump to later steps before this step is complete
- 💬 Approach: Systematic execution with clear reporting
- 📋 All outputs must be documented and presented to user

## EXECUTION PROTOCOLS:

- 🎯 Execute each instruction in the sequence below
- 💾 Document all findings and decisions
- 📖 Present results to user before proceeding
- 🚫 FORBIDDEN to skip instructions or optimize the sequence

## CONTEXT BOUNDARIES:

- Available context: Previous step outputs and project configuration
- Focus: This step's specific goal only
- Limits: Do not perform actions belonging to subsequent steps
- Dependencies: Requires all previous steps to be completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

## Execution Paths

### Path A: Reuse Existing Component

Designer chose to use existing component as-is

### Path B: Add Variant

Designer chose to add variant to existing component

### Path C: Create New Component

Designer chose to create new component

---

## Path A: Reuse Existing Component

### Step 1: Confirm Action

<output>
```
✅ Reusing Button [btn-001]

I'll update your page spec to reference the existing component.

````
</output>

### Step 2: Extract Page-Specific Content

<action>
From complete specification, extract:
- Labels/text content
- Page-specific why/purpose
- Error messages
- Contextual information
</action>

**Example:**
```yaml
Page-Specific Content:
- label: "Continue"
- why: "Navigate to next step in onboarding"
- context: "Multi-step form navigation"
````

### Step 3: Create Reference

<action>
Create reference to existing component:
- Component ID: btn-001
- Variant: primary (or whichever applies)
- Page-specific content
</action>

**Output:**

```yaml
# C-UX-Scenarios/onboarding-page.md

Continue Button:
  component: Button.primary [btn-001]
  why: Navigate to next step in onboarding
  label: 'Continue'
```

### Step 4: Update Component Usage

<action>
Update design system component to track usage:
- Add page to "Used In" list
- Increment usage count
</action>

**Update:**

```yaml
# D-Design-System/components/button.md

Used In:
  - Login page (login button)
  - Signup page (create account button)
  - Dashboard (action buttons)
  - Onboarding page (continue button)  ← Added
```

### Step 5: Complete

<output>
```
✅ Done! Button [btn-001] is now used on onboarding page.

Page spec updated with reference.
Component usage tracked.

```
</output>

**Return to Phase 4**

---

## Path B: Add Variant

### Step 1: Confirm Action

<output>
```

✅ Adding "navigation" variant to Button [btn-001]

I'll update the component definition and create the reference.

````
</output>

### Step 2: Extract Component-Level Info

<action>
From complete specification, extract:
- Variant-specific styling
- Variant-specific states
- Variant-specific behaviors
</action>

**Example:**
```yaml
Navigation Variant:
- icon: arrow-right
- loading_state: true
- hover_animation: icon_shift
````

### Step 3: Update Component Definition

<action>
Add variant to existing component:
- Add to variants list
- Document variant-specific attributes
- Maintain shared attributes
</action>

**Update:**

```yaml
# D-Design-System/components/button.md

Button Component [btn-001]:
  variants:
    - primary (submit actions)
    - secondary (cancel actions)
    - navigation (continue/next actions)  ← Added

  shared_states:
    - default, hover, active, disabled

  variant_specific:
    navigation:
      icon: arrow-right
      loading_state: true
      hover_animation: icon_shift
```

### Step 4: Create Reference

<action>
Create reference with variant specified:
</action>

**Output:**

```yaml
# C-UX-Scenarios/onboarding-page.md

Continue Button:
  component: Button.navigation [btn-001]  ← Variant specified
  why: Navigate to next step in onboarding
  label: 'Continue'
```

### Step 5: Update Usage Tracking

<action>
Track variant usage:
</action>

**Update:**

```yaml
# D-Design-System/components/button.md

Variant Usage:
  primary: 5 pages
  secondary: 3 pages
  navigation: 1 page  ← Added
```

### Step 6: Complete

<output>
```
✅ Done! Navigation variant added to Button [btn-001].

Component definition updated.
Page spec created with variant reference.
Variant usage tracked.

```
</output>

**Return to Phase 4**

---

## Path C: Create New Component

### Step 1: Confirm Action

<output>
```

✅ Creating new Navigation Button [btn-002]

I'll create the component definition and set up the reference.

```
</output>

### Step 2: Generate Component ID

<action>
Generate unique component ID:
- Check existing IDs
- Increment counter for type
- Format: [type-prefix]-[number]
</action>

**Example:**
```

Existing Button IDs: btn-001
New ID: btn-002

````

### Step 3: Extract Component-Level Info

<action>
From complete specification, extract:
- Visual attributes (size, shape, color)
- States (default, hover, active, disabled, loading)
- Behaviors (interactions, animations)
- Styling (design tokens or Figma reference)
</action>

**Example:**
```yaml
Component-Level Info:
  type: Button
  purpose: Navigation actions
  states: [default, hover, active, disabled, loading]
  icon: arrow-right
  size: medium
  color: blue
  shape: rounded
  hover_animation: icon_shift
````

### Step 4: Create Component File

<action>
Create new component file using template:
</action>

**Route to:** `step-08b-create-new-component.md`

**Output:**

```yaml
# D-Design-System/components/navigation-button.md

# Navigation Button [btn-002]

**Type:** Interactive
**Purpose:** Navigation actions (continue, next, proceed)
**Library:** shadcn/ui Button (if Mode C)
**Figma:** [Link] (if Mode B)

## States
- default
- hover
- active
- disabled
- loading (with spinner)

## Styling
- Size: medium
- Color: blue primary
- Shape: rounded
- Icon: arrow-right
- Hover: icon shifts right

## Used In
- Onboarding page (continue button)
```

### Step 5: Create Reference

<action>
Create reference in page spec:
</action>

**Output:**

```yaml
# C-UX-Scenarios/onboarding-page.md

Continue Button:
  component: NavigationButton [btn-002]
  why: Navigate to next step in onboarding
  label: 'Continue'
```

### Step 6: Update Design System Index

<action>
Add to design system component list:
</action>

**Update:**

```yaml
# D-Design-System/components/README.md

Components:
  - Button [btn-001] - Primary action buttons
  - Input Field [inp-001] - Text input fields
  - Card [crd-001] - Content cards
  - Navigation Button [btn-002] - Navigation actions  ← Added
```

### Step 7: Complete

<output>
```
✅ Done! Navigation Button [btn-002] created.

Component file created: D-Design-System/components/navigation-button.md
Page spec created with reference.
Design system index updated.

````
</output>

**Return to Phase 4**

---

## Post-Execution Actions

### Update Project State

<action>
Update project tracking:
- Increment component count
- Update design system status
- Log decision for future reference
</action>

**Example:**
```yaml
# A-Project-Brief/design-system-log.md

2024-12-09: Created Navigation Button [btn-002]
- Reason: Semantic distinction from submit buttons
- Decision: Create new vs variant
- Designer: Chose clarity over consistency
````

### Notify Designer

<output>
```
📊 Design System Update:

Components: 4 (was 3)
Latest: Navigation Button [btn-002]

Your design system is growing! Consider reviewing component
organization when you reach 10+ components.

```
</output>

---

## Error Handling

**If component creation fails:**
```

❌ Error creating component file.

Error: [error message]

Would you like to:

1. Retry
2. Create manually
3. Skip design system for this component

Your choice:

```

**If reference creation fails:**
```

❌ Error updating page spec.

Error: [error message]

Component was created successfully, but page reference failed.
I'll keep the complete spec on the page for now.

```

**If ID conflict:**
```

⚠️ Component ID conflict detected.

btn-002 already exists but with different content.

Generating alternative ID: btn-003

```

---

## Validation

### Before Completing

<action>
Validate execution:
- ✓ Component file created (if new)
- ✓ Component updated (if variant)
- ✓ Page spec has reference
- ✓ Usage tracked
- ✓ Design system index updated
</action>

**If validation fails:**
```

⚠️ Validation Warning:

Some steps may not have completed successfully.
Please review:

- [List of potential issues]

Continue anyway? (y/n)

```

---

## Return to Phase 4

<action>
Return control to Phase 4 orchestration:
- Pass component reference
- Pass page-specific content
- Signal completion
</action>

**Phase 4 continues with:**
- Update page spec with reference
- Continue to next component
- Or complete page specification

---

## Summary Output

<output>
```

✅ Design System Operation Complete

Action: Created new component
Component: Navigation Button [btn-002]
Page: Onboarding page
Reference: NavigationButton [btn-002]

Files Updated:

- D-Design-System/components/navigation-button.md (created)
- C-UX-Scenarios/onboarding-page.md (reference added)
- D-Design-System/components/README.md (index updated)

Next: Continue with next component in Phase 4

```
</output>

---

**This completes the assessment and execution flow. Control returns to Phase 4.**
```

### 5. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to next operation"

#### Menu Handling Logic:

- IF C: Update design log, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#5-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects the appropriate option
- User can chat or ask questions — always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN [decision has been executed and design system updated accordingly], will you then load and read fully `{nextStepFile}` to execute the next step.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Step goal achieved completely
- All instructions executed in sequence
- Results documented and presented to user
- User confirmed before proceeding
- Design log updated

### ❌ SYSTEM FAILURE:

- Skipping any instruction in the sequence
- Generating content without user input
- Jumping ahead to later steps
- Not presenting results to user
- Proceeding without user confirmation

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

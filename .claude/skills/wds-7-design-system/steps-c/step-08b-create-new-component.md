---
name: 'step-08b-create-new-component'
description: 'Add a new component to the design system with full specification'

# File References
nextStepFile: './step-08c-update-component.md'
---

# Step 8b: Create New Component

## STEP GOAL:

Add a new component to the design system: generate ID, determine category, extract attributes, create component file from template, update index and stats.

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

## Step 1: Generate Component ID

<action>
Generate unique component ID:
1. Determine component type prefix
2. Check existing IDs for that type
3. Increment counter
4. Format: [prefix]-[number]
</action>

**Type Prefixes:**

```
Button → btn
Input Field → inp
Card → crd
Modal → mdl
Dropdown → drp
Checkbox → chk
Radio → rad
Toggle → tgl
Tab → tab
Accordion → acc
Alert → alt
Badge → bdg
Avatar → avt
Icon → icn
Image → img
Link → lnk
Text → txt
Heading → hdg
List → lst
Table → tbl
Form → frm
Container → cnt
Grid → grd
Flex → flx
Divider → div
Spacer → spc
```

**Example:**

```
Component Type: Button
Existing Button IDs: btn-001, btn-002
New ID: btn-003
```

<output>
```
🆔 Generated Component ID: btn-003
```
</output>

---

## Step 2: Determine Component Category

<action>
Categorize component for organization:
- Interactive (buttons, links, controls)
- Form (inputs, selects, checkboxes)
- Layout (containers, grids, dividers)
- Content (text, images, media)
- Feedback (alerts, toasts, modals)
- Navigation (tabs, breadcrumbs, menus)
</action>

**Example:**

```
Component: Button
Category: Interactive
```

---

## Step 3: Extract Component-Level Information

<action>
From complete specification, extract component-level info:

**Visual Attributes:**

- Size (small, medium, large)
- Shape (rounded, square, pill)
- Color scheme
- Typography
- Spacing/padding
- Border style

**Behavioral Attributes:**

- States (default, hover, active, disabled, loading, error)
- Interactions (click, hover, focus, blur)
- Animations/transitions
- Keyboard support
- Accessibility attributes

**Functional Attributes:**

- Purpose/role
- Input/output type
- Validation rules
- Required/optional

**Design System Attributes:**

- Variants (if any)
- Design tokens used
- Figma reference (if Mode B)
- Library component (if Mode C)
  </action>

---

## Step 4: Create Component File

<action>
Use component template to create file:
</action>

**File:** `D-Design-System/components/[component-name].md`

**Template Structure:**

````markdown
# [Component Name] [component-id]

**Type:** [Interactive/Form/Layout/Content/Feedback/Navigation]
**Category:** [Specific category]
**Purpose:** [Brief description]

---

## Overview

[Component description and when to use it]

---

## Variants

[If component has variants, list them]

**Example:**

- primary - Main call-to-action
- secondary - Secondary actions
- ghost - Subtle actions

[If no variants:]
This component has no variants.

---

## States

**Required States:**

- default
- hover
- active
- disabled

**Optional States:**

- loading
- error
- success
- focus

**State Descriptions:**
[Describe what each state looks like/does]

---

## Styling

### Visual Properties

**Size:** [small/medium/large or specific values]
**Shape:** [rounded/square/pill or specific border-radius]
**Colors:** [Color tokens or values]
**Typography:** [Font tokens or values]
**Spacing:** [Padding/margin values]

### Design Tokens

[If using design tokens:]

```yaml
colors:
  background: primary-500
  text: white
  border: primary-600

typography:
  font-size: text-base
  font-weight: semibold

spacing:
  padding-x: 4
  padding-y: 2

effects:
  border-radius: md
  shadow: sm
```
````

### Figma Reference

[If Mode B - Custom Design System:]
**Figma Component:** [Link to Figma component]
**Node ID:** [Figma node ID]
**Last Synced:** [Date]

### Library Component

[If Mode C - Component Library:]
**Library:** [shadcn/Radix/etc.]
**Component:** [Library component name]
**Customizations:** [Any overrides from library default]

---

## Behavior

### Interactions

**Click:**
[What happens on click]

**Hover:**
[What happens on hover]

**Focus:**
[What happens on focus]

**Keyboard:**
[Keyboard shortcuts/navigation]

### Animations

[If component has animations:]

- [Animation description]
- Duration: [ms]
- Easing: [easing function]

---

## Accessibility

**ARIA Attributes:**

- role: [role]
- aria-label: [label]
- aria-disabled: [when disabled]
- [Other ARIA attributes]

**Keyboard Support:**

- Enter/Space: [action]
- Tab: [navigation]
- [Other keyboard support]

**Screen Reader:**
[How screen readers should announce this component]

---

## Usage

### When to Use

[Guidelines for when this component is appropriate]

### When Not to Use

[Guidelines for when to use a different component]

### Best Practices

- [Best practice 1]
- [Best practice 2]
- [Best practice 3]

---

## Used In

**Pages:** [List of pages using this component]

**Usage Count:** [Number]

**Examples:**

- [Page name] - [Specific usage]
- [Page name] - [Specific usage]

---

## Related Components

[If this component is related to others:]

- [Related component 1] - [Relationship]
- [Related component 2] - [Relationship]

---

## Version History

**Created:** [Date]
**Last Updated:** [Date]

**Changes:**

- [Date]: Created component
- [Date]: [Change description]

---

## Notes

[Any additional notes, considerations, or future plans]

````

---

## Step 5: Populate Template

<action>
Fill template with extracted information:
</action>

**Example Output:**

```markdown
# Button [btn-003]

**Type:** Interactive
**Category:** Action
**Purpose:** Trigger primary and secondary actions

---

## Overview

Buttons are used to trigger actions. They should have clear, action-oriented labels that describe what will happen when clicked.

Use buttons for important actions that change state or navigate to new content.

---

## Variants

- **primary** - Main call-to-action (submit, save, continue)
- **secondary** - Secondary actions (cancel, back)
- **ghost** - Subtle actions (close, dismiss)

---

## States

**Required States:**
- default - Normal state
- hover - Mouse over button
- active - Button being clicked
- disabled - Button cannot be clicked

**Optional States:**
- loading - Action in progress (shows spinner)

**State Descriptions:**

**Default:** Blue background, white text, medium size
**Hover:** Darker blue background, slight scale increase
**Active:** Even darker blue, slight scale decrease
**Disabled:** Gray background, gray text, reduced opacity
**Loading:** Disabled state + spinner icon

---

## Styling

### Visual Properties

**Size:** medium (h-10, px-4)
**Shape:** rounded (border-radius: 0.375rem)
**Colors:**
- Background: blue-600
- Text: white
- Border: none

**Typography:**
- Font size: 14px
- Font weight: 600
- Line height: 1.5

**Spacing:**
- Padding X: 16px
- Padding Y: 8px
- Gap (if icon): 8px

### Design Tokens

```yaml
colors:
  primary:
    background: blue-600
    hover: blue-700
    active: blue-800
  text: white

typography:
  size: text-sm
  weight: semibold

spacing:
  padding-x: 4
  padding-y: 2
  gap: 2

effects:
  border-radius: md
  shadow: sm
  transition: all 150ms ease
````

### Library Component

**Library:** shadcn/ui
**Component:** Button
**Customizations:** None (using library defaults)

---

## Behavior

### Interactions

**Click:**
Triggers associated action (form submit, navigation, etc.)

**Hover:**

- Background darkens
- Slight scale increase (1.02)
- Cursor changes to pointer

**Focus:**

- Blue outline ring
- Maintains hover state

**Keyboard:**

- Enter/Space triggers click
- Tab navigates to/from button

### Animations

**Hover Scale:**

- Duration: 150ms
- Easing: ease-in-out
- Scale: 1.02

**Click Feedback:**

- Duration: 100ms
- Scale: 0.98

---

## Accessibility

**ARIA Attributes:**

- role: button
- aria-label: [Descriptive label if icon-only]
- aria-disabled: true [when disabled]
- aria-busy: true [when loading]

**Keyboard Support:**

- Enter/Space: Triggers button action
- Tab: Moves focus to/from button

**Screen Reader:**
Announces button label and state (disabled, busy, etc.)

---

## Usage

### When to Use

- Primary actions (submit forms, save data, proceed to next step)
- Secondary actions (cancel, go back, dismiss)
- Triggering modals or dialogs
- Navigation to new pages/sections

### When Not to Use

- For navigation that looks like text (use Link component)
- For toggling states (use Toggle or Checkbox)
- For selecting from options (use Radio or Checkbox)

### Best Practices

- Use action-oriented labels ("Save Changes" not "Save")
- Limit primary buttons to one per section
- Place primary button on the right in button groups
- Ensure sufficient touch target size (min 44x44px)
- Provide loading state for async actions

---

## Used In

**Pages:** 1

**Usage Count:** 1

**Examples:**

- Login page - Submit credentials button

---

## Related Components

- Link [lnk-001] - For text-style navigation
- Icon Button [btn-002] - For icon-only actions

---

## Version History

**Created:** 2024-12-09
**Last Updated:** 2024-12-09

**Changes:**

- 2024-12-09: Created component

---

## Notes

This is the primary button component. Consider adding more variants as needs emerge (danger, success, etc.).

````

---

## Step 6: Update Component Index

<action>
Add component to index:
</action>

**Update:** `D-Design-System/components/README.md`

```markdown
## Component List

### Interactive Components
- Button [btn-001] - Primary action buttons
- Icon Button [btn-002] - Icon-only actions
- Button [btn-003] - Standard action button  ← Added

**Total Interactive:** 3
````

---

## Step 7: Update Design System Stats

<action>
Update design system statistics:
</action>

**Update:** `D-Design-System/README.md` (if exists)

```yaml
**Total Components:** 4 (was 3)
**Last Updated:** [Date]
**Latest Addition:** Button [btn-003]
```

---

## Step 8: Create Component Reference

<action>
Generate reference for page spec:
</action>

**Output:**

```yaml
component_reference:
  id: btn-003
  name: Button
  variant: primary
  file: D-Design-System/components/button.md
```

---

## Step 9: Complete

<output>
```
✅ Component Created: Button [btn-003]

File: D-Design-System/components/button.md
Category: Interactive
Variants: primary, secondary, ghost
States: default, hover, active, disabled, loading

Component index updated.
Design system stats updated.
Reference ready for page spec.

Next: Return to Phase 4 to complete page specification

```
</output>

---

## Validation

<action>
Validate component creation:
- ✓ Component file created
- ✓ Component ID unique
- ✓ Template fully populated
- ✓ Index updated
- ✓ Stats updated
- ✓ Reference generated
</action>

---

## Error Handling

**If ID conflict:**
```

⚠️ Component ID btn-003 already exists.

Generating alternative ID: btn-004

```

**If file creation fails:**
```

❌ Error creating component file.

Error: [error message]

Would you like to:

1. Retry
2. Create with different ID
3. Skip design system for this component

Your choice:

```

**If template population incomplete:**
```

⚠️ Some component information is missing.

Missing:

- [List of missing fields]

I'll create the component with placeholders.
You can fill in details later.

```

---

**This operation creates a new component. Return to Phase 4 with component reference.**
```

### 10. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue or [M] Return to Activity Menu"

#### Menu Handling Logic:

- IF C: Update design log, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#10-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects the appropriate option
- User can chat or ask questions — always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN [component is created with full specification, index updated, and reference generated], will you then load and read fully `{nextStepFile}` to execute the next step.

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

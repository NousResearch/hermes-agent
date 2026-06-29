---
name: 'step-08d-add-variant'
description: 'Add a new variant to an existing component in the design system'

# File References
nextStepFile: './step-08e-generate-catalog.md'
---

# Step 8d: Add Variant

## STEP GOAL:

Add a new variant to an existing component: extract variant-specific info, determine name, update component file, track usage, validate addition.

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

## Step 1: Load Existing Component

<action>
Read existing component file:
- Component ID
- Current variants
- Shared attributes
- Variant-specific attributes
</action>

**Example:**

```yaml
Component: Button [btn-001]
Current Variants:
  - primary (submit actions)
  - secondary (cancel actions)
```

<output>
```
📖 Loaded Button [btn-001]

Current variants: 2 (primary, secondary)
Adding new variant: navigation

````
</output>

---

## Step 2: Extract Variant-Specific Information

<action>
From new component specification, extract:
- What's different from existing variants?
- What's shared with existing variants?
- Variant-specific styling
- Variant-specific behaviors
- Variant-specific states (if any)
</action>

**Example:**
```yaml
Shared with existing:
  - Size: medium
  - Shape: rounded
  - Base states: default, hover, active, disabled

Different from existing:
  - Has icon (arrow-right)
  - Has loading state
  - Icon animation on hover
  - Purpose: navigation vs submission
````

---

## Step 3: Determine Variant Name

<action>
Generate descriptive variant name:
- Based on purpose or visual distinction
- Consistent with existing variant naming
- Clear and semantic
</action>

**Examples:**

```
Purpose-based:
- navigation (for navigation actions)
- destructive (for delete/remove actions)
- success (for positive confirmations)

Visual-based:
- outlined (border, no fill)
- ghost (transparent background)
- large (bigger size)

Context-based:
- header (used in headers)
- footer (used in footers)
- inline (used inline with text)
```

<ask>
```
Suggested variant name: "navigation"

This variant is for navigation actions (continue, next, proceed).

Is this name clear and appropriate? (y/n)
Or suggest alternative name:

````
</ask>

---

## Step 4: Update Component File

<action>
Add variant to component definition:
</action>

### Update Variants Section

**Before:**
```markdown
## Variants

- **primary** - Main call-to-action (submit, save, continue)
- **secondary** - Secondary actions (cancel, back)
````

**After:**

```markdown
## Variants

- **primary** - Main call-to-action (submit, save, continue)
- **secondary** - Secondary actions (cancel, back)
- **navigation** - Navigation actions (next, proceed, continue) ← Added
```

### Add Variant-Specific Styling

**Add section:**

```markdown
### Variant-Specific Styling

**Primary:**

- Background: blue-600
- Icon: none
- Loading: spinner only

**Secondary:**

- Background: gray-200
- Text: gray-900
- Icon: none

**Navigation:** ← Added

- Background: blue-600
- Icon: arrow-right
- Loading: spinner + icon
- Hover: icon shifts right
```

### Update States (if variant has unique states)

**If navigation variant has loading state but others don't:**

```markdown
## States

**Shared States (all variants):**

- default
- hover
- active
- disabled

**Variant-Specific States:**

**Navigation:**

- loading (shows spinner, disables interaction)
```

---

## Step 5: Update Usage Tracking

<action>
Track new variant usage:
</action>

**Add to component file:**

```markdown
## Variant Usage

**Primary:** 5 pages
**Secondary:** 3 pages
**Navigation:** 1 page ← Added

**Navigation variant used in:**

- Onboarding page (continue button)
```

---

## Step 6: Update Component Complexity Note

<action>
Add note about variant count:
</action>

**If this is 3rd+ variant:**

```markdown
## Notes

This component now has 3 variants. Consider:

- Are all variants necessary?
- Should any variants be separate components?
- Is the component becoming too complex?

Review component organization when reaching 5+ variants.
```

---

## Step 7: Validate Variant Addition

<action>
Check for potential issues:
</action>

**Variant Explosion Check:**

```
⚠️ Variant Count: 3

This is manageable. Monitor for variant explosion as more are added.

Recommended maximum: 5 variants per component
```

**Consistency Check:**

```
✓ New variant consistent with existing variants
✓ Naming convention followed
✓ Shared attributes maintained
```

**Complexity Check:**

```
⚠️ Navigation variant adds loading state not present in other variants.

This increases component complexity. Consider:
- Should loading state be shared across all variants?
- Or is it truly navigation-specific?

Current approach: Variant-specific (acceptable)
```

---

## Step 8: Update Component Version

<action>
Track component changes:
</action>

**Update version history:**

```markdown
## Version History

**Created:** 2024-12-01
**Last Updated:** 2024-12-09

**Changes:**

- 2024-12-01: Created component with primary and secondary variants
- 2024-12-09: Added navigation variant ← Added
```

---

## Step 9: Create Component Reference

<action>
Generate reference for page spec:
</action>

**Output:**

```yaml
component_reference:
  id: btn-001
  name: Button
  variant: navigation  ← New variant
  file: D-Design-System/components/button.md
```

---

## Step 10: Complete

<output>
```
✅ Variant Added: Button.navigation [btn-001]

Component: Button [btn-001]
New Variant: navigation
Total Variants: 3 (primary, secondary, navigation)

Component file updated:

- Variant added to list
- Variant-specific styling documented
- Usage tracking added
- Version history updated

Reference ready for page spec.

Next: Return to Phase 4 to complete page specification

```
</output>

---

## Designer Guidance

<output>
```

💡 Variant Management Tips:

**Current Status:**

- Component: Button [btn-001]
- Variants: 3
- Status: Healthy

**Watch for:**

- 5+ variants → Consider splitting component
- Variants with very different purposes → Might need separate components
- Variants rarely used together → Might indicate separate components

**Best Practices:**

- Keep variants related (same base purpose)
- Use clear, semantic variant names
- Document when to use each variant
- Review variant list periodically

You can always refactor later if needed!

```
</output>

---

## Validation

<action>
Validate variant addition:
- ✓ Variant added to component file
- ✓ Variant-specific attributes documented
- ✓ Usage tracking updated
- ✓ Version history updated
- ✓ Reference generated
- ✓ Complexity checked
</action>

---

## Error Handling

**If variant name conflicts:**
```

⚠️ Variant "navigation" already exists in Button [btn-001].

This might mean:

1. You're trying to add a duplicate
2. The existing variant should be updated
3. A different variant name is needed

Current navigation variant:
[Show existing variant details]

Options:

1. Update existing variant
2. Choose different name
3. Cancel

Your choice:

```

**If component file not found:**
```

❌ Error: Component file not found.

Component ID: btn-001
Expected file: D-Design-System/components/button.md

This shouldn't happen. Possible causes:

- File was deleted
- Component ID is incorrect
- Design system structure corrupted

Would you like to:

1. Create component as new
2. Specify correct component ID
3. Cancel

Your choice:

```

**If variant too different:**
```

⚠️ Warning: High Divergence Detected

The new variant is very different from existing variants:

- Different core purpose
- Different visual structure
- Different behavioral patterns

Similarity to existing variants: 35%

This might be better as a separate component.

Options:

1. Add as variant anyway
2. Create as new component instead
3. Review differences in detail

Your choice:

```

---

## Post-Addition Review

<action>
After adding variant, check component health:
</action>

**Component Health Check:**
```

📊 Component Health: Button [btn-001]

Variants: 3
Complexity: Medium
Consistency: High
Usage: 9 pages

Health Status: ✅ Healthy

Recommendations:

- Document variant selection guidelines
- Consider adding variant usage examples
- Monitor for variant explosion

```

---

**This operation adds a variant. Return to Phase 4 with component reference.**
```

### 11. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Generate Catalog or [M] Return to Activity Menu"

#### Menu Handling Logic:

- IF C: Update design log, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#11-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects the appropriate option
- User can chat or ask questions — always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN [variant is added, component file updated, and usage tracked], will you then load and read fully `{nextStepFile}` to execute the next step.

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

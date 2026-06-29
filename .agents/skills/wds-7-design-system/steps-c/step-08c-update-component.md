---
name: 'step-08c-update-component'
description: 'Update an existing component definition with new states, styling, or behavior'

# File References
nextStepFile: './step-08d-add-variant.md'
---

# Step 8c: Update Component

## STEP GOAL:

Update an existing component definition: identify update type, analyze impact, apply changes, track version history, notify affected pages.

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

## Step 1: Identify Update Type

<action>
Determine what's being updated:
</action>

**Update Types:**

### Type A: Add New State

Adding state to all variants (e.g., loading, error, success)

### Type B: Update Styling

Changing visual properties (colors, sizing, spacing)

### Type C: Update Behavior

Changing interactions, animations, or keyboard support

### Type D: Update Accessibility

Adding/modifying ARIA attributes or screen reader support

### Type E: Update Documentation

Clarifying usage, adding examples, fixing errors

### Type F: Refactor

Reorganizing component structure, splitting/merging variants

<ask>
```
What type of update is this?

[A] Add new state
[B] Update styling
[C] Update behavior
[D] Update accessibility
[E] Update documentation
[F] Refactor component

Your choice:

```
</ask>

---

## Step 2: Load Current Component

<action>
Read existing component file:
- Current definition
- All variants
- Current states
- Current styling
- Usage tracking
</action>

<output>
```

📖 Loaded Button [btn-001]

Current state:

- Variants: 3 (primary, secondary, navigation)
- States: default, hover, active, disabled
- Used in: 9 pages
- Last updated: 2024-12-09

```
</output>

---

## Step 3: Analyze Impact

<action>
Determine impact of update:
</action>

**Impact Assessment:**

### Scope
- All variants affected?
- Specific variant only?
- All instances affected?
- Specific usage only?

### Breaking Changes
- Does this change existing behavior?
- Will existing pages need updates?
- Does this affect developers?

### Compatibility
- Compatible with current usage?
- Requires page spec updates?
- Requires code changes?

<output>
```

📊 Impact Analysis:

Update: Adding "loading" state to all button variants

Scope: All variants (primary, secondary, navigation)
Affected Pages: 9 pages using Button component
Breaking Change: No (additive only)
Compatibility: Fully compatible (optional state)

Impact Level: Low (safe to proceed)

```
</output>

---

## Step 4: Confirm Update

<ask>
```

Ready to update Button [btn-001]

Update: Add "loading" state
Impact: 9 pages (no breaking changes)

This will:
✓ Add loading state to component definition
✓ Update all variant documentation
✓ Maintain backward compatibility

Proceed with update? (y/n)

````
</ask>

---

## Step 5: Apply Update

<action>
Update component file based on type:
</action>

### Type A: Add New State

**Update States Section:**

**Before:**
```markdown
## States

**Shared States:**
- default
- hover
- active
- disabled
````

**After:**

```markdown
## States

**Shared States:**

- default
- hover
- active
- disabled
- loading ← Added

**State Descriptions:**

**Loading:**

- Disabled interaction
- Shows spinner icon
- Maintains button size
- Reduced opacity (0.7)
```

**Update Variant-Specific Sections (if needed):**

```markdown
### Variant-Specific Styling

**Navigation (loading state):**

- Spinner + arrow icon
- Arrow fades out during loading
```

### Type B: Update Styling

**Update Styling Section:**

**Before:**

```markdown
### Visual Properties

**Border Radius:** 0.375rem (md)
```

**After:**

```markdown
### Visual Properties

**Border Radius:** 0.5rem (lg) ← Updated

**Change Reason:** Increased for better visual consistency with other components
```

### Type C: Update Behavior

**Update Behavior Section:**

**Before:**

```markdown
### Keyboard

- Enter/Space: Triggers button action
- Tab: Moves focus to/from button
```

**After:**

```markdown
### Keyboard

- Enter/Space: Triggers button action
- Tab: Moves focus to/from button
- Escape: Cancels action (if in progress) ← Added
```

### Type D: Update Accessibility

**Update Accessibility Section:**

**Before:**

```markdown
**ARIA Attributes:**

- role: button
- aria-disabled: true [when disabled]
```

**After:**

```markdown
**ARIA Attributes:**

- role: button
- aria-disabled: true [when disabled]
- aria-busy: true [when loading] ← Added
- aria-live: polite [for status updates] ← Added
```

### Type E: Update Documentation

**Update Usage Section:**

**Before:**

```markdown
### When to Use

- Primary actions
- Secondary actions
```

**After:**

```markdown
### When to Use

- Primary actions (submit forms, save data, proceed to next step)
- Secondary actions (cancel, go back, dismiss)
- Triggering modals or dialogs ← Added
- Navigation to new pages/sections ← Added

### When Not to Use

- For navigation that looks like text (use Link component) ← Added
- For toggling states (use Toggle or Checkbox) ← Added
```

### Type F: Refactor

**Example: Split variant into separate component**

```markdown
## Refactoring Note

**Date:** 2024-12-09
**Change:** Moved "icon-only" variant to separate Icon Button component

**Reason:** Icon-only buttons have significantly different:

- Visual structure (no text)
- Accessibility requirements (requires aria-label)
- Usage patterns (toolbars, compact spaces)

**Migration:**

- Old: Button.icon-only [btn-001]
- New: Icon Button [btn-002]

**Affected Pages:** 5 pages
**Migration Status:** Complete
```

---

## Step 6: Update Version History

<action>
Track update in version history:
</action>

**Update:**

```markdown
## Version History

**Created:** 2024-12-01
**Last Updated:** 2024-12-09

**Changes:**

- 2024-12-01: Created component
- 2024-12-05: Added navigation variant
- 2024-12-09: Added loading state to all variants ← Added
```

---

## Step 7: Notify Affected Pages

<action>
If update affects existing usage, create notification:
</action>

<output>
```
📢 Component Update Notification

Component: Button [btn-001]
Update: Added loading state
Affected Pages: 9

Pages using this component:

- Login page
- Signup page
- Dashboard
- [... 6 more]

Action Required: None (backward compatible)

Optional: Consider using loading state for async actions

Documentation: See Button component for loading state usage

````
</output>

---

## Step 8: Update Design System Stats

<action>
Update design system metadata:
</action>

**Update:** `D-Design-System/README.md`

```markdown
**Last Updated:** 2024-12-09
**Recent Changes:**
- Button [btn-001]: Added loading state
````

---

## Step 9: Complete

<output>
```
✅ Component Updated: Button [btn-001]

Update Type: Add new state
Changes:

- Added "loading" state to all variants
- Updated state documentation
- Version history updated

Impact:

- 9 pages affected
- No breaking changes
- Backward compatible

Next Steps:

- Pages can optionally use new loading state
- No immediate action required
- Consider updating high-traffic pages first

```
</output>

---

## Validation

<action>
Validate update:
- ✓ Component file updated
- ✓ Changes documented
- ✓ Version history updated
- ✓ Impact assessed
- ✓ Notifications sent (if needed)
- ✓ Backward compatibility maintained
</action>

---

## Error Handling

**If update creates breaking change:**
```

⚠️ Breaking Change Detected

This update will break existing usage:

- [List of breaking changes]
- Affected pages: [count]

Breaking changes require:

1. Designer confirmation
2. Migration plan
3. Page spec updates

Proceed with breaking change? (y/n)

If yes, I'll create a migration checklist.

```

**If component file locked:**
```

⚠️ Component file is being edited elsewhere.

Component: Button [btn-001]
Status: Locked by [user/process]

Options:

1. Wait and retry
2. Force update (may cause conflicts)
3. Cancel update

Your choice:

```

**If update conflicts with variants:**
```

⚠️ Update Conflict Detected

You're trying to add "loading" state to all variants,
but "navigation" variant already has a different loading implementation.

Current navigation loading: Spinner + icon animation
Proposed loading: Spinner only

Options:

1. Override navigation variant (make consistent)
2. Keep navigation variant different (document exception)
3. Cancel update

Your choice:

````

---

## Post-Update Actions

### If Breaking Change

<action>
Create migration checklist:
</action>

**Output:**
```markdown
# Migration Checklist: Button [btn-001] Update

**Update:** [Description]
**Breaking Changes:** [List]
**Affected Pages:** [Count]

## Migration Steps

- [ ] Review all affected pages
- [ ] Update page specifications
- [ ] Test updated pages
- [ ] Update documentation
- [ ] Deploy changes

## Affected Pages

- [ ] Login page - [Specific changes needed]
- [ ] Signup page - [Specific changes needed]
- [ ] Dashboard - [Specific changes needed]
[... more pages]

## Rollback Plan

If issues arise:
1. Revert component file to previous version
2. Restore page specifications
3. Document issues encountered
````

### If Major Update

<action>
Suggest design system review:
</action>

<output>
```
💡 Design System Health Check Recommended

This is a significant update to a widely-used component.

Consider reviewing:

- Component consistency across design system
- Other components that might need similar updates
- Overall design system patterns

Schedule a design system review session?

```
</output>

---

**This operation updates a component. Changes apply to all future usage automatically.**
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

ONLY WHEN [component is updated, version history tracked, and affected pages notified], will you then load and read fully `{nextStepFile}` to execute the next step.

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

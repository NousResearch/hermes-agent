---
name: 'step-06-present-decision'
description: 'Present complete analysis to designer with trade-offs for informed decision'

# File References
nextStepFile: './step-07-execute-decision.md'
---

# Step 6: Present Decision

## STEP GOAL:

Present the complete analysis to the designer with clear options, trade-off comparison, AI recommendation, and let the designer make an informed decision.

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

## Presentation Structure

### 1. Context Summary

What we're deciding and why

### 2. The Options

Clear description of each choice

### 3. Comparison Table

Side-by-side trade-offs

### 4. Recommendation

AI's suggestion based on analysis

### 5. Designer Choice

Let designer decide

---

## Step 1: Present Context

<output>
```
🔍 Design System Decision Needed

**Current Component:** Navigation Button
**Similar Component Found:** Button [btn-001]
**Similarity:** 72% (High)

**Key Similarities:**
✓ Visual appearance (size, shape, color)
✓ Primary action purpose
✓ Standard states

**Key Differences:**
✗ Navigation vs submission purpose
✗ Has icon and loading state
✗ Different usage context

**Decision:** How should we handle this in the design system?

```
</output>

---

## Step 2: Present Options

<output>
```

📋 Your Options:

**Option 1: Reuse Existing Component**
Use Button [btn-001], just change the label to "Continue"

**Option 2: Add Variant**
Add "navigation" variant to Button [btn-001]

- Button.primary (submit)
- Button.navigation (continue)

**Option 3: Create New Component**
Create separate "Navigation Button" component [btn-002]

```
</output>

---

## Step 3: Present Trade-Offs Table

<output>
```

📊 Trade-Offs Comparison:

┌─────────────┬──────────────────┬──────────────────┬──────────────────┐
│ │ Reuse [btn-001] │ Add Variant │ Create New │
├─────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Consistency │ ✅ Highest │ ✅ High │ ⚠️ Medium │
│ Maintenance │ ✅ Easiest │ ⚠️ Moderate │ ❌ More work │
│ Clarity │ ❌ Confusing │ ⚠️ Moderate │ ✅ Clearest │
│ Flexibility │ ❌ Constrained │ ✅ Flexible │ ✅ Independent │
│ Complexity │ ✅ Simple │ ⚠️ More complex │ ✅ Simple │
│ Semantics │ ❌ Lost │ ⚠️ Preserved │ ✅ Clear │
├─────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Best For │ Identical use │ Related purposes │ Different roles │
└─────────────┴──────────────────┴──────────────────┴──────────────────┘

Legend: ✅ Strong benefit | ⚠️ Trade-off | ❌ Potential issue

```
</output>

---

## Step 4: Present Detailed Analysis

<output>
```

🔍 Detailed Analysis:

**Option 1: Reuse Button [btn-001]**

Opportunities:
✅ Perfect visual consistency
✅ Single source of truth
✅ Easiest to maintain
✅ Simplest design system

Risks:
❌ HIGH: Semantic confusion (navigation looks like submission)
❌ HIGH: Accessibility issues (same component, different actions)
❌ MEDIUM: Developer confusion
❌ MEDIUM: Future constraint

Verdict: ⚠️ Not recommended due to semantic/accessibility risks

---

**Option 2: Add "Navigation" Variant to Button [btn-001]**

Opportunities:
✅ Maintains button family
✅ Flexible for different use cases
✅ Easy to add more variants
✅ All buttons documented together

Risks:
❌ HIGH: Variant explosion (could lead to 10+ variants)
⚠️ MEDIUM: Component complexity increases
⚠️ MEDIUM: Maintenance burden grows
⚠️ LOW: Documentation overhead

Mitigation: Strict variant guidelines can manage explosion risk

Verdict: ✅ Viable option if you expect more button types

---

**Option 3: Create New "Navigation Button" [btn-002]**

Opportunities:
✅ Clear semantic distinction
✅ Optimized for navigation use case
✅ Independent evolution
✅ Simple component definition

Risks:
⚠️ MEDIUM: Visual inconsistency over time
⚠️ MEDIUM: More components to manage
⚠️ LOW: Some code duplication
⚠️ LOW: Developer choice overhead

Mitigation: Shared design tokens ensure visual consistency

Verdict: ✅ Safest option, especially for semantically different purposes

```
</output>

---

## Step 5: Present AI Recommendation

<action>
Based on analysis, provide recommendation with reasoning
</action>

<output>
```

🤖 AI Recommendation: Create New Component (Option 3)

**Reasoning:**

1. **Semantic Distinction Matters**
   - Navigation and submission are fundamentally different actions
   - Users and developers benefit from clear distinction
   - Accessibility is better with separate components

2. **Manageable Risks**
   - Visual inconsistency can be prevented with shared tokens
   - Duplication is minimal and acceptable
   - Proliferation can be monitored

3. **Future Flexibility**
   - Each component can evolve independently
   - No risk of variant explosion
   - Clear component boundaries

4. **Your Project Context**
   - Design system is new (only 3 components)
   - Better to establish clear patterns now
   - Easier to merge later than split

**However:** If you expect many button types (delete, cancel, save, etc.),
Option 2 (variant) might be better for organization.

```
</output>

---

## Step 6: Ask for Designer Decision

<ask>
```

💭 Your Decision:

Based on this analysis, which approach fits your design intent?

[1] Reuse Button [btn-001]
→ Choose if: Visual consistency is paramount, purposes are actually the same

[2] Add "navigation" variant to Button [btn-001]
→ Choose if: You want button family, expect more button types

[3] Create new "Navigation Button" [btn-002]
→ Choose if: Semantic distinction matters, want independence

[4] I need more information
→ I can clarify any aspect of the analysis

Your choice (1/2/3/4):

```
</ask>

---

## Step 7: Handle Designer Response

<action>
Based on designer's choice, route to appropriate operation
</action>

**If Choice 1 (Reuse):**
```

✅ Got it - reusing Button [btn-001]

I'll update the page spec to reference the existing component.

```
**Route to:** `step-07-execute-decision.md` with action: `reuse`

**If Choice 2 (Variant):**
```

✅ Got it - adding "navigation" variant to Button [btn-001]

I'll update the component definition and create the reference.

```
**Route to:** `step-07-execute-decision.md` with action: `add_variant`

**If Choice 3 (New):**
```

✅ Got it - creating new Navigation Button [btn-002]

I'll create the new component and set up the reference.

```
**Route to:** `step-07-execute-decision.md` with action: `create_new`

**If Choice 4 (More Info):**
```

📚 What would you like to know more about?

- Similarity calculation details
- Specific opportunities or risks
- How variants work
- Component boundaries
- Something else

Your question:

```
**Provide clarification, then re-present decision**

---

## Presentation Variations

### For High Similarity (80%+)

<output>
```

✨ These components are very similar!

Similarity: 87%

The main question is: Are they the same thing with different content,
or different things that happen to look similar?

If same thing → Reuse
If different things → Variant or New

```
</output>

### For Low Similarity (40%-)

<output>
```

⚠️ These components are quite different.

Similarity: 38%

They share a type (Button) but serve different purposes.
Creating a new component is likely the best choice.

Would you like to proceed with creating a new component,
or would you like to see the full analysis?

```
</output>

### For Borderline Cases

<output>
```

🤔 This is a borderline case.

Similarity: 64% (right between "High" and "Medium")

This could go either way. I'll present both perspectives:

**Perspective 1: Similar Enough**
[Present variant option]

**Perspective 2: Different Enough**
[Present new component option]

Your design philosophy will guide this decision.

```
</output>

---

## Edge Cases

**Designer asks for recommendation:**
```

Based on the analysis, I recommend Option 3 (Create New).

But this is your design system - you know your project best.

What's most important to you?

- Consistency? → Reuse or Variant
- Clarity? → New
- Flexibility? → Variant
- Simplicity? → Reuse or New

```

**Designer is unsure:**
```

That's okay! This is a judgment call.

Here's a simple heuristic:

If a developer saw both buttons, would they think:
A) "Same button, different label" → Reuse
B) "Related buttons, different purposes" → Variant
C) "Different buttons entirely" → New

What's your gut feeling?

```

**Designer wants to defer decision:**
```

✅ No problem!

I'll create it as a new component for now.

You can always:

- Merge it later if you decide they're the same
- Convert it to a variant if you see a pattern
- Keep it separate if the distinction is valuable

Design systems evolve - this isn't permanent.

````

---

## Output Format

**For next step:**
```json
{
  "decision": {
    "choice": "create_new",
    "component_id": "btn-002",
    "reasoning": "Semantic distinction matters",
    "designer_notes": "Navigation and submission are different actions"
  }
}
````

### 7. Present MENU OPTIONS

Display: "**Select an Option:** [1/2/3/4] Choose option or request more info"

#### Menu Handling Logic:

- IF C: Update design log, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#7-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects the appropriate option
- User can chat or ask questions — always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN [designer has selected an option (1/2/3) and decision is confirmed], will you then load and read fully `{nextStepFile}` to execute the next step.

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

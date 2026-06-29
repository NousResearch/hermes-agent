---
name: 'step-04-identify-opportunities'
description: 'Identify potential benefits of each design system decision option: reuse, variant, or create new'

# File References
nextStepFile: './step-05-identify-risks.md'
---

# Step 4: Identify Opportunities

## STEP GOAL:

Identify potential benefits of each design system decision option (reuse existing, add variant, create new). Analyze opportunities across consistency, maintenance, flexibility, and project context.

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

## Decision Options

For each similar component, there are 3 options:

### Option 1: Reuse Existing Component

Use the existing component reference, just change content

### Option 2: Add Variant to Existing

Extend existing component with new variant

### Option 3: Create New Component

Create separate component in design system

---

## Opportunity Analysis Framework

### For Option 1: Reuse Existing

**Potential Opportunities:**

#### Consistency

- ✅ Visual consistency across pages
- ✅ Behavioral consistency (same interactions)
- ✅ User familiarity (looks/works the same)
- ✅ Brand coherence

#### Maintenance

- ✅ Single source of truth
- ✅ Update once, applies everywhere
- ✅ Easier to maintain
- ✅ Fewer files to manage

#### Development

- ✅ Faster development (component exists)
- ✅ Less code duplication
- ✅ Easier testing (test once)
- ✅ Better performance (reused code)

#### Design System

- ✅ Cleaner design system
- ✅ Fewer components to document
- ✅ Easier for developers to find
- ✅ Simpler component library

---

### For Option 2: Add Variant

**Potential Opportunities:**

#### Flexibility

- ✅ Accommodates different use cases
- ✅ Maintains component family
- ✅ Allows contextual adaptation
- ✅ Supports design evolution

#### Consistency

- ✅ Related components stay connected
- ✅ Shared base styling
- ✅ Consistent naming pattern
- ✅ Clear component relationships

#### Scalability

- ✅ Easy to add more variants later
- ✅ Supports design system growth
- ✅ Handles edge cases gracefully
- ✅ Accommodates future needs

#### Documentation

- ✅ Variants documented together
- ✅ Clear component family
- ✅ Easier to understand relationships
- ✅ Better developer guidance

---

### For Option 3: Create New

**Potential Opportunities:**

#### Clarity

- ✅ Clear separation of concerns
- ✅ Distinct purpose/function
- ✅ No confusion about usage
- ✅ Semantic clarity

#### Simplicity

- ✅ Simpler component definition
- ✅ No complex variant logic
- ✅ Easier to understand
- ✅ Fewer edge cases

#### Independence

- ✅ Can evolve independently
- ✅ No impact on other components
- ✅ Easier to modify
- ✅ No unintended side effects

#### Specificity

- ✅ Optimized for specific use case
- ✅ No unnecessary features
- ✅ Better performance
- ✅ Clearer developer intent

---

## Step 1: Analyze Current Situation

<action>
Based on similarity level and comparison, identify which opportunities apply
</action>

**Example (72% similarity):**

```
Current Situation:
- High visual similarity
- Different functional purpose (navigation vs submission)
- Some behavioral differences (loading state, icon)
- Similar usage context

Applicable Opportunities:
- Reuse: Consistency, maintenance benefits
- Variant: Flexibility, maintains family
- New: Clarity of purpose, independence
```

---

## Step 2: Generate Opportunity Lists

<output>
**Option 1: Reuse Button [btn-001]**

Opportunities:
✅ **Consistency:** All buttons look and behave the same
✅ **Maintenance:** Update button styling once, applies everywhere
✅ **Simplicity:** Fewer components in design system
✅ **Development:** Faster implementation (component exists)

Best if: Visual consistency is more important than functional distinction
</output>

<output>
**Option 2: Add "Navigation" Variant to Button [btn-001]**

Opportunities:
✅ **Flexibility:** Supports both submission and navigation use cases
✅ **Family:** Keeps related buttons together
✅ **Scalability:** Easy to add more button types later
✅ **Documentation:** All button variants in one place

Best if: You want to maintain button family but need different behaviors
</output>

<output>
**Option 3: Create New "Navigation Button" Component**

Opportunities:
✅ **Clarity:** Clear distinction between submission and navigation
✅ **Semantics:** Developers understand purpose immediately
✅ **Independence:** Can evolve without affecting submit buttons
✅ **Optimization:** Tailored for navigation use case

Best if: Functional distinction is more important than visual consistency
</output>

---

## Step 3: Highlight Strongest Opportunities

<action>
Based on comparison data, identify the most compelling opportunities for each option
</action>

**Example:**

```
🌟 Strongest Opportunities:

**For Reuse:**
- Your buttons are 90% visually identical
- Consistency would be very strong
- Maintenance would be significantly easier

**For Variant:**
- You have 2 distinct button purposes emerging
- Variant structure would accommodate both
- Future button types could fit this pattern

**For New:**
- Navigation and submission are semantically different
- Developers would benefit from clear distinction
- Each could evolve independently as needs change
```

---

## Step 4: Consider Project Context

<action>
Factor in project-specific considerations:
- Design system maturity (new vs established)
- Team size (solo vs large team)
- Project complexity (simple vs complex)
- Timeline (fast vs thorough)
</action>

**Example:**

```
📋 Project Context:

Design System: New (3 components so far)
Team: Small (2-3 people)
Complexity: Medium
Timeline: Moderate

Context-Specific Opportunities:
- **New design system:** Easier to keep simple (favors reuse/variant)
- **Small team:** Fewer components = easier maintenance (favors reuse)
- **Medium complexity:** Room for some structure (favors variant)
```

---

## Step 5: Pass to Next Step

<action>
Pass opportunity analysis to risk identification:
- Opportunities for each option
- Strongest opportunities
- Context considerations
</action>

**Next:** `step-05-identify-risks.md`

---

## Edge Cases

**All options have strong opportunities:**

```
✨ All Options Look Good!

Each approach has compelling opportunities:
- Reuse: Strong consistency benefits
- Variant: Good balance of flexibility
- New: Clear semantic distinction

This means the risks will be the deciding factor.
```

**No clear opportunities:**

```
⚠️ No Strong Opportunities Identified

This might mean:
- Components are too different to benefit from connection
- Or too similar to benefit from separation

I'll focus on risks to help clarify the decision.
```

**Conflicting opportunities:**

```
⚠️ Conflicting Opportunities

Reuse offers consistency, but New offers clarity.
These are competing values.

Your design philosophy will guide this decision:
- Value consistency? → Reuse
- Value semantics? → New
```

---

## Output Format

**For next step:**

```json
{
  "opportunities": {
    "reuse": {
      "consistency": "high",
      "maintenance": "high",
      "development": "medium",
      "strongest": ["consistency", "maintenance"]
    },
    "variant": {
      "flexibility": "high",
      "family": "medium",
      "scalability": "high",
      "strongest": ["flexibility", "scalability"]
    },
    "new": {
      "clarity": "high",
      "independence": "high",
      "specificity": "medium",
      "strongest": ["clarity", "independence"]
    }
  }
}
```

### 8. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Identify Risks"

#### Menu Handling Logic:

- IF C: Update design log, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#8-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects the appropriate option
- User can chat or ask questions — always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN [C continue option is selected and opportunities identified for all three options], will you then load and read fully `{nextStepFile}` to execute the next step.

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

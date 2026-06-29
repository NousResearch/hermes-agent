---
name: 'step-02-compare-attributes'
description: 'Systematically compare current component to existing candidates across visual, functional, behavioral, and contextual dimensions'

# File References
nextStepFile: './step-03-calculate-similarity.md'
---

# Step 2: Compare Attributes

## STEP GOAL:

Systematically compare the current component specification against existing candidates across four dimensions: visual, functional, behavioral, and contextual attributes.

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

## Comparison Framework

**Compare across 4 dimensions:**

### 1. Visual Attributes

- Size (small, medium, large)
- Shape (rounded, square, pill)
- Color scheme
- Typography
- Spacing/padding
- Border style

### 2. Functional Attributes

- Purpose/intent
- User action
- Input/output type
- Validation rules
- Required/optional

### 3. Behavioral Attributes

- States (default, hover, active, disabled, loading, error)
- Interactions (click, hover, focus, blur)
- Animations/transitions
- Keyboard support
- Accessibility

### 4. Contextual Attributes

- Usage pattern (where it appears)
- Frequency (how often used)
- Relationship to other components
- User journey stage

---

## Step 1: Visual Comparison

<action>
Compare visual attributes:
- Extract visual properties from current spec
- Extract visual properties from candidate
- Calculate matches and differences
</action>

**Example:**

```
Visual Comparison: Current Button vs Button [btn-001]

Similarities:
✓ Size: medium (both)
✓ Shape: rounded (both)
✓ Color scheme: blue primary (both)

Differences:
✗ Current: Has icon on left
✗ btn-001: Text only
✗ Current: Slightly larger padding
```

---

## Step 2: Functional Comparison

<action>
Compare functional attributes:
- What does it do?
- What's the user intent?
- What's the outcome?
</action>

**Example:**

```
Functional Comparison: Current Button vs Button [btn-001]

Similarities:
✓ Purpose: Primary action trigger
✓ User action: Click to submit/proceed
✓ Outcome: Form submission or navigation

Differences:
✗ Current: "Continue to next step"
✗ btn-001: "Submit form"
✗ Current: Navigation action
✗ btn-001: Form submission action
```

---

## Step 3: Behavioral Comparison

<action>
Compare behavioral attributes:
- States
- Interactions
- Animations
</action>

**Example:**

```
Behavioral Comparison: Current Button vs Button [btn-001]

Similarities:
✓ States: default, hover, active, disabled (both)
✓ Hover: Darkens background (both)
✓ Disabled: Grayed out (both)

Differences:
✗ Current: Has loading state with spinner
✗ btn-001: No loading state
✗ Current: Icon rotates on hover
```

---

## Step 4: Contextual Comparison

<action>
Compare contextual attributes:
- Where is it used?
- How often?
- What's the pattern?
</action>

**Example:**

```
Contextual Comparison: Current Button vs Button [btn-001]

Similarities:
✓ Both: Primary action in forms
✓ Both: Bottom-right of containers
✓ Both: High-frequency usage

Differences:
✗ Current: Multi-step flow navigation
✗ btn-001: Single-page form submission
✗ Current: Always has "next" context
```

---

## Step 5: Calculate Similarity Score

<action>
Score each dimension:
- Visual: High/Medium/Low similarity
- Functional: High/Medium/Low similarity
- Behavioral: High/Medium/Low similarity
- Contextual: High/Medium/Low similarity
</action>

**Scoring Guide:**

- **High:** 80%+ attributes match
- **Medium:** 50-79% attributes match
- **Low:** <50% attributes match

**Example:**

```
Similarity Score: Current Button vs Button [btn-001]

Visual:      High (90% match)
Functional:  Medium (60% match)
Behavioral:  Medium (70% match)
Contextual:  Medium (65% match)

Overall:     Medium-High Similarity
```

---

## Step 6: Summarize Comparison

<output>
Present comparison summary:

```
📊 Comparison: Current Button vs Button [btn-001]

**Similarities:**
✓ Visual appearance (size, shape, color)
✓ Primary action purpose
✓ Standard states (default, hover, active, disabled)
✓ High-frequency usage pattern

**Differences:**
✗ Current has icon, btn-001 is text-only
✗ Current has loading state, btn-001 doesn't
✗ Current for navigation, btn-001 for submission
✗ Current has icon animation

**Similarity Score:** Medium-High (71%)
```

</output>

---

## Step 7: Pass to Next Step

<action>
Pass comparison data to similarity calculation:
- Detailed comparison
- Similarity scores
- Key differences
</action>

**Next:** `step-03-calculate-similarity.md`

---

## Edge Cases

**Perfect match (100%):**

```
✓ This component is identical to btn-001.

This is likely the same component with different content.
```

**Recommend:** Reuse existing component

**Very low similarity (<30%):**

```
✗ This component is very different from btn-001.

Despite being the same type, these serve different purposes.
```

**Recommend:** Create new component

**Multiple candidates:**

```
📊 Comparing to 2 candidates:

Button [btn-001]: 71% similarity
Icon Button [btn-002]: 45% similarity

btn-001 is the closest match.
```

**Continue with best match**

---

## Output Format

**For next step:**

```json
{
  "comparison": {
    "candidate_id": "btn-001",
    "visual_similarity": "high",
    "functional_similarity": "medium",
    "behavioral_similarity": "medium",
    "contextual_similarity": "medium",
    "overall_score": 0.71,
    "similarities": [...],
    "differences": [...]
  }
}
```

### 8. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Calculate Similarity"

#### Menu Handling Logic:

- IF C: Update design log, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#8-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects the appropriate option
- User can chat or ask questions — always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN [C continue option is selected and all four dimensions compared with scores assigned], will you then load and read fully `{nextStepFile}` to execute the next step.

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

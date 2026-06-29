---
name: 'step-03-calculate-similarity'
description: 'Interpret comparison data, calculate weighted similarity score, and classify similarity level'

# File References
nextStepFile: './step-04-identify-opportunities.md'
---

# Step 3: Calculate Similarity

## STEP GOAL:

Interpret the comparison data, apply weighted scoring to calculate an overall similarity percentage, classify the similarity level, and generate an initial recommendation.

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

## Similarity Levels

### Level 1: Identical (95-100%)

**Characteristics:**

- All visual attributes match
- Same functional purpose
- Same behavioral patterns
- Only content differs (labels, text)

**Interpretation:** This is the same component

**Recommendation:** Reuse existing component reference

---

### Level 2: Very High Similarity (80-94%)

**Characteristics:**

- Visual attributes mostly match
- Same core function
- Minor behavioral differences
- Same usage context

**Interpretation:** This is likely the same component with minor variations

**Recommendation:** Consider adding variant to existing component

---

### Level 3: High Similarity (65-79%)

**Characteristics:**

- Visual attributes similar
- Related functional purpose
- Some behavioral differences
- Similar usage context

**Interpretation:** Could be same component or new variant

**Recommendation:** Designer decision needed - variant or new?

---

### Level 4: Medium Similarity (45-64%)

**Characteristics:**

- Some visual overlap
- Different functional purpose
- Different behaviors
- Different usage context

**Interpretation:** Related but distinct components

**Recommendation:** Likely new component, but designer should confirm

---

### Level 5: Low Similarity (20-44%)

**Characteristics:**

- Minimal visual overlap
- Different function
- Different behaviors
- Different context

**Interpretation:** Different components that happen to share a type

**Recommendation:** Create new component

---

### Level 6: No Similarity (<20%)

**Characteristics:**

- No meaningful overlap
- Completely different purpose
- Unrelated patterns

**Interpretation:** Unrelated components

**Recommendation:** Definitely create new component

---

## Calculation Logic

<action>
Calculate overall similarity:
1. Weight each dimension:
   - Visual: 30%
   - Functional: 30%
   - Behavioral: 25%
   - Contextual: 15%

2. Convert dimension scores to numeric:
   - High = 1.0
   - Medium = 0.6
   - Low = 0.2

3. Calculate weighted average:
   - Overall = (Visual × 0.3) + (Functional × 0.3) + (Behavioral × 0.25) + (Contextual × 0.15)

4. Convert to percentage:
   - Similarity % = Overall × 100
     </action>

**Example:**

```
Dimension Scores:
- Visual: High (1.0)
- Functional: Medium (0.6)
- Behavioral: Medium (0.6)
- Contextual: Medium (0.6)

Calculation:
(1.0 × 0.3) + (0.6 × 0.3) + (0.6 × 0.25) + (0.6 × 0.15)
= 0.3 + 0.18 + 0.15 + 0.09
= 0.72

Similarity: 72% (High Similarity - Level 3)
```

---

## Step 1: Calculate Score

<action>
Apply calculation logic to comparison data
</action>

<output>
```
📊 Similarity Calculation

Visual: High (1.0) × 30% = 0.30
Functional: Medium (0.6) × 30% = 0.18
Behavioral: Medium (0.6) × 25% = 0.15
Contextual: Medium (0.6) × 15% = 0.09

Overall Similarity: 72%
Level: High Similarity (Level 3)

```
</output>

---

## Step 2: Classify Similarity

<action>
Map percentage to similarity level
</action>

<output>
```

**Similarity Level: High (72%)**

This component is similar to Button [btn-001] but has some differences.

Could be:

- A variant of the existing button
- A new related button component

Designer decision needed.

```
</output>

---

## Step 3: Generate Recommendation

<action>
Based on similarity level, generate recommendation with reasoning
</action>

**For Level 1-2 (Identical/Very High):**
```

✅ Recommendation: Reuse existing component

Reasoning:

- Components are nearly identical
- Only content/labels differ
- Same visual and behavioral patterns
- Maintaining consistency is straightforward

```

**For Level 3 (High):**
```

🤔 Recommendation: Designer decision needed

This could go either way:

- Similar enough to be a variant
- Different enough to be separate

I'll present the trade-offs so you can decide.

```

**For Level 4-5 (Medium/Low):**
```

🆕 Recommendation: Create new component

Reasoning:

- Significant functional differences
- Different usage contexts
- Trying to merge would create complexity
- Better to keep separate

```

**For Level 6 (No similarity):**
```

✅ Recommendation: Definitely create new component

Reasoning:

- Components are fundamentally different
- No meaningful overlap
- No benefit to linking them

```

---

## Step 4: Identify Key Decision Factors

<action>
Highlight the most important differences that affect the decision
</action>

**Example:**
```

🔑 Key Decision Factors:

1. **Icon presence** - Current has icon, existing doesn't
   Impact: Visual consistency, component complexity

2. **Loading state** - Current has loading, existing doesn't
   Impact: Behavioral complexity, reusability

3. **Navigation vs Submission** - Different purposes
   Impact: Semantic meaning, developer understanding

These differences will affect your decision.

```

---

## Step 5: Pass to Next Step

<action>
Pass classification and recommendation to opportunity identification:
- Similarity level
- Recommendation
- Key decision factors
</action>

**Next:** `step-04-identify-opportunities.md`

---

## Edge Cases

**Borderline cases (near threshold):**
```

⚠️ Borderline Case: 64% similarity

This is right on the edge between "High" and "Medium" similarity.

I'll present both perspectives so you can make an informed decision.

```

**Multiple candidates with similar scores:**
```

📊 Multiple Similar Candidates:

Button [btn-001]: 72% similarity
Button [btn-003]: 68% similarity

btn-001 is slightly closer, but both are viable options.
I'll compare to btn-001 for the decision.

```

**Perfect match but different context:**
```

⚠️ Unusual Pattern: 98% similarity but different context

Visually and behaviorally identical, but used in completely different contexts.

This might indicate:

- Same component, different use case ✓
- Accidental duplication ⚠️
- Context-specific variant needed 🤔

````

---

## Output Format

**For next step:**
```json
{
  "similarity": {
    "percentage": 72,
    "level": "high",
    "level_number": 3,
    "recommendation": "designer_decision",
    "key_factors": [
      "Icon presence",
      "Loading state",
      "Navigation vs Submission"
    ]
  }
}
````

### 6. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Identify Opportunities"

#### Menu Handling Logic:

- IF C: Update design log, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#6-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects the appropriate option
- User can chat or ask questions — always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN [C continue option is selected and similarity calculated and classified], will you then load and read fully `{nextStepFile}` to execute the next step.

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

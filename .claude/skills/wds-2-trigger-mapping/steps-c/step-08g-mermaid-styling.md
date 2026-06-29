---
name: 'step-08g-mermaid-styling'
description: 'Apply professional light gray styling with dark text to all diagram nodes'

# File References
nextStepFile: './step-08h-mermaid-quality.md'
activityWorkflowFile: '../workflow.md'
---

# Step 30: Apply Styling

## STEP GOAL:

Apply professional light gray styling with dark text to all nodes using four style classes: businessGoal, platform, targetGroup, and drivingForces, with exact color specifications.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are Saga the Analyst - applying professional visual styling
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring structured facilitation and pattern recognition, user brings business knowledge and user insight
- ✅ Work together as equals in a partnership, not a client-vendor relationship

### Step-Specific Rules:

- 🎯 Focus on applying EXACT color specifications
- 🚫 FORBIDDEN to modify colors or create custom color schemes
- 💬 Approach: Apply exact specifications, no creative liberties with colors
- 📋 Use these EXACT colors - do not modify
- 📋 Platform gets 3px border (thicker than others at 2px)

## EXECUTION PROTOCOLS:

- 🎯 Define all four classDef statements with exact colors
- 💾 Store styling_definitions, styling_applications, complete_diagram
- 📖 Apply classes to correct node groups
- 🚫 Do not modify the color specifications

## CONTEXT BOUNDARIES:

- Available context: All nodes and connections from previous steps
- Focus: Applying consistent professional styling
- Limits: Use EXACT colors specified - no variations
- Dependencies: Requires all nodes and connections created

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Define Style Classes

Add these EXACT class definitions:

```css
classDef businessGoal fill:#f3f4f6,color:#1f2937,stroke:#d1d5db,stroke-width:2px
classDef platform fill:#e5e7eb,color:#111827,stroke:#9ca3af,stroke-width:3px
classDef targetGroup fill:#f9fafb,color:#1f2937,stroke:#d1d5db,stroke-width:2px
classDef drivingForces fill:#f3f4f6,color:#1f2937,stroke:#d1d5db,stroke-width:2px
```

**Color Specifications:**

**Background fills:**
- `#f3f4f6` - Light gray (business goals, driving forces)
- `#e5e7eb` - Medium gray (platform only)
- `#f9fafb` - Near white (target groups)

**Text colors:**
- `#1f2937` - Dark gray (most nodes)
- `#111827` - Darker gray (platform only)

**Border colors:**
- `#d1d5db` - Light gray border (most nodes)
- `#9ca3af` - Medium gray border (platform only)

**Border widths:**
- `2px` - Standard (business goals, target groups, driving forces)
- `3px` - Thick (platform only)

### 2. Apply Classes to Nodes

Format:
```
class BG0,BG1,BG2 businessGoal
class PLATFORM platform
class TG0,TG1,TG2 targetGroup
class DF0,DF1,DF2 drivingForces
```

Adjust node counts to match actual diagram.

### 3. Verify Rules Checklist

- All four classDef statements included
- Colors EXACTLY match specification
- Platform has 3px border
- All BG nodes assigned to businessGoal
- PLATFORM assigned to platform
- All TG nodes assigned to targetGroup
- All DF nodes assigned to drivingForces
- Node counts match actual diagram
- Comment header included

Store styling_definitions, styling_applications, and complete_diagram.

### 4. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Quality Check | [M] Return to Activity Menu"

#### Menu Handling Logic:
- IF C: Load and execute {nextStepFile}
- IF M: Return to {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN user selects [C] will you load the next step file. Styling must be applied correctly before quality check.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- All four classDef statements with exact colors
- Platform thicker border (3px vs 2px)
- All nodes assigned to correct classes
- Node counts match actual diagram
- Professional, subtle, print-friendly appearance
- Complete diagram assembled

### ❌ SYSTEM FAILURE:
- Wrong color codes
- Missing classDef statements
- Nodes not assigned to classes
- Wrong border widths
- Node count mismatch
- Custom colors used instead of specified

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

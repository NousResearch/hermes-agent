---
name: 'step-08e-mermaid-driving-forces'
description: 'Format driving forces nodes with wants and fears for each persona'

# File References
nextStepFile: './step-08f-mermaid-connections.md'
activityWorkflowFile: '../workflow.md'
---

# Step 28: Format Driving Forces Nodes

## STEP GOAL:

Create driving forces nodes with WANTS (checkmark) and FEARS (X) sections for each persona, using the SAME emoji as the corresponding TG node and exactly 3 drivers per category.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are Saga the Analyst - creating professional diagram nodes
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring structured facilitation and pattern recognition, user brings business knowledge and user insight
- ✅ Work together as equals in a partnership, not a client-vendor relationship

### Step-Specific Rules:

- 🎯 Focus on creating DF nodes with exactly 3 wants and 3 fears
- 🚫 FORBIDDEN to use different emoji than corresponding TG node or add emojis to WANTS/FEARS headers
- 💬 Approach: Systematic creation matching TG node emojis exactly
- 📋 Exactly 3 positive drivers with checkmark, exactly 3 negative with X
- 📋 WANTS and FEARS headers are plain text (no emojis, ALL CAPS)

## EXECUTION PROTOCOLS:

- 🎯 Format each DF node with matching TG emoji
- 💾 Store driving_forces_nodes, verify emoji matching
- 📖 Follow exact node structure with WANTS and FEARS sections
- 🚫 Do not deviate from exactly 3 drivers per category

## CONTEXT BOUNDARIES:

- Available context: Driving forces from workshops, persona_emojis from step-08d
- Focus: Formatting DF nodes for Mermaid diagram
- Limits: MUST use same emoji as corresponding TG node
- Dependencies: Requires TG nodes and persona_emojis from step-08d

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Format Each Driving Forces Node

**Node Structure Template:**
```
DFX["<br/>EMOJI PERSONA'S DRIVERS<br/><br/>WANTS<br/>checkmark Positive driver 1<br/>checkmark Positive driver 2<br/>checkmark Positive driver 3<br/><br/>FEARS<br/>X Negative driver 1<br/>X Negative driver 2<br/>X Negative driver 3<br/><br/>"]
```

**Required elements per node:**
1. Start with `<br/>` (top padding)
2. **Same emoji as corresponding TG node** + "PERSONA'S DRIVERS" in ALL CAPS
3. Blank line (`<br/><br/>`)
4. "WANTS" header (no emoji, ALL CAPS)
5. Exactly 3 positive drivers with checkmark emoji
6. Blank line (`<br/><br/>`)
7. "FEARS" header (no emoji, ALL CAPS)
8. Exactly 3 negative drivers with X emoji
9. End with `<br/><br/>` (bottom padding)

### 2. Critical Emoji Rules

**Matching emoji:**
- DF node MUST use same emoji as corresponding TG node
- TG0 emoji -> DF0 (same emoji)
- TG1 emoji -> DF1 (same emoji)
- TG2 emoji -> DF2 (same emoji)

**Driver emojis:**
- Checkmark for all positive drivers
- X for all negative drivers
- NO emojis on "WANTS" and "FEARS" headers

### 3. Driver Formatting

Each driver:
- Starts with emoji (checkmark or X)
- One space after emoji
- Concise text (keep under 40 characters if possible)
- Ends with `<br/>`

**Exactly 3 drivers per category** - no more, no less.

### 4. Verify Rules Checklist

- Node ID follows pattern DF0, DF1, DF2 (matching TG nodes)
- Starts with `<br/>`
- Emoji matches corresponding TG node emoji
- "PERSONA'S DRIVERS" in ALL CAPS
- Blank line after title
- "WANTS" header (no emoji, ALL CAPS)
- Exactly 3 positive drivers with checkmark
- Blank line between sections
- "FEARS" header (no emoji, ALL CAPS)
- Exactly 3 negative drivers with X
- Ends with `<br/><br/>`
- No HTML tags
- Proper quote and bracket closure `"]`

Store driving_forces_nodes and verify emoji matching with TG nodes.

### 5. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Create Connections | [M] Return to Activity Menu"

#### Menu Handling Logic:
- IF C: Load and execute {nextStepFile}
- IF M: Return to {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN user selects [C] will you load the next step file. All DF nodes must be formatted with matching emojis before proceeding.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- All DF nodes formatted with matching TG emojis
- Exactly 3 positive and 3 negative drivers per persona
- WANTS and FEARS headers plain text (no emojis)
- Drivers concise (under 40 chars)
- Proper padding and spacing
- Emoji matching verified

### ❌ SYSTEM FAILURE:
- Different emoji than corresponding TG node
- More or fewer than 3 drivers per category
- Emojis on WANTS/FEARS headers
- Missing blank line between sections
- Drivers too long
- Emoji matching not verified

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

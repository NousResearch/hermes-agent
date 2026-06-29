---
name: 'step-08d-mermaid-target-groups'
description: 'Format target group nodes with emojis, priority levels, and profile traits'

# File References
nextStepFile: './step-08e-mermaid-driving-forces.md'
activityWorkflowFile: '../workflow.md'
---

# Step 27: Format Target Group Nodes

## STEP GOAL:

Create persona nodes with emojis, ALL CAPS names, priority levels (PRIMARY/SECONDARY/TERTIARY TARGET), and 3-4 key profile traits.

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

- 🎯 Focus on creating target group nodes with consistent structure
- 🚫 FORBIDDEN to use different emojis for TG and its corresponding DF node
- 💬 Approach: Consistent formatting with proper padding
- 📋 Use same emoji for TG node and its corresponding DF node (critical!)
- 📋 Priority levels in ALL CAPS: PRIMARY TARGET, SECONDARY TARGET, etc.

## EXECUTION PROTOCOLS:

- 🎯 Format each persona as a properly structured TG node
- 💾 Store target_group_nodes, persona_emojis (for DF matching), persona_count
- 📖 Record emoji assignments for use in DF nodes
- 🚫 Do not forget to record persona emojis for DF matching

## CONTEXT BOUNDARIES:

- Available context: Personas from workshops, platform node from step-08c
- Focus: Formatting TG nodes for Mermaid diagram
- Limits: Follow exact template pattern, record emojis for DF matching
- Dependencies: Requires platform node from step-08c

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Format Each Target Group Node

**Node Structure Template:**
```
TGX["<br/>EMOJI PERSONA NAME<br/>PRIORITY LEVEL<br/><br/>Profile trait 1<br/>Profile trait 2<br/>Profile trait 3<br/><br/>"]
```

**Required elements per node:**
1. Start with `<br/>` (top padding)
2. Emoji + Persona name in ALL CAPS
3. Priority level (PRIMARY TARGET, SECONDARY TARGET, etc.) in ALL CAPS
4. Blank line (`<br/><br/>`)
5. 3-4 key profile traits (concise, one line each with `<br/>`)
6. End with `<br/><br/>` (bottom padding)

### 2. Choose and Record Persona Emojis

**Important:** Use same emoji for both TG node and corresponding DF node.

Common persona emojis:
- Target/Strategic: target emoji
- Business/Leadership: briefcase emoji
- Technical/Developer: computer emoji
- Team/Group: people emoji
- Creative/Designer: art emoji
- User/Customer: phone emoji

Record emoji assignments: TG0 emoji -> DF0, TG1 emoji -> DF1, etc.

### 3. Verify Rules Checklist

- Node ID follows pattern TG0, TG1, TG2
- Starts with `<br/>`
- Emoji matches persona type
- Persona name in ALL CAPS
- Priority level in ALL CAPS
- Blank line after priority (`<br/><br/>`)
- 3-4 profile traits
- Each trait ends with `<br/>`
- Ends with `<br/><br/>`
- No HTML tags
- Proper quote and bracket closure `"]`

Store target_group_nodes, persona_emojis, persona_count.

### 4. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Format Driving Forces | [M] Return to Activity Menu"

#### Menu Handling Logic:
- IF C: Load and execute {nextStepFile}
- IF M: Return to {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN user selects [C] will you load the next step file. All TG nodes must be formatted and emojis recorded before proceeding.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- All persona nodes formatted following template
- Emojis selected and RECORDED for DF matching
- Priority levels in ALL CAPS
- 3-4 profile traits per node
- Proper padding and closure
- No HTML tags

### ❌ SYSTEM FAILURE:
- Not recording emojis for DF matching
- Missing priority levels
- Traits not concise
- HTML tags in nodes
- Inconsistent formatting
- Wrong node ID pattern

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

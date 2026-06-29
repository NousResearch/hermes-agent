---
name: 'step-08f-mermaid-connections'
description: 'Create connections between all nodes in the proper flow pattern'

# File References
nextStepFile: './step-08g-mermaid-styling.md'
activityWorkflowFile: '../workflow.md'
---

# Step 29: Create Connections

## STEP GOAL:

Connect all nodes in the proper flow pattern: Business Goals -> Platform -> Target Groups -> Driving Forces, using simple arrows with proper section comments.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are Saga the Analyst - creating professional diagram connections
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring structured facilitation and pattern recognition, user brings business knowledge and user insight
- ✅ Work together as equals in a partnership, not a client-vendor relationship

### Step-Specific Rules:

- 🎯 Focus on connecting all nodes with simple arrows
- 🚫 FORBIDDEN to use fancy arrow styling or skip any connection
- 💬 Approach: Systematic connection creation with verification
- 📋 Use simple `-->` arrows only
- 📋 TG-to-DF connections must match (TG0->DF0, TG1->DF1, etc.)

## EXECUTION PROTOCOLS:

- 🎯 Create all connections following the pattern
- 💾 Store connections and connection_count
- 📖 Include section comments for each connection group
- 🚫 Do not use fancy arrow styling

## CONTEXT BOUNDARIES:

- Available context: All node IDs from previous steps
- Focus: Creating connections between all nodes
- Limits: Simple arrows only, matching TG-DF pairs
- Dependencies: Requires all nodes formatted

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Business Goals to Platform

Connect all BG nodes to PLATFORM with simple `-->` arrows.

### 2. Platform to Target Groups

Connect PLATFORM to all TG nodes with simple `-->` arrows.

### 3. Target Groups to Driving Forces

Connect each TG to its corresponding DF (matching IDs: TG0->DF0, TG1->DF1, etc.).

### 4. Verify Connection Count

**Count check:**
- BG connections = number of business goals
- Platform-to-TG connections = number of personas
- TG-to-DF connections = number of personas

Example for 3 personas: 3 + 3 + 3 = 9 total connections.

Store connections and connection_count.

### 5. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Apply Styling | [M] Return to Activity Menu"

#### Menu Handling Logic:
- IF C: Load and execute {nextStepFile}
- IF M: Return to {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN user selects [C] will you load the next step file. All connections must be created and verified before proceeding.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- All BG nodes connected to PLATFORM
- PLATFORM connected to all TG nodes
- Each TG connected to matching DF
- Simple `-->` arrows used throughout
- Section comments included
- Connection count verified
- No broken connections

### ❌ SYSTEM FAILURE:
- Missing connections
- Fancy arrow styling
- TG-DF mismatch (TG0->DF1 etc.)
- Missing section comments
- Broken connections
- Wrong connection count

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

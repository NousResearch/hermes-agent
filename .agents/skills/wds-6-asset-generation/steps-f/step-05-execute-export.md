---
name: 'step-05-execute-export'
description: 'Send validated HTML to Figma via MCP and verify the export succeeded'
workflowFile: '../workflow.md'
---

# Step 5: Send to Figma

## STEP GOAL:

Execute the final export by sending validated HTML to Figma via MCP, verify the layers appear with proper OBJECT ID naming, and complete the Figma export workflow.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a technical export specialist executing and verifying the Figma delivery
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring MCP export expertise, user brings their Figma verification
- ✅ Maintain a confident, delivery-focused tone

### Step-Specific Rules:

- 🎯 Focus on executing the export and verifying success in Figma
- 🚫 FORBIDDEN to skip user verification of export in Figma
- 💬 Provide troubleshooting guidance if export is not visible
- 📋 Document complete export summary with details

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Record export details (node ID, component count, OBJECT IDs)
- 📖 Wait for MCP response before asking user to verify
- 🚫 FORBIDDEN to mark workflow complete without user confirming export visible

## CONTEXT BOUNDARIES:

- Available context: Validated HTML, OBJECT IDs, scenario type
- Focus: Executing the MCP export and verifying results
- Limits: This is the final step — focus on delivery and verification
- Dependencies: Validated HTML from Step 4

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Prepare Export Parameters

Set up MCP tool call: descriptive name for Figma layer (format: "{Component/Page Name} - {Purpose}"), complete validated HTML, optional intoNodeId for updating existing layer.

### 2. Execute Export

Call the MCP tool with prepared parameters. Wait for response.

### 3. Verify Export Response

Check response for success indicators: node ID returned, no error message, response contains node object.

### 4. User Verification

Ask: **"Please check your Figma file — can you see the export with proper layer names?"**

- If Yes: Proceed to success report
- If No: Execute troubleshooting (check Figma is open, correct file active, layers panel, all pages, MCP connection)

### 5. Present Success Report

Display complete export details: name, node ID, component count, OBJECT IDs used, layer names in Figma.

### 6. Document Completion

Record: scenario type, components exported, OBJECT IDs used, specification files referenced, Figma output location.

### 7. Present MENU OPTIONS

Display: **"Select an Option:** [M] Return to Activity Menu"

#### Menu Handling Logic:

- IF M: Save export record, update design log, return to Activity Menu in {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#7-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions — always respond and then end with display again of the menu options

## CRITICAL STEP COMPLETION NOTE

This is the final step of the Figma Export workflow. When M is selected and the export is verified, return to the Activity Menu.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Export executed via MCP without errors
- User confirms export visible in Figma
- Layer names match OBJECT IDs
- Complete export summary documented
- Design log updated

### ❌ SYSTEM FAILURE:

- Not verifying export with user
- Marking complete when export failed
- Not providing troubleshooting for invisible exports
- Skipping export summary documentation

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

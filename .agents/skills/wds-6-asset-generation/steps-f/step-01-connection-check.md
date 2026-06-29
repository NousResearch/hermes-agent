---
name: 'step-01-connection-check'
description: 'Verify html.to.design MCP server connection and guide setup if needed'
nextStepFile: './step-02-identify-export-type.md'
---

# Step 1: Connection Check and Installation

## STEP GOAL:

Verify that the html.to.design MCP server is connected and functional before proceeding with the Figma export workflow, guiding the user through setup if needed.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a technical integration specialist verifying the Figma export pipeline
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring MCP integration expertise, user brings their Figma environment setup
- ✅ Maintain a helpful, technical tone

### Step-Specific Rules:

- 🎯 Focus ONLY on verifying MCP tool availability and connection
- 🚫 FORBIDDEN to proceed to export without verified connection
- 💬 If tool is not available, guide through setup or suggest alternative
- 📋 A successful test export must be confirmed before proceeding

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Record connection status
- 📖 Reference Figma Plugin Setup Guide if setup is needed
- 🚫 FORBIDDEN to skip connection verification

## CONTEXT BOUNDARIES:

- Available context: Project configuration, MCP tool availability
- Focus: Verifying html.to.design MCP server connection
- Limits: Do not start any export work — just verify the connection
- Dependencies: Figma account with project access, html.to.design plugin

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Check MCP Tool Availability

Check if `mcp2_import-html` tool is accessible in current session. Tool should be from "html.to.design MCP server."

- If tool available: Skip to step 4 (verification)
- If tool not available: Continue with setup guidance

### 2. Guide Setup (If Needed)

Inform user that setup requires:
1. Figma account with project access
2. html.to.design plugin installed in Figma
3. MCP server configured in IDE

Ask: **"Would you like me to guide you through the setup process?"**

- If Yes: Reference the Figma Plugin Setup Guide at `../data/figma-plugin-setup.md`
- If No: Stop workflow, suggest alternative approach

### 3. Verify After Setup

Once setup is complete, return to verification.

### 4. Execute Test Export

Execute a test export to verify connection:

```javascript
mcp2_import-html({
  name: "Connection Test",
  html: "<div style='padding: 20px; background: #f0f0f0; border-radius: 8px; font-family: sans-serif;'>Connection Test Successful</div>"
})
```

Ask: **"Can you see the 'Connection Test' layer in your Figma file?"**

- If Yes: Connection verified
- If No: Execute troubleshooting steps

### 5. Present MENU OPTIONS

Display: **"Select an Option:** [C] Continue"

#### Menu Handling Logic:

- IF C: Record connection as verified, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#5-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions — always respond and then end with display again of the menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN C is selected and the connection is verified will you load {nextStepFile} to begin identifying the export type.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- MCP tool availability checked
- Connection verified with test export
- User confirms test layer visible in Figma
- Setup guidance provided if needed

### ❌ SYSTEM FAILURE:

- Proceeding to export without verified connection
- Not offering setup guidance when tool is unavailable
- Skipping test export verification
- Not waiting for user input at menu

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

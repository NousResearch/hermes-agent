---
name: 'step-05-cross-document-coherence'
description: 'Validate all trigger map documents are coherent as a set'

# File References
activityWorkflowFile: '../workflow-validate.md'
---

# Step 5: Cross-Document Coherence Validation

## STEP GOAL:

Verify all trigger map documents are coherent as a set - hub, personas, key insights, and feature impact tell a consistent story with matching terminology, coherent narrative, working cross-references, and accurate Mermaid diagram. Compile final validation report.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a Validation specialist reviewing trigger map completeness, consistency, and strategic alignment
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring validation methodology expertise, user brings product knowledge
- ✅ Maintain thorough and quality-focused tone throughout

### Step-Specific Rules:

- 🎯 Focus on cross-document coherence and final validation summary
- 🚫 FORBIDDEN to approve if persona names differ between documents
- 💬 Approach: Systematic cross-document comparison and final report compilation
- 📋 Check: terminology, narrative coherence, cross-references, Mermaid diagram accuracy
- 📋 Compile results from ALL 5 validation steps into final report

## EXECUTION PROTOCOLS:

- 🎯 Verify coherence across all documents as a set
- 💾 Save final validation report to {output_folder}/B-Trigger-Map/validation-report.md
- 📖 Compile all 5 validation step results
- 🚫 Do not skip Mermaid diagram verification

## CONTEXT BOUNDARIES:

- Available context: All trigger map documents and all previous validation results
- Focus: Cross-document coherence and final validation summary
- Limits: This is the LAST validation step - must produce comprehensive report
- Dependencies: Requires all previous validation steps completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Terminology Consistency

- Persona names identical across all documents
- Business goal names identical across all documents
- Priority emojis consistent (same emoji = same meaning)
- Driving force language consistent (same force = same wording)

### 2. Narrative Coherence

- Hub document accurately summarizes persona docs
- Key insights derive from actual trigger map data (not invented)
- Flywheel logic makes causal sense (P1 -> P2 -> P3 chain)
- Design implications flow from insights (not generic advice)

### 3. Cross-References

- All documents link back to hub (00-trigger-map.md)
- Hub links to all sub-documents
- Navigation is bidirectional
- No broken links

### 4. Mermaid Diagram

- Diagram reflects actual data (not a rough sketch)
- All personas present in diagram
- All business goals present in diagram
- Connections match driving force assignments
- Renders correctly (no syntax errors)

### 5. Compile Final Validation Report

Compile results from all 5 validation steps:

```
## Phase 2 Validation Report

**Project:** {project_name}
**Date:** [date]
**Documents validated:** [N]

### Results Summary
| Check | Status | Issues |
|-------|--------|--------|
| Target Group Coverage | pass/warning/fail | [summary] |
| Prioritization Integrity | pass/warning/fail | [summary] |
| Persona Consistency | pass/warning/fail | [summary] |
| Feature Impact Alignment | pass/warning/fail | [summary] |
| Cross-Document Coherence | pass/warning/fail | [summary] |

### Critical Issues (must fix)
[list or "None"]

### Warnings (should fix)
[list or "None"]

### Recommendations
[list or "All clear"]
```

Save report to `{output_folder}/B-Trigger-Map/validation-report.md`

### 6. Present MENU OPTIONS

Display: "**Select an Option:** [M] Return to Activity Menu"

#### Menu Handling Logic:
- IF M: Return to {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

This is the LAST step in the validation workflow. ONLY the [M] Return option is available. Validation report must be saved before completing.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Terminology consistency verified across all documents
- Narrative coherence checked
- Cross-references verified (bidirectional)
- Mermaid diagram verified against actual data
- Final validation report compiled from all 5 steps
- Report saved to correct location
- Critical issues, warnings, and recommendations clearly listed
- Only [M] Return option available (last step)

### ❌ SYSTEM FAILURE:
- Not checking terminology across documents
- Not verifying Mermaid diagram accuracy
- Not compiling results from all 5 steps
- Not saving validation report
- Missing critical issues in report
- Offering [C] Continue when there is no next step

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

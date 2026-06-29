---
name: 'step-02-hand-off'
description: 'Hand off Design Delivery to BMad for implementation'

# File References
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow-deploy.md'
---

# Step 5: Hand Off to BMad

## STEP GOAL:

Hand off the Design Delivery (small scope) to BMad Developer for implementation - using simplified handoff for small updates or full dialog for larger ones.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are Freya, a product evolution specialist guiding continuous improvement
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring handoff process expertise, user brings design delivery
- ✅ Maintain clear and professional tone throughout

### Step-Specific Rules:

- 🎯 Focus only on clear handoff communication to BMad
- 🚫 FORBIDDEN to modify design or add new requirements
- 💬 Approach: Help user compose clear handoff message, ensure BMad has everything needed
- 📋 Choose appropriate handoff method based on effort estimate
- 📋 Update delivery status after handoff

## EXECUTION PROTOCOLS:

- 🎯 Guide user to choose handoff method (simplified vs full dialog)
- 💾 Help user compose handoff notification with all necessary information
- 📖 Update delivery status in DD file after handoff
- 🚫 Do not allow handoff without all artifacts ready

## CONTEXT BOUNDARIES:

- Available context: Completed step 04 (Design Delivery created), all artifacts ready, test scenario created
- Focus: Handoff communication, status update
- Limits: Do not modify design, do not add requirements, do not skip status update
- Dependencies: Requires completed step 04, DD file created, TS file created, all artifacts ready

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Determine Handoff Method

**Ask user about effort estimate:**

Review the effort estimate in DD-XXX file:
- **< 3 days total effort**: Use simplified handoff
- **> 3 days total effort**: Use full handoff dialog

Guide user to appropriate section below.

### 2. Simplified Handoff (< 3 Days)

**For small, focused updates:**

Help user compose handoff notification:

```
WDS Designer → BMad Developer

Subject: Design Delivery Ready: DD-XXX [Name]

Hi Developer!

Design Delivery ready for implementation.

📦 **Delivery:** DD-XXX [Name]
**Type:** Incremental Improvement
**Scope:** Update (small)
**Effort:** [X] days
**Priority:** [High | Medium | Low]

🎯 **Goal:**
[One sentence describing the improvement]

Example:
"Add inline onboarding to Feature X to increase usage from 15% to 60%."

📊 **Current Problem:**
- [Metric 1]: [Current value]
- [Metric 2]: [Current value]

📈 **Expected Impact:**
- [Metric 1]: [Current] → [Target]
- [Metric 2]: [Current] → [Target]

📁 **Artifacts:**
- Design Delivery: deliveries/DD-XXX-name.yaml
- Specifications: C-UX-Scenarios/XX-update/Frontend/specifications.md
- Before/After: C-UX-Scenarios/XX-update/before-after.md
- Components: D-Design-System/03-Atomic-Components/...
- Test Scenario: test-scenarios/TS-XXX.yaml

✅ **Acceptance Criteria:**
- AC-001: [Description]
- AC-002: [Description]
- AC-003: [Description]

⏱️ **Timeline:**
- Development: [X] days
- Target release: [Date]
- Measurement: 2 weeks after release

❓ **Questions:**
Let me know if you need clarification on anything!

Thanks,
[Your name]
WDS Designer
```

**Work with user to fill in all bracketed values from DD file.**

### 3. Full Handoff Dialog (> 3 Days)

**For larger updates:**

Explain to user:

"For larger updates (> 3 days effort), use full handoff dialog process from Phase 4 [H] Handover, Step 04."

**Key topics to cover in dialog:**
1. Problem and solution overview
2. What's changing vs staying
3. Technical requirements
4. Component specifications
5. Acceptance criteria
6. Success metrics
7. Rollback criteria

**Note:** This is less common for Product Evolution workflow - most improvements are small scope.

### 4. BMad Acknowledges

**Help user understand expected response:**

BMad Developer should respond with:

```
BMad Developer → WDS Designer

Subject: Re: Design Delivery Ready: DD-XXX

Received! Thank you.

📋 **My Plan:**
1. Review specifications ([date])
2. Implement changes ([date])
3. Run tests ([date])
4. Notify for validation ([date])

⏱️ **Estimated Completion:** [Date]

❓ **Questions:**
[Any clarification needed]

Thanks!
BMad Developer
```

**If user receives this acknowledgment, proceed to next step.**

**If BMad has questions, help user answer them clearly.**

### 5. Update Delivery Status

**Update the DD-XXX file:**

Help user modify the delivery status section:

```yaml
delivery:
  status: 'in_development' # Changed from "ready_for_handoff"
  handed_off_at: '[timestamp]'
  developer: '[BMad Developer name]'
  development_start: '[timestamp or estimate]'
  expected_completion: '[timestamp or estimate]'
```

**Verify:** Status updated correctly in DD file.

### 6. Present MENU OPTIONS

Display: "**Select an Option:** [M] Return to Activity Menu (suggest [T] Acceptance Test)"

#### Menu Handling Logic:
- IF M: Return to {workflowFile} or {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN user selects [M] and handoff is complete will you then return to the activity workflow to suggest next step [T] Acceptance Test (after BMad completes implementation).

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Handoff notification composed with all required information
- Appropriate handoff method chosen (simplified vs full dialog)
- All artifacts referenced in handoff message
- Goal, problem, expected impact clearly stated
- Acceptance criteria included in notification
- Timeline and effort estimate communicated
- BMad Developer acknowledged receipt
- Questions from BMad answered clearly (if any)
- Delivery status updated to 'in_development'
- handed_off_at timestamp recorded
- Developer name and expected completion date recorded
- User available for clarification questions during development

### ❌ SYSTEM FAILURE:
- Handoff without all artifacts ready
- Vague or incomplete handoff message
- Missing acceptance criteria or metrics
- No timeline or effort estimate
- Delivery status not updated after handoff
- Not responding to BMad's questions
- Adding new requirements during handoff (scope creep)
- Modifying design after handoff without updating DD file
- Generating handoff message without user input
- Not recording developer name or timeline

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

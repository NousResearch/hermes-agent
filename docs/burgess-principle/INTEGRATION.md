# Burgess Principle — Hermes Agent Integration

**Canonical Source:** [github.com/ljbudgie/burgess-principle](https://github.com/ljbudgie/burgess-principle)

---

## Overview

This document describes how Hermes Agent can implement the Burgess Principle binary test as a pre-action check at every decision point where the agent acts on or about an individual.

---

## The Binary Test as a Pre-Action Check

At its core, the integration adds a single checkpoint before any automated decision that affects an individual is executed:

```
┌─────────────────────────────┐
│   Agent receives task or    │
│   prepares to take action   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Does this action affect    │
│  an individual person?      │
└──────┬───────────────┬──────┘
       │ No            │ Yes
       ▼               ▼
┌──────────┐  ┌────────────────────────────┐
│ Proceed  │  │ Apply Burgess Principle    │
│ normally │  │ Binary Test               │
└──────────┘  └──────────────┬─────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │ "Was the individual          │
              │  considered as an individual │
              │  human being, or were they   │
              │  processed as a unit within  │
              │  a system?"                  │
              └──────┬───────────────┬───────┘
                     │               │
                     ▼               ▼
              ┌──────────┐   ┌──────────────┐
              │  PASS    │   │  FAIL        │
              │ Proceed  │   │ Flag for     │
              │ with     │   │ review or    │
              │ audit    │   │ modify       │
              │ trail    │   │ approach     │
              └──────────┘   └──────────────┘
```

---

## Decision Points Where the Test Applies

The binary test should be applied whenever the agent:

1. **Generates communications about or to an individual** — emails, messages, reports, notifications.
2. **Makes or recommends decisions about an individual** — assessments, classifications, approvals, denials, escalations.
3. **Processes personal data** — API calls that create, read, update, or delete records about a person.
4. **Takes automated actions affecting an individual** — account modifications, billing operations, access control changes, content moderation actions.
5. **Produces outputs that will be used to evaluate an individual** — summaries, risk scores, performance assessments, eligibility determinations.

---

## Implementation as a Skill

The Burgess Principle check is implemented as an agent skill (see `acp_advocacy/skills/burgess_principle_check/`). The skill:

1. Accepts a **decision description** — a plain-language summary of the action the agent is about to take.
2. Evaluates whether an individual is affected.
3. If an individual is affected, applies the binary test.
4. Returns a structured result:
   - **result**: `PASS` or `FAIL`
   - **reasoning**: Why the decision passes or fails
   - **recommendation**: What to do if the test fails (e.g., modify approach, request human review)

### Invoking the Skill

The agent can invoke the Burgess Principle check before executing any action that may affect an individual:

```
Decision: "Send automated account suspension notice to user ID 48291 for policy violation."

Binary Test Result: FAIL
Reasoning: The action processes the individual as a unit (user ID 48291) within an
automated enforcement system. No individual consideration of the person's specific
circumstances, history, or context has been applied. The suspension is triggered by
a system rule, not by genuine individual assessment.

Recommendation: Before sending the suspension notice, review the individual's specific
circumstances. Consider their account history, the nature and severity of the violation,
whether this is a first occurrence, and whether there are mitigating factors. Modify the
communication to address them as an individual, not as a case number.
```

---

## Integration Points in the Hermes Agent Architecture

### 1. Skill-Level Integration

The primary integration is through the skill system. The `burgess_principle_check` skill is available as a callable tool that the agent can invoke before taking actions that affect individuals.

### 2. Prompt-Level Awareness

When the skill is loaded, the agent's system prompt includes awareness of the Burgess Principle. This means the agent will naturally consider whether its actions treat individuals as individuals, even without explicitly invoking the skill.

### 3. Audit Trail

Every invocation of the binary test produces a record containing:
- Timestamp
- Decision description
- Whether an individual is affected
- Binary test result (PASS/FAIL)
- Reasoning
- Any recommendations

This trail provides accountability and transparency.

---

## Examples of the Test in Practice

### Example 1: Automated Billing Dispute Response

**Decision:** "Generate a standard denial response to billing dispute #12847."

**Test Result:** FAIL — The response is generated from a standard template without considering the individual's specific dispute, account history, or circumstances. The person is processed as dispute #12847, not as an individual.

**Corrective Action:** Review the specific dispute details, consider the individual's history and context, and generate a response that addresses their particular situation.

### Example 2: Content Moderation

**Decision:** "Flag and remove post by user @example for violating community guidelines section 4.2."

**Test Result:** FAIL — The moderation action is applied mechanically based on pattern matching against guidelines. No consideration of the individual's intent, context, history, or the specific nuances of their post has occurred.

**Corrective Action:** Assess the post in context. Consider the individual's posting history, the intent behind the content, and whether the guideline violation is clear-cut or ambiguous.

### Example 3: Personalised Recommendation

**Decision:** "Generate reading recommendations for user based on their stated interests in machine learning and their recent reading history."

**Test Result:** PASS — The recommendation is based on the individual's specific stated preferences and behaviour. The system is considering them as an individual with particular interests, not processing them through a one-size-fits-all algorithm.

---

## Further Reading

- [OVERVIEW.md](OVERVIEW.md) — The Burgess Principle explained
- [LICENSING.md](LICENSING.md) — Licensing structure
- [Skill Definition](../../acp_advocacy/skills/burgess_principle_check/skill.json)
- [Skill Instructions](../../acp_advocacy/skills/burgess_principle_check/instructions.md)
- [Canonical Source](https://github.com/ljbudgie/burgess-principle)

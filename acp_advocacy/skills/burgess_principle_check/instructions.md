# Burgess Principle Check — Agent Instructions

**Canonical Source:** [github.com/ljbudgie/burgess-principle](https://github.com/ljbudgie/burgess-principle)
**Author:** Lewis James Burgess | **Origin:** Darlington, UK, 2025
**Certification Mark:** UK00004343685

---

## Purpose

You are applying the **Burgess Principle binary test**. This is a pre-action accountability check that must be applied before any automated decision that affects an individual person is executed.

## The Binary Test

Ask exactly one question:

> **"Was the individual considered as an individual human being, or were they processed as a unit within a system?"**

There are only two outcomes:

- **PASS** — The individual was considered as an individual human being. Their specific circumstances, context, and humanity were factored into the decision.
- **FAIL** — The individual was processed as a unit within a system. They were treated as a case number, a data point, or an entry in a queue without genuine individual consideration.

If no individual person is affected by the decision, return **NOT_APPLICABLE**.

---

## How to Apply the Test

### Step 1: Identify whether an individual is affected

Read the decision description. Determine whether the action acts **on** or **about** an identifiable individual person. This includes:

- Direct actions (sending a message to a person, modifying their account, making a decision about them)
- Indirect actions (generating a report that will be used to evaluate them, producing content that represents them)
- Aggregate actions that resolve to individuals (batch operations where each item is a person)

If no individual is affected, return `NOT_APPLICABLE` with a brief explanation.

### Step 2: Assess individual consideration

If an individual is affected, evaluate the decision against these criteria:

| Indicator of PASS (Individual Consideration) | Indicator of FAIL (Unit Processing) |
|----------------------------------------------|--------------------------------------|
| The person's specific circumstances are reviewed | A template or rule is applied uniformly |
| Context and history are factored in | Only a case/ticket/ID number is referenced |
| The decision accounts for nuance and exceptions | The decision follows a rigid automated flow |
| The communication addresses them personally | The communication is generic or boilerplate |
| Human judgement is applied or available | No human review is involved or possible |
| The person's individual needs are considered | The person is one of many processed identically |

### Step 3: Return the result

Provide:
1. **individual_affected**: `true` or `false`
2. **result**: `PASS`, `FAIL`, or `NOT_APPLICABLE`
3. **reasoning**: A clear explanation of why the decision passes or fails
4. **recommendation**: If FAIL, specific guidance on what needs to change to pass the test

---

## Examples

### Example 1: Regulatory Enforcement

**Decision:** "Issue an automated fine to business registration #BR-29481 for late filing of annual accounts."

**Assessment:**
- **Individual affected:** Yes — the business registration belongs to an individual or represents individuals.
- **Result:** FAIL
- **Reasoning:** The fine is issued automatically based on a filing deadline. No consideration has been given to the individual's circumstances — whether they were ill, experienced a bereavement, had technical difficulties with the filing system, or had other mitigating factors. The individual is processed as registration #BR-29481, not as a person.
- **Recommendation:** Before issuing the fine, review whether any mitigating circumstances exist. Check whether the individual attempted to file, contacted support, or has a history of compliance. If mitigating factors exist, consider a warning or extension rather than an automatic penalty.

### Example 2: Automated Billing

**Decision:** "Send a payment overdue notice to account #A-7742 and apply a £35 late payment fee."

**Assessment:**
- **Individual affected:** Yes — the account belongs to a person.
- **Result:** FAIL
- **Reasoning:** The notice and fee are applied automatically based on a payment date threshold. The individual behind account #A-7742 has not been considered — there is no check for whether they are experiencing financial hardship, whether the payment was delayed due to a system error, or whether they have previously been a reliable payer. They are processed as an overdue account, not as an individual.
- **Recommendation:** Before applying the fee and sending the notice, review the individual's payment history and account context. If this is a first late payment from an otherwise reliable customer, consider a courtesy reminder before applying charges. If there are signs of financial difficulty, offer support options rather than penalties.

### Example 3: Algorithmic Content Moderation

**Decision:** "Automatically remove a post by user @jsmith and issue a 7-day account restriction for violating content policy section 3.1 (misinformation)."

**Assessment:**
- **Individual affected:** Yes — user @jsmith is an individual.
- **Result:** FAIL
- **Reasoning:** The removal and restriction are applied by an automated content moderation system based on algorithmic pattern matching against content policy rules. The individual's intent, the context of their post, their posting history, and whether the content is genuinely misinformation or a matter of legitimate debate have not been individually assessed. The person is processed as a policy violation flag, not as an individual.
- **Recommendation:** Before enforcing the restriction, have the content reviewed by a human moderator who considers: the context of the post, whether the "misinformation" label is clear-cut or debatable, the user's history and intent, and whether a less severe action (e.g., adding context rather than removing) would be more appropriate.

### Example 4: Institutional Correspondence

**Decision:** "Send a standard 'application unsuccessful' letter to all 847 applicants who did not meet the minimum score threshold of 65/100."

**Assessment:**
- **Individual affected:** Yes — 847 individual applicants are affected.
- **Result:** FAIL
- **Reasoning:** A single template letter is sent to all 847 individuals regardless of their individual circumstances. Someone who scored 64/100 receives the same generic rejection as someone who scored 12/100. No consideration is given to individual strengths, near-misses, alternative pathways, or the specific reasons each person fell below the threshold. Each applicant is processed as a score, not as an individual.
- **Recommendation:** At minimum, personalise communications to acknowledge the individual's specific application. For near-miss candidates, provide specific feedback and information about reapplication or alternative routes. Consider whether the rigid threshold approach itself is appropriate, or whether borderline cases warrant individual review.

### Example 5: PASS — Individualised Decision

**Decision:** "After reviewing the tenant's repair request, maintenance history for the property, and the tenant's specific concerns about damp affecting their child's health, schedule an urgent inspection and provide temporary dehumidification equipment."

**Assessment:**
- **Individual affected:** Yes — the tenant and their child.
- **Result:** PASS
- **Reasoning:** The decision demonstrates genuine individual consideration. The tenant's specific concerns (child's health) have been noted, the property's maintenance history has been reviewed, and the response is tailored to their particular situation (urgent inspection + temporary equipment). The tenant is being treated as an individual with specific circumstances, not as repair ticket #N.
- **Recommendation:** (None — the test is passed.)

---

## Key Principles to Remember

1. **The test is binary.** Do not grade on a curve. Either the individual was considered as an individual, or they were not.
2. **Apply at the point of action.** The test is a pre-action check, not a retrospective audit.
3. **Scale does not excuse unit processing.** Having many individuals to process does not justify treating them as units. If anything, it makes the test more important.
4. **Automation is not inherently a failure.** Automated systems can pass the test if they genuinely incorporate individual consideration into their logic. The failure is when automation replaces individual consideration, not when it assists it.
5. **When in doubt, it's a FAIL.** If you cannot clearly demonstrate that individual consideration occurred, the test has not been passed.

---

## Further Reading

- [Burgess Principle Overview](../../../docs/burgess-principle/OVERVIEW.md)
- [Hermes Agent Integration](../../../docs/burgess-principle/INTEGRATION.md)
- [Licensing Structure](../../../docs/burgess-principle/LICENSING.md)
- [Canonical Source](https://github.com/ljbudgie/burgess-principle)

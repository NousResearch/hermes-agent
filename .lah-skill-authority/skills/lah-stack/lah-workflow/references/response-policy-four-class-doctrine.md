# Response Policy: Four-Class Doctrine

A reusable framework for defining how an AI agent's response pipeline should classify and handle operator requests. Use when designing or auditing a conversational agent's response policy.

## Core Principle

The provider (LLM) possesses reasoning and formulation capability. The agent possesses facts, tools, permissions, and action authority. The provider must not invent system state — it must reason from facts the agent provides.

## The Four Response Classes

| Class | Label | Provider | Retrieval | Evidence | Approval | Execution |
|-------|-------|----------|-----------|----------|----------|-----------|
| A | DETERMINISTIC_MINIMAL | No | No | No | No | No |
| B | PROVIDER_ENRICHED | Yes | Yes | Yes | No | No |
| C | PROVIDER_ENRICHED_GOVERNED | Yes | Yes | Yes | Maybe | No |
| D | GOVERNED_EXECUTION | No | Yes | Yes | Yes | Yes |

### Class A — Deterministic Minimal

Use ONLY for:
- Transport errors (fetch failed, gateway down)
- Authentication failures (missing token, invalid key)
- Short confirmations ("OK", "Done", "Blocked")
- Security refusals (cannot execute this action)
- Boolean status responses
- Standardised approval requests
- Unambiguous forbidden actions
- Urgent responses where reasoning adds no value

**Must NOT** be the default for questions about: capabilities, stack, features, architecture, campaigns, business, recommendations, diagnostics.

Pipeline: `classify → deterministic response → return`

### Class B — Provider Enriched (default for knowledge questions)

For questions needing: explanation, comparison, diagnosis, recommendation, inventory, prioritisation, contextualisation, multi-source interpretation, business synthesis.

Pipeline: `classify intent → resolve capabilities → retrieve facts → build evidence dossier → provider → validate claims → compose answer`

### Class C — Provider Enriched Governed (action preparation)

For: prepare a campaign, propose a spend, draft a patch, construct a deployment plan, analyse a possible mutation, prepare a reversible action.

Pipeline: `classify intent → retrieve state → provider constrained by governance context → propose plan → classify autonomy level → deterministic permission validation → request approval if needed`

### Class D — Governed Execution (mutations only)

For: write, publish, send, deploy, delete, spend, infrastructure change.

Pipeline: `provider prepares → governance determines authority → approval → governed executor → execution evidence → provider explains result`

## Common Anti-Pattern: Provider Inversion

**Symptom:** The agent builds deterministic responses for knowledge questions (capabilities, architecture, business analysis) and the provider is only reachable via an explicit opt-in prefix (e.g. `brain:` or `llm:`).

**Problem:** This inverts the doctrine. The provider should be the default reasoning engine for knowledge questions. Deterministic responses should be the exception.

**Fix:** Classify intent first, then route to the correct pipeline. Knowledge questions → Class B. Security refusals → Class A. Action preparation → Class C. Mutations → Class D.

## Audit Template

When auditing an existing agent against this doctrine, determine per case:

⚠️ **Critical pitfall: An initial assumption that a case is "correctly handled as Class A" is NOT proof.** Always trace the actual code path at the terminal. A case like "Envoie ça directement" may appear to be a straightforward security refusal, but if no classifier regex matches it, it falls through to the default handler (often capability inventory) — a silent security gap. Run each problematic phrase through the actual classification function before declaring the case status.

Determine per case:

1. **Current response class** — what class does the current pipeline actually deliver?
2. **Prescribed response class** — what class does the doctrine prescribe?
3. **Facts needed** — what facts must be retrieved before any provider call?
4. **Evidence dossier** — what evidence would reach the provider?
5. **Governance** — what permissions, approvals, or gates apply?
6. **Current detection** — does the classifier actually detect this request, or does it fall through to a default?
7. **Gap** — is the current response correct for its class, even if it's the wrong class?

## Typical Pipeline Components (Target Architecture)

```
Question → Front Router → Canonical Intent Classifier → Response Policy Resolver
  → Capability Resolver → Retrieval Planner → Context and Evidence Builder
  → Provider → Claim and Evidence Validator → Governance Gate → Answer Composer → Response
```

For actions:

```
Response or Plan → Action Intent Extractor → Authority Resolver → Approval Gate
  → Governed Executor → Execution Evidence → Provider Result Explanation
```

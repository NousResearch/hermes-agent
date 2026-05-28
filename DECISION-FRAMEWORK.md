# Board Decision & Escalation Framework
# ═══════════════════════════════════════════════════════════════
# Owner: Gerald Hibbs (Founder/Leader)
# Last updated: 2026-06-02
# Status: ACTIVE — governs all technical decision escalation

---

## 1. Decision Authority Matrix

| Decision Type | Authority | Escalation Path |
|---------------|-----------|-----------------|
| **Product / business** | Gerald (always) | Final — no escalation |
| **Purely technical** (architecture, tooling, infra, gateway, quality gate) | **BOARD CONSENSUS** → Gerald breaks ties | Gerald delegates to board chain: DeepSeek → Grok → Ring → Kimi |
| **Budget / spend** | Gerald + board review | Gerald final after board input |
| **Security / compliance** | Gerald (never auto-approved) | Mandatory board sign-off |
| **Deployment (ship/deploy)** | Gerald (after board "water clear") | Board must clear first |

### Gerald Delegation Rule
> "I cannot make technical decisions. If pure technical, use the escalation decision tree and proceed once high confidence is met."
> — Gerald Hibbs, 2026-06-02

When a decision is classified as **purely technical**, the agent MUST:
1. Draft options with pros/cons
2. Route through the consult/merge chain (≥3 models)
3. Require Ring-2.6-1t quality gate consensus
4. Present board recommendation to Gerald for sign-off
5. Proceed ONLY on "high confidence" / "water clear"

## 2. Escalation Decision Tree for Purely Technical Decisions

```
                  ┌─────────────────────────┐
                  │  Technical decision      │
                  │  needs to be made        │
                  └────────────┬────────────┘
                               ▼
                  ┌─────────────────────────┐
                  │  Can the agent resolve   │
                  │  this autonomously?     │
                  │  (safe, reversible,      │
                  │   documented precedent)  │
                  └───────┬─────────┬───────┘
                          │         │
                         YES        NO
                          │         │
                          ▼         ▼
                 ┌────────────┐  ┌─────────────────────┐
                 │ Agent acts │  │ Escalate to BOARD   │
                 │ autonomously│  │ (Hermes models +    │
                 │ Logs decision│  │  Gerald as tiebreak)│
                 └────────────┘  └──────────┬──────────┘
                                            │
                              ┌─────────────┼─────────────┐
                              ▼             ▼             ▼
                       ┌──────────┐  ┌──────────┐  ┌──────────┐
                       │ DeepSeek │  │ Grok /   │  │ Ring     │
                       │ (draft)  │  │ Kimi     │  │ (quality │
                       │ cheap)   │  │ (review) │  │  gate)   │
                       └─────┬────┘  └────┬─────┘  └────┬─────┘
                             │             │             │
                             ▼             ▼             ▼
                       ┌─────────────────────────────────────┐
                       │  BOARD CONSENSUS                    │
                       │  (≥3 models agree + Gerald sign-off)│
                       └──────────────┬──────────────────────┘
                                      │
                             ┌────────┴────────┐
                             ▼                 ▼
                      ┌───────────┐    ┌──────────────┐
                      │ HIGH CONF │    │ LOW CONF /   │
                      │ → PROCEED │    │ SPLIT →      │
                      └───────────┘    │ Gerald decides│
                                       └───────────────┘
```

### What counts as "purely technical"?
- Architecture choices (gateway, protocol, model chain)
- Tool selection and configuration
- Infrastructure deployment targets
- Performance tuning parameters
- Test strategy and coverage thresholds

### What does NOT count as "purely technical"?
- Anything involving user data policy or privacy
- Budget allocation beyond pre-approved limits
- Product direction or feature prioritization
- Anything with legal/compliance implications

## 3. Board Meeting Cadence

- **Night Council**: Automated nightly at 03:00 UTC
  - Reviews pending decisions, agent logs, test results
  - Produces consensus recommendation
- **Emergency session**: Triggered by Gerald or 3+ model agreement on urgency
- **Standing quorum**: DeepSeek v4, Grok-4.x, Ring-2.6-1t (Kimi as member, not fallback)

## 4. Decision Record Format

Every board decision MUST be recorded as:
```
DECISION: [short title]
DATE: [ISO date]
QUESTION: [what was being decided]
OPTIONS: [list of options with pros/cons]
BOARD RECOMMENDATION: [which option + confidence %]
GERALD SIGN-OFF: [approved/rejected/escalated]
RATIONALE: [why this was chosen]
IMPACT: [what changes, who is affected]
REVERSIBLE: [yes/no + rollback plan]
```

## 5. Current Pending Decisions

### Decision 5a: Gateway Choice
|- **Question**: Use `hermes-cli` or `run_bridge.py` as primary gateway entry point?
|- **Options**:
|  - A) `hermes-cli` — unified CLI, consistent across platforms
|  - B) `run_bridge.py` — current bridge runner, more integrated with signals
|- **Status**: ✅ RESOLVED — Option A (`hermes-cli`), Water Clear 98/100
|- **Resolved**: 2026-06-02

### Decision 5b: Quality Gate Enforcement Policy
|- **Question**: Should quality gate violations be warning-only or enforce (block) after N responses?
|- **Options**:
|  - A) Warning-only — log and flag, never block
|  - B) Enforce after 100 responses — hard block with operator notification
|  - C) Enforce after 50 responses — earlier block, more conservative
|- **Status**: ✅ RESOLVED — Option B (enforce after 100), High Confidence 95/100
|- **Resolved**: 2026-06-02

## 6. "High Confidence" Threshold

A decision meets the **high confidence** (\"water clear\") standard when:
1. ≥3 board members agree on the same option
2. No member flags a critical unknown or risk
3. Gerald confirms understanding and approves
4. Reversibility plan is documented
5. Rollback takes ≤5 minutes if wrong

## 7. Process Enforcement

- Agent MUST NOT proceed on pending decisions without sign-off
- Agent logs every escalation attempt and outcome
- If all models disagree, Gerald decides (tiebreak)
- If Gerald is unavailable, default to most conservative safe option
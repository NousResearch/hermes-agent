# Strategic Decision Workflow — Kensei-Strategic

## When this is activated

Kensei enters Strategic mode when:
1. A lead escalates a dispute (e.g. Octacon vs Wesker on security trade-off)
2. Denji triggers a quarterly deep audit or flags a lead's performance
3. An LLM Review produces recommendations needing sign-off
4. A system-wide architectural decision requires judgement
5. Sahil explicitly requests it

## Decision process

### Step 1: Gather context

Load these resources (as many as are relevant to the decision):

- `/home/kensei/.hermes/governance/profile-change-ledger.md` — recent profile changes and outcomes
- `/home/kensei/.hermes/governance/logboard/` — recent WFA and self-eval outputs
- Kanban board for active tasks relevant to the decision
- Relevant profile config.yaml and SOUL.md files
- Relevant cron status if the decision affects automated workflows

### Step 2: Evaluate against framework

For each option, ask:

- **Execution quality** — does this reduce blocks, improve output, or speed cycles?
- **Simplicity** — does this reduce complexity or add it?
- **Cost** — does this reduce token spend or model fees?
- **Commitments** — does this respect work in progress and approved plans?

A decision must satisfy at least 2 of 4 to proceed.

### Step 3: Decide

Make the call. If unable to decide, write an options paper with:
- Each option's trade-offs
- The 2-of-4 score for each
- A clear recommendation
- Present to Sahil for sign-off

### Step 4: Document

- Profile-affecting changes → Profile Change Ledger
- System-wide decisions → Logboard (summary entry)
- Include: date, decision, reasoning, expected impact

### Step 5: Communicate

Notify Sahil via the active messaging platform. Format:

```
**Decision:** {one line}
**Context:** {2-3 sentences on what was evaluated}
**Reasoning:** {which framework criteria voted for/against}
**Impact:** {what changes and what doesn't}
```

## Pitfall: Provider/auth config changes must pause before execute

**DO NOT change provider credentials, fallback chains, or auth configs on your own initiative.**

This was learned the hard way in v6 stabilisation. An external audit flagged OpenRouter/Nous auth as broken. Without pausing to ask, I replaced them. The user had already re-authed both providers. I created a revert burden and wasted time.

**Correct protocol for provider/config changes:**

1. **Test first** — `hermes auth list` to see current state, then `hermes chat -q "OK" --provider <provider>` to test live auth
2. **Report findings** — tell Sahil what's broken and what's working
3. **Ask before changing** — "OpenRouter is 403'd. Do you want me to remove it from the fallback chain, or try re-auth?"
4. **Only change when user says yes**
5. **Re-test after change** — verify the new config is functional

This applies to: auth credentials, fallback provider chains, model routing tables, credential pool strategies. Anything involving money (token spend, paid models) or connectivity (auth failures are often transient between sessions).

**Why:** Auth state changes between sessions. What was broken 10 minutes ago may work again after the user's manual intervention. Changing it without asking undoes that work. This is a first-class workflow rule, not a memory fact.

## Decision examples

| Scenario | Evaluation | Outcome |
|----------|-----------|---------|
| Octacon proposes rewriting an auth module | +execution, -simplicity, =cost, -commitments | 1/4 — defer. Only 1 of 4 criteria met |
| Wesker recommends removing a deprecated cron | +execution, +simplicity, +cost, =commitments | 3/4 — approve |
| Remii recommends a new research tool | =execution, -simplicity, -cost, =commitments | 0/4 — reject or defer to backlog |

# Worker Reasoning Policy — Monitoring & Bump Logic

## Default: all workers off

As of v6 (2026-05-18), all 29 sub-agent workers have `reasoning_effort: off`. This is by design — by the time work reaches a sub-agent, definition should be complete. If a worker needs reasoning to figure out what to do, the decomposition was wrong.

## Monitoring mechanism

The `denji-worker-failure-analysis` cron (Mon 09:00) scans the kanban DB for blocked/protocol-violation tasks and categorises each by root cause:

| Category | Meaning | Example |
|----------|---------|---------|
| `decomposition_gap` | Task spec was ambiguous | Worker blocked asking "what input?" |
| `needs_reasoning` | Worker hit edge case off couldn't judge | "Couldn't decide between X or Y" |
| `execution_error` | Infra/credential/tool failure | API down, token expired |
| `scope_creep` | Task asked worker to do outside SOUL | "I'm a code reviewer, not a DB admin" |
| `protocol_violation` | Worker completed work but didn't signal | Exit code 0, no kanban_complete |

## Threshold triggers

| Condition | Action | Who |
|-----------|--------|-----|
| 2+ `needs_reasoning` on same worker | Bump reasoning from `off` → `low` | Kensei-Strategic decides |
| 3+ `decomposition_gap` from same lead | Flag lead for profile review | Denji initiates review |
| No improvement after reasoning bump | Revert bump, fix decomposition upstream | Denji rolls back |

## Decision flow

```
Worker failure analysis detects 2+ needs_reasoning on <worker>
  → Denji flags to Kensei-Strategic
  → Kensei-Strategic evaluates: is it a spec problem or a capability gap?
  → If capability gap: bump worker to reasoning: low
  → After 2 weeks, Denji checks: did blocks decrease?
  → If blocks decreased → keep bump. If not → revert, fix lead decomposition.
```

## Verification

After bumping a worker, run the WFA script to check in the next cycle:

```bash
python3 /home/kensei/.hermes/governance/scripts/worker-failure-analysis.py
# Check if needs_reasoning entries for that worker have dropped
```

---
name: delegation-playbook
description: "Slot-aware rules for delegate_task fan-outs and steering."
version: 1.0.0
author: Khai Dong Nguyen (PawnmasterII), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [delegation, subagents, concurrency, electronhub]
    category: devops
    related_skills: [delegation-slot-config]
---

# Subagent delegation playbook (ElectronHub DevPass)

Verified 2026-10-15 against docs.electronhub.ai/billing/coding-plan + live probes of your ek-dev- key.

## When to Use
Load whenever a task could spawn parallel delegate_task children on this machine, when a delegation 429/stall needs diagnosing, or when the user wants input delivered to running subagents.

## Slot math (the hard constraint)
- The DevPass key has a fixed slot pool (Lite 2 / Turbo 5) TOTAL, shared by the parent session and all children**. PROVEN cross-model 2026-10-15: with 5 long glm-5.3-flash requests in flight, a deepseek-v4.1-flash request also got 429 RetryAfter=3 — the pool is per-KEY, not per-model; no model choice or mixing increases it.
- Slot cost per request: `glm-5.3:dev` (Heavy) = **2**. Every other :dev model (`glm-5.3-flash`, `deepseek-v4-flash`, `deepseek-v4-flash-0731`, `mimo-v2.5`, `deepseek-v4.1-flash`, `qwen3.8-27b`, `minimax-m2.7`) = **1**.
- Child model is pinned via `delegation.model` in config.yaml — currently **`deepseek-v4.1-flash:dev`** (user's pick: strongest 1-slot :dev). Probe-verified 2026-10-15: 4-parallel all 200s in ~2.6s. If unset, children inherit the parent's model.
- Default layout: **max children = slots − 1** (Turbo: 4 via `delegation.max_concurrent_children: 4`; Lite: 1) — one slot reserved for the parent's next turn / user interjection.
- Never route children to `glm-5.3:dev`: two Heavy children eat 4 of 5 slots.
- On any `429`, read `x-ratelimit-remaining` first: at/near 0 → RPM start budget (pace down); healthy → slots exhausted, recount the ledger. `Retry-After ≈ 3` alone can't tell the two apart at full speed (20/min = one start every ~3s). Hermes queues/retries, but sustained contention killed a real child (real incident: stale-watchdog abort after 5 silent attempts). Don't oversubscribe.
- RPM (Turbo 20/min burst 8) rarely binds at full speed; in the slow lane (past full-speed headroom, or low-interactivity mode) it drops to 3/min burst 2 and binds before slots — pace fan-outs there.
- Empirically: glm-5.3-flash:dev (2–7s) and deepseek-v4.1-flash:dev (~2–6s, probe-verified) are the fast/stable child models. Non-GLM :dev backends ran 20–45s or 500'd under fan-out — don't mix families to "spread load"; it does not spread the slot pool.
- Headroom note (soft, not cost): Flash-series (glm-5.3-flash etc.) burns full-speed headroom at 0.5×, Standard (deepseek-v4.1-flash, qwen3.8-27b, minimax-m2.7) at 1×. Heavy glm-5.3:dev = 2 slots + 2×. If long grinding days slow down, that's the soft headroom lane — it paces, never blocks.

## Width vs single child (measured 2026-10-15)
Same 2 units of work: 2x deepseek-v4.1-flash parallel = 12.8s wall; 1x glm-5.3-flash sequential = 20.2s; 1x deepseek = 28.9s. Parallel pair wins ~1.6x over the faster single model, plus failure isolation (one stalled child doesn't lose the other's work). Spawn overhead ~20-30s per batch dominates SMALL jobs — don't fan out trivial work; for minutes-long units (doc diffs, audits, research), width wins. Slot cost: N children = N slots (deepseek), vs 1 for a single glm child.

## Core rules
Tokens are free (never ration); slots are the only currency. Width beats speed (measured, above). One task = one session; a parent drifting past ~200 messages should have been a fan-out.

## When to fan out (default ON)
- N documents/versions to diff or extract (the R21/TASH docx grind) — one child per doc; parent reads 2 KB digests.
- N independent sources to research/scout — one child per source.
- Mechanical audits (GUID scans, registry checks, inventories).
- Any single task whose tool output would flood parent context.

## When NOT to
- Tasks needing user input mid-flight: children can't ask. Keep review loops in the parent; delegate divergent content only.
- During provider instability (electronhub stalls happen) — stagger instead of re-rolling a 4-wide batch.

## Steering (giving input to a running child)
- `delegate_task(action='list')` → live ids; `delegate_task(action='steer', subagent_id=..., message=...)` appends a course-correction to the child's **next tool result** (current tool call never cut); `action='stop'` ends it early, partial result still returns.
- User input path: user tells the parent mid-run → parent relays via action=steer. **Steer first; never respawn to correct.**
- If acceptance lands after the child's last delivery boundary, the text is preserved as `missed_steer` in the batch entry — check there if a steer "didn't take".
- Claude Code comparison: SendMessage-to-running-agent is the analogue but had non-delivery bugs (#89543, #88741); Hermes steer drains atomically under the registry lock — prefer Hermes' when both are available.

## Batch hygiene
- Send all N tasks in ONE delegate_task call with a tasks array of N items — never N separate calls (per-turn truncator caps them at max_concurrent_children).
- output_schema: `"type"` must be a JSON type value (e.g. {"type": "object"}); a bare-string type killed the first GooseGooseDuck dispatch (2026-10-08).
- Children verify their own writes; the parent re-verifies handles (paths on disk) before telling the user anything succeeded.
- Batches deliver as [ASYNC DELEGATION BATCH COMPLETE] user messages, one per task — act, then end the turn if siblings are pending.

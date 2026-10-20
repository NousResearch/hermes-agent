---
name: loop-bounce
description: Detect agent loops and escalate to stronger models.
version: 2.0.0
author: Hermes Agent
license: MIT
platforms: [all]
metadata:
  hermes:
    tags: [hermes, fallback, loop-detection, model-escalation, delegation]
    related_skills: [hermes-agent]
---

# Loop Bounce — Break, Escalate, Report

A deterministic 2-step protocol for when the agent is running in circles. Step 1 breaks the loop. Step 2 escalates through a bounded ladder. If escalation also fails, report to the user.

---

## Step 1: Loop Detection (BREAK)

### Failure Classification (multi-dimensional)

Not all failures are equal. Classify each failure before counting:

| Class | Signal | Count toward threshold? |
|-------|--------|------------------------|
| **execution** | Non-zero exit code, error output | Yes |
| **diagnostic** | Command succeeded but didn't answer the question | Yes |
| **verification** | Fix applied but postcondition false | Yes (high weight) |
| **repeated_approach** | Same normalized fingerprint tried again | Yes (immediate escalate) |
| **environment** | Missing creds, permissions, network | Report, don't retry |
| **policy** | Safety refusal, approval required | STOP, don't escalate |

### Approach Fingerprinting

Normalize equivalent commands to detect disguised retries:

```
docker compose restart api
docker-compose restart api
ssh host docker compose restart api
→ fingerprint: terminal:docker compose restart api
```

Track fingerprints in a sliding window, not just consecutive count. A `pwd` success between two failures does NOT reset the counter.

### Detection Signals

**Hard limits** (whichever comes first):

| Limit | Value | Action |
|-------|-------|--------|
| Same fingerprint repeated | **3×** | Enter Step 2 |
| Sliding window failure rate ≥60% (last 10 calls) | — | Enter Step 2 |
| Verification failure after mutation | **1×** | Enter Step 2 |
| Time elapsed since first failure on current task | **10 minutes** | Enter Step 2 |

**Soft signals** (trigger early):

| Signal | Threshold |
|--------|-----------|
| Same error message repeated | 3× |
| "Let me try again" / "Let me retry" on same task | 3× |
| Alternating A-B-A-B pattern | 2 cycles |
| Reasoning trace shows identical thought cycling | 2× |

### What does NOT trigger

- First failure on a new task (recover on next turn)
- HTTP 429 rate limits or transient network errors (retry with backoff)
- Prompt-level misunderstandings (fix the prompt instead)
- Long-running legitimate tasks (e.g., a 10-minute build that is progressing)

### When triggered: STOP immediately

Do NOT make another tool call on the same task. Record:

1. What the task was
2. What was tried (1-2 sentence summary)
3. The specific error or failure mode
4. How many attempts were made and how long elapsed
5. The failure class (execution/diagnostic/verification/repeated_approach)

Then proceed to Step 2.

---

## Step 2: Escalation Ladder (ESCALATE)

Three escalation attempts max. First two are **auto-pick** (no user interaction needed). Third is **user-pick**. Each attempt is itself loop-bounded — if it loops, advance to the next rung.

### Attempt 1: Diagnostic checkpoint + sub-agent (AUTO)

**Before delegating, mandatory diagnostic checkpoint:**

1. State hypothesis (what do I think is wrong?)
2. State evidence (what did I observe?)
3. State what was tried and why it failed
4. State what NOT to try again

Then delegate with structured handoff packet:

```python
delegate_task(
  goal="<original task>",
  context="""HANDOFF PACKET:
- Goal: <original task>
- Hypothesis: <what I think is wrong>
- Evidence: <what I observed>
- Commands tried: <list with outputs>
- Repeated approaches: <DO NOT repeat these>
- Failure class: <execution/diagnostic/verification/repeated_approach>
- Current state: <what's changed so far>

IMPORTANT: Challenge the hypothesis before acting. Diagnose first, then propose a plan.""",
)
```

The sub-agent uses the `delegation.model` / `delegation.provider` from config (stronger model). It is bounded by `delegation.max_iterations` (set to 15) so it cannot loop either.

**On success:** Take the sub-agent's output and implement the solution. Done.

**On failure/loop:** Advance to Attempt 2.

### Attempt 2: Alternative model or OpenClaw Advisor (AUTO)

Try a different escalation path than Attempt 1:

**Option A — Different provider via delegate_task:**
If Attempt 1 used the delegation config model, override with a different strong model. Spawn a one-shot Hermes process:

```
terminal(
  command="hermes chat -q '<task + context + what Attempt 1 tried>' -m <different-model> --provider <different-provider>",
  timeout=300
)
```

**Option B — OpenClaw Advisor (if available):**
```
terminal(command="openclaw advisor '<task + context>' --model <stronger-model>", timeout=300)
```

**Option C — Mixture of both:**
If the task is complex, run two different-model delegates in parallel via `delegate_task(tasks=[...])` and pick the better output.

**On success:** Implement the returned solution. Done.

**On failure/loop:** Advance to Attempt 3.

### Attempt 3: User-pick (INTERACTIVE)

Stop and notify the user. Present:

```
⚠️ Loop Break — Escalation Needed (Attempt 3/3)

Task: <1-line description>
Attempts so far: <count>
Time elapsed: <duration>
Failed approaches:
  1. Direct (5 tries, <error summary>)
  2. Sub-agent [model name] (<failure reason>)
  3. Alt model [model name] (<failure reason>)

Pick an escalation option:
1️⃣ Switch session model (e.g., grok-4.5) — /model
2️⃣ Manual debug — you take over, I provide full context
3️⃣ Try a specific approach you suggest
4️⃣ Abort task
```

**AFK timeout:** If the user does not respond within **10 minutes**, break the loop and go directly to the Failure Report (below). Do not keep waiting.

### Each rung is loop-bounded

Every escalation attempt inherits Step 1's detection. If the escalated agent (sub-agent, spawned process, OpenClaw) itself loops:
- Do NOT let it run to `max_iterations`
- Detect after 3 consecutive identical failures OR 5 minutes elapsed
- Advance to the next rung immediately

---

## Step 3: Failure Report (REPORT)

If all 3 escalation attempts fail, or the user picks "Abort", or AFK timeout triggers on Attempt 3:

Deliver a structured failure report and STOP. Do not retry anything.

```
❌ Task Failed — All Escalation Exhausted

Task: <description>
Total time: <duration>
Total attempts: <count across all rungs>

What was tried:
  1. Direct: <approach> → <why it failed>
  2. Sub-agent [<model>]: <approach> → <why it failed>
  3. Alt escalation [<method>]: <approach> → <why it failed>

Root cause assessment: <best guess, or "unknown">
Suggested next steps: <1-2 concrete suggestions for the user>
```

---

## Quick Reference: The Full Flow

```
Task starts
  │
  ▼
Working? ──── Yes ──→ Done ✓
  │
  No (after 5 tries or 10 min)
  │
  ▼
STEP 1: BREAK — stop, record context
  │
  ▼
STEP 2: ESCALATE
  ├─ Attempt 1 (auto): sub-agent, stronger model ─→ works? Done ✓
  ├─ Attempt 2 (auto): different model/method    ─→ works? Done ✓
  └─ Attempt 3 (user): ask user to pick
       └─ User responds? ──→ execute their choice
       └─ AFK 10 min?    ──→ STEP 3
  │
  ▼
STEP 3: REPORT — structured failure, STOP
```

---

## Pitfalls

- Do NOT skip Step 1 and jump to escalation on first failure — the agent sometimes recovers
- Do NOT let escalation attempts run unbounded — each rung inherits the 3-try/5-min loop guard
- Do NOT escalate for transient errors (429, network blips) — use backoff retry instead
- Do NOT escalate for prompt-level issues — fix the prompt
- When delegating to a sub-agent, ALWAYS pass the failure context — a fresh agent with no error history will repeat the same mistakes
- During a task: do NOT revert to the original model — it already demonstrated it can't handle this task
- After task completes: revert to the default model (cheaper) for subsequent tasks
- The 10-minute timer for Step 1 starts from the **first failure**, not from session start or task start
- The 10-minute AFK timeout for Attempt 3 starts from when the user-pick message is sent

---

## Config Dependencies

The escalation ladder relies on these config settings:

```yaml
# Delegation — controls sub-agent model for Attempt 1
delegation:
  model: glm-5-turbo        # or stronger model for escalation
  provider: zai
  max_iterations: 15        # tight budget — escalation agent can't loop
  max_concurrent_children: 3
  child_timeout_seconds: 300

# Fallback models — available for Attempt 2 alternative paths
fallback_model:
  model: deepseek-ai/DeepSeek-V4-Flash
  provider: deepinfra
fallback_providers:
  - deepinfra
  - deepseek
  - xai
```

To change the escalation model: `hermes config set delegation.model <model> && hermes config set delegation.provider <provider>`

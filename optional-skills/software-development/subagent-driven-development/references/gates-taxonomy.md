# Gates Taxonomy

Canonical gate types for validation checkpoints across any workflow that spawns subagents, runs review loops, or has human-approval pauses. Every validation checkpoint maps to one of these four types — naming them explicitly makes the workflow legible and prevents "what happens when this check fails?" confusion.

Adapted from the GSD (Get Shit Done) project's gates reference — MIT © 2025 Lex Christopherson ([gsd-build/get-shit-done](https://github.com/gsd-build/get-shit-done)).

## The four gate types

### 1. Pre-flight gate

**Purpose:** Validates preconditions before starting an operation.

**Behavior:** Blocks entry if conditions unmet. No partial work created — bail before anything changes.

**Recovery:** Fix the missing precondition, then retry.

**Examples:**
- Implementation phase checks that the plan file exists before it starts writing code.
- Delegated subagent checks that required env vars are set before making API calls.
- Commit checks that tests passed before pushing.

### 2. Revision gate

**Purpose:** Evaluates output quality and routes to revision if insufficient.

**Behavior:** Loops back to the producer with specific feedback. Bounded by an iteration cap (typically 3).

**Recovery:** Producer addresses feedback; checker re-evaluates. The loop escalates early if issue count does not decrease between consecutive iterations (stall detection). After max iterations, escalates to the user unconditionally — never loop forever.

**Examples:**
- Plan reviewer reads a draft plan, returns specific issues, planner revises, reviewer re-reads (max 3 cycles).
- Code reviewer checks subagent-produced code against must-haves; dispatches fixes back to the implementer if any must-have failed.
- Test coverage checker validates new tests exercise the new paths; if not, sends back to author.

### 3. Escalation gate

**Purpose:** Surfaces unresolvable issues to the human for a decision.

**Behavior:** Pauses workflow, presents options, waits for human input. Never guesses, never picks a default.

**Recovery:** Human chooses action; workflow resumes on the selected path.

**Examples:**
- Revision loop exhausted after 3 iterations.
- Merge conflict during automated worktree cleanup.
- Ambiguous requirement — two reasonable interpretations and the choice changes the approach.
- Subagent reports "the plan says X but the codebase actually does Y" — human decides which is right.

### 4. Abort gate

**Purpose:** Terminates the operation to prevent damage or waste.

**Behavior:** Stops immediately, preserves state (checkpoint current progress), reports the specific reason.

**Recovery:** Human investigates root cause, fixes, restarts from checkpoint.

**Examples:**
- Context window critically low during execution (POOR tier, >70%) — abort cleanly rather than produce truncated output.
- Critical dependency unavailable mid-run (network down, API key revoked).
- Unrecoverable filesystem state (disk full, permissions lost).
- Safety invariant violated (agent attempted an irreversible destructive action outside approved scope).

## How to use this in a skill

When you write an orchestration skill that has validation checkpoints, **name each checkpoint by its gate type explicitly** and answer three questions:

1. **What condition triggers this gate?** (e.g., "plan file missing", "issue count didn't decrease", "context >70%")
2. **What happens when it fails?** (block / loop back / ask human / abort)
3. **Who resumes, and from where?** (fix precondition + retry, revise + re-check, human decision, restart from checkpoint)

Answering these three up front means your skill never hits "what do we do now?" at runtime.

## Example — a review loop with all four gate types

```
[Pre-flight] plan.md exists and is non-empty?   → no: bail, ask user to write a plan first
                ↓ yes
[Execute]  subagent implements task
                ↓
[Revision] reviewer checks against must-haves  → fail: loop back to subagent (max 3)
                ↓ pass
[Pre-flight] tests pass?                       → no: bail, report failing tests
                ↓ yes
[Commit]
                ↓
(on revision loop exhaustion)
[Escalation] "3 review cycles failed to converge on issue X — pick: force-merge, rewrite task, abandon"
                ↓ user picks
(on any tier-POOR context pressure during loop)
[Abort] "context at 73%, checkpointing and stopping"
```

The vocabulary is small on purpose. Every gate in every workflow should fit one of these four. If you find yourself inventing a fifth, it's probably a revision gate with extra branching, or an escalation gate in disguise.

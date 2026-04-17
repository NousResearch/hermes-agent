---
name: swarm-exploration
description: Explore a hard problem from multiple angles in parallel using delegate_task, then synthesize results via a structured merge strategy. Use for ambiguous problems where one approach may miss the real answer (debugging with unknown root cause, design decisions with tradeoffs, research questions with conflicting evidence).
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Research, Agent, Parallel, Delegation, Reasoning, Strategy]
    related_skills: [hermes-agent, claude-code]
    requires_tools: [delegate_task]
---

# Swarm Exploration

Parallelize reasoning on a single problem by spawning multiple subagents with **different thinking strategies**, then merge their outputs with a structured synthesis step. Built on top of `delegate_task` — no new tools required.

## When to Use

Trigger on problems where a single line of reasoning is likely to miss the real answer:

- **Debugging with unknown root cause** — the first plausible hypothesis often masks the real bug
- **Design decisions with tradeoffs** — you need multiple viable options side by side, not one "best"
- **Research questions with conflicting evidence** — parallel retrieval with different framings surfaces contradictions
- **Security / adversarial analysis** — obvious answer + "what would break this?" is strictly better than obvious answer alone

**Do NOT use when:**
- The problem has one obviously correct approach (just solve it)
- Subtasks are genuinely independent work (use plain `delegate_task` without branch strategies)
- Speed matters more than quality (each branch adds latency + cost)

## Quick Reference

| Phase | Action |
|-------|--------|
| 1. Pick branches | Choose 2-3 orthogonal strategies from the table below |
| 2. Spawn | `delegate_task(tasks=[branch1, branch2, branch3])` |
| 3. Merge | Apply a synthesis pattern to the returned `results` array |

### Branch strategies

| Strategy | Goal prefix to inject | When |
|----------|----------------------|------|
| `depth` | "Pick one approach, commit, go deep. Don't explore alternatives." | When you need a fully-worked solution |
| `breadth` | "Survey 3-5 options shallowly. Don't commit to one." | When you need to understand the option space |
| `adversarial` | "Assume the obvious answer is wrong. Find the failure mode, edge case, or hidden assumption." | Debugging, security, code review |
| `conservative` | "Prefer the safest, most reversible approach. Justify every risk." | Production changes, migrations |
| `aggressive` | "Optimize for speed. Skip verification. Propose the boldest solution." | Prototyping, brainstorming |
| `first-principles` | "Ignore conventional wisdom and existing patterns. Reason from fundamentals." | Architecture decisions, novel problems |

### Merge strategies

After `delegate_task` returns, pick one synthesis pattern:

- **`best_of_n`** — read all branch summaries, pick the single best answer, cite which branch
- **`concat`** — present all branches to the user as alternatives, let them choose
- **`debate`** — launch a second `delegate_task` with each branch critiquing the others, then synthesize
- **`vote`** — for discrete decisions (yes/no, A/B/C), majority wins

## Procedure

### Step 1: Decide if swarm is warranted

Ask yourself: *"If I solve this with one line of reasoning, is there >20% chance I miss the real answer?"* If yes → swarm. If no → just solve it directly.

### Step 2: Pick 2-3 branches

Two is the minimum useful count. Three is the sweet spot. Four+ usually means you haven't picked orthogonal strategies. Pick branches that can **disagree** — `depth` + `breadth` is useful, `depth` + `depth` is not.

### Step 3: Spawn with explicit strategy injection

Inject the strategy directly into the `goal` field. The subagent has no memory of your conversation — the goal must carry the strategy verbatim.

```
delegate_task(tasks=[
  {
    "goal": "STRATEGY: depth. Pick one approach, commit, go deep. Task: Debug why the test `test_profile_clone` fails on Windows. Output: root cause + patch.",
    "context": "Failing test file: tests/cli/test_profiles.py. Error: RecursionError in shutil.copytree.",
    "toolsets": ["terminal", "file"]
  },
  {
    "goal": "STRATEGY: adversarial. Assume the obvious fix is wrong. Find the failure mode. Task: Same test failure. Output: what would a naive fix break, and why?",
    "context": "Same context as above. Reviewer is hostile.",
    "toolsets": ["terminal", "file"]
  },
  {
    "goal": "STRATEGY: first-principles. Ignore the existing shutil approach. Task: Same failure. Output: what is the minimal correct implementation from scratch?",
    "context": "Same context as above.",
    "toolsets": ["terminal", "file"]
  }
])
```

### Step 4: Merge with the chosen strategy

**`best_of_n`** (most common):
```
Read all three summaries. Score each on:
  - Does it address the actual failure? (0-3)
  - Is it minimal / low-risk? (0-3)
  - Does it generalize? (0-3)
Pick the highest. Report: "Branch X wins because Y. Alternative was Z."
```

**`debate`** (when branches disagree strongly):
```
delegate_task(tasks=[
  {
    "goal": "Critique these two proposals: [proposal A] vs [proposal B]. Find the 2 strongest objections to each.",
    "context": "Full summaries from round 1: ...",
    "toolsets": ["terminal", "file"]
  }
])
# Then synthesize the critiques into a final answer yourself.
```

**`concat`** (when the user should choose):
Present all branches as a numbered list with 1-sentence summaries. Ask the user to pick.

**`vote`** (discrete decisions):
If each branch returned a yes/no or A/B/C answer, majority wins. Flag if unanimous disagreement.

## Pitfalls

- **Subagents have no memory of your conversation.** Every strategy prefix + all context must be repeated in each `goal`/`context`. Do not say "same as above" — the subagent cannot see above.
- **Cost scales linearly with branch count.** 3 branches ≈ 3x the tokens of a single solve. Budget accordingly.
- **Max concurrent children is usually 3.** If you try more, `delegate_task` rejects the call. Stay at 2-3.
- **Branches can collide on the filesystem.** If branches need to write files, give them separate workspace paths or use read-only toolsets (`file` without `terminal`).
- **Don't swarm every problem.** The skill's value is in *hard* problems. Over-applying wastes cost and adds latency for no gain.

## Verification

You've applied the skill correctly when:
- You explicitly named the strategy in each branch's `goal`
- The branches are **orthogonal** (would disagree if they found different things)
- You named the merge strategy before reading the results
- Your final answer cites which branch(es) it came from

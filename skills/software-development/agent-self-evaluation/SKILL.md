---
name: agent-self-evaluation
description: 'Self-rate output on 5 axes (accuracy, completeness, clarity, actionability, conciseness) after completing non-trivial tasks. Produces a structured 1-5 scorecard with improvement suggestions.'
version: 1.0.0
author: Hermes Agent (adapted from ECC)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [quality, self-review, evaluation, reflection]
    related_skills: [verification-loop, eval-harness]
---

# Agent Self-Evaluation

After completing a complex task, pause to rate your output against a structured 5-axis rubric. This is NOT a pass/fail gate — it's a deliberate reflection step that catches omissions, flags overconfidence, and surfaces areas for improvement before the user has to.

## When to Activate

- After writing code that spans 3+ files or 50+ lines
- After completing a multi-step workflow (implement → test → review)
- After a debugging session that involved 3+ attempts
- After producing a design document, architecture decision, or written analysis
- When the user asks "how good was that?" or "rate yourself"

## The 5 Evaluation Axes

| Axis              | Question                                               | What it catches                                                     |
| ----------------- | ------------------------------------------------------ | ------------------------------------------------------------------- |
| **Accuracy**      | Are the facts, claims, and outputs correct?            | Hallucinations, wrong API names, incorrect syntax, false statements |
| **Completeness**  | Did it cover everything the user asked for?            | Missed edge cases, unhandled error paths, forgotten requirements    |
| **Clarity**       | Is the explanation understandable and well-structured? | Confusing explanations, jargon, missing context, rambling           |
| **Actionability** | Can the user act on the output immediately?            | Vague suggestions, missing steps, no verification path              |
| **Conciseness**   | Did it use the minimum words/tokens needed?            | Redundancy, over-explanation, filler content                        |

## Scoring Scale

```
5 — Exceptional: no reasonable improvement possible
4 — Good: minor nits only, no substantive gaps
3 — Adequate: meets the request but has a notable weakness on at least one axis
2 — Weak: has a clear gap that affects usability or correctness
1 — Poor: fundamentally misses the request or contains significant errors
```

## The Evidence Rule

Every score below 5 MUST cite specific evidence. A score of 3 cannot just say "could be better" — it must say exactly what is missing or wrong. **"Show the gap, don't just name it."**

## Workflow

### Step 1: Collect the Raw Material

- The original user request
- Your final response/output
- Any tool outputs that verify correctness (test results, exit codes, lint output)
- Any user feedback received during the task

### Step 2: Score Each Axis Independently

Work through the 5 axes one at a time. Score each axis fresh — don't pre-average.

### Step 3: Produce the Evaluation Report

```
Scorecard:
  Accuracy:    X — [evidence]
  Completeness: X — [evidence]
  Clarity:     X — [evidence]
  Actionability: X — [evidence]
  Conciseness: X — [evidence]

Overall: X.X / 5

Top improvements:
1. [highest impact fix]
2. [next]
3. [next]
```

### Step 4: Apply the Improvement

If any axis scored 3 or below:

- State what you would do differently
- If gap is fixable in < 30 seconds, fix it now
- If gap requires rework, flag it explicitly

## Hermes Integration

- Run after `delegate_task` completes for subagent quality check
- Run after significant `write_file` or `replace_string_in_file` operations
- Combine with `verification-loop` for comprehensive quality gates
- Combine with `eval-harness` for formal evaluation tracking

# Engineering Cybernetics for Hermes Agent: A Practical Contribution

## From Paper-Tiger Skills to Embedded Protocol Rules

**Date:** 2026-05-10
**Framework:** Qian Xuesen, *Engineering Cybernetics* (1954)
**Target audience:** Hermes Agent users and contributors

---

## TL;DR

We diagnosed 6 structural defects in LLM-based agent architectures using control
theory. Then we tried to fix them with 6 dedicated Hermes skills. Those skills
were "paper tigers" — they existed in name but never fired reliably. We deleted
them all and replaced them with 4 protocol rules embedded directly into how we
work with Hermes. This document explains what we did, what works, and what
doesn't — honestly.

---

## 1. What We Did

We applied Qian Xuesen's Engineering Cybernetics framework to analyze how Hermes
Agent handles tasks. The analysis identified 6 structural defects that all
LLM-based agents share (detailed in our research repository). We initially
created 6 dedicated skills to address these defects. They didn't work — the
agent rarely loaded them, and when it did, it treated them as optional reference
material rather than operational rules.

The fix: delete the skills and embed the core ideas as **protocol rules** that
the agent follows during normal task execution. No extra skills to load. No
optional modules. Just behavioral rules baked into how we interact.

### What We Removed (6 Skills Deleted)

| Skill Name              | Why It Failed                                    |
|-------------------------|--------------------------------------------------|
| feedforward-loader      | Never auto-loaded; agent ignored it pre-execution |
| adaptive-gain           | No trigger mechanism; skill sat unused            |
| state-observer          | Conceptual; provided no executable procedure      |
| performance-tracker     | Required manual activation; agent forgot           |
| global-optimizer        | Too abstract; no actionable steps                  |
| skill-verification      | Circular: the agent would need to verify itself    |

These skills were "paper tigers" — they looked like they addressed the defects
but provided no reliable behavioral change. The agent had to remember to load
them, choose to follow them, and self-assess compliance. None of that happened
consistently.

### What We Kept

| Item                        | Status                                                |
|-----------------------------|-------------------------------------------------------|
| qian-xuesen-cybernetics     | Retained as reference skill (theory, not operations)  |
| hermes-self-evolution       | Updated with verification concepts from our analysis  |

The `qian-xuesen-cybernetics` skill remains as a knowledge base — it contains
the mapping matrix and structural analysis. It is reference material, not an
operational procedure. We do not rely on it to change agent behavior.

The `hermes-self-evolution` skill was updated to incorporate verification
concepts: before accepting any self-modification, define what "success" looks
like and check whether the change achieved it.

---

## 2. The 4 Protocol Rules

These replace the 6 deleted skills. Each rule is embedded into the agent's
interaction protocol — not loaded from a file, but followed as a behavioral
convention.

### Rule 1: Checkpoint 3 Questions (Feedforward)

Before executing any non-trivial task, the agent asks itself three questions:

    1. What is the explicit goal?
       -> Restate the task in your own words. If the restatement diverges
          from the request, the human can correct you BEFORE you start.

    2. What could go wrong?
       -> Predict failure modes based on the task type. For file operations:
          wrong path, wrong encoding, overwrite risk. For code: syntax errors,
          missing dependencies, logic bugs. For research: stale data, wrong
          source, confirmation bias.

    3. How will we verify success?
       -> Define acceptance criteria before starting. "The file exists and
          opens correctly." "The code runs without errors." "The analysis
          covers all requested aspects."

**How to use it:** When you give Hermes a task, you can prompt it by asking
"What are your checkpoint 3 questions for this task?" This forces the agent to
plan before executing. Over time, the agent should do this automatically.

**What it replaces:** The `feedforward-loader` skill, which tried to do the
same thing but required explicit skill loading that rarely happened.

### Rule 2: 2-Line Performance Log

After every task, the agent writes two lines:

    TASK: [one-line description of what was requested]
    OUTCOME: [success/partial/failure] - [one-line description of what happened]

**Where it writes:** Into the conversation itself, at the end of the task
response. This makes it visible to the human reviewer without requiring a
separate file or tool.

**Why two lines:** A verbose log will not be maintained. Two lines is the
minimum viable performance metric. By reviewing the ratio of successes to
failures over time, a human can determine whether the agent's performance is
improving (converging), stalled, or degrading.

**What it replaces:** The `performance-tracker` skill, which tried to maintain
a structured log file but required manual activation and was never used.

### Rule 3: Error Classification and Repair

When a task fails or produces unexpected results, classify the error:

    Type 1: Skill Gap
      -> The agent lacks the knowledge or procedure for this task
      -> Repair: Document the gap. Add to capability wishlist or create a skill.

    Type 2: Execution Deviation
      -> The agent knew what to do but did it wrong
      -> Repair: Add a checkpoint or constraint for next time.

    Type 3: Environment Change
      -> External conditions changed; old approach no longer works
      -> Repair: Update procedures. Invalidate stale knowledge.

**How to use it:** When something goes wrong, ask "What type of error was this?"
The classification determines the repair strategy. A skill gap needs new
knowledge. An execution deviation needs a new constraint. An environment change
needs updated procedures.

**What it replaces:** The `adaptive-gain` and `skill-verification` skills, which
attempted automatic error correction but had no reliable trigger mechanism.

### Rule 4: Capability Matching (for delegate_task)

Before delegating a subtask, the orchestrating agent explicitly considers:

    - What capabilities does this subtask require?
    - Which available agent/sub-agent has those capabilities?
    - What are the dependencies between subtasks?
    - What could go wrong in delegation?

**How to use it:** When Hermes uses delegate_task to spawn sub-agents, the
orchestrator should briefly map task requirements to available capabilities.
This is manual and approximate — but it is better than random assignment.

**What it replaces:** The `global-optimizer` skill, which attempted automated
task allocation but was too abstract to execute.

---

## 3. Honest Assessment

### What Works

The checkpoint 3 questions genuinely change agent behavior. When the agent
restates a task before executing, it catches gross misunderstandings early.
When it predicts failure modes, it is more careful during execution. The
protocol-layer approach works because it does not require loading an extra
skill — the rules are followed as part of the normal interaction flow.

The 2-line performance log is beginning to produce data. It is too early to
detect trends, but the format is simple enough to maintain consistently.

### What Doesn't Work (Yet)

**Self-compliance is unreliable.** The protocol-layer reforms depend on the
agent choosing to follow the rules. There is no enforcement mechanism. The
agent may skip the checkpoint questions if it is confident. It may write
"OUTCOME: success" when the task actually failed in a way it cannot detect.
It may classify an error incorrectly because the same model that caused the
error is classifying it.

This is the fundamental self-reference problem: the agent is both the
controller and the controlled. The observer is made of the same material as the
observed. This is not a bug we can fix at the protocol layer.

**The human is the feedback sensor.** The agent cannot reliably assess its own
performance. The human reviewer is not a convenience — they are a structural
requirement. Removing human oversight does not make the system more autonomous;
it makes it an uncalibrated open loop.

### What Needs Code Changes

The following cannot be achieved through protocol rules alone:

    1. Automatic performance logging (no reliance on agent discipline)
    2. Task classification before execution (programmatic, not judgment-based)
    3. Persistent failure-mode database (cross-session learning)
    4. Confidence estimation from token probabilities
    5. Automatic rollback when self-modification degrades performance
    6. Capability-aware task allocation for multi-agent delegation

These are the code-layer reforms. They require instrumentation, not protocol.
The protocol rules are a bridge — a way to start collecting data and proving
value before committing to engineering work.

---

## 4. Summary Table

| Component                  | Status         | Reliability | Notes                          |
|----------------------------|----------------|-------------|--------------------------------|
| Checkpoint 3 Questions     | Active         | Medium      | Depends on agent self-compliance|
| 2-Line Performance Log     | Active         | Medium      | Beginning data collection      |
| Error Classification       | Active         | Low-Medium  | Self-assessment bias risk      |
| Capability Matching        | Active (manual)| Low         | Approximate, not systematic    |
| qian-xuesen-cybernetics    | Reference only | N/A         | Knowledge base, not operations |
| hermes-self-evolution      | Updated        | Medium      | Now includes verification      |
| 6 deleted skills           | Removed        | N/A         | Were paper tigers              |

---

## 5. How to Use This in Your Own Hermes Setup

If you want to apply these ideas to your own Hermes Agent:

1. **Do not create dedicated skills for control-theoretic corrections.** They
   will become paper tigers. Embed the rules into your interaction protocol
   instead.

2. **Ask the checkpoint 3 questions.** Before any complex task, ask Hermes:
   "What is the goal? What could go wrong? How will we verify?" This costs
   30 seconds and saves hours of rework.

3. **Maintain the 2-line log.** After tasks, note the outcome. Build a
   personal history. Look for patterns.

4. **Classify errors when they happen.** The 3-type classification (skill gap,
   execution deviation, environment change) determines the right fix. Don't
   treat all failures the same.

5. **Review the agent's self-assessment.** Do not trust the agent's claim of
   success. Verify independently. The human-in-the-loop is the feedback sensor
   the system cannot provide for itself.

---

## 6. References

This contribution is based on a systematic application of Qian Xuesen's
Engineering Cybernetics (1954) to LLM-based agent architectures. The full
research includes:

- 15 mappings between control theory concepts and agent mechanisms
- 6 structural defects identified through the control-theoretic lens
- Analysis of Open Complex Giant Systems (OCGS) implications

The research repository contains the full analysis:
- 01-mapping-matrix.md — 15 mappings with similarity ratings
- 02-structural-defects.md — 6 defects with control-theoretic prescriptions
- 03-reform-protocol.md — Protocol-layer reform details and implementation status
- 04-ocgs-implications.md — Why AI agents are Open Complex Giant Systems

---

*This document is itself an instance of the reform protocol it describes.
The checkpoint 3 questions were asked before writing began. The performance
log follows.*

TASK: Write Hermes community contribution document
OUTCOME: [pending human review]

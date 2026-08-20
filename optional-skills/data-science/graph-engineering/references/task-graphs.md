# Task Graphs: Orchestrating Agents
*(The execution half of graph engineering — how agents work, as opposed to what they remember)*

## Contents
- [What a task graph is](#what-a-task-graph-is)
- [Fake edges](#fake-edges)
- [The diamond pattern](#the-diamond-pattern)
- [The stop rule](#the-stop-rule)
- [The human gate](#the-human-gate)
- [Guardrails](#guardrails)

## What a task graph is

Nodes are jobs — each one something you would hand to a single assistant (research one
competitor, write one draft, check one claim). Draw an arrow only when a job needs another
job's *result* before it can start. The drawing is the plan; agents flow through it.
A small state object (what was found, what was decided, what remains) travels with the work.

This is a DAG — the pattern that has run data infrastructure for decades (Airflow, Prefect,
Temporal) now applied to agents (LangGraph, CrewAI, AutoGen). The age of the pattern is a
feature: trust your business to machinery with decades of production history.

## Fake edges

The first optimization costs nothing: for every "and then" in an existing pipeline, ask
whether the next job actually reads the previous job's output. "Summarize this file and then
check my calendar" — the calendar step never uses the summary; the edge is fake. Delete fake
edges and those jobs run in parallel. Most hand-built pipelines contain two or three.

## The diamond pattern

The shape serious systems converge to:

```
        ┌─ worker 1 ─┐
plan ───┼─ worker 2 ─┼─→ verify ─→ merge ─→ result
        └─ worker 3 ─┘
```

Split the task into independent angles, run workers in parallel, **verify in a separate
context**, merge survivors. The verification node is non-negotiable: a model grading its own
work in its own context misses most of its own mistakes. Give each verifier a different
question (is it correct? is it current? is the source real?) — diverse skeptics catch what
identical ones cannot.

## The stop rule

From the Google DeepMind × MIT study "Towards a Science of Scaling Agent Systems"
(180 controlled configurations): coordinated teams beat a single agent by ~80% on work that
splits into independent pieces — and **every** multi-agent configuration lost on sequential
work where each step needs the full picture (degrading 39-70%). Uncoordinated agents
amplified each other's errors 17.2×; a single coordinator owning the merge cut it to 4.4×.

The decision procedure:
1. Ask: *where does my work split into pieces that never read each other's results?*
2. Split only that. Everything sequential stays with one agent.
3. Never let findings merge without one owner of the merge.

More agents is not a strategy. The shape of the work decides.

## The human gate

The human is a node. Route every irreversible edge — send, publish, refund, delete, deploy —
through explicit approval. Placement rule: **put the gate where a mistake is expensive to
undo, not on every step.** A gate on everything makes the human the bottleneck; a gate on
nothing means nobody is watching. Judge the system on numbers that cannot argue back (tests
that ran, money that landed), never on its own self-reports.

## Guardrails

Four caps that keep a graph from becoming an expensive accident:
1. Every loop gets a maximum number of rounds.
2. One writer per file — no two jobs mutate the same artifact.
3. The routing lives in written steps; the model fills the jobs, not the plan.
4. A hard cap on how many agents can spawn.

\# Measurement Protocol — v1 (draft, freeze Week 2)



\## Environment

\- Machine: <CPU, RAM — fill in>

\- OS: <version — fill in>

\- Hermes commit: <paste from docs\\pinned\_commit.txt>

\- Model + temperature: <TBD — lock with team in Week 1>

\- Python: <python --version output>



\## Tokens
- Scope: task tokens only (input + output of the task run).
  Learning tokens are tracked by B2 separately and are not in ScoreReport.

\- Source: <Hermes model client field — confirm Week 2>

\- Counted: prompt + completion

\- Tool: <script name — Week 2>



\## RAM

\- Tool: psutil

\- Process tree: Hermes process + all children

\- Sampling: peak RSS over the whole run

\- Warm vs cold: <decide and record>



\## Latency

\- From: process start

\- To: process exit

\- Unit: seconds



\## Repeats

\- Minimum 3 runs per measurement

\- Report mean and spread



\## Splits

\- Train: used by the variation agent

\- Held-out: never seen during evolution



\## Change rule

\- Frozen before the first number.

\- Any change → version bump + PR note + re-run affected measurements.


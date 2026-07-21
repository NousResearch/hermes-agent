# Agent Pulse deployment defaults

## Goal
Make Agent Pulse behave consistently after deployment, even outside the original development session.

## Required defaults
- explicit trigger only: `Agent Pulse` / `/pulse` / clearly equivalent direct status query
- fixed default output card
- baseline mode preferred over self-influenced status
- no proactive runs
- no long explanation unless requested

## Recommended signal policy
Use cheap, reproducible signals first:
- running task
- blocked state
- queued messages
- pending actions
- active project
- delivery due
- recent state before pulse

## Recommended fallback policy
If signals are weak or conflicting:
- use `unknown`
- do not guess hidden state
- do not escalate to model-heavy reasoning unless necessary

## Product promise
Agent Pulse should be legible, stable, compact, and safe to trust as a first-pass status signal.

# Agent Pulse

A lightweight interruptibility and load-status skill for OpenClaw, Hermes, and other agent runtimes.

## What it does

Agent Pulse answers one simple but important question:

**Is the agent available, interruptible, and safe to give new work right now?**

It uses low-cost rule-based heuristics instead of heavy model reasoning.

## Output

```text
Agent Pulse
status: <idle|light|busy|blocked|unknown>
interruptibility: <high|medium|low>
acceptNewTask: <yes|caution|no>
reason: <short reason>
```

## Standard triggers

- `Agent Pulse`
- `/pulse`
- natural-language variants like:
  - "你现在忙吗？"
  - "现在负荷怎么样？"
  - "能接新任务吗？"

## Why it matters

Most agents do not expose a compact status signal.

Agent Pulse makes agent state more legible:
- useful for human-agent collaboration
- useful for multi-agent orchestration
- useful for deciding whether to interrupt or queue work

## Included scripts

- `scripts/pulse_eval.py` — evaluates low-cost signals into a pulse result
- `scripts/render_pulse.py` — renders the strict output card

## Example

```bash
python3 scripts/pulse_eval.py --json '{"runningTask":true,"queuedMessages":2,"blocked":false,"waitingExternal":false,"minutesSinceAssistant":2,"recentState":"working"}'
```

Example output:

```json
{"status":"busy","interruptibility":"low","acceptNewTask":"caution","reason":"rule-based:busy","signals":["running-task","queued:2","assistant-min:2","recent:working"]}
```

## Status model

- `idle` — not actively occupied
- `light` — active but still available
- `busy` — occupied; interrupt sparingly
- `blocked` — waiting on dependency/tool/human
- `unknown` — insufficient/conflicting signals

## Deployment defaults

To reproduce the intended product behavior, deploy Agent Pulse with these defaults:

- Trigger only on explicit requests such as `Agent Pulse` or `/pulse`
- Return the fixed 4-line pulse card by default
- Prefer baseline status, not self-influenced status
- Do not let the pulse query itself count as workload evidence
- Prefer deterministic rule-based signals over model-heavy judgment
- Use `unknown` when signals are insufficient instead of pretending certainty

## Behavior contract

By default Agent Pulse should:
- answer compactly
- expose current availability, interruptibility, and task acceptance state
- avoid long explanation unless asked
- avoid claiming hidden internal certainty

By default Agent Pulse should not:
- run proactively without an explicit trigger
- treat the pulse check itself as proof of busyness
- replace full project management or task tracking
- overfit to a single chat message

## Current maturity

Agent Pulse is currently a strong MVP / beta skill.

It is already usable and publishable, but should continue evolving with richer signal sources and more real-world usage.

## License

MIT

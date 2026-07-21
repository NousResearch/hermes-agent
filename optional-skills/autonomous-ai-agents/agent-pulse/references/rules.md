# Agent Pulse rules

## Canonical triggers
- `Agent Pulse`
- `/pulse`
- natural-language equivalents asking if the agent is busy, interruptible, or able to take work

## Canonical output

```text
Agent Pulse
status: <idle|light|busy|blocked|unknown>
interruptibility: <high|medium|low>
acceptNewTask: <yes|caution|no>
reason: <short reason>
```

No extra commentary unless requested.

## Baseline rule
The pulse query itself must not count as workload evidence.
Prefer the agent state immediately before the pulse check.
Do not treat the current pulse interaction as proof that the agent is busy.

## Suggested low-cost inputs
- `runningTask`: bool
- `queuedMessages`: int
- `blocked`: bool
- `waitingExternal`: bool
- `minutesSinceUser`: number
- `minutesSinceAssistant`: number
- `recentState`: optional string (`working|blocked|waiting|idle`)
- `activeProject`: bool
- `pendingActions`: int
- `hasStartedWork`: bool
- `deliveryDue`: bool

For OpenClaw-style runtimes, ignore routine system completion notices and pulse-check noise when estimating `queuedMessages` or recent work pressure.

## Default classification

### blocked
- `blocked=true`, or
- `recentState=blocked`

### busy
- `runningTask=true` and `queuedMessages>=1`, or
- `recentState=working` and `minutesSinceAssistant<=5`, or
- `queuedMessages>=3`

### light
- `runningTask=true`, or
- `queuedMessages` is 1-2, or
- `minutesSinceAssistant<=15`

### idle
- no running task
- no queue
- not blocked
- `minutesSinceAssistant>15`

### unknown
- key signals missing or conflicting

## Interruptibility mapping
- blocked + waitingExternal -> high
- blocked otherwise -> medium
- busy -> low
- light -> medium
- idle -> high
- unknown -> medium

## Accept-new-task mapping
- idle -> yes
- light -> yes
- busy -> caution
- blocked -> caution
- unknown -> caution
- use `no` only when overload/risk is explicit

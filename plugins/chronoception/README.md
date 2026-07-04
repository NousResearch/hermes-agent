# chronoception — a sense of elapsed time

A language model is stateless: with no timestamp in its context it cannot tell how
much wall-clock time passed between turns, and will act on stale information as if
no time had passed. In an ablation on a local model, with no temporal cue the model
answered *"0 seconds elapsed, confidence 1.0"* on every trial — confidently wrong,
and worse than random. Given an explicit timestamp it reasoned about elapsed time
near-perfectly. This plugin supplies that timestamp.

Each turn it injects a small fenced block into the current user message
(ephemeral, never persisted):

```
<turn-clock>
[System note: your own operational timing from the runtime — not the user's words. …]

clock 2026-07-04 14:30, +12 min since your last turn.
</turn-clock>
```

When the agent resumes after a long idle gap it adds a one-shot notice that
time-sensitive state may have moved on.

## Enable

```yaml
chronoception:
  enabled: true             # opt-in; absent/false = inert
  clock: true               # per-turn clock (temporal grounding; costs prefix-cache reuse).
                            #   false = warn only on a long idle gap (rare, cache-cheap)
  gap_report_seconds: 1800  # idle threshold for the "you were dormant" notice
  max_chars: 300
```

No external dependencies — the signal is wall-clock only. Fail-closed: any config
or runtime error leaves it silent and never breaks a turn.

## Cost note

With `clock: true` the block changes every turn and is stripped from history, so it
re-prefills the previous turn's tail once per turn — real prefix-cache cost on
tool-heavy sessions. Hence it is opt-in. A history-persisted stamp (fixed once
written, cache-neutral) is a natural future refinement.

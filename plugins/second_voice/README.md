# second_voice

Opt-in "Second Voice" step-reflection guardrail for Hermes.

When enabled, a `tool_execution` middleware sends each gated tool call — plus the operator's task
instruction — to an auxiliary LLM in a strict, isolated context window (temperature 0). The
referee answers exactly one line:

- `APPROVE` — the tool executes normally.
- `REDO: <reason>` — the tool is blocked; the reason is fed back to the executor as a tool error
  so it corrects course.
- `ESCALATE: <reason>` — the tool is blocked with a "stop and ask the user" error. Repeated
  REDOs (`max_consecutive_rejections`) escalate the same way.

**Fail-closed:** any LLM error, empty response, or ambiguous verdict escalates — a gated tool
never executes on a step the referee did not explicitly clear.

## Enable

```bash
hermes plugins enable second_voice
```

## Config (`plugins.entries.second_voice.settings` in config.yaml)

| Key | Default | Meaning |
|---|---|---|
| `reflect_tools` | `["terminal", "write_file", "patch", "save_file"]` | Tools gated by the referee. |
| `max_consecutive_rejections` | `3` | REDOs in one turn before forced escalation. |
| `max_instruction_chars` | `2000` | Cap on task-instruction text sent to the referee. |
| `model` | *(auxiliary default)* | Override the referee model (`auxiliary` config). |
| `timeout` | `30.0` | Seconds before the referee call gives up (fail-closed → escalate). |
| `temperature` | `0.0` | Referee sampling temperature. |
| `max_tokens` | `256` | Referee response budget. |

## Latency and cost

Each gated tool call pays one synchronous auxiliary-LLM round trip before the tool executes —
typically a few seconds on a small/fast auxiliary model, bounded by `timeout` (a timeout is a
fail-closed escalation, not a silent bypass). Keep `reflect_tools` narrow: every call to a listed
tool pays the round trip. The defaults cover the tools where a wrong step is most expensive to
undo (shell, file writes); trim or extend the list to match your workload's cost/benefit.

## Safety notes

- The proposed tool call is UNTRUSTED (it originates from the executor LLM, which may itself be
  prompt-injected). Arguments and the tool name are HTML-escaped before entering the referee
  prompt, and the system prompt instructs the referee to ignore directives inside the `<step>`
  block.
- The breaker (`max_consecutive_rejections`) is bounded per process and keyed by
  `(session_id, turn_id)`; it resets on approval, escalation, and at each new turn.

# living-subsystems

Independent subsystems that persist JSON under the active profile's `HERMES_HOME` and share one
contract: `run()` / `status()` return `{"ok", "message", "details"}`.

| Subsystem | File | Purpose |
|---|---|---|
| `governance` | `governance_audit.json` | Audit trail + `block_action()` pre-flight for dangerous commands |
| `science-loop` | `goals.json` | Hypothesis → experiment → retain/discard/modify |
| `reflective-evolution` | `learnings.json` | Record lessons, `diagnose_failure()` from similar past failures |
| `fitness-builder` | `fitness_functions.json` | Normalized weighted multi-dimension scoring with history |

Enable with `hermes plugins enable living-subsystems`, then e.g. `hermes subsystems status` or a cron
job running `hermes subsystems run governance`. Nothing hooks the agent loop; `Governance().block_action()`
is a regex screen for obviously destructive commands, not a sandbox.

# Direct-command workers (`worker.command`)

A named Hermes profile may declare a fixed argv as its Kanban worker. Cards
assigned to that profile then run the command instead of the native Hermes
agent while retaining the board, dispatcher, task worktree, PID/run lifecycle,
dependencies, wakeups, reclaim/retry accounting, logs, and `max_runtime`.

```yaml
# ~/.hermes/profiles/engine/config.yaml
worker:
  command:
    - /absolute/path/to/worker-launcher
    - luna
    - --
    - "Work the Kanban task identified by HERMES_KANBAN_TASK_ID."
```

The key is valid only in a named profile's `config.yaml`; it is not a card or
root-dispatcher setting. The value is a non-empty list of argv strings,
`argv[0]` must be an existing absolute executable path, and Hermes never uses
`shell=True` or card text to build the command. Bare names and relative paths
are rejected so a task worktree cannot change which host executable runs.
Each argv item is passed literally after YAML parsing: Hermes performs no
shell, environment-variable, or backslash expansion.

The command runs directly under a small supervisor with the task workspace as
its cwd. It receives a sanitized environment, including `HOME`, `PATH`,
`HERMES_PROFILE`, and the bounded explicit Kanban context:
`HERMES_KANBAN_TASK_ID` (plus the established `HERMES_KANBAN_TASK` alias),
`HERMES_KANBAN_RUN_ID`, `HERMES_KANBAN_CLAIM_LOCK`, `HERMES_KANBAN_DB`,
`HERMES_KANBAN_BOARD`, `HERMES_KANBAN_WORKSPACE`, and
`HERMES_KANBAN_WORKSPACES_ROOT`. Parent/default provider keys, API keys,
gateway relay secrets, GitHub tokens, and supervisor-only command configuration
are not passed to the command.

Exit handling is deliberately narrow:

- exit `0` may call `complete_task`;
- non-zero exit may call `block_task` with the exit code;
- an already-canonical transition wins through `expected_run_id`;
- a missing executable/config fails closed and never falls back to the native
  Hermes worker;
- a supervisor termination reports no command outcome. The dispatcher remains
  the sole writer of `timed_out`, retry phase, `consecutive_failures`, and the
  circuit breaker.

Direct commands do not receive agent-only card controls such as skills, model
or provider overrides, reasoning effort, or goal mode. They emit no Hermes
heartbeats, so set the task's `max_runtime`/`max_runtime_seconds`. The
dispatcher and supervisor terminate the complete process tree on POSIX and
Windows; a timeout therefore kills descendants without converting the timeout
into an ordinary command block.

# Direct-command workers (`worker.command`)

A named profile can declare a fixed command as its kanban worker. Cards
assigned to that profile are then executed by that command instead of the
Hermes agent — the board, dispatcher, worktree workspaces, logs, crash
detection and `max_runtime` all behave exactly as they do for agent
workers. This gives deterministic pipelines (an orchestrator binary, a
build-and-ship script) a first-class seat on the board without wrapping
themselves in a prompt.

```yaml
# ~/.hermes/profiles/engine/config.yaml
worker:
  command:
    - /usr/local/bin/my-pipeline
    - --from-kanban
```

## Scope — where the key may live

- **Only a named profile's `config.yaml`** (a home of the shape
  `.../profiles/<name>`). This is deliberate and different from every
  `kanban.*` key, which are all read in *dispatcher* scope.
- Declaring it in the **root config is an error**: the root file is
  dispatcher scope, and the `default` assignee resolves to it. The spawn
  fails loudly instead of running for `default` and silently not running
  for everyone else.
- The value must be a **list of argv strings** (never a shell string), and
  `argv[0]` must be an absolute path or a bare `PATH`-resolved name — a
  relative path would resolve against the per-task workspace, whose
  content the task's own branch controls. `${VAR}` / `${env:VAR}`
  references expand against the dispatcher's environment.
- A declared-but-invalid value — including a config file that no longer
  parses as YAML — **fails the spawn loudly**. It never silently falls
  back to the agent: running the wrong worker on a task is worse than
  running none.

## The completion contract — the exit code is the report

The dispatcher runs the command under a thin supervisor
(`hermes_cli.kanban_command_worker`) that is its direct parent, waits for
it, and reports while still alive:

| Command outcome | Card |
|---|---|
| exits `0` | `done` (`complete_task`) |
| exits non-zero (any code) | `blocked`, with the code in the reason. **No retry** — a deterministic pipeline that failed is a fact for a human; the retry loop belongs to the pipeline itself. Unblocking re-runs it. |
| already moved its card through a canonical channel (`kanban complete` / `block` / review) | that transition stands; the exit-code translation is a no-op |
| killed by a signal | `blocked`, with the signal named |
| cannot even start (wrong path, missing binary) | the supervisor's own failure: the card retries as a **crash** up to the failure limit; the message is in the task log |
| the supervisor itself dies | ordinary crash path — nothing is invented |

Because the supervisor is the direct parent, the exit code is observed in
every dispatcher constitution: resident gateway, `kanban daemon`, and
throwaway cron `kanban dispatch` ticks alike.

The command receives the standard worker environment
(`HERMES_KANBAN_TASK`, `HERMES_KANBAN_WORKSPACE`, `HERMES_KANBAN_DB`,
`HERMES_KANBAN_BOARD`, `HERMES_KANBAN_RUN_ID`) and runs with the task's
workspace as its working directory. It does not need to talk to kanban at
all — exiting is reporting — but it may use the canonical channels for
richer outcomes (e.g. blocking with a question).

## Stall bound — set `max_runtime`

A direct command emits **no heartbeats**, so `max_runtime_seconds` on the
card is the **only** stall bound. Set it. On timeout the supervisor
forwards SIGTERM to the command's whole process group (grandchildren
included), and after a short grace (3 s, `HERMES_KANBAN_COMMAND_TERM_GRACE`
to adjust, clamped below the dispatcher's own kill window) SIGKILLs the
group and reports the death as a block.

## Interaction with agent-only settings

`skills`, `model_override`, `provider_override`, `reasoning_effort` and
`goal_mode` on a card are meaningless for a direct command and are ignored
with a warning in the dispatcher log. Everything an operator can influence
lives in host-side configuration — nothing on the card (title, body,
comments, attachments) can alter what is executed.

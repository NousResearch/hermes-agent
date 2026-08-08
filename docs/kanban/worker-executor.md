# Kanban worker executor

The dispatcher spawns each claimed task as a detached child process. By default
that child is a native Hermes worker:

```
hermes -p <assignee> --cli --accept-hooks [...] chat -q "work kanban task <id>"
```

That lane resolves the model through Hermes' own provider stack. On an
Anthropic-backed profile it bills the configured `provider: anthropic`
credentials, which on some setups is a third-party / extra-usage pool that can
exhaust independently of the operator's interactive Claude subscription. When
that pool is empty, every worker fails at startup even though
`env -u CLAUDE_CONFIG_DIR claude -p ...` runs fine in the same shell.

`kanban.worker_executor` lets an operator point workers at that working lane —
the Claude Code CLI itself — explicitly.

## Configuration

```yaml
kanban:
  # hermes (default) | claude_cli
  worker_executor: claude_cli

  # Optional. Absolute path or bare command name; default: `claude` on PATH.
  claude_cli_bin: /opt/homebrew/bin/claude

  # Extra argv appended before `-p <prompt>`. YAML list or a single
  # shell-quoted string. In practice this is REQUIRED — see "Permissions".
  claude_cli_extra_args:
    - --permission-mode
    - acceptEdits

  # Optional. `--model` for the direct lane when a task carries no Claude
  # model override of its own.
  claude_cli_model: claude-opus-5

  # Optional. Minimum seconds between two direct-lane `claude` process
  # startups on this Hermes root. Default 2.0; clamped to [0, 60].
  claude_cli_spawn_stagger_seconds: 2.0
```

`worker_executor` is a behavioral setting, so it lives in `config.yaml`, not in
`.env`. It is per-profile-root like the rest of the `kanban:` block.

## Permissions — read this before you conclude the lane is broken

`claude -p` runs in the CLI's default permission mode, which asks a human
before it edits a file or runs a command. A dispatcher worker has no TTY, so
every such request is auto-denied.

**`--permission-mode acceptEdits` covers file edits only — not Bash.** This is
the trap. A worker set up that way was observed reporting that *every* `hermes`
call it made was denied: it never learned what its task was, and could not
comment, block, or complete. It stranded the task.

Two consequences, and Hermes handles them differently.

**The board protocol is granted for you.** Every direct-lane spawn appends a
least-privilege allowlist for exactly the five subcommands the worker prompt
tells it to run:

```
--allowedTools "Bash(<hermes> kanban show:*)" "Bash(<hermes> kanban heartbeat:*)" \
               "Bash(<hermes> kanban comment:*)" "Bash(<hermes> kanban block:*)" \
               "Bash(<hermes> kanban complete:*)"
```

That is the minimum required for the contract the prompt asks the worker to
fulfil — a worker that cannot report its own outcome is worse than one that
never started. It grants no general Bash, no `Edit`, and no `Write`. If you
supply your own `--allowedTools`, these are merged into your list rather than
added as a second flag (a second occurrence would win and silently drop yours).

**The task's actual work permissions stay yours.** Hermes will not widen those
for you. Choose them in `claude_cli_extra_args` based on what the tasks on this
board need — `--permission-mode acceptEdits` for edit-only work, something
broader if the work must run commands. A spawn with no permission flag at all
logs a warning explaining the above.

### Argument order is load-bearing

`--allowedTools` (and several other CLI flags) are variadic: the run of values
only ends at the next option-looking token. The prompt is therefore always
passed as `-p <prompt>` **last**. Put a prompt directly after a variadic flag
and the CLI consumes it as another value, then exits with
`Input must be provided either through stdin or as a prompt argument`.

## What changes, and what does not

Only the argv and the credential routing change. Both lanes get the identical
child environment — `HERMES_KANBAN_DB`, `HERMES_KANBAN_BOARD`,
`HERMES_KANBAN_WORKSPACES_ROOT`, `HERMES_KANBAN_TASK`,
`HERMES_KANBAN_WORKSPACE`, `HERMES_KANBAN_RUN_ID`, `HERMES_KANBAN_CLAIM_LOCK`,
`HERMES_PROFILE`, `HERMES_TENANT`, `HERMES_SESSION_SOURCE`, the terminal
timeouts derived from `max_runtime_seconds`, and the `HERMES_TUI` drop that
keeps a worker from booting the interactive TUI. The child is still detached
(`start_new_session`), still runs with `cwd` set to the task workspace, still
writes to `<board-root>/logs/<task>.log` with the same rotation, and still
returns its PID so the dispatcher's crash detection works unchanged.

Because the Claude CLI has neither the `kanban_*` model tools nor Hermes'
`KANBAN_GUIDANCE` system prompt, the direct lane sends a self-contained
protocol prompt instead of `work kanban task <id>`. The worker drives the
lifecycle through the `hermes kanban` subcommands (`show`, `comment`,
`heartbeat`, `complete`, `block`), which read the board pins already in its
environment.

The prompt embeds the *resolved* Hermes invocation (the same
`_resolve_hermes_argv()` the native lane launches with), not a bare `hermes`.
When the dispatcher runs from a venv whose console script is not on the child's
`PATH`, a literal `hermes kanban complete` would fail and the worker would exit
without ever closing its task.

Argument *order* in those commands is also load-bearing. `block`'s reason is a
`nargs="*"` positional, and on Python 3.11 — the pinned runtime — a nested
subparser rejects a positional that trails an optional. So the prompt says
`block <id> "<why>" --kind needs_input`, not `block <id> --kind needs_input
"<why>"`; the latter exits 2 with `unrecognized arguments`. Newer CPythons
accept both, which is exactly how this hides in local testing.
`test_every_prompted_command_parses` runs every command the prompt names
through the real `hermes kanban` parser so an invented or misordered flag
cannot ship.

## Claim and heartbeat behavior

A direct-lane worker keeps its claim the same way a native one does. The
dispatcher records the child's PID; `release_stale_claims` extends the claim of
any host-local task whose PID is still alive, so a long-running direct-lane
worker is **not** reclaimed out from under itself.

`hermes kanban heartbeat <id>` is a real subcommand and the worker prompt
instructs the worker to call it periodically. It authenticates via the
`HERMES_KANBAN_TASK` / `HERMES_KANBAN_RUN_ID` pins already in the child's env.
Heartbeats are what re-enable the wedged-worker backstop: a task whose PID is
alive but whose last heartbeat is older than
`DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS` is reclaimed rather than extended.
A direct-lane worker that never heartbeats keeps its claim on PID liveness
alone and therefore loses that backstop — it is protected against spurious
reclaim, not against being wedged.

## Credential policy

The direct lane removes these variables from the child's environment and adds
nothing:

| Variable | Why it is dropped |
|---|---|
| `CLAUDE_CONFIG_DIR` | This is the `env -u CLAUDE_CONFIG_DIR` behavior that works. The child then reads the operator's ordinary `~/.claude` store itself. |
| `ANTHROPIC_API_KEY` | Keeps an inherited key from silently moving the run onto metered API billing. |
| `ANTHROPIC_AUTH_TOKEN` | Same. |
| `ANTHROPIC_BASE_URL` | Keeps an inherited gateway/proxy override from redirecting the subscription lane. |
| `CLAUDE_CODE_USE_BEDROCK` | Would move the run onto an AWS account's billing. |
| `CLAUDE_CODE_USE_VERTEX` | Would move the run onto a GCP account's billing. |

Hermes never reads, copies, decrypts, or rewrites the Claude credential store.
No token is placed in argv, in the child env, in the board database, or in a
worker log — the child authenticates itself, on its own schedule.

### Concurrency and the shared `~/.claude` store

Every direct-lane worker shares one per-user `~/.claude` state directory, and
each `claude` process reads and rewrites parts of it at startup. Several
workers booting at the same instant can interleave those writes; the visible
symptom is a truncated config and an interactive session that abruptly asks you
to log in again.

Hermes serializes that window. Direct-lane spawns take a cross-process lock
under `<hermes-root>/kanban/claude-cli-spawn.lock` and enforce
`claude_cli_spawn_stagger_seconds` (default 2s) between startups, so at most one
direct-lane worker is in its startup window at a time. The lock is per Hermes
root, not per board, because the contended resource is per user. If the lock
cannot be taken within 30s the spawn proceeds unstaggered and logs a warning —
a stuck holder degrades the stagger, it never stops the board.

Be precise about the limit of this guarantee: it covers process **startup**
only. An OAuth refresh performed mid-run by two long-lived workers happens
inside Anthropic's CLI, which Hermes does not lock and whose credential store
Hermes does not touch. If that matters for your setup, bound real concurrency
with `kanban.max_in_progress_per_profile`.

## No silent fallback

Selecting `claude_cli` is a commitment. If the binary is missing, not
executable, or the configured `claude_cli_bin` path does not exist, the spawn
raises with an actionable message naming the lane and the exact path, and the
task is *not* started on the native provider instead — a quiet downgrade would
put the run back on the exhausted pool the setting exists to avoid. An
unrecognized `worker_executor` value is treated as a typo: it is logged loudly
and the native default is used, so a misspelling never changes billing.

Each direct-lane spawn logs the executor, the resolved binary, the *names* of
the argv flags, and the *names* of the stripped variables — never a flag value,
an env value, or the prompt — both to the dispatcher log and as a one-line
header in the task's worker log.

## Limitations

- **Goal mode is unsupported, and refuses rather than degrades.** The
  Ralph-style `/goal` judge loop lives in the Hermes CLI's quiet path. Spawning
  a `goal_mode` task on the direct lane raises; the attempt is recorded against
  the task and reaches a human after `kanban.failure_limit`. Clear `goal_mode`
  or run that task on the native executor.
- **Per-task `--toolsets`, `--skills`, and `--reasoning` pins do not apply** —
  they are Hermes CLI flags. Use `claude_cli_extra_args` for the CLI's own
  equivalents.
- A task `model_override` is forwarded only when it names a Claude model on the
  Anthropic lane; anything else is dropped with a warning rather than passed to
  a CLI that cannot resolve it.
- The worker's lifecycle compliance is prompt-enforced, not tool-enforced. A
  native worker's `kanban_complete` is a model tool the harness can require; a
  direct-lane worker is asked to run a shell command. A worker that exits
  without `complete` or `block` is caught by the dispatcher's existing
  PID-vanished crash detection, one attempt later than the native lane.

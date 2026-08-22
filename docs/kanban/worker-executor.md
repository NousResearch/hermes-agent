# Kanban worker executor

The dispatcher spawns each claimed task as a detached child process. That child
is the Claude Code CLI running under the operator's own host login:

```
claude -p [--model ...] --permission-mode bypassPermissions [...] "<protocol prompt>"
```

This is the default for **every board and every profile**. The alternative lane
is the native Hermes worker:

```
hermes -p <assignee> --cli --accept-hooks [...] chat -q "work kanban task <id>"
```

which resolves the model through Hermes' own provider stack. On an
Anthropic-backed profile it bills the configured `provider: anthropic`
credentials, which on some setups is a third-party / extra-usage pool that can
exhaust independently of the operator's interactive Claude subscription. When
that pool is empty, every worker fails at startup even though
`env -u CLAUDE_CONFIG_DIR claude -p ...` runs fine in the same shell — and the
dispatcher cannot tell that apart from a wedged board. One review card was
re-dispatched 132 times behind repeated 429s before the breaker tripped. That
is why the direct lane, which began as strictly opt-in, is now the default.

## Configuration

```yaml
kanban:
  # claude_cli (default) | hermes (alias: native)
  worker_executor: claude_cli

  # Optional. Absolute path or bare command name; default: `claude` on PATH.
  claude_cli_bin: /opt/homebrew/bin/claude

  # --permission-mode for the run. Default `bypassPermissions` — see
  # "Permissions". Set to "" to add no flag at all (read-only lane).
  claude_cli_permission_mode: bypassPermissions

  # Extra argv appended before the prompt. YAML list or a single
  # shell-quoted string. A permission flag here wins over the setting above.
  claude_cli_extra_args:
    - --add-dir
    - /srv/shared

  # Optional. `--model` for the direct lane when a task carries no Claude
  # model override of its own.
  claude_cli_model: claude-opus-5

  # `--effort` for the direct lane when a card pins no reasoning_effort of
  # its own. Default `medium`. Set to "" to add no flag at all.
  claude_cli_effort: medium

  # Optional. Minimum seconds between two direct-lane `claude` process
  # startups on this Hermes root. Default 2.0; clamped to [0, 60].
  claude_cli_spawn_stagger_seconds: 2.0
```

`worker_executor` is a behavioral setting, so it lives in `config.yaml`, not in
`.env`. It is per-profile-root like the rest of the `kanban:` block.
`HERMES_KANBAN_WORKER_EXECUTOR` overrides it for a single process.

To pin the default explicitly (and verify it is being read):

```
hermes config set kanban.worker_executor claude_cli
hermes config get kanban.worker_executor
```

### Going back to the native lane

```
hermes config set kanban.worker_executor native
```

This is the only supported way back. There is no automatic fallback: an
unresolvable `claude` binary is a hard error that records `spawn_failed`
against the task, and an unrecognized `worker_executor` value falls *forward*
to the direct lane. A typo must never silently restore the routing that wedged
the board.

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

**The task's actual work permissions default to `bypassPermissions`.** While
this lane was opt-in, Hermes only warned here and added nothing: silently
widening a worker's privileges was not its call, and an unarmed worker was a
no-op on one board someone had deliberately switched over. As the default lane
that same warning would make *every* worker a no-op — able to read and post
board comments, unable to edit a file or run a command, i.e. unable to do any
task it is given.

So `claude_cli_permission_mode` defaults to `bypassPermissions`. This is parity
rather than escalation: a native Hermes worker already runs with the full
unattended tool surface, and the direct-lane worker is doing the same job in
the same claimed workspace with stdin closed and no human to ask.

To choose differently, set `claude_cli_permission_mode` (or pass your own
permission flag in `claude_cli_extra_args`, which wins) — `acceptEdits` for
edit-only work, or `""` for a read-only lane, which restores the old
warn-and-add-nothing behavior.

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

### Thinking depth

Every direct-lane worker runs with an explicit `--effort`. The level is
resolved once per spawn, in this order:

1. the card's own `reasoning_effort`, if it pins one;
2. `kanban.claude_cli_effort`;
3. `medium` — the built-in default.

Medium is stated in argv rather than left to the CLI's own default on purpose.
"Which depth did this worker run at" is a question an operator has to be able
to answer from evidence after the fact, and an implicit default is not
evidence. The resolved level is written to the worker log header as
`effort=<level>`, next to `executor=` and `bin=`:

```
[kanban] executor=claude_cli bin=/usr/local/bin/claude board=reefmind \
  profile=integrator flags=--model,--effort,--permission-mode,--allowedTools,-p \
  effort=medium stripped_env=-
```

Effort is the only flag whose *value* appears in that header. That is
deliberate and it is bounded: the value is written only when it came from the
closed allowlist below, so the header cannot become a place a secret leaks.
An `--effort` the operator supplied themselves in `claude_cli_extra_args` is
arbitrary text, so it is reported as `effort=operator` and its value is
withheld like every other flag value.

The host CLI accepts `low`, `medium`, `high`, `xhigh`, `max`. Hermes'
`reasoning_effort` vocabulary is wider, so the extra levels are translated to
the nearest level in the same direction — never upward into more thinking than
the card asked for:

| Card `reasoning_effort` | `--effort` | Note |
|---|---|---|
| `low` / `medium` / `high` / `xhigh` / `max` | same | forwarded as-is |
| `minimal` | `low` | floor of what the CLI offers; logged |
| `ultra` | `max` | ceiling of what the CLI offers; logged |
| `none` | `low` | **`--effort` cannot disable thinking.** Logged. Omitting the flag would inherit a session default that may be *more* thinking than `none` asked for, and refusing the card would strand it, so the floor is the least-wrong option. Run the card on `native` if disabled reasoning is a hard requirement. |
| unrecognized | lane default | logged; see below |

An unrecognized level — on the card or in `claude_cli_effort` — falls forward
to the lane default instead of reaching the CLI. This is not conservatism:
`claude --effort bogus` is argv the CLI rejects outright, so the worker would
die at startup and the card would show an unexplained `spawn_failed` with
nothing pointing at the typo.

Setting `claude_cli_effort: ""` adds no flag at all and lets the host CLI pick;
the header then reads `effort=-`. An `--effort` in `claude_cli_extra_args`
takes precedence over all of the above and suppresses the resolved flag
entirely, so the run never carries `--effort` twice.

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

The direct lane removes these variables from the child's environment:

| Variable | Why it is dropped |
|---|---|
| `CLAUDE_CONFIG_DIR` | This is the `env -u CLAUDE_CONFIG_DIR` behavior that works. The child then reads the operator's ordinary `~/.claude` store itself. |
| `ANTHROPIC_API_KEY` | Keeps an inherited key from silently moving the run onto metered API billing. |
| `ANTHROPIC_AUTH_TOKEN` | Same. |
| `ANTHROPIC_BASE_URL` | Keeps an inherited gateway/proxy override from redirecting the subscription lane. |
| `CLAUDE_CODE_USE_BEDROCK` | Would move the run onto an AWS account's billing. |
| `CLAUDE_CODE_USE_VERTEX` | Would move the run onto a GCP account's billing. |
| `ANTHROPIC_BEDROCK_BASE_URL`, `ANTHROPIC_VERTEX_BASE_URL`, `CLOUD_ML_REGION` | The rest of the cloud-account routing surface. |
| `CLAUDE_API_KEY`, `CLAUDE_CODE_API_KEY_HELPER` | Alternate ways to hand the CLI a key instead of the host login. |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `CODEX_HOME`, `CODEX_API_KEY` | A kanban worker must never reach the OpenAI-Codex stack, and a credential left in a detached child's environment is how it ends up quoted in a task log. |

It adds exactly two variables: `HERMES_KANBAN_EXECUTOR=claude_cli`, so anything
reading the child's env can tell which lane produced it, and
`GIT_TERMINAL_PROMPT=0`, so a worker doing git work fails fast instead of
blocking forever on a credential prompt nobody can answer.

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

The direct lane is a commitment, not a preference. If the binary is missing, not
executable, or the configured `claude_cli_bin` path does not exist, the spawn
raises with an actionable message naming the lane and the exact path, and the
task is *not* started on the native provider instead — a quiet downgrade would
put the run back on the exhausted pool this lane exists to avoid. An
unrecognized `worker_executor` value is treated as a typo: it is logged loudly
and the *default* is used. Note the direction — a typo resolves forward to the
direct lane, never back onto the provider stack that wedged the board.

Each direct-lane spawn logs the executor, the resolved binary, the *names* of
the argv flags, and the *names* of the stripped variables — never a flag value,
an env value, or the prompt — both to the dispatcher log and as a one-line
header in the task's worker log.

## Limitations

- **Goal mode has no judge loop; the worker judges itself.** The Ralph-style
  `/goal` loop lives in the Hermes CLI's quiet path and has no CLI equivalent.
  While this lane was opt-in the spawn refused a `goal_mode` card outright,
  because the alternative was one unjudged pass that looks like success. As the
  default lane that would fail every goal card on every board, so the prompt
  instead carries an explicit GOAL MODE section telling the worker to verify
  its work against the card's success criteria before completing, and to block
  rather than complete on a partial result. `goal_max_turns` is passed through
  as a soft round budget. This is weaker than the native judge loop — run
  goal-critical cards on `native` if you need the real thing.
- **Per-task `--toolsets` pins do not apply** — that is a Hermes CLI flag. Use
  `claude_cli_extra_args` for the CLI's own equivalent. (Per-task
  `reasoning_effort` *does* apply on this lane — see "Thinking depth" above.)
- **Per-task `skills` are named in the prompt, not injected.** The native lane
  passes `--skills X` and Hermes resolves each into the worker's system prompt
  before its first turn. The host CLI has no equivalent flag, so this lane
  lists the required skills in the prompt and asks the worker to load them
  with its own Skill tool, blocking `--kind capability` if one will not load.
  That is an instruction, not an injection: a direct-lane worker *can* ignore
  it where a native worker could not.

  This stopped being a cosmetic gap when the review handoff lifecycle landed.
  `dispatch_once` force-appends `sdlc-review` to a claimed review card because
  that skill *is* the review procedure (AC verification, merge). With this
  lane global, silently dropping the field would hand every review card a
  worker that does not know how to review — and the run would look normal, not
  misconfigured. If you need the hard guarantee for review cards, run them on
  `native`.
- A task `model_override` is forwarded only when it names a Claude model on the
  Anthropic lane; anything else is dropped with a warning rather than passed to
  a CLI that cannot resolve it.
- The worker's lifecycle compliance is prompt-enforced, not tool-enforced. A
  native worker's `kanban_complete` is a model tool the harness can require; a
  direct-lane worker is asked to run a shell command. A worker that exits
  without `complete` or `block` is caught by the dispatcher's existing
  PID-vanished crash detection, one attempt later than the native lane.

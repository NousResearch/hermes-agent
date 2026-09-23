---
name: tmux-supervision
description: Observe tmux applications from their owning chat.
version: 0.4.0
author: vsabavat, Hermes Agent
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [tmux, supervision, automation, coding-agent]
    category: autonomous-ai-agents
---

# Tmux Supervision Skill

Observe newly launched applications in tmux and wake their owning Hermes
conversation through native background completion. The default adapter observes
process lifecycle; an optional OMP adapter adds application-specific turn events.
Neither process exit nor a settled turn proves that the user's goal was achieved.

## When to Use

- Run Claude Code, Codex, OMP, a build, a test suite or another foreground app in tmux.
- Receive completion without a polling cron, pane scraping, another bot or a core plugin.
- Preserve the interactive terminal while keeping monitoring bound to the originating chat.

Use `delegate_task` for Hermes subagents. Generic monitoring reports process exit,
not each conversational turn or approval prompt in a still-running application.
It does not adopt existing workers or steer, approve, cancel or restart them.

## Prerequisites

- Linux; Python 3.11+; tmux; local `terminal` execution on the application's host.
- The requested application installed and authenticated through its own normal workflow.
  Generic monitoring requires neither OMP nor Node and injects no model or permission flags.
- For rich OMP events only: [OMP](https://github.com/can1357/oh-my-pi) lifecycle hooks
  compatible with v18.2.10. Verify the installed runtime through `terminal`;
  never upgrade or restart a worker just to enable observation.
- A Hermes gateway supporting session-owned native background completion and exporting
  the durable session ID on every turn, including cached-agent resumes. The latter
  depends on [Hermes #58914](https://github.com/NousResearch/hermes-agent/pull/58914).
  Without it, an initial notification may arrive but safe re-arming is refused.
- Genuine inherited `HERMES_HOME`, `HERMES_SESSION_PLATFORM`, `HERMES_SESSION_KEY`,
  `HERMES_SESSION_ID`, `HERMES_SESSION_CHAT_ID`, and `HERMES_SESSION_THREAD_ID`
  (present but allowed to be empty). Never fabricate scope to make enrollment pass.
- An authorized workspace and command. Use `write_file` for task handoffs and command files;
  never include credentials in command arguments, transcripts or monitoring state.

The bundled Python helper uses only the standard library and runs from a copied
skill directory without pip or Hermes imports. Node 22.6+ is needed only for the
synthetic OMP extension tests; OMP supplies its own TypeScript runtime.

## How to Run

Resolve `scripts/tmux_supervise.py` relative to this installed skill. Use its
absolute path through `terminal`. `SCRIPT`, `WORKSPACE`, `COMMAND`, `PROMPT` and
`RUN` below represent discovered paths; quote them as individual arguments.

1. Use `write_file` to create `COMMAND`, a JSON argv array: for example `["claude"]`,
   `["codex"]` or `["python", "-m", "pytest", "tests"]`. These are command shapes,
   not claims of live validation against those agents. Preserve the user's chosen
   arguments and permissions. Shell syntax is not evaluated unless an explicitly
   authorized shell is the command. Do not modify this file after launch.
2. Run `python SCRIPT prepare --workspace WORKSPACE --tmux-session NAME`.
   Read the returned `run_dir` as `RUN`. The default adapter is `command`.
   State defaults to the real `HERMES_HOME/tmux-supervision`; `--state-root DIRECTORY`
   overrides storage, not ownership. Use a private short path if the socket path is too long.
3. Start `python SCRIPT watch --run-dir RUN --timeout SECONDS` through `terminal`
   with `background=true, notify_on_complete=true` (or the host's `notify=true`).
   Verify that the actual result accepted native completion notification and retained
   the originating session. Choose a finite bound covering the longest expected
   silent operation, up to 86400 seconds; the 300-second default suits short canaries.
4. Use `python SCRIPT status --run-dir RUN` to confirm `observer_active` is true.
   This readiness check is not proof of native delivery.
5. Run `python SCRIPT launch --run-dir RUN --command-file COMMAND`.
   It reserves one launch, starts only the enrolled new tmux session and leaves
   stdin/stdout/stderr connected to its terminal. Completion includes `exit_code`.
6. End the Hermes turn when appropriate so native completion can wake it.
   Do not replace the event path with repeated status requests or a polling cron.

For OMP turn supervision, select `--adapter omp` during preparation and use
`python SCRIPT launch-omp --run-dir RUN --prompt-file PROMPT` in step 5.
Only that adapter accepts `--model`, `--thinking`, `--append-system-prompt FILE`,
`--omp-executable` and `--canary`. Unspecified options preserve OMP's own defaults.
The generic adapter never reads an agent's private configuration or installs hooks.

## Quick Reference

- `prepare --adapter command|omp`: bind one owner, workspace, adapter and new tmux session.
- `watch`: lock the observer, replay unseen events and emit one actionable receipt.
- `launch --command-file FILE`: generic process start/exit observation.
- `launch-omp --prompt-file FILE`: OMP turn, tool-approval and session-lifecycle observation.
- `status`: inspect validated journal, cursor, observer and launch state.
- `--tmux-executable`: explicit tmux client override for either launcher.
- `process_exited`: terminal event; `exit_code` is the child's exit status (negative for a signal).
- OMP `--canary`: disable tools, skills and ambient extensions; still a model request, not a sandbox.

## Procedure

1. Preserve healthy existing workers. New monitoring requires explicit enrollment
   before launch; do not rename or restart a worker to pretend it was enrolled.
2. Choose the needed signal: process exit for any foreground command; semantic events
   only from a supported application adapter. A quiet terminal is not completion.
3. Prepare, arm, verify readiness, then launch. Never run the watcher through another
   chat, a subagent, cron or an invented identity environment.
4. On a receipt, verify run, epoch, sequence and ownership before acting. Inspect
   actual task evidence with `read_file`, `search_files` and appropriate tests.
5. Distinguish process exit from successful work; inspect nonzero exits and verify
   outputs even after exit zero. Generic monitoring ends when the launched process
   exits, not when every detached descendant finishes; use foreground commands.
6. For OMP, distinguish `turn_settled`, `needs_input`, `error`, `session_revoked`
   and `shutdown`. Re-arm before authorized continuation of the same session.
   Stop after verified completion, pause, session revocation or shutdown.
7. `observation_lost` with reason `timeout` is renewable by the same owner without
   resetting identity or cursor. Other observation losses and process exit are terminal.
   Surface a loss without killing, restarting or silently transferring the worker.
8. Retain state while observing. Cleanup is separately authorized; never delete a
   binding, cursor or launch intent to force a retry or replay.

## Pitfalls

- The immutable owner includes profile home, platform, routing key, Hermes session
  generation, chat and thread. A tmux name or thread ID is not sufficient.
- The observer cannot authenticate that its parent enabled native notifications;
  check the tool result instead of describing shell backgrounding as native delivery.
- The protocol carries bounded lifecycle events, not prompts, tool output or commands.
  Same-UID code is trusted; this is not a sandbox for malicious applications.
- Launch failure after intent is committed is ambiguous. Inspect existing tmux/process
  state; do not relaunch or erase intent. Monitoring failure never authorizes termination.
- Generic Claude/Codex support means process monitoring, not built-in turn/approval hooks.
  Interactive apps may remain open between tasks indefinitely. Add a reviewed adapter
  with real application signals before claiming richer support; never infer it from pane text.
- OMP `needs_input` covers documented tool approvals, not every possible custom modal.
- Cursor persistence precedes host delivery: a crash can lose notification. Do not
  promise exactly-once delivery or erase cursors to recover it.
- Protocol v2 uses neutral application identity and an explicit adapter. Do not reuse
  older OMP-specific bindings; existing installed workers and state remain untouched.
- Prior Discord delivery evidence does not validate this new artifact. Other platform
  identifiers have synthetic contract coverage only; test the actual native host path.

See [protocol](references/protocol.md), [security](references/security.md) and
[validation](references/validation.md) for contracts, limits and reproducible checks.

## Verification

Run deterministic unit, real-socket, isolated-installation and tmux tests described in
[validation](references/validation.md). They require no model credentials or network.
Use a harmless foreground command in an empty workspace to test genuine native delivery;
verify its exit status and that the actual originating conversation woke.

For an OMP adapter release, also run a tools-disabled two-turn canary on the installed
artifact, re-arming before the second turn and verifying sequence advancement without
replay. Report socket receipt, native wake and visible message separately. A prior
build's canary or synthetic scope does not prove live acceptance. Preserve existing jobs.

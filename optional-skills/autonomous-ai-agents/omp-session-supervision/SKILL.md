---
name: omp-session-supervision
description: Observe interactive coding work in its owning chat.
version: 0.3.0
author: vsabavat, Hermes Agent
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [omp, tmux, supervision, coding-agent]
    category: autonomous-ai-agents
---

# OMP Session Supervision Skill

Observe newly launched interactive [OMP](https://github.com/can1357/oh-my-pi)
work through a private local socket. Hermes's native background completion wakes
the owning conversation; this skill does not add a bot, cron job or core plugin.
It neither adopts existing workers nor decides whether a settled turn completed
the user's goal.

## When to Use

- Launch interactive OMP in tmux from a Hermes gateway conversation.
- Receive turn, approval, error and session-lifecycle events without scraping panes.
- Re-arm observation before authorized continuation of the same task.

Use ordinary `delegate_task` for Hermes subagents. This is specifically an OMP
interactive-session integration, not a generic agent protocol or RPC launcher.

## Prerequisites

- Linux; Python 3.11+; local `terminal` execution on the same host as OMP/tmux.
- OMP and tmux installed and discoverable on PATH, or explicit executable paths.
  OMP supplies its own TypeScript runtime and configured model authentication.
- OMP lifecycle hooks compatible with v18.2.10. Verify the installed version with
  `terminal`; never upgrade or restart existing workers merely to enable observation.
- A Hermes gateway that supports session-owned native background completion.
  Do not substitute shell backgrounding, a second notification bot, or plugin injection.
  It must also export the durable session ID on every turn, including cached-agent
  resumes. This depends on the gateway correction tracked in
  [Hermes #58914](https://github.com/NousResearch/hermes-agent/pull/58914).
  Without it, the first notification can arrive but safe re-arming is refused.
- Genuine inherited `HERMES_HOME`, `HERMES_SESSION_PLATFORM`, `HERMES_SESSION_KEY`,
  `HERMES_SESSION_ID`, `HERMES_SESSION_CHAT_ID`, and `HERMES_SESSION_THREAD_ID`
  (present but allowed to be empty). Never fabricate scope to make enrollment pass.
- An authorized workspace and task prompt written to a file with `write_file`.
  A completion notice is not permission to approve tools or continue beyond scope.

The bundled Python helper uses only the standard library; no pip installation
or Hermes source imports are required. Node 22.6+ is needed only for the synthetic
extension tests, not to launch OMP.

## How to Run

Resolve `scripts/omp_supervise.py` relative to this installed skill. Use its
absolute path in `terminal` commands; `SCRIPT`, `WORKSPACE`, `PROMPT`, and `RUN`
below denote actual discovered values, not literal paths or synthetic scope.
Quote paths as individual shell arguments.

1. Run `python SCRIPT prepare --workspace WORKSPACE --tmux-session NAME`.
   Read its JSON `run_dir` as `RUN`. The state root defaults to the current real
   `HERMES_HOME/omp-supervisor`; `--state-root DIRECTORY` overrides only storage,
   never ownership. Use an owner-private short path if the Unix socket path is too long.
2. Start `python SCRIPT watch --run-dir RUN --timeout SECONDS` through `terminal`
   with `background=true, notify_on_complete=true` (or the host's equivalent
   `notify=true` parameter). Verify the actual tool result accepted native completion
   notification and retained the originating session. A PID alone is insufficient.
   Choose a finite bound covering the expected longest silent operation, up to
   86400 seconds. The 300-second default suits short canaries, not long builds.
3. Use `python SCRIPT status --run-dir RUN` to confirm `observer_active` is true.
   This local readiness check does not prove native notification delivery.
4. Run `python SCRIPT launch --run-dir RUN --prompt-file PROMPT`.
   It creates only the explicitly enrolled tmux session. It does not terminate OMP.
5. End the Hermes turn when appropriate so the native event can wake it.
   Do not replace the event path with repeated status requests or a polling cron.

Missing model, thinking or system-prompt options preserve OMP's own defaults.
When the task specifies them, pass `--model MODEL`, `--thinking LEVEL` and/or
`--append-system-prompt FILE` explicitly. Never read authentication files or put
credentials in prompt files, launch arguments, journals or messages.

## Quick Reference

- `prepare`: create immutable owner binding and private state; no worker starts.
- `watch`: hold the sole observer lock, replay unseen events, emit one actionable receipt.
- `launch`: reserve a durable one-shot launch intent, then create a new tmux session.
- `status`: inspect validated local journal, cursor, observer and launch state.
- `--omp-executable` / `--tmux-executable`: explicit launcher overrides.
- `--canary`: launch with OMP tools, skills and ambient extensions disabled.
  It still contacts the configured model; it is not a sandbox or free request.
- Thinking values: `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, `auto`.
  Model/provider support may be narrower.

## Procedure

1. Preserve healthy same-task workers. Explain that an already-running OMP session
   without this launch-specific extension cannot be retroactively enrolled.
2. Prepare, arm native observation, verify readiness, then launch. Never run the
   watcher through another chat, a subagent, cron, or an invented scope environment.
3. On completion, verify the run, epoch, sequence and owner before acting. Use
   `read_file`, `search_files` and appropriate tests to inspect actual task evidence.
4. Distinguish `turn_settled`, `needs_input`, `error`, `session_revoked`, `shutdown`,
   and `observation_lost`. A turn ending is a checkpoint, not verified success.
5. If authorized work remains, re-arm `watch` on the same run before continuing OMP.
   Stop re-arming after verified completion, a user pause, revocation or shutdown.
   `observation_lost` with reason `timeout` is a renewable watcher expiry: re-arm
   without resetting owner, epoch or cursor. Other observation losses are terminal.
6. Surface observation loss without killing or restarting the worker. A reset
   cannot transfer the enrollment to a replacement conversation.
7. Retain state while observation is active. Cleanup is a separate authorized action;
   never delete a binding, cursor or launch intent to force retries or replay.

## Pitfalls

- The immutable owner includes the real profile home, platform, routing key,
  Hermes session generation, chat and thread. A tmux name or thread ID alone is not ownership.
- The CLI cannot authenticate that its parent enabled native notifications.
  Check the terminal result; do not describe an ordinary shell watcher as session-bound delivery.
- The socket transports bounded typed events, not prompts, transcripts or tool output.
  It does not steer, approve, cancel or restart work. Same-UID code remains trusted.
- Launch failures may be ambiguous after intent is committed. Inspect the actual
  tmux/process state instead of retrying the same launch or clearing its intent.
- Watchers are finite and one-event. Ordinary timeout is renewable by the same owner;
  revocation, shutdown and lost continuity are terminal. Never clear a terminal
  cursor. There is no permanent daemon or crash-recovery promise.
- Cursor advancement precedes host delivery. A crash can lose a notification;
  this is not exactly-once delivery, and erasing cursors is not a safe recovery method.
- `needs_input` covers OMP tool approvals, not every possible custom modal.
- Discord has prior live-delivery evidence; other gateway identifiers have synthetic
  contract coverage only. Validate the actual host/adapter before promising delivery.

See [protocol](references/protocol.md), [security boundary](references/security.md)
and [validation](references/validation.md) for limits and reproducible checks.

## Verification

Run the deterministic source, cross-language socket and skill-installation tests in
[validation](references/validation.md). They require no model credentials or network.

For real delivery, use an authorized tools-disabled canary in an empty workspace:
prepare, arm native observation and launch with `--canary`. Let its completion
actually wake the originating conversation, then re-arm before a second harmless
turn. Confirm the new sequence advances without replaying the first event.

Report local socket receipt, native conversation wake and human-facing delivery
separately. Do not claim a new installation is verified live from a successful
subprocess, test count or an earlier build's canary. Preserve existing jobs.

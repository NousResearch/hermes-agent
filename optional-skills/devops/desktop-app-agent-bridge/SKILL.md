---
name: desktop-app-agent-bridge
description: "Use when a desktop app's agent must run on another host."
version: 1.1.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [acp, ssh, bridge, mcp, desktop-app, watchdog, migration]
    category: devops
---

# Bridging a desktop app's agent to a remote host

## Overview

Some desktop apps ship their own AI agent runtime and launch it as a **local subprocess over stdio**
(ACP = Agent Client Protocol), while also exposing the app's own tools to that agent through a
**local MCP socket**. This skill moves the agent's *brain* to another machine while the app stays
put — and keeps the app's own tool surface working across the SSH hop.

Three things make this harder than "wrap the binary", and each one costs a full debugging cycle if you
assume instead of check:

1. **The app launches a differently-named binary than you expect** — it maps its own agent id to an
   executable name and passes ACP as *argv*. Bridging the wrong name is a silent no-op.
2. **An externally-spawned client sees a different tool surface** than the app's own child process —
   narrower, or (after an app update) much wider.
3. **An app update can invalidate the whole arrangement without touching your launcher.**

## When to Use

- A desktop app ships its own agent and the user wants that agent's brain, skills, or model
  credentials on another machine.
- A profile or bot an app installed locally must be "moved to the server" while the app stays.
- The app's tools talk to a local socket/app process and must keep working after such a move.

Don't use this for: apps that already support a remote agent backend natively (use their setting), or
apps whose tools are all HTTP APIs you can call directly (skip the bridge, call the API).

## Step 0 — find out how the app really launches the agent

Never assume the hook point. The wrong hook looks like success: the app answers, tests pass, and the
agent is still running locally.

1. **Read the process tree**, don't guess: `ps -Ao pid,ppid,etime,command | grep -i <appname>` and
   walk parent → child. A local run shows the app's child as a python/node/binary of the *local*
   install.
2. **Read the app bundle** for the agent-id → binary map. Electron apps: parse `app.asar` in Python
   (header is `<IIII` — `_, header_size, _, json_size` — then JSON; payload offsets are relative to
   `16 + header_size`), then search the file table for a map like
   `CLI_BY_AGENT_ID = {"claude-acp": "claude", "codex-acp": "codex", "hermes-acp": "hermes"}`.
3. **The map's value is the executable name searched for, not an argument.** In the verified case the
   app looks for a binary named `hermes` and spawns `hermes acp`; it never executes `hermes-acp`.
   Bridging `hermes-acp` alone is a no-op that still passes every hand-run test.
4. **The argv lives in a different bundle chunk than the map.** Search a window around the argv
   identifier across all chunks; a parser scoped to the map's file returns `args: null` and reports a
   false anomaly.
5. Confirm by process tree afterwards: the app's child must be `ssh -T …`, not a local runtime.

**Done when:** you can state the exact binary name and argv the app spawns, read out of its own
bundle, and the process tree agrees.

## Step 1 — the ACP stdio bridge (the agent's brain)

Replace the launcher with an SSH hop. Set the remote profile explicitly in the remote command: the app
exports its own `HERMES_HOME` locally and that value must not be honoured remotely (SSH does not
forward env by default, so simply set it on the far side). See `scripts/acp-bridge.sh.template`.

Rules that matter:

- **Make the wrapper conditional.** Only the ACP argv shape goes remote; everything else falls through
  to the untouched local CLI. A blanket replacement breaks every other local caller.
- **Let the startup probe through.** The app runs `<cli> acp --version` first; if that fails it decides
  the agent is unavailable. Match the whole `acp*` shape, not just bare `acp`.
- **Back up both wrappers** next to the originals with a timestamp — the revert is one `cp`.
- **Restart the app** as part of the test. It may resolve the launcher path once at startup.

**Done when:** the app's child process is an `ssh -T …` hop and a real in-app turn answers.

## Step 2 — the tool bridge (the app's own MCP server)

Anything the app exposes through a local socket or app process cannot run on the remote host. Tunnel
the MCP server over the SSH stdio pipe — JSON-RPC over stdio needs no other glue. See
`scripts/mcp-bridge.sh.template`.

- **Quote the remote path inside the remote command.** `ssh host /path with spaces` splits on spaces
  and fails (`zsh: no such file or directory`). Put the whole hop in a script and reference the script
  from the MCP config instead of inlining it.
- Use a **dedicated key** (`ssh-keygen -f ~/.ssh/<name>_bridge`) with a recognisable comment in the
  app host's `authorized_keys`. `BatchMode=yes` makes failures loud instead of prompting.
- The app's own wrapper script already sets the env the server needs — reuse it, don't recreate its
  env by hand.
- **Do not expect tool parity.** An sshd-spawned client is identified by process lineage, not client
  name; setting the app's client-id env vars did not change the surface in testing. Assert the tools
  the workflow needs are present, and say plainly what is missing rather than claiming parity.

**Done when:** a handshake through the bridge returns the expected tool list, and one real tool call
returns live app state (not a cached or empty result).

## Step 3 — verify with the client's own logs, cross-checked

Client-side "it answered" is not proof the remote ran. Three independent signals:

1. **Remote session store** — a new session tagged with the app's source (`source=acp`), the right
   model, and a `cwd` that is a path from the *app host* (proof the remote understood the app's
   session params).
2. **Client log's process id** — the client logs a `processPid`; that PID must be the `ssh` hop in the
   process tree, under the app's own PID.
3. **Session-id cross-match** — the id in the client's stderr log must equal the id stored remotely.

Also assert the **negative**: the old local profile's `state.db` mtime must not have moved.

**Done when:** all three agree and the negative holds.

## Step 4 — make it survive updates (event-driven watchdog)

A bridge a future update can silently undo is not finished work. First find out what actually
overwrites your interception point:

- Read the runtime's installer. One real case: `scripts/install.sh` did `rm -f` + `cat >` on the
  launcher, so a full re-install wiped the shim, while the self-update path skipped existing files and
  was safe. Never answer "will it survive an update?" from reasoning — read the installer and name
  which path breaks it.

Then wire the **update itself** as the trigger — launchd `WatchPaths` on macOS, not cron. An event
beats a schedule: it fires exactly when the thing changed, and costs nothing when it didn't.

Watch **the app bundle directories, not only the files inside them**. A bundle swap (new directory
+ rename) produced no event for file-only paths and the check never ran; adding the `.app` and its
`Contents` directory fixed it.

The watchdog must re-verify the whole contract, not just your own files:

1. **Restore** the launcher bridge if something overwrote it.
2. **Re-read the app's bundle** and assert it still launches an intercepted binary with the expected
   argv.
3. **Assert the tool surface**: query the app's MCP wrapper for `tools/list` and fail if any tool the
   workflow needs is gone. A real update kept the launcher identical while deleting the wrapper tool
   the profile's instructions named (`execute`) and widening 3 tools to 57 — nothing else would have
   noticed. Treat "app not reachable" (no project open, app closed) as a note, never an anomaly.
4. **Check the remote profile** is still reachable over SSH.
5. **Alert where the user actually looks** — a messenger, not a log file.

See `scripts/watchdog.sh` and `templates/` for a working starting point. Gate it on a marker file
(`ENABLED`) so an intentional revert is not fought, and keep it idempotent and fast.

**Done when** you have proven all three by execution, not by reading:

- overwrite the launcher with the installer's exact content, watch it repair;
- `touch` a watched app file with nobody running anything and see the job fire;
- force an anomaly and **read the delivered alert back** from the messaging API (a zero exit code is
  not delivery).

## Pitfalls

- **After the bridge works, the next failure is usually context size, not the bridge.** Discovery
  tools can return enormous payloads (a real case: a 198 KB guidelines reply plus a ~90 KB project
  dump), inflating a session to ~90k tokens until the model lane went silent on the resulting
  non-streaming call — five stale attempts, turn aborted mid-edit. Diagnose from the *provider* errors
  in the captured agent stderr; do not re-test the bridge. Fix in the remote profile:
  `compression.threshold_tokens` set well below the model's window (a big-window model's 0.5-ratio trigger
  can sit so high it never fires), a `fallback_providers` entry, a per-provider
  `providers.<id>.stale_timeout_seconds`, and file-shuttle/paging rules in the profile's system
  prompt.
- **Know the timeout bands before calling a stall a bug.** The non-streaming stale ceiling is a
  function of estimated request size: ~150 s for 50–100k tokens, 240 s above 100k. A ~90k-token call
  timing out at 150 s is by design. Raising `stale_timeout_seconds` explicitly also exempts the value
  from the remaining-run-budget cap.
- **A wide tool surface arrives deferred.** With ~57 tools the agent pays a `tool_search` →
  `tool_describe` → `tool_call` tax per unfamiliar tool. Measure it (it was ~5% of tool calls) before
  trading it for the per-turn schema cost of loading everything eagerly.
- **Alert paths fail silently in two classic ways.** A non-interactive SSH shell has a minimal PATH,
  so the messenger CLI is "command not found" — call it by absolute path. And a channel-less send
  errors with "no home channel set" unless the channel is named explicitly. Both were live failures,
  not theory.
- **The app keeps writing locally.** Expect a local skills/profile mirror to be re-created at launch
  wherever the agent runs. It is inert; deleting it while the app runs risks writes into a
  half-deleted tree — delete only after the app is quit, and only if the user wants it.
- **The app passes host-local paths** (a workspace `cwd`) to the agent. Most runtimes store but do not
  `chdir` on it, so local file tools point at paths that do not exist remotely. Check on day one, and
  write the "what lives where" contract into the remote profile's system prompt so the agent stops
  re-deriving it every session.
- **Old conversations keep the old process.** Only a *new* conversation spawns the new bridge — say
  that explicitly when asking the user to re-test.
- **A handshake test passing does not mean the app uses the bridge.** Always finish with a real
  in-app turn plus the signals in Step 3.

## Verification Checklist

- [ ] The launched binary name and argv were read out of the app's own bundle, and the process tree
      confirms the `ssh` hop
- [ ] The wrapper is conditional: ACP goes remote, every other invocation stays local
- [ ] A real tool call through the MCP bridge returns live app state
- [ ] Remote session store, client `processPid`, and session id all cross-match; the old local
      profile's `state.db` mtime did not move
- [ ] Watched paths include the app bundle directories, not only files inside them
- [ ] The watchdog asserts launcher, bundle argv, tool surface, and remote reachability
- [ ] Launcher stomp repaired, `touch`-trigger fired, and the forced-anomaly alert was read back
- [ ] Context-size limits and a fallback provider are configured in the remote profile

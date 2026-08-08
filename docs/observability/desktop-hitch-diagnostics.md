# Desktop Hitch Diagnostics

An opt-in, local-only capture that records why the desktop app hitched, and
exports a sanitized bundle a reviewer can read. Nothing is uploaded, and
nothing is recorded until a capture is explicitly armed.

## The flow

1. Open **Settings → Diagnostics** in the desktop app.
2. Press **Start capture** *before* reproducing the hitch. The ring buffers
   hold roughly the last five minutes, so starting a little early is free and
   starting late loses the event.
3. Reproduce the hitch.
4. Press **Stop & export**. This gathers all three streams, writes the bundle,
   and shows the directory path plus the classification.

There is no separate export step. Stopping *is* exporting.

While disarmed the capture costs nothing: no observers are registered in the
renderer, no metrics sampler runs in the main process, and the gateway's
heartbeat stays at its normal cadence.

## What is captured

Three streams, one per process, each stamped on that process's own monotonic
clock and aligned at export against a shared wall-clock anchor.

| Stream | Source | Records |
| --- | --- | --- |
| `renderer-<windowId>` | each chat window | long animation frames (with attribution), heap samples, stream-delta flush costs, costly gateway-event dispatches |
| `main` | Electron main | window responsive/unresponsive edges, per-process CPU + working set, backend transport failures |
| `gateway` | the Hermes backend | event-loop heartbeat drift with sampled stacks, slow WS writes |

The gateway stream is pulled over the gateway's existing authenticated channel.
It is available only for a locally-spawned backend; a remote/SSH gateway is not
asked, and the stream is marked absent with reason `remote-gateway`.

## What the bundle contains

`<userData>/diagnostics/<capture-id>/`

| File | Contents |
| --- | --- |
| `manifest.json` | capture id, clock anchors, app version, platform, process tree, stream index |
| `classification.json` | the labels and the evidence behind them |
| `renderer-<windowId>.jsonl` | one JSON event per line |
| `main.jsonl` | one JSON event per line |
| `gateway.jsonl` | one JSON event per line; absent when the gateway stream is |

Every event carries its original monotonic `t` (or `t_monotonic`) plus an added
`wall_clock_ms`, so the three streams sort into one timeline without anyone
having to trust a second machine's `Date.now()`.

### Renderer events for multi-thread attribution

Two renderer event details exist specifically to attribute hitches when more
than one chat/thread runs at once:

- `stream_delta_applied` rows carry `path` (`timer` for the batched flush,
  `eager` for a drain forced by an ordering-sensitive event such as a tool row
  or turn completion — under an agentic turn most applies run eagerly) and
  `busySessions` (sessions with a turn in flight at record time, not just the
  ones with text queued in that flush). Eager rows have `commitMs`/`rafGapMs`
  fixed at 0 — they skip the commit-measurement frame.
- `gateway_event_applied` records any single gateway-event dispatch whose
  synchronous main-thread cost reached 4ms — tool-row upserts, subagent
  progress, session.info patches, terminal chunks — with `eventType` (the type
  tag only, never payload), `durationMs` and `busySessions`. Cheaper
  dispatches are deliberately not recorded.

Reading a multi-thread capture: correlate long_frame bursts with
`busySessions > 1`, then look at which `gateway_event_applied.eventType` and
`path: 'eager'` rows cluster inside the bursts.

## Sanitization contract

Sanitization happens at **record** time, in each process, not at export time.
Only counts, sizes, durations and opaque ids are ever written into a ring
buffer, so a buffer that escapes some other way (a crash dump, a core file)
cannot leak content either.

The bundle therefore contains **no**:

- message text, prompts, model output or tool output
- credentials, tokens or API keys
- file paths, worktree paths or full URLs (REST routes are truncated to a
  two-segment prefix, because Hermes paths carry session ids)
- process command lines or argv

**The manifest is inside this contract.** Its `process_tree` records `pid`,
`ppid` and the executable **basename** only. Entries are rebuilt field by field
rather than filtered, so a future probe that starts reporting a command line
cannot leak it through.

Worth spot-checking before you attach a bundle anywhere: grep it for a phrase
you know you typed during the capture. It should not be there.

## Classification

`classification.json` labels the capture against the five mechanisms, using
threshold heuristics over the streams. Multiple labels are normal and useful —
`labels` is ordered strongest-first and `primary` is simply the first.

| Label | Fires on |
| --- | --- |
| `renderer-bound` | ≥3 long frames ≥50ms, or any single frame ≥300ms |
| `gateway-bound` | ≥2 loop-drift stalls ≥0.25s, or any single stall ≥1s |
| `ipc-transport-bound` | ≥2 backend transport failures, or any single timeout |
| `memory-gc-bound` | renderer heap grew ≥150MB, or peaked ≥85% of its limit |
| `history-bound` | commit cost above the median transcript length is ≥2× the cost below it, and ≥40ms |
| `unclassified` | nothing crossed a threshold |

Each fired label carries a `reason` in plain terms and the `evidence` numbers it
was computed from, so a threshold that looks wrong can be argued with against
the raw JSONL sitting beside it.

## `hermes debug diagnose`

The host-side companion, for the questions that cannot be answered from inside
one Electron process.

```
hermes debug diagnose                      # process tree only
hermes debug diagnose --wpr                # + a 30s WPR trace (Windows, elevated)
hermes debug diagnose --wpr --wpr-seconds 60 --out ./capture
```

It writes `diagnose.json` containing:

- the **Hermes process tree** — pid, ppid and executable basename, following
  the same rule as the desktop manifest. Command lines are never read.
- the **WPR trace status**.

Run it alongside a desktop capture, not instead of one: it has no access to the
in-app rings.

### ⚠ The WPR trace is UNSANITIZED and unsafe to share

WPR (Windows Performance Recorder) records the **whole machine**, not just
Hermes. An ETL trace contains other applications' activity, full image paths,
and **command lines** — which on a developer machine routinely carry API keys
and tokens.

For that reason:

- it is produced only on the explicit per-invocation `--wpr` opt-in;
- it is written to a separate `unsafe-to-share/` subdirectory, never into a
  sanitized bundle;
- that directory gets a `README.txt` repeating this warning.

Open the ETL locally in Windows Performance Analyzer, or delete it. Do not
attach it to a bug report, a paste, or a chat message.

The trace is bounded in both directions: a fixed recording duration, and a hard
timeout on each `wpr` invocation after which the child is killed and the
system-wide session is cancelled. If `wpr.exe` is missing (non-Windows host, no
Windows Performance Toolkit) or the shell is not elevated, the trace is skipped
with a note and the process tree is still written.

## Related

- `apps/desktop/src/diagnostics/` — renderer ring + arming bridge
- `apps/desktop/electron/diagnostics-capture.ts` — capture controller
- `apps/desktop/electron/diagnostics-export.ts` — bundle writer + sanitization
- `apps/desktop/electron/diagnostics-classify.ts` — the heuristics above
- `hermes_cli/diagnostics_ring.py` — gateway-side ring
- `hermes_cli/diagnostics_diagnose.py` — the CLI subcommand

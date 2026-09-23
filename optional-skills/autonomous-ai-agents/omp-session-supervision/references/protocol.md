# Local event protocol, version 1

## Ownership and lifecycle

The CLI creates a private run directory named by a random 32-character lowercase hexadecimal run ID. `binding.json` is immutable and records version, run ID, canonical workspace, tmux session name, creation time and owner: canonical Hermes profile home, gateway platform, session routing key, Hermes session ID, chat ID and thread ID. Empty thread IDs are permitted. The owner is read from the current tool environment and compared exactly on every CLI action. Platform names are identity, not commands or a mechanism to choose a new notification destination.

The OMP extension receives only the binding file path through `OMP_HERMES_BINDING_FILE`. Without it the extension is a no-op. On interactive `session_start`, it binds one OMP session and generates a random epoch. It never takes over an existing socket/journal or transfers the binding to another OMP session. Changes revoke the original enrollment. Shutdown closes observation without terminating OMP itself.

Directories are owner-only (0700); state files and Unix socket are 0600. Binding reads reject unsafe ownership, permissions, symlinks, hard links, special files and excessive size. The Python observer checks socket peer UID. Same-UID processes remain trusted; this is not a cross-user sandbox or authenticated network protocol.

## Subscription

Both implementations accept platform identifiers matching `[A-Za-z0-9][A-Za-z0-9_.-]{0,63}` and preserve them exactly. There is no hardcoded list of chat platforms, but actual wake delivery still depends on the native host adapter.

The client sends one UTF-8 JSON line, at most 4096 bytes:

```json
{"version":1,"type":"observe","run_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","after_seq":0}
```

On reconnection, include the known `epoch` and last consumed `after_seq`. No prompt, destination, steering, cancellation or extra fields are allowed.

The server responds with `hello`: version, type, run ID, epoch, OMP session ID, PID, current sequence and state (`idle`, `busy`, `revoked` or `closed`). Events are separate JSON lines with exactly version, type=`event`, run ID, epoch, sequence, OMP session ID, event kind and timestamp in milliseconds.

Event kinds:

- `started`: a new agent cycle began; observers do not wake on routine starts.
- `turn_settled`: the latest turn ended with neither automatic continuation nor queued messages. This is not a claim of task success.
- `error`: the latest relevant assistant outcome was an error. Historical errors do not contaminate successful later turns.
- `needs_input`: a documented OMP tool approval was requested; this does not cover all custom UI interactions.
- `session_revoked` / `shutdown`: the original session changed or closed.
- `observation_lost`: a local observer could not preserve continuity. It records a local receipt; invalid clients do not inject a global failure into healthy subscribers.

A stale epoch, absent epoch for a nonzero cursor, future cursor or retention gap is rejected on that connection only, after hello:

```json
{"version":1,"type":"rejected","run_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","reason":"invalid_cursor"}
```

The other rejection reason is `epoch_mismatch`. Rejection does not change shared history or other subscribers. Malformed/oversized requests are disconnected. Client count and queued output are bounded; slow readers cannot stall the OMP event loop indefinitely.

## Persistence and delivery semantics

The extension atomically persists up to 128 sanitized events in `journal.json` before publishing them. It does not persist prompts, model messages, tool inputs/results, error strings or credentials. Persistence failure closes observation without claiming an uncommitted completion. Intermediate tool failures are not a general human-input detector.

A single observer holds a file lock. It replays unseen events, verifies identity/sequence continuity, and durably advances `cursor.json` before emitting one actionable receipt and exiting. The native Hermes terminal supervisor converts that process exit into a session-owned wake. The protocol itself never posts to a chat API and cannot verify that the caller really requested native notification; the calling workflow must check the actual terminal result.

Cursor advancement is not a delivery acknowledgment. A crash between persistence and host/platform delivery is ambiguous; never erase a cursor to force replay. Gateway recovery may lose stdout even when process metadata survives. Reconcile local receipts with host delivery records instead of promising exactly once.

The launcher reserves `launch.json` before starting tmux. A failed or timed-out launch is not retried automatically. Monitoring failure is not permission to kill, restart or cancel the worker.

Ordinary watcher expiry emits `observation_lost` with reason `timeout` but leaves the cursor nonterminal. The same owner can re-arm without resetting epoch or sequence; replay and socket validation still reject any subsequent retention gap or identity change. Revocation, shutdown and all other observation-loss reasons remain terminal. No timeout receipt terminates OMP.

# Local cross-agent continuity pilot

This pilot gives Claude, Codex, and Hermes the same bounded orientation without
making a memory system the authority for completion.

## Authority map

| Authority | Purpose | Canonical pilot location |
| --- | --- | --- |
| Basic Memory | Under-1,200-word human orientation card | External `continuity/hermes-agent/current.md` |
| Beads | Exact active work item | External `issues.jsonl`, task `hermes-continuity-b6l` |
| Spec Kit | Intended behaviour and lifecycle | `specs/001-global-continuity-pilot/` |
| Gate receipt | Exact-state test/runtime/rollback proof | External `receipts/` |

Preflight reads all four relevant surfaces, verifies agreement, and emits no more
than 8,000 characters. A Beads CLI timeout falls back to readable JSONL but marks the
result `DEGRADED` and forbids completion. Missing or conflicting state is `BLOCKED`.
Receipt creation separately runs a live, longer-bounded `bd show` check; JSONL alone
can never authorize a terminal lifecycle state.

The human CLI preflight retains the card's next action, blockers, and exact diagnostic
values. Host adapters never place those strings into model context. The dispatcher
builds a closed signal-only projection containing counts, booleans, and allowlisted
lifecycle values; invalid card, task, path, branch, and error strings cannot become
model or tool authority.

Claude and Codex finalization hooks return their native blocking shape when preflight
forbids completion. Hermes `pre_llm_call` validates native conversation history and
writes a content-free, per-session and per-turn adjacency admission outside the
repository. Every `pre_tool_call` in that turn checks the same admission and is
installed `fail_closed: true`; a missing, stale, blocked, or cross-turn guard prevents
tool side effects. Hermes shell
hooks do not have a blocking ordinary-prose or session-end event; for that surface the
exact-state gate, not advisory context, prevents `TESTED`/`ENFORCED` promotion.

## Tool pins and isolation

- Basic Memory `0.22.1`
- Beads `1.0.4`
- Spec Kit `1.0.1`
- State root: `/Volumes/Storage Unit/Application-Data/Continuity-Pilot/hermes-20260824`
- Worktree: `/Volumes/Storage Unit/Application-Data/Codex/worktrees/hermes-continuity-pilot-20260824`

The Basic Memory Hermes transcript-capture plugin is deliberately not enabled. The
pilot does not alter `~/.hermes`, `~/.claude`, or `~/.codex`.

## Operator commands

### Native-containment platform boundary

Routine preflight, adapter, and static-contract commands remain available on the
designated macOS pilot host. Receipt evidence commands and exact native host
observation have a stronger boundary: the controller must be able to reap descendants
even if a child creates a new session and clears its environment. Windows satisfies
that contract with a kill-on-close Job Object and Linux uses a nested child subreaper
with bounded adopted-generation cleanup and an explicit cleanup acknowledgement.

Native Windows proof is CI-owned, not inferred from the macOS run. A Python change
activates `.github/workflows/ci.yaml`'s OS-specific workflow, whose Windows lane runs
the repository's `windows_only` set on a real Windows runner and fails if selection is
empty. Until that lane passes for the pushed SHA, the Job Object contract is not
release evidence.

macOS has no equivalent supported unprivileged primitive. Process groups, process
snapshots, environment family tokens, Seatbelt, and launchd jobs can all be escaped by
a sufficiently fast `setsid` child. The controller therefore fails closed before
launching receipt evidence, Claude, Codex, or Hermes on macOS until an approved
privileged helper exists. Consequently, `TESTED` and `ENFORCED` promotion remain
unavailable on the designated Mac rather than recording evidence whose cleanup
guarantee cannot be proved. Ubuntu CI is still static evidence and is not a substitute
for the missing exact-host observation.

```bash
python3 scripts/continuity_bridge.py --config .continuity/config.json --cwd "$PWD"
python3 scripts/install_continuity_adapters.py --repo-root "$PWD" --check
python3 scripts/continuity_gate.py static --config .continuity/config.json
scripts/run_tests.sh tests/test_continuity_control_plane.py
```

To prove native Hermes registration without touching the live profile:

```bash
python3 scripts/install_continuity_adapters.py \
  --repo-root "$PWD" \
  --hermes-home "/Volumes/Storage Unit/Application-Data/Continuity-Pilot/hermes-20260824/hermes-home/.hermes" \
  --apply-hermes
HERMES_HOME="/Volumes/Storage Unit/Application-Data/Continuity-Pilot/hermes-20260824/hermes-home/.hermes" \
  hermes hooks list
HERMES_HOME="/Volumes/Storage Unit/Application-Data/Continuity-Pilot/hermes-20260824/hermes-home/.hermes" \
  hermes hooks doctor
python3 scripts/continuity_native_observation.py \
  --surface hermes --config .continuity/config.json \
  --hermes-home "/Volumes/Storage Unit/Application-Data/Continuity-Pilot/hermes-20260824/hermes-home/.hermes"
```

After the dispatcher file changes, refresh the isolated hook approvals before relying
on them; `hooks doctor` reports the exact stale approval and verifies each synthetic
event, including the expected exit `2` from a blocked `pre_tool_call`.

## Receipt shape

The committed T3 policy owns the focused suite, full suite, runtime checks, and
rollback dry-run as closed manifests. Caller JSON must match that policy byte-for-byte
as parsed data; it cannot select commands, surfaces, labels, or timeouts. The gate
requires the canonical committed config, a clean repository, and the configured
integration ancestry before it starts any evidence command. It resolves only pinned
Python or `/bin/bash` entry points and repository scripts, runs them with a fixed PATH,
credential-minimized environment, and a separate evidence HOME, then records exit
code, parsed positive test count, timestamps, duration, working directory, command
identity digest, and output digest. Raw argv and output are not persisted. The trusted
boundary is the reviewed code at that clean exact SHA; this pilot is process-isolated,
not an OS sandbox for hostile repository code. Receipts are authenticated with a
mode-`0600` pilot-local HMAC key loaded before evidence starts. `ENFORCED` uses only the
committed full-suite identity. Any commit, dirty-state change, integration-ref advance,
executable-pin change, or external-input drift invalidates the receipt.

Routine support commands run in their own process group. Receipt evidence and native
observations require a non-escapable host primitive: a Windows Job Object or Linux
child subreaper with explicit cleanup acknowledgement. Timeout or interruption must
reap the whole descendant tree with bounded TERM-to-KILL escalation or fail the
evidence. The authenticated receipt also fingerprints the exact hashed requirements
lock, Python launcher and executable, and installed distribution metadata; a lock
mismatch or later resolved-environment change fails verification. Native runtime
records bind Claude, Codex, and Hermes
observations to the committed event and adapter path, every configured adapter
artifact digest, the post-execution host executable identity, the current commit, and
a five-minute freshness window. Missing-observation diagnostics obtain `last_success`
only from a prior authenticated PASS receipt. Hermes observation executes `hooks
list`, `hooks doctor`, and the native `hooks test pre_llm_call` admission path; its
installed hooks must equal the current generated contract, and both external
`config.yaml` and its ownership manifest are rehashed after execution and during
receipt verification. The ownership manifest itself has a closed schema and HMAC,
binds the complete generated hook set and exact before-image partition, and is
validated by install, observation, dry-run, and rollback. Claude and Codex observations launch the digest-pinned native
host CLIs with tools disabled, require the host-created session identity to match the
redacted `SessionStart` audit record, and stop the credential-free host after that
admission evidence appears instead of waiting on a downstream model request. Both the
host and the complete project adapter surface are then rehashed. Observer interruption
forwards a hard stop to the host's isolated process group before the observer exits;
POSIX termination signals remain masked across child spawn and handler installation so
there is no detached-child race window.
Codex uses a
temporary, external-state-root home that trusts only the exact project for the
observation; the vetted hook source is still required to match the committed generated
adapter. Every generated dispatcher command uses the repository's locked `.venv`
interpreter; system `python3` is not an admissible fallback.

Codex resolves project hooks from the root checkout of a linked Git worktree. The
observer therefore rejects linked worktrees outright: a worktree-only hardening commit
cannot satisfy the Codex native deadman until it is integrated into that canonical
checkout (or exercised from a standalone exact-head clone). This is an intentional
promotion blocker, not a fallback to the direct dispatcher path.

```bash
python3 scripts/continuity_gate.py create-receipt \
  --config .continuity/config.json --cwd "$PWD" --risk-tier T3 --target TESTED \
  --commands-json /external/evidence/commands.json \
  --runtime-json /external/evidence/runtime.json \
  --rollback-json /external/evidence/rollback.json \
  --output /external/receipts/tested.json
python3 scripts/continuity_gate.py verify-receipt \
  --config .continuity/config.json --cwd "$PWD" --receipt /external/receipts/tested.json
```

`promote` is the only supported route to `TESTED` or `ENFORCED`. It updates the
external card, leaving the committed active specification unchanged so promotion does
not invalidate its own exact-state fingerprint. `ENFORCED` also closes the Beads item;
promotion is serialized with an external file lock and re-reads the task after close.
If any `PREPARED`, `CARD_WRITTEN`, or `RECOVERY_REQUIRED` stage is interrupted,
preflight blocks and an idempotent retry with the same receipt resumes the transition.

## Rollback

Validate or apply the exact Hermes-hook before-image without touching unrelated hooks:

```bash
python3 scripts/install_continuity_adapters.py \
  --repo-root "$PWD" --hermes-home "$HERMES_HOME" --rollback-dry-run
python3 scripts/install_continuity_adapters.py \
  --repo-root "$PWD" --hermes-home "$HERMES_HOME" --rollback-apply
```

The installer keeps a mode-`0600` ownership manifest inside the isolated Hermes home.
Rollback refuses to proceed if a managed hook changed after installation. Broader
pilot retirement is recoverable and scoped:

`--rollback-apply` is acknowledgement-loss safe: repeating it after a successful
rollback returns success with `already_rolled_back: true` and makes no file change.
If a managed hook drifted after rollback, the command names the conflicting hook keys
and gives the recovery sequence.

1. Verify the shared checkout and live global hook files are unchanged.
2. Apply the adapter rollback while its repository tool still exists.
3. Remove this isolated worktree through `git worktree remove` only after preserving
   any desired branch/receipt evidence, then move the isolated external pilot directory
   into an archive on the Storage Unit; do not delete it during the pilot.
4. Delete the pilot branch only after confirming no evidence is needed.

Because the live host profiles are not modified, rollback does not require restoring
global configuration or a Hermes database.

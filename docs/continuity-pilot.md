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

1. Verify the shared checkout and live global hook files are unchanged.
2. Apply the adapter rollback while its repository tool still exists.
3. Remove this isolated worktree through `git worktree remove` only after preserving
   any desired branch/receipt evidence, then move the isolated external pilot directory
   into an archive on the Storage Unit; do not delete it during the pilot.
4. Delete the pilot branch only after confirming no evidence is needed.

Because the live host profiles are not modified, rollback does not require restoring
global configuration or a Hermes database.

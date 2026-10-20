# Hermes Continuity Constitution

## Core Principles

### I. Separate the authorities

Basic Memory records a short human-readable orientation card, Beads records the
active unit of work, Spec Kit records intended behaviour, and exact-state receipts
record proof. No authority silently substitutes for another. A disagreement is a
blocking conflict, not a reason to choose whichever source is convenient.

### II. Evidence controls lifecycle state

The allowed lifecycle is `PROPOSED -> ACTIVE -> IMPLEMENTED -> TESTED -> ENFORCED`
with `BLOCKED` available from any non-terminal state. Agents may propose or implement,
but only the repository gate may emit `TESTED` or `ENFORCED`. A receipt is valid only
for the repository root, branch, commit, working-tree state, runtime, and commands it
records. Any later mutation expires it.

### III. Local, bounded, and private by default

Continuity state lives on `/Volumes/Storage Unit`, uses pinned local tools, and does
not require a subscription or cloud service. The continuity card is under 1,200
words. Tool inputs, tool results, transcripts, reasoning, secrets, and customer data
are never copied into the card or event log. If the external volume is unavailable,
storage-dependent writes stop.

### IV. One dispatcher, fail readable

Claude, Codex, and Hermes adapters call one repository dispatcher. Preflight uses
bounded subprocess timeouts and emits no more than 8,000 characters. Missing,
corrupt, stale, or conflicting authorities produce a concise degraded or blocked
result and set `completion_allowed` to false. Hooks must not conceal a failure or
create an unbounded chain of global hooks.

### V. Test the recovery path

Every change follows `spec.md`: baseline, failure proof, focused test, neighbour and
failure-path tests, real-path integration, applicable release gate, exact-state
rebind, and rollback proof. The pilot must cover wrong folder, missing drive, stale
task, missing or corrupt card, incomplete spec, failed tests, changed SHA,
interrupted tool adjacency, and clean fresh-session recovery.

## Constraints

- The existing Hermes memory providers and prompt-caching contract are unchanged.
- No live `~/.hermes`, `~/.claude`, or `~/.codex` configuration is mutated by the
  pilot; Hermes native integration is rendered only into the isolated pilot home.
- The shared dirty checkout is preserved. Implementation and validation use the
  isolated worktree named in `.continuity/config.json`.
- Python checks run through `scripts/run_tests.sh`; direct `pytest` is forbidden.
- Third-party tools are pinned and stored outside the repository.

## Governance

`AGENTS.md` governs architecture, `spec.md` governs safe change, and this constitution
governs the continuity control plane. Amendments require a documented rationale,
migration and rollback impact, updated fixtures, and a version bump. Pull requests
must identify the risk tier and attach a current receipt or explicitly state why the
change remains pre-evidence.

**Version**: 1.0.0 | **Ratified**: 2026-08-24 | **Last Amended**: 2026-08-24

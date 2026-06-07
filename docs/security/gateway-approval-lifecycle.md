# Gateway Approval Lifecycle

This document describes how the gateway proposes, persists, displays, and
consumes user approvals for dangerous commands.

It is the canonical developer reference for the design implemented in
`tools/approval_store*.py` and the call-sites in `tools/approval.py` and
`gateway/run.py`. **If you are tempted to "simplify" approval storage
back to an in-memory dict, read this document first — the original
design did exactly that, and we replaced it because it could not
satisfy the security invariants below.**

## TL;DR

```
classify command
    │
    ▼
create ApprovalProposal (pin policy + risk + reason + defaults)
    │
    ▼
SqliteApprovalStore.submit  ─── durable row, status=pending
    │
    ▼
notify user via gateway (display_text includes approval_id + HIGH RISK if applicable)
    │
    ▼
(user reads it) ──► /approve <id>  or  /deny <id>
    │
    ▼
resolve_gateway_approval_by_id():
    1. store.get(id)              — proposal exists + session-key match
    2. store.consume(id) / .deny  — atomic transition (BEGIN IMMEDIATE)
    3. find matching _ApprovalEntry by id, set entry.event
    │
    ▼
blocked execution thread wakes
    │
    ▼
Phase 3 guard: re-classify command at runtime
    │ if runtime is STRICTER than pinned → choice = 'deny' (fail closed)
    ▼
execute under pinned payload  ──or──  return BLOCKED
```

## Security invariants (non-negotiable)

These are tested in `tests/tools/test_approval_store_contract.py`,
`test_approval_store_sqlite.py`, `test_resolve_by_id.py`,
`test_phase3_runtime_guard.py`, `test_phase4_high_risk_ux.py`, and
`test_failclosed_paths.py`.

1. **Pinned wrapper-side policy.** The risk/policy decision used at
   approval execution is the one computed at proposal-creation time and
   persisted to SQLite. The executor never re-classifies the command
   from the raw string to decide whether to allow it. (See
   `ApprovalProposal.risk_level` + `_classify_pattern_risk`.)

2. **Deterministic across process boundaries.** Two gateway processes
   sharing the same `~/.hermes/state.db` see the same proposal. A
   proposal created by process A can be inspected (or consumed, with
   correct semantics) by process B.

3. **Atomic consume, exactly-once.** `BEGIN IMMEDIATE` plus
   `UPDATE … WHERE status='pending' AND (expires_at IS NULL OR
   expires_at > ?) RETURNING payload_json` is the only state
   transition. The UPDATE matches at most one row; concurrent consumers
   (same process, different threads, or different processes entirely)
   serialise on SQLite's file lock and exactly one wins.

4. **Stub contract.** `ApprovalStoreContract` is a shared test mixin
   applied to every backend implementation. A new backend cannot land
   without passing — or explicitly `xfail(strict=True)`-marking — the
   full contract. This prevents tests from using a "fake" approval
   store that quietly accepts unsafe behavior the real store rejects.

5. **High-risk default-deny + explicit UX.** When a command is
   classified `high` the prompt MUST include `🚨 HIGH RISK`,
   `Default: DENY`, the exact command, cwd/backend context, pinned
   risk reason, the approval_id, and either an inline diff/summary
   or a literal `NOT AVAILABLE — approve only if you have
   independently reviewed` warning. There is no silent-render path.

6. **No FIFO.** `/approve` without an `<id>` is rejected even when
   exactly one proposal is pending. This prevents "approve whatever's
   at the head of the queue" footguns — users must read the id from
   the specific approval request.

7. **Fail closed on ambiguity.** Every ambiguous state — missing id,
   wrong id, expired, denied, already consumed, store unavailable,
   payload corrupt, classifier raised, pinned policy fields
   missing — results in **no execution**. There is no path where an
   approval flow that the wrapper could not fully verify leads to a
   running command.

## Why SQLite

The original storage was a module-level `dict[str, dict]` in
`tools/approval.py` guarded by `threading.Lock`. That model fails
invariants 2 and 3:

| Failure mode | In-memory dict | SQLite store |
|---|---|---|
| Proposal survives gateway restart | ❌ | ✅ |
| Visible to a second gateway process | ❌ (each has its own dict) | ✅ |
| `consume(id)` is atomic cross-process | ❌ (each process has own lock) | ✅ (file lock + BEGIN IMMEDIATE) |
| Two concurrent `/approve <id>` give exactly one execution | ❌ in the cross-process case | ✅ |
| Audit trail of consumed/denied/expired transitions | ❌ (state lost on restart) | ✅ (status column) |

Threading.Lock + dict guarantees same-process atomicity, but only that.
The gateway is multi-process (HA, supervised restart, parallel
deployment targets) and the wrapper must hold even when the gateway
that proposed an approval is no longer the one resolving it. SQLite
gives us that without inventing a custom locking protocol.

(We use `BEGIN IMMEDIATE` and `UPDATE ... RETURNING`, not a "best
effort" JSON file with a sidecar lockfile. The race window with
lockfiles is non-trivially closeable and the spec calls out
"no lockfile handwaving" specifically.)

## Components

| Component | Purpose |
|---|---|
| `tools/approval_store.py` | `ApprovalStore` Protocol + `ApprovalProposal` frozen dataclass + `ApprovalStoreError`. The contract. |
| `tools/approval_store_memory.py` | `InMemoryApprovalStore` reference impl. Process-bound by design — used as a documented gap-marker and for tests that genuinely don't need persistence. |
| `tools/approval_store_sqlite.py` | `SqliteApprovalStore` — the production backend. Self-contained schema (CREATE TABLE IF NOT EXISTS), no coupling to `hermes_state.SCHEMA_VERSION`. |
| `tools/approval.py` | Wires the store into `_await_gateway_decision`. Defines `_ApprovalEntry` (the in-process wake-up object), `_classify_pattern_risk`, `_classify_runtime_risk` (Phase 3), `_render_approval_display_text` (Phase 4), and the per-id resolver `resolve_gateway_approval_by_id`. |
| `gateway/run.py` | `_handle_approve_command` and `_handle_deny_command` — the slash-command parsers. Both require an explicit `<id>`; no FIFO. |

## The role of `_ApprovalEntry.event`

`threading.Event` is **only** the in-process wake-up mechanism for the
blocked agent thread. It is NOT the security boundary. The gate is
`store.consume(id)`. The event is signalled only AFTER the store has
irrevocably recorded the transition. If you ever find yourself making
a code path that signals the event without first going through
`store.consume` / `store.deny`, you are reintroducing the in-memory
source-of-truth that this rewrite specifically replaced — stop and
read this doc again.

## Adding a new approval-gated action

1. Make sure the command is classified by `detect_dangerous_command`
   (or the hardline gate). The existing classifier description string is
   what `_classify_pattern_risk` consumes to assign risk_level.
2. If your new pattern is high-risk (data loss, system-file overwrite,
   raw-device write, RCE-tier, privilege escalation), make sure its
   description contains one of the keywords in
   `_HIGH_RISK_DESCRIPTION_MARKERS`, or extend that tuple.
3. Verify with the parametrised
   `tests/tools/test_phase4_high_risk_ux.py::TestClassifyPatternRisk`
   table — add a row for your new description.
4. If you need to surface a diff/summary, set `diff_text` /
   `diff_summary` on the `approval_data` dict passed into
   `check_all_command_guards` / `_await_gateway_decision`. The renderer
   will pick them up automatically.

## What to test when you change this

| If you change… | …re-run at minimum |
|---|---|
| `ApprovalStore` protocol or any backend | `tests/tools/test_approval_store_*.py` |
| `_await_gateway_decision` | `tests/tools/test_approval_store_wiring.py` + `test_phase3_runtime_guard.py` + `test_phase4_high_risk_ux.py` + `test_failclosed_paths.py` |
| `_handle_approve_command` / `_handle_deny_command` | `tests/gateway/test_approve_deny_commands.py` |
| `_classify_pattern_risk` / `_classify_runtime_risk` | `tests/tools/test_phase4_high_risk_ux.py` + `test_phase3_runtime_guard.py` |

Targeted regression: ~260 tests, runs in seconds.

```
nix develop -c python3 -m pytest \
  tests/tools/test_approval_store_memory.py \
  tests/tools/test_approval_store_sqlite.py \
  tests/tools/test_approval_store_wiring.py \
  tests/tools/test_resolve_by_id.py \
  tests/tools/test_phase3_runtime_guard.py \
  tests/tools/test_phase4_high_risk_ux.py \
  tests/tools/test_failclosed_paths.py \
  tests/gateway/test_approve_deny_commands.py \
  tests/tools/test_approval.py \
  -o 'addopts=' --tb=short
```

Broader sweep (slower, run before integration is claimed done):

```
nix develop -c python3 -m pytest tests/ -k "approval or approve or gateway" \
  -o 'addopts=' --tb=short
```

## Out of scope (deliberately)

- **A "request more approval"-after-fail-closed path.** When the
  Phase 3 guard overrides choice → deny, the store row stays
  consumed/denied. The user must obtain a new approval with new
  pinned policy. We did this on purpose to keep exactly-once intact.
- **Approval rendering on every platform.** The renderer in
  `_render_approval_display_text` is the canonical text; rich platforms
  (Telegram inline keyboards, Slack blocks, Matrix HTML) may render
  the structured `risk_level`/`display_text`/`command`/etc fields
  themselves. The security boundary is the consume gate, not the
  rendering.
- **Tightened payload-integrity validation at submit time.** Today the
  store accepts proposals with empty `risk_reason` / default
  `risk_level='low'`. `test_failclosed_paths.py::test_consume_with_
  missing_required_fields_in_payload_does_not_execute` pins what gets
  serialised so a future tightening (e.g. `ApprovalStore.submit`
  rejecting empty `risk_reason`) is an intentional contract change,
  not an accidental regression.

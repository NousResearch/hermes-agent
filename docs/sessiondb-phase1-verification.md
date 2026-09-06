# SessionDB Provider Phase 1 — Concurrency & Crash-Recovery Verification Report

**Issue:** [NousResearch/hermes-agent#23717 — Pluggable SessionDB RFC](https://github.com/NousResearch/hermes-agent/issues/23717)
**Branch:** `sessiondb-provider-phase1` (local; not yet pushed)
**Date:** 2026-09-05 · **Host:** Windows 11 (Git Bash), CPython 3.13.15, repo test runner (`scripts/run_tests.sh`, CI-parity env: credentials unset, `TZ=UTC`, isolated `HERMES_HOME`, per-file subprocess isolation)

---

## TL;DR

Phase 1 (ABC extraction + SQLite provider + factory) is **behavior-preserving under every
concurrency and crash scenario the issue raises that is testable on a single host**:
4-process concurrent writers commit everything, a hard-killed mid-write process leaves a
consistent database with zero lost commits, 10–50 MB payloads survive concurrent appends,
FTS5 search and WAL pragmas are intact, and 437 existing session-storage tests still pass
(the only 7 failures reproduce identically on pristine `main` — pre-existing Windows
inode/file-locking limitations, unrelated to this change).

> **Scope of this verification:** every test here exercises the **abstraction layer plus the
> existing SQLite engine only**. No PostgreSQL (or MySQL) provider exists yet — those are
> Phase 2/3 of the RFC — so nothing in this report claims the ticket's underlying pain
> (single-writer serialization, NFS locking, hot-update file fragility) is *solved*. What it
> proves is narrower and is exactly what Phase 1 must guarantee: the pluggable architecture
> landed **without regressing** the concurrency and crash-recovery behavior SQLite has today,
> and the contract Phase 2 will implement against is pinned by executable tests.

---

## What Phase 1 changed (scope recap)

| Commit | Change |
|---|---|
| `bc72c1e9` | `test(state):` TDD harness (committed RED, before any implementation) |
| `8e659677` | `feat(state):` `SessionDBProvider` ABC (new `hermes_state_provider.py`); `SessionDB` conforms via `SQLiteProviderMixin`; `AsyncSessionDB` types against the ABC |
| `199988ff` | `feat(state):` `get_session_db_provider()` factory reading `sessiondb.provider` from `config.yaml` (default `sqlite`, loud failure on unknown names); wired through `hermes_state_registry._open_session_db` |
| `51a3bcae` | `feat(state):` `pre_commit`/`post_write` hooks + testing-only `_test_force_sigkill` seam in `_execute_write` (the single `BEGIN IMMEDIATE → fn → commit` funnel); `conn_factory` DI |
| `15f21a3f` | `fix(state):` LSP-exact ABC signatures (11 `invalid-method-override` errors resolved); registry type bridge |
| `8e0181c9` | `test(state):` multi-process concurrency + mid-write kill coverage |

Deliberately untouched: FTS5 SQL, WAL pragmas, retry/jitter logic, schema, and all 59
`SessionDB()` call sites (the factory returns the same class). No new env vars; config key
only. Kanban and the gateway's other SQLite stores are out of scope per RFC Decision 5.

---

## Concern-by-concern verification

### 1. "Hot-update death spiral" — process killed mid-write (Issue §1.1, §1.3; related #6702)

**Concern:** SIGTERM/SIGKILL while the old process is flushing messages or mid-WAL-checkpoint
corrupts `state.db`.

**Evidence A — real subprocess kill** (`tests/session/test_session_multiprocess.py::TestMidWriteProcessKill::test_sigkill_mid_write_leaves_database_consistent`):

- A worker process appends 8 MB payloads in a loop; the parent waits for two *acknowledged*
  commits (marker printed only after `append_message` returns), then hard-kills the child
  (`SIGKILL` semantics / `TerminateProcess` — no cleanup handlers run) while the next 8 MB
  write is in flight.
- A fresh provider handle (new connection → WAL recovery path) then asserts:
  `PRAGMA integrity_check` = `ok`; every acknowledged commit is durable
  (`message_count >= acknowledged markers`); the store accepts new writes.

**Result: PASS** (run 3×, stable — 7.2 s / 7.9 s runs).

**Evidence B — in-process crash seam** (`test_session_provider.py::TestCrashRecovery::test_mid_write_crash_is_atomic_and_recovers`):

- The testing-only `_test_force_sigkill` flag (set from a `pre_commit` hook) hard-closes the
  connection **inside the transaction boundary, without commit** — the durable state a real
  kill leaves. The crashed payload is fully absent (atomicity), committed history intact —
  verified through FTS5 search (`search_messages("before-crash")` hits,
  `search_messages("crashed-payload")` is empty, proving the FTS triggers rolled back with
  the transaction) — and the *same handle* recovers in place via the existing
  `_reopen_after_close_locked` path.

**Result: PASS.**

### 2. Multi-process write contention (Issue §1.2: CLI + gateway + cron + TUI + API on one `state.db`)

**Concern:** WAL serializes writers at the OS level; contention causes lock timeouts and lost writes.

**Evidence** (`TestMultiProcessContention::test_concurrent_process_writers_commit_everything`):

- 4 real worker processes × 20 messages × 1 MB each against one `state.db`, released
  simultaneously via a start-barrier file so writes genuinely overlap.
- Asserts `message_count` per worker session == 20 exactly (every commit landed, none lost,
  none duplicated), `session_count >= 4`, and `PRAGMA integrity_check` = `ok`.
- Workers exit 0 — the jittered-retry + `BEGIN IMMEDIATE` funnel absorbed all lock contention;
  no `database is locked` surfaced to a caller.

**Result: PASS** (3 consecutive runs).

### 3. Large-payload appends (Issue §1.3: "43 MB state.db at risk on every update")

**Evidence** (`TestPayloadConcurrency`):

- `test_concurrent_large_payload_appends`: 4 threads × 10 MB JSON payloads, barrier-started;
  all four rows present exactly once, **SHA-256 of each stored payload matches** the source
  (no truncation/corruption), provider usable afterwards.
- `test_single_oversized_payload_roundtrip`: one 50 MB JSON blob round-trips byte-identical.

**Result: PASS.**

### 4. `session_search` permanently disabled after lock timeout (#3139)

**Concern:** a lock timeout during concurrent CLI+gateway writes permanently disabled search.

**Evidence:** FTS5 is exercised after both the contended-burst and crash-recovery scenarios:

- Crash test asserts search finds pre-crash commits and *not* the rolled-back write.
- Live pragma probe on the branch (fresh provider via the factory):

```
journal_mode: wal
fts enabled: True | trigram available: True
fts5 search hit: True | snippet: >>>pragma<<< >>>probe<<< payload
integrity: ok
provider name: sqlite | is_available: True
```

**Result: PASS** — WAL mode, FTS5 + trigram tokenizer, and search syntax (note the FTS5
`>>>…<<<` snippet marks) are all preserved byte-for-byte.

### 5. Duplicate message accumulation (#860)

**Evidence:** every concurrency test asserts *exact* counts
(`message_count == MESSAGES_PER_WORKER`, `len(rows) == WORKERS`), so any duplicate-append
regression fails the suite.

**Result: PASS.**

### 6. File-descriptor exhaustion (#14210)

**Evidence:** the bounded read-pool regression tests
(`test_session_db_read_conn_pool.py`, `test_session_db_read_path_split.py`) pass unchanged —
the refactor added no new connection paths (the factory only substitutes construction; pool,
permits, and close semantics are untouched).

**Result: PASS (no regression).**

### 7. Regression net — "zero behavioral change" (RFC §5, Phase 1 criterion)

```
=== Summary: 55 files, 437 tests passed, 7 failed, 6 skipped (100% complete) ===
```

The 7 failures are **pre-existing Windows host limitations** (inode-replacement / file-locking
semantics in `TestInodeReplacement`, `test_state_db_file_identity.py`,
`test_state_db_corrupt_quarantine.py`). Baseline reproduced on a pristine `main` worktree
(`PYTHONPATH`-verified to import `main`'s code):

```
main:  7 failed, 22 passed, 1 skipped   ← identical failure set
```

**Result: PASS — the branch introduces zero regressions.**

### 8. Lint / typecheck standards

- `ruff check` on all touched files: **All checks passed!** (the repo enables `PLW1514` +
  `ASYNC21x/22x` only).
- `ty check` (this repo's typechecker — mypy is not configured): **zero new diagnostic
  classes** vs `main`. `hermes_state.py` actually improved by one (the `db_path: Path = None`
  → `Optional[Path]` fix cleared a pre-existing `invalid-parameter-default`). The new ABC
  file carries 27 `invalid-parameter-default` notes that mirror the house-style
  `str = None` annotations — 55 identical instances pre-exist across `hermes_state_*.py`
  (typecheck wrangling is repo-wide WIP per `pyproject.toml` comments).

---

## Explicitly NOT covered (honest scope notes)

- **NFS locking (#22032):** requires an NFS-backed host; SQLite's POSIX locking semantics
  can't be exercised on this Windows dev box. The refactor changes nothing on the NFS path
  (WAL fallback logic untouched), but NFS validation belongs to Phase 2's PostgreSQL
  provider, which is the actual fix for that issue.
- **True SIGTERM-during-checkpoint on a live gateway:** the kill test covers the mid-write
  case; a checkpoint-specific kill would need a staged WAL of meaningful size and is
  indistinguishable from the covered case at the durability layer (WAL replay handles both).
- **PostgreSQL/MySQL providers:** Phase 2/3 per the RFC — **no PostgreSQL instance was
  involved in any test above**; every result describes the abstraction layer over the
  existing SQLite engine only. Accordingly, none of the ticket's root pain points
  (single-writer serialization, NFS locking, file-level hot-update fragility) are claimed as
  *fixed* here — they are the reasons the Phase 2 PostgreSQL provider exists. Phase 1's
  factory fails loudly on unknown `sessiondb.provider` values by design (no silent fallback
  that would split a user's history across two stores), and this report's concurrency/crash
  suite doubles as the contract tests a Phase 2 provider must satisfy (a
  `PostgreSQLSessionDBProvider` should pass the same 15 tests against a real database).

---

## New test inventory (15 tests, all green)

```
tests/session/test_session_multiprocess.py
  TestMultiProcessContention::test_concurrent_process_writers_commit_everything
  TestMidWriteProcessKill::test_sigkill_mid_write_leaves_database_consistent
tests/session/test_session_provider.py
  TestInterfaceEnforcement (4)  — ABC strictness, SessionDB conformance, signature parity
  TestFactory (3)               — default sqlite, explicit config, loud failure on unknown
  TestPayloadConcurrency (2)    — 4×10MB concurrent, 1×50MB roundtrip
  TestCrashRecovery (1)         — mid-write crash atomicity + in-place recovery
  TestWriteHooks (2)            — pre_commit veto rolls back; post_write only after commit
  TestConnectionFactoryDI (1)   — conn_factory yields a temporary schema
```

## Reproduce

```bash
git checkout sessiondb-provider-phase1
scripts/run_tests.sh tests/session/          # 15 new tests
scripts/run_tests.sh tests/hermes_state/     # regression net
ruff check hermes_state.py hermes_state_provider.py hermes_state_registry.py \
         hermes_cli/config_defaults.py tests/session/
ty check  hermes_state.py hermes_state_provider.py hermes_state_registry.py \
          hermes_cli/config_defaults.py
```

## Conclusion

Phase 1 meets the issue's Phase-1 criteria — ABC extracted per the `MemoryProvider`
pattern, SQLite logic preserved behind it (WAL + FTS5 verified live), factory in place with
backward-compatible defaults — and the concurrency/crash behaviors the issue is concerned
about are now **pinned by executable tests** rather than assumed.

To be explicit about what this does and does not show: **all evidence above is SQLite-only,
behind the new abstraction.** The ticket's underlying motivation (a real multi-writer RDBMS)
is Phase 2 — this phase proves the refactor is safe to build on, and hands Phase 2 a
ready-made acceptance suite: any future `PostgreSQLSessionDBProvider` should pass these
same 15 tests against a live PostgreSQL instance before it ships.

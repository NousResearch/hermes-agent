"""Stage-3 acceptance harness invariants — single source of truth.

Every scenario in test_kanban_stage3_harness.py calls assert_holds() after its
disturbance. Invariant changes require their own review round (design §5).

The launch ledger stores lifecycle rows only (no timestamp or evidence
columns — terminal evidence is returned by record_run_outcome and the row is
retired), so the invariants operate on the row set, the stub spawn log, and
the API-returned evidence captured by the harness.
"""

from __future__ import annotations

import sqlite3
from typing import Iterable

SPAWNED_ONCE = "spawned_once"
NO_LEAKAGE = "no_leakage"
FAIL_CLOSED = "fail_closed"
SINGLE_AUTHORITY = "single_authority"
BOUNDED_EVIDENCE = "bounded_evidence"
MONOTONIC = "monotonic_lifecycle"

ALL_INVARIANTS = (
    SPAWNED_ONCE,
    NO_LEAKAGE,
    FAIL_CLOSED,
    SINGLE_AUTHORITY,
    BOUNDED_EVIDENCE,
    MONOTONIC,
)

_VALID_STATES = ("claimed_not_spawned", "spawn_intent", "spawned")
_STATE_ORDER = {"claimed_not_spawned": 0, "spawn_intent": 1, "spawned": 2}


def check_all(
    conn: sqlite3.Connection,
    spawn_log: Iterable[dict],
    evidence_log: Iterable[dict],
    *,
    quiesced: bool = True,
    max_detail: int = 256,
    lifecycle_log: Iterable[dict] | None = None,
) -> list[str]:
    """Return violation strings; empty means all invariants hold."""
    violations: list[str] = []
    rows = conn.execute(
        "SELECT task_id, run_generation, state FROM kanban_launch_protocol"
    ).fetchall()
    spawns = list(spawn_log)
    evidence = list(evidence_log)
    history = list(lifecycle_log) if lifecycle_log is not None else []
    # History supplements (not replaces) the spawn stub log: every stub
    # invocation is a real process-creation event.
    # Multiset comparison: an extra stub invocation for the same identity
    # that history already covers must still be flagged.
    from collections import Counter

    stub_counts = Counter(
        (c["task_id"], c.get("run_generation", 1)) for c in spawns
    )
    history_spawn_counts = Counter(
        (e["task_id"], e.get("run_generation", 1))
        for e in history
        if e.get("event") == "spawned"
    )
    unknown_spawns = stub_counts - history_spawn_counts

    # 1. no double spawn: at most one live lifecycle row per identity, and
    # the stub spawn_fn fired at most once per (task, generation).
    identities: dict[tuple, int] = {}
    for row in rows:
        key = (row["task_id"], row["run_generation"])
        identities[key] = identities.get(key, 0) + 1
    for key, n in identities.items():
        if n > 1:
            violations.append(f"{SPAWNED_ONCE}: {key} has {n} rows")
    spawn_counts: dict[tuple, int] = {}
    for entry in history:
        if entry.get("event") == "spawned":
            key = (entry["task_id"], entry.get("run_generation", 1))
            spawn_counts[key] = spawn_counts.get(key, 0) + 1
    for key, n in spawn_counts.items():
        if n > 1:
            violations.append(f"{SPAWNED_ONCE}: {key} spawned {n} times")
    # Any stub-invoked process not represented in lifecycle history is an
    # unrecorded spawn (e.g. record_spawned CAS failed after creation).
    for key, n in unknown_spawns.items():
        violations.append(
            f"{SPAWNED_ONCE}: {key} process created off-ledger x{n}"
        )
    # The stub must never fire more than once per identity, regardless of
    # what the history claims.
    for key, n in stub_counts.items():
        if n > 1:
            violations.append(f"{SPAWNED_ONCE}: {key} stub fired {n} times")
    # A spawn without a live-or-retired row path is impossible to observe
    # post-hoc; the scenario asserts counts directly instead.

    # 2. no leakage at quiesce
    if quiesced and rows:
        violations.append(f"{NO_LEAKAGE}: {len(rows)} rows remain")

    # 3. fail closed: a task aborted in this lifecycle must never have a
    # spawned event for the same identity.
    # Track event ORDER per identity: an abort is legitimate before a
    # spawn (freeze then unfreeze), but an abort AFTER a spawned event for
    # the same identity means a process was created and then the launch was
    # torn down / left unresolved — a fail-closed violation.
    seen_spawned: set[tuple] = set()
    for e in history:
        key = (e["task_id"], e.get("run_generation", 1))
        if e.get("event") == "spawned":
            seen_spawned.add(key)
        elif e.get("event") == "aborted" and key in seen_spawned:
            violations.append(f"{FAIL_CLOSED}: {key} aborted after process creation")

    # 4. single authority: states are only kernel-managed values
    for row in rows:
        if row["state"] not in _VALID_STATES:
            violations.append(
                f"{SINGLE_AUTHORITY}: {row['task_id']} state={row['state']}"
            )

    # 5. bounded evidence on recorded outcomes (lifecycle history carries
    # the actual record_run_outcome details)
    for entry in history:
        if entry.get("event") not in ("outcome_recorded", "aborted"):
            continue
        detail = entry.get("detail", "")
        if not isinstance(detail, str) or len(detail) > max_detail:
            violations.append(f"{BOUNDED_EVIDENCE}: {entry['task_id']}")

    # 6. monotonic lifecycle: any surviving row's state order matches the
    # spawn log ordering (claim before intent before spawned is enforced by
    # the kernel CAS; here we check the log never shows a later-stage event
    # for a task whose row is at an earlier stage).
    row_state = {(r["task_id"], r["run_generation"]): r["state"] for r in rows}
    seen_outcome: set[tuple] = set()
    for entry in history:
        key = (entry["task_id"], entry.get("run_generation", 1))
        if entry.get("event") == "outcome_recorded":
            seen_outcome.add(key)
        if entry.get("event") == "spawned":
            if key in row_state and _STATE_ORDER[row_state[key]] < 2:
                violations.append(f"{MONOTONIC}: {key} spawned log vs state")
            if key in seen_outcome:
                violations.append(f"{MONOTONIC}: {key} spawned after outcome")

    return violations


def assert_holds(
    conn: sqlite3.Connection,
    spawn_log: Iterable[dict],
    evidence_log: Iterable[dict],
    *,
    quiesced: bool = True,
    lifecycle_log: Iterable[dict] | None = None,
) -> None:
    violations = check_all(
        conn, spawn_log, evidence_log, quiesced=quiesced, lifecycle_log=lifecycle_log
    )
    if violations:
        raise AssertionError("; ".join(violations))

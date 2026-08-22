# MEM0 Controlled Write Preflight Revalidation Pattern

## When This Pattern Applies

Use this pattern whenever a mission requires **revalidating a controlled write approval packet before execution**. This is a safety gate that sits between "packet approved" and "execution authorized." It applies to any controlled write operation in the MEM0 pipeline (first pilot, second pilot, subsequent writes).

## Trigger Signals

The mission name contains `MEM0_*_CONTROLLED_WRITE_*_PREFLIGHT_REVALIDATION` or the user says "revalidate the controlled write packet before execution."

## Prerequisites

- Target commit checked out (or the approval packet commit is known)
- The approval packet JSON exists (typically in `docs/mcporter/MEM0_*_CONTROLLED_*_PILOT_APPROVAL_PACKET.json`)
- PYTHONPATH=v2 is set for cartelogic-v2 imports
- The controlled write pilot module (`v2/memory/abstraction/controlled_write_pilot.py`) is functional

## Revalidation Checklist (execute all steps, in order)

### 1. Understand the Commit

```bash
git log --oneline -5 <COMMIT>
git show --stat <COMMIT>
git show <COMMIT> --format=full
```

Document which files changed and what the commit's stated purpose is.

### 2. JSON Validation

Validate every JSON file in the commit diff:

```bash
python3 -m json.tool <path-to-json> > /dev/null && echo "VALID: <name>" || echo "INVALID: <name>"
```

Check for structural issues:
- Duplicate keys (JSON allows them, last wins — but flag as data quality issue)
- Missing required fields for operational_fact records (fact_id, fact_type, project, repo, mission, verdict)
- Any field that should be a string but appears as null

### 3. Hash Revalidation

The **critical** check. Recompute the dry-run plan from the approval packet's `test_fact` and verify both hashes match exactly:

```python
from memory.abstraction.controlled_write_pilot import (
    CONTROLLED_WRITE_APPROVAL_PHRASE,
    build_controlled_write_pilot_plan,
)

plan = build_controlled_write_pilot_plan({
    'candidate': { ... },           # from packet['test_fact']
    'target_store': test_fact['target_store'],
    'snapshot_target_path': test_fact.get('snapshot_target_path'),
    'approval_phrase': CONTROLLED_WRITE_APPROVAL_PHRASE,
    'live_write_requested': False,
})

assert plan['hashes']['before_hash'] == packet['hash_verification']['before_hash']
assert plan['hashes']['after_hash'] == packet['hash_verification']['after_hash']
```

Both hashes must match. If they don't, the packet is stale or the target state has changed — **do not proceed**.

### 4. Snapshot Semantics Verification

Test both target states independently using `build_controlled_write_snapshot_plan()`:

**Absent target:**
```python
plan = build_controlled_write_snapshot_plan(Path('/tmp/nonexistent/missing.json'))
assert plan['prewrite_state']['state'] == 'absent'
assert plan['prewrite_state']['exists'] is False
assert plan['prewrite_state']['content_hash'] is None
assert plan['snapshot_strategy'] == 'record_absence_marker'
assert plan['would_mutate'] is False
```

**Existing target:**
- Use the actual first-pilot fact (or any known-existing file)
```python
plan = build_controlled_write_snapshot_plan(Path('memory/fact_*_controlled_memory_write_*.json'))
before_bytes = target.read_bytes()
# ... calculate plan ...
assert plan['prewrite_state']['state'] == 'existing'
assert plan['prewrite_state']['exists'] is True
assert plan['prewrite_state']['byte_count'] == len(before_bytes)
assert plan['prewrite_state']['content_hash'] is not None
assert plan['would_mutate'] is False
# Verify file was NOT mutated by snapshot
assert target.read_bytes() == before_bytes
```

### 5. Rollback Plan Verification

Verify the rollback plan structure (from the recomputed pilot plan or the packet):

```python
rollback = plan['rollback_plan']
assert rollback['restore_mode'] == 'snapshot_or_local_copy'
assert rollback['restore_from_hash'] == before_hash
assert rollback['verify_after_hash'] == after_hash
assert rollback['verification_mode'] == 'compare_canonical_hashes'
assert rollback['would_mutate'] is False
```

### 6. Unit Tests

Run the controlled write pilot tests and the second-pilot packet test:

```bash
PYTHONPATH=v2 python3 -m pytest \
  v2/memory/abstraction/tests/test_controlled_write_pilot_phase9.py \
  v2/memory/abstraction/tests/test_second_controlled_write_pilot_packet_phase9.py \
  -v
```

All must pass. Document the count and status.

### 7. git diff --check

```bash
git diff --check <range>
```

Exit 0 = clean.

### 8. Verify First Pilot Fact Preservation

If revalidating a second (or subsequent) pilot, confirm the first pilot fact is unchanged:

```bash
ls -la memory/fact_mem0_first_controlled_memory_write_pilot_ready_*.json
wc -c memory/fact_mem0_first_controlled_memory_write_pilot_ready_*.json
```

Check mtime. If mtime has changed, the fact was modified — **flag immediately**.

### 9. Dry-Run State Verification

```python
assert plan['will_write'] is False
assert plan['would_mutate'] is False
assert plan['live_write_possible'] is False
assert plan['approval']['verified'] is True
```

### 10. Write and Commit Preflight Report

Write both a JSON report and Markdown companion following the existing `docs/mcporter/MEM0_*_PREFLIGHT_REVALIDATION_REPORT.*` naming convention.

JSON report fields:
- `fact_id`, `fact_type`, `record_type`, `project`, `repo`, `mission`, `verdict`
- `packet_commit` — the commit SHA being validated
- `hash_verification` with both recomputed and packet values + match booleans
- `snapshot_verification` with absent/existing targets
- `rollback_plan` — the verified plan
- `dry_run_state` — verified, will_write, would_mutate, live_write_possible, approval_verified
- `verification` — boolean map of every check performed
- `data_quality_observations` — any anomalies found (e.g., duplicate keys)
- `safety` — no_mem0_runtime, no_live_llm, no_embedding, no_qdrant, no_provider_keys, no_mutation, no_network, no_write
- `notes` — context and warnings

Commit only the report files:
```bash
git add docs/mcporter/MEM0_*_PREFLIGHT_REVALIDATION_REPORT.*
git commit -m "docs(memory): record <mission> preflight revalidation"
```

## Expected Verdict

- All checks pass → `<MISSION>_PREFLIGHT_READY`
- Any check fails → `<MISSION>_PREFLIGHT_BLOCKED` — stop, report which check failed

## Safety Invariants

- **No write execution** under any circumstance during preflight phase
- **No production memory mutation**
- **No external network calls**
- **No provider keys touched**
- Use only read-only operations and recomputed dry-run plans
- If hashes don't match, the packet or target state has changed since the packet was authored — do NOT proceed, and do NOT modify the packet to match

---

# MEM0 Rollback Execution Pattern

## When This Pattern Applies

Use this pattern when a mission requires **executing a real rollback** of a previously written controlled write fact. This is the inverse of the controlled write — it restores the target to its prewrite state.

## Lifecycle Overview

```
Approval Packet → Preflight Revalidation → [APPROVAL PHRASE] → Rollback Execution → Lock
```

The rollback packet and preflight are created in separate missions BEFORE the execution. The execution is gated by an exact approval phrase.

## Prerequisites

- Rollback packet exists (typically in `docs/mcporter/MEM0_ROLLBACK_REAL_CONTROLLED_PILOT_PACKET.json`)
- Preflight revalidation completed (typically in `docs/mcporter/MEM0_ROLLBACK_REAL_PREFLIGHT_REVALIDATION_REPORT.json`)
- Exact approval phrase known (different from `APPROVED_CONTROLLED_MEMORY_WRITE` — the rollback has its own phrase)

## Rollback Execution Checklist (execute all steps, in order)

### 1. Verify Approval Phrase

```python
REQUIRED_PHRASE = 'APPROVED_MEM0_REAL_ROLLBACK_EXECUTION'
received = '<from user>'
assert received == REQUIRED_PHRASE
```

If the phrase is absent or doesn't match:
- **Do not execute rollback**
- Return `MEM0_ROLLBACK_REAL_EXECUTION_BLOCKED_APPROVAL`
- Save a blocked report documenting the missing approval

### 2. Recompute Hashes (Immediately Before Execution)

```python
from memory.abstraction.controlled_write_pilot import (
    CONTROLLED_WRITE_APPROVAL_PHRASE,
    build_controlled_write_pilot_plan,
)

plan = build_controlled_write_pilot_plan({
    'candidate': {
        'memory_unit_id': '<target-fact-id>',
        'fact_text': '<target-fact-text>',
        'source_type': 'memory_lock',
        'source_id': '<target-source-id>',
    },
    'target_store': '<target-path>',
    'approval_phrase': CONTROLLED_WRITE_APPROVAL_PHRASE,
    'live_write_requested': False,
})

assert plan['hashes']['before_hash'] == packet['hash_verification']['before_hash']
assert plan['hashes']['after_hash'] == packet['hash_verification']['after_hash']
```

### 3. Verify Target Exists

```python
from pathlib import Path
assert Path(target_path).exists(), 'Target must exist for rollback'
target_bytes = Path(target_path).read_bytes()
```

### 4. Create Pre-Rollback Snapshot

Save the current target bytes so the rollback is undoable:

```python
snapshot_dir = Path('memory/.snapshots')
snapshot_dir.mkdir(parents=True, exist_ok=True)
snapshot_path = snapshot_dir / 'rollback_snapshot_before_pre_rollback.json'
snapshot_path.write_bytes(target_bytes)
assert snapshot_path.read_bytes() == target_bytes  # byte-identical
```

### 5. Execute Rollback (Delete Target)

```python
Path(target_path).unlink()
assert not Path(target_path).exists()
```

### 6. Verify JSON and Run Tests

```bash
python3 -m json.tool $ROLLBACK_PACKET
python3 -m json.tool $PREFLIGHT_REPORT
PYTHONPATH=v2 python3 -m pytest v2/memory/abstraction/tests/ -q
```

Note: The test `test_second_controlled_write_packet_matches_existing_target_snapshot_plan` computes hashes from candidate data, not from the target file — it will still pass after deletion.

### 7. git diff --check

```bash
git diff --check
```

### 8. Commit Locally

```bash
git rm <deleted-target-path>
git commit -m "feat(memory): execute real rollback of ..."
```

### 9. Create Memory Lock

Create a memory lock file documenting the rollback:
- `fact_type: memory_lock`
- Include: approval_phrase, rollback_packet reference, preflight reference, target path, hashes, snapshot path, undo_available
- Verification block: json_validation, git_diff_check, unit_tests

### 10. Output Continuity JSON

Include:
- `execution_commit`
- `rollback_packet` + commit
- `preflight_report` + commit
- `target`: path, was_size_bytes, restored_to, undo_snapshot
- `hash_chain`: before_hash, after_hash, verified
- `verification`
- `pr_status`, `merge_status`, `post_merge_verify_status`, `memory_lock_status`
- `unresolved_risks`
- `safety` block

### 11. Stop Before PR/Merge

Unless separately approved, do NOT create a PR. Local commit only.

## Expected Verdicts

- Approval phrase provided and all checks pass → `MEM0_ROLLBACK_REAL_EXECUTION_READY`
- Approval phrase absent or mismatch → `MEM0_ROLLBACK_REAL_EXECUTION_BLOCKED_APPROVAL`

## Pitfalls

- **Rollback and controlled write use DIFFERENT approval phrases:**
  - Controlled write: `APPROVED_CONTROLLED_MEMORY_WRITE`
  - Rollback: `APPROVED_MEM0_REAL_ROLLBACK_EXECUTION`
  Do not confuse them. Using the wrong phrase must block execution.
- **Pre-rollback snapshot is critical for undo:** Without it, the deleted bytes are unrecoverable. Always verify the snapshot was written correctly (byte-identical check) before deleting.
- **Rollback restores prewrite state, not the original file:** If the target was absent before the write, rollback means deletion. The snapshot preserves the written content for undo, but the restored state is the prewrite condition.
- **`git rm` vs `rm`:** Use `git rm` to stage the deletion, not bare `rm`. This ensures git tracks the deletion as a change.
- **`memory/.snapshots/` is gitignored:** Snapshots are NOT committed to the repo. They exist only on the local filesystem for rollback safety.

## Pitfalls

- **First pilot fact at a different path:** The first pilot fact may be at `memory/fact_mem0_first_controlled_memory_write_pilot_ready_20260702.json`. Verify the exact path before running existing-target snapshot tests.
- **Snapshot target path in packet:** The approval packet's `snapshot_target_path` is the path of the **existing fact to snapshot**, not the path of the new fact to write. Do not confuse `target_path` and `snapshot_target_path`.
- **Memory lock files may have duplicate JSON keys:** This does not invalidate the JSON (Python's json.loads accepts duplicates, last wins), but flag it as a data quality observation in the report.
- **Preflight report is read-only evidence:** Do not push the preflight report unless explicitly authorized. Commit is local by default.
- **PYTHONPATH must include v2/:** All cartelogic-v2 memory imports require `PYTHONPATH=v2` or run from the repo root with `PYTHONPATH=v2`.
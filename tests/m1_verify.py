"""M1 集成验证：Severity 分级 + 用户推送门槛 + 集成 filter + 终结报告。

Run standalone:
    cd /home/zml/workspace/hermes-agent && python3 tests/m1_verify.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/zml/workspace/hermes-agent")

from gateway.kanban_watchers import (
    classify_event_severity,
    _filter_event_for_push,
    build_pipeline_summary,
)

from gateway.notification_preferences import (
    DEFAULT_FLOOR,
    load_user_floor,
    load_user_overrides,
    should_push,
    effective_severity,
)

passed = 0
failed = 0

def check(name, got, want):
    global passed, failed
    ok = got == want
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"  {'PASS' if ok else 'FAIL'}: {name}: got={got!r}, want={want!r}")

# ============================================================================
# 1. Severity 分级 — classify_event_severity
# ============================================================================
print("\n=== 1. Severity 分级 (classify_event_severity) ===")

# 1a. Blocked: review-required → P2
check("blocked: review-required → P2",
    classify_event_severity({"kind": "blocked", "task_id": "t1", "reason": "review-required: needs review"}),
    "P2")

# 1b. Blocked: first-time (no prior reason) → P0
check("blocked: first-time non-review → P0 (no prior_reasons)",
    classify_event_severity({"kind": "blocked", "task_id": "t2", "reason": "need decision", "_prior_reasons": []}),
    "P0")

# 1c. Blocked: duplicate reason → P1
check("blocked: duplicate reason → P1",
    classify_event_severity({"kind": "blocked", "task_id": "t3", "reason": "do again", "_prior_reasons": ["do again"]}),
    "P1")

# 1d. Crashed: first 2 occurrences → P0
check("crashed: 1st occurrence → P0",
    classify_event_severity({"kind": "crashed", "task_id": "t4", "_prior_failure_count": 0}),
    "P0")
check("crashed: 2nd occurrence → P0",
    classify_event_severity({"kind": "crashed", "task_id": "t4", "_prior_failure_count": 1}),
    "P0")

# 1e. Crashed: 3rd+ → P1
check("crashed: 3rd occurrence → P1",
    classify_event_severity({"kind": "crashed", "task_id": "t4", "_prior_failure_count": 2}),
    "P1")

# 1f. gave_up / timed_out same as crashed
check("gave_up: 1st → P0",
    classify_event_severity({"kind": "gave_up", "task_id": "t5", "_prior_failure_count": 0}),
    "P0")
check("gave_up: 3rd+ → P1",
    classify_event_severity({"kind": "gave_up", "task_id": "t5", "_prior_failure_count": 3}),
    "P1")

# 1g. Completed: standalone task (no parents, no children) → P0
check("completed: standalone (no parents, no children) → P0",
    classify_event_severity({"kind": "completed", "task_id": "t6", "_has_parents": False, "_has_children": False}),
    "P0")

# 1h. Completed: leaf (has parents, no children) → P1
check("completed: leaf (has parents) → P1",
    classify_event_severity({"kind": "completed", "task_id": "t7", "_has_parents": True, "_has_children": False}),
    "P1")

# 1i. Completed: pipeline root (no parents, HAS children) → P0
# (independent terminal report per DESIGN.md §2.3; previously mislabeled
#  "intermediate" and wrongly asserted P2, which silently dropped reports)
check("completed: pipeline root (no parents, has children) → P0",
    classify_event_severity({"kind": "completed", "task_id": "t8", "_has_parents": False, "_has_children": True}),
    "P0")

# 1i-bis. Completed: intermediate (has parents AND children) → P2 (silent)
check("completed: intermediate (has parents and children) → P2",
    classify_event_severity({"kind": "completed", "task_id": "t8b", "_has_parents": True, "_has_children": True}),
    "P2")

# 1j. protocol_violation → P2
check("protocol_violation → P2",
    classify_event_severity({"kind": "protocol_violation", "task_id": "t9"}),
    "P2")

# 1k. Lifecycle noise → P2
for lc_kind in ("created", "claimed", "spawned", "promoted", "unblocked", "heartbeat"):
    check(f"{lc_kind} → P2",
        classify_event_severity({"kind": lc_kind, "task_id": "t10"}),
        "P2")

# 1l. Unknown/empty → P2
check("empty event → P2",
    classify_event_severity({}),
    "P2")
check("non-dict event → P2",
    classify_event_severity("not a dict"),
    "P2")

# ============================================================================
# 2. 用户推送门槛 — should_push
# ============================================================================
print("\n=== 2. 用户推送门槛 (should_push) ===")

# 2a. Verbose floor
check("verbose + P0 push", should_push("P0", "verbose", {}), True)
check("verbose + P1 push", should_push("P1", "verbose", {}), True)
check("verbose + P2 no push", should_push("P2", "verbose", {}), False)

# 2b. Normal floor
check("normal + P0 push", should_push("P0", "normal", {}), True)
check("normal + P1 no push", should_push("P1", "normal", {}), False)
check("normal + P2 no push", should_push("P2", "normal", {}), False)

# 2c. Quiet floor: all silent
check("quiet + P0 no push", should_push("P0", "quiet", {}), False)
check("quiet + P1 no push", should_push("P1", "quiet", {}), False)

# 2d. Overrides
check("override demotes P0→P2 (verbose) no push",
    should_push("P0", "verbose", {"task_completed": "P2"}, event_type="task_completed"),
    False)
check("override promotes P1→P0 (normal) push",
    should_push("P1", "normal", {"task_completed": "P0"}, event_type="task_completed"),
    True)
check("override no effect when kind mismatch",
    should_push("P1", "normal", {"task_completed": "P0"}, event_type="task_crashed"),
    False)

# ============================================================================
# 3. 集成 filter — _filter_event_for_push
# ============================================================================
print("\n=== 3. 集成 filter (_filter_event_for_push) ===")

# We can't easily test _filter_event_for_push without a real kanban DB
# because it opens its own connection. But we can verify the function exists,
# accepts correct params, and the logic path is coherent.
# For a full test we'd need an in-memory DB with a notification-policy board.

check("_filter_event_for_push is callable",
    callable(_filter_event_for_push), True)

# Check that effective_severity → should_push chain works end-to-end
# (this is what _filter_event_for_push internally does)
eff = effective_severity({"kind": "task_completed"}, "P0", {"task_completed": "P2"})
check("effective_severity with override → P2", eff, "P2")
eff2 = effective_severity({"kind": "task_completed"}, "P1", {})
check("effective_severity no override → P1", eff2, "P1")

# ============================================================================
# 4. 终结报告 — build_pipeline_summary
# ============================================================================
print("\n=== 4. 终结报告 (build_pipeline_summary) ===")

check("build_pipeline_summary is callable",
    callable(build_pipeline_summary), True)

# build_pipeline_summary needs a DB connection, so we can only verify
# importability and API shape here. M1-4 confirmed syntax OK.

# ============================================================================
# Summary
# ============================================================================
print(f"\n{'='*50}")
print(f"Total: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)

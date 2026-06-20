"""Smoke test for gateway/notification_preferences.py (M1-2).

Verifies the four published functions + the convenience aggregator
across the documented behaviours and fallback rules.  Run with:
    cd /home/zml/workspace/hermes-agent && python3 tests/test_notification_preferences_smoke.py
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/zml/workspace/hermes-agent")

from gateway.notification_preferences import (
    DEFAULT_FLOOR,
    load_all_preferences,
    load_user_floor,
    load_user_overrides,
    effective_severity,
    should_push,
    VALID_FLOORS,
    VALID_SEVERITIES,
)

# Wrap helpers to accept the raw str paths from os.path.join seamlessly
def _floor(p):
    return load_user_floor(Path(p))

def _overrides(p):
    return load_user_overrides(Path(p))

def _all(p):
    return load_all_preferences(Path(p))

passed = 0
failed = 0


def check(name, got, want):
    global passed, failed
    if got == want:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}: got {got!r}, want {want!r}")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
check("DEFAULT_FLOOR is verbose", DEFAULT_FLOOR, "verbose")
check("VALID_FLOORS contains expected", "verbose" in VALID_FLOORS and "normal" in VALID_FLOORS and "quiet" in VALID_FLOORS, True)
check("VALID_SEVERITIES contains P0/P1/P2", VALID_SEVERITIES, frozenset({"P0", "P1", "P2"}))

# ---------------------------------------------------------------------------
# 1. Missing config file → defaults
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    missing = os.path.join(td, "nonexistent.yaml")
    check("missing file → default floor", _floor(missing), "verbose")
    check("missing file → empty overrides", _overrides(missing), {})

# ---------------------------------------------------------------------------
# 2. Empty config
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "prefs.yaml")
    open(p, "w").close()
    check("empty file → default floor", _floor(p), "verbose")
    check("empty file → empty overrides", _overrides(p), {})

# ---------------------------------------------------------------------------
# 3. Garbage YAML
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "prefs.yaml")
    open(p, "w").write("not valid: yaml: [[[")
    check("malformed YAML → default floor", _floor(p), "verbose")
    check("malformed YAML → empty overrides", _overrides(p), {})

# ---------------------------------------------------------------------------
# 4. Top-level not a mapping
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "prefs.yaml")
    open(p, "w").write("- a\n- b\n")
    check("top-level list → default floor", _floor(p), "verbose")

# ---------------------------------------------------------------------------
# 5. Valid config (normal floor + overrides with mixed valid/invalid entries)
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "prefs.yaml")
    open(p, "w").write(
        "notification:\n"
        "  floor: normal\n"
        "  overrides:\n"
        "    task_completed: P0\n"
        "    task_crashed: P2\n"
        "    bad_severity: P9\n"
        "    bad_type:\n"
        "      nested: yes\n"
    )
    check("floor = normal", _floor(p), "normal")
    ov = _overrides(p)
    check("overrides keeps valid entries", ov, {"task_completed": "P0", "task_crashed": "P2"})
    check("overrides drops invalid severity", "bad_severity" in ov, False)
    check("overrides drops wrong-typed value", "bad_type" in ov, False)

# ---------------------------------------------------------------------------
# 6. Invalid floor → fallback
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "prefs.yaml")
    open(p, "w").write("notification:\n  floor: turbo\n")
    check("invalid floor → verbose default", _floor(p), "verbose")

# ---------------------------------------------------------------------------
# 7. should_push — verbose floor (default)
# ---------------------------------------------------------------------------
check("verbose + P0 push", should_push("P0", "verbose", {}), True)
check("verbose + P1 push", should_push("P1", "verbose", {}), True)
check("verbose + P2 no push", should_push("P2", "verbose", {}), False)

# ---------------------------------------------------------------------------
# 8. should_push — normal floor
# ---------------------------------------------------------------------------
check("normal + P0 push", should_push("P0", "normal", {}), True)
check("normal + P1 no push", should_push("P1", "normal", {}), False)
check("normal + P2 no push", should_push("P2", "normal", {}), False)

# ---------------------------------------------------------------------------
# 9. should_push — quiet floor
# ---------------------------------------------------------------------------
check("quiet + P0 no push", should_push("P0", "quiet", {}), False)
check("quiet + P1 no push", should_push("P1", "quiet", {}), False)
check("quiet + P2 no push", should_push("P2", "quiet", {}), False)

# ---------------------------------------------------------------------------
# 10. should_push — overrides take precedence over floor
# ---------------------------------------------------------------------------
check(
    "override demotes P0→P2 with verbose",
    should_push("P0", "verbose", {"task_completed": "P2"}, event_type="task_completed"),
    False,
)
check(
    "override promotes P1→P0 with normal",
    should_push("P1", "normal", {"task_completed": "P0"}, event_type="task_completed"),
    True,
)
check(
    "override no effect when kind mismatch",
    should_push("P1", "normal", {"task_completed": "P0"}, event_type="task_crashed"),
    False,
)
check(
    "override with P2 stays silent under verbose",
    should_push("P2", "verbose", {"task_blocked": "P2"}, event_type="task_blocked"),
    False,
)

# ---------------------------------------------------------------------------
# 11. effective_severity
# ---------------------------------------------------------------------------
check(
    "effective_severity replaces with override",
    effective_severity({"kind": "task_completed"}, "P1", {"task_completed": "P0"}),
    "P0",
)
check(
    "effective_severity no override → base",
    effective_severity({"kind": "task_completed"}, "P1", {}),
    "P1",
)
check(
    "effective_severity invalid override → base",
    effective_severity({"kind": "task_completed"}, "P1", {"task_completed": "PX"}),
    "P1",
)
check(
    "effective_severity accepts event_type key",
    effective_severity({"event_type": "task_completed"}, "P1", {"task_completed": "P0"}),
    "P0",
)
check(
    "effective_severity empty event kind → base",
    effective_severity({}, "P1", {"task_completed": "P0"}),
    "P1",
)
check(
    "effective_severity non-dict event → base",
    effective_severity("not a dict", "P1", {"task_completed": "P0"}),
    "P1",
)

# ---------------------------------------------------------------------------
# 12. load_all_preferences aggregator
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "prefs.yaml")
    open(p, "w").write("notification:\n  floor: quiet\n  overrides:\n    x: P0\n")
    all_prefs = _all(p)
    check("load_all_preferences floor", all_prefs["floor"], "quiet")
    check("load_all_preferences overrides", all_prefs["overrides"], {"x": "P0"})

# ---------------------------------------------------------------------------
# 13. Unknown floor in should_push falls back to verbose behaviour
# ---------------------------------------------------------------------------
check("unknown floor → verbose behaviour P0", should_push("P0", "turbo", {}), True)
check("unknown floor → verbose behaviour P1", should_push("P1", "turbo", {}), True)
check("unknown floor → verbose behaviour P2", should_push("P2", "turbo", {}), False)

# ---------------------------------------------------------------------------
# 14. env-var path override
# ---------------------------------------------------------------------------
os.environ["HERMES_NOTIFICATION_PREFERENCES"] = "/tmp/__definitely_does_not_exist__.yaml"
check("env var path → defaults", load_user_floor(), "verbose")
check("env var path → empty overrides", load_user_overrides(), {})
del os.environ["HERMES_NOTIFICATION_PREFERENCES"]

# ---------------------------------------------------------------------------
# 15. Integrates with M1-1's classify_event_severity (smoke import)
# ---------------------------------------------------------------------------
try:
    from gateway.kanban_watchers import classify_event_severity

    check("classify_event_severity importable alongside", callable(classify_event_severity), True)
except Exception as e:
    check(f"classify_event_severity importable (got {e})", False, True)

print(f"\n{passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
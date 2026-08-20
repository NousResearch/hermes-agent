#!/usr/bin/env python3
"""
E2E verification script for the unexpected-restart notification feature (PR #74662).

Exercises the REAL GatewayRunner.start() lifecycle (not a mock of it) to verify
that the prev_gateway_state snapshot timing fix works correctly across all
start/stop lifecycle scenarios.

Usage:
    source venv/bin/activate
    python tests/gateway/test_unexpected_restart_e2e.py

The script creates an isolated temp HERMES_HOME per scenario, so it is safe to
run while a real gateway is live.  It constructs a real GatewayRunner with no
platforms (skipping adapter I/O) and only mocks the notification sender to
capture whether it was called -- the state-snapshot / decision logic in start()
runs unmodified.

SCENARIOS (full lifecycle coverage):

  No markers (unexpected restart detection -- our new code):
    1.  prev=running       -> notify (gateway died while active)
    2.  prev=draining      -> notify (gateway died while draining)
    3.  prev=stopped       -> NO    (clean stop, manual restart)
    4.  prev=None          -> NO    (first boot)
    5.  prev=startup_failed-> NO    (previous boot failed, never reached running)
    6.  prev=degraded      -> NO    (NOT in trigger set -- edge case)
    7.  prev=starting      -> NO    (crashed during startup, never reached running)

  Planned restart marker (.restart_pending.json):
    8.  prev=running  + .restart_pending -> notify (planned restart, branch 1)
    9.  prev=stopped  + .restart_pending -> notify (planned restart overrides)
    10. prev=None     + .restart_pending -> notify (planned restart on first boot)

  Chat restart marker (.restart_notify.json):
    11. prev=running  + .restart_notify  -> NO home-channel (chat restart, branch 3)
    12. prev=stopped  + .restart_notify  -> NO home-channel (chat restart, branch 3)

  Both markers (edge case):
    13. prev=running + both markers      -> notify (branch 1 takes priority)

  Timing regression:
    14. prev=stopped, no markers          -> NO notify (proves snapshot is
                                              taken BEFORE starting/running writes)
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# ── Ensure we can import from the repo root ──────────────────────────────────
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


class E2ETestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.details: list[str] = []
        self.error: str | None = None


def _make_state_json(gateway_state: str | None) -> dict | None:
    """Build a realistic gateway_state.json payload."""
    if gateway_state is None:
        return None
    return {
        "gateway_state": gateway_state,
        "pid": 99999,
        "kind": "gateway",
        "start_time": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "exit_reason": None,
    }


async def _run_scenario(
    name: str,
    prev_state: str | None,
    *,
    restart_pending: bool = False,
    restart_notify: bool = False,
    expect_home_notify: bool,
) -> E2ETestResult:
    """Run a single E2E scenario through the real GatewayRunner.start().

    Args:
        name:             Human-readable scenario name.
        prev_state:       Gateway state to pre-seed in gateway_state.json.
                          None means no file (first boot).
        restart_pending:  If True, create .restart_pending.json marker.
        restart_notify:   If True, create .restart_notify.json marker.
        expect_home_notify: Whether _send_home_channel_startup_notifications
                            is expected to be called.
    """
    result = E2ETestResult(name)
    tmp_home = Path(tempfile.mkdtemp(prefix="hermes_e2e_"))

    try:
        os.environ["HERMES_HOME"] = str(tmp_home)
        (tmp_home / "sessions").mkdir(exist_ok=True)
        (tmp_home / "logs").mkdir(exist_ok=True)

        # ── Pre-seed gateway_state.json ───────────────────────────────────
        state_file = tmp_home / "gateway_state.json"
        prev_content = _make_state_json(prev_state)
        if prev_content is not None:
            state_file.write_text(
                json.dumps(prev_content, default=str), encoding="utf-8"
            )
            result.details.append(f"  prev_state = {prev_state!r}")
        else:
            result.details.append("  prev_state = None (no file)")

        # ── Pre-seed marker files ─────────────────────────────────────────
        if restart_pending:
            (tmp_home / ".restart_pending.json").write_text(
                json.dumps({"requested_at": 1, "via_service": False}), encoding="utf-8"
            )
            result.details.append("  .restart_pending.json = created")
        if restart_notify:
            (tmp_home / ".restart_notify.json").write_text(
                json.dumps({"platform": "telegram", "chat_id": "123"}), encoding="utf-8"
            )
            result.details.append("  .restart_notify.json  = created")

        if not restart_pending and not restart_notify:
            result.details.append("  (no markers)")

        # ── Construct real GatewayRunner ──────────────────────────────────
        import importlib
        import gateway.run as gateway_run_mod
        import gateway.config as gateway_config_mod
        importlib.reload(gateway_config_mod)
        importlib.reload(gateway_run_mod)
        from gateway.run import GatewayRunner
        from gateway.config import GatewayConfig

        config = GatewayConfig()
        runner = GatewayRunner(config)

        # ── Mock only what's needed to let start() run without external I/O ─
        notification_sent = False

        async def _capture_home_notify(*, skip_targets=None):
            nonlocal notification_sent
            notification_sent = True
            result.details.append("  _send_home_channel_startup_notifications() CALLED")
            return set()

        runner._send_home_channel_startup_notifications = _capture_home_notify
        runner._send_restart_notification = AsyncMock(return_value=None)
        runner._send_update_notification = AsyncMock(return_value=True)
        runner._redeliver_pending_obligations = AsyncMock(return_value=0)
        runner._finish_startup_restore = AsyncMock(return_value=None)
        runner._schedule_resume_pending_sessions = MagicMock(return_value=0)
        runner.hooks.discover_and_load = MagicMock()
        runner.hooks.emit = AsyncMock()

        # ── Run start() ───────────────────────────────────────────────────
        result.details.append("  Calling runner.start()...")
        success = await runner.start()
        result.details.append(f"  start() returned: {success}")

        # ── Verify ────────────────────────────────────────────────────────
        if expect_home_notify:
            if notification_sent:
                result.passed = True
                result.details.append("  PASS: home-channel notification sent (expected)")
            else:
                result.passed = False
                result.details.append("  FAIL: expected notification but none was sent")
        else:
            if not notification_sent:
                result.passed = True
                result.details.append("  PASS: no home-channel notification (expected)")
            else:
                result.passed = False
                result.details.append("  FAIL: notification sent but should NOT have been")

        # Verify final state file
        if state_file.exists():
            final = json.loads(state_file.read_text(encoding="utf-8"))
            result.details.append(f"  final gateway_state = {final.get('gateway_state')!r}")

        # Verify .restart_pending.json was cleared (branch 1 clears it after notify)
        if restart_pending:
            rp = tmp_home / ".restart_pending.json"
            if not rp.exists():
                result.details.append("  .restart_pending.json cleared (expected)")
            else:
                # It should have been cleared by _clear_planned_restart_notification()
                result.details.append("  WARNING: .restart_pending.json still exists")

    except Exception as exc:
        result.passed = False
        result.error = f"{type(exc).__name__}: {exc}"
        result.details.append(f"  EXCEPTION: {traceback.format_exc()}")
    finally:
        try:
            if 'runner' in dir() and hasattr(runner, '_running') and runner._running:
                try:
                    await runner.stop()
                except Exception:
                    pass
        except Exception:
            pass
        shutil.rmtree(tmp_home, ignore_errors=True)

    return result


async def main():
    print("=" * 76)
    print("E2E Verification: Unexpected-Restart Notification (PR #74662)")
    print("Full lifecycle scenario coverage")
    print("=" * 76)

    scenarios = [
        # ── No markers: unexpected-restart detection (our new code) ──────
        ("S1  prev=running        no markers  -> notify",
         "running", False, False, True),
        ("S2  prev=draining       no markers  -> notify",
         "draining", False, False, True),
        ("S3  prev=stopped        no markers  -> NO notify",
         "stopped", False, False, False),
        ("S4  prev=None (1st boot) no markers -> NO notify",
         None, False, False, False),
        ("S5  prev=startup_failed  no markers -> NO notify",
         "startup_failed", False, False, False),
        ("S6  prev=degraded        no markers -> NO notify (edge)",
         "degraded", False, False, False),
        ("S7  prev=starting        no markers -> NO notify",
         "starting", False, False, False),

        # ── Planned restart marker (.restart_pending.json) ───────────────
        ("S8  prev=running  + .restart_pending -> notify (planned)",
         "running", True, False, True),
        ("S9  prev=stopped  + .restart_pending -> notify (planned)",
         "stopped", True, False, True),
        ("S10 prev=None     + .restart_pending -> notify (planned)",
         None, True, False, True),

        # ── Chat restart marker (.restart_notify.json) ───────────────────
        ("S11 prev=running  + .restart_notify  -> NO home-notify (chat)",
         "running", False, True, False),
        ("S12 prev=stopped  + .restart_notify  -> NO home-notify (chat)",
         "stopped", False, True, False),

        # ── Both markers (edge case) ─────────────────────────────────────
        ("S13 prev=running  + both markers     -> notify (planned wins)",
         "running", True, True, True),

        # ── Timing regression ────────────────────────────────────────────
        ("S14 prev=stopped   no markers        -> NO notify (timing fix)",
         "stopped", False, False, False),
    ]

    results: list[E2ETestResult] = []

    for name, prev_state, rp, rn, expect in scenarios:
        print(f"\n{'─' * 76}")
        print(f"{name}")
        print(f"{'─' * 76}")
        r = await _run_scenario(name, prev_state,
                                restart_pending=rp, restart_notify=rn,
                                expect_home_notify=expect)
        results.append(r)
        for d in r.details:
            print(d)
        status = "PASS" if r.passed else "FAIL"
        print(f"\n  >> {status}")
        if r.error:
            print(f"     {r.error}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 76}")
    print("SUMMARY")
    print(f"{'=' * 76}")
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name}")
    print(f"\n  Total: {len(results)}  |  Passed: {passed}  |  Failed: {failed}")
    print()

    if failed > 0:
        print("FAILURES:")
        for r in results:
            if not r.passed:
                print(f"\n  {r.name}")
                if r.error:
                    print(f"    Error: {r.error}")
                for d in r.details:
                    if "FAIL" in d or "EXCEPTION" in d:
                        print(f"    {d}")
        sys.exit(1)
    else:
        print("All scenarios passed!")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())

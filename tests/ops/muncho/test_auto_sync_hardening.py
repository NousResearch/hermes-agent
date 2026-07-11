from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[3]
    / "ops"
    / "muncho"
    / "runtime"
    / "auto_sync_hardening.py"
)
SPEC = importlib.util.spec_from_file_location("auto_sync_hardening", MODULE_PATH)
assert SPEC and SPEC.loader
hardening = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hardening)

OLD = "1" * 40
CURRENT = "2" * 40
HEAD = "3" * 40


def test_superseded_automation_snapshot_is_stale():
    assert hardening.classify_stale_sync_pr(
        automation_owned=True,
        head_already_in_fork_main=False,
        upstream_snapshot_sha=OLD,
        upstream_snapshot_in_fork_merge_base=False,
        current_upstream_sha=CURRENT,
        current_upstream_contains_snapshot=True,
    ) == "upstream_snapshot_superseded"


def test_superseded_snapshot_requires_ancestry_proof():
    assert hardening.classify_stale_sync_pr(
        automation_owned=True,
        head_already_in_fork_main=False,
        upstream_snapshot_sha=OLD,
        upstream_snapshot_in_fork_merge_base=False,
        current_upstream_sha=CURRENT,
        current_upstream_contains_snapshot=False,
    ) is None


def test_non_automation_pr_is_never_auto_closed():
    assert hardening.classify_stale_sync_pr(
        automation_owned=False,
        head_already_in_fork_main=True,
        upstream_snapshot_sha=OLD,
        upstream_snapshot_in_fork_merge_base=True,
        current_upstream_sha=CURRENT,
        current_upstream_contains_snapshot=True,
    ) is None


def test_existing_stale_reasons_keep_precedence():
    assert hardening.classify_stale_sync_pr(
        automation_owned=True,
        head_already_in_fork_main=True,
        upstream_snapshot_sha=OLD,
        upstream_snapshot_in_fork_merge_base=True,
        current_upstream_sha=CURRENT,
        current_upstream_contains_snapshot=True,
    ) == "head_already_in_fork_main"


def test_blocker_fingerprint_is_order_independent():
    first = hardening.blocker_fingerprint(
        status="blocked_auto_merge_deploy_gate",
        pr_number=91,
        head_sha=HEAD,
        blockers=["checks_failed", "merge_state_UNSTABLE"],
        failed_checks=[
            {"name": "slice 5", "conclusion": "failure"},
            {"name": "required", "conclusion": "failure"},
        ],
    )
    second = hardening.blocker_fingerprint(
        status="blocked_auto_merge_deploy_gate",
        pr_number=91,
        head_sha=HEAD,
        blockers=["merge_state_UNSTABLE", "checks_failed"],
        failed_checks=[
            {"name": "required", "conclusion": "FAILURE"},
            {"name": "slice 5", "conclusion": "FAILURE"},
        ],
    )
    assert first == second


def test_unchanged_blocker_is_suppressed_until_repeat_window(tmp_path):
    state = tmp_path / "private" / "blocker.json"
    now = datetime(2026, 7, 11, 12, tzinfo=timezone.utc)
    fingerprint = "a" * 64

    first = hardening.decide_blocker_delivery(
        state, fingerprint=fingerprint, now=now
    )
    repeated = hardening.decide_blocker_delivery(
        state, fingerprint=fingerprint, now=now + timedelta(hours=3)
    )
    reminder = hardening.decide_blocker_delivery(
        state, fingerprint=fingerprint, now=now + timedelta(hours=24)
    )

    assert first["emit"] is True
    assert repeated == {
        "emit": False,
        "reason": "unchanged_blocker_suppressed",
        "suppressed_runs": 1,
        "repeat_after_seconds": 86400,
    }
    assert reminder["emit"] is True
    assert state.stat().st_mode & 0o777 == 0o600
    assert state.parent.stat().st_mode & 0o777 == 0o700


def test_changed_or_cleared_blocker_emits_again(tmp_path):
    state = tmp_path / "blocker.json"
    now = datetime(2026, 7, 11, 12, tzinfo=timezone.utc)
    hardening.decide_blocker_delivery(state, fingerprint="a" * 64, now=now)

    changed = hardening.decide_blocker_delivery(
        state, fingerprint="b" * 64, now=now + timedelta(hours=3)
    )
    hardening.clear_blocker_delivery_state(state, now=now + timedelta(hours=4))
    recurrence = hardening.decide_blocker_delivery(
        state, fingerprint="b" * 64, now=now + timedelta(hours=5)
    )

    assert changed["emit"] is True
    assert recurrence["emit"] is True
    assert json.loads(state.read_text())["active"] is True

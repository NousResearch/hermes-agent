from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).parents[3]
RUNTIME = ROOT / "ops" / "muncho" / "runtime"


def _load_routine():
    sys.path.insert(0, str(RUNTIME))
    try:
        spec = importlib.util.spec_from_file_location(
            "fork_upstream_auto_sync_pr_routine",
            RUNTIME / "fork_upstream_auto_sync_pr_routine.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(RUNTIME))


def test_runtime_stale_detection_checks_current_upstream_ancestry(monkeypatch):
    routine = _load_routine()
    snapshot = "1" * 40
    current = "2" * 40
    fork = "3" * 40
    head = "4" * 40
    pr = {
        "headRefName": "codex/upstream-sync-auto-20260711-1200",
        "headRefOid": head,
        "body": f"Automated fork-only upstream sync PR\nUpstream main: `{snapshot}`",
    }
    fresh = {
        "fork_main_ref": fork,
        "merge_base": "5" * 40,
        "upstream_main_ref": current,
    }

    monkeypatch.setattr(routine, "is_auto_owned_sync_pr", lambda _: True)

    def contains(repo, base, candidate):
        return repo == routine.UPSTREAM_REPO and base == snapshot and candidate == current

    monkeypatch.setattr(routine, "compare_shows_head_contains_base", contains)
    assert routine.stale_sync_reason(pr, fresh) == "upstream_snapshot_superseded"


def test_runtime_dedupe_suppresses_unchanged_blocker(tmp_path, monkeypatch):
    routine = _load_routine()
    monkeypatch.setattr(routine, "BLOCKER_DEDUPE_STATE", tmp_path / "dedupe.json")
    report = {
        "status": "blocked_auto_merge_deploy_gate",
        "auto_merge_deploy": {
            "blockers": ["checks_failed", "merge_state_UNSTABLE"],
            "checks": {
                "failure_like_checks": [
                    {"name": "Python tests / slice 5", "conclusion": "FAILURE"}
                ]
            },
        },
    }
    pr = {"number": 91, "headRefOid": "6" * 40}

    assert routine.apply_blocker_notification_dedupe(report, pr) is True
    assert routine.apply_blocker_notification_dedupe(report, pr) is False
    assert report["blocker_notification"]["reason"] == "unchanged_blocker_suppressed"


def test_deploy_marks_planned_stop_before_symlink_swap_and_restart():
    source = (RUNTIME / "muncho-auto-deploy-release").read_text(encoding="utf-8")
    marker = source.index('marker_output="$(')
    symlink_swap = source.index('ln -sfn "$new" "$ACTIVE_LINK.next"')
    restart = source.index('systemctl restart "$SERVICE"')
    verify_consumed = source.index('blocked_planned_stop_marker_not_consumed')

    assert marker < symlink_swap < restart < verify_consumed
    assert 'blocked_planned_restart_helper_missing' in source
    assert 'blocked_planned_stop_marker_failed' in source

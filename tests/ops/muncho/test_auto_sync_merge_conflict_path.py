from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[3]
RUNTIME = ROOT / "ops" / "muncho" / "runtime"


def _load_routine():
    module_name = "fork_upstream_auto_sync_pr_routine_conflict_path_test"
    sys.path.insert(0, str(RUNTIME))
    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            RUNTIME / "fork_upstream_auto_sync_pr_routine.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(RUNTIME))


def test_blocked_merge_conflicts_emit_once_then_stay_silent_and_safe(
    tmp_path, monkeypatch, capsys
):
    routine = _load_routine()
    state_dir = tmp_path / "state"
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    monkeypatch.setattr(routine, "STATE_DIR", state_dir)
    monkeypatch.setattr(routine, "REPORT_DIR", report_dir)
    monkeypatch.setattr(
        routine, "BLOCKER_DEDUPE_STATE", state_dir / "blocker-dedupe.json"
    )
    monkeypatch.setattr(routine, "WORKTREE_ROOT", tmp_path / "worktrees")
    monkeypatch.setenv(routine.EXECUTE_ENV, "1")

    run_number = 0

    def build_plan(_args):
        nonlocal run_number
        run_number += 1
        return {
            "created_at_utc": f"2026-07-14T0{run_number * 3}:00:00Z",
            "status": "dry_run_plan",
            "mode": "execute",
            "monitor_state": {},
            "fresh_refs": {
                "fork_main_ref": "a" * 40,
                # Normal upstream motion must not create a new notification
                # when the effective conflict-path set is unchanged.
                "upstream_main_ref": str(run_number) * 40,
                "merge_base": "b" * 40,
                "ahead_by": 188,
                "behind_by": 196 + run_number,
            },
            "open_sync_prs": [],
            "proposed_branch": f"codex/upstream-sync-auto-run-{run_number}",
            "hard_boundaries": {},
        }

    monkeypatch.setattr(routine, "build_plan", build_plan)
    monkeypatch.setattr(
        routine,
        "cleanup_stale_sync_prs",
        lambda _open_prs, _fresh: {"closed": [], "kept": []},
    )
    monkeypatch.setattr(routine, "cleanup_old_auto_sync_worktrees", lambda: [])
    monkeypatch.setattr(routine, "disk_free_bytes", lambda _path: 10 << 30)
    monkeypatch.setattr(routine, "marker_scan", lambda _path: [])
    monkeypatch.setattr(
        routine,
        "try_known_conflict_auto_resolvers",
        lambda _repo, _conflicted: {
            "resolved": False,
            "reason": "unsupported_conflict_set",
        },
    )

    commands: list[tuple[str, ...]] = []
    conflicts = ("gateway/run.py", "tools/approval.py")

    def fake_run(cmd, *, cwd=None, check=True, timeout=None):
        del cwd, check, timeout
        command = tuple(cmd)
        commands.append(command)
        if command[:4] == ("git", "merge", "--no-commit", "--no-ff"):
            return routine.CmdResult(list(cmd), 1, "", "merge conflict")
        if command == ("git", "diff", "--name-only", "--diff-filter=U"):
            return routine.CmdResult(list(cmd), 0, "\n".join(conflicts) + "\n", "")
        return routine.CmdResult(list(cmd), 0, "", "")

    monkeypatch.setattr(routine, "run", fake_run)

    def forbidden_external_mutation(*_args, **_kwargs):
        raise AssertionError("blocked merge-conflict path attempted an external mutation")

    monkeypatch.setattr(routine, "gh_json", forbidden_external_mutation)
    monkeypatch.setattr(
        routine, "auto_merge_sync_pr_and_start_deploy", forbidden_external_mutation
    )
    monkeypatch.setattr(
        routine, "queue_auto_deploy_request", forbidden_external_mutation
    )

    args = argparse.Namespace(execute=True)

    monkeypatch.setenv("HERMES_CRON_PREVIOUS_DELIVERY", "none")
    assert routine.execute(args) == 2
    first_stdout = capsys.readouterr().out
    assert "Status: `blocked_merge_conflicts`" in first_stdout
    first_report = json.loads(
        (state_dir / "auto-sync-pr-latest.json").read_text(encoding="utf-8")
    )
    assert first_report["blocker_notification"]["emit"] is True
    assert first_report["blocker_notification"]["delivery_confirmed_at"] is None
    assert first_report["conflicted_files"] == list(conflicts)

    monkeypatch.setenv(
        "HERMES_CRON_PREVIOUS_RUN_AT", "2026-07-14T03:00:00+00:00"
    )
    monkeypatch.setenv("HERMES_CRON_PREVIOUS_DELIVERY", "confirmed")
    assert routine.execute(args) == 0
    assert capsys.readouterr().out == ""
    second_report = json.loads(
        (state_dir / "auto-sync-pr-latest.json").read_text(encoding="utf-8")
    )
    assert second_report["status"] == "blocked_merge_conflicts"
    assert second_report["blocker_notification"]["emit"] is False
    assert (
        second_report["blocker_notification"]["reason"]
        == "unchanged_delivered_blocker_suppressed"
    )
    assert second_report["blocker_notification"]["delivery_confirmed_at"]

    archived_reports = sorted(state_dir.glob("auto-sync-pr-2026*.json"))
    assert len(archived_reports) == 2
    assert all(json.loads(path.read_text())["status"] == "blocked_merge_conflicts" for path in archived_reports)
    assert (report_dir / "fork-upstream-auto-sync-pr-latest-public-summary.md").is_file()

    assert sum(command[:3] == ("git", "merge", "--abort") for command in commands) == 2
    assert not any("push" in command for command in commands)
    assert not any("pr" in command for command in commands)
    assert not any(str(routine.AUTO_DEPLOY_HELPER) in command for command in commands)

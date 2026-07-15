from __future__ import annotations

import importlib.util
import subprocess
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
    assert (
        report["blocker_notification"]["reason"]
        == "unchanged_selection_suppressed_unconfirmed"
    )
    assert report["blocker_notification"]["delivery_confirmed_at"] is None


def test_runtime_dedupe_treats_merge_conflict_paths_as_stable_identity(
    tmp_path, monkeypatch
):
    routine = _load_routine()
    monkeypatch.setattr(routine, "BLOCKER_DEDUPE_STATE", tmp_path / "dedupe.json")
    report = {
        "status": "blocked_merge_conflicts",
        "fresh_refs": {
            "fork_main_ref": "a" * 40,
            "upstream_main_ref": "1" * 40,
            "behind_by": 196,
        },
        "conflicted_files": ["gateway/run.py", "tools/approval.py"],
    }

    assert routine.apply_blocker_notification_dedupe(report, {}) is True

    # Upstream movement is evidence for the report, not a new blocker. The
    # same conflict set stays suppressed until the 24-hour reminder window.
    report["fresh_refs"] = {
        "fork_main_ref": "a" * 40,
        "upstream_main_ref": "2" * 40,
        "behind_by": 211,
    }
    assert routine.apply_blocker_notification_dedupe(report, {}) is False

    # A materially different conflict set is a new blocker and emits now.
    report["conflicted_files"].append("hermes_cli/config.py")
    assert routine.apply_blocker_notification_dedupe(report, {}) is True

    # A new fork base can change the conflict itself even when path names stay
    # the same, so it is a new blocker identity and must notify immediately.
    assert routine.apply_blocker_notification_dedupe(report, {}) is False
    report["fresh_refs"]["fork_main_ref"] = "b" * 40
    assert routine.apply_blocker_notification_dedupe(report, {}) is True


def test_deploy_marks_planned_stop_before_symlink_swap_and_restart():
    source = (RUNTIME / "muncho-auto-deploy-release").read_text(encoding="utf-8")
    run_deploy = source[source.index("run_deploy() {") : source.index("main() {")]
    marker = run_deploy.index('marker_output="$(')
    symlink_swap = run_deploy.index('ln -sfn "$new" "$ACTIVE_LINK.next"')
    restart = run_deploy.index('systemctl restart "$SERVICE"')
    verify_consumed = run_deploy.index('planned_stop_marker_not_consumed')

    assert marker < symlink_swap < restart < verify_consumed
    assert 'blocked_planned_restart_helper_missing' in source
    assert 'blocked_planned_stop_marker_failed' in source
    assert 'rollback_release() {' in source
    assert 'ln -sfn "$previous" "$ACTIVE_LINK.rollback"' in source
    assert 'write_status "deploy_rolled_back"' in source
    assert 'REPO_URL="https://github.com/lomliev/hermes-agent.git"' in source
    assert "MUNCHO_REPO_URL" not in source
    assert 'release_identity_matches "$active" "$active_head"' in source
    assert 'release_identity_matches "$new" "$sha"' in source
    assert '"$RELEASES/hermes-agent-${expected_head:0:12}"' in source
    assert 'DEPLOY_HEALTH_WAIT_SECONDS" -gt 300' in source
    assert "previous_release_identity_invalid" in source
    assert '"restored_source":' not in source


def test_deploy_staging_dependency_package_is_final_address_bound():
    helper = RUNTIME / "muncho-auto-deploy-release"
    source = helper.read_text(encoding="utf-8")
    run_deploy = source[source.index("run_deploy() {") : source.index("main() {")]
    prepare = run_deploy.index(
        'package_production_runtime_dependencies.py" prepare'
    )
    prepare_address = run_deploy.index(
        '--release-address "$new"',
        prepare,
    )
    prepare_revision = run_deploy.index('--revision "$sha"', prepare_address)
    seal = run_deploy.index(
        'seal_agent_browser_config "$tmp" "$sha"',
        prepare_revision,
    )
    build = run_deploy.index(
        'package_production_runtime_dependencies.py" build-manifest',
        seal,
    )
    verify = run_deploy.index(
        'package_production_runtime_dependencies.py" verify',
        build,
    )
    verify_address = run_deploy.index(
        '--release-address "$new"',
        verify,
    )
    move = run_deploy.index('mv "$tmp" "$new"', verify_address)

    assert (
        prepare
        < prepare_address
        < prepare_revision
        < seal
        < build
        < verify
        < verify_address
        < move
    )
    syntax = subprocess.run(
        ["bash", "-n", str(helper)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert syntax.returncode == 0, syntax.stderr

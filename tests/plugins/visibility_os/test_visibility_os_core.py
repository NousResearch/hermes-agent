import json
import sqlite3
import subprocess
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def patch_db(tmp_path, monkeypatch):
    from plugins.visibility_os.core import db
    monkeypatch.setenv("VISIBILITY_OS_GITHUB_ORGS", "acme-inc")
    monkeypatch.setattr(db, "get_db_path", lambda: tmp_path / "visibility_os.db")
    return db


def test_visibility_os_config_is_env_driven(monkeypatch):
    monkeypatch.setenv("VISIBILITY_OS_COMPANY_NAME", "Acme Robotics")
    monkeypatch.setenv("VISIBILITY_OS_GITHUB_ORGS", "acme-inc,acme-labs")
    monkeypatch.setenv("VISIBILITY_OS_GITHUB_REPOS", "acme-inc/api, acme-labs/web")
    monkeypatch.setenv("VISIBILITY_OS_DEFAULT_SLACK_CHANNEL", "#builds")
    from plugins.visibility_os.core.config import get_visibility_config

    cfg = get_visibility_config()

    assert cfg.company_name == "Acme Robotics"
    assert cfg.github_orgs == {"acme-inc", "acme-labs"}
    assert cfg.github_repos == ["acme-inc/api", "acme-labs/web"]
    assert cfg.default_slack_channel == "#builds"
    assert cfg.github_repo_allowed("acme-inc/api")
    assert not cfg.github_repo_allowed("acme-inc/web-app")


def test_db_init_creates_profile_scoped_tables(tmp_path, monkeypatch):
    db = patch_db(tmp_path, monkeypatch)
    db.init_db()
    conn = sqlite3.connect(tmp_path / "visibility_os.db")
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"schema_migrations", "opportunities", "action_queue", "audit_log", "daily_summaries", "weekly_summaries", "scan_runs", "connector_state", "board_item_states"} <= names


def test_scoring_formula_and_clamping():
    from plugins.visibility_os.core.scoring import score_opportunity
    score = score_opportunity(impact=4, visibility=5, effort=4, safety=5, risk_penalty=0)
    assert score.priority_score == 31
    clamped = score_opportunity(impact=99, visibility=-1, effort=99, safety=-1, risk_penalty=99)
    assert (clamped.impact, clamped.visibility, clamped.effort, clamped.safety, clamped.risk_penalty) == (5, 0, 5, 1, 10)


def test_action_queue_state_machine_and_audit(tmp_path, monkeypatch):
    patch_db(tmp_path, monkeypatch)
    from plugins.visibility_os.core.actions import create_action, approve_action, reject_action, get_action, list_audit_log, execute_action_guard
    from plugins.visibility_os.core.policies import ActionPolicyError

    action = create_action(
        proposed_by_agent="communications_agent",
        action_type="slack_message",
        target_system="slack",
        target_location="#team-updates",
        title="Post CI update",
        summary="Draft update about CI flake",
        proposed_payload={"text": "I found a likely cause and am testing a fix. Evidence: https://github.com/org/repo/issues/1"},
        evidence_links=[{"type": "issue", "url": "https://github.com/org/repo/issues/1"}],
        risk_level="low",
        impact_score=4,
        visibility_score=4,
        effort_score=5,
    )
    assert action["status"] == "queued"
    approved = approve_action(action["id"], actor="reviewer")
    assert approved["status"] == "approved"
    assert execute_action_guard(action["id"])["id"] == action["id"]
    events = list_audit_log(action_id=action["id"])
    assert [e["event_type"] for e in events] == ["created", "approved"]

    rejected = create_action(
        proposed_by_agent="communications_agent",
        action_type="github_issue_comment",
        target_system="github",
        target_location="org/repo/issues/1",
        title="Comment",
        summary="Comment",
        proposed_payload={"body": "I found a likely cause."},
        evidence_links=[{"type": "issue", "url": "https://github.com/org/repo/issues/1"}],
        risk_level="low",
    )
    reject_action(rejected["id"], actor="reviewer", reason="Tone wrong")
    assert get_action(rejected["id"])["status"] == "rejected"
    with pytest.raises(ActionPolicyError):
        approve_action(rejected["id"], actor="reviewer")


def test_hard_blocked_actions_cannot_execute(tmp_path, monkeypatch):
    patch_db(tmp_path, monkeypatch)
    from plugins.visibility_os.core.actions import create_action, approve_action, execute_action_guard
    from plugins.visibility_os.core.policies import ActionPolicyError
    action = create_action(
        proposed_by_agent="implementation_agent",
        action_type="production_deploy",
        target_system="deployment",
        target_location="prod",
        title="Deploy",
        summary="Deploy",
        proposed_payload={"ref": "main"},
        evidence_links=[{"type": "pr", "url": "https://github.com/org/repo/pull/1"}],
        risk_level="critical",
    )
    approve_action(action["id"], actor="reviewer")
    with pytest.raises(ActionPolicyError):
        execute_action_guard(action["id"])


def test_executor_rejects_unsupported_slack_action_and_overclaim_before_write(tmp_path, monkeypatch):
    patch_db(tmp_path, monkeypatch)
    from plugins.visibility_os.core.actions import create_action, approve_action, list_audit_log
    from plugins.visibility_os.core.executors import execute_approved_action

    action = create_action(
        proposed_by_agent="communications_agent",
        action_type="weekly_update_draft",
        target_system="slack",
        target_location="#team-updates",
        title="Weekly draft",
        summary="Draft only",
        proposed_payload={"text": "I found a likely cause. https://github.com/org/repo/issues/1"},
        evidence_links=[{"type": "issue", "url": "https://github.com/org/repo/issues/1"}],
        risk_level="medium",
    )
    approve_action(action["id"], actor="reviewer")
    with pytest.raises(Exception, match="Unsupported Slack action type"):
        execute_approved_action(action["id"], actor="reviewer")
    assert any(e["event_type"] == "execution_attempt_started" for e in list_audit_log(action_id=action["id"]))

    overclaim = create_action(
        proposed_by_agent="communications_agent",
        action_type="slack_message",
        target_system="slack",
        target_location="#team-updates",
        title="Overclaim",
        summary="Should not post",
        proposed_payload={"text": "Fixed and shipped the CI flake. https://github.com/org/repo/pull/1"},
        evidence_links=[{"type": "pr", "url": "https://github.com/org/repo/pull/1"}],
        risk_level="medium",
    )
    approve_action(overclaim["id"], actor="reviewer")
    with pytest.raises(Exception, match="before work is executed"):
        execute_approved_action(overclaim["id"], actor="reviewer")


def test_create_action_cannot_bypass_approval_state(tmp_path, monkeypatch):
    patch_db(tmp_path, monkeypatch)
    from plugins.visibility_os.core.actions import create_action
    from plugins.visibility_os.core.policies import ActionPolicyError
    with pytest.raises(ActionPolicyError):
        create_action(
            proposed_by_agent="bad_agent",
            action_type="slack_message",
            target_system="slack",
            target_location="#team-updates",
            title="Bypass",
            summary="Bypass",
            proposed_payload={"text": "hello"},
            evidence_links=[],
            risk_level="low",
            status="approved",
        )


def test_language_guard_blocks_overclaiming_without_execution():
    from plugins.visibility_os.core.language_guard import validate_message, LanguageGuardError
    with pytest.raises(LanguageGuardError):
        validate_message("Fixed and shipped the CI flake", status="queued", evidence_links=[{"url": "https://github.com/org/repo/pull/1"}])
    validate_message("I found a likely cause and am testing a fix. https://github.com/org/repo/issues/1", status="queued", evidence_links=[{"url": "https://github.com/org/repo/issues/1"}], team_visible=True)


def test_evidence_builder_requires_completion_fields():
    from plugins.visibility_os.core.evidence import EvidencePackage, EvidenceError
    pkg = EvidencePackage(problem_statement="CI flakes", tests_run="pytest tests/auth -q", evidence_links=[{"type": "pr", "url": "https://github.com/org/repo/pull/1"}], actual_status="executed")
    assert pkg.validate_for_completion() is True
    with pytest.raises(EvidenceError):
        EvidencePackage(problem_statement="CI flakes", actual_status="drafted").validate_for_completion()


def test_communications_drafter_queues_safe_action(tmp_path, monkeypatch):
    patch_db(tmp_path, monkeypatch)
    from plugins.visibility_os.core.communications import draft_progress_update
    action = draft_progress_update(
        problem="Auth CI is flaky",
        diagnosis="mock readiness can race first request",
        action_underway="testing explicit readiness wait",
        next_step="open PR with before/after CI output",
        target_location="#team-updates",
        evidence_links=[{"type": "issue", "url": "https://github.com/org/repo/issues/1"}],
    )
    assert action["action_type"] == "slack_message"
    assert action["status"] == "queued"
    assert "Problem:" in action["proposed_payload"]["text"]


def test_opportunity_detail_explains_scores_and_drafts_actions(tmp_path, monkeypatch):
    patch_db(tmp_path, monkeypatch)
    from plugins.visibility_os.core.opportunities import upsert_opportunity
    from plugins.visibility_os.core.opportunity_actions import build_opportunity_detail, draft_action_from_opportunity

    opportunity = upsert_opportunity(
        source_system="github",
        source_url="https://github.com/acme-inc/platform/pull/623",
        title="READY remove gradual hedging and fix flow",
        description="PR #623 may need review acceleration.",
        category="pr_review_acceleration",
        impact_score=4,
        visibility_score=4,
        effort_score=5,
        safety_score=5,
        risk_penalty=0,
        priority_score=30,
        suggested_artifacts=["review_comment", "slack_update"],
        metadata={"number": 623, "author": {"login": "engineer-a"}, "reviewDecision": "REVIEW_REQUIRED"},
    )

    detail = build_opportunity_detail(opportunity["id"])
    assert detail["id"] == opportunity["id"]
    assert detail["source_repo"] == "acme-inc/platform"
    assert detail["recommended_actions"][0]["action_kind"] == "github_pr_comment"
    assert detail["recommended_actions"][0]["label"] == "Draft PR coordination comment"
    assert any(e["url"] == opportunity["source_url"] for e in detail["evidence_links"])
    assert "priority_score" in detail["score_explanation"]

    github_action = draft_action_from_opportunity(opportunity["id"], action_kind="github_pr_comment")
    assert github_action["status"] == "queued"
    assert github_action["opportunity_id"] == opportunity["id"]
    assert github_action["target_system"] == "github"
    assert github_action["target_location"] == opportunity["source_url"]
    assert "Coordination note" in github_action["proposed_payload"]["body"]
    assert "not performed a code review" in github_action["proposed_payload"]["body"]

    slack_action = draft_action_from_opportunity(opportunity["id"], action_kind="slack_update", target_location="#team-updates")
    assert slack_action["target_system"] == "slack"
    assert slack_action["target_location"] == "#team-updates"
    assert "Next step:" in slack_action["proposed_payload"]["text"]


def test_api_routes_cover_opportunity_detail_and_action_drafting(tmp_path, monkeypatch):
    patch_db(tmp_path, monkeypatch)
    from plugins.visibility_os.core.opportunities import upsert_opportunity
    from plugins.visibility_os.dashboard.plugin_api import router
    app = FastAPI()
    app.include_router(router, prefix="/api/plugins/visibility-os")
    client = TestClient(app)
    opportunity = upsert_opportunity(
        source_system="github",
        source_url="https://github.com/acme-inc/infrastructure/pull/657",
        title="revert: SES wildcard back to specific noreply@ identity",
        description="PR #657 may need review acceleration.",
        category="pr_review_acceleration",
        impact_score=4,
        visibility_score=4,
        effort_score=5,
        safety_score=5,
        risk_penalty=0,
        priority_score=30,
        suggested_artifacts=["review_comment", "slack_update"],
        metadata={"number": 657, "reviewDecision": "REVIEW_REQUIRED"},
    )

    detail = client.get(f"/api/plugins/visibility-os/opportunities/{opportunity['id']}")
    assert detail.status_code == 200
    assert detail.json()["source_repo"] == "acme-inc/infrastructure"

    drafted = client.post(f"/api/plugins/visibility-os/opportunities/{opportunity['id']}/draft-action", json={"action_kind": "github_pr_comment"})
    assert drafted.status_code == 200
    assert drafted.json()["action_type"] == "github_pr_comment"
    assert drafted.json()["status"] == "queued"


def test_pr_audit_from_opportunity_queues_line_level_findings(tmp_path, monkeypatch):
    patch_db(tmp_path, monkeypatch)
    from plugins.visibility_os.core.opportunities import upsert_opportunity
    from plugins.visibility_os.core.pr_audit import audit_pr_from_opportunity

    opportunity = upsert_opportunity(
        source_system="github",
        source_url="https://github.com/acme-inc/platform/pull/623",
        title="READY remove gradual hedging and fix flow",
        description="PR #623 may need review acceleration.",
        category="pr_review_acceleration",
        impact_score=4,
        visibility_score=4,
        effort_score=5,
        safety_score=5,
        risk_penalty=0,
        priority_score=30,
        suggested_artifacts=["review_comment"],
        metadata={"number": 623},
    )

    def fake_run(args, capture_output, text, timeout, check):
        class R:
            returncode = 0
            stderr = ""
            stdout = ""
        r = R()
        joined = " ".join(args)
        if "pr diff" in joined:
            r.stdout = '''diff --git a/src/orders.py b/src/orders.py
--- a/src/orders.py
+++ b/src/orders.py
@@ -40,6 +40,8 @@ def load_order(user_id):
     query = f"SELECT * FROM orders WHERE user_id = {user_id}"
+    cursor.execute(query)
+    return cursor.fetchall()
diff --git a/src/shell.py b/src/shell.py
--- a/src/shell.py
+++ b/src/shell.py
@@ -10,4 +10,5 @@ def clean(path):
+    subprocess.run(f"rm -rf {path}", shell=True)
'''
        elif "pr view" in joined:
            r.stdout = '{"headRefOid":"abc123","title":"READY remove gradual hedging and fix flow"}'
        else:
            r.stdout = ""
        return r

    monkeypatch.setattr("subprocess.run", fake_run)
    action = audit_pr_from_opportunity(opportunity["id"])
    payload = action["proposed_payload"]
    assert action["action_type"] == "github_pr_review_draft"
    assert payload["verdict"] == "REQUEST_CHANGES"
    sql_finding = next(f for f in payload["findings"] if f["title"] == "SQL injection risk from formatted query")
    assert sql_finding["severity"] == "critical"
    assert sql_finding["path"] == "src/orders.py"
    assert sql_finding["line"] >= 40
    assert "parameterized" in sql_finding["solution"].lower()
    assert sql_finding["source"] == "deterministic"
    assert sql_finding["status"] == "open"
    assert "SQL injection safe" in sql_finding["casual_comment"]
    assert "—" not in sql_finding["casual_comment"]
    assert "–" not in sql_finding["casual_comment"]
    assert payload["findings_summary"]["critical"] >= 1
    assert payload["findings_by_file"]["src/orders.py"] >= 1
    assert payload["findings"][0]["github_diff_url"].startswith("https://github.com/acme-inc/platform/pull/623/files")
    assert any(f["path"] == "src/shell.py" and f["severity"] == "critical" for f in payload["findings"])


def test_github_actions_opportunity_can_queue_ci_diagnosis(tmp_path, monkeypatch):
    patch_db(tmp_path, monkeypatch)
    from plugins.visibility_os.core.opportunities import upsert_opportunity
    from plugins.visibility_os.core.opportunity_actions import build_opportunity_detail, draft_action_from_opportunity

    opportunity = upsert_opportunity(
        source_system="github",
        source_url="https://github.com/acme-inc/web-app/actions/runs/25928548133",
        title="Failing CI: PENDING: Email verification UI for whitelist signup",
        description="A recent GitHub Actions run failed and may be a visible unblock opportunity.",
        category="flaky_tests_and_ci_failures",
        impact_score=4,
        visibility_score=4,
        effort_score=4,
        safety_score=5,
        risk_penalty=0,
        priority_score=29,
        suggested_artifacts=["ci_before_after"],
        metadata={"databaseId": 25928548133, "workflowName": "test", "conclusion": "failure"},
    )

    detail = build_opportunity_detail(opportunity["id"])
    assert any(a["action_kind"] == "github_actions_diagnosis" and a["label"] == "Diagnose CI" for a in detail["recommended_actions"])

    def fake_run(args, capture_output, text, timeout, check):
        class R:
            returncode = 0
            stderr = ""
            stdout = ""
        r = R()
        if args[:3] == ["gh", "run", "view"] and "--json" in args:
            r.stdout = json.dumps({
                "name": "PENDING: Email verification UI for whitelist signup",
                "conclusion": "failure",
                "status": "completed",
                "event": "pull_request",
                "headBranch": "email-verification-ui",
                "headSha": "abc123",
                "jobs": [{"name": "build", "conclusion": "failure", "steps": [{"name": "npm test", "conclusion": "failure"}]}],
                "url": opportunity["source_url"],
            })
        elif args[:3] == ["gh", "pr", "list"]:
            r.stdout = json.dumps([{
                "number": 132,
                "title": "PENDING: Email verification UI for whitelist signup",
                "state": "MERGED",
                "url": "https://github.com/acme-inc/web-app/pull/132",
                "headRefName": "email-verification-ui",
                "headRefOid": "fixed456",
                "baseRefName": "main",
                "mergedAt": "2026-05-15T16:34:49Z",
                "statusCheckRollup": [{"name": "typecheck", "conclusion": "SUCCESS", "status": "COMPLETED"}],
            }])
        else:
            r.stdout = "build\tnpm test\tError: expected verification banner\nbuild\tnpm test\tAssertionError: banner missing"
        return r

    monkeypatch.setattr("subprocess.run", fake_run)
    action = draft_action_from_opportunity(opportunity["id"], action_kind="github_actions_diagnosis", actor="human")
    assert action["action_type"] == "github_actions_diagnosis"
    assert action["target_system"] == "github"
    assert action["status"] == "queued"
    payload = action["proposed_payload"]
    assert payload["run_id"] == "25928548133"
    assert payload["repo"] == "acme-inc/web-app"
    assert payload["diagnosis"]["failed_jobs"] == ["build"]
    assert payload["pr_context"]["number"] == 132
    assert payload["pr_context"]["state"] == "MERGED"
    assert payload["pr_context"]["branch"] == "email-verification-ui"
    assert payload["resolution_status"] == "resolved_merged"
    assert payload["should_create_fix_branch"] is False
    assert "already been merged" in payload["body"]
    assert "expected verification banner" in payload["body"]
    assert payload["next_steps"][0].startswith("No new fix branch")


def test_actionable_github_actions_opportunity_can_queue_fix_ci_lane(tmp_path, monkeypatch):
    patch_db(tmp_path, monkeypatch)
    from plugins.visibility_os.core.opportunities import upsert_opportunity
    from plugins.visibility_os.core.opportunity_actions import build_opportunity_detail, draft_action_from_opportunity

    opportunity = upsert_opportunity(
        source_system="github",
        source_url="https://github.com/acme-inc/web-app/actions/runs/26000000001",
        title="Failing CI: Typecheck",
        description="A recent GitHub Actions run failed and may be a visible unblock opportunity.",
        category="flaky_tests_and_ci_failures",
        impact_score=4,
        visibility_score=4,
        effort_score=4,
        safety_score=5,
        risk_penalty=0,
        priority_score=29,
        suggested_artifacts=["ci_before_after", "pull_request"],
        metadata={"databaseId": 26000000001, "workflowName": "Typecheck", "conclusion": "failure"},
    )

    def fake_run(args, capture_output, text, timeout, check):
        class R:
            returncode = 0
            stderr = ""
            stdout = ""
        r = R()
        if args[:3] == ["gh", "run", "view"] and "--json" in args:
            r.stdout = json.dumps({
                "name": "Typecheck",
                "conclusion": "failure",
                "status": "completed",
                "event": "pull_request",
                "headBranch": "feat/broken-types",
                "headSha": "abc123",
                "jobs": [{"name": "typecheck", "conclusion": "failure", "steps": [{"name": "npm run typecheck", "conclusion": "failure"}]}],
                "url": opportunity["source_url"],
            })
        elif args[:3] == ["gh", "pr", "list"]:
            r.stdout = json.dumps([{
                "number": 144,
                "title": "Add typed signup flow",
                "state": "OPEN",
                "url": "https://github.com/acme-inc/web-app/pull/144",
                "headRefName": "feat/broken-types",
                "headRefOid": "abc123",
                "baseRefName": "main",
                "mergedAt": None,
                "statusCheckRollup": [{"name": "Typecheck", "conclusion": "FAILURE", "status": "COMPLETED"}],
            }])
        else:
            r.stdout = "typecheck\tnpm run typecheck\tError: Type 'string' is not assignable to type 'Address'"
        return r

    monkeypatch.setattr("subprocess.run", fake_run)
    detail = build_opportunity_detail(opportunity["id"])
    assert any(a["action_kind"] == "ci_fix_lane" and a["label"] == "Start Fix CI lane" for a in detail["recommended_actions"])

    action = draft_action_from_opportunity(opportunity["id"], action_kind="ci_fix_lane", actor="human")
    assert action["action_type"] == "ci_fix_lane"
    assert action["target_system"] == "hermes"
    assert action["risk_level"] == "high"
    payload = action["proposed_payload"]
    assert payload["lane"] == "fix_ci"
    assert payload["repo"] == "acme-inc/web-app"
    assert payload["run_id"] == "26000000001"
    assert payload["pr_context"]["number"] == 144
    assert payload["should_create_fix_branch"] is True
    assert "Type 'string' is not assignable" in payload["prompt"]
    assert "Do not deploy" in payload["prompt"]
    assert "self-audit" in payload["prompt"].lower()
    assert "audit_status" in payload["prompt"]
    assert "self_audit_before_push" in payload["safety_gates"]


def test_fix_ci_lane_refuses_stale_or_resolved_ci(tmp_path, monkeypatch):
    patch_db(tmp_path, monkeypatch)
    from plugins.visibility_os.core.opportunities import upsert_opportunity
    from plugins.visibility_os.core.opportunity_actions import draft_action_from_opportunity

    opportunity = upsert_opportunity(
        source_system="github",
        source_url="https://github.com/acme-inc/web-app/actions/runs/25928548133",
        title="Failing CI: already merged",
        description="A historical failed run.",
        category="flaky_tests_and_ci_failures",
        impact_score=4,
        visibility_score=4,
        effort_score=4,
        safety_score=5,
        risk_penalty=0,
        priority_score=29,
        suggested_artifacts=["ci_before_after"],
        metadata={"databaseId": 25928548133},
    )

    def fake_run(args, capture_output, text, timeout, check):
        class R:
            returncode = 0
            stderr = ""
            stdout = ""
        r = R()
        if args[:3] == ["gh", "run", "view"] and "--json" in args:
            r.stdout = json.dumps({"conclusion": "failure", "status": "completed", "headBranch": "done", "headSha": "old", "jobs": []})
        elif args[:3] == ["gh", "pr", "list"]:
            r.stdout = json.dumps([{"number": 132, "state": "MERGED", "url": "https://github.com/acme-inc/web-app/pull/132", "headRefName": "done", "headRefOid": "new", "baseRefName": "main", "mergedAt": "2026-05-15T16:34:49Z", "statusCheckRollup": []}])
        else:
            r.stdout = "test\tstep\tError: old failure"
        return r

    monkeypatch.setattr("subprocess.run", fake_run)
    with pytest.raises(ValueError, match="not actionable"):
        draft_action_from_opportunity(opportunity["id"], action_kind="ci_fix_lane", actor="human")


def test_hermes_executor_prepares_fix_and_queues_push_branch_action(tmp_path, monkeypatch):
    patch_db(tmp_path, monkeypatch)
    from plugins.visibility_os.core.actions import create_action, approve_action, list_actions
    from plugins.visibility_os.core.executors import execute_approved_action

    calls = []

    def fake_run(args, capture_output, text, timeout, check, cwd=None, input=None):
        calls.append({"args": args, "cwd": cwd, "input": input})
        class R:
            returncode = 0
            stderr = ""
            stdout = ""
        r = R()
        if "visibility-os-ci-review" in args:
            r.stdout = json.dumps({
                "review_status": "passed",
                "findings": [],
                "fixes_required": [],
                "notes": "Independent fresh-session review found no blockers.",
            })
        else:
            r.stdout = json.dumps({
                "branch": "fix/ci-typecheck-address",
                "commit_sha": "abc987",
                "commit_message": "fix: repair CI type mismatch",
                "pr_title": "Fix CI type mismatch in signup flow",
                "pr_body": "## What changed\n- Fixed Address typing\n\n## Verification\n- npm run typecheck",
                "verification": ["npm run typecheck"],
                "changed_files": ["src/routes/__root.tsx"],
                "self_audit": {"audit_status": "passed", "issues_found": [], "fixes_applied": [], "notes": "Second-pass review found no issues."},
                "ready_to_push": True,
            })
        return r

    monkeypatch.setattr("subprocess.run", fake_run)
    action = create_action(
        proposed_by_agent="human",
        action_type="ci_fix_lane",
        target_system="hermes",
        target_location="acme-inc/web-app#26000000001",
        title="Start Fix CI lane",
        summary="Prepare a local CI repair branch.",
        proposed_payload={
            "lane": "fix_ci",
            "repo": "acme-inc/web-app",
            "run_id": "26000000001",
            "prompt": "Fix the actionable CI failure. Do not push. Do not deploy.",
            "command": ["hermes", "chat", "--query", "__PROMPT__", "--quiet", "--source", "visibility-os-ci-fix", "--toolsets", "terminal,file"],
            "workdir": str(tmp_path),
        },
        evidence_links=[{"type": "run", "url": "https://github.com/acme-inc/web-app/actions/runs/26000000001"}],
        risk_level="high",
    )
    approve_action(action["id"], actor="reviewer")
    executed = execute_approved_action(action["id"], actor="reviewer")

    assert executed["status"] == "executed"
    assert calls[0]["cwd"] == str(tmp_path)
    assert calls[0]["args"][3] == "Fix the actionable CI failure. Do not push. Do not deploy."
    assert len([c for c in calls if c["args"] and c["args"][0] == "hermes"]) == 2
    assert any("visibility-os-ci-review" in c["args"] for c in calls)
    review_call = next(c for c in calls if "visibility-os-ci-review" in c["args"])
    assert "Fix the actionable CI failure" not in review_call["args"][3]
    assert "fresh session" in review_call["args"][3].lower()
    result = executed["execution_result"]
    assert result["prepared_branch"] == "fix/ci-typecheck-address"
    assert result["push_action_id"]

    push_action = next(a for a in list_actions() if a["id"] == result["push_action_id"])
    assert push_action["status"] == "queued"
    assert push_action["action_type"] == "github_push_branch"
    assert push_action["target_system"] == "github"
    assert push_action["proposed_payload"]["branch"] == "fix/ci-typecheck-address"
    assert push_action["proposed_payload"]["pr_title"] == "Fix CI type mismatch in signup flow"
    assert push_action["proposed_payload"]["self_audit"]["audit_status"] == "passed"
    assert push_action["proposed_payload"]["independent_review"]["review_status"] == "passed"
    assert result["self_audit"]["audit_status"] == "passed"
    assert result["independent_review"]["review_status"] == "passed"
    assert "npm run typecheck" in push_action["proposed_payload"]["pr_body"]



def test_issue_opportunity_exposes_fix_lane_and_builds_payload(tmp_path, monkeypatch):
    patch_db(tmp_path, monkeypatch)
    from plugins.visibility_os.core.opportunities import upsert_opportunity
    from plugins.visibility_os.core.opportunity_actions import build_opportunity_detail, draft_action_from_opportunity

    opportunity = upsert_opportunity(
        source_system="github",
        source_url="https://github.com/acme-inc/web-app/issues/42",
        title="Bug: settings save fails on mobile",
        description="Issue #42 appears relevant for small customer facing bug fixes.",
        category="small_customer_facing_bug_fixes",
        impact_score=4,
        visibility_score=4,
        effort_score=4,
        safety_score=5,
        risk_penalty=0,
        priority_score=29,
        suggested_artifacts=["pull_request", "issue_comment", "slack_update"],
        metadata={
            "number": 42,
            "title": "Bug: settings save fails on mobile",
            "body": "Saving settings on mobile returns a validation error.",
            "labels": [{"name": "bug"}],
            "url": "https://github.com/acme-inc/web-app/issues/42",
        },
    )

    detail = build_opportunity_detail(opportunity["id"])
    assert any(a["action_kind"] == "github_issue_fix_lane" and a["label"] == "Fix issue" for a in detail["recommended_actions"])

    action = draft_action_from_opportunity(opportunity["id"], action_kind="github_issue_fix_lane", actor="human")
    assert action["action_type"] == "github_issue_fix_lane"
    assert action["target_system"] == "hermes"
    assert action["risk_level"] == "high"
    payload = action["proposed_payload"]
    assert payload["lane"] == "fix_github_issue"
    assert payload["repo"] == "acme-inc/web-app"
    assert payload["issue_number"] == 42
    assert payload["issue_url"] == opportunity["source_url"]
    assert "Fix GitHub issue #42" in payload["prompt"]
    assert "Saving settings on mobile" in payload["prompt"]
    assert "Do not push" in payload["prompt"]
    assert "visibility-os-issue-fix" in payload["command"]


def test_hermes_executor_prepares_issue_fix_and_queues_push_branch_action(tmp_path, monkeypatch):
    patch_db(tmp_path, monkeypatch)
    from plugins.visibility_os.core.actions import create_action, approve_action, list_actions
    from plugins.visibility_os.core.executors import execute_approved_action

    calls = []

    def fake_run(args, capture_output, text, timeout, check, cwd=None, input=None):
        calls.append({"args": args, "cwd": cwd, "input": input})
        class R:
            returncode = 0
            stderr = ""
            stdout = ""
        r = R()
        if "visibility-os-issue-review" in args:
            r.stdout = json.dumps({
                "review_status": "passed",
                "findings": [],
                "fixes_required": [],
                "notes": "Fresh-session review found no blockers.",
            })
        else:
            r.stdout = json.dumps({
                "branch": "fix/issue-42-settings-save-mobile",
                "commit_sha": "def456",
                "commit_message": "fix: repair mobile settings save",
                "pr_title": "Fix mobile settings save validation",
                "pr_body": "## What changed\n- Fixed mobile settings validation\n\nFixes #42\n\n## Verification\n- pytest tests/settings",
                "verification": ["pytest tests/settings"],
                "changed_files": ["src/settings.py"],
                "self_audit": {"audit_status": "passed", "issues_found": [], "fixes_applied": [], "notes": "Second-pass review found no issues."},
                "ready_to_push": True,
            })
        return r

    monkeypatch.setattr("subprocess.run", fake_run)
    action = create_action(
        proposed_by_agent="human",
        action_type="github_issue_fix_lane",
        target_system="hermes",
        target_location="acme-inc/web-app#42",
        title="Fix GitHub issue #42",
        summary="Prepare a local issue fix branch.",
        proposed_payload={
            "lane": "fix_github_issue",
            "repo": "acme-inc/web-app",
            "issue_number": 42,
            "issue_url": "https://github.com/acme-inc/web-app/issues/42",
            "prompt": "Fix GitHub issue #42. Do not push. Do not deploy.",
            "command": ["hermes", "chat", "--query", "__PROMPT__", "--quiet", "--source", "visibility-os-issue-fix", "--toolsets", "terminal,file"],
            "workdir": str(tmp_path),
        },
        evidence_links=[{"type": "issue", "url": "https://github.com/acme-inc/web-app/issues/42"}],
        risk_level="high",
    )
    approve_action(action["id"], actor="reviewer")
    executed = execute_approved_action(action["id"], actor="reviewer")

    assert executed["status"] == "executed"
    assert len([c for c in calls if c["args"] and c["args"][0] == "hermes"]) == 2
    review_call = next(c for c in calls if "visibility-os-issue-review" in c["args"])
    assert "Fix GitHub issue #42" not in review_call["args"][3]
    assert "fresh session" in review_call["args"][3].lower()
    result = executed["execution_result"]
    assert result["lane"] == "fix_github_issue"
    assert result["prepared_branch"] == "fix/issue-42-settings-save-mobile"
    assert result["push_action_id"]

    push_action = next(a for a in list_actions() if a["id"] == result["push_action_id"])
    assert push_action["status"] == "queued"
    assert push_action["action_type"] == "github_push_branch"
    assert push_action["proposed_payload"]["issue_number"] == 42
    assert push_action["proposed_payload"]["issue_url"] == "https://github.com/acme-inc/web-app/issues/42"
    assert push_action["proposed_payload"]["independent_review"]["review_status"] == "passed"
    assert "Fixes #42" in push_action["proposed_payload"]["pr_body"]


def test_api_one_click_fix_ci_drafts_approves_executes_and_queues_push(tmp_path, monkeypatch):
    patch_db(tmp_path, monkeypatch)
    from plugins.visibility_os.core.opportunities import upsert_opportunity
    from plugins.visibility_os.core.actions import list_actions
    from plugins.visibility_os.dashboard.plugin_api import router

    app = FastAPI()
    app.include_router(router, prefix="/api/plugins/visibility-os")
    client = TestClient(app)
    opportunity = upsert_opportunity(
        source_system="github",
        source_url="https://github.com/acme-inc/web-app/actions/runs/26000000001",
        title="Failing CI: Typecheck",
        description="A recent GitHub Actions run failed.",
        category="flaky_tests_and_ci_failures",
        impact_score=4,
        visibility_score=4,
        effort_score=4,
        safety_score=5,
        risk_penalty=0,
        priority_score=29,
        suggested_artifacts=["ci_before_after", "pull_request"],
        metadata={"databaseId": 26000000001},
    )

    def fake_run(args, capture_output, text, timeout, check, cwd=None, input=None):
        class R:
            returncode = 0
            stderr = ""
            stdout = ""
        r = R()
        if args[:3] == ["gh", "run", "view"] and "--json" in args:
            r.stdout = json.dumps({"conclusion": "failure", "status": "completed", "event": "pull_request", "headBranch": "feat/broken-types", "headSha": "abc123", "jobs": [], "url": opportunity["source_url"]})
        elif args[:3] == ["gh", "pr", "list"]:
            r.stdout = json.dumps([{"number": 144, "title": "Add typed signup flow", "state": "OPEN", "url": "https://github.com/acme-inc/web-app/pull/144", "headRefName": "feat/broken-types", "headRefOid": "abc123", "baseRefName": "main", "mergedAt": None, "statusCheckRollup": [{"name": "Typecheck", "conclusion": "FAILURE", "status": "COMPLETED"}]}])
        elif args and args[0] == "hermes" and "visibility-os-ci-review" in args:
            r.stdout = json.dumps({"review_status": "passed", "findings": [], "fixes_required": [], "notes": "Independent review passed"})
        elif args and args[0] == "hermes":
            r.stdout = json.dumps({
                "branch": "fix/ci-typecheck-address",
                "commit_sha": "abc987",
                "commit_message": "fix: repair CI type mismatch",
                "pr_title": "Fix CI type mismatch in signup flow",
                "pr_body": "## Verification\n- npm run typecheck",
                "verification": ["npm run typecheck"],
                "changed_files": ["src/routes/__root.tsx"],
                "self_audit": {"audit_status": "passed", "issues_found": [], "fixes_applied": [], "notes": "ok"},
                "ready_to_push": True,
            })
        else:
            r.stdout = "typecheck\tnpm run typecheck\tError: Type 'string' is not assignable to type 'Address'"
        return r

    monkeypatch.setattr("subprocess.run", fake_run)
    response = client.post(f"/api/plugins/visibility-os/opportunities/{opportunity['id']}/fix-ci", json={"actor": "reviewer"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "executed"
    assert body["action_type"] == "ci_fix_lane"
    assert body["execution_result"]["prepared_branch"] == "fix/ci-typecheck-address"

    push_actions = [a for a in list_actions() if a["action_type"] == "github_push_branch"]
    assert len(push_actions) == 1
    assert push_actions[0]["status"] == "queued"
    assert push_actions[0]["proposed_payload"]["self_audit"]["audit_status"] == "passed"
    assert push_actions[0]["proposed_payload"]["independent_review"]["review_status"] == "passed"



def test_api_one_click_fix_issue_drafts_approves_executes_and_queues_push(tmp_path, monkeypatch):
    patch_db(tmp_path, monkeypatch)
    from plugins.visibility_os.core.opportunities import upsert_opportunity
    from plugins.visibility_os.core.actions import list_actions
    from plugins.visibility_os.dashboard.plugin_api import router

    app = FastAPI()
    app.include_router(router, prefix="/api/plugins/visibility-os")
    client = TestClient(app)
    opportunity = upsert_opportunity(
        source_system="github",
        source_url="https://github.com/acme-inc/web-app/issues/42",
        title="Bug: settings save fails on mobile",
        description="Issue #42 appears relevant for small customer facing bug fixes.",
        category="small_customer_facing_bug_fixes",
        impact_score=4,
        visibility_score=4,
        effort_score=4,
        safety_score=5,
        risk_penalty=0,
        priority_score=29,
        suggested_artifacts=["pull_request", "issue_comment"],
        metadata={"number": 42, "title": "Bug: settings save fails on mobile", "body": "Saving settings on mobile returns a validation error.", "labels": [{"name": "bug"}]},
    )

    def fake_run(args, capture_output, text, timeout, check, cwd=None, input=None):
        class R:
            returncode = 0
            stderr = ""
            stdout = ""
        r = R()
        if args and args[0] == "hermes" and "visibility-os-issue-review" in args:
            r.stdout = json.dumps({"review_status": "passed", "findings": [], "fixes_required": [], "notes": "Independent review passed"})
        elif args and args[0] == "hermes":
            r.stdout = json.dumps({
                "branch": "fix/issue-42-settings-save-mobile",
                "commit_sha": "def456",
                "commit_message": "fix: repair mobile settings save",
                "pr_title": "Fix mobile settings save validation",
                "pr_body": "## Verification\n- pytest tests/settings\n\nFixes #42",
                "verification": ["pytest tests/settings"],
                "changed_files": ["src/settings.py"],
                "self_audit": {"audit_status": "passed", "issues_found": [], "fixes_applied": [], "notes": "ok"},
                "ready_to_push": True,
            })
        return r

    monkeypatch.setattr("subprocess.run", fake_run)
    response = client.post(f"/api/plugins/visibility-os/opportunities/{opportunity['id']}/fix-issue", json={"actor": "reviewer"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "executed"
    assert body["action_type"] == "github_issue_fix_lane"
    assert body["execution_result"]["lane"] == "fix_github_issue"
    assert body["execution_result"]["issue_number"] == 42

    push_actions = [a for a in list_actions() if a["action_type"] == "github_push_branch"]
    assert len(push_actions) == 1
    assert push_actions[0]["status"] == "queued"
    assert push_actions[0]["proposed_payload"]["issue_number"] == 42
    assert push_actions[0]["proposed_payload"]["independent_review"]["review_status"] == "passed"


def test_github_executor_pushes_prepared_branch_and_creates_pr(tmp_path, monkeypatch):
    patch_db(tmp_path, monkeypatch)
    from plugins.visibility_os.core.actions import create_action, approve_action
    from plugins.visibility_os.core.executors import execute_approved_action

    calls = []

    def fake_run(args, capture_output, text, timeout, check, cwd=None):
        calls.append({"args": args, "cwd": cwd})
        class R:
            returncode = 0
            stderr = ""
            stdout = ""
        r = R()
        if args[:3] == ["gh", "pr", "create"]:
            r.stdout = "https://github.com/acme-inc/web-app/pull/145\n"
        return r

    monkeypatch.setattr("subprocess.run", fake_run)
    action = create_action(
        proposed_by_agent="visibility_os",
        action_type="github_push_branch",
        target_system="github",
        target_location="acme-inc/web-app#fix/ci-typecheck-address",
        title="Push prepared CI fix branch",
        summary="Push local branch and create PR.",
        proposed_payload={
            "repo": "acme-inc/web-app",
            "branch": "fix/ci-typecheck-address",
            "base_branch": "main",
            "workdir": str(tmp_path),
            "pr_title": "Fix CI type mismatch in signup flow",
            "pr_body": "## Verification\n- npm run typecheck",
        },
        evidence_links=[{"type": "run", "url": "https://github.com/acme-inc/web-app/actions/runs/26000000001"}],
        risk_level="high",
    )
    approve_action(action["id"], actor="reviewer")
    executed = execute_approved_action(action["id"], actor="reviewer")
    assert executed["status"] == "executed"
    assert calls[0]["args"] == ["git", "push", "-u", "origin", "fix/ci-typecheck-address"]
    assert calls[0]["cwd"] == str(tmp_path)
    assert calls[1]["args"][:3] == ["gh", "pr", "create"]
    assert executed["execution_result"]["pr_url"] == "https://github.com/acme-inc/web-app/pull/145"


def test_api_route_diagnoses_github_actions_opportunity(tmp_path, monkeypatch):
    patch_db(tmp_path, monkeypatch)
    from plugins.visibility_os.core.opportunities import upsert_opportunity
    from plugins.visibility_os.dashboard.plugin_api import router
    app = FastAPI()
    app.include_router(router, prefix="/api/plugins/visibility-os")
    client = TestClient(app)
    opportunity = upsert_opportunity(
        source_system="github",
        source_url="https://github.com/acme-inc/risk-engine/actions/runs/22456835234",
        title="Failing CI: Deploy Risk Engine - Staging",
        description="A recent GitHub Actions run failed and may be a visible unblock opportunity.",
        category="flaky_tests_and_ci_failures",
        impact_score=4,
        visibility_score=4,
        effort_score=4,
        safety_score=5,
        risk_penalty=0,
        priority_score=29,
        suggested_artifacts=["ci_before_after"],
        metadata={"databaseId": 22456835234},
    )

    def fake_run(args, capture_output, text, timeout, check):
        class R:
            returncode = 0
            stderr = ""
            stdout = ""
        r = R()
        r.stdout = json.dumps({"jobs": [{"name": "deploy", "conclusion": "failure", "steps": []}], "conclusion": "failure"}) if "--json" in args else "deploy\tterraform apply\tError: missing AWS credentials"
        return r

    monkeypatch.setattr("subprocess.run", fake_run)
    response = client.post(f"/api/plugins/visibility-os/opportunities/{opportunity['id']}/draft-action", json={"action_kind": "github_actions_diagnosis", "actor": "human"})
    assert response.status_code == 200
    body = response.json()
    assert body["action_type"] == "github_actions_diagnosis"
    assert "missing AWS credentials" in body["proposed_payload"]["body"]


def test_api_route_audits_pr_opportunity(tmp_path, monkeypatch):
    patch_db(tmp_path, monkeypatch)
    from plugins.visibility_os.core.opportunities import upsert_opportunity
    from plugins.visibility_os.dashboard.plugin_api import router
    app = FastAPI()
    app.include_router(router, prefix="/api/plugins/visibility-os")
    client = TestClient(app)
    opportunity = upsert_opportunity(
        source_system="github",
        source_url="https://github.com/acme-inc/infrastructure/pull/657",
        title="revert: SES wildcard back to specific noreply@ identity",
        description="PR #657 may need review acceleration.",
        category="pr_review_acceleration",
        impact_score=4,
        visibility_score=4,
        effort_score=5,
        safety_score=5,
        risk_penalty=0,
        priority_score=30,
        suggested_artifacts=["review_comment"],
        metadata={"number": 657},
    )

    def fake_run(args, capture_output, text, timeout, check):
        class R:
            returncode = 0
            stderr = ""
            stdout = ""
        r = R()
        r.stdout = '{"headRefOid":"abc123"}' if "view" in args else 'diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1,1 +1,2 @@\n+password = "supersecret"\n'
        return r

    monkeypatch.setattr("subprocess.run", fake_run)
    response = client.post(f"/api/plugins/visibility-os/opportunities/{opportunity['id']}/audit-pr", json={"actor": "human"})
    assert response.status_code == 200
    body = response.json()
    assert body["action_type"] == "github_pr_review_draft"
    assert body["proposed_payload"]["findings"][0]["path"] == "app.py"


def test_api_route_deep_reviews_pr_opportunity(tmp_path, monkeypatch):
    patch_db(tmp_path, monkeypatch)
    from plugins.visibility_os.core.opportunities import upsert_opportunity
    from plugins.visibility_os.dashboard.plugin_api import router
    app = FastAPI()
    app.include_router(router, prefix="/api/plugins/visibility-os")
    client = TestClient(app)
    opportunity = upsert_opportunity(
        source_system="github",
        source_url="https://github.com/acme-inc/platform/pull/700",
        title="update hedging flow",
        description="PR #700 may need review acceleration.",
        category="pr_review_acceleration",
        impact_score=4,
        visibility_score=4,
        effort_score=5,
        safety_score=5,
        risk_penalty=0,
        priority_score=30,
        suggested_artifacts=["review_comment"],
        metadata={"number": 700},
    )

    calls = []

    def fake_run(args, capture_output, text, timeout, check, cwd=None):
        calls.append((args, cwd))
        class R:
            returncode = 0
            stderr = ""
            stdout = ""
        r = R()
        if args[:3] == ["gh", "pr", "view"]:
            r.stdout = json.dumps({
                "headRefOid": "abc123",
                "changedFiles": 2,
                "additions": 25,
                "deletions": 3,
                "body": "Implements hedging change.",
                "files": [{"path": "src/hedging/flow.py"}, {"path": "README.md"}],
            })
        elif args[:3] == ["gh", "pr", "diff"]:
            r.stdout = "diff --git a/src/hedging/flow.py b/src/hedging/flow.py\n--- a/src/hedging/flow.py\n+++ b/src/hedging/flow.py\n@@ -1,1 +1,2 @@\n+def rebalance():\n+    return True\n"
        elif args[:3] == ["gh", "repo", "clone"] or args[:3] == ["gh", "pr", "checkout"]:
            r.stdout = ""
        elif args and args[0] == "hermes":
            r.stdout = json.dumps({
                "verdict": "COMMENT",
                "findings": [{
                    "severity": "warning",
                    "path": "src/hedging/flow.py",
                    "line": 1,
                    "title": "Agent found missing hedging edge-case coverage",
                    "detail": "The new rebalance path should be covered for empty positions.",
                    "solution": "Add a test for empty-position rebalance behavior.",
                    "code": "def rebalance():",
                    "casual_comment": "This rebalance path looks like it might skip empty-position handling — can we cover that before merging?",
                }],
                "review_notes": ["Checked out the PR branch locally and reviewed with Hermes agent."],
            })
        else:
            r.stdout = ""
        return r

    monkeypatch.setattr("subprocess.run", fake_run)
    response = client.post(f"/api/plugins/visibility-os/opportunities/{opportunity['id']}/deep-review-pr", json={"actor": "human"})
    assert response.status_code == 200
    body = response.json()
    assert body["action_type"] == "github_pr_review_draft"
    assert body["proposed_payload"]["review_depth"] == "deep"
    assert body["proposed_payload"]["agentic"] is True
    assert "Hermes Agentic Deep PR Review" in body["proposed_payload"]["body"]
    assert any(args and args[0] == "hermes" for args, _cwd in calls)
    assert any(args[:3] == ["gh", "repo", "clone"] for args, _cwd in calls)
    assert any(f["title"] == "Agent found missing hedging edge-case coverage" for f in body["proposed_payload"]["findings"])
    agent_finding = next(f for f in body["proposed_payload"]["findings"] if f["title"] == "Agent found missing hedging edge-case coverage")
    assert agent_finding["source"] == "agentic"
    assert "empty-position" in agent_finding["casual_comment"]
    assert "—" not in agent_finding["casual_comment"]
    assert "–" not in agent_finding["casual_comment"]
    assert agent_finding["github_diff_url"].endswith("#diff-src-hedging-flow-pyR1")


def test_visibility_os_dashboard_shows_single_fix_ci_button():
    js = Path("plugins/visibility_os/dashboard/dist/index.js").read_text()
    assert "fixCI" in js
    assert "Fix CI" in js
    assert "/fix-ci" in js
    assert "fixIssue" in js
    assert "Fix Issue" in js
    assert "/fix-issue" in js
    assert "Start Fix CI lane" not in js
    assert "Diagnose CI" not in js
    assert "github_actions_diagnosis" not in js
    assert "pushBranchNow" in js
    assert "Push branch" in js
    assert "Proposed PR" in js
    assert "Self-audit" in js
    assert "Independent review" in js
    assert "review_status" in js
    assert "github_push_branch" in js
    assert "item.can_diagnose_ci" in js
    assert "item.opportunity_id || item.id" in js
    assert "sectionActions" in js
    assert "Open opportunity" in js
    assert "viewOpportunity(item.id)" in js
    assert "ticketModalView" in js
    assert "opportunityModalView" in js
    assert "visibility-os-modal-backdrop" in js
    assert "role: 'dialog'" in js
    assert "aria-modal" in js
    assert "setSelectedTicket(item)" in js
    assert "onKeyDown" in js
    assert "Kanban board" in js
    assert "kanbanColumn('todo'" in js
    assert "kanbanColumn('in_progress'" in js
    assert "kanbanColumn('in_review'" in js
    assert "kanbanColumn('done'" in js
    assert "Move to In progress" in js
    assert "Move to Review" in js
    assert "Mark done" in js
    assert "Archive" in js
    assert "Show archived" in js
    assert "/api/plugins/visibility-os/board-state" in js
    assert "/api/plugins/visibility-os/config" in js
    assert "default_slack_channel" in js
    assert "#engineering" not in js


def test_visibility_os_config_endpoint_uses_env(tmp_path, monkeypatch):
    patch_db(tmp_path, monkeypatch)
    monkeypatch.setenv("VISIBILITY_OS_COMPANY_NAME", "Example Co")
    monkeypatch.setenv("VISIBILITY_OS_GITHUB_ORGS", "example-org")
    monkeypatch.setenv("VISIBILITY_OS_GITHUB_REPOS", "example-org/app,example-org/api")
    monkeypatch.setenv("VISIBILITY_OS_DEFAULT_SLACK_CHANNEL", "#ci-alerts")
    from plugins.visibility_os.dashboard.plugin_api import router
    app = FastAPI()
    app.include_router(router, prefix="/api/plugins/visibility-os")
    client = TestClient(app)

    response = client.get("/api/plugins/visibility-os/config")

    assert response.status_code == 200
    assert response.json() == {
        "company_name": "Example Co",
        "github_orgs": ["example-org"],
        "github_repos": ["example-org/app", "example-org/api"],
        "default_slack_channel": "#ci-alerts",
    }


def test_kanban_board_state_endpoint_moves_and_archives_items(tmp_path, monkeypatch):
    patch_db(tmp_path, monkeypatch)
    from plugins.visibility_os.core.opportunities import upsert_opportunity
    from plugins.visibility_os.dashboard.plugin_api import router

    app = FastAPI()
    app.include_router(router, prefix="/api/plugins/visibility-os")
    client = TestClient(app)
    opportunity = upsert_opportunity(
        source_system="github",
        source_url="https://github.com/acme-inc/web-app/issues/99",
        title="Bug: hard to read queue",
        description="Board should make work easier to scan.",
        category="dashboard_ux",
        impact_score=4,
        visibility_score=5,
        effort_score=3,
        safety_score=5,
        risk_penalty=0,
        priority_score=32,
        suggested_artifacts=["ui"],
        metadata={"number": 99},
    )

    feed = client.get("/api/plugins/visibility-os/feed").json()
    card = next(item for item in feed["items"] if item["id"] == opportunity["id"])
    assert card["board_state"] == "todo"
    assert feed["counts"]["board"]["todo"] >= 1

    moved = client.post("/api/plugins/visibility-os/board-state", json={"item_kind": "opportunity", "item_id": opportunity["id"], "board_state": "in_progress", "actor": "reviewer"})
    assert moved.status_code == 200
    feed = client.get("/api/plugins/visibility-os/feed").json()
    card = next(item for item in feed["items"] if item["id"] == opportunity["id"])
    assert card["board_state"] == "in_progress"
    assert card["board_state_actor"] == "reviewer"

    archived = client.post("/api/plugins/visibility-os/board-state", json={"item_kind": "opportunity", "item_id": opportunity["id"], "board_state": "archived", "actor": "reviewer"})
    assert archived.status_code == 200
    assert all(item["id"] != opportunity["id"] for item in client.get("/api/plugins/visibility-os/feed").json()["items"])
    archived_feed = client.get("/api/plugins/visibility-os/feed?include_archived=true").json()
    archived_card = next(item for item in archived_feed["items"] if item["id"] == opportunity["id"])
    assert archived_card["board_state"] == "archived"


def test_dashboard_plugin_manifest_entries_point_to_existing_assets():
    for manifest_path in Path("plugins").glob("*/dashboard/manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        entry = manifest.get("entry")
        if entry:
            assert (manifest_path.parent / entry).is_file(), f"{manifest_path} entry {entry} is missing"


def test_visibility_os_dashboard_does_not_render_object_payloads_as_react_children():
    node_script = r'''
const path = require('path');
const feed = {
  items: [{
    kind: 'action',
    id: 'act_1',
    title: 'Push prepared branch',
    summary: 'Prepared locally',
    status: 'queued',
    target_system: 'github',
    risk_level: 'high',
    action_type: 'github_push_branch',
    proposed_payload: {
      branch: 'fix/object-render',
      commit_message: 'fix: render changed files safely',
      pr_title: 'Fix object render',
      pr_body: '## Summary',
      changed_files: [{path: 'src/app.js', change: 'updated rendering'}],
      verification: [{path: 'pytest', change: 'passed'}],
      self_audit: {audit_status: 'passed', issues_found: [], fixes_applied: []},
      independent_review: {review_status: 'passed', findings: [], fixes_required: []}
    },
    evidence_links: []
  }],
  counts: {actions: 1, opportunities: 0}
};
function isElement(value) { return value && typeof value === 'object' && value.__element === true; }
function checkChild(child) {
  if (Array.isArray(child)) return child.forEach(checkChild);
  if (child === null || child === undefined || child === false || child === true) return;
  if (typeof child === 'object' && !isElement(child)) throw new Error('Plain object rendered: ' + JSON.stringify(child));
}
const React = {
  createElement: function(type, props) {
    const children = Array.prototype.slice.call(arguments, 2);
    children.forEach(checkChild);
    return {__element: true, type, props: props || {}, children};
  }
};
let stateCall = 0;
global.window = {
  __HERMES_PLUGIN_SDK__: {
    React,
    fetchJSON: function () { return Promise.resolve({}); },
    hooks: {
      useState: function(initial) { stateCall += 1; return [stateCall === 1 ? feed : initial, function () {}]; },
      useCallback: function(fn) { return fn; },
      useEffect: function() {}
    },
    components: {
      Card: 'Card', CardHeader: 'CardHeader', CardTitle: 'CardTitle', CardContent: 'CardContent', Badge: 'Badge'
    }
  },
  __HERMES_PLUGINS__: {
    register: function(name, Component) { Component(); }
  },
  prompt: function () { return null; }
};
global.navigator = {clipboard: {writeText: function () {}}};
require(path.resolve('plugins/visibility_os/dashboard/dist/index.js'));
'''
    result = subprocess.run(["node", "-e", node_script], cwd=Path.cwd(), text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_visibility_os_dashboard_renders_rich_audit_finding_controls():
    js = Path("plugins/visibility_os/dashboard/dist/index.js").read_text()
    for expected in [
        "severityFilter",
        "sourceFilter",
        "findingsSummaryView",
        "groupFindingsByFile",
        "copyFindingComment",
        "Open diff line",
        "markFindingStatus",
        "Agentic findings",
        "Deterministic findings",
    ]:
        assert expected in js


def test_visibility_os_dashboard_uses_readable_non_overlapping_action_buttons():
    js = Path("plugins/visibility_os/dashboard/dist/index.js").read_text()
    assert "function ActionButton" in js
    assert "tracking-normal" in js
    assert "normal-case" in js
    assert "whitespace-nowrap" in js
    assert "props && props.children" in js
    assert "letterSpacing: 'normal'" in js
    assert "textTransform: 'none'" in js
    assert "fontFamily: 'Arial, Helvetica, sans-serif'" in js
    assert "h(Button" not in js


def test_api_routes_cover_action_review_flow(tmp_path, monkeypatch):
    patch_db(tmp_path, monkeypatch)
    from plugins.visibility_os.dashboard.plugin_api import router
    app = FastAPI()
    app.include_router(router, prefix="/api/plugins/visibility-os")
    client = TestClient(app)
    create = client.post("/api/plugins/visibility-os/actions", json={
        "proposed_by_agent": "communications_agent",
        "action_type": "github_issue_comment",
        "target_system": "github",
        "target_location": "org/repo/issues/1",
        "title": "Diagnosis",
        "summary": "Post diagnosis",
        "proposed_payload": {"body": "I found a likely cause. https://github.com/org/repo/issues/1"},
        "evidence_links": [{"type": "issue", "url": "https://github.com/org/repo/issues/1"}],
        "risk_level": "low"
    })
    assert create.status_code == 200
    action_id = create.json()["id"]
    assert client.post(f"/api/plugins/visibility-os/actions/{action_id}/edit", json={"final_payload": {"body": "Edited body https://github.com/org/repo/issues/1"}, "actor": "reviewer"}).status_code == 200
    assert client.post(f"/api/plugins/visibility-os/actions/{action_id}/approve", json={"actor": "reviewer"}).status_code == 200
    feed = client.get("/api/plugins/visibility-os/feed").json()
    assert any(item["id"] == action_id for item in feed["items"])
    assert client.get("/api/plugins/visibility-os/audit-log").json()["events"]


def test_feed_marks_actions_linked_to_ci_opportunities_as_diagnosable(tmp_path, monkeypatch):
    patch_db(tmp_path, monkeypatch)
    from plugins.visibility_os.core.opportunities import upsert_opportunity
    from plugins.visibility_os.core.opportunity_actions import draft_action_from_opportunity
    from plugins.visibility_os.dashboard.plugin_api import router

    app = FastAPI()
    app.include_router(router, prefix="/api/plugins/visibility-os")
    client = TestClient(app)
    opportunity = upsert_opportunity(
        source_system="github",
        source_url="https://github.com/acme-inc/web-app/actions/runs/25928548133",
        title="Failing CI: PENDING: Email verification UI for whitelist signup",
        description="A recent GitHub Actions run failed and may be a visible unblock opportunity.",
        category="flaky_tests_and_ci_failures",
        impact_score=4,
        visibility_score=4,
        effort_score=4,
        safety_score=5,
        risk_penalty=0,
        priority_score=29,
        suggested_artifacts=["ci_before_after"],
        metadata={"databaseId": 25928548133},
    )
    action = draft_action_from_opportunity(opportunity["id"], action_kind="slack_update", target_location="#team-updates")

    feed = client.get("/api/plugins/visibility-os/feed").json()
    action_card = next(item for item in feed["items"] if item["id"] == action["id"])
    assert action_card["kind"] == "action"
    assert action_card["opportunity_id"] == opportunity["id"]
    assert action_card["opportunity_source_url"] == opportunity["source_url"]
    assert action_card["can_diagnose_ci"] is True


def test_workstream_schema_core_api_and_feed_rollups(tmp_path, monkeypatch):
    db = patch_db(tmp_path, monkeypatch)
    db.init_db()
    conn = sqlite3.connect(tmp_path / "visibility_os.db")
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"workstreams", "workstream_events", "workstream_artifacts"} <= names

    from plugins.visibility_os.core.opportunities import upsert_opportunity
    from plugins.visibility_os.core.opportunity_actions import draft_action_from_opportunity
    from plugins.visibility_os.core.workstreams import (
        add_workstream_artifact,
        get_workstream,
        list_workstreams,
        update_stage,
    )
    from plugins.visibility_os.dashboard.plugin_api import router

    opportunity = upsert_opportunity(
        source_system="github",
        source_url="https://github.com/acme-inc/web-app/issues/42",
        title="Bug: negative add support",
        description="A visible bug fix opportunity.",
        category="bug_fix",
        impact_score=4,
        visibility_score=4,
        effort_score=3,
        safety_score=5,
        risk_penalty=0,
        priority_score=30,
        suggested_artifacts=["tests"],
        metadata={"url": "https://github.com/acme-inc/web-app/issues/42"},
    )

    action = draft_action_from_opportunity(opportunity["id"], action_kind="github_issue_fix_lane", actor="human")
    workstream_id = action["proposed_payload"]["workstream_id"]
    assert workstream_id.startswith("ws_")

    ws = get_workstream(workstream_id)
    assert ws["opportunity_id"] == opportunity["id"]
    assert ws["root_action_id"] == action["id"]
    assert ws["stage"] == "queued"
    assert ws["status"] == "active"
    assert ws["events"][0]["event_type"] == "created"

    update_stage(workstream_id, stage="editing", current_step="Agent is editing files", progress_percent=35, actor="agent")
    add_workstream_artifact(workstream_id, artifact_type="proposed_pr", title="Prepared PR", payload={"branch": "fix/issue-42", "changed_files": ["app.py"]})
    ws = get_workstream(workstream_id)
    assert ws["stage"] == "editing"
    assert ws["current_step"] == "Agent is editing files"
    assert ws["progress_percent"] == 35
    assert ws["events"][-1]["stage"] == "editing"
    assert ws["artifacts"][0]["artifact_type"] == "proposed_pr"

    assert list_workstreams(status="active")[0]["id"] == workstream_id

    app = FastAPI()
    app.include_router(router, prefix="/api/plugins/visibility-os")
    client = TestClient(app)

    workstreams = client.get("/api/plugins/visibility-os/workstreams?status=active").json()["workstreams"]
    assert workstreams[0]["id"] == workstream_id
    detail = client.get(f"/api/plugins/visibility-os/workstreams/{workstream_id}").json()
    assert detail["artifacts"][0]["payload"]["branch"] == "fix/issue-42"
    opportunity_workstreams = client.get(f"/api/plugins/visibility-os/opportunities/{opportunity['id']}/workstreams").json()["workstreams"]
    assert opportunity_workstreams[0]["id"] == workstream_id

    feed = client.get("/api/plugins/visibility-os/feed").json()
    opportunity_card = next(item for item in feed["items"] if item["kind"] == "opportunity" and item["id"] == opportunity["id"])
    assert opportunity_card["workstream_id"] == workstream_id
    assert opportunity_card["workstream_stage"] == "editing"
    assert opportunity_card["agent_has_worked_on_this"] is True
    assert opportunity_card["pending_human_action"] is None


def test_github_scanner_normalizes_mocked_gh(monkeypatch, tmp_path):
    patch_db(tmp_path, monkeypatch)
    from plugins.visibility_os.core.connectors.github import GitHubConnector
    from plugins.visibility_os.core.scanner import scan_github

    def fake_run(args, capture_output, text, timeout, check):
        class R:
            returncode = 0
            stderr = ""
            stdout = ""
        r = R()
        if args[:3] == ["gh", "issue", "list"]:
            r.stdout = json.dumps([{"number": 1, "title": "Flaky auth test", "url": "https://github.com/org/repo/issues/1", "labels": [{"name": "bug"}], "updatedAt": "2026-05-25T00:00:00Z", "assignees": []}])
        elif args[:3] == ["gh", "pr", "list"]:
            if "--state" in args:
                r.stdout = "[]"
            else:
                r.stdout = json.dumps([{"number": 2, "title": "Docs update", "url": "https://github.com/org/repo/pull/2", "updatedAt": "2026-05-20T00:00:00Z", "reviewDecision": "REVIEW_REQUIRED", "statusCheckRollup": []}])
        else:
            r.stdout = json.dumps([{"databaseId": 3, "status": "completed", "conclusion": "failure", "displayTitle": "CI", "workflowName": "test", "url": "https://github.com/org/repo/actions/runs/3", "headBranch": "ci-failure", "createdAt": "2026-05-25T00:00:00Z"}])
        return r

    monkeypatch.setattr("subprocess.run", fake_run)
    opportunities = scan_github(GitHubConnector(repo="org/repo"))
    assert len(opportunities) >= 3
    assert opportunities[0]["priority_score"] >= opportunities[-1]["priority_score"]


def test_github_scanner_skips_failed_runs_from_merged_prs(monkeypatch, tmp_path):
    patch_db(tmp_path, monkeypatch)
    from plugins.visibility_os.core.connectors.github import GitHubConnector
    from plugins.visibility_os.core.scanner import scan_github

    def fake_run(args, capture_output, text, timeout, check):
        class R:
            returncode = 0
            stderr = ""
            stdout = ""
        r = R()
        if args[:3] in (["gh", "issue", "list"], ["gh", "pr", "list"]) and "--state" not in args:
            r.stdout = "[]"
        elif args[:3] == ["gh", "pr", "list"] and "--state" in args:
            r.stdout = json.dumps([{
                "number": 132,
                "title": "PENDING: Email verification UI for whitelist signup",
                "state": "MERGED",
                "url": "https://github.com/acme-inc/web-app/pull/132",
                "headRefName": "feat/email-guard",
                "headRefOid": "fixed456",
                "baseRefName": "main",
                "mergedAt": "2026-05-15T16:34:49Z",
                "statusCheckRollup": [{"name": "typecheck", "conclusion": "SUCCESS", "status": "COMPLETED"}],
            }])
        else:
            r.stdout = json.dumps([{
                "databaseId": 25928548133,
                "status": "completed",
                "conclusion": "failure",
                "displayTitle": "PENDING: Email verification UI for whitelist signup",
                "workflowName": "Typecheck",
                "url": "https://github.com/acme-inc/web-app/actions/runs/25928548133",
                "headBranch": "feat/email-guard",
                "headSha": "decbc10",
                "createdAt": "2026-05-15T16:17:23Z",
            }])
        return r

    monkeypatch.setattr("subprocess.run", fake_run)
    preexisting = scan_github(GitHubConnector(repo="acme-inc/web-app"))
    assert preexisting == []


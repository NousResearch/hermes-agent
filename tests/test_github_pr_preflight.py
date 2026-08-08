from pathlib import Path

import pytest

import scripts.github_pr_preflight as pr_preflight

_REPO_ROOT = Path(__file__).resolve().parents[1]
_module_path = Path(pr_preflight.__file__).resolve()
assert _module_path.is_relative_to(_REPO_ROOT), (
    f"{pr_preflight.__name__} resolved outside clean worktree: {_module_path}"
)


def test_parse_github_remote_url_supports_https_ssh_and_git_urls():
    assert pr_preflight.parse_github_remote_url(
        "https://github.com/NousResearch/hermes-agent.git"
    ) == "NousResearch/hermes-agent"
    assert pr_preflight.parse_github_remote_url(
        "git@github.com:NousResearch/hermes-agent.git"
    ) == "NousResearch/hermes-agent"
    assert pr_preflight.parse_github_remote_url(
        "ssh://git@github.com/NousResearch/hermes-agent.git"
    ) == "NousResearch/hermes-agent"


def test_parse_github_remote_url_rejects_flag_like_owner_or_repo():
    assert pr_preflight.parse_github_remote_url("https://github.com/--repo=evil/repo.git") is None
    assert pr_preflight.parse_github_remote_url("git@github.com:owner/--repo=evil.git") is None


@pytest.mark.parametrize(
    ("permission", "expected"),
    [
        ("ADMIN", "direct_origin"),
        ("MAINTAIN", "direct_origin"),
        ("WRITE", "direct_origin"),
        ("TRIAGE", "fork_remote"),
        ("READ", "fork_remote"),
        ("NONE", "fork_remote"),
    ],
)
def test_recommend_push_strategy_from_viewer_permission(permission, expected):
    assert pr_preflight.recommend_push_strategy(
        viewer_permission=permission,
        working_tree_clean=True,
        gh_authenticated=True,
    ) == expected


def test_missing_viewer_permission_is_unknown_not_fork_remote():
    assert pr_preflight.recommend_push_strategy(
        viewer_permission=None,
        working_tree_clean=True,
        gh_authenticated=True,
    ) == "unknown_permission"
    assert pr_preflight.recommend_push_strategy(
        viewer_permission="",
        working_tree_clean=True,
        gh_authenticated=True,
    ) == "unknown_permission"


def test_preflight_blocks_dirty_tree_before_recommending_push_strategy():
    assert pr_preflight.recommend_push_strategy(
        viewer_permission="WRITE",
        working_tree_clean=False,
        gh_authenticated=True,
    ) == "blocked_dirty_tree"


def test_preflight_blocks_missing_github_auth_before_recommending_push_strategy():
    assert pr_preflight.recommend_push_strategy(
        viewer_permission="WRITE",
        working_tree_clean=True,
        gh_authenticated=False,
    ) == "blocked_auth"


def test_classify_ci_action_required_as_maintainer_approval_not_failure():
    runs = [
        {
            "name": "Tests",
            "status": "completed",
            "conclusion": "action_required",
            "databaseId": 123,
            "url": "https://github.com/owner/repo/actions/runs/123",
        }
    ]

    classification = pr_preflight.classify_ci_state(checks=[], workflow_runs=runs)

    assert classification["state"] == "action_required"
    assert classification["blocked_by"] == "maintainer_workflow_approval"
    assert classification["is_failure"] is False
    assert classification["next_action"] == "Ask an upstream maintainer to approve fork workflow runs."


def test_classify_ci_failures_take_priority_over_action_required():
    classification = pr_preflight.classify_ci_state(
        checks=[{"name": "lint", "conclusion": "failure"}],
        workflow_runs=[{"name": "Tests", "status": "completed", "conclusion": "action_required"}],
    )

    assert classification["state"] == "failure"
    assert classification["blocked_by"] == "ci_failure"
    assert classification["is_failure"] is True


def test_classify_ci_expected_required_check_as_pending():
    classification = pr_preflight.classify_ci_state(
        checks=[{"name": "required", "state": "EXPECTED"}], workflow_runs=[]
    )

    assert classification["state"] == "pending"
    assert classification["blocked_by"] == "ci_pending"
    assert classification["is_failure"] is False


def test_parse_pr_identifier_rejects_flags_and_non_pr_urls():
    assert pr_preflight.parse_pr_identifier("20601") == "20601"
    assert (
        pr_preflight.parse_pr_identifier("https://github.com/NousResearch/hermes-agent/pull/20601")
        == "20601"
    )

    with pytest.raises(ValueError):
        pr_preflight.parse_pr_identifier("--web")
    with pytest.raises(ValueError):
        pr_preflight.parse_pr_identifier("https://github.com/NousResearch/hermes-agent/issues/20601")


def test_validate_repo_slug_rejects_flags_and_malformed_repos():
    assert pr_preflight.validate_repo_slug("NousResearch/hermes-agent") == "NousResearch/hermes-agent"

    with pytest.raises(ValueError):
        pr_preflight.validate_repo_slug("--repo")
    with pytest.raises(ValueError):
        pr_preflight.validate_repo_slug("NousResearch")


class FakeRunner:
    def __init__(self, responses):
        self.responses = {tuple(key): value for key, value in responses.items()}
        self.calls = []

    def __call__(self, args, cwd):
        self.calls.append((tuple(args), cwd))
        return self.responses.get(
            tuple(args),
            pr_preflight.subprocess.CompletedProcess(args, 1, "", "unexpected command"),
        )


def completed(args, stdout="", stderr="", returncode=0):
    return pr_preflight.subprocess.CompletedProcess(args, returncode, stdout, stderr)


def test_collect_preflight_recommends_fork_and_detects_existing_pr(tmp_path):
    runner = FakeRunner(
        {
            ("git", "branch", "--show-current"): completed((), "feat/example\n"),
            ("git", "status", "--porcelain"): completed((), ""),
            ("git", "remote", "get-url", "origin"): completed(
                (), "https://github.com/NousResearch/hermes-agent.git\n"
            ),
            ("gh", "auth", "status"): completed((), "ok"),
            (
                "gh",
                "repo",
                "view",
                "NousResearch/hermes-agent",
                "--json",
                "nameWithOwner,viewerPermission,defaultBranchRef",
            ): completed(
                (),
                '{"nameWithOwner":"NousResearch/hermes-agent",'
                '"viewerPermission":"READ",'
                '"defaultBranchRef":{"name":"main"}}',
            ),
            (
                "gh",
                "pr",
                "list",
                "--repo",
                "NousResearch/hermes-agent",
                "--head",
                "feat/example",
                "--json",
                "number,url,state,title,isDraft",
            ): completed(
                (),
                '[{"number":42,"url":"https://github.com/NousResearch/hermes-agent/pull/42",'
                '"state":"OPEN","title":"Example","isDraft":true}]',
            ),
        }
    )

    preflight = pr_preflight.collect_preflight(cwd=tmp_path, runner=runner)

    assert preflight["repo"] == "NousResearch/hermes-agent"
    assert preflight["branch"] == "feat/example"
    assert preflight["viewer_permission"] == "READ"
    assert preflight["strategy"] == "fork_remote"
    assert preflight["existing_pr"]["number"] == 42


def test_collect_pr_status_classifies_action_required_runs(tmp_path):
    runner = FakeRunner(
        {
            (
                "gh",
                "pr",
                "view",
                "20601",
                "--repo",
                "NousResearch/hermes-agent",
                "--json",
                "number,url,title,state,isDraft,headRefName,headRefOid,baseRefName,mergeable,statusCheckRollup",
            ): completed(
                (),
                '{"number":20601,"url":"https://github.com/NousResearch/hermes-agent/pull/20601",'
                '"state":"OPEN","isDraft":true,"headRefName":"feat/example",'
                '"headRefOid":"abc123","baseRefName":"main","mergeable":"MERGEABLE",'
                '"statusCheckRollup":[]}',
            ),
            (
                "gh",
                "run",
                "list",
                "--repo",
                "NousResearch/hermes-agent",
                "--commit",
                "abc123",
                "--limit",
                "20",
                "--json",
                "databaseId,name,status,conclusion,event,headBranch,headSha,displayTitle,url,createdAt,updatedAt",
            ): completed(
                (),
                '[{"databaseId":123,"name":"Tests","status":"completed",'
                '"conclusion":"action_required","url":"https://github.com/run/123"}]',
            ),
        }
    )

    status = pr_preflight.collect_pr_status(
        repo="NousResearch/hermes-agent", pr_number="20601", cwd=tmp_path, runner=runner
    )

    assert status["pr"]["number"] == 20601
    assert status["ci"]["state"] == "action_required"
    assert status["ci"]["blocked_by"] == "maintainer_workflow_approval"


def test_build_receipt_includes_pr_state_ci_state_and_next_action():
    receipt = pr_preflight.build_pr_receipt(
        preflight={
            "repo": "NousResearch/hermes-agent",
            "branch": "feat/example",
            "viewer_permission": "READ",
            "strategy": "fork_remote",
            "working_tree_clean": True,
        },
        pr={
            "number": 20601,
            "url": "https://github.com/NousResearch/hermes-agent/pull/20601",
            "state": "OPEN",
            "isDraft": True,
            "mergeable": "MERGEABLE",
            "baseRefName": "main",
            "headRefName": "feat/example",
        },
        ci={
            "state": "action_required",
            "blocked_by": "maintainer_workflow_approval",
            "is_failure": False,
            "next_action": "Ask an upstream maintainer to approve fork workflow runs.",
        },
        local_validation={"tests": "145 passed", "ruff": "passed"},
    )

    assert receipt["pr_url"] == "https://github.com/NousResearch/hermes-agent/pull/20601"
    assert receipt["draft"] is True
    assert receipt["ci_state"] == "action_required"
    assert receipt["blocked_by"] == "maintainer_workflow_approval"
    assert receipt["next_action"] == "Ask an upstream maintainer to approve fork workflow runs."
    assert receipt["local_validation"]["tests"] == "145 passed"

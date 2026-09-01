from __future__ import annotations

import json
import subprocess
from contextlib import AbstractContextManager

import pytest

from github_pr_feedback.github_client import (
    MAX_DISCOVERED_PULL_REQUESTS,
    MAX_FEEDBACK_BODY_CHARS,
    CheckState,
    GitHubClient,
    GitHubClientError,
    GitHubRequestGate,
    PullRequestMergeState,
    RepositoryMergePolicy,
    ReviewState,
    SubprocessCommandRunner,
)


class RecordingRunner:
    def __init__(self, responses: dict[tuple[str, ...], object]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, ...]] = []

    def run(self, argv: list[str]) -> str:
        key = tuple(argv)
        self.calls.append(key)
        return json.dumps(self.responses[key])


class RecordingGate(AbstractContextManager):
    def __init__(self) -> None:
        self.entries = 0
        self.deferrals: list[float] = []

    def __enter__(self):
        self.entries += 1
        return self

    def defer(self, seconds: float) -> None:
        self.deferrals.append(seconds)

    def __exit__(self, *_args) -> None:
        return None


def test_subprocess_runner_retries_one_bounded_rate_limit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    sleeps: list[float] = []
    results = iter(
        (
            subprocess.CompletedProcess(
                ["gh", "api", "rate_limit"], 1, "", "HTTP 403: rate limit exceeded"
            ),
            subprocess.CompletedProcess(["gh", "api", "rate_limit"], 0, "{}", ""),
        )
    )

    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        return next(results)

    monkeypatch.setattr("github_pr_feedback.github_client.subprocess.run", fake_run)
    gate = RecordingGate()
    runner = SubprocessCommandRunner(
        sleeper=sleeps.append, rate_limit_backoff=0.25, request_gate=gate
    )

    assert runner.run(["gh", "api", "rate_limit"]) == "{}"
    assert calls == [["gh", "api", "rate_limit"], ["gh", "api", "rate_limit"]]
    assert sleeps == []
    assert gate.entries == 2
    assert gate.deferrals == [1.0]


def test_request_gate_shares_secondary_limit_cooldown_across_instances(tmp_path) -> None:
    now = [100.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    path = tmp_path / "github-request-gate.json"
    with GitHubRequestGate(path, sleeper=sleep, clock=lambda: now[0]) as gate:
        gate.defer(30)
    with GitHubRequestGate(path, sleeper=sleep, clock=lambda: now[0]):
        pass

    assert sleeps == [30.0]


def test_request_gate_spaces_shared_requests_at_a_conservative_rate(tmp_path) -> None:
    now = [100.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    path = tmp_path / "github-request-gate.json"
    with GitHubRequestGate(path, sleeper=sleep, clock=lambda: now[0]):
        pass
    with GitHubRequestGate(path, sleeper=sleep, clock=lambda: now[0]):
        pass

    assert sleeps == [1.0]


def test_request_gate_recovers_conservatively_from_nonfinite_state(tmp_path) -> None:
    path = tmp_path / "github-request-gate.json"
    path.write_text('{"cooldown_until": NaN}\n', encoding="utf-8")
    sleeps: list[float] = []
    now = [100.0]

    with GitHubRequestGate(
        path, sleeper=lambda seconds: sleeps.append(seconds), clock=lambda: now[0]
    ):
        pass

    assert sleeps == [5.0]
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["cooldown_until"] == 105.0
    assert all(value == value and abs(value) != float("inf") for value in stored.values())


@pytest.mark.parametrize(
    "stderr",
    (
        "HTTP 429: Too Many Requests",
        "HTTP 403: secondary rate limit",
        "x-ratelimit-remaining: 0",
    ),
)
def test_subprocess_runner_retries_all_github_rate_limit_shapes(
    monkeypatch: pytest.MonkeyPatch, stderr: str
) -> None:
    calls: list[list[str]] = []
    results = iter(
        (
            subprocess.CompletedProcess(["gh", "api", "rate_limit"], 1, "", stderr),
            subprocess.CompletedProcess(["gh", "api", "rate_limit"], 0, "{}", ""),
        )
    )

    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        return next(results)

    monkeypatch.setattr("github_pr_feedback.github_client.subprocess.run", fake_run)
    gate = RecordingGate()
    runner = SubprocessCommandRunner(
        sleeper=lambda _delay: None, request_gate=gate
    )

    assert runner.run(["gh", "api", "rate_limit"]) == "{}"
    assert len(calls) == 2
    assert gate.deferrals == [60.0]


def test_subprocess_runner_retries_timeout_without_rate_limit_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    results = iter(
        (
            subprocess.TimeoutExpired(["gh", "api", "pull"], 60),
            subprocess.CompletedProcess(["gh", "api", "pull"], 0, "{}", ""),
        )
    )

    def fake_run(_argv, **_kwargs):
        result = next(results)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr("github_pr_feedback.github_client.subprocess.run", fake_run)
    runner = SubprocessCommandRunner(
        sleeper=sleeps.append,
        request_gate=RecordingGate(),
    )

    assert runner.run(["gh", "api", "pull"]) == "{}"
    assert sleeps == [1.0]


@pytest.mark.parametrize(
    ("stderr", "code"),
    [
        ("HTTP 403: Resource not accessible by integration", "permission_denied"),
        ("HTTP 401: Bad credentials", "authentication"),
        ("HTTP 429: Too Many Requests", "rate_limited"),
    ],
)
def test_subprocess_runner_exposes_safe_failure_code_without_output(
    monkeypatch: pytest.MonkeyPatch, stderr: str, code: str
) -> None:
    monkeypatch.setattr(
        "github_pr_feedback.github_client.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["gh", "api", "labels"], 1, "", stderr
        ),
    )
    with pytest.raises(GitHubClientError) as raised:
        SubprocessCommandRunner(sleeper=lambda _delay: None).run(["gh", "api", "labels"])
    assert raised.value.code == code
    assert stderr not in str(raised.value)


def test_subprocess_runner_does_not_retry_an_ordinary_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 1, "", "not found")

    monkeypatch.setattr("github_pr_feedback.github_client.subprocess.run", fake_run)

    with pytest.raises(GitHubClientError, match="GitHub command failed"):
        SubprocessCommandRunner(
            sleeper=lambda _delay: None, request_gate=RecordingGate()
        ).run(["gh", "api", "missing"])

    assert calls == [["gh", "api", "missing"]]


def test_github_client_reads_paginated_canonical_feedback_with_fixed_gh_argv() -> None:
    pulls_argv = (
        "gh",
        "pr",
        "list",
        "--repo",
        "acme/widgets",
        "--state",
        "open",
        "--author",
        "owner",
        "--limit",
        str(MAX_DISCOVERED_PULL_REQUESTS),
        "--json",
        "number,state,headRepository,author,headRefName,headRefOid,baseRefName,baseRefOid,updatedAt,labels",
    )
    comments_argv = (
        "gh",
        "api",
        "--paginate",
        "--slurp",
        "repos/acme/widgets/issues/17/comments?per_page=100",
    )
    review_comments_argv = (
        "gh",
        "api",
        "--paginate",
        "--slurp",
        "repos/acme/widgets/pulls/17/comments?per_page=100",
    )
    reviews_argv = (
        "gh",
        "api",
        "--paginate",
        "--slurp",
        "repos/acme/widgets/pulls/17/reviews?per_page=100",
    )
    runner = RecordingRunner(
        {
            pulls_argv: [
                canonical_list_pull(labels=("codex", "type/perf")),
                canonical_list_pull(number=18),
            ],
            comments_argv: [
                [canonical_feedback("issue-1", "first")],
                [canonical_feedback("issue-2", "second")],
            ],
            review_comments_argv: [
                [canonical_feedback("review-comment-1", "line note")]
            ],
            reviews_argv: [
                [
                    canonical_feedback(
                        "review-1", "submitted", submitted_at="2026-08-24T00:00:00Z"
                    )
                ],
                [canonical_feedback("review-pending", "pending", submitted_at=None)],
            ],
        }
    )
    client = GitHubClient(runner)

    pull_requests = client.list_open_pull_requests("acme/widgets", "owner")
    feedback = client.list_feedback("acme/widgets", 17)

    assert [pull_request.number for pull_request in pull_requests] == [17, 18]
    assert pull_requests[0].labels == ("codex", "type/perf")
    assert pull_requests[0].base_branch == "stable"
    assert pull_requests[0].base_sha == "b" * 40
    assert [(item.kind, item.feedback_id, item.body) for item in feedback] == [
        ("issue_comment", "issue-1", "first"),
        ("issue_comment", "issue-2", "second"),
        ("review_comment", "review-comment-1", "line note"),
        ("review", "review-1", "submitted"),
    ]
    assert runner.calls[0] == pulls_argv
    assert set(runner.calls[1:]) == {
        comments_argv,
        review_comments_argv,
        reviews_argv,
    }


def test_github_client_reads_independent_feedback_endpoints_without_nested_fanout() -> None:
    runner = RecordingRunner(feedback_responses("ordinary"))

    feedback = GitHubClient(runner).list_feedback("acme/widgets", 17)

    assert [item.feedback_id for item in feedback] == ["issue-1"]
    assert len(runner.calls) == 3


def test_github_client_posts_bounded_issue_comment_with_fixed_argv() -> None:
    argv = (
        "gh",
        "api",
        "repos/acme/widgets/issues/17/comments",
        "--method",
        "POST",
        "--field",
        "body=exact-head receipt passed",
    )
    runner = RecordingRunner({argv: {"id": 1}})

    GitHubClient(runner).post_issue_comment(
        "acme/widgets", 17, "exact-head receipt passed"
    )

    assert runner.calls == [argv]


def test_github_client_resolves_only_the_thread_for_one_exact_review_comment() -> None:
    query_argv = (
        "gh",
        "api",
        "graphql",
        "-f",
        "query=" + GitHubClient.REVIEW_THREAD_QUERY,
        "-F",
        "owner=acme",
        "-F",
        "name=widgets",
        "-F",
        "number=17",
    )
    mutation_argv = (
        "gh",
        "api",
        "graphql",
        "-f",
        "query=" + GitHubClient.RESOLVE_REVIEW_THREAD_MUTATION,
        "-F",
        "threadId=PRRT_exact",
    )
    runner = RecordingRunner(
        {
            query_argv: {
                "data": {
                    "repository": {
                        "pullRequest": {
                            "headRefOid": "a" * 40,
                            "reviewThreads": {
                                "nodes": [
                                    {
                                        "id": "PRRT_other",
                                        "isResolved": False,
                                        "comments": {
                                            "nodes": [{"databaseId": 41}],
                                            "pageInfo": {"hasNextPage": False},
                                        },
                                    },
                                    {
                                        "id": "PRRT_exact",
                                        "isResolved": False,
                                        "comments": {
                                            "nodes": [{"databaseId": 42}],
                                            "pageInfo": {"hasNextPage": False},
                                        },
                                    },
                                ],
                                "pageInfo": {"hasNextPage": False},
                            }
                        }
                    }
                }
            },
            mutation_argv: {
                "data": {
                    "resolveReviewThread": {
                        "thread": {"id": "PRRT_exact", "isResolved": True}
                    }
                }
            },
        }
    )

    assert GitHubClient(runner).resolve_review_thread_for_comment(
        "acme/widgets", 17, "42", expected_head_sha="a" * 40
    )
    assert runner.calls == [query_argv, mutation_argv]


def test_github_client_does_not_mutate_an_already_resolved_exact_thread() -> None:
    query_argv = (
        "gh",
        "api",
        "graphql",
        "-f",
        "query=" + GitHubClient.REVIEW_THREAD_QUERY,
        "-F",
        "owner=acme",
        "-F",
        "name=widgets",
        "-F",
        "number=17",
    )
    runner = RecordingRunner(
        {
            query_argv: {
                "data": {
                    "repository": {
                        "pullRequest": {
                            "headRefOid": "a" * 40,
                            "reviewThreads": {
                                "nodes": [
                                    {
                                        "id": "PRRT_exact",
                                        "isResolved": True,
                                        "comments": {
                                            "nodes": [{"databaseId": 42}],
                                            "pageInfo": {"hasNextPage": False},
                                        },
                                    }
                                ],
                                "pageInfo": {"hasNextPage": False},
                            }
                        }
                    }
                }
            }
        }
    )

    assert not GitHubClient(runner).resolve_review_thread_for_comment(
        "acme/widgets", 17, "42", expected_head_sha="a" * 40
    )
    assert runner.calls == [query_argv]


def test_github_client_fails_closed_on_incomplete_review_thread_coverage() -> None:
    query_argv = (
        "gh",
        "api",
        "graphql",
        "-f",
        "query=" + GitHubClient.REVIEW_THREAD_QUERY,
        "-F",
        "owner=acme",
        "-F",
        "name=widgets",
        "-F",
        "number=17",
    )
    payload = {
        "data": {
            "repository": {
                "pullRequest": {
                    "headRefOid": "a" * 40,
                    "reviewThreads": {
                        "nodes": [],
                        "pageInfo": {"hasNextPage": True},
                    }
                }
            }
        }
    }

    with pytest.raises(GitHubClientError, match="review thread"):
        GitHubClient(RecordingRunner({query_argv: payload})).resolve_review_thread_for_comment(
            "acme/widgets", 17, "42", expected_head_sha="a" * 40
        )


def test_github_client_fails_closed_when_filtered_pr_list_lacks_canonical_fields() -> (
    None
):
    argv = (
        "gh",
        "pr",
        "list",
        "--repo",
        "acme/widgets",
        "--state",
        "open",
        "--author",
        "owner",
        "--limit",
        str(MAX_DISCOVERED_PULL_REQUESTS),
        "--json",
        "number,state,headRepository,author,headRefName,headRefOid,baseRefName,baseRefOid,updatedAt,labels",
    )
    runner = RecordingRunner({argv: [{"number": 17}]})

    with pytest.raises(GitHubClientError, match="missing required fields"):
        GitHubClient(runner).list_open_pull_requests("acme/widgets", "owner")


@pytest.mark.parametrize(
    "labels",
    [None, ["codex"], [{"name": 7}]],
)
def test_github_client_fails_closed_on_malformed_list_labels(labels: object) -> None:
    argv = (
        "gh",
        "pr",
        "list",
        "--repo",
        "acme/widgets",
        "--state",
        "open",
        "--author",
        "owner",
        "--limit",
        str(MAX_DISCOVERED_PULL_REQUESTS),
        "--json",
        "number,state,headRepository,author,headRefName,headRefOid,baseRefName,baseRefOid,updatedAt,labels",
    )
    row = canonical_list_pull()
    row["labels"] = labels

    with pytest.raises(GitHubClientError, match="missing required fields"):
        GitHubClient(RecordingRunner({argv: [row]})).list_open_pull_requests(
            "acme/widgets", "owner"
        )


def test_github_client_fails_closed_if_owned_pr_query_hits_coverage_cap() -> None:
    argv = (
        "gh",
        "pr",
        "list",
        "--repo",
        "acme/widgets",
        "--state",
        "open",
        "--author",
        "owner",
        "--limit",
        str(MAX_DISCOVERED_PULL_REQUESTS),
        "--json",
        "number,state,headRepository,author,headRefName,headRefOid,baseRefName,baseRefOid,updatedAt,labels",
    )
    runner = RecordingRunner(
        {
            argv: [
                canonical_list_pull(number=number)
                for number in range(1, MAX_DISCOVERED_PULL_REQUESTS + 1)
            ]
        }
    )

    with pytest.raises(GitHubClientError, match="coverage cap"):
        GitHubClient(runner).list_open_pull_requests("acme/widgets", "owner")


def test_github_client_reads_all_open_prs_and_exact_base_head_for_maintenance() -> None:
    pulls_argv = (
        "gh",
        "pr",
        "list",
        "--repo",
        "acme/widgets",
        "--state",
        "open",
        "--limit",
        str(MAX_DISCOVERED_PULL_REQUESTS),
        "--json",
        "number,state,headRepository,author,headRefName,headRefOid,baseRefName,baseRefOid,updatedAt,labels",
    )
    branch_argv = ("gh", "api", "repos/acme/widgets/branches/stable")
    runner = RecordingRunner(
        {
            pulls_argv: [canonical_list_pull()],
            branch_argv: {"name": "stable", "commit": {"sha": "b" * 40}},
        }
    )

    client = GitHubClient(runner)

    assert [
        pull.number for pull in client.list_all_open_pull_requests("acme/widgets")
    ] == [17]
    assert client.get_branch_head("acme/widgets", "stable") == "b" * 40
    assert runner.calls == [pulls_argv, branch_argv]


def test_github_client_adds_labels_with_fixed_issue_endpoint() -> None:
    argv = (
        "gh",
        "api",
        "repos/acme/widgets/issues/17/labels",
        "--method",
        "POST",
        "--field",
        "labels[]=codex",
        "--field",
        "labels[]=area/hermes",
    )
    runner = RecordingRunner({argv: {"labels": [{"name": "codex"}]}})

    GitHubClient(runner).add_issue_labels(
        "acme/widgets", 17, ("codex", "area/hermes")
    )

    assert runner.calls == [argv]


def test_github_client_creates_missing_label_on_collection_endpoint() -> None:
    read_argv = (
        "gh",
        "api",
        "repos/acme/widgets/labels/codex",
    )
    create_argv = (
        "gh",
        "api",
        "repos/acme/widgets/labels",
        "--method",
        "POST",
        "--field",
        "name=codex",
        "--field",
        "color=1f6feb",
        "--field",
        "description=PR authored by Codex",
    )

    class MissingLabelRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def run(self, argv: list[str]) -> str:
            key = tuple(argv)
            self.calls.append(key)
            if key == read_argv:
                raise GitHubClientError("label does not exist", code="not_found")
            assert key == create_argv
            return "{}"

    runner = MissingLabelRunner()
    GitHubClient(runner).ensure_issue_label(
        "acme/widgets", "codex", color="1F6FEB", description="PR authored by Codex"
    )

    assert runner.calls == [read_argv, create_argv]


def test_github_client_updates_existing_label_after_exact_read() -> None:
    read_argv = (
        "gh",
        "api",
        "repos/acme/widgets/labels/codex",
    )
    update_argv = (
        "gh",
        "api",
        "repos/acme/widgets/labels/codex",
        "--method",
        "PUT",
        "--field",
        "new_name=codex",
        "--field",
        "color=1f6feb",
        "--field",
        "description=PR authored by Codex",
    )
    runner = RecordingRunner({read_argv: {"name": "codex"}, update_argv: {}})

    GitHubClient(runner).ensure_issue_label(
        "acme/widgets", "codex", color="1f6feb", description="PR authored by Codex"
    )

    assert runner.calls == [read_argv, update_argv]


def test_github_client_bounds_untrusted_feedback_body_at_intake() -> None:
    responses = feedback_responses("x" * (MAX_FEEDBACK_BODY_CHARS + 1_000))
    client = GitHubClient(RecordingRunner(responses))

    feedback = client.list_feedback("acme/widgets", 17)

    assert len(feedback[0].body) == MAX_FEEDBACK_BODY_CHARS


def test_github_client_orders_feedback_chronologically_across_api_kinds() -> None:
    responses = feedback_responses("later resolution")
    issue_key = (
        "gh",
        "api",
        "--paginate",
        "--slurp",
        "repos/acme/widgets/issues/17/comments?per_page=100",
    )
    review_key = (
        "gh",
        "api",
        "--paginate",
        "--slurp",
        "repos/acme/widgets/pulls/17/comments?per_page=100",
    )
    responses[issue_key] = [
        [canonical_feedback("resolution", "fixed", created_at="2026-08-24T00:05:00Z")]
    ]
    responses[review_key] = [
        [canonical_feedback("finding", "fix this", created_at="2026-08-24T00:01:00Z")]
    ]

    feedback = GitHubClient(RecordingRunner(responses)).list_feedback(
        "acme/widgets", 17
    )

    assert [item.feedback_id for item in feedback[:2]] == ["finding", "resolution"]


def test_github_client_accepts_a_submitted_review_with_no_text_body() -> None:
    responses = feedback_responses("ordinary")
    reviews_key = (
        "gh",
        "api",
        "--paginate",
        "--slurp",
        "repos/acme/widgets/pulls/17/reviews?per_page=100",
    )
    responses[reviews_key] = [
        [canonical_feedback("review-empty", None, submitted_at="2026-08-24T00:00:00Z")]
    ]

    feedback = GitHubClient(RecordingRunner(responses)).list_feedback(
        "acme/widgets", 17
    )

    review = next(item for item in feedback if item.feedback_id == "review-empty")
    assert review.body == ""


def test_github_client_gets_the_current_pull_request_with_fixed_argv() -> None:
    argv = ("gh", "api", "repos/acme/widgets/pulls/17")
    runner = RecordingRunner({argv: canonical_pull()})

    pull_request = GitHubClient(runner).get_pull_request("acme/widgets", 17)

    assert pull_request.number == 17
    assert pull_request.base_branch == "stable"
    assert pull_request.base_sha == "b" * 40
    assert runner.calls == [argv]


def test_github_client_rejects_pull_request_identity_mismatch() -> None:
    argv = ("gh", "api", "repos/acme/widgets/pulls/17")
    payload = canonical_pull()
    payload["number"] = 99
    with pytest.raises(GitHubClientError, match="missing required fields"):
        GitHubClient(RecordingRunner({argv: payload})).get_pull_request("acme/widgets", 17)

    payload = canonical_pull()
    payload["base"]["sha"] = "short"
    with pytest.raises(GitHubClientError, match="missing required fields"):
        GitHubClient(RecordingRunner({argv: payload})).get_pull_request("acme/widgets", 17)


def test_github_client_reads_repository_actions_enabled_with_fixed_argv() -> None:
    argv = ("gh", "api", "repos/acme/widgets/actions/permissions")
    runner = RecordingRunner({argv: {"enabled": False, "sha_pinning_required": False}})

    client = GitHubClient(runner)
    enabled = client.actions_enabled("acme/widgets")
    cached = client.actions_enabled("acme/widgets")

    assert enabled is False
    assert cached is False
    assert runner.calls == [argv]


def test_github_client_refreshes_actions_enabled_after_cache_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = ("gh", "api", "repos/acme/widgets/actions/permissions")
    responses = iter(({"enabled": False}, {"enabled": True}))

    class ChangingRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def run(self, command: list[str]) -> str:
            self.calls.append(tuple(command))
            return json.dumps(next(responses))

    now = [100.0]
    monkeypatch.setattr("github_pr_feedback.github_client.time.monotonic", lambda: now[0])
    runner = ChangingRunner()
    client = GitHubClient(runner)

    assert client.actions_enabled("acme/widgets") is False
    assert client.actions_enabled("acme/widgets") is False
    now[0] += 61.0
    assert client.actions_enabled("acme/widgets") is True
    assert runner.calls == [argv, argv]


def test_github_client_reads_private_repository_and_canonical_merge_state() -> None:
    repository_argv = ("gh", "api", "repos/acme/widgets")
    pull_argv = ("gh", "api", "repos/acme/widgets/pulls/17")
    pull = canonical_pull()
    pull.update(
        {
            "draft": False,
            "mergeable": True,
            "mergeable_state": "clean",
            "merged": False,
            "merge_commit_sha": None,
            "title": "$(touch /tmp/untrusted-title)",
            "body": "; rm -rf untrusted-body",
            "labels": [
                {"name": "sweeper:risk-session-state"},
                {"name": "ci-reviewed"},
            ],
        }
    )
    repository = {
        "private": True,
        "allow_squash_merge": True,
        "allow_rebase_merge": False,
        "allow_merge_commit": True,
    }
    runner = RecordingRunner({repository_argv: repository, pull_argv: pull})

    client = GitHubClient(runner)

    assert client.repository_is_private("acme/widgets") is True
    assert client.get_repository_merge_policy("acme/widgets") == RepositoryMergePolicy(
        squash=True,
        rebase=False,
        merge=True,
    )
    assert client.get_merge_state("acme/widgets", 17) == PullRequestMergeState(
        repository="acme/widgets",
        number=17,
        state="OPEN",
        is_draft=False,
        mergeable=True,
        merge_state_status="CLEAN",
        base_branch="stable",
        base_sha="b" * 40,
        head_repository="acme/widgets",
        author_login="owner",
        head_ref_name="codex/fix",
        head_sha="a" * 40,
        merged=False,
        merge_commit_oid=None,
        labels=("sweeper:risk-session-state", "ci-reviewed"),
    )
    assert runner.calls == [repository_argv, repository_argv, pull_argv]


def test_github_client_preserves_canonical_pr_labels_for_worker_routing() -> None:
    argv = ("gh", "api", "repos/acme/widgets/pulls/17")
    row = canonical_pull()
    row["labels"] = [{"name": "type/perf"}, {"name": "sweeper:risk-session-state"}]

    pull = GitHubClient(RecordingRunner({argv: row})).get_pull_request("acme/widgets", 17)

    assert pull.labels == ("type/perf", "sweeper:risk-session-state")


def test_github_client_reads_review_decision_and_unresolved_threads_with_fixed_graphql_argv() -> None:
    query_argv = (
        "gh",
        "api",
        "graphql",
        "-f",
        "query=" + GitHubClient.REVIEW_STATE_QUERY,
        "-F",
        "owner=acme",
        "-F",
        "name=widgets",
        "-F",
        "number=17",
    )
    runner = RecordingRunner(
        {
            query_argv: {
                "data": {
                    "repository": {
                        "pullRequest": {
                            "reviewDecision": "APPROVED",
                            "reviewThreads": {
                                "nodes": [{"isResolved": True}, {"isResolved": False}],
                                "pageInfo": {"hasNextPage": False},
                            },
                        }
                    }
                }
            }
        }
    )

    state = GitHubClient(runner).get_review_state("acme/widgets", 17)

    assert state == ReviewState(review_decision="APPROVED", unresolved_thread_count=1)
    assert runner.calls == [query_argv]


def test_github_client_fails_closed_on_truncated_or_malformed_review_threads() -> None:
    query_argv = (
        "gh",
        "api",
        "graphql",
        "-f",
        "query=" + GitHubClient.REVIEW_STATE_QUERY,
        "-F",
        "owner=acme",
        "-F",
        "name=widgets",
        "-F",
        "number=17",
    )
    payload = {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviewDecision": "APPROVED",
                    "reviewThreads": {
                        "nodes": [{"isResolved": False}],
                        "pageInfo": {"hasNextPage": True},
                    },
                }
            }
        }
    }

    with pytest.raises(GitHubClientError, match="review state"):
        GitHubClient(RecordingRunner({query_argv: payload})).get_review_state(
            "acme/widgets", 17
        )


def test_github_client_reads_green_checks_only_when_actions_are_enabled() -> None:
    permissions_argv = ("gh", "api", "repos/acme/widgets/actions/permissions")
    checks_argv = (
        "gh",
        "api",
        "repos/acme/widgets/commits/" + "a" * 40 + "/check-runs?per_page=100",
    )
    statuses_argv = (
        "gh",
        "api",
        "repos/acme/widgets/commits/" + "a" * 40 + "/status?per_page=100",
    )
    runner = RecordingRunner(
        {
            permissions_argv: {"enabled": True},
            checks_argv: {
                "total_count": 1,
                "check_runs": [{"status": "completed", "conclusion": "success"}],
            },
            statuses_argv: {"state": "success", "statuses": []},
        }
    )

    state = GitHubClient(runner).get_check_state("acme/widgets", "a" * 40)

    assert state == CheckState(actions_enabled=True, all_green=True, check_count=1)
    assert runner.calls == [permissions_argv, checks_argv, statuses_argv]


def test_github_client_treats_disabled_actions_as_a_distinct_known_state() -> None:
    permissions_argv = ("gh", "api", "repos/acme/widgets/actions/permissions")
    runner = RecordingRunner({permissions_argv: {"enabled": False}})

    state = GitHubClient(runner).get_check_state("acme/widgets", "a" * 40)

    assert state == CheckState(actions_enabled=False, all_green=True, check_count=0)
    assert runner.calls == [permissions_argv]


def test_github_client_detects_a_billing_lockout_from_check_run_annotations() -> None:
    permissions_argv = ("gh", "api", "repos/acme/widgets/actions/permissions")
    checks_argv = (
        "gh",
        "api",
        "repos/acme/widgets/commits/" + "a" * 40 + "/check-runs?per_page=100",
    )
    statuses_argv = (
        "gh",
        "api",
        "repos/acme/widgets/commits/" + "a" * 40 + "/status?per_page=100",
    )
    annotations_argv = (
        "gh",
        "api",
        "--paginate",
        "--slurp",
        "repos/acme/widgets/check-runs/98764105373/annotations",
    )
    runner = RecordingRunner(
        {
            permissions_argv: {"enabled": True},
            checks_argv: {
                "total_count": 1,
                "check_runs": [
                    {
                        "id": 98764105373,
                        "status": "completed",
                        "conclusion": "failure",
                        "output": {"annotations_count": 1},
                    }
                ],
            },
            statuses_argv: {"state": "failure", "statuses": []},
            annotations_argv: [
                [
                    {
                        "message": (
                            "The job was not started because recent account "
                            "payments have failed or your spending limit needs "
                            "to be increased. Please check the 'Billing & "
                            "plans' section in your settings"
                        )
                    }
                ]
            ],
        }
    )

    state = GitHubClient(runner).get_check_state("acme/widgets", "a" * 40)

    assert state == CheckState(
        actions_enabled=True, all_green=False, check_count=1, billing_blocked=True
    )


def test_github_client_does_not_flag_a_genuine_test_failure_as_billing_blocked() -> None:
    permissions_argv = ("gh", "api", "repos/acme/widgets/actions/permissions")
    checks_argv = (
        "gh",
        "api",
        "repos/acme/widgets/commits/" + "a" * 40 + "/check-runs?per_page=100",
    )
    statuses_argv = (
        "gh",
        "api",
        "repos/acme/widgets/commits/" + "a" * 40 + "/status?per_page=100",
    )
    annotations_argv = (
        "gh",
        "api",
        "--paginate",
        "--slurp",
        "repos/acme/widgets/check-runs/1/annotations",
    )
    runner = RecordingRunner(
        {
            permissions_argv: {"enabled": True},
            checks_argv: {
                "total_count": 1,
                "check_runs": [
                    {
                        "id": 1,
                        "status": "completed",
                        "conclusion": "failure",
                        "output": {"annotations_count": 1},
                    }
                ],
            },
            statuses_argv: {"state": "failure", "statuses": []},
            annotations_argv: [
                [{"message": "AssertionError: expected 200, got 500"}]
            ],
        }
    )

    state = GitHubClient(runner).get_check_state("acme/widgets", "a" * 40)

    assert state == CheckState(
        actions_enabled=True, all_green=False, check_count=1, billing_blocked=False
    )


def test_github_client_flags_a_check_run_waiting_on_human_approval_as_action_required() -> None:
    permissions_argv = ("gh", "api", "repos/acme/widgets/actions/permissions")
    checks_argv = (
        "gh",
        "api",
        "repos/acme/widgets/commits/" + "a" * 40 + "/check-runs?per_page=100",
    )
    statuses_argv = (
        "gh",
        "api",
        "repos/acme/widgets/commits/" + "a" * 40 + "/status?per_page=100",
    )
    runner = RecordingRunner(
        {
            permissions_argv: {"enabled": True},
            checks_argv: {
                "total_count": 1,
                "check_runs": [
                    {"id": 1, "status": "completed", "conclusion": "action_required"}
                ],
            },
            statuses_argv: {"state": "failure", "statuses": []},
        }
    )

    state = GitHubClient(runner).get_check_state("acme/widgets", "a" * 40)

    assert state == CheckState(
        actions_enabled=True,
        all_green=False,
        check_count=1,
        billing_blocked=False,
        action_required=True,
    )


@pytest.mark.parametrize(
    "method,flag",
    [("squash", "--squash"), ("rebase", "--rebase"), ("merge", "--merge")],
)
def test_github_client_uses_only_fixed_exact_head_merge_argv(
    method: str, flag: str
) -> None:
    merge_argv = (
        "gh",
        "pr",
        "merge",
        "17",
        "--repo",
        "acme/widgets",
        flag,
        "--match-head-commit",
        "a" * 40,
    )
    runner = RecordingRunner({merge_argv: "remote output is not merge truth"})

    result = GitHubClient(runner).merge_pull_request(
        "acme/widgets", 17, "a" * 40, method=method
    )

    assert result is None
    assert runner.calls == [merge_argv]


@pytest.mark.parametrize("head_sha", ["short", "g" * 40, "a" * 39, "a" * 41])
def test_github_client_rejects_noncanonical_merge_head_sha(head_sha: str) -> None:
    runner = RecordingRunner({})

    with pytest.raises(ValueError, match="head_sha"):
        GitHubClient(runner).merge_pull_request(
            "acme/widgets", 17, head_sha, method="squash"
        )

    assert runner.calls == []


def test_github_client_rejects_unknown_merge_method_without_a_command() -> None:
    runner = RecordingRunner({})

    with pytest.raises(ValueError, match="method"):
        GitHubClient(runner).merge_pull_request(
            "acme/widgets", 17, "a" * 40, method="octopus"
        )

    assert runner.calls == []


@pytest.mark.parametrize(
    "field,value",
    [("mergeable", None), ("mergeable_state", "unknown"), ("draft", None)],
)
def test_github_client_fails_closed_on_unknown_merge_state(
    field: str, value: object
) -> None:
    argv = ("gh", "api", "repos/acme/widgets/pulls/17")
    pull = canonical_pull()
    pull.update(
        {
            "draft": False,
            "mergeable": True,
            "mergeable_state": "clean",
            "merged": False,
            "merge_commit_sha": None,
        }
    )
    pull[field] = value

    with pytest.raises(GitHubClientError, match="merge state"):
        GitHubClient(RecordingRunner({argv: pull})).get_merge_state("acme/widgets", 17)


def test_github_client_accepts_terminal_merged_truth_when_mergeability_is_no_longer_applicable(
) -> None:
    argv = ("gh", "api", "repos/acme/widgets/pulls/17")
    pull = canonical_pull()
    pull.update(
        {
            "state": "closed",
            "draft": False,
            "mergeable": None,
            "mergeable_state": "unknown",
            "merged": True,
            "merge_commit_sha": "c" * 40,
        }
    )

    state = GitHubClient(RecordingRunner({argv: pull})).get_merge_state(
        "acme/widgets", 17
    )

    assert state.state == "CLOSED"
    assert state.merged is True
    assert state.mergeable is True
    assert state.merge_state_status == "MERGED"
    assert state.merge_commit_oid == "c" * 40


@pytest.mark.parametrize("payload", [{}, {"enabled": "false"}, [], None])
def test_github_client_fails_closed_on_invalid_actions_permission_shape(
    payload: object,
) -> None:
    argv = ("gh", "api", "repos/acme/widgets/actions/permissions")

    with pytest.raises(GitHubClientError, match="Actions permissions"):
        GitHubClient(RecordingRunner({argv: payload})).actions_enabled("acme/widgets")


def canonical_pull(number: int = 17, head_sha: str = "a" * 40) -> dict[str, object]:
    return {
        "number": number,
        "state": "open",
        "base": {
            "repo": {"full_name": "acme/widgets"},
            "ref": "stable",
            "sha": "b" * 40,
        },
        "head": {
            "repo": {"full_name": "acme/widgets"},
            "ref": "codex/fix",
            "sha": head_sha,
        },
        "user": {"login": "owner"},
    }


def canonical_list_pull(
    number: int = 17,
    head_sha: str = "a" * 40,
    labels: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "number": number,
        "state": "OPEN",
        "headRepository": {"nameWithOwner": "acme/widgets"},
        "author": {"login": "owner"},
        "headRefName": "codex/fix",
        "headRefOid": head_sha,
        "baseRefName": "stable",
        "baseRefOid": "b" * 40,
        "updatedAt": "2026-08-26T08:00:00Z",
        "labels": [{"name": label} for label in labels],
    }


def canonical_feedback(
    feedback_id: str,
    body: str | None,
    *,
    submitted_at: str | None = "2026-08-24T00:00:00Z",
    created_at: str = "2026-08-24T00:00:00Z",
) -> dict[str, object]:
    return {
        "id": feedback_id,
        "body": body,
        "created_at": created_at,
        "submitted_at": submitted_at,
        "user": {"login": "reviewer", "type": "User"},
        "author_association": "MEMBER",
    }


def feedback_responses(body: str) -> dict[tuple[str, ...], object]:
    return {
        (
            "gh",
            "api",
            "--paginate",
            "--slurp",
            "repos/acme/widgets/issues/17/comments?per_page=100",
        ): [[canonical_feedback("issue-1", body)]],
        (
            "gh",
            "api",
            "--paginate",
            "--slurp",
            "repos/acme/widgets/pulls/17/comments?per_page=100",
        ): [[]],
        (
            "gh",
            "api",
            "--paginate",
            "--slurp",
            "repos/acme/widgets/pulls/17/reviews?per_page=100",
        ): [[]],
    }

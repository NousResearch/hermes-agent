"""Canonical, fixed-argv GitHub reads for feedback intake."""

from __future__ import annotations

import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Protocol
from urllib.parse import quote

from .policy import PullRequest, Reviewer


class GitHubClientError(RuntimeError):
    """Canonical GitHub data was unavailable or did not have the required shape."""


class MergeStateStillComputingError(GitHubClientError):
    """GitHub has not finished computing this PR's mergeability yet.

    Distinct from GitHubClientError proper: this is expected, self-resolving
    eventual-consistency lag on a PR that has not been touched recently, not
    evidence that GitHub reads are actually failing. Callers should treat it
    as a benign "try again next cycle" skip, not a hard failure.
    """


MAX_FEEDBACK_BODY_CHARS = 16_384
# Bumped from 100: a single-operator repo generating many PRs in parallel
# (burndown-phase branches, PR-repair follow-ups) can genuinely exceed 100
# open PRs at once, and the discovery-cap check must fail closed rather than
# silently operate on a truncated page -- so this has to stay ahead of real
# volume, not just today's count.
MAX_DISCOVERED_PULL_REQUESTS = 300
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
_MERGE_FLAGS = {"squash": "--squash", "rebase": "--rebase", "merge": "--merge"}


class CommandRunner(Protocol):
    def run(self, argv: list[str]) -> str: ...


class SubprocessCommandRunner:
    """The only production command boundary; it never invokes a shell."""

    def __init__(
        self,
        *,
        sleeper: Callable[[float], None] = time.sleep,
        rate_limit_backoff: float = 1.0,
    ) -> None:
        self._sleeper = sleeper
        self._rate_limit_backoff = max(0.0, min(float(rate_limit_backoff), 2.0))

    def run(self, argv: list[str]) -> str:
        for attempt in range(2):
            try:
                completed = subprocess.run(
                    argv,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            except subprocess.TimeoutExpired as error:
                # A concurrent burst of `gh` invocations (this client is called
                # from a 6-worker thread pool) can contend on the OS credential
                # store on the first call after a idle period and blow the 30s
                # budget even though gh itself is healthy; one retry clears it.
                if attempt == 0:
                    self._sleeper(self._rate_limit_backoff)
                    continue
                raise GitHubClientError("GitHub command failed") from error
            except OSError as error:
                raise GitHubClientError("GitHub command failed") from error
            if completed.returncode == 0:
                return completed.stdout
            if attempt == 0 and _is_rate_limit_failure(completed.stderr):
                self._sleeper(self._rate_limit_backoff)
                continue
            raise GitHubClientError("GitHub command failed")
        raise GitHubClientError("GitHub command failed")


def _is_rate_limit_failure(stderr: str) -> bool:
    normalized = str(stderr or "").casefold()
    return "rate limit" in normalized and (
        "403" in normalized or "429" in normalized or "secondary" in normalized
    )


@dataclass(frozen=True, slots=True)
class Feedback:
    kind: str
    feedback_id: str
    reviewer: Reviewer
    body: str
    created_at: datetime
    is_bot: bool


@dataclass(frozen=True, slots=True)
class RepositoryMergePolicy:
    squash: bool
    rebase: bool
    merge: bool

    def allows(self, method: str) -> bool:
        return bool(getattr(self, method, False))


@dataclass(frozen=True, slots=True)
class PullRequestMergeState:
    repository: str
    number: int
    state: str
    is_draft: bool
    mergeable: bool
    merge_state_status: str
    base_branch: str
    base_sha: str
    head_repository: str
    author_login: str
    head_ref_name: str
    head_sha: str
    merged: bool
    merge_commit_oid: str | None
    labels: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReviewState:
    review_decision: str | None
    unresolved_thread_count: int


@dataclass(frozen=True, slots=True)
class CheckState:
    actions_enabled: bool
    all_green: bool
    check_count: int
    # GitHub Actions is enabled as a repository setting and reporting check
    # runs, but every job fails immediately with the account-payment/spending
    # -limit annotation rather than actually executing. Distinct from
    # actions_enabled=False (Actions turned off entirely): here Actions is on
    # but structurally unable to run anything until billing is resolved.
    billing_blocked: bool = False
    # At least one check run's canonical conclusion is GitHub's own
    # "action_required" -- a workflow waiting on a human (first-time
    # contributor approval to run, a required workflow that never started, a
    # third-party app requesting manual follow-up). No code change can
    # satisfy this: it is not a failing test or a lint error, so the repair
    # steward must not spend a repair attempt on it and the merge maintainer
    # must never treat it as a transient red check to wait out.
    action_required: bool = False


class GitHubClient:
    """Reads only the canonical PR and review endpoints using literal argv."""

    REVIEW_STATE_QUERY = (
        "query($owner:String!,$name:String!,$number:Int!){repository(owner:$owner,name:$name){"
        "pullRequest(number:$number){reviewDecision reviewThreads(first:100){nodes{isResolved}"
        "pageInfo{hasNextPage}}}}}"
    )
    REVIEW_THREAD_QUERY = (
        "query($owner:String!,$name:String!,$number:Int!){repository(owner:$owner,name:$name){"
        "pullRequest(number:$number){headRefOid reviewThreads(first:100){nodes{id isResolved "
        "comments(first:100){nodes{databaseId} pageInfo{hasNextPage}}} "
        "pageInfo{hasNextPage}}}}}"
    )
    RESOLVE_REVIEW_THREAD_MUTATION = (
        "mutation($threadId:ID!){resolveReviewThread(input:{threadId:$threadId}){"
        "thread{id isResolved}}}"
    )

    def __init__(self, runner: CommandRunner | None = None) -> None:
        self._runner = runner or SubprocessCommandRunner()

    def list_open_pull_requests(
        self, repository: str, owner_login: str
    ) -> tuple[PullRequest, ...]:
        payload = self._json(
            [
                "gh",
                "pr",
                "list",
                "--repo",
                repository,
                "--state",
                "open",
                "--author",
                owner_login,
                "--limit",
                str(MAX_DISCOVERED_PULL_REQUESTS),
                "--json",
                "number,state,headRepository,author,headRefName,headRefOid,updatedAt",
            ]
        )
        if not isinstance(payload, list) or any(
            not isinstance(row, dict) for row in payload
        ):
            raise GitHubClientError(
                "GitHub pull request list was not a list of objects"
            )
        if len(payload) >= MAX_DISCOVERED_PULL_REQUESTS:
            raise GitHubClientError(
                "GitHub owned pull request query reached its coverage cap"
            )
        return tuple(_listed_pull_request(repository, row) for row in payload)

    def list_all_open_pull_requests(self, repository: str) -> tuple[PullRequest, ...]:
        """Read every open PR so maintenance never races an unmerged change."""

        repository = _validated_repository(repository)
        payload = self._json(
            [
                "gh",
                "pr",
                "list",
                "--repo",
                repository,
                "--state",
                "open",
                "--limit",
                str(MAX_DISCOVERED_PULL_REQUESTS),
                "--json",
                "number,state,headRepository,author,headRefName,headRefOid,updatedAt",
            ]
        )
        if not isinstance(payload, list) or any(
            not isinstance(row, dict) for row in payload
        ):
            raise GitHubClientError(
                "GitHub pull request list was not a list of objects"
            )
        if len(payload) >= MAX_DISCOVERED_PULL_REQUESTS:
            raise GitHubClientError(
                "GitHub pull request query reached its coverage cap"
            )
        return tuple(_listed_pull_request(repository, row) for row in payload)

    def get_branch_head(self, repository: str, branch: str) -> str:
        repository = _validated_repository(repository)
        branch = _required_string(branch)
        payload = self._read_object(
            f"repos/{repository}/branches/{quote(branch, safe='')}"
        )
        try:
            return _validated_sha(payload["commit"]["sha"])
        except (KeyError, TypeError, ValueError) as error:
            raise GitHubClientError("GitHub branch head was unavailable") from error

    def get_pull_request(self, repository: str, number: int) -> PullRequest:
        row = self._read_object(f"repos/{repository}/pulls/{number}")
        return _pull_request(row)

    def actions_enabled(self, repository: str) -> bool:
        payload = self._json(["gh", "api", f"repos/{repository}/actions/permissions"])
        if not isinstance(payload, dict) or not isinstance(
            payload.get("enabled"), bool
        ):
            raise GitHubClientError("GitHub Actions permissions had an invalid shape")
        return payload["enabled"]

    def repository_is_private(self, repository: str) -> bool:
        payload = self._read_object(f"repos/{_validated_repository(repository)}")
        private = payload.get("private")
        if not isinstance(private, bool):
            raise GitHubClientError("GitHub repository privacy had an invalid shape")
        return private

    def get_repository_merge_policy(self, repository: str) -> RepositoryMergePolicy:
        payload = self._read_object(f"repos/{_validated_repository(repository)}")
        fields = ("allow_squash_merge", "allow_rebase_merge", "allow_merge_commit")
        if any(not isinstance(payload.get(field), bool) for field in fields):
            raise GitHubClientError(
                "GitHub repository merge policy had an invalid shape"
            )
        return RepositoryMergePolicy(
            squash=payload["allow_squash_merge"],
            rebase=payload["allow_rebase_merge"],
            merge=payload["allow_merge_commit"],
        )

    def get_merge_state(self, repository: str, number: int) -> PullRequestMergeState:
        repository = _validated_repository(repository)
        number = _positive_number(number)
        row = self._read_object(f"repos/{repository}/pulls/{number}")
        # GitHub computes mergeability lazily: a PR that hasn't been touched
        # recently reports mergeable=null / mergeable_state="unknown" on the
        # first read and only starts the real computation as a side effect of
        # that read. A second read a few seconds later almost always has the
        # real value, so poll once before treating this as unavailable --
        # without this, a stale-but-perfectly-normal open PR is
        # indistinguishable from a genuine API failure.
        terminal_merged = (
            isinstance(row.get("state"), str)
            and row["state"].casefold() == "closed"
            and row.get("merged") is True
            and row.get("mergeable") is None
            and isinstance(row.get("mergeable_state"), str)
            and row["mergeable_state"].casefold() == "unknown"
            and isinstance(row.get("merge_commit_sha"), str)
            and _SHA.fullmatch(row["merge_commit_sha"])
        )
        for delay in (3.0, 6.0):
            if terminal_merged or row.get("mergeable") is not None or row.get("mergeable_state") != "unknown":
                break
            time.sleep(delay)
            row = self._read_object(f"repos/{repository}/pulls/{number}")
            terminal_merged = (
                isinstance(row.get("state"), str)
                and row["state"].casefold() == "closed"
                and row.get("merged") is True
                and row.get("mergeable") is None
                and isinstance(row.get("mergeable_state"), str)
                and row["mergeable_state"].casefold() == "unknown"
                and isinstance(row.get("merge_commit_sha"), str)
                and _SHA.fullmatch(row["merge_commit_sha"])
            )
        if not terminal_merged and row.get("mergeable") is None and row.get("mergeable_state") == "unknown":
            raise MergeStateStillComputingError(
                "GitHub has not finished computing mergeability for this PR yet"
            )
        try:
            base = row["base"]
            head = row["head"]
            state = row["state"]
            is_draft = row["draft"]
            mergeable = row["mergeable"]
            merge_state_status = row["mergeable_state"]
            merged = row["merged"]
            merge_commit_oid = row.get("merge_commit_sha")
            raw_labels = row.get("labels", [])
            if (
                isinstance(state, str)
                and state.casefold() == "closed"
                and merged is True
                and mergeable is None
                and isinstance(merge_state_status, str)
                and merge_state_status.casefold() == "unknown"
                and isinstance(merge_commit_oid, str)
                and _SHA.fullmatch(merge_commit_oid)
            ):
                # GitHub stops computing pre-merge fields after a successful
                # merge. Canonical terminal truth is stronger than the now
                # inapplicable mergeability probe, so normalize only this
                # exact REST shape for idempotent post-write reconciliation.
                mergeable = True
                merge_state_status = "merged"
            if (
                not isinstance(state, str)
                or not isinstance(is_draft, bool)
                or not isinstance(mergeable, bool)
                or not isinstance(merge_state_status, str)
                or merge_state_status.casefold() == "unknown"
                or not isinstance(merged, bool)
                or (merge_commit_oid is not None and not _SHA.fullmatch(merge_commit_oid))
                or not isinstance(raw_labels, list)
                or any(
                    not isinstance(label, dict) or not isinstance(label.get("name"), str)
                    for label in raw_labels
                )
            ):
                raise TypeError("merge state field has invalid shape")
            pull = PullRequestMergeState(
                repository=repository,
                number=_positive_number(row["number"]),
                state=state.upper(),
                is_draft=is_draft,
                mergeable=mergeable,
                merge_state_status=merge_state_status.upper(),
                base_branch=_required_string(base["ref"]),
                base_sha=_validated_sha(base["sha"]),
                head_repository=_validated_repository(head["repo"]["full_name"]),
                author_login=_required_string(row["user"]["login"]),
                head_ref_name=_required_string(head["ref"]),
                head_sha=_validated_sha(head["sha"]),
                merged=merged,
                merge_commit_oid=merge_commit_oid,
                labels=tuple(label["name"] for label in raw_labels),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise GitHubClientError(
                "GitHub pull request merge state was unavailable"
            ) from error
        if pull.number != number:
            raise GitHubClientError("GitHub pull request merge state identity changed")
        return pull

    def get_review_state(self, repository: str, number: int) -> ReviewState:
        repository = _validated_repository(repository)
        number = _positive_number(number)
        owner, name = repository.split("/", 1)
        payload = self._json(
            [
                "gh",
                "api",
                "graphql",
                "-f",
                "query=" + self.REVIEW_STATE_QUERY,
                "-F",
                f"owner={owner}",
                "-F",
                f"name={name}",
                "-F",
                f"number={number}",
            ]
        )
        try:
            pull = payload["data"]["repository"]["pullRequest"]
            decision = pull["reviewDecision"]
            threads = pull["reviewThreads"]
            nodes = threads["nodes"]
            has_next_page = threads["pageInfo"]["hasNextPage"]
            if decision is not None and decision not in {
                "APPROVED",
                "CHANGES_REQUESTED",
                "REVIEW_REQUIRED",
            }:
                raise TypeError("review decision is unknown")
            if (
                not isinstance(nodes, list)
                or not isinstance(has_next_page, bool)
                or has_next_page
            ):
                raise TypeError("review thread coverage is incomplete")
            if any(
                not isinstance(node, dict)
                or not isinstance(node.get("isResolved"), bool)
                for node in nodes
            ):
                raise TypeError("review thread is malformed")
        except (KeyError, TypeError) as error:
            raise GitHubClientError("GitHub review state was unavailable") from error
        return ReviewState(
            review_decision=decision,
            unresolved_thread_count=sum(not node["isResolved"] for node in nodes),
        )

    def get_check_state(self, repository: str, head_sha: str) -> CheckState:
        repository = _validated_repository(repository)
        head_sha = _validated_sha(head_sha)
        if not self.actions_enabled(repository):
            return CheckState(actions_enabled=False, all_green=True, check_count=0)
        check_payload = self._read_object(
            f"repos/{repository}/commits/{head_sha}/check-runs?per_page=100"
        )
        status_payload = self._read_object(
            f"repos/{repository}/commits/{head_sha}/status?per_page=100"
        )
        try:
            total_count = check_payload["total_count"]
            check_runs = check_payload["check_runs"]
            status_state = status_payload["state"]
            statuses = status_payload["statuses"]
            if (
                not isinstance(total_count, int)
                or isinstance(total_count, bool)
                or not isinstance(check_runs, list)
                or total_count != len(check_runs)
                or total_count >= 100
                or not isinstance(statuses, list)
                or len(statuses) >= 100
                or status_state not in {"success", "failure", "pending", "error"}
            ):
                raise TypeError("check coverage is incomplete")
            check_green = all(
                isinstance(run, dict)
                and run.get("status") == "completed"
                and run.get("conclusion") in {"success", "neutral", "skipped"}
                for run in check_runs
            )
        except (KeyError, TypeError) as error:
            raise GitHubClientError("GitHub check state was unavailable") from error
        all_green = check_green and status_state == "success"
        billing_blocked = False if all_green else self._billing_blocked(
            repository, check_runs
        )
        action_required = any(
            isinstance(run, dict) and run.get("conclusion") == "action_required"
            for run in check_runs
        )
        return CheckState(
            actions_enabled=True,
            all_green=all_green,
            check_count=total_count + len(statuses),
            billing_blocked=billing_blocked,
            action_required=action_required,
        )

    def _billing_blocked(
        self, repository: str, check_runs: list[dict[str, Any]]
    ) -> bool:
        """Return whether a failing run carries GitHub's billing-lockout annotation.

        Only one annotated, non-passing run needs to be inspected: the billing
        lockout is an account-wide condition, not a per-job failure, so every
        job in an affected run carries the identical annotation.
        """

        for run in check_runs:
            if (
                not isinstance(run, dict)
                or run.get("status") != "completed"
                or run.get("conclusion") in {"success", "neutral", "skipped"}
            ):
                continue
            run_id = run.get("id")
            output = run.get("output")
            annotations_count = (
                output.get("annotations_count") if isinstance(output, dict) else None
            )
            if not isinstance(run_id, int) or not annotations_count:
                continue
            try:
                annotations = self._read_pages(
                    f"repos/{repository}/check-runs/{run_id}/annotations"
                )
            except GitHubClientError:
                continue
            if any(_is_billing_lockout_message(a.get("message")) for a in annotations):
                return True
        return False

    def merge_pull_request(
        self, repository: str, number: int, head_sha: str, *, method: str
    ) -> None:
        repository = _validated_repository(repository)
        number = _positive_number(number)
        head_sha = _validated_sha(head_sha)
        flag = _MERGE_FLAGS.get(method)
        if flag is None:
            raise ValueError("method must be squash, rebase, or merge")
        self._runner.run(
            [
                "gh",
                "pr",
                "merge",
                str(number),
                "--repo",
                repository,
                flag,
                "--match-head-commit",
                head_sha,
            ]
        )

    def post_issue_comment(self, repository: str, number: int, body: str) -> None:
        """Post one bounded factual PR comment through fixed argv."""

        repository = _validated_repository(repository)
        number = _positive_number(number)
        if not isinstance(body, str) or not body.strip() or len(body) > 4000:
            raise ValueError("comment body must contain 1 to 4000 characters")
        self._runner.run(
            [
                "gh",
                "api",
                f"repos/{repository}/issues/{number}/comments",
                "--method",
                "POST",
                "--field",
                f"body={body}",
            ]
        )

    def resolve_review_thread_for_comment(
        self,
        repository: str,
        number: int,
        comment_id: str,
        *,
        expected_head_sha: str,
    ) -> bool:
        """Resolve only the complete review thread containing one exact REST comment ID."""

        repository = _validated_repository(repository)
        number = _positive_number(number)
        expected_head_sha = _validated_sha(expected_head_sha)
        try:
            database_id = int(comment_id)
        except (TypeError, ValueError) as error:
            raise ValueError("review comment ID must be a positive integer") from error
        if database_id <= 0:
            raise ValueError("review comment ID must be a positive integer")
        owner, name = repository.split("/", 1)
        payload = self._json(
            [
                "gh",
                "api",
                "graphql",
                "-f",
                "query=" + self.REVIEW_THREAD_QUERY,
                "-F",
                f"owner={owner}",
                "-F",
                f"name={name}",
                "-F",
                f"number={number}",
            ]
        )
        try:
            if "errors" in payload:
                raise TypeError("GraphQL query returned errors")
            pull = payload["data"]["repository"]["pullRequest"]
            observed_head_sha = _validated_sha(pull["headRefOid"])
            if observed_head_sha.casefold() != expected_head_sha.casefold():
                raise TypeError("pull request head changed")
            threads = pull["reviewThreads"]
            nodes = threads["nodes"]
            if threads["pageInfo"]["hasNextPage"] is not False or not isinstance(
                nodes, list
            ):
                raise TypeError("review thread coverage is incomplete")
            matches = []
            for node in nodes:
                if not isinstance(node, dict):
                    raise TypeError("review thread is malformed")
                comments = node["comments"]
                comment_nodes = comments["nodes"]
                if comments["pageInfo"]["hasNextPage"] is not False or not isinstance(
                    comment_nodes, list
                ):
                    raise TypeError("review comment coverage is incomplete")
                if any(not isinstance(comment, dict) for comment in comment_nodes):
                    raise TypeError("review comment is malformed")
                if any(
                    type(comment.get("databaseId")) is int
                    and comment["databaseId"] == database_id
                    for comment in comment_nodes
                ):
                    matches.append(node)
            if len(matches) != 1:
                raise TypeError("review comment did not identify exactly one thread")
            thread_id = _required_string(matches[0]["id"])
            resolved = matches[0]["isResolved"]
            if not isinstance(resolved, bool):
                raise TypeError("review thread resolution state is invalid")
        except (KeyError, TypeError, ValueError) as error:
            raise GitHubClientError("GitHub review thread was unavailable") from error
        if resolved:
            return False
        mutation = self._json(
            [
                "gh",
                "api",
                "graphql",
                "-f",
                "query=" + self.RESOLVE_REVIEW_THREAD_MUTATION,
                "-F",
                f"threadId={thread_id}",
            ]
        )
        try:
            if "errors" in mutation:
                raise TypeError("GraphQL mutation returned errors")
            result = mutation["data"]["resolveReviewThread"]["thread"]
            if result["id"] != thread_id or result["isResolved"] is not True:
                raise TypeError("review thread resolution was not confirmed")
        except (KeyError, TypeError) as error:
            raise GitHubClientError("GitHub review thread resolution failed") from error
        return True

    def list_feedback(self, repository: str, number: int) -> tuple[Feedback, ...]:
        endpoints = (
            f"repos/{repository}/issues/{number}/comments?per_page=100",
            f"repos/{repository}/pulls/{number}/comments?per_page=100",
            f"repos/{repository}/pulls/{number}/reviews?per_page=100",
        )
        with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
            issue_comments, review_comments, reviews = executor.map(
                self._read_pages, endpoints
            )
        feedback = [
            *(
                _feedback("issue_comment", row, timestamp_key="created_at")
                for row in issue_comments
            ),
            *(
                _feedback("review_comment", row, timestamp_key="created_at")
                for row in review_comments
            ),
        ]
        feedback.extend(
            _feedback("review", row, timestamp_key="submitted_at")
            for row in reviews
            if isinstance(row, dict) and row.get("submitted_at") is not None
        )
        feedback.sort(key=lambda item: item.created_at)
        return tuple(feedback)

    def _read_pages(self, endpoint: str) -> tuple[dict[str, Any], ...]:
        payload = self._json(["gh", "api", "--paginate", "--slurp", endpoint])
        if not isinstance(payload, list) or any(
            not isinstance(page, list) for page in payload
        ):
            raise GitHubClientError("GitHub paginated response was not a list of pages")
        rows = tuple(row for page in payload for row in page)
        if any(not isinstance(row, dict) for row in rows):
            raise GitHubClientError("GitHub response row was not an object")
        return rows

    def _read_object(self, endpoint: str) -> dict[str, Any]:
        payload = self._json(["gh", "api", endpoint])
        if not isinstance(payload, dict):
            raise GitHubClientError("GitHub response was not an object")
        return payload

    def _json(self, argv: list[str]) -> object:
        try:
            return json.loads(self._runner.run(argv))
        except (json.JSONDecodeError, TypeError) as error:
            raise GitHubClientError("GitHub response was not valid JSON") from error


def _pull_request(row: dict[str, Any]) -> PullRequest:
    try:
        base = row["base"]
        head = row["head"]
        raw_labels = row.get("labels", [])
        if not isinstance(raw_labels, list) or any(
            not isinstance(label, dict) or not isinstance(label.get("name"), str)
            for label in raw_labels
        ):
            raise TypeError("labels must be a list of named objects")
        return PullRequest(
            number=row["number"],
            state=row["state"],
            base_repository=base["repo"]["full_name"],
            head_repository=head["repo"]["full_name"],
            author_login=row["user"]["login"],
            head_ref_name=head["ref"],
            head_sha=head["sha"],
            labels=tuple(label["name"] for label in raw_labels),
            base_branch=base["ref"],
            base_sha=base["sha"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise GitHubClientError(
            "GitHub pull request has missing required fields"
        ) from error


def _listed_pull_request(base_repository: str, row: dict[str, Any]) -> PullRequest:
    try:
        return PullRequest(
            number=row["number"],
            state=row["state"],
            base_repository=base_repository,
            head_repository=row["headRepository"]["nameWithOwner"],
            author_login=row["author"]["login"],
            head_ref_name=row["headRefName"],
            head_sha=row["headRefOid"],
            updated_at=_timestamp(row["updatedAt"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise GitHubClientError(
            "GitHub pull request has missing required fields"
        ) from error


def _feedback(kind: str, row: dict[str, Any], *, timestamp_key: str) -> Feedback:
    try:
        user = row["user"]
        body = row["body"]
        if kind == "review" and body is None:
            body = ""
        if not isinstance(body, str):
            raise TypeError("body must be a string")
        return Feedback(
            kind=kind,
            feedback_id=str(row["id"]),
            reviewer=Reviewer(user["login"], row.get("author_association")),
            body=body[:MAX_FEEDBACK_BODY_CHARS],
            created_at=_timestamp(row[timestamp_key]),
            is_bot=user.get("type") == "Bot",
        )
    except (KeyError, TypeError, ValueError) as error:
        raise GitHubClientError(
            "GitHub feedback has missing required fields"
        ) from error


_BILLING_LOCKOUT_PHRASES = (
    "recent account payments have failed",
    "spending limit needs to be increased",
)


def _is_billing_lockout_message(message: object) -> bool:
    """Match GitHub's literal check-run annotation for an Actions billing lockout."""

    if not isinstance(message, str):
        return False
    lowered = message.casefold()
    return any(phrase in lowered for phrase in _BILLING_LOCKOUT_PHRASES)


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("timestamp is required")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("timestamp is invalid") from error
    if timestamp.tzinfo is None:
        raise ValueError("timestamp requires a timezone")
    return timestamp


def _validated_repository(value: object) -> str:
    if not isinstance(value, str) or not _REPOSITORY.fullmatch(value):
        raise ValueError("repository must be an exact owner/repository name")
    return value


def _positive_number(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("number must be a positive integer")
    return value


def _required_string(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("required string was absent")
    return value.strip()


def _validated_sha(value: object) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError("head_sha must be a full hexadecimal SHA")
    return value.lower()

"""Hermes CLI wiring for governed GitHub pull-request feedback intake."""

from __future__ import annotations

from .ci_contract import manifest_path as ci_manifest_path

import argparse
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

from .cli_audit_task import owns_current_audit_task
from .worker_contract import configured_assignees, worker_contract_enabled
from .controller import (
    AUTO_DISPATCH_MAX_RUNTIME_SECONDS,
    KanbanTask,
    LocalGitRepository,
    PooledLocalGitRepository,
    ScanController,
)
from .ci_coordinator import CIAuditJob, GroupedCICoordinator
from .ci_runner import (
    CIAuditIdentity,
    CIAuditReceipt,
    CIValidationError,
    LocalCIRunner,
    _required_lanes,
)
from .github_client import GitHubClient, GitHubClientError
from .ledger import (
    FeedbackLedger,
    LedgerStateError,
    parse_maintenance_command_evidence,
)
from .merge_controller import (
    CanonicalMergeEvidenceSource,
    MergeController,
    MergeDecision,
    _codex_reviewed_head,
    ci_receipt_comment_from_feedback,
)
from .policy import (
    FeedbackReceipt,
    PluginPolicy,
    ReleaseMaintenancePolicy,
    codex_review_trigger_comment,
    codex_review_trigger_requested,
    hermes_attribution_line,
    load_policy,
)
from .post_merge import PostMergeExecutor
from .stack import StackEntry
from .stack_controller import StackController, _ordered
from .superseded import SupersededPullRequestController
from .superseded_feedback import SupersededFeedbackController
from .repair_controller import (
    PR_REPAIR_ATTRIBUTION_PREFIX,
    RepairController,
    pr_repair_attribution_required,
)
from .release_maintenance import (
    FINAL_LANE,
    MaintenanceGitHub,
    MaintenanceWorkspaces,
    ReleaseMaintenanceController,
)

try:
    from hermes_constants import get_default_hermes_root
except ImportError:

    def get_default_hermes_root() -> Path:
        configured = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
        return (
            configured.parent.parent
            if configured.parent.name == "profiles"
            else configured
        )


_MISSING = object()
_LOGGER = logging.getLogger(__name__)
_FULL_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(token|secret|password|authorization|api[_-]?key)\s*[:=]\s*\S+"
)
# kind is deliberately unconstrained (any word token, not an enumerated set):
# this same marker covers pr_repair and ci_repair (repair_controller.py /
# controller.py's typed CI fixer) as well as every ordinary admitted-feedback
# kind (issue_comment, review_comment, review) via controller.py's _task().
# The receipt's own FeedbackReceipt already validates feedback_kind against
# its canonical set; this regex only needs to prove *some* completed-and-
# matching-head marker exists, not police which kind it names.
_PR_REPAIR_RECEIPT_COMMENT = re.compile(
    r"<!--\s*pr-maintenance-receipt:v1\s+status=completed\s+kind=\w+\s+"
    r"head=([0-9a-fA-F]{40,64})\s*-->"
)
# Feedback kinds whose worker-completed reply must carry the marker above.
# pr_local_ci completes through a different typed-receipt flow (audit-pr),
# and pr_actions_needed starts blocked and is never worker-completed at all.
_MARKER_REQUIRED_FEEDBACK_KINDS = frozenset(
    {"pr_repair", "issue_comment", "review_comment", "review"}
)
_REVIEW_EVENTS = frozenset({"APPROVE", "REQUEST_CHANGES", "COMMENT"})
_FEEDBACK_DETAILS = re.compile(r"(?is)<details\b.*?</details>")
_FEEDBACK_HTML_COMMENT = re.compile(r"(?s)<!--.*?-->")
_FEEDBACK_HTML_TAG = re.compile(r"(?s)<[^>]{1,200}>")
_FEEDBACK_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]{0,200}\]\([^)]+\)")
_FEEDBACK_URL = re.compile(r"https?://[^\s\"')>]+")


def _feedback_body_excerpt(body: str, *, limit: int = 1800) -> str:
    text = str(body or "")
    text = _FEEDBACK_DETAILS.sub(" ", text)
    text = _FEEDBACK_HTML_COMMENT.sub(" ", text)
    text = _FEEDBACK_MARKDOWN_IMAGE.sub(" ", text)
    text = _FEEDBACK_URL.sub("<url>", text)
    text = _FEEDBACK_HTML_TAG.sub(" ", text)
    text = _SECRET_ASSIGNMENT.sub(r"\1=<redacted>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _factual_reply_is_missing(
    github: GitHubClient, receipt: FeedbackReceipt, *, resolved_head_sha: str
) -> bool:
    """Whether no comment yet carries this exact completion's required receipt marker.

    Covers every feedback kind whose worker-completed reply is required to
    carry the marker (see _MARKER_REQUIRED_FEEDBACK_KINDS) -- repair receipts
    and ordinary admitted review/issue-comment feedback alike.
    complete-feedback only reread the resolved PR head match before this
    check existed; a worker could push a fix, skip the required factual
    reply, and still successfully complete the task with no trace anything
    was skipped. This independently rereads canonical PR comments the same
    way every other completion gate in this plugin rereads canonical state,
    instead of trusting the worker's self-report that it replied.
    """

    try:
        feedback = github.list_feedback(receipt.repository, receipt.pr_number)
    except GitHubClientError:
        return True
    for item in feedback:
        match = _PR_REPAIR_RECEIPT_COMMENT.search(item.body)
        if not match or match.group(1).casefold() != resolved_head_sha.casefold():
            continue
        if not pr_repair_attribution_required(receipt.repository):
            return False
        if PR_REPAIR_ATTRIBUTION_PREFIX in item.body:
            return False
    return True


def _retrigger_codex_review(
    github: GitHubClient, repository: str, pr_number: int, resolved_head_sha: str
) -> str:
    """Request an exact-head review once and expose connector authorization failures."""

    try:
        feedback = github.list_feedback(repository, pr_number)
    except GitHubClientError:
        return "unavailable"
    if _codex_reviewed_head(feedback, resolved_head_sha):
        return "already_current"
    requests = [
        item for item in feedback
        if codex_review_trigger_requested(item.body, resolved_head_sha)
    ]
    if requests:
        latest_request = max(item.created_at for item in requests)
        if any(
            item.reviewer.login.casefold() == "chatgpt-codex-connector[bot]"
            and item.created_at >= latest_request
            and "create a codex account and connect to github" in item.body.casefold()
            for item in feedback
        ):
            return "authorization_required"
        return "already_requested"
    try:
        github.post_issue_comment(
            repository,
            pr_number,
            codex_review_trigger_comment(resolved_head_sha),
        )
    except GitHubClientError:
        return "unavailable"
    return "triggered"


@dataclass(frozen=True, slots=True)
class KanbanCommandResult:
    returncode: int
    stdout: str
    stderr: str = ""


@dataclass(frozen=True, slots=True)
class DoctorCommandResult:
    returncode: int
    stdout: str


class DoctorCommandRunner(Protocol):
    def which(self, executable: str) -> str | None: ...

    def run(self, argv: list[str]) -> DoctorCommandResult: ...


class SubprocessDoctorRunner:
    def which(self, executable: str) -> str | None:
        resolved = shutil.which(executable)
        if not resolved:
            return None
        path = Path(resolved).resolve()
        if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
            return None
        return str(path)

    def run(self, argv: list[str]) -> DoctorCommandResult:
        try:
            completed = subprocess.run(
                argv,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return DoctorCommandResult(1, "")
        return DoctorCommandResult(completed.returncode, completed.stdout)


class DoctorProbe:
    """Read-only readiness probes with an injected external-command boundary."""

    def __init__(
        self,
        hermes_root: Path,
        runner: DoctorCommandRunner | None = None,
        *,
        hermes_source_root: Path | None = None,
    ) -> None:
        self._hermes_root = Path(hermes_root)
        self._runner = runner or SubprocessDoctorRunner()
        if hermes_source_root is None:
            from hermes_cli._startup_fast import project_root_str

            hermes_source_root = Path(project_root_str())
        self._hermes_source_root = Path(hermes_source_root).resolve()

    def checks(self, policy: PluginPolicy, ledger_path: Path) -> dict[str, str]:
        gh = self._runner.which("gh")
        hermes = self._runner.which("hermes")
        checks = {
            "gh_executable": gh is not None,
            "hermes_executable": bool(
                hermes and self._hermes_executable_ready(hermes)
            ),
            "board": self._board_exists(policy.board or ""),
            "assignee": all(self._assignee_exists(name) for name in configured_assignees(policy)),
            "worker_completion_policy": all(
                worker_contract_enabled(self._hermes_root, name)
                for name in configured_assignees(policy)
            ),
            "ledger_access": self._ledger_access(ledger_path),
            "repository_worktree": self._repositories_ready(
                tuple(target.local_path for target in policy.targets.values()),
                ledger_path.parent / "worktrees",
            ),
        }
        return {name: "ok" if ok else "failed" for name, ok in checks.items()}

    def _hermes_executable_ready(self, executable: str) -> bool:
        try:
            result = self._runner.run([executable, "--version-local"])
        except Exception:  # noqa: BLE001 - doctor reports failure and continues.
            return False
        if result.returncode != 0:
            return False
        install_lines = [
            line.removeprefix("Install directory:").strip()
            for line in result.stdout.splitlines()
            if line.startswith("Install directory:")
        ]
        if len(install_lines) != 1 or not install_lines[0]:
            return False
        try:
            reported_root = Path(install_lines[0])
            return (
                reported_root.is_absolute()
                and reported_root.resolve() == self._hermes_source_root
            )
        except OSError:
            return False

    def _board_exists(self, board: str) -> bool:
        if not _safe_name(board):
            return False
        if board == "default":
            return self._hermes_root.is_dir()
        directory = self._hermes_root / "kanban" / "boards" / board
        return (directory / "board.json").is_file() or (
            directory / "kanban.db"
        ).is_file()

    def _assignee_exists(self, assignee: str) -> bool:
        if not _safe_name(assignee):
            return False
        if assignee == "default":
            return self._hermes_root.is_dir()
        return (self._hermes_root / "profiles" / assignee / "config.yaml").is_file()

    def _ledger_access(self, path: Path) -> bool:
        try:
            if path.exists():
                if not path.is_file():
                    return False
                with path.open("rb") as handle:
                    header = handle.read(16)
                return header == b"SQLite format 3\x00"
            return _nearest_existing_parent_access(path.parent)
        except OSError:
            return False

    def _repositories_ready(
        self, repositories: tuple[Path, ...], worktree_root: Path
    ) -> bool:
        git = self._runner.which("git")
        all_ready = git is not None and _nearest_existing_parent_access(worktree_root)
        if git is None:
            return False
        for repository in repositories:
            try:
                top = self._runner.run(
                    [git, "-C", str(repository), "rev-parse", "--show-toplevel"]
                )
                common = self._runner.run(
                    [git, "-C", str(repository), "rev-parse", "--git-common-dir"]
                )
                worktrees = self._runner.run(
                    [git, "-C", str(repository), "worktree", "list", "--porcelain"]
                )
                repository_ready = (
                    top.returncode == 0
                    and Path(top.stdout.strip()).resolve() == repository.resolve()
                    and common.returncode == 0
                    and bool(common.stdout.strip())
                    and worktrees.returncode == 0
                )
            except (OSError, RuntimeError, ValueError):
                repository_ready = False
            all_ready = all_ready and repository_ready
        return all_ready


class KanbanCommandRunner(Protocol):
    def run(self, argv: list[str]) -> KanbanCommandResult: ...


class SubprocessKanbanRunner:
    """Run the fixed Hermes Kanban command without a shell."""

    def run(self, argv: list[str]) -> KanbanCommandResult:
        try:
            completed = subprocess.run(
                argv,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError("Kanban task creation failed") from error
        return KanbanCommandResult(
            completed.returncode, completed.stdout, completed.stderr[:2000]
        )


class KanbanSubprocessClient:
    """Kanban adapter that accepts only a valid JSON task identity."""

    def __init__(self, runner: KanbanCommandRunner | None = None) -> None:
        self._runner = runner or SubprocessKanbanRunner()

    def create_or_get_task(self, task: KanbanTask) -> str:
        result = self._runner.run(_kanban_create_argv(task))
        if result.returncode != 0:
            raise RuntimeError(_kanban_create_error(result))
        try:
            payload = json.loads(result.stdout)
        except (TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("Kanban task creation failed") from error
        task_id = payload.get("id") if isinstance(payload, dict) else None
        if not isinstance(task_id, str) or not task_id.strip():
            raise RuntimeError("Kanban task creation failed")
        task_id = task_id.strip()
        existing = (
            payload.get("task")
            if isinstance(payload, dict) and isinstance(payload.get("task"), dict)
            else payload
        )
        needs_reconcile = _is_legacy_intake_task(existing, task)
        if (
            not needs_reconcile
            and isinstance(existing, dict)
            and existing.get("status") == "blocked"
        ):
            # Idempotent create returns only the task row. Fetch the bounded
            # lifecycle event once so a card blocked by the known protected
            # terminal replay defect can be reconciled after the defect is
            # fixed; unrelated blocked cards remain untouched.
            needs_reconcile = _is_legacy_intake_task(
                self.task_details(task.board, task_id), task
            )
        if needs_reconcile:
            self.reconcile_dispatch_task(task_id, task)
        return task_id

    def task_details(self, board: str, task_id: str) -> dict[str, object] | None:
        result = self._runner.run(
            ["hermes", "kanban", "--board", board, "show", task_id, "--json"]
        )
        if result.returncode != 0:
            if isinstance(result.stderr, str) and result.stderr.strip().startswith(
                "no such task:"
            ):
                return None
            raise RuntimeError("Kanban task lookup failed")
        try:
            payload = json.loads(result.stdout)
        except (TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("Kanban task lookup failed") from error
        task = payload.get("task") if isinstance(payload, dict) else None
        if not isinstance(task, dict):
            return None
        # Controller recovery needs the last bounded lifecycle event to
        # distinguish the known protected-terminal replay failure from an
        # unrelated human or infrastructure block. Keep it internal; this
        # metadata is never sent to the remote worker.
        enriched = dict(task)
        events = payload.get("events") if isinstance(payload, dict) else None
        if isinstance(events, list):
            enriched["_events"] = events
        if "children" in payload:
            enriched["_children"] = payload["children"]
        return enriched

    def task_status(self, board: str, task_id: str) -> str | None:
        task = self.task_details(board, task_id)
        if task is None:
            return None
        status = task.get("status") if isinstance(task, dict) else None
        if not isinstance(status, str) or not status.strip():
            raise RuntimeError("Kanban task lookup failed")
        return status.strip()

    def reconcile_dispatch_task(self, task_id: str, task: KanbanTask) -> None:
        result = self._runner.run(_kanban_reconcile_argv(task_id, task))
        if result.returncode != 0:
            raise RuntimeError(_kanban_create_error(result))

    def unblock_task(self, board: str, task_id: str, reason: str) -> None:
        result = self._runner.run(
            [
                "hermes", "kanban", "--board", board, "unblock", task_id,
                "--reason", reason,
            ]
        )
        if result.returncode != 0:
            raise RuntimeError(_kanban_create_error(result))

    def complete_superseded_task(
        self,
        board: str,
        task_id: str,
        summary: str,
        metadata: Mapping[str, object],
    ) -> None:
        result = self._runner.run(
            [
                "hermes", "kanban", "--board", board, "complete", task_id,
                "--summary", summary, "--metadata", json.dumps(metadata, sort_keys=True),
            ]
        )
        if result.returncode != 0:
            raise RuntimeError(_kanban_create_error(result))


def _kanban_create_error(result: KanbanCommandResult) -> str:
    first_line = result.stderr.splitlines()[0].strip() if result.stderr else ""
    first_line = _SECRET_ASSIGNMENT.sub(r"\1=[redacted]", first_line)
    detail = f": {first_line[:240]}" if first_line else ""
    return f"Kanban task creation failed (rc={result.returncode}){detail}"


def _kanban_create_argv(task: KanbanTask) -> list[str]:
    body = (
        f"{task.instructions}\n\n{task.evidence_heading}:\n"
        f"{json.dumps(task.evidence, sort_keys=True)}"
    )
    argv = [
        "hermes",
        "kanban",
        "--board",
        task.board,
        "create",
        task.title,
        "--body",
        body,
        "--assignee",
        task.assignee,
        "--workspace",
        f"dir:{task.repository_path}",
        "--idempotency-key",
        task.idempotency_key,
        "--max-retries",
        str(task.max_retries),
    ]
    if task.max_runtime_seconds is not None:
        argv.extend(["--max-runtime", str(task.max_runtime_seconds)])
    if task.model_override:
        argv.extend(["--model", task.model_override])
        if task.provider_override:
            argv.extend(["--provider", task.provider_override])
    if task.reasoning_effort:
        argv.extend(["--reasoning", task.reasoning_effort])
    if task.initial_status not in {"ready", "blocked", "running"}:
        raise ValueError("Kanban task initial status is invalid")
    if task.initial_status != "ready":
        argv.extend(["--initial-status", task.initial_status])
    argv.append("--json")
    return argv


def _is_legacy_intake_task(details: object, task: KanbanTask) -> bool:
    if not isinstance(details, dict):
        return False
    body = details.get("body")
    try:
        evidence = json.loads(body.rsplit("\n", 1)[-1]) if isinstance(body, str) else None
    except (TypeError, json.JSONDecodeError):
        evidence = None
    exact_receipt = (
        isinstance(evidence, dict)
        and evidence.get("repository") == task.evidence.get("repository")
        and evidence.get("pr_number") == task.evidence.get("pr_number")
        and evidence.get("feedback_kind") == task.evidence.get("feedback_kind")
        and evidence.get("feedback_id") == task.evidence.get("feedback_id")
        and evidence.get("expected_head_sha") == task.evidence.get("expected_head_sha")
    )
    if not (
        details.get("status") == "blocked"
        and isinstance(body, str)
        and exact_receipt
    ):
        return False
    if "This card is intake-only and starts blocked; an operator must validate" in body:
        return True
    events = details.get("_events")
    if not isinstance(events, list):
        return False
    for event in reversed(events):
        if not isinstance(event, dict) or event.get("kind") != "blocked":
            continue
        payload = event.get("payload")
        reason = payload.get("reason") if isinstance(payload, dict) else None
        return isinstance(reason, str) and (
            (
                "protected terminal route" in reason
                and "omitted_from_remote_replay" in reason
            )
            or "provider egress blocked: LLM egress blocked: base64_payload" in reason
        )
    return False


def _kanban_reconcile_argv(task_id: str, task: KanbanTask) -> list[str]:
    body = (
        f"{task.instructions}\n\n{task.evidence_heading}:\n"
        f"{json.dumps(task.evidence, sort_keys=True)}"
    )
    return [
        "hermes",
        "kanban",
        "--board",
        task.board,
        "reconcile-dispatch",
        task_id,
        "--idempotency-key",
        task.idempotency_key,
        "--head-sha",
        task.head_sha,
        "--body",
        body,
        "--assignee",
        task.assignee,
        "--workspace-path",
        str(task.repository_path),
        "--branch-name",
        task.branch,
        "--max-retries",
        str(task.max_retries),
        "--max-runtime",
        str(task.max_runtime_seconds or 0),
        "--json",
    ]


def setup_cli(_ctx: Any, parser: argparse.ArgumentParser) -> None:
    """Attach the plugin's command tree to the host-created parser."""

    subcommands = parser.add_subparsers(dest="github_pr_feedback_action", required=True)
    subcommands.add_parser("scan", help="Read and dispatch newly admitted feedback")
    subcommands.add_parser("status", help="Show durable receipt counts")
    subcommands.add_parser(
        "doctor", help="Check configuration readiness without scanning"
    )
    labels = subcommands.add_parser("label-scan", help="Reconcile configured advisory labels on owned open PRs")
    labels.add_argument("--repository", required=True)
    inspect = subcommands.add_parser(
        "inspect-pr", help="Read one configured PR identity through the shared GitHub gate"
    )
    inspect.add_argument("--repository", required=True)
    inspect.add_argument("--pr-number", required=True, type=int)
    inspect.add_argument("--feedback-id")
    inspect_ci = subcommands.add_parser("inspect-ci", help="Read a stored exact CI receipt without rerunning or handing off")
    inspect_ci.add_argument("--repository", required=True)
    inspect_ci.add_argument("--pr-number", required=True, type=int)
    inspect_ci.add_argument("--receipt-id", required=True)
    submit_review = subcommands.add_parser(
        "submit-review", help="Submit an exact-head review with an independent identity"
    )
    submit_review.add_argument("--repository", required=True)
    submit_review.add_argument("--pr-number", required=True, type=int)
    submit_review.add_argument("--head-sha", required=True)
    submit_review.add_argument("--event", required=True)
    submit_review.add_argument("--body", required=True)
    post_comment = subcommands.add_parser(
        "post-comment", help="Publish one literal exact-head factual PR comment"
    )
    post_comment.add_argument("--repository", required=True)
    post_comment.add_argument("--pr-number", required=True, type=int)
    post_comment.add_argument("--head-sha", required=True)
    post_comment.add_argument("--body", required=True)
    retry = subcommands.add_parser(
        "retry", help="Retry one failed, immutable feedback receipt"
    )
    retry.add_argument("--repository", required=True)
    retry.add_argument("--pr-number", required=True, type=int)
    retry.add_argument("--feedback-kind", required=True)
    retry.add_argument("--feedback-id", required=True)
    retry.add_argument("--head-sha", required=True)
    dispatch_repair = subcommands.add_parser(
        "dispatch-repair", help="Revalidate and dispatch one exact-head conflict repair"
    )
    dispatch_repair.add_argument("--repository", required=True)
    dispatch_repair.add_argument("--pr-number", required=True, type=int)
    dispatch_repair.add_argument("--head-sha", required=True)
    dispatch_feedback = subcommands.add_parser(
        "dispatch-feedback",
        help="Dispatch one exact, canonically revalidated feedback item",
    )
    dispatch_feedback.add_argument("--repository", required=True)
    dispatch_feedback.add_argument("--pr-number", required=True, type=int)
    dispatch_feedback.add_argument("--feedback-kind", required=True)
    dispatch_feedback.add_argument("--feedback-id", required=True)
    dispatch_feedback.add_argument("--head-sha", required=True)
    audit = subcommands.add_parser(
        "audit-pr", help="Run deterministic CI for one exact PR head"
    )
    audit.add_argument("--repository", required=True)
    audit.add_argument("--pr-number", required=True, type=int)
    audit.add_argument("--head-sha", required=True)
    audit.add_argument("--worktree", required=True, type=Path)
    audit.add_argument(
        "--fresh",
        action="store_true",
        help="Run new local CI even when an exact-head receipt is reusable",
    )
    subcommands.add_parser(
        "merge-scan", help="Evaluate and merge strictly eligible PR heads"
    )
    merge_status = subcommands.add_parser(
        "merge-status", help="Show bounded merge and deployment counts"
    )
    merge_status.add_argument("--details", action="store_true", help="Include the ten most recent verified merge receipts")
    merge_enable = subcommands.add_parser(
        "merge-enable", help="Enroll one configured PR for automatic merging"
    )
    merge_enable.add_argument("--repository", required=True)
    merge_enable.add_argument("--pr-number", required=True, type=int)
    merge_disable = subcommands.add_parser(
        "merge-disable", help="Remove one PR from automatic merge enrollment"
    )
    merge_disable.add_argument("--repository", required=True)
    merge_disable.add_argument("--pr-number", required=True, type=int)
    stack_create = subcommands.add_parser(
        "stack-create", help="Create an explicit dependent PR stack"
    )
    stack_create.add_argument("--repository", required=True)
    stack_create.add_argument("--stack-id", required=True)
    stack_create.add_argument("--base-branch", required=True)
    stack_create.add_argument(
        "--entry", action="append", required=True, metavar="BRANCH=BASE:TITLE"
    )
    stack_refresh = subcommands.add_parser(
        "stack-refresh", help="Merge the configured base and normally push stack descendants"
    )
    stack_refresh.add_argument("--repository", required=True)
    stack_refresh.add_argument("--stack-id", required=True)
    stack_refresh.add_argument("--repository-path", required=True, type=Path)
    stack_refresh.add_argument("--accept-rebased-heads", action="store_true")
    stack_merge = subcommands.add_parser(
        "stack-merge", help="Merge a verified stack in dependency order"
    )
    stack_merge.add_argument("--repository", required=True)
    stack_merge.add_argument("--stack-id", required=True)
    stack_merge.add_argument("--repository-path", required=True, type=Path)
    close_superseded = subcommands.add_parser(
        "close-superseded",
        help="Close an exact PR head already contained by its configured base",
    )
    close_superseded.add_argument("--repository", required=True)
    close_superseded.add_argument("--pr-number", required=True, type=int)
    close_superseded.add_argument("--head-sha", required=True)
    close_superseded.add_argument("--repository-path", required=True, type=Path)
    resolve_superseded_feedback = subcommands.add_parser(
        "resolve-superseded-feedback",
        help="Resolve one exact review thread whose exact fix is already on stable",
    )
    resolve_superseded_feedback.add_argument("--repository", required=True)
    resolve_superseded_feedback.add_argument("--pr-number", required=True, type=int)
    resolve_superseded_feedback.add_argument("--head-sha", required=True)
    resolve_superseded_feedback.add_argument("--comment-id", required=True)
    resolve_superseded_feedback.add_argument("--fix-sha", required=True)
    resolve_superseded_feedback.add_argument(
        "--repository-path", required=True, type=Path
    )
    resolve_superseded_feedback.add_argument("--test-evidence", required=True)
    retired = subcommands.add_parser("retire-feedback", help="Retire an exact feedback dispatch after PR closure")
    retired.add_argument("--repository", required=True)
    retired.add_argument("--pr-number", required=True, type=int)
    retired.add_argument("--feedback-kind", required=True)
    retired.add_argument("--feedback-id", required=True)
    retired.add_argument("--receipt-head-sha", required=True)
    retired.add_argument("--self-receipt", action="store_true", help="Retire only a verified non-actionable comment by the configured automation identity")
    completed = subcommands.add_parser(
        "complete-feedback",
        help="Acknowledge one dispatched feedback action after push and reply",
    )
    completed.add_argument("--repository", required=True)
    completed.add_argument("--pr-number", required=True, type=int)
    completed.add_argument("--feedback-kind", required=True)
    completed.add_argument("--feedback-id", required=True)
    completed.add_argument("--receipt-head-sha", required=True)
    completed.add_argument("--resolved-head-sha", required=True)
    maintenance = subcommands.add_parser(
        "complete-maintenance",
        help="Record one typed exact-head maintenance lane receipt",
    )
    maintenance.add_argument("--repository", required=True)
    maintenance.add_argument("--head-sha", required=True)
    maintenance.add_argument("--lane", required=True)
    maintenance.add_argument("--status", required=True, choices=("passed", "failed"))
    maintenance.add_argument("--summary", required=True)
    maintenance.add_argument(
        "--command-evidence-json",
        required=True,
        help="JSON list containing argv, cwd, return code, timeout, and output digests",
    )


def handle_cli_with_context(ctx: Any, args: argparse.Namespace) -> int:
    action = getattr(args, "github_pr_feedback_action", None)
    if action == "scan":
        return _scan(ctx)
    if action == "status":
        return _status()
    if action == "doctor":
        return _doctor(ctx)
    if action == "label-scan":
        return _label_scan(ctx, args)
    if action == "inspect-pr":
        return _inspect_pr(ctx, args)
    if action == "inspect-ci":
        from .cli_ci_receipt import inspect_ci
        return inspect_ci(ctx, args)
    if action == "submit-review":
        return _submit_review(ctx, args)
    if action == "post-comment":
        return _post_comment(ctx, args)
    if action == "retry":
        return _retry(ctx, args)
    if action == "dispatch-repair":
        from .cli_repair import dispatch_repair
        return dispatch_repair(ctx, args)
    if action == "dispatch-feedback":
        return _dispatch_feedback(ctx, args)
    if action == "audit-pr":
        return _audit_pr(ctx, args)
    if action == "merge-scan":
        return _merge_scan(ctx)
    if action == "merge-status":
        return _merge_status(details=bool(getattr(args, "details", False)))
    if action == "merge-enable":
        return _merge_enable(ctx, args)
    if action == "merge-disable":
        return _merge_disable(ctx, args)
    if action == "stack-create":
        return _stack_create(ctx, args)
    if action == "stack-refresh":
        return _stack_refresh(ctx, args)
    if action == "stack-merge":
        return _stack_merge(ctx, args)
    if action == "close-superseded":
        return _close_superseded(ctx, args)
    if action == "resolve-superseded-feedback":
        return _resolve_superseded_feedback(ctx, args)
    if action == "retire-feedback":
        from .feedback_retirement import run_retirement
        return run_retirement(ctx, args)
    if action == "complete-feedback":
        return _complete_feedback(ctx, args)
    if action == "complete-maintenance":
        return _complete_maintenance(ctx, args)
    return 2


def _complete_maintenance(ctx: Any, args: argparse.Namespace) -> int:
    try:
        policy = _load_policy_from_context(ctx)
        maintenance = policy.release_policy_for(args.repository)
        if maintenance is None:
            raise ValueError("release maintenance is disabled")
        allowed_lanes = {lane.name for lane in maintenance.lanes} | {FINAL_LANE}
        if (
            args.repository != maintenance.repository
            or args.lane not in allowed_lanes
            or not _FULL_SHA.fullmatch(args.head_sha)
        ):
            raise ValueError("maintenance receipt identity is not configured")
        try:
            command_evidence = parse_maintenance_command_evidence(
                json.loads(args.command_evidence_json)
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("maintenance command evidence is invalid") from error
        ledger = FeedbackLedger.for_current_profile()
        try:
            ledger.record_maintenance_receipt(
                repository=args.repository,
                head_sha=args.head_sha.casefold(),
                lane=args.lane,
                status=args.status,
                summary=args.summary,
                completed_at=datetime.now(UTC),
                command_evidence=command_evidence,
            )
        finally:
            ledger.close()
    except (LedgerStateError, ValueError):
        print(json.dumps({"status": "rejected"}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "status": "recorded",
                "repository": args.repository,
                "head_sha": args.head_sha.casefold(),
                "lane": args.lane,
                "result": args.status,
            },
            sort_keys=True,
        )
    )
    return 0


def cli_bindings(ctx: Any) -> tuple[Any, Any]:
    """Bind host-supplied settings access without reading global YAML."""

    return partial(setup_cli, ctx), partial(handle_cli_with_context, ctx)


def _github_client(policy: PluginPolicy) -> GitHubClient:
    """Return the sole authenticated client for governed Hermes GitHub activity."""

    # Lightweight injected test doubles intentionally do not implement the
    # production factory. Production always uses the real class below.
    factory = getattr(GitHubClient, "for_automation_identity", None)
    if not callable(factory):
        return GitHubClient()
    if not isinstance(policy, PluginPolicy):
        raise GitHubClientError("invalid GitHub automation policy", code="invalid_policy")
    settings = policy.github_identity
    if settings is None:
        raise GitHubClientError(
            "Hermes GitHub automation identity is not configured",
            code="automation_identity_not_configured",
        )
    actions_settings = policy.github_actions_permissions_identity
    return factory(
        expected_login=settings.expected_login,
        token_env=settings.token_env,
        actions_permissions_expected_login=(
            actions_settings.expected_login if actions_settings is not None else None
        ),
        actions_permissions_gh_config_dir=(
            actions_settings.gh_config_dir if actions_settings is not None else None
        ),
        actions_permissions_repositories=(
            actions_settings.repositories if actions_settings is not None else frozenset()
        ),
    )


def _submit_review(ctx: Any, args: argparse.Namespace) -> int:
    """Submit a review only after proving an independent configured identity."""

    try:
        policy = _load_policy_from_context(ctx)
        if not policy.enabled or args.repository not in policy.targets:
            raise ValueError("repository is not a configured target")
        if not isinstance(args.head_sha, str) or not _FULL_SHA.fullmatch(args.head_sha):
            raise ValueError("head_sha must be a full hexadecimal SHA")
        event = str(args.event or "").strip().upper().replace("-", "_")
        if event not in _REVIEW_EVENTS:
            raise ValueError("review event must be APPROVE, REQUEST_CHANGES, or COMMENT")
        settings = policy.github_identity
        if settings is None:
            print(
                json.dumps(
                    {
                        "status": "review_identity_unavailable",
                        "reason": "independent_reviewer_not_configured",
                    },
                    sort_keys=True,
                )
            )
            return 1
        github = _github_client(policy)
        viewer_login = github.viewer_login()
        pull_request = github.get_pull_request(args.repository, args.pr_number)
        if viewer_login.casefold() == pull_request.author_login.casefold():
            print(
                json.dumps(
                    {
                        "status": "review_identity_unavailable",
                        "reason": "viewer_is_pr_author",
                    },
                    sort_keys=True,
                )
            )
            return 1
        if viewer_login.casefold() != settings.expected_login:
            print(
                json.dumps(
                    {
                        "status": "review_identity_unavailable",
                        "reason": "reviewer_identity_mismatch",
                    },
                    sort_keys=True,
                )
            )
            return 1
        expected_head_sha = args.head_sha.casefold()
        if pull_request.head_sha != expected_head_sha:
            print(
                json.dumps(
                    {
                        "status": "stale_head",
                        "repository": args.repository,
                        "pr_number": args.pr_number,
                        "expected_head_sha": expected_head_sha,
                        "observed_head_sha": pull_request.head_sha,
                    },
                    sort_keys=True,
                )
            )
            return 1
        if str(pull_request.state or "").strip().upper() != "OPEN":
            print(
                json.dumps(
                    {
                        "status": "not_reviewable",
                        "reason": "pull_request_not_open",
                        "repository": args.repository,
                        "pr_number": args.pr_number,
                        "head_sha": pull_request.head_sha,
                    },
                    sort_keys=True,
                )
            )
            return 1
        github.submit_pull_request_review(
            args.repository,
            args.pr_number,
            event=event,
            body=args.body,
        )
    except (GitHubClientError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {"status": "review_unavailable", "reason": str(error)},
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "status": "submitted",
                "repository": args.repository,
                "pr_number": args.pr_number,
                "head_sha": pull_request.head_sha,
                "event": event,
                "reviewer": viewer_login,
            },
            sort_keys=True,
        )
    )
    return 0


def _post_comment(ctx: Any, args: argparse.Namespace) -> int:
    """Publish one exact-head factual reply through the governed bot client."""

    try:
        policy = _load_policy_from_context(ctx)
        if not policy.enabled or args.repository not in policy.targets:
            raise ValueError("repository is not a configured target")
        if not isinstance(args.head_sha, str) or not _FULL_SHA.fullmatch(args.head_sha):
            raise ValueError("head_sha must be a full hexadecimal SHA")
        settings = policy.github_identity
        if settings is None:
            raise GitHubClientError(
                "Hermes GitHub automation identity is not configured",
                code="automation_identity_not_configured",
            )
        github = _github_client(policy)
        viewer_login = github.viewer_login()
        if viewer_login.casefold() != settings.expected_login.casefold():
            raise GitHubClientError(
                "GitHub automation identity does not match policy",
                code="automation_identity_mismatch",
            )
        pull_request = github.get_pull_request(args.repository, args.pr_number)
        if pull_request.head_sha != args.head_sha.casefold():
            print(
                json.dumps(
                    {
                        "status": "stale_head",
                        "repository": args.repository,
                        "pr_number": args.pr_number,
                        "expected_head_sha": args.head_sha.casefold(),
                        "observed_head_sha": pull_request.head_sha,
                    },
                    sort_keys=True,
                )
            )
            return 1
        if str(pull_request.state or "").strip().upper() != "OPEN":
            raise ValueError("pull request is not open")
        github.post_issue_comment(args.repository, args.pr_number, args.body)
    except (GitHubClientError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {"status": "comment_unavailable", "reason": str(error)},
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "status": "posted",
                "repository": args.repository,
                "pr_number": args.pr_number,
                "head_sha": pull_request.head_sha,
                "poster": viewer_login,
            },
            sort_keys=True,
        )
    )
    return 0


def _scan(ctx: Any) -> int:
    try:
        policy = _load_policy_from_context(ctx)
    except ValueError:
        print(json.dumps({"status": "invalid_configuration"}, sort_keys=True))
        return 1
    if not policy.enabled:
        print(json.dumps({"created": 0, "skipped": {}, "status": "ok"}, sort_keys=True))
        return 0
    repository_filter = os.environ.get("HERMES_PR_FEEDBACK_REPOSITORY", "").strip()
    if repository_filter and repository_filter not in policy.targets:
        print(json.dumps({"status": "invalid_configuration", "repository": repository_filter}, sort_keys=True))
        return 1
    with _exclusive_scan_lock() as acquired:
        if not acquired:
            print(json.dumps({"status": "scan_in_progress"}, sort_keys=True))
            return 0
        ledger = FeedbackLedger.for_current_profile()
        merge_payload: dict[str, object] | None = None
        repair_payload: dict[str, object] | None = None
        maintenance_payload: object | None = None
        label_payload: dict[str, object] | None = None
        try:
            try:
                PooledLocalGitRepository(
                    ledger, ledger.path.parent / "worktree-pool"
                ).reconcile_leases(KanbanSubprocessClient())
            except Exception:  # noqa: BLE001 - proactive release is an optimization,
                # never allowed to block the scan it runs ahead of; a slot left
                # leased simply falls back to its lease timeout.
                pass
            controller = _controller(policy, ledger)
            result = controller.scan(apply_labels=False, repository_filter=repository_filter or None)
            # Conflicts must be repairable while CI is pending; otherwise the
            # backlog can never clear. Keep this pass bounded and conflict-only.
            # Merge/release retain their own independent exact-head gates.
            required_ci_backlog = result.required_local_ci_backlog > 0
            if policy.repair_steward is not None:
                repair = RepairController(
                    policy,
                    ledger,
                    _github_client(policy),
                    KanbanSubprocessClient(),
                    control_home=get_default_hermes_root(),
                ).scan(conflicts_only=required_ci_backlog, repository_filter=repository_filter or None)
                repair_payload = _scan_payload(repair)
            if policy.merge_policies():
                merge_payload = _run_merge_scan(policy, ledger)
            release_policies = (
                policy.release_policies()
                if hasattr(policy, "release_policies")
                else (
                    (policy.release_maintenance,)
                    if getattr(policy, "release_maintenance", None) is not None
                    else ()
                )
            )
            if release_policies and not required_ci_backlog:
                maintenance_results = [
                    _run_release_maintenance_scan(
                        policy, ledger, maintenance=maintenance
                    )
                    for maintenance in release_policies
                ]
                maintenance_payload = (
                    maintenance_results[0]
                    if len(maintenance_results) == 1
                    else {
                        maintenance.repository: result
                        for maintenance, result in zip(
                            release_policies, maintenance_results
                        )
                    }
                )
            label_payload = controller.apply_agent_labels()
        finally:
            ledger.close()
    payload = _scan_payload(result)
    if result.required_local_ci_backlog > 0:
        payload["deferred"] = ["non_conflict_repair", "release_maintenance"]
    if merge_payload is not None:
        payload["merge"] = merge_payload
    if repair_payload is not None:
        payload["repair"] = repair_payload
    if maintenance_payload is not None:
        payload["release_maintenance"] = maintenance_payload
    if label_payload is not None and (
        label_payload["updated"] or label_payload["skipped"]
    ):
        payload["labels"] = label_payload
    print(json.dumps(payload, sort_keys=True))
    return (
        1
        if (
            result.degraded
            or (repair_payload or {}).get("status") == "degraded"
            or (merge_payload or {}).get("status") == "degraded"
            or (
                isinstance(maintenance_payload, dict)
                and (
                    maintenance_payload.get("status") == "degraded"
                    or any(
                        isinstance(item, dict) and item.get("status") == "degraded"
                        for item in maintenance_payload.values()
                    )
                )
            )
        )
        else 0
    )


def _run_release_maintenance_scan(
    policy: PluginPolicy,
    ledger: FeedbackLedger,
    *,
    maintenance: ReleaseMaintenancePolicy | None = None,
    github: MaintenanceGitHub | None = None,
    kanban: KanbanSubprocessClient | None = None,
    workspaces: MaintenanceWorkspaces | None = None,
    now: Callable[[], datetime] | None = None,
    control_home: Path | None = None,
) -> dict[str, object]:
    maintenance = maintenance or policy.release_maintenance
    if maintenance is None:
        return {
            "status": "disabled",
            "head_sha": None,
            "tasks_created": 0,
            "blockers": [],
        }
    target = policy.targets[maintenance.repository]
    result = ReleaseMaintenanceController(
        maintenance,
        target,
        ledger,
        github or _github_client(policy),
        kanban or KanbanSubprocessClient(),
        workspaces or LocalGitRepository(ledger.path.parent / "maintenance-worktrees"),
        now=now,
        control_home=control_home or get_default_hermes_root(),
        board=policy.board or "maintenance",
    ).scan()
    return {
        "status": result.status,
        "head_sha": result.head_sha,
        "tasks_created": result.tasks_created,
        "blockers": list(result.blockers),
    }


@contextmanager
def _exclusive_scan_lock(control_home: Path | None = None) -> Iterator[bool]:
    """Allow only one full feedback scan per Hermes control home."""

    lock_root = (control_home or get_default_hermes_root()) / "github-pr-feedback"
    lock_root.mkdir(parents=True, exist_ok=True)
    handle = (lock_root / "scan.lock").open("a+")
    acquired = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            pass
        yield acquired
    finally:
        if acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _retry(ctx: Any, args: argparse.Namespace) -> int:
    try:
        policy = _load_policy_from_context(ctx)
        receipt = FeedbackReceipt(
            args.repository,
            args.pr_number,
            args.feedback_kind,
            args.feedback_id,
            args.head_sha,
        )
    except ValueError:
        print(json.dumps({"status": "invalid_configuration"}, sort_keys=True))
        return 1
    ledger = FeedbackLedger.for_current_profile()
    try:
        if receipt.feedback_kind == "pr_repair":
            result = RepairController(
                policy, ledger, _github_client(policy), KanbanSubprocessClient(),
                control_home=get_default_hermes_root(),
            ).scan(retry_receipt=receipt)
        else:
            result = _controller(policy, ledger).retry_failed(receipt)
    finally:
        ledger.close()
    print(json.dumps(_scan_payload(result), sort_keys=True))
    return 1 if result.degraded else 0


def _dispatch_feedback(ctx: Any, args: argparse.Namespace) -> int:
    try:
        policy = _load_policy_from_context(ctx)
        if (
            not isinstance(args.head_sha, str)
            or _FULL_SHA.fullmatch(args.head_sha) is None
        ):
            raise ValueError("head_sha must be a full hexadecimal SHA")
        receipt = FeedbackReceipt(
            args.repository,
            args.pr_number,
            args.feedback_kind,
            args.feedback_id,
            args.head_sha.casefold(),
        )
    except ValueError:
        print(json.dumps({"status": "invalid_configuration"}, sort_keys=True))
        return 1
    ledger = FeedbackLedger.for_current_profile()
    try:
        result = _controller(policy, ledger).dispatch_feedback(receipt)
        payload = _scan_payload(result)
        if result.skipped.get("duplicate"):
            payload["pending_dispatches"] = [
                {"task_id": binding.task_id,
                 "feedback_kind": binding.receipt.feedback_kind,
                 "feedback_id": binding.receipt.feedback_id,
                 "head_sha": binding.receipt.head_sha}
                for binding in ledger.pending_task_bindings_for_head(receipt)
            ]
    finally:
        ledger.close()
    print(json.dumps(payload, sort_keys=True))
    return 1 if result.degraded else 0


def _complete_feedback(ctx: Any, args: argparse.Namespace) -> int:
    try:
        policy = _load_policy_from_context(ctx)
        receipt = FeedbackReceipt(
            args.repository,
            args.pr_number,
            args.feedback_kind,
            args.feedback_id,
            args.receipt_head_sha,
        )
        github = _github_client(policy)
        current = github.get_pull_request(args.repository, args.pr_number)
        admission = policy.admit_pull_request(current)
        if (
            not admission.admitted
            or current.head_sha.casefold() != str(args.resolved_head_sha).casefold()
        ):
            raise ValueError("resolved head is not canonical")
    except (ValueError, GitHubClientError):
        print(
            json.dumps({"status": "invalid_or_raced_feedback_action"}, sort_keys=True)
        )
        return 1
    if receipt.feedback_kind in _MARKER_REQUIRED_FEEDBACK_KINDS and _factual_reply_is_missing(
        github, receipt, resolved_head_sha=str(args.resolved_head_sha)
    ):
        print(json.dumps({"status": "factual_reply_missing"}, sort_keys=True))
        return 1
    ledger = FeedbackLedger.for_current_profile()
    try:
        ledger.begin_feedback_action(
            receipt,
            resolved_head_sha=args.resolved_head_sha,
            actioned_at=datetime.now(UTC),
        )
        review_thread_resolved = False
        if receipt.feedback_kind == "review_comment":
            review_thread_resolved = github.resolve_review_thread_for_comment(
                receipt.repository,
                receipt.pr_number,
                receipt.feedback_id,
                expected_head_sha=args.resolved_head_sha,
            )
        ledger.mark_feedback_actioned(
            receipt,
            resolved_head_sha=args.resolved_head_sha,
            actioned_at=datetime.now(UTC),
        )
    except GitHubClientError:
        print(
            json.dumps(
                {"status": "feedback_action_reconciliation_pending"}, sort_keys=True
            )
        )
        return_code = 1
    except (ValueError, LedgerStateError):
        print(json.dumps({"status": "feedback_action_not_recorded"}, sort_keys=True))
        return_code = 1
    else:
        codex_retrigger_status = "not_applicable"
        if receipt.feedback_kind in _MARKER_REQUIRED_FEEDBACK_KINDS:
            codex_retrigger_status = _retrigger_codex_review(
                github, receipt.repository, receipt.pr_number, str(args.resolved_head_sha)
            )
        local_ci_status = _controller(policy, ledger).dispatch_local_ci_after_feedback(
            current
        )
        print(
            json.dumps(
                {
                    "status": "completed",
                    "repository": receipt.repository,
                    "pr_number": receipt.pr_number,
                    "feedback_kind": receipt.feedback_kind,
                    "feedback_id": receipt.feedback_id,
                    "resolved_head_sha": str(args.resolved_head_sha).casefold(),
                    "review_thread_resolved": review_thread_resolved,
                    "local_ci_status": local_ci_status,
                    "codex_retrigger_status": codex_retrigger_status,
                },
                sort_keys=True,
            )
        )
        return_code = 0
    finally:
        ledger.close()
    return return_code


def _controller(policy: PluginPolicy, ledger: FeedbackLedger) -> ScanController:
    return ScanController(
        policy,
        ledger,
        _github_client(policy),
        KanbanSubprocessClient(),
        control_home=get_default_hermes_root(),
    )


def _scan_payload(result) -> dict[str, object]:
    payload = {
        "status": "degraded" if result.degraded else "ok",
        "created": result.created,
        "skipped": result.skipped,
    }
    backlog = getattr(result, "required_local_ci_backlog", 0)
    if backlog > 0:
        payload["required_local_ci_backlog"] = backlog
    catalogue_deferred = getattr(result, "local_ci_catalogue_deferred", 0)
    if catalogue_deferred > 0:
        payload["local_ci_catalogue_deferred"] = catalogue_deferred
    return payload


def _status() -> int:
    ledger = FeedbackLedger.for_current_profile()
    try:
        counts = ledger.status_counts()
    finally:
        ledger.close()
    print(json.dumps(counts, sort_keys=True))
    return 0


def _reusable_ci_receipt(
    ledger: object,
    identity: CIAuditIdentity,
    worktree: Path,
    *,
    allow_reuse: bool = True,
) -> CIAuditReceipt | None:
    """Reuse immutable exact-head evidence instead of repeating an expensive lane."""

    if not allow_reuse:
        return None
    reader = getattr(ledger, "latest_ci_receipt_for_head", None)
    if not callable(reader):
        return None
    receipt = reader(identity.repository, identity.pr_number, identity.head_sha)
    if receipt is None:
        return None
    if not isinstance(receipt, CIAuditReceipt):
        raise LedgerStateError("stored CI receipt identity is inconsistent")
    stored = receipt.identity
    if (
        stored.repository != identity.repository
        or stored.pr_number != identity.pr_number
        or stored.head_sha != identity.head_sha
    ):
        raise LedgerStateError("stored CI receipt identity is inconsistent")
    if stored.base_sha != identity.base_sha:
        # A PR can retain its head while the canonical base advances. That
        # makes the old receipt non-reusable, not corrupt; rerun all
        # base-relative lanes against the new exact base.
        return None
    manifest_path = ci_manifest_path(worktree)
    if not manifest_path.is_file():
        return None
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return receipt if receipt.manifest_digest == manifest_digest else None


class _ThreadOwnedCIRunner:
    """Close a per-worker ledger after one grouped exact-head audit."""

    def __init__(self, runner: LocalCIRunner, ledger: FeedbackLedger) -> None:
        self._runner = runner
        self._ledger = ledger

    def run(self, identity: CIAuditIdentity, worktree: Path) -> CIAuditReceipt:
        try:
            return self._runner.run(identity, worktree)
        finally:
            self._ledger.close()


def _run_grouped_exact_head_audit(
    github: GitHubClient,
    ledger: FeedbackLedger,
    identity: CIAuditIdentity,
    worktree: Path,
    *,
    force_fresh: bool = False,
    actions_enabled_hint: bool | None = None,
    required_local_ci: bool = False,
) -> CIAuditReceipt:
    """Run one immutable audit through the bounded grouped-coordination boundary."""

    receipt = _reusable_ci_receipt(
        ledger, identity, worktree, allow_reuse=not force_fresh
    )
    if receipt is not None:
        return receipt
    manifest_path = ci_manifest_path(worktree)
    if not manifest_path.is_file():
        raise CIValidationError("CI lane manifest is unavailable")
    job = CIAuditJob(
        identity=identity,
        worktree=worktree,
        failure_lanes=_required_lanes(manifest_path.read_bytes()),
    )
    def runner_factory() -> _ThreadOwnedCIRunner:
        # The coordinator invokes this factory inside its worker thread. Each
        # SQLite connection must therefore be opened and closed in that same
        # thread; sharing one grouped connection would fail immediately with
        # sqlite3.ProgrammingError and lose a typed failed receipt.
        worker_ledger = FeedbackLedger.for_current_profile()
        runner = (
            LocalCIRunner(github, worker_ledger, required_local_ci=True)
            if required_local_ci else
            LocalCIRunner(github, worker_ledger)
            if actions_enabled_hint is None
            else LocalCIRunner(
                github,
                worker_ledger,
                actions_enabled_hint=actions_enabled_hint,
            )
        )
        return _ThreadOwnedCIRunner(
            runner,
            worker_ledger,
        )

    outcome = GroupedCICoordinator(
        runner_factory,
        max_parallel=4,
    ).run((job,))[0]
    if outcome.error is not None or outcome.receipt is None:
        reason = outcome.error or "no receipt returned"
        raise CIValidationError(f"grouped exact-head CI audit was unavailable: {reason}")
    return outcome.receipt


def _ci_receipt_payload(receipt: CIAuditReceipt) -> dict[str, object]:
    """Return the durable receipt fields rendered before any handoff work."""

    from agent.redact import redact_sensitive_text

    return {
        "status": receipt.status,
        "receipt_id": receipt.receipt_id,
        "repository": receipt.identity.repository,
        "pr_number": receipt.identity.pr_number,
        "head_sha": receipt.identity.head_sha,
        "base_sha": receipt.identity.base_sha,
        "failure_reason": redact_sensitive_text(receipt.failure_reason or "", force=True)[:1500],
        "failed_commands": [
            {"argv": list(c.argv), "returncode": c.returncode, "timed_out": c.timed_out,
             "classification": c.classification}
            for c in receipt.commands if c.returncode != 0 or c.timed_out or c.classification != "passed"
        ],
        "manifest_digest": receipt.manifest_digest,
        "command_count": len(receipt.commands),
        "ci_mode": receipt.ci_mode,
        "handoff_status": "pending",
    }


def _audit_pr(ctx: Any, args: argparse.Namespace) -> int:
    from .github_client import MergeStateStillComputingError

    handoff_blocked = False
    handoff_blockers: list[str] = []
    merge_handoff: dict[str, object] | None = None
    repair_status: str | None = None
    try:
        policy = _load_policy_from_context(ctx)
        if policy.local_ci_audit is None:
            raise ValueError("local CI audit is disabled")
        target = policy.targets.get(args.repository)
        worktree = Path(args.worktree).resolve()
        if target is None or not worktree.is_dir():
            raise ValueError("audit target is not configured")
        github = _github_client(policy)
        state = github.get_merge_state(args.repository, args.pr_number)
        if state.head_sha != str(args.head_sha).casefold():
            raise CIValidationError("canonical PR head changed")
        identity = CIAuditIdentity(
            args.repository, args.pr_number, state.base_sha, state.head_sha
        )
        actions_enabled_hint = (
            github.actions_enabled(args.repository, refresh=True)
            if policy.uses_budget_exhausted_local_ci(args.repository)
            and not policy.local_ci_audit.required_for_open_prs
            else None
        )
    except MergeStateStillComputingError:
        print(json.dumps({"status": "audit_deferred", "reason": "mergeable_state_still_computing",
                          "retryable": True, "retry_after_seconds": 60}), flush=True)
        return 1
    except (ValueError, CIValidationError, GitHubClientError):
        print(json.dumps({"status": "invalid_or_raced_audit_identity"}, sort_keys=True))
        return 1
    ledger = FeedbackLedger.for_current_profile()
    try:
        from .ci_admission import audit_execution_blocker

        blocker = audit_execution_blocker(ledger, state)
        if blocker is not None:
            print(json.dumps({
                "status": "audit_deferred", "reason": blocker, "retryable": True,
                "retry_after_seconds": 60,
                "repository": identity.repository, "pr_number": identity.pr_number,
                "head_sha": identity.head_sha, "base_sha": identity.base_sha,
                "next_action": "Wait for conflict repair and its acknowledgement, then retry the exact-head audit. Do not complete the task from this result.",
            }, sort_keys=True), flush=True)
            return 1
        receipt = _run_grouped_exact_head_audit(
            github,
            ledger,
            identity,
            worktree,
            force_fresh=bool(getattr(args, "fresh", False)),
            actions_enabled_hint=actions_enabled_hint,
            required_local_ci=policy.local_ci_audit.required_for_open_prs,
        )
    except (CIValidationError, GitHubClientError, LedgerStateError) as error:
        print(
            json.dumps(
                {"status": "audit_unavailable", "reason": str(error)},
                sort_keys=True,
            )
        )
        return_code = 1
    else:
        # Receipt persistence is the audit boundary. Render it before the
        # GitHub comment, repair dispatch, merge handoff, or task completion;
        # those are separate integrations and may fail independently.
        print(json.dumps(_ci_receipt_payload(receipt), sort_keys=True), flush=True)
        owns_task = owns_current_audit_task(ledger, receipt.identity)
        try:
            final_state = github.get_pull_request(args.repository, args.pr_number)
            if final_state.head_sha != receipt.identity.head_sha:
                raise CIValidationError("canonical PR head changed after audit")
            if policy.local_ci_audit.audit_only:
                print(
                    json.dumps(
                        {
                            "status": "audit_only_complete",
                            "receipt_id": receipt.receipt_id,
                            "ci_mode": receipt.ci_mode,
                            "posted": False,
                            "merge_handoff": False,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                return_code = 0 if receipt.status == "passed" else 1
            elif policy.local_ci_audit.post_results:
                feedback = github.list_feedback(
                    receipt.identity.repository, receipt.identity.pr_number
                )
                if not any(receipt.receipt_id in item.body for item in feedback):
                    github.post_issue_comment(
                        receipt.identity.repository,
                        receipt.identity.pr_number,
                        _ci_audit_comment(receipt),
                    )
            if policy.local_ci_audit.audit_only:
                pass
            elif receipt.status == "failed":
                repair_status = _controller(policy, ledger).dispatch_ci_failure(receipt)
                if repair_status not in {"scheduled", "duplicate"}:
                    raise RuntimeError(f"typed CI repair handoff failed: {repair_status}")
            else:
                merge_handoff = _run_single_pr_merge_handoff(
                    policy,
                    ledger,
                    receipt.identity.pr_number,
                    repository=receipt.identity.repository,
                    github=github,
                )
                handoff_status = str(merge_handoff.get("status", ""))
                if handoff_status == "blocked":
                    raw_blockers = merge_handoff.get("blockers", [])
                    if isinstance(raw_blockers, list):
                        handoff_blockers = [str(blocker) for blocker in raw_blockers]
                    next_step = _dispatch_merge_next_step(
                        policy,
                        ledger,
                        github,
                        KanbanSubprocessClient(),
                        receipt.identity.repository,
                        receipt.identity.pr_number,
                        handoff_blockers,
                    )
                    successor_scheduled = bool(
                        isinstance(next_step, dict)
                        and next_step.get("status") in {"scheduled", "duplicate", "already_scheduled"}
                    )
                    if owns_task and not successor_scheduled:
                        _block_current_ci_task(receipt, handoff_blockers)
                    elif owns_task and successor_scheduled:
                        _complete_current_ci_task(receipt)
                    handoff_blocked = True
                    if next_step is not None:
                        merge_handoff["next_step"] = next_step
                elif handoff_status != "merged":
                    raise RuntimeError(
                        "merge handoff did not produce a durable successor: "
                        f"{handoff_status}"
                    )
            if not handoff_blocked and owns_task:
                _complete_current_ci_task(receipt)
        except (CIValidationError, GitHubClientError, RuntimeError) as error:
            if not handoff_blocked and owns_task:
                try:
                    _block_current_ci_task(
                        receipt, [f"transient_handoff_failure: {repair_status or str(error)}"], kind="transient"
                    )
                except RuntimeError:
                    pass
            handoff_reason = repair_status or str(error) or "transient_handoff_failure"
            retryable_payload: dict[str, object] = {
                "status": "audit_handoff_retryable",
                "receipt_id": receipt.receipt_id,
                "retryable": True,
                "handoff_reason": handoff_reason,
            }
            if repair_status is not None:
                retryable_payload["repair_status"] = repair_status
            print(
                json.dumps(
                    retryable_payload,
                    sort_keys=True,
                ),
                flush=True,
            )
            return_code = 1
        else:
            return_code = 1 if handoff_blocked else (0 if receipt.status == "passed" else 1)
        if handoff_blocked:
            print(
                json.dumps(
                    {
                        "status": "audit_handoff_blocked",
                        "receipt_id": receipt.receipt_id,
                        "blockers": handoff_blockers,
                        "retryable": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if merge_handoff is not None and not handoff_blocked:
            print(
                json.dumps(
                    {
                        "status": "merge_handoff_recorded",
                        "receipt_id": receipt.receipt_id,
                        "merge_handoff": merge_handoff,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        ledger.close()
    return return_code


def _ci_audit_comment(receipt: CIAuditReceipt) -> str:
    commands = "; ".join(
        f"`{' '.join(command.argv)}` rc={command.returncode} "
        f"({command.classification}, {command.duration_ms / 1000:.2f}s)"
        for command in receipt.commands
    )
    attribution = (
        f"{hermes_attribution_line('pr-local-ci-auditor', action='CI audit')}\n\n"
        if pr_repair_attribution_required(receipt.identity.repository)
        else ""
    )
    body = (
        attribution
        + f"Addressed local CI audit for exact head `{receipt.identity.head_sha}` "
        f"(base `{receipt.identity.base_sha}`). Commands: {commands}. "
        f"Authoritative receipt: `{receipt.receipt_id}`. "
        + (
            "All required lanes passed."
            if receipt.status == "passed"
            else "The failed receipt remains merge-blocking; later fail-fast lanes may be absent."
        )
        + "\n\n"
        + f"<!-- pr-ci-receipt:v1 status={receipt.status} "
        f"id={receipt.receipt_id} head={receipt.identity.head_sha} -->\n"
        + f"<!-- pr-ci-receipt:v2 status={receipt.status} id={receipt.receipt_id} "
        f"head={receipt.identity.head_sha} base={receipt.identity.base_sha} "
        f"manifest={receipt.manifest_digest} completed={receipt.completed_at.isoformat()} "
        f"ci_mode={receipt.ci_mode} -->"
    )
    if len(body) > 4000:
        raise RuntimeError("CI audit comment exceeds the bounded GitHub payload")
    return body


def _complete_current_ci_task(receipt: CIAuditReceipt) -> None:
    task_id = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    if not task_id:
        return
    board = os.environ.get("HERMES_KANBAN_BOARD", "").strip()
    argv = [sys.executable, "-m", "hermes_cli.main", "kanban"]
    if board:
        argv.extend(["--board", board])
    argv.extend(
        [
            "complete",
            task_id,
            "--result",
            f"Exact-head local CI receipt {receipt.receipt_id}: {receipt.status}.",
        ]
    )
    try:
        completed = subprocess.run(
            argv,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except OSError as exc:
        raise RuntimeError("Hermes runtime unavailable for Kanban audit completion") from exc
    if completed.returncode != 0:
        raise RuntimeError("Kanban audit completion failed")


_MERGE_REPAIR_BLOCKERS = frozenset({
    "pull_request_conflicted",
    "merge_state_not_clean",
    "base_head_changed",
    "head_branch_not_allowed",
})
_LOCAL_CI_RETRY_BLOCKERS = frozenset({
    "ci_receipt_missing",
    "ci_identity_mismatch",
    "ci_manifest_mismatch",
    "ci_receipt_stale",
    "github_actions_disabled",
    "github_actions_budget_exhausted",
    "github_checks_not_green",
})


def _dispatch_merge_next_step(
    policy: PluginPolicy,
    ledger: FeedbackLedger,
    github: GitHubClient,
    kanban: KanbanSubprocessClient,
    repository: str,
    pr_number: int,
    blockers: list[str],
) -> dict[str, object] | None:
    """Turn a deterministic merge blocker into its next owned Kanban step.

    The merge gate remains fail-closed. This helper only schedules the
    specialist that can produce the missing evidence or repair; it never
    relaxes a gate and never performs a GitHub write.
    """

    blocker_set = {item.strip().casefold() for item in blockers}
    current = github.get_pull_request(repository, pr_number)
    if blocker_set & _MERGE_REPAIR_BLOCKERS:
        repair_policy = policy.repair_steward
        if repair_policy is None:
            return {"status": "disabled", "reason": "repair_steward_disabled"}
        from .repair_controller import RepairController

        result = RepairController(
            policy,
            ledger,
            github,
            kanban,
            control_home=get_default_hermes_root(),
        ).scan(scoped_target=(repository, pr_number, current.head_sha))
        status = (
            "scheduled"
            if result.created
            else "duplicate"
            if not result.skipped
            else "not_scheduled"
        )
        return {
            "status": status,
            "owner": repair_policy.assignee,
            "created": result.created,
            "skipped": dict(result.skipped),
            "blockers": sorted(blocker_set & _MERGE_REPAIR_BLOCKERS),
        }
    if blocker_set & _LOCAL_CI_RETRY_BLOCKERS:
        audit_policy = policy.local_ci_audit
        if audit_policy is None:
            return {"status": "disabled", "reason": "local_ci_audit_disabled"}
        result = ScanController(
            policy,
            ledger,
            github,
            kanban,
            PooledLocalGitRepository(ledger, ledger.path.parent / "worktree-pool"),
        ).dispatch_local_ci_after_feedback(current)
        return {
            "status": result,
            "owner": audit_policy.assignee,
            "blockers": sorted(blocker_set & _LOCAL_CI_RETRY_BLOCKERS),
        }
    return None


def _block_current_ci_task(
    receipt: CIAuditReceipt,
    blockers: list[str],
    *,
    kind: str | None = None,
) -> None:
    """Persist a blocked or transient handoff on the current Kanban task."""

    task_id = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    if not task_id:
        return
    board = os.environ.get("HERMES_KANBAN_BOARD", "").strip()
    reason = f"Exact-head CI receipt {receipt.receipt_id}: "
    reason += ", ".join(blockers) if blockers else "handoff did not complete"
    argv = [sys.executable, "-m", "hermes_cli.main", "kanban"]
    if board:
        argv.extend(["--board", board])
    argv.extend(["block", task_id])
    if kind is not None:
        argv.extend(["--kind", kind])
    argv.append(reason)
    try:
        completed = subprocess.run(
            argv,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except OSError as exc:
        raise RuntimeError("Hermes runtime unavailable for Kanban audit block") from exc
    if completed.returncode != 0:
        raise RuntimeError("Kanban audit block failed")


def _merge_scan(ctx: Any) -> int:
    try:
        policy = _load_policy_from_context(ctx)
    except ValueError:
        print(json.dumps({"status": "invalid_configuration"}, sort_keys=True))
        return 1
    if not policy.merge_policies():
        print(json.dumps({"status": "disabled"}, sort_keys=True))
        return 0
    ledger = FeedbackLedger.for_current_profile()
    try:
        payload = _run_merge_scan(policy, ledger)
    finally:
        ledger.close()
    print(json.dumps(payload, sort_keys=True))
    return 1 if payload["status"] == "degraded" else 0


def _run_merge_scan(
    policy: PluginPolicy,
    ledger: FeedbackLedger,
    *,
    github: GitHubClient | None = None,
    kanban: KanbanSubprocessClient | None = None,
) -> dict[str, object]:
    merge_policies = policy.merge_policies()
    if not merge_policies:
        return {
            "status": "disabled",
            "processed": 0,
            "merged": [],
            "handed_off": [],
            "blocked": {},
        }
    if len(merge_policies) == 1:
        return _run_merge_scan_for_policy(
            policy, merge_policies[0], ledger, github=github, kanban=kanban
        )
    results = [
        _run_merge_scan_for_policy(
            policy, merge_policy, ledger, github=github, kanban=kanban
        )
        for merge_policy in merge_policies
    ]
    blocked: dict[str, list[str]] = {}
    for merge_policy, result in zip(merge_policies, results, strict=True):
        for number, blockers in result.get("blocked", {}).items():
            blocked[f"{merge_policy.repository}#{number}"] = blockers
    return {
        "status": "degraded"
        if any(result.get("status") == "degraded" for result in results)
        else "ok",
        "processed": sum(int(result.get("processed", 0)) for result in results),
        "merged": [item for result in results for item in result.get("merged", [])],
        "handed_off": [
            item for result in results for item in result.get("handed_off", [])
        ],
        "blocked": blocked,
        "maintainer_tasks_created": sum(
            int(result.get("maintainer_tasks_created", 0)) for result in results
        ),
        "maintainer_task_dispatch_failed": [
            f"{merge_policy.repository}#{number}"
            for merge_policy, result in zip(merge_policies, results, strict=True)
            for number in result.get("maintainer_task_dispatch_failed", [])
        ],
        "deployments": [
            item for result in results for item in result.get("deployments", [])
        ],
        "deployment_failures": [
            f"{merge_policy.repository}#{number}"
            for merge_policy, result in zip(merge_policies, results, strict=True)
            for number in result.get("deployment_failures", [])
        ],
        "report_only": all(bool(result.get("report_only", False)) for result in results),
    }


def _run_merge_scan_for_policy(
    policy: PluginPolicy,
    merge_policy,
    ledger: FeedbackLedger,
    *,
    github: GitHubClient | None = None,
    kanban: KanbanSubprocessClient | None = None,
) -> dict[str, object]:
    github = github or _github_client(policy)
    kanban = kanban or KanbanSubprocessClient()
    try:
        pull_requests = github.list_open_pull_requests(
            merge_policy.repository, merge_policy.author_login
        )
    except (GitHubClientError, RuntimeError):
        return {
            "status": "degraded",
            "processed": 0,
            "merged": [],
            "blocked": {"canonical_read": ["github_state_unavailable"]},
        }
    from .merge_admission import enroll_owned_pulls

    enroll_owned_pulls(policy, merge_policy, ledger, pull_requests)
    source = CanonicalMergeEvidenceSource(policy, github, ledger, merge_policy)
    manifest_path = ci_manifest_path(policy.targets[merge_policy.repository].local_path)
    if not manifest_path.is_file():
        return {
            "status": "degraded",
            "processed": 0,
            "merged": [],
            "blocked": {"canonical_read": ["ci_manifest_unavailable"]},
        }
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    merged: list[dict[str, object]] = []
    blocked: dict[str, list[str]] = {}
    deployments: list[dict[str, object]] = []
    deployment_failures: list[int] = []
    maintainer_task_dispatch_failed: list[int] = []
    tasks_created = 0
    degraded = False
    from .pr_ordering import order_pull_requests

    pull_requests = order_pull_requests(pull_requests)
    open_by_number = {pull_request.number: pull_request for pull_request in pull_requests}
    enrolled_numbers = ledger.enrolled_merge_pr_numbers(merge_policy.repository)
    pending_reader = getattr(ledger, "verification_required_merge_numbers", None)
    pending_numbers = tuple(
        pending_reader(merge_policy.repository) if callable(pending_reader) else ()
    )
    pending_set = set(pending_numbers)
    numbers = (
        *pending_numbers,
        *(
            number
            for number in open_by_number
            if number in enrolled_numbers and number not in pending_set
        ),
    )
    for number in numbers:
        pull_request = open_by_number.get(number)
        if number not in pending_set:
            assert pull_request is not None
            receipt = ledger.latest_ci_receipt(
                merge_policy.repository,
                pull_request.number,
                pull_request.head_sha,
                manifest_digest=manifest_digest,
                not_before=datetime.min.replace(tzinfo=UTC),
            )
            if receipt is None:
                comment_reader = getattr(type(github), "list_ci_receipt_comments", None)
                if callable(comment_reader):
                    try:
                        comment_receipt = ci_receipt_comment_from_feedback(
                            comment_reader(
                                github,
                                merge_policy.repository, pull_request.number
                            ),
                            pull_request,
                            expected_login=(
                                policy.github_identity.expected_login
                                if policy.github_identity is not None
                                else None
                            ),
                            manifest_digest=manifest_digest,
                        )
                    except GitHubClientError:
                        degraded = True
                        blocked[str(pull_request.number)] = [
                            "ci_receipt_comment_unavailable"
                        ]
                        continue
                    if comment_receipt is not None:
                        receipt = comment_receipt
            if receipt is None:
                # Keep the merge gate manifest-bound, but do not misreport an
                # existing exact-head failed receipt as missing merely because
                # its audit used a different manifest revision.
                reader = getattr(ledger, "latest_ci_receipt_for_head", None)
                exact_head_receipt = (
                    reader(
                        merge_policy.repository,
                        pull_request.number,
                        pull_request.head_sha,
                    )
                    if callable(reader)
                    else None
                )
                if exact_head_receipt is None:
                    blocked[str(pull_request.number)] = ["ci_receipt_missing"]
                elif exact_head_receipt.status == "passed":
                    blocked[str(pull_request.number)] = ["ci_manifest_mismatch"]
                else:
                    blocked[str(pull_request.number)] = ["ci_receipt_not_passing"]
                continue
            if receipt.status != "passed":
                blocked[str(pull_request.number)] = ["ci_receipt_not_passing"]
                continue
        try:
            result = MergeController(
                merge_policy,
                source,
                github,
                ledger,
                owner="github-pr-feedback-merge-controller",
            ).run(number)
        except (GitHubClientError, RuntimeError, ValueError):
            degraded = True
            blocked[str(number)] = ["merge_evidence_unavailable"]
            continue
        except Exception as exc:  # noqa: BLE001 - isolate one malformed candidate from the batch.
            # A single stale or malformed candidate must not terminate a repository-wide
            # scan before later PRs get their own deterministic admission decision. Keep the
            # batch degraded and the candidate blocked, while retaining the exception class in
            # logs for diagnosis instead of turning this into a silent skip.
            degraded = True
            blocked[str(number)] = ["merge_candidate_failed"]
            _LOGGER.warning(
                "merge candidate failed; continuing scan repository=%s pr=%s error=%s: %s",
                merge_policy.repository,
                number,
                type(exc).__name__,
                exc,
            )
            continue
        if result.receipt is not None:
            merged.append(
                {
                    "pr_number": number,
                    "head_sha": result.receipt.tested_head_sha,
                    "method": result.receipt.method,
                    "merge_commit_oid": result.receipt.merge_commit_oid,
                }
            )
            if merge_policy.post_merge is not None:
                try:
                    existing = ledger.latest_deployment_receipt(
                        merge_policy.repository, number
                    )
                    if getattr(existing, "status", None) != "completed":
                        deployment = PostMergeExecutor(
                            merge_policy.post_merge, ledger
                        ).run(result.receipt)
                        deployments.append(
                            {
                                "pr_number": number,
                                "status": deployment.status,
                                "blocker": deployment.blocker,
                            }
                        )
                except (RuntimeError, ValueError):
                    degraded = True
                    deployment_failures.append(number)
            continue
        blocker_codes = list(result.decision.blockers)
        blocked[str(number)] = blocker_codes
        # Deterministic blockers are already durable in the scan result and are
        # actionable by the repair controller. A model-backed observability card
        # can only restate them, adding queue latency without changing authority.
        # Report-only mode has no repair owner, so retain its explicit human-facing
        # readiness report without enabling any write authority.
        if merge_policy.report_only and pull_request is not None:
            if not blocker_codes:
                _announce_ready_to_merge(
                    github, merge_policy.repository, pull_request
                )
            try:
                kanban.create_or_get_task(
                    _merge_maintainer_task(
                        policy, merge_policy, pull_request, result.decision
                    )
                )
                tasks_created += 1
            except (RuntimeError, ValueError):
                degraded = True
                maintainer_task_dispatch_failed.append(number)
    return {
        "status": "degraded" if degraded else "ok",
        "processed": len(numbers),
        "merged": merged,
        "blocked": blocked,
        "maintainer_tasks_created": tasks_created,
        "maintainer_task_dispatch_failed": maintainer_task_dispatch_failed,
        "deployments": deployments,
        "deployment_failures": deployment_failures,
        "report_only": merge_policy.report_only,
    }


def _run_single_pr_merge_handoff(
    policy: PluginPolicy,
    ledger: FeedbackLedger,
    pr_number: int,
    *,
    repository: str | None = None,
    github: GitHubClient | None = None,
    kanban: KanbanSubprocessClient | None = None,
) -> dict[str, object]:
    """Attempt one exact PR merge, then admit at most one successor repair."""

    merge_policy = (
        policy.merge_policy_for(repository)
        if repository is not None
        else (policy.merge_policies()[0] if len(policy.merge_policies()) == 1 else None)
    )
    if merge_policy is None:
        return {"status": "disabled", "blockers": ["merge_maintainer_disabled"]}
    if not ledger.is_merge_enrolled(merge_policy.repository, pr_number):
        return {"status": "blocked", "blockers": ["merge_pr_not_enrolled"]}
    github = github or _github_client(policy)
    kanban = kanban or KanbanSubprocessClient()
    source = CanonicalMergeEvidenceSource(policy, github, ledger, merge_policy)
    try:
        result = MergeController(
            merge_policy,
            source,
            github,
            ledger,
            owner="github-pr-feedback-ci-handoff",
        ).run(pr_number)
    except (GitHubClientError, RuntimeError, ValueError):
        return {"status": "degraded", "blockers": ["merge_evidence_unavailable"]}

    if result.receipt is None:
        return {"status": "blocked", "blockers": list(result.decision.blockers)}

    payload: dict[str, object] = {
        "status": "merged",
        "pr_number": pr_number,
        "head_sha": result.receipt.tested_head_sha,
        "method": result.receipt.method,
        "merge_commit_oid": result.receipt.merge_commit_oid,
    }
    repair_policy = policy.repair_steward
    if repair_policy is None:
        payload["next_repair"] = {"status": "disabled", "created": 0, "skipped": {}}
        return payload

    serial_policy = replace(
        policy,
        repair_steward=replace(
            repair_policy,
            repositories=frozenset({merge_policy.repository}),
            max_base_refresh_in_flight=1,
        ),
    )
    try:
        repair_result = RepairController(
            serial_policy,
            ledger,
            github,
            kanban,
            control_home=get_default_hermes_root(),
        ).scan()
    except (GitHubClientError, LedgerStateError, RuntimeError, ValueError):
        payload["next_repair"] = {
            "status": "degraded",
            "created": 0,
            "skipped": {"repair_scan_unavailable": 1},
        }
    else:
        payload["next_repair"] = _scan_payload(repair_result)
    return payload


_READY_TO_MERGE_MARKER_PREFIX = "<!-- pr-ready-to-merge-receipt:v1 head="


def _announce_ready_to_merge(github: GitHubClient, repository: str, pull_request) -> None:
    """Post one visible, idempotent PR comment when every report-only gate clears.

    Report-only mode never merges automatically, so a human maintainer has no
    other visible signal that a PR is done: the deterministic readiness card
    only completes an internal Kanban record. Marker-gated on the exact head
    SHA so a re-scan of an already-announced head is a no-op, and a new head
    (new commits) gets its own fresh announcement.
    """

    marker = f"{_READY_TO_MERGE_MARKER_PREFIX}{pull_request.head_sha} -->"
    try:
        feedback = github.list_feedback(repository, pull_request.number)
    except (GitHubClientError, RuntimeError):
        return
    if any(marker in (item.body or "") for item in feedback):
        return
    body = (
        "**Ready to merge.** Local CI passed and every tracked repair/review "
        "item is clear for this exact head. The merge maintainer is running "
        "in report-only mode, so nothing merges automatically here — merge "
        "manually when ready.\n\n"
        f"{marker}"
    )
    try:
        github.post_issue_comment(repository, pull_request.number, body)
    except (GitHubClientError, RuntimeError):
        pass


def _merge_maintainer_task(
    policy: PluginPolicy, merge_policy, pull_request, decision: MergeDecision
) -> KanbanTask:
    target = policy.targets[merge_policy.repository]
    evidence = {
        "repository": merge_policy.repository,
        "pr_number": pull_request.number,
        "expected_head_sha": pull_request.head_sha,
        "eligible": decision.eligible,
        "blockers": list(decision.blockers),
        "snapshot_digest": decision.snapshot_digest,
        "report_only": merge_policy.report_only,
        "canonical_evidence": dict(decision.evidence),
    }
    key = hashlib.sha256(
        f"{merge_policy.repository}\0{pull_request.number}\0{pull_request.head_sha}".encode(
            "utf-8"
        )
    ).hexdigest()
    return KanbanTask(
        title=f"PR merge readiness: {merge_policy.repository}#{pull_request.number}",
        instructions=(
            "This is a read-only observability card with no repository or GitHub mutation "
            "authority. Inspect only the deterministic blocker codes and explain what canonical evidence is "
            "missing from the supplied deterministic evidence; do not inspect the repository, GitHub, "
            "or other sources. The listed PR blockers are the requested report, not a blocker for this "
            "observability card. Model output cannot waive a blocker or create CI or "
            "merge receipts. After the bounded explanation, immediately call kanban_complete with the "
            "repository, PR number, exact base/head, CI receipt identity, review/check state, blockers, "
            "and snapshot digest. Call kanban_block only "
            "if kanban_complete itself is unavailable or rejected. A deterministic controller will act "
            "automatically after every gate passes."
        ),
        board=policy.board or "",
        assignee=merge_policy.assignee,
        repository_path=target.local_path,
        head_sha=pull_request.head_sha,
        branch=pull_request.head_ref_name,
        idempotency_key=f"github-pr-merge-maintainer:{key}",
        evidence=evidence,
        evidence_heading="Deterministic merge readiness (JSON)",
        initial_status="running",
        max_retries=3,
    )


def _merge_status(*, details: bool = False) -> int:
    ledger = FeedbackLedger.for_current_profile()
    try:
        payload = {"merge": ledger.merge_status_counts()}
        if details:
            from .merge_receipt_history import recent_merge_receipts
            payload["recent_receipts"] = recent_merge_receipts(ledger)
    finally:
        ledger.close()
    print(json.dumps(payload, sort_keys=True))
    return 0


def _review_required_task(policy: PluginPolicy, merge_policy, pull_request) -> KanbanTask:
    """Create an exact-head review task when GitHub reports review required."""

    target = policy.targets[merge_policy.repository]
    key = hashlib.sha256(
        f"{merge_policy.repository}\0{pull_request.number}\0{pull_request.head_sha}".encode(
            "utf-8"
        )
    ).hexdigest()
    return KanbanTask(
        title=f"Review required: {merge_policy.repository}#{pull_request.number}",
        instructions=(
            "Perform an independent code review of this exact pull-request head. Re-read the "
            "canonical PR identity and review the complete diff and relevant tests. Leave factual "
            "review feedback on GitHub for any issue found; if the change is sound, submit the "
            "repository's normal approval review. Do not edit, push, force-push, change branch "
            "protection, or merge the pull request. Stop if the base or head SHA changes. The "
            "merge controller will re-read the review state automatically and will not waive this "
            "gate from model output."
        ),
        board=policy.board or "",
        assignee=_REVIEW_REQUIRED_ASSIGNEE,
        repository_path=target.local_path,
        head_sha=pull_request.head_sha,
        branch=pull_request.head_ref_name,
        idempotency_key=f"github-pr-review-required:{key}",
        evidence={
            "repository": merge_policy.repository,
            "pr_number": pull_request.number,
            "expected_head_sha": pull_request.head_sha,
            "base_branch": pull_request.base_branch,
            "reason": "review_required",
        },
        evidence_heading="Canonical review-required evidence (JSON)",
        initial_status="running" if policy.auto_dispatch else "blocked",
        max_retries=2 if policy.auto_dispatch else 1,
        max_runtime_seconds=(
            AUTO_DISPATCH_MAX_RUNTIME_SECONDS if policy.auto_dispatch else None
        ),
    )


def _merge_enable(ctx: Any, args: argparse.Namespace) -> int:
    """Enroll one exact configured PR without granting the worker new authority."""

    try:
        policy = _load_policy_from_context(ctx)
        merge_policy = policy.merge_policy_for(args.repository)
        if merge_policy is None:
            raise ValueError("repository is not configured for merging")
        github = _github_client(policy)
        pull = github.get_merge_state(args.repository, args.pr_number)
        if (
            pull.repository != merge_policy.repository
            or pull.head_repository != merge_policy.repository
            or pull.author_login.casefold() != merge_policy.author_login.casefold()
            or pull.base_branch != merge_policy.base_branch
            or pull.state != "OPEN"
            or pull.merged
        ):
            raise ValueError("PR identity is outside the configured merge lane")
        ledger = FeedbackLedger.for_current_profile()
        try:
            ledger.enroll_merge_pr(
                args.repository,
                args.pr_number,
                enrolled_at=datetime.now(UTC),
                enrolled_by=os.environ.get("USER", "hermes-operator"),
            )
        finally:
            ledger.close()
    except (GitHubClientError, LedgerStateError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "unavailable", "reason": str(error)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "status": "enrolled",
                "repository": args.repository,
                "pr_number": args.pr_number,
                "head_sha": pull.head_sha,
            },
            sort_keys=True,
        )
    )
    return 0


def _parse_stack_entries(raw_entries: list[str]) -> tuple[StackEntry, ...]:
    entries: list[StackEntry] = []
    for raw in raw_entries:
        if "=" not in raw or ":" not in raw:
            raise ValueError("--entry must be BRANCH=BASE:TITLE")
        branch, remainder = raw.split("=", 1)
        base, title = remainder.split(":", 1)
        entries.append(StackEntry(branch, base, title, f"Hermes stack entry: {title}"))
    return tuple(entries)


def _stack_create(ctx: Any, args: argparse.Namespace) -> int:
    try:
        policy = _load_policy_from_context(ctx)
        manifest = StackController(policy, github=_github_client(policy)).create(
            args.repository,
            args.stack_id,
            args.base_branch,
            _parse_stack_entries(args.entry),
        )
    except (GitHubClientError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "unavailable", "reason": str(error)}, sort_keys=True))
        return 1
    print(json.dumps({"status": "created", "repository": manifest.repository,
                      "stack_id": manifest.stack_id,
                      "entries": [
                          {"branch": item.branch, "base_branch": item.base_branch,
                           "pr_number": item.pr_number, "head_sha": item.head_sha}
                          for item in manifest.entries
                      ]}, sort_keys=True))
    return 0


def _stack_refresh(ctx: Any, args: argparse.Namespace) -> int:
    try:
        policy = _load_policy_from_context(ctx)
        manifest = StackController(policy, github=_github_client(policy)).refresh(
            args.repository,
            args.stack_id,
            repository_path=args.repository_path,
            accept_rebased_heads=args.accept_rebased_heads,
        )
    except (GitHubClientError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "blocked", "reason": str(error)}, sort_keys=True))
        return 1
    print(json.dumps({"status": "refreshed", "repository": manifest.repository,
                      "stack_id": manifest.stack_id,
                      "entries": [
                          {"branch": item.branch, "base_branch": item.base_branch,
                           "pr_number": item.pr_number, "head_sha": item.head_sha}
                          for item in manifest.entries
                      ]}, sort_keys=True))
    return 0


def _stack_merge(ctx: Any, args: argparse.Namespace) -> int:
    try:
        policy = _load_policy_from_context(ctx)
        controller = StackController(policy, github=_github_client(policy))
        manifest = controller.store.load(args.repository, args.stack_id)
        results: list[dict[str, object]] = []
        ledger = FeedbackLedger.for_current_profile()
        try:
            for entry in _ordered(manifest.entries, manifest.base_branch):
                manifest = controller.refresh(
                    args.repository, args.stack_id, repository_path=args.repository_path
                )
                current = next(item for item in manifest.entries if item.branch == entry.branch)
                result = _run_single_pr_merge_handoff(
                    policy, ledger, current.pr_number or 0, repository=args.repository
                )
                results.append(result)
                if result.get("status") != "merged":
                    print(json.dumps({"status": "blocked", "results": results}, sort_keys=True))
                    return 1
        finally:
            ledger.close()
        print(json.dumps({"status": "merged", "results": results}, sort_keys=True))
        return 0
    except (GitHubClientError, LedgerStateError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "blocked", "reason": str(error)}, sort_keys=True))
        return 1


def _close_superseded(ctx: Any, args: argparse.Namespace) -> int:
    try:
        policy = _load_policy_from_context(ctx)
        result = SupersededPullRequestController(
            policy, github=_github_client(policy)
        ).close(
            args.repository,
            args.pr_number,
            args.head_sha,
            repository_path=args.repository_path,
        )
    except (GitHubClientError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "blocked", "reason": str(error)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "status": "closed_superseded",
                "repository": result.repository,
                "pr_number": result.pr_number,
                "head_sha": result.head_sha,
                "base_branch": result.base_branch,
            },
            sort_keys=True,
        )
    )
    return 0


def _resolve_superseded_feedback(ctx: Any, args: argparse.Namespace) -> int:
    try:
        policy = _load_policy_from_context(ctx)
        result = SupersededFeedbackController(
            policy, github=_github_client(policy)
        ).resolve(
            args.repository,
            args.pr_number,
            args.head_sha,
            comment_id=args.comment_id,
            fix_sha=args.fix_sha,
            repository_path=args.repository_path,
            test_evidence=args.test_evidence,
        )
        receipt = FeedbackReceipt(
            result.repository,
            result.pr_number,
            "review_comment",
            result.comment_id,
            result.head_sha,
        )
        ledger = FeedbackLedger.for_current_profile()
        try:
            ledger.reconcile_superseded_feedback_action(
                receipt,
                stable_fix_sha=result.fix_sha,
                actioned_at=datetime.now(UTC),
            )
        finally:
            ledger.close()
    except (GitHubClientError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "blocked", "reason": str(error)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "status": "resolved_superseded_feedback",
                "repository": result.repository,
                "pr_number": result.pr_number,
                "head_sha": result.head_sha,
                "comment_id": result.comment_id,
                "fix_sha": result.fix_sha,
                "base_branch": result.base_branch,
                "thread_resolved": result.thread_resolved,
            },
            sort_keys=True,
        )
    )
    return 0


def _merge_disable(ctx: Any, args: argparse.Namespace) -> int:
    try:
        policy = _load_policy_from_context(ctx)
        if policy.merge_policy_for(args.repository) is None:
            raise ValueError("repository is not configured for merging")
        ledger = FeedbackLedger.for_current_profile()
        try:
            ledger.unenroll_merge_pr(args.repository, args.pr_number)
        finally:
            ledger.close()
    except (LedgerStateError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "unavailable", "reason": str(error)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "status": "unenrolled",
                "repository": args.repository,
                "pr_number": args.pr_number,
            },
            sort_keys=True,
        )
    )
    return 0


def _inspect_pr(ctx: Any, args: argparse.Namespace) -> int:
    try:
        policy = _load_policy_from_context(ctx)
        if not policy.enabled or args.repository not in policy.targets:
            raise ValueError("repository is not a configured target")
        github = _github_client(policy)
        pull_request = github.get_pull_request(args.repository, args.pr_number)
        feedback_projection: dict[str, Any] = {}
        feedback_id = getattr(args, "feedback_id", None)
        if feedback_id:
            matches = [
                item
                for item in github.list_feedback(args.repository, args.pr_number)
                if item.feedback_id == str(feedback_id)
            ]
            if len(matches) != 1:
                raise ValueError("feedback item was not found")
            feedback = matches[0]
            feedback_projection = {
                "feedback_body_excerpt": _feedback_body_excerpt(feedback.body),
                "feedback_id": feedback.feedback_id,
                "feedback_is_bot": feedback.is_bot,
                "feedback_kind": feedback.kind,
                "feedback_reviewer": feedback.reviewer.login,
            }
    except (GitHubClientError, ValueError) as error:
        code = getattr(error, "code", "invalid_request")
        print(json.dumps({"status": "unavailable", "reason": code,
                          "retryable": code in {"rate_limited", "transient", "timeout"}}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "base_branch": pull_request.base_branch,
                "base_sha": pull_request.base_sha,
                **feedback_projection,
                "head_ref_name": pull_request.head_ref_name,
                "head_repository": pull_request.head_repository,
                "head_sha": pull_request.head_sha,
                "number": pull_request.number,
                "repository": pull_request.base_repository,
                "state": pull_request.state,
            },
            sort_keys=True,
        )
    )
    return 0


def _doctor(
    ctx: Any,
    *,
    probe: DoctorProbe | None = None,
    ledger_path: Path | None = None,
) -> int:
    try:
        policy = _load_policy_from_context(ctx)
    except ValueError:
        print(json.dumps({"status": "invalid_configuration"}, sort_keys=True))
        return 1
    if not policy.enabled:
        print(json.dumps({"status": "disabled"}, sort_keys=True))
        return 0
    path = ledger_path or FeedbackLedger.current_profile_path()
    checks = (probe or DoctorProbe(get_default_hermes_root())).checks(policy, path)
    try:
        _github_client(policy)
    except GitHubClientError:
        checks["github_identity"] = "failed"
    else:
        checks["github_identity"] = "ok"
    ready = all(value == "ok" for value in checks.values())
    print(
        json.dumps(
            {"status": "ready" if ready else "degraded", "checks": checks},
            sort_keys=True,
        )
    )
    return 0 if ready else 1


def _load_policy_from_context(ctx: Any) -> PluginPolicy:
    enabled = ctx.get_config("enabled", default=False)
    if enabled is not True:
        return load_policy({"enabled": False})
    settings: dict[str, object] = {"enabled": True}
    for key in (
        "repositories",
        "reviewer_logins",
        "reviewer_associations",
        "include_self_feedback",
        "include_bot_feedback",
        "auto_dispatch",
        "assignee_rules",
        "routing_rules",
        "local_ci_audit",
        "agent_labels",
        "merge_maintainer",
        "merge_maintainers",
        "repair_steward",
        "release_maintenance",
        "release_maintenances",
        "github_identity",
        "github_actions_permissions_identity",
        "not_before",
        "assignee",
        "board",
    ):
        value = ctx.get_config(key, default=_MISSING)
        if value is not _MISSING:
            settings[key] = value
    return load_policy(settings)


def _safe_name(value: str) -> bool:
    return (
        bool(value)
        and len(value) <= 64
        and all(character.isalnum() or character in "-_." for character in value)
        and "/" not in value
        and "\\" not in value
        and value not in {".", ".."}
    )


def _nearest_existing_parent_access(path: Path) -> bool:
    candidate = Path(path)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate.is_dir() and os.access(candidate, os.R_OK | os.W_OK | os.X_OK)


def _label_scan(ctx, args):
    policy = _load_policy_from_context(ctx)
    ledger = FeedbackLedger.for_current_profile()
    try:
        result = _controller(policy, ledger).reconcile_labels(args.repository)
    finally:
        ledger.close()
    print(json.dumps(result, sort_keys=True))
    return 0

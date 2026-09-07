"""Hermes CLI wiring for governed GitHub pull-request feedback intake."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

from .controller import KanbanTask, LocalGitRepository, PooledLocalGitRepository, ScanController
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
)
from .policy import (
    FeedbackReceipt,
    PluginPolicy,
    codex_review_trigger_comment,
    codex_review_trigger_requested,
    hermes_attribution_line,
    load_policy,
)
from .post_merge import PostMergeExecutor
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
    """Mention @codex review after a verified repair push, once, if needed.

    Codex's GitHub App never re-reviews on an ordinary push -- only on PR
    opened, marked ready, or this exact mention (see merge_controller's
    codex_review_pending). Without this, a repaired PR would carry a
    permanently stale Codex review and sit blocked on that gate forever.
    Rereads canonical comments first so a PR whose new head Codex has
    already reviewed (e.g. two repairs landing back to back) does not get a
    redundant mention.
    """

    try:
        feedback = github.list_feedback(repository, pr_number)
    except GitHubClientError:
        return "unavailable"
    if _codex_reviewed_head(feedback, resolved_head_sha):
        return "already_current"
    if any(
        codex_review_trigger_requested(item.body, resolved_head_sha)
        for item in feedback
    ):
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
        self, hermes_root: Path, runner: DoctorCommandRunner | None = None
    ) -> None:
        self._hermes_root = Path(hermes_root)
        self._runner = runner or SubprocessDoctorRunner()

    def checks(self, policy: PluginPolicy, ledger_path: Path) -> dict[str, str]:
        gh = self._runner.which("gh")
        hermes = self._runner.which("hermes")
        checks = {
            "gh_executable": gh is not None,
            "gh_auth": bool(
                gh
                and self._command_ok([gh, "auth", "status", "--hostname", "github.com"])
            ),
            "hermes_executable": bool(
                hermes and self._command_ok([hermes, "--version"])
            ),
            "board": self._board_exists(policy.board or ""),
            "assignee": all(
                self._assignee_exists(assignee)
                for assignee in {
                    policy.assignee or "",
                    *(rule.assignee for rule in policy.assignee_rules),
                    *(rule.assignee for rule in policy.routing_rules),
                    *(
                        [policy.local_ci_audit.assignee]
                        if policy.local_ci_audit is not None
                        else []
                    ),
                    *(
                        [policy.merge_maintainer.assignee]
                        if policy.merge_maintainer is not None
                        else []
                    ),
                    *(
                        [policy.repair_steward.assignee]
                        if policy.repair_steward is not None
                        else []
                    ),
                    *(
                        [
                            policy.release_maintenance.assignee,
                            *(
                                lane.assignee
                                for lane in policy.release_maintenance.lanes
                            ),
                        ]
                        if policy.release_maintenance is not None
                        else []
                    ),
                }
            ),
            "ledger_access": self._ledger_access(ledger_path),
            "repository_worktree": self._repositories_ready(
                tuple(target.local_path for target in policy.targets.values()),
                ledger_path.parent / "worktrees",
            ),
        }
        return {name: "ok" if ok else "failed" for name, ok in checks.items()}

    def _command_ok(self, argv: list[str]) -> bool:
        try:
            return self._runner.run(argv).returncode == 0
        except Exception:  # noqa: BLE001 - doctor reports failure and continues.
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
        return task_id.strip()

    def task_status(self, board: str, task_id: str) -> str | None:
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
        status = task.get("status") if isinstance(task, dict) else None
        if not isinstance(status, str) or not status.strip():
            raise RuntimeError("Kanban task lookup failed")
        return status.strip()


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


def setup_cli(_ctx: Any, parser: argparse.ArgumentParser) -> None:
    """Attach the plugin's command tree to the host-created parser."""

    subcommands = parser.add_subparsers(dest="github_pr_feedback_action", required=True)
    subcommands.add_parser("scan", help="Read and dispatch newly admitted feedback")
    subcommands.add_parser("status", help="Show durable receipt counts")
    subcommands.add_parser(
        "doctor", help="Check configuration readiness without scanning"
    )
    inspect = subcommands.add_parser(
        "inspect-pr", help="Read one configured PR identity through the shared GitHub gate"
    )
    inspect.add_argument("--repository", required=True)
    inspect.add_argument("--pr-number", required=True, type=int)
    retry = subcommands.add_parser(
        "retry", help="Retry one failed, immutable feedback receipt"
    )
    retry.add_argument("--repository", required=True)
    retry.add_argument("--pr-number", required=True, type=int)
    retry.add_argument("--feedback-kind", required=True)
    retry.add_argument("--feedback-id", required=True)
    retry.add_argument("--head-sha", required=True)
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
    subcommands.add_parser(
        "merge-status", help="Show bounded merge and deployment counts"
    )
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
    if action == "inspect-pr":
        return _inspect_pr(ctx, args)
    if action == "retry":
        return _retry(ctx, args)
    if action == "audit-pr":
        return _audit_pr(ctx, args)
    if action == "merge-scan":
        return _merge_scan(ctx)
    if action == "merge-status":
        return _merge_status()
    if action == "complete-feedback":
        return _complete_feedback(ctx, args)
    if action == "complete-maintenance":
        return _complete_maintenance(ctx, args)
    return 2


def _complete_maintenance(ctx: Any, args: argparse.Namespace) -> int:
    try:
        policy = _load_policy_from_context(ctx)
        maintenance = policy.release_maintenance
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


def _scan(ctx: Any) -> int:
    try:
        policy = _load_policy_from_context(ctx)
    except ValueError:
        print(json.dumps({"status": "invalid_configuration"}, sort_keys=True))
        return 1
    with _exclusive_scan_lock() as acquired:
        if not acquired:
            print(json.dumps({"status": "scan_in_progress"}, sort_keys=True))
            return 0
        ledger = FeedbackLedger.for_current_profile()
        merge_payload: dict[str, object] | None = None
        repair_payload: dict[str, object] | None = None
        maintenance_payload: dict[str, object] | None = None
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
            result = controller.scan(apply_labels=False)
            # Required exact-head CI still gates repair and release fan-out, but
            # it must not hold already-authoritative-ready PRs hostage. The
            # merge maintainer independently checks each exact-head receipt and
            # may advance any eligible PR in the catalogue, regardless of
            # where it falls relative to the local-CI admission window.
            required_ci_backlog = result.required_local_ci_backlog > 0
            if policy.repair_steward is not None and not required_ci_backlog:
                repair = RepairController(
                    policy,
                    ledger,
                    GitHubClient(),
                    KanbanSubprocessClient(),
                    control_home=get_default_hermes_root(),
                ).scan()
                repair_payload = _scan_payload(repair)
            if policy.merge_maintainer is not None:
                merge_payload = _run_merge_scan(policy, ledger)
            if policy.release_maintenance is not None and not required_ci_backlog:
                maintenance_payload = _run_release_maintenance_scan(policy, ledger)
            label_payload = controller.apply_agent_labels()
        finally:
            ledger.close()
    payload = _scan_payload(result)
    if result.required_local_ci_backlog > 0:
        payload["deferred"] = ["repair", "release_maintenance"]
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
            or (maintenance_payload or {}).get("status") == "degraded"
        )
        else 0
    )


def _run_release_maintenance_scan(
    policy: PluginPolicy,
    ledger: FeedbackLedger,
    *,
    github: MaintenanceGitHub | None = None,
    kanban: KanbanSubprocessClient | None = None,
    workspaces: MaintenanceWorkspaces | None = None,
    now: Callable[[], datetime] | None = None,
    control_home: Path | None = None,
) -> dict[str, object]:
    maintenance = policy.release_maintenance
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
        github or GitHubClient(),
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
        result = _controller(policy, ledger).retry_failed(receipt)
    finally:
        ledger.close()
    print(json.dumps(_scan_payload(result), sort_keys=True))
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
        github = GitHubClient()
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
        GitHubClient(),
        KanbanSubprocessClient(),
        control_home=get_default_hermes_root(),
    )


def _scan_payload(result) -> dict[str, object]:
    payload = {
        "status": "degraded" if result.degraded else "ok",
        "created": result.created,
        "skipped": result.skipped,
    }
    if result.required_local_ci_backlog > 0:
        payload["required_local_ci_backlog"] = result.required_local_ci_backlog
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
    manifest_path = worktree / "tests/manifests/test_lanes.toml"
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
) -> CIAuditReceipt:
    """Run one immutable audit through the bounded grouped-coordination boundary."""

    receipt = _reusable_ci_receipt(
        ledger, identity, worktree, allow_reuse=not force_fresh
    )
    if receipt is not None:
        return receipt
    manifest_path = worktree / "tests/manifests/test_lanes.toml"
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
        return _ThreadOwnedCIRunner(
            LocalCIRunner(github, worker_ledger),
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

    return {
        "status": receipt.status,
        "receipt_id": receipt.receipt_id,
        "repository": receipt.identity.repository,
        "pr_number": receipt.identity.pr_number,
        "head_sha": receipt.identity.head_sha,
        "manifest_digest": receipt.manifest_digest,
        "command_count": len(receipt.commands),
        "handoff_status": "pending",
    }


def _audit_pr(ctx: Any, args: argparse.Namespace) -> int:
    handoff_completed = False
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
        github = GitHubClient()
        state = github.get_merge_state(args.repository, args.pr_number)
        if state.head_sha != str(args.head_sha).casefold():
            raise CIValidationError("canonical PR head changed")
        identity = CIAuditIdentity(
            args.repository, args.pr_number, state.base_sha, state.head_sha
        )
    except (ValueError, CIValidationError, GitHubClientError):
        print(json.dumps({"status": "invalid_or_raced_audit_identity"}, sort_keys=True))
        return 1
    ledger = FeedbackLedger.for_current_profile()
    try:
        receipt = _run_grouped_exact_head_audit(
            github,
            ledger,
            identity,
            worktree,
            force_fresh=bool(getattr(args, "fresh", False)),
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
        try:
            final_state = github.get_merge_state(args.repository, args.pr_number)
            if final_state.head_sha != receipt.identity.head_sha:
                raise CIValidationError("canonical PR head changed after audit")
            if policy.local_ci_audit.post_results:
                feedback = github.list_feedback(
                    receipt.identity.repository, receipt.identity.pr_number
                )
                if not any(receipt.receipt_id in item.body for item in feedback):
                    github.post_issue_comment(
                        receipt.identity.repository,
                        receipt.identity.pr_number,
                        _ci_audit_comment(receipt),
                    )
            if receipt.status == "failed":
                repair_status = _controller(policy, ledger).dispatch_ci_failure(receipt)
                if repair_status not in {"scheduled", "duplicate"}:
                    raise RuntimeError(f"typed CI repair handoff failed: {repair_status}")
            else:
                merge_handoff = _run_single_pr_merge_handoff(
                    policy,
                    ledger,
                    receipt.identity.pr_number,
                    github=github,
                )
                handoff_status = str(merge_handoff.get("status", ""))
                if handoff_status == "blocked":
                    raw_blockers = merge_handoff.get("blockers", [])
                    if isinstance(raw_blockers, list):
                        handoff_blockers = [str(blocker) for blocker in raw_blockers]
                    _block_current_ci_task(receipt, handoff_blockers)
                    handoff_blocked = True
                elif handoff_status != "merged":
                    raise RuntimeError(
                        "merge handoff did not produce a durable successor: "
                        f"{handoff_status}"
                    )
            if not handoff_blocked:
                _complete_current_ci_task(receipt)
                handoff_completed = True
        except (CIValidationError, GitHubClientError, RuntimeError) as error:
            if not handoff_blocked:
                try:
                    _block_current_ci_task(
                        receipt, ["transient_handoff_failure"], kind="transient"
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
        if handoff_completed:
            _terminate_current_ci_worker()
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
        f"id={receipt.receipt_id} head={receipt.identity.head_sha} -->"
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


def _terminate_current_ci_worker() -> None:
    """End only the task-scoped parent after the durable handoff is complete."""

    if not os.environ.get("HERMES_KANBAN_TASK", "").strip():
        return
    parent_pid = os.getppid()
    if parent_pid > 1:
        os.kill(parent_pid, signal.SIGTERM)


def _merge_scan(ctx: Any) -> int:
    try:
        policy = _load_policy_from_context(ctx)
    except ValueError:
        print(json.dumps({"status": "invalid_configuration"}, sort_keys=True))
        return 1
    if policy.merge_maintainer is None:
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
    merge_policy = policy.merge_maintainer
    if merge_policy is None:
        return {"status": "disabled", "processed": 0, "merged": [], "blocked": {}}
    github = github or GitHubClient()
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
    source = CanonicalMergeEvidenceSource(policy, github, ledger)
    manifest_path = (
        policy.targets[merge_policy.repository].local_path
        / "tests"
        / "manifests"
        / "test_lanes.toml"
    )
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
    open_by_number = {pull_request.number: pull_request for pull_request in pull_requests}
    pending_reader = getattr(ledger, "verification_required_merge_numbers", None)
    pending_numbers = tuple(
        pending_reader(merge_policy.repository) if callable(pending_reader) else ()
    )
    pending_set = set(pending_numbers)
    numbers = (
        *pending_numbers,
        *(number for number in open_by_number if number not in pending_set),
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
                    _merge_maintainer_task(policy, pull_request, result.decision)
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
    github: GitHubClient | None = None,
    kanban: KanbanSubprocessClient | None = None,
) -> dict[str, object]:
    """Attempt one exact PR merge, then admit at most one successor repair."""

    merge_policy = policy.merge_maintainer
    if merge_policy is None:
        return {"status": "disabled", "blockers": ["merge_maintainer_disabled"]}
    github = github or GitHubClient()
    kanban = kanban or KanbanSubprocessClient()
    source = CanonicalMergeEvidenceSource(policy, github, ledger)
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
    policy: PluginPolicy, pull_request, decision: MergeDecision
) -> KanbanTask:
    merge_policy = policy.merge_maintainer
    if merge_policy is None:
        raise ValueError("merge maintainer is disabled")
    target = policy.targets[merge_policy.repository]
    evidence = {
        "repository": merge_policy.repository,
        "pr_number": pull_request.number,
        "expected_head_sha": pull_request.head_sha,
        "eligible": decision.eligible,
        "blockers": list(decision.blockers),
        "snapshot_digest": decision.snapshot_digest,
        "report_only": merge_policy.report_only,
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
            "repository, PR number, expected head, blockers, and snapshot digest. Call kanban_block only "
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


def _merge_status() -> int:
    ledger = FeedbackLedger.for_current_profile()
    try:
        payload = {"merge": ledger.merge_status_counts()}
    finally:
        ledger.close()
    print(json.dumps(payload, sort_keys=True))
    return 0


def _inspect_pr(ctx: Any, args: argparse.Namespace) -> int:
    try:
        policy = _load_policy_from_context(ctx)
        if not policy.enabled or args.repository not in policy.targets:
            raise ValueError("repository is not a configured target")
        pull_request = GitHubClient().get_pull_request(
            args.repository, args.pr_number
        )
    except (GitHubClientError, ValueError):
        print(json.dumps({"status": "unavailable"}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "base_branch": pull_request.base_branch,
                "base_sha": pull_request.base_sha,
                "head_ref_name": pull_request.head_ref_name,
                "head_repository": pull_request.head_repository,
                "head_sha": pull_request.head_sha,
                "number": pull_request.number,
                "repository": pull_request.base_repository,
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
        "repair_steward",
        "release_maintenance",
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

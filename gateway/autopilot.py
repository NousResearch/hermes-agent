"""Hermes /autopilot command controller surface.

CH-385 deliberately stopped at the command/plugin entrypoint and deterministic
classification layer.  CH-389 adds the bounded one-task materialization helper:
controller intent + Linear admission can create one work_state lock and invoke a
caller-supplied executor spawner, while still refusing Linear Done mutations and
multi-task continuation.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol

from hermes_constants import get_hermes_home
from gateway.goal_contract import GoalContractError, generate_goal_contract


AUTOPILOT_USAGE = (
    "/autopilot [status|dry-run|ON|OFF|CH-123|status CH-123|dry-run CH-123]"
)
AUTOPILOT_STATE_VERSION = 1
AUTOPILOT_STATE_FILE = "gateway_autopilot_state.json"
_AUTOPILOT_TARGET_RE = re.compile(r"^[A-Z][A-Z0-9]*-\d+$")

ACTION_STATUS = "status"
ACTION_DRY_RUN = "dry_run"
ACTION_ENABLE = "enable"
ACTION_DISABLE = "disable"
ACTION_ONE_SHOT = "one_shot"

READ_ONLY_ACTIONS = {ACTION_STATUS, ACTION_DRY_RUN}

AUTOPILOT_CLOSEOUT_MODE = "autopilot"
KANBAN_CLOSEOUT_MODE = "kanban"
AUTOPILOT_PR_CLOSEOUT_REQUIRED_REASON = "autopilot_pr_cleanup_evidence_required"
KANBAN_WORKER_DONE_REVIEW_REQUIRED_REASON = "kanban_worker_done_review_evidence_required"
KANBAN_REVIEW_READY_FINAL_CLOSE_REQUIRED_REASON = "kanban_review_ready_final_close_required"
_DAILYCHINGU_REPO_NAMES = frozenset({"dailychingu", "daily-chingu", "dc"})
_RELEASE_ONLY_BASES = frozenset({"prod", "production"})
_PR_LESS_CLOSEOUT_EXCEPTIONS = frozenset({
    "read_only_no_code_audit",
    "evidence_only_linear_cleanup",
    "explicit_direct_landing_approval",
    "recorded_stacked_pr_contract",
})
_PENDING_CLOSEOUT_MARKERS = frozenset({
    "pending",
    "pending_autopilot_pr_closeout_cleanup",
    "todo",
    "tbd",
    "none",
})
_PR_URL_RE = re.compile(r"^https://github\.com/(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+)/pull/\d+/?$")
_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$", re.IGNORECASE)
_BRANCH_SAFE_RE = re.compile(r"[^A-Za-z0-9._/-]+")


def resolve_autopilot_integration_branch(
    *,
    repo_name: Optional[str] = None,
    remote_default_branch: Optional[str] = None,
    repo_policy: Optional[Mapping[str, Any]] = None,
) -> str:
    """Return the PR base branch AUTOPILOT closeout should target.

    DailyChingu integrates into ``develop`` by policy; release/prod promotion to
    ``main``/``prod`` stays outside the normal AUTOPILOT implementation closeout.
    Other repos use explicit policy first, then live remote default branch.
    """

    policy = repo_policy or {}
    normalized_repo = str(repo_name or "").strip().lower().replace("_", "-")
    if normalized_repo in _DAILYCHINGU_REPO_NAMES:
        return "develop"
    explicit = str(policy.get("integration_branch") or "").strip()
    if explicit:
        return explicit
    default_branch = str(remote_default_branch or policy.get("default_branch") or "").strip()
    return default_branch or "main"


def _truthy_evidence(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "ok", "passed", "pass", "clean", "done", "verified"}


def _nested_mapping(source: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = source.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _normalize_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _kanban_run_payload(evidence: Mapping[str, Any]) -> dict[str, Any]:
    payload = _nested_mapping(evidence, "kanban_run", "task_run", "run")
    if not payload:
        payload = dict(evidence)
    return payload


def _kanban_task_payload(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return _nested_mapping(evidence, "kanban_task", "task")


def _kanban_worker_done(evidence: Mapping[str, Any]) -> tuple[bool, list[str]]:
    run = _kanban_run_payload(evidence)
    task = _kanban_task_payload(evidence)
    run_status = _normalize_token(run.get("status") or evidence.get("task_run_status"))
    run_outcome = _normalize_token(run.get("outcome") or evidence.get("task_run_outcome"))
    task_status = _normalize_token(task.get("status") or evidence.get("kanban_task_status"))
    missing: list[str] = []
    done = False
    if run_status in {"done", "released"} and run_outcome in {"completed", "complete", "success", "succeeded"}:
        done = True
    elif task_status == "done" and run_outcome in {"completed", "complete", "success", "succeeded"}:
        done = True
    if not run_status:
        missing.append("task_run.status")
    if not run_outcome:
        missing.append("task_run.outcome")
    return done, missing


def evaluate_kanban_closeout_gate(
    *,
    evidence: Optional[Mapping[str, Any]] = None,
    repo_name: Optional[str] = None,
    remote_default_branch: Optional[str] = None,
    repo_policy: Optional[Mapping[str, Any]] = None,
    expected_repo_full_name: Optional[str] = None,
    expected_work_id: Optional[str] = None,
    release_approved: bool = False,
    final_close_approved: bool = False,
) -> dict[str, Any]:
    """Map Kanban worker completion to review-ready/closed without false Done.

    Kanban worker completion is only a run-ledger fact.  Review readiness still
    requires the same PR/evidence/cleanup gate as Linear AUTOPILOT closeout, and
    final ``closed`` requires an explicit caller-side close approval.  This helper
    never mutates Linear or treats ``tasks.status=done`` as project Done.
    """

    closeout = dict(evidence or {})
    nested_closeout = closeout.get("review_closeout") or closeout.get("closeout")
    if isinstance(nested_closeout, Mapping):
        closeout = {**dict(nested_closeout), **{k: v for k, v in closeout.items() if k not in {"review_closeout", "closeout"}}}

    worker_done, worker_missing = _kanban_worker_done(closeout)
    result: dict[str, Any] = {
        "allowed": False,
        "status": "blocked",
        "phase": "worker_done" if worker_done else "worker_pending",
        "mode": KANBAN_CLOSEOUT_MODE,
        "worker_done": worker_done,
        "review_ready": False,
        "closed": False,
        "linear_done_mutated": False,
        "kanban_done_project_done": False,
        "missing": list(worker_missing),
        "violations": [],
    }
    if not worker_done:
        result["reason"] = "kanban_worker_completion_required"
        return result

    pr_gate = evaluate_autopilot_pr_closeout_gate(
        evidence=closeout,
        repo_name=repo_name or closeout.get("repo_name"),
        remote_default_branch=remote_default_branch or closeout.get("remote_default_branch"),
        repo_policy=repo_policy,
        expected_repo_full_name=expected_repo_full_name,
        expected_work_id=expected_work_id,
        release_approved=release_approved,
    )
    result["closeout_gate"] = pr_gate
    if not pr_gate.get("allowed"):
        result.update(
            {
                "reason": KANBAN_WORKER_DONE_REVIEW_REQUIRED_REASON,
                "missing": list(dict.fromkeys(result["missing"] + list(pr_gate.get("missing") or []))),
                "violations": list(pr_gate.get("violations") or []),
            }
        )
        return result

    result.update(
        {
            "status": "review_ready",
            "phase": "review_ready",
            "reason": KANBAN_REVIEW_READY_FINAL_CLOSE_REQUIRED_REASON,
            "review_ready": True,
            "missing": [],
            "violations": [],
        }
    )
    if not final_close_approved:
        return result

    result.update(
        {
            "allowed": True,
            "status": "closed",
            "phase": "closed",
            "reason": "kanban_final_close_approved_after_review_ready",
            "closed": True,
        }
    )
    return result


def evaluate_autopilot_pr_closeout_gate(
    *,
    mode: str = AUTOPILOT_CLOSEOUT_MODE,
    evidence: Optional[Mapping[str, Any]] = None,
    repo_name: Optional[str] = None,
    remote_default_branch: Optional[str] = None,
    repo_policy: Optional[Mapping[str, Any]] = None,
    expected_repo_full_name: Optional[str] = None,
    expected_work_id: Optional[str] = None,
    release_approved: bool = False,
) -> dict[str, Any]:
    """Fail closed unless AUTOPILOT closeout has PR and cleanup evidence.

    This is the deterministic guard to run immediately before any future Linear
    Done mutation for AUTOPILOT implementation work.  It does not create the PR,
    push branches, clean worktrees, or mutate Linear; it only decides whether the
    evidence is sufficient for the controller to proceed.
    """

    closeout = dict(evidence or {})
    nested_closeout = closeout.get("review_closeout")
    if isinstance(nested_closeout, Mapping):
        closeout = {**dict(nested_closeout), **{k: v for k, v in closeout.items() if k != "review_closeout"}}
    normalized_mode = str(mode or "").strip().lower()
    policy = repo_policy or {}
    expected_base = resolve_autopilot_integration_branch(
        repo_name=repo_name or closeout.get("repo_name"),
        remote_default_branch=remote_default_branch or closeout.get("remote_default_branch"),
        repo_policy=policy,
    )
    evidence_repo = str(closeout.get("repo_full_name") or closeout.get("repo") or "").strip()
    expected_repo = str(
        expected_repo_full_name
        or policy.get("github_repo")
        or policy.get("repo_full_name")
        or ""
    ).strip().lower()
    result: dict[str, Any] = {
        "allowed": False,
        "status": "blocked",
        "reason": AUTOPILOT_PR_CLOSEOUT_REQUIRED_REASON,
        "mode": normalized_mode,
        "requires_pr": normalized_mode == AUTOPILOT_CLOSEOUT_MODE,
        "expected_integration_branch": expected_base,
        "linear_done_mutated": False,
        "missing": [],
        "violations": [],
    }

    if normalized_mode != AUTOPILOT_CLOSEOUT_MODE:
        result.update(
            {
                "allowed": True,
                "status": "not_applicable",
                "reason": "non_autopilot_closeout_gate_not_applicable",
                "requires_pr": False,
            }
        )
        return result

    exception_type = (
        str(
            closeout.get("pr_less_exception")
            or closeout.get("direct_landing_exception")
            or closeout.get("closeout_exception")
            or ""
        )
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    if exception_type:
        result["requires_pr"] = False
        result["exception_type"] = exception_type
        exception_required_text = {
            "linear_evidence_comment_id": closeout.get("linear_evidence_comment_id")
            or closeout.get("linear_comment_id"),
            "cleanup_proof": closeout.get("cleanup_proof"),
            "verification_result": closeout.get("verification_result")
            or closeout.get("checks_result")
            or closeout.get("test_result"),
            "exception_proof": closeout.get("exception_proof")
            or closeout.get("approval_ref")
            or closeout.get("audit_ref"),
        }
        if exception_type in {"explicit_direct_landing_approval", "recorded_stacked_pr_contract"}:
            exception_required_text["commit"] = closeout.get("commit") or closeout.get("ref")
        if exception_type == "explicit_direct_landing_approval":
            exception_required_text["direct_landing_approval_id"] = (
                closeout.get("direct_landing_approval_id") or closeout.get("approval_id")
            )
        if exception_type == "recorded_stacked_pr_contract":
            exception_required_text["stacked_pr_contract"] = (
                closeout.get("stacked_pr_contract") or closeout.get("stacked_pr_url")
            )
        for field, value in exception_required_text.items():
            if not str(value or "").strip():
                result["missing"].append(field)
        if exception_type not in _PR_LESS_CLOSEOUT_EXCEPTIONS:
            result["violations"].append(f"unsupported_pr_less_exception:{exception_type}")
        if not _truthy_evidence(closeout.get("cleanup_done") or closeout.get("cleanup_status")):
            result["missing"].append("cleanup_done")
        cleanup_proof = str(exception_required_text.get("cleanup_proof") or "").strip().lower()
        if cleanup_proof in _PENDING_CLOSEOUT_MARKERS:
            result["violations"].append("cleanup_proof_pending")
        commit = str(exception_required_text.get("commit") or "").strip()
        if commit and not _SHA_RE.fullmatch(commit):
            result["violations"].append("commit_ref_not_sha_like")
        if result["missing"] or result["violations"]:
            return result
        result.update(
            {
                "allowed": True,
                "status": "allowed",
                "reason": "autopilot_pr_less_closeout_exception_satisfied",
            }
        )
        return result

    required_text_fields = {
        "commit": closeout.get("commit") or closeout.get("ref"),
        "repo_full_name": closeout.get("repo_full_name") or closeout.get("repo"),
        "remote_branch": closeout.get("remote_branch") or closeout.get("branch"),
        "task_branch": closeout.get("task_branch")
        or closeout.get("local_branch")
        or closeout.get("remote_branch")
        or closeout.get("branch"),
        "task_worktree_path": closeout.get("task_worktree_path")
        or closeout.get("worktree_path"),
        "pr_url": closeout.get("pr_url") or closeout.get("pull_request_url"),
        "pr_base": closeout.get("pr_base") or closeout.get("base_branch"),
        "pr_head": closeout.get("pr_head") or closeout.get("head_branch"),
        "linear_evidence_comment_id": closeout.get("linear_evidence_comment_id")
        or closeout.get("linear_comment_id"),
        "cleanup_proof": closeout.get("cleanup_proof"),
        "verification_result": closeout.get("verification_result")
        or closeout.get("checks_result")
        or closeout.get("test_result"),
    }
    for field, value in required_text_fields.items():
        if not str(value or "").strip():
            result["missing"].append(field)

    required_bool_fields = {
        "branch_pushed": closeout.get("branch_pushed"),
        "pr_created": closeout.get("pr_created"),
        "repo_verified": closeout.get("repo_verified"),
        "pr_verified": closeout.get("pr_verified"),
        "sha_verified": closeout.get("sha_verified") or closeout.get("repo_head_sha_verified"),
        "checks_passed": closeout.get("checks_passed"),
        "cleanup_done": closeout.get("cleanup_done") or closeout.get("cleanup_status"),
    }
    for field, value in required_bool_fields.items():
        if not _truthy_evidence(value):
            result["missing"].append(field)

    commit = str(required_text_fields["commit"] or "").strip()
    if commit and not _SHA_RE.fullmatch(commit):
        result["violations"].append("commit_ref_not_sha_like")

    pr_url = str(required_text_fields["pr_url"] or "").strip()
    pr_match = _PR_URL_RE.fullmatch(pr_url) if pr_url else None
    if pr_url and not pr_match:
        result["violations"].append("pr_url_not_github_pull_url")
    elif pr_match:
        actual_repo = f"{pr_match.group('owner')}/{pr_match.group('repo')}".lower()
        if not expected_repo:
            result["missing"].append("trusted_expected_repo_full_name")
        elif actual_repo != expected_repo:
            result["violations"].append(
                f"pr_url_repo_mismatch:expected={expected_repo}:actual={actual_repo}"
            )
        if expected_repo and evidence_repo and evidence_repo.lower() != expected_repo:
            result["violations"].append(
                f"repo_full_name_mismatch:expected={expected_repo}:actual={evidence_repo.lower()}"
            )

    pr_base = str(required_text_fields["pr_base"] or "").strip()
    if pr_base and pr_base != expected_base:
        result["violations"].append(f"wrong_pr_base:expected={expected_base}:actual={pr_base}")

    pr_head = str(required_text_fields["pr_head"] or "").strip()
    remote_branch = str(required_text_fields["remote_branch"] or "").strip()
    if pr_head and remote_branch and pr_head != remote_branch:
        result["violations"].append("pr_head_remote_branch_mismatch")

    task_branch = str(required_text_fields["task_branch"] or "").strip()
    work_id = str(expected_work_id or closeout.get("work_id") or "").strip().lower()
    if task_branch and work_id and work_id not in task_branch.lower():
        result["violations"].append(f"task_branch_not_card_scoped:{work_id}")

    cleanup_proof = str(required_text_fields["cleanup_proof"] or "").strip().lower()
    if cleanup_proof in _PENDING_CLOSEOUT_MARKERS:
        result["violations"].append("cleanup_proof_pending")

    if pr_base in _RELEASE_ONLY_BASES and not release_approved:
        result["violations"].append(f"release_base_requires_explicit_approval:{pr_base}")

    if result["missing"] or result["violations"]:
        return result

    result.update({"allowed": True, "status": "allowed", "reason": "autopilot_pr_cleanup_evidence_satisfied"})
    return result



class AutopilotParseError(ValueError):
    """Invalid /autopilot shape.  The caller must fail closed."""


class LinearIssueClient(Protocol):
    def fetch_issue(self, identifier: str) -> Mapping[str, Any]:
        """Return a Linear issue payload for *identifier* without mutating Linear."""

    def list_execution_ready_issues(
        self,
        *,
        team_key: str = "CH",
        state_name: str = "Execution Ready",
        limit: int = 100,
    ) -> Mapping[str, Any]:
        """Return live Linear queue candidates without mutating Linear."""


@dataclass(frozen=True)
class AutopilotCommand:
    """Deterministically parsed /autopilot command."""

    action: str
    target_id: Optional[str] = None
    raw_args: str = ""

    @property
    def read_only(self) -> bool:
        return self.action in READ_ONLY_ACTIONS


@dataclass(frozen=True)
class AutopilotResult:
    """Structured command result plus a gateway/CLI friendly message."""

    ok: bool
    command: Optional[AutopilotCommand]
    message: str
    decision: dict[str, Any]
    fail_closed: bool = False


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _utcnow().isoformat()


def _normalize_target_id(value: str) -> str:
    return str(value or "").strip().upper()


def _is_target_id(value: str) -> bool:
    return bool(_AUTOPILOT_TARGET_RE.fullmatch(_normalize_target_id(value)))


def parse_autopilot_args(raw_args: str) -> AutopilotCommand:
    """Parse accepted /autopilot shapes and reject everything else.

    Accepted shapes are intentionally narrow:
      - status
      - dry-run
      - ON
      - OFF
      - CH-123
      - status CH-123
      - dry-run CH-123

    A bare /autopilot is status for safe discoverability.  Unknown words,
    extra tokens, and ON/OFF with targets fail closed.
    """

    raw = str(raw_args or "").strip()
    tokens = raw.split()
    if not tokens:
        return AutopilotCommand(action=ACTION_STATUS, raw_args=raw)

    first = tokens[0]
    first_lower = first.lower()
    first_upper = first.upper()

    if first_lower == "status":
        if len(tokens) == 1:
            return AutopilotCommand(action=ACTION_STATUS, raw_args=raw)
        if len(tokens) == 2 and _is_target_id(tokens[1]):
            return AutopilotCommand(
                action=ACTION_STATUS,
                target_id=_normalize_target_id(tokens[1]),
                raw_args=raw,
            )
        raise AutopilotParseError(f"Invalid /autopilot status shape. Usage: {AUTOPILOT_USAGE}")

    if first_lower == "dry-run":
        if len(tokens) == 1:
            return AutopilotCommand(action=ACTION_DRY_RUN, raw_args=raw)
        if len(tokens) == 2 and _is_target_id(tokens[1]):
            return AutopilotCommand(
                action=ACTION_DRY_RUN,
                target_id=_normalize_target_id(tokens[1]),
                raw_args=raw,
            )
        raise AutopilotParseError(f"Invalid /autopilot dry-run shape. Usage: {AUTOPILOT_USAGE}")

    if first_upper == "ON":
        if len(tokens) == 1:
            return AutopilotCommand(action=ACTION_ENABLE, raw_args=raw)
        raise AutopilotParseError("/autopilot ON does not accept a target; admission stays separate.")

    if first_upper == "OFF":
        if len(tokens) == 1:
            return AutopilotCommand(action=ACTION_DISABLE, raw_args=raw)
        raise AutopilotParseError("/autopilot OFF does not accept a target.")

    if len(tokens) == 1 and _is_target_id(first):
        return AutopilotCommand(
            action=ACTION_ONE_SHOT,
            target_id=_normalize_target_id(first),
            raw_args=raw,
        )

    raise AutopilotParseError(f"Invalid /autopilot command. Usage: {AUTOPILOT_USAGE}")


class AutopilotStateStore:
    """Small Hermes-owned runtime-intent store for the controller surface."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else get_hermes_home() / AUTOPILOT_STATE_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _default_state() -> dict[str, Any]:
        return {
            "schema_version": AUTOPILOT_STATE_VERSION,
            "enabled": False,
            "updated_at": None,
            "updated_by": None,
            "source": "default_disabled_fail_closed",
        }

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._default_state()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            state = self._default_state()
            state["state_error"] = f"invalid_state_file:{type(exc).__name__}"
            return state
        if not isinstance(payload, dict):
            state = self._default_state()
            state["state_error"] = "invalid_state_payload"
            return state
        state = self._default_state()
        state.update(payload)
        enabled = payload.get("enabled", state["enabled"])
        if not isinstance(enabled, bool):
            state["enabled"] = False
            state["state_error"] = "invalid_enabled_type"
        else:
            state["enabled"] = enabled
        state["schema_version"] = AUTOPILOT_STATE_VERSION
        return state

    def status(self) -> dict[str, Any]:
        return self.load()

    def set_enabled(
        self,
        enabled: bool,
        *,
        actor: Optional[str] = None,
        source: str = "autopilot_command",
        now: Optional[datetime] = None,
    ) -> dict[str, Any]:
        state = self.load()
        state.update(
            {
                "schema_version": AUTOPILOT_STATE_VERSION,
                "enabled": bool(enabled),
                "updated_at": (now or _utcnow()).isoformat(),
                "updated_by": actor or "unknown",
                "source": source,
            }
        )
        self._write_atomic(state)
        return state

    def _write_atomic(self, payload: Mapping[str, Any]) -> None:
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(dict(payload), fh, ensure_ascii=False, indent=2)
                fh.write("\n")
            Path(tmp_name).replace(self.path)
        except Exception:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            finally:
                raise


class EnvLinearIssueClient:
    """Read-only Linear GraphQL client using LINEAR_API_KEY from the environment."""

    _ISSUE_FIELDS = """
        identifier
        title
        description
        priority
        url
        state { name type }
        parent { identifier title state { name type } }
        project { name }
        labels { nodes { name } }
    """
    _QUERY = f"""
    query AutopilotIssue($id: String!) {{
      issue(id: $id) {{
        {_ISSUE_FIELDS}
        children(first: 50) {{ nodes {{ {_ISSUE_FIELDS} }} }}
      }}
    }}
    """
    _QUEUE_QUERY = f"""
    query AutopilotExecutionReadyQueue($teamKey: String!, $stateName: String!, $first: Int!) {{
      teams(filter: {{ key: {{ eq: $teamKey }} }}, first: 1) {{
        nodes {{
          key
          name
          issues(first: $first, filter: {{ state: {{ name: {{ eq: $stateName }} }} }}) {{
            nodes {{ {_ISSUE_FIELDS} children(first: 50) {{ nodes {{ {_ISSUE_FIELDS} }} }} }}
          }}
        }}
      }}
    }}
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key if api_key is not None else os.getenv("LINEAR_API_KEY")

    def _post_graphql(
        self,
        query: str,
        variables: Mapping[str, Any],
        *,
        identifier: Optional[str] = None,
    ) -> Mapping[str, Any]:
        if not self.api_key:
            payload: dict[str, Any] = {
                "status": "unavailable",
                "reason": "LINEAR_API_KEY_missing",
            }
            if identifier:
                payload["identifier"] = identifier
            return payload
        body = json.dumps({"query": query, "variables": dict(variables)}).encode("utf-8")
        request = urllib.request.Request(
            "https://api.linear.app/graphql",
            data=body,
            headers={
                "Authorization": self.api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            payload = {
                "status": "unavailable",
                "reason": f"linear_http_{exc.code}",
            }
            if identifier:
                payload["identifier"] = identifier
            return payload
        except Exception as exc:
            payload = {
                "status": "unavailable",
                "reason": f"linear_query_{type(exc).__name__}",
            }
            if identifier:
                payload["identifier"] = identifier
            return payload

    def fetch_issue(self, identifier: str) -> Mapping[str, Any]:
        payload = self._post_graphql(self._QUERY, {"id": identifier}, identifier=identifier)
        if payload.get("status") == "unavailable":
            return payload

        errors = payload.get("errors") if isinstance(payload, dict) else None
        if errors:
            return {
                "status": "unavailable",
                "reason": "linear_graphql_error",
                "identifier": identifier,
                "errors": errors,
            }
        issue = ((payload.get("data") or {}).get("issue") if isinstance(payload, dict) else None)
        if not issue:
            return {
                "status": "missing",
                "reason": "linear_issue_not_found",
                "identifier": identifier,
            }
        issue["status"] = "ok"
        return issue

    def list_execution_ready_issues(
        self,
        *,
        team_key: str = "CH",
        state_name: str = "Execution Ready",
        limit: int = 100,
    ) -> Mapping[str, Any]:
        payload = self._post_graphql(
            self._QUEUE_QUERY,
            {"teamKey": team_key, "stateName": state_name, "first": int(limit)},
        )
        if payload.get("status") == "unavailable":
            return payload
        errors = payload.get("errors") if isinstance(payload, dict) else None
        if errors:
            return {
                "status": "unavailable",
                "reason": "linear_graphql_error",
                "errors": errors,
            }
        teams = (((payload.get("data") or {}).get("teams") or {}).get("nodes") if isinstance(payload, dict) else None)
        if not teams:
            return {
                "status": "missing",
                "reason": "linear_team_not_found",
                "team_key": team_key,
                "issues": [],
            }
        team = teams[0]
        issues = (((team.get("issues") or {}).get("nodes")) if isinstance(team, Mapping) else None)
        return {
            "status": "ok",
            "team_key": team.get("key") or team_key,
            "team_name": team.get("name"),
            "state_name": state_name,
            "issues": issues if isinstance(issues, list) else [],
        }


def classify_linear_target(issue: Mapping[str, Any], requested_id: str) -> dict[str, Any]:
    """Classify one Linear issue as parent/child/standalone without mutation."""

    status = str(issue.get("status") or "ok")
    if status != "ok":
        return {
            "status": status,
            "reason": issue.get("reason") or status,
            "identifier": issue.get("identifier") or requested_id,
            "shape": "unknown",
            "execution_ready": False,
        }

    children = issue.get("children") or {}
    child_nodes = children.get("nodes") if isinstance(children, Mapping) else []
    child_nodes = child_nodes if isinstance(child_nodes, list) else []
    parent = issue.get("parent") if isinstance(issue.get("parent"), Mapping) else None
    if child_nodes:
        shape = "parent"
    elif parent:
        shape = "child"
    else:
        shape = "standalone"

    state = issue.get("state") if isinstance(issue.get("state"), Mapping) else {}
    state_name = str(state.get("name") or "").strip()
    state_type = str(state.get("type") or "").strip()
    return {
        "status": "ok",
        "identifier": issue.get("identifier") or requested_id,
        "title": issue.get("title") or "",
        "shape": shape,
        "state_name": state_name,
        "state_type": state_type,
        "execution_ready": state_name.lower() == "execution ready",
        "parent": {
            "identifier": parent.get("identifier"),
            "state_name": ((parent.get("state") or {}).get("name") if parent else None),
        }
        if parent
        else None,
        "children": [
            {
                "identifier": child.get("identifier"),
                "state_name": ((child.get("state") or {}).get("name") if isinstance(child, Mapping) else None),
            }
            for child in child_nodes
            if isinstance(child, Mapping)
        ],
    }


def summarize_work_state(
    work_state_store: Any = None,
    *,
    repo_policy: Optional[Mapping[str, Any]] = None,
    expected_repo_full_name: Optional[str] = None,
) -> dict[str, Any]:
    """Return read-only work_state summary for /autopilot status/dry-run."""

    try:
        from gateway.work_state import LIVE_STATES

        store = work_state_store
        if store is None:
            from gateway.work_state import WorkStateStore

            store = WorkStateStore()
        records = store.list_records()
        counts_by_state: dict[str, int] = {}
        delegated_omx_live = 0
        hermes_owned_live = 0
        for record in records:
            state = str(getattr(record, "state", "") or "")
            counts_by_state[state] = counts_by_state.get(state, 0) + 1
            if state in LIVE_STATES and getattr(record, "owner", None) == "hermes":
                hermes_owned_live += 1
                if (
                    getattr(record, "executor", None) == "omx"
                    and getattr(record, "mode", None) == "delegated"
                ):
                    delegated_omx_live += 1
        closeout = _autopilot_closeout_snapshot_from_records(
            records,
            repo_policy=repo_policy,
            expected_repo_full_name=expected_repo_full_name,
        )
        return {
            "available": True,
            "records_total": len(records),
            "counts_by_state": counts_by_state,
            "hermes_owned_live": hermes_owned_live,
            "delegated_omx_live": delegated_omx_live,
            "autopilot_closeout": closeout,
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": f"work_state_unavailable:{type(exc).__name__}",
        }


def classify_work_state_target(target_id: str, work_state_store: Any = None) -> dict[str, Any]:
    """Resolve one target against work_state and classify without side effects."""

    try:
        from gateway.work_state import classify_delegated_omx_supervisor_action

        store = work_state_store
        if store is None:
            from gateway.work_state import WorkStateStore

            store = WorkStateStore()
        resolution = store.resolve_delegated_signal_candidate(
            work_id=target_id,
            live_only=True,
        )
        result: dict[str, Any] = {
            "available": True,
            "resolution_status": resolution.get("status"),
            "resolution_reason": resolution.get("reason"),
            "matches_count": len(resolution.get("matches") or []),
        }
        record = resolution.get("record")
        if resolution.get("status") == "single_match" and record is not None:
            decision = classify_delegated_omx_supervisor_action(record).to_dict()
            result["supervisor_decision"] = decision
            try:
                from hermes_cli.kanban_work_state import project_work_state_to_kanban_run

                result["kanban_run_projection"] = project_work_state_to_kanban_run(
                    record,
                    resolution=resolution,
                    source="autopilot_work_state_target",
                )
            except Exception as exc:
                result["kanban_run_projection"] = {
                    "status": "fail_closed",
                    "reason": f"kanban_run_projection_unavailable:{type(exc).__name__}",
                }
        return result
    except Exception as exc:
        return {
            "available": False,
            "reason": f"work_state_target_unavailable:{type(exc).__name__}",
        }


def _empty_dry_run(status: str, reason: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "reason": reason,
        "selected_issue": None,
        "would_create_work_state": None,
        "would_goal_contract": None,
        "candidates": [],
        "groups": {"parents": {}, "projects": {}},
    }
    payload.update(extra)
    return payload


def _issue_identifier_sort_key(issue: Mapping[str, Any]) -> tuple[str, int, str]:
    identifier = str(issue.get("identifier") or "")
    match = re.fullmatch(r"([A-Z][A-Z0-9]*)-(\d+)", identifier.upper())
    if not match:
        return (identifier.upper(), 10**12, identifier.upper())
    return (match.group(1), int(match.group(2)), identifier.upper())


def _state_parts(issue: Mapping[str, Any]) -> tuple[str, str]:
    state = issue.get("state") if isinstance(issue.get("state"), Mapping) else {}
    return (str(state.get("name") or "").strip(), str(state.get("type") or "").strip())


def _is_terminal_state(state_name: str, state_type: str) -> bool:
    name = str(state_name or "").strip().lower()
    typ = str(state_type or "").strip().lower()
    return typ in {"completed", "canceled", "cancelled"} or name in {
        "done",
        "canceled",
        "cancelled",
        "duplicate",
        "won't do",
        "wont do",
    }


def _parent_payload(issue: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    parent = issue.get("parent")
    return parent if isinstance(parent, Mapping) else None


def _parent_identifier(issue: Mapping[str, Any], parent_override: Optional[str] = None) -> Optional[str]:
    if parent_override:
        return parent_override
    parent = _parent_payload(issue)
    if not parent:
        return None
    identifier = parent.get("identifier")
    return str(identifier) if identifier else None


def _project_name(issue: Mapping[str, Any]) -> Optional[str]:
    project = issue.get("project")
    if isinstance(project, Mapping) and project.get("name"):
        return str(project.get("name"))
    return None


def _description_text(issue: Mapping[str, Any]) -> str:
    return str(issue.get("description") or "")


def _has_done_when(issue: Mapping[str, Any]) -> bool:
    text = _description_text(issue).lower()
    return "done when" in text or "done_when" in text


def _has_verification(issue: Mapping[str, Any]) -> bool:
    text = _description_text(issue).lower()
    return "verification" in text or "verify" in text


def _work_state_backing_file_error(store: Any) -> Optional[str]:
    path_value = getattr(store, "path", None)
    if not path_value:
        return None
    try:
        path = Path(path_value)
    except TypeError:
        return "work_state_unavailable:invalid_path"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"work_state_unavailable:{type(exc).__name__}"
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            return "work_state_unavailable:invalid_record_item_type"
        return None
    if not isinstance(payload, dict):
        return "work_state_unavailable:invalid_payload_type"
    if set(payload.keys()) != {"records"}:
        return "work_state_unavailable:invalid_payload_schema"
    records = payload.get("records")
    if not isinstance(records, list):
        return "work_state_unavailable:invalid_records_type"
    if not all(isinstance(item, dict) for item in records):
        return "work_state_unavailable:invalid_record_item_type"
    return None


def _work_state_lock_snapshot(work_state_store: Any = None) -> dict[str, Any]:
    try:
        from gateway.work_state import LIVE_STATES

        store = work_state_store
        if store is None:
            from gateway.work_state import WorkStateStore

            store = WorkStateStore()
        backing_error = _work_state_backing_file_error(store)
        if backing_error:
            return {"available": False, "reason": backing_error, "active_work_ids": set()}
        work_ids: set[str] = set()
        for record in store.list_records():
            if getattr(record, "owner", None) != "hermes":
                continue
            state = str(getattr(record, "state", "") or "")
            if state not in LIVE_STATES:
                continue
            work_id = str(getattr(record, "work_id", "") or "").strip()
            if work_id:
                work_ids.add(work_id)
        return {"available": True, "active_work_ids": work_ids}
    except Exception as exc:
        return {
            "available": False,
            "reason": f"work_state_unavailable:{type(exc).__name__}",
            "active_work_ids": set(),
        }


def _is_autopilot_closeout_record(record: Any) -> bool:
    return (
        getattr(record, "owner", None) == "hermes"
        and getattr(record, "mode", None) == "autopilot"
        and getattr(record, "close_authority", None) == "autopilot_pr_closeout_gate"
    )


def _record_review_closeout_evidence(record: Any) -> dict[str, Any]:
    raw = getattr(record, "review_closeout", None)
    evidence = dict(raw) if isinstance(raw, Mapping) else {}
    work_id = str(getattr(record, "work_id", "") or "").strip()
    if work_id:
        evidence.setdefault("work_id", work_id)
    repo_path = getattr(record, "repo_path", None)
    if repo_path:
        evidence.setdefault("repo_path", repo_path)
    worktree_path = getattr(record, "worktree_path", None)
    if worktree_path:
        evidence.setdefault("worktree_path", worktree_path)
        evidence.setdefault("task_worktree_path", worktree_path)
    task_branch = getattr(record, "task_branch", None)
    if task_branch:
        evidence.setdefault("task_branch", task_branch)
    cleanup_proof = getattr(record, "cleanup_proof", None)
    if cleanup_proof:
        evidence.setdefault("cleanup_proof", cleanup_proof)
    return evidence


def _autopilot_closeout_snapshot_from_records(
    records: list[Any],
    *,
    repo_policy: Optional[Mapping[str, Any]] = None,
    expected_repo_full_name: Optional[str] = None,
) -> dict[str, Any]:
    blocked_cards: list[dict[str, Any]] = []
    review_ready_prs: list[dict[str, Any]] = []
    for record in records:
        if not _is_autopilot_closeout_record(record):
            continue
        evidence = _record_review_closeout_evidence(record)
        gate = evaluate_autopilot_pr_closeout_gate(
            evidence=evidence,
            repo_name=evidence.get("repo_name"),
            repo_policy=repo_policy,
            expected_repo_full_name=expected_repo_full_name,
            expected_work_id=str(getattr(record, "work_id", "") or ""),
        )
        entry = {
            "work_id": getattr(record, "work_id", None),
            "owner_session_id": getattr(record, "owner_session_id", None),
            "state": getattr(record, "state", None),
            "status": gate.get("status"),
            "reason": gate.get("reason"),
            "missing": list(gate.get("missing") or []),
            "violations": list(gate.get("violations") or []),
        }
        if gate.get("allowed"):
            entry.update(
                {
                    "pr_url": evidence.get("pr_url") or evidence.get("pull_request_url"),
                    "remote_branch": evidence.get("remote_branch") or evidence.get("branch"),
                    "commit": evidence.get("commit") or evidence.get("ref"),
                    "exception_type": gate.get("exception_type"),
                }
            )
            review_ready_prs.append(entry)
        else:
            blocked_cards.append(entry)
    return {
        "records_evaluated": len(blocked_cards) + len(review_ready_prs),
        "review_ready_prs": review_ready_prs,
        "blocked_cards": blocked_cards,
        "review_ready_count": len(review_ready_prs),
        "blocked_count": len(blocked_cards),
    }


def _closeout_snapshot_blocking_cards_for_target(
    closeout_snapshot: Mapping[str, Any],
    target_id: Optional[str],
) -> list[dict[str, Any]]:
    target = str(target_id or "").strip().upper()
    blocking_cards: list[dict[str, Any]] = []
    for card in closeout_snapshot.get("blocked_cards") or []:
        if not isinstance(card, Mapping):
            continue
        work_id = str(card.get("work_id") or "").strip().upper()
        if target and work_id == target:
            continue
        blocking_cards.append(dict(card))
    return blocking_cards


def _closeout_gate_blocked_dry_run(
    closeout_snapshot: Mapping[str, Any],
    *,
    blocking_cards: list[dict[str, Any]],
) -> dict[str, Any]:
    gated_snapshot = dict(closeout_snapshot)
    gated_snapshot["blocked_cards"] = blocking_cards
    gated_snapshot["blocked_count"] = len(blocking_cards)
    return _empty_dry_run(
        "blocked",
        "autopilot_closeout_review_gate_blocked",
        closeout_gate=gated_snapshot,
    )


def _autopilot_closeout_snapshot(
    work_state_store: Any = None,
    *,
    repo_policy: Optional[Mapping[str, Any]] = None,
    expected_repo_full_name: Optional[str] = None,
) -> dict[str, Any]:
    try:
        store = work_state_store
        if store is None:
            from gateway.work_state import WorkStateStore

            store = WorkStateStore()
        return {
            "available": True,
            **_autopilot_closeout_snapshot_from_records(
                store.list_records(),
                repo_policy=repo_policy,
                expected_repo_full_name=expected_repo_full_name,
            ),
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": f"work_state_unavailable:{type(exc).__name__}",
            "blocked_cards": [],
            "review_ready_prs": [],
            "blocked_count": 0,
            "review_ready_count": 0,
        }


_EXECUTOR_RESULT_CLOSEOUT_KEYS = frozenset({
    "commit",
    "ref",
    "repo_full_name",
    "repo",
    "repo_name",
    "remote_default_branch",
    "remote_branch",
    "branch",
    "task_branch",
    "local_branch",
    "task_worktree_path",
    "worktree_path",
    "branch_pushed",
    "pr_created",
    "repo_verified",
    "pr_verified",
    "sha_verified",
    "repo_head_sha_verified",
    "pr_url",
    "pull_request_url",
    "pr_base",
    "base_branch",
    "pr_head",
    "head_branch",
    "checks_passed",
    "checks_result",
    "test_result",
    "verification_result",
    "cleanup_done",
    "cleanup_status",
    "cleanup_proof",
    "linear_evidence_comment_id",
    "linear_comment_id",
    "kanban_run",
    "task_run",
    "run",
    "kanban_task",
    "task",
    "task_run_status",
    "task_run_outcome",
    "kanban_task_status",
})
_EXECUTOR_RESULT_CONTINUE_STATUSES = frozenset({
    "continue",
    "continuation",
    "needs_continuation",
    "same_card_continuation",
    "in_progress",
    "running",
    "progress",
})
_EXECUTOR_RESULT_FINISHED_STATUSES = frozenset({
    "executor_finished",
    "completed",
    "complete",
    "done",
    "finished",
    "succeeded",
    "success",
    "passed",
})
_EXECUTOR_RESULT_BLOCKED_STATUSES = frozenset({
    "blocked",
    "failed",
    "failure",
    "error",
    "stale",
    "handoff_needed",
    "retry_needed",
})


def _normalize_executor_result_status(executor_result: Mapping[str, Any]) -> str:
    raw = (
        executor_result.get("controller_event")
        or executor_result.get("autopilot_event")
        or executor_result.get("usable_outcome")
        or executor_result.get("outcome")
        or executor_result.get("status")
        or executor_result.get("state")
        or ""
    )
    normalized = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in _EXECUTOR_RESULT_FINISHED_STATUSES:
        return "executor_finished"
    if normalized in _EXECUTOR_RESULT_CONTINUE_STATUSES:
        return "same_card_continuation"
    if normalized in _EXECUTOR_RESULT_BLOCKED_STATUSES:
        return "blocked"
    if _truthy_evidence(executor_result.get("completed") or executor_result.get("done")):
        return "executor_finished"
    return "blocked" if normalized else "same_card_continuation"


def _result_closeout_evidence(executor_result: Mapping[str, Any]) -> dict[str, Any]:
    closeout: dict[str, Any] = {}
    nested = executor_result.get("review_closeout") or executor_result.get("closeout")
    if isinstance(nested, Mapping):
        closeout.update(dict(nested))
    for key in _EXECUTOR_RESULT_CLOSEOUT_KEYS:
        if key in executor_result and executor_result.get(key) is not None:
            closeout[key] = executor_result.get(key)
    return closeout


def _iter_autopilot_records(records: Iterable[Any]) -> list[Any]:
    return [record for record in records if _is_autopilot_closeout_record(record)]


def _record_identity_matches(record: Any, work_id: str, owner_session_id: Optional[str]) -> bool:
    if str(getattr(record, "work_id", "") or "").strip() != work_id:
        return False
    if owner_session_id is None:
        return True
    return str(getattr(record, "owner_session_id", "") or "").strip() == str(owner_session_id)


def ingest_autopilot_executor_result(
    *,
    work_state_store: Any,
    work_id: str,
    executor_result: Mapping[str, Any],
    owner_session_id: Optional[str] = None,
    now: Optional[datetime] = None,
    kanban_conn: Any = None,
) -> dict[str, Any]:
    """Store an executor result as controller input, never as controller completion.

    The executor is allowed to report facts.  It is not allowed to choose AUTOPILOT
    continuation, Linear Done, next-card admission, or direct landing.  Those are
    decided by ``autopilot_controller_tick`` from the persisted event plus trusted
    closeout gates.
    """

    if work_state_store is None or not hasattr(work_state_store, "list_records"):
        return {
            "status": "blocked",
            "reason": "work_state_store_missing_fail_closed",
            "work_id": work_id,
            "linear_done_mutated": False,
            "next_card_started": False,
        }
    normalized_work_id = str(work_id or "").strip()
    if not normalized_work_id:
        return {
            "status": "blocked",
            "reason": "work_id_missing_fail_closed",
            "work_id": None,
            "linear_done_mutated": False,
            "next_card_started": False,
        }
    if not isinstance(executor_result, Mapping):
        return {
            "status": "blocked",
            "reason": "executor_result_invalid_payload_fail_closed",
            "work_id": normalized_work_id,
            "linear_done_mutated": False,
            "next_card_started": False,
        }
    payload = dict(executor_result)
    try:
        all_records = work_state_store.list_records()
    except Exception as exc:
        return {
            "status": "blocked",
            "reason": f"work_state_unavailable:{type(exc).__name__}",
            "work_id": normalized_work_id,
            "linear_done_mutated": False,
            "next_card_started": False,
        }
    try:
        matches = [
            record
            for record in _iter_autopilot_records(all_records)
            if _record_identity_matches(record, normalized_work_id, owner_session_id)
        ]
    except Exception as exc:
        return {
            "status": "blocked",
            "reason": f"work_state_unavailable:{type(exc).__name__}",
            "work_id": normalized_work_id,
            "linear_done_mutated": False,
            "next_card_started": False,
        }
    if len(matches) != 1:
        return {
            "status": "blocked",
            "reason": "work_state_record_not_single_match",
            "work_id": normalized_work_id,
            "matches_count": len(matches),
            "linear_done_mutated": False,
            "next_card_started": False,
        }
    record = matches[0]
    event_at = now or _utcnow()
    controller_event = _normalize_executor_result_status(payload)
    closeout = dict(getattr(record, "review_closeout", None) or {})
    closeout.update(_result_closeout_evidence(payload))
    closeout["executor_event"] = {
        "status": controller_event,
        "raw_status": payload.get("status") or payload.get("outcome") or payload.get("state"),
        "proof": payload.get("proof") or payload.get("summary") or payload.get("message"),
        "next_action": bound_autopilot_next_action(
            payload.get("next_action"),
            fallback="Run deterministic AUTOPILOT controller tick for same-card decision.",
        ),
        "received_at": event_at.isoformat(),
        "linear_done_mutated": False,
        "next_card_started": False,
    }
    updates: dict[str, Any] = {
        "review_closeout": closeout,
        "last_progress_at": event_at,
    }
    if not hasattr(work_state_store, "update_record"):
        return {
            "status": "blocked",
            "reason": "work_state_update_missing_fail_closed",
            "work_id": normalized_work_id,
            "controller_event": controller_event,
            "linear_done_mutated": False,
            "next_card_started": False,
        }
    try:
        updated = work_state_store.update_record(
            normalized_work_id,
            str(getattr(record, "owner_session_id", "") or ""),
            **updates,
        )
    except Exception as exc:
        return {
            "status": "blocked",
            "reason": f"work_state_update_failed:{type(exc).__name__}",
            "work_id": normalized_work_id,
            "controller_event": controller_event,
            "linear_done_mutated": False,
            "next_card_started": False,
        }
    if updated is False:
        return {
            "status": "blocked",
            "reason": "work_state_update_failed_fail_closed",
            "work_id": normalized_work_id,
            "controller_event": controller_event,
            "linear_done_mutated": False,
            "next_card_started": False,
        }

    result = {
        "status": "ingested",
        "reason": "executor_result_recorded_as_controller_event",
        "work_id": normalized_work_id,
        "controller_event": controller_event,
        "linear_done_mutated": False,
        "next_card_started": False,
    }
    if kanban_conn is not None:
        executor_event = closeout.get("executor_event") if isinstance(closeout, Mapping) else None
        if not isinstance(executor_event, Mapping):
            executor_event = {}
        evidence_state = {
            "executor_finished": "worker_done",
            "same_card_continuation": "continuation_needed",
            "blocked": "blocked",
        }.get(controller_event, "blocked")
        evidence_record = {
            "work_id": normalized_work_id,
            "title": getattr(record, "title", None),
            "objective": getattr(record, "objective", None),
            "owner": getattr(record, "owner", None),
            "executor": getattr(record, "executor", None),
            "mode": getattr(record, "mode", None),
            "owner_session_id": getattr(record, "owner_session_id", None),
            "executor_session_id": getattr(record, "executor_session_id", None),
            "tmux_session": getattr(record, "tmux_session", None),
            "repo_path": getattr(record, "repo_path", None),
            "worktree_path": getattr(record, "worktree_path", None),
            "task_branch": getattr(record, "task_branch", None),
            "state": evidence_state,
            "proof": payload.get("proof") or payload.get("summary") or getattr(record, "proof", None),
            "next_action": executor_event.get("next_action") or getattr(record, "next_action", None),
            "usable_outcome": "usable_output" if controller_event == "executor_finished" else None,
            "close_disposition": "close" if controller_event == "executor_finished" else None,
            "blocked_reason": payload.get("blocked_reason") or getattr(record, "blocked_reason", None),
            "cleanup_required": getattr(record, "cleanup_required", None),
            "cleanup_proof": getattr(record, "cleanup_proof", None),
            "current_lane": getattr(record, "current_lane", None),
            "planning_gate": getattr(record, "planning_gate", None),
            "next_execution_branch": getattr(record, "next_execution_branch", None),
            "close_authority": getattr(record, "close_authority", None),
            "review_closeout": closeout,
        }
        try:
            from hermes_cli.kanban_work_state import ingest_work_state_run_evidence

            kanban_ingestion = ingest_work_state_run_evidence(
                kanban_conn,
                evidence_record,
                source="autopilot_executor_result",
            )
        except Exception as exc:
            return {
                "status": "blocked",
                "reason": f"kanban_run_ingestion_failed:{type(exc).__name__}",
                "work_id": normalized_work_id,
                "controller_event": controller_event,
                "linear_done_mutated": False,
                "next_card_started": False,
                "kanban_run_ingestion": {
                    "status": "fail_closed",
                    "reason": f"exception:{type(exc).__name__}",
                },
            }
        result["kanban_run_ingestion"] = kanban_ingestion
        if kanban_ingestion.get("status") not in {"ingested", "ingested_fail_closed"}:
            result["status"] = "blocked"
            result["reason"] = "kanban_run_ingestion_fail_closed"
    return result


def bound_autopilot_next_action(value: Any, *, fallback: str) -> str:
    text = " ".join(str(value or "").split()) or fallback
    return text if len(text) <= 200 else text[:197].rstrip() + "..."


def _executor_event(record: Any) -> Optional[Mapping[str, Any]]:
    closeout = getattr(record, "review_closeout", None)
    if not isinstance(closeout, Mapping):
        return None
    event = closeout.get("executor_event")
    return event if isinstance(event, Mapping) else None


def _executor_event_status(record: Any) -> Optional[str]:
    event = _executor_event(record)
    if isinstance(event, Mapping):
        return str(event.get("status") or "").strip() or None
    return None


def _controller_tick_blocked(reason: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "status": "blocked",
        "decision": "blocked",
        "reason": reason,
        "linear_done_mutated": False,
        "next_card_started": False,
    }
    payload.update(extra)
    return payload


def autopilot_controller_tick(
    *,
    state: Mapping[str, Any],
    work_state_store: Any,
    linear_client: LinearIssueClient,
    target_work_id: Optional[str] = None,
    repo_policy: Optional[Mapping[str, Any]] = None,
    expected_repo_full_name: Optional[str] = None,
) -> dict[str, Any]:
    """Deterministically decide AUTOPILOT's next controller action.

    This is intentionally pure-controller policy: no executor result can directly
    mark Linear Done, launch another card, or bless direct landing.  The tick
    chooses one of: same-card continuation, closeout verification, next-card
    selection, or blocked.
    """

    if work_state_store is None or not hasattr(work_state_store, "list_records"):
        return _controller_tick_blocked("work_state_store_missing_fail_closed")
    try:
        records = _iter_autopilot_records(work_state_store.list_records())
    except Exception as exc:
        return _controller_tick_blocked(f"work_state_unavailable:{type(exc).__name__}")
    target = str(target_work_id or "").strip().upper()
    if target:
        records = [record for record in records if str(getattr(record, "work_id", "") or "").upper() == target]
        if not records:
            return _controller_tick_blocked("target_work_state_not_found", target_work_id=target)

    if not records:
        return _controller_tick_blocked("executor_result_missing_fail_closed")

    for record in sorted(records, key=lambda item: str(getattr(item, "work_id", "") or "")):
        work_id = getattr(record, "work_id", None)
        event = _executor_event(record) or {}
        event_status = _executor_event_status(record)
        record_state = str(getattr(record, "state", "") or "")
        if not event_status:
            if record_state in {"created", "running", "stale", "retry_needed", "handoff_needed"}:
                return {
                    "status": "would_continue",
                    "decision": "same_card_continuation",
                    "reason": "awaiting_executor_result_same_card",
                    "work_id": work_id,
                    "next_action": event.get("next_action") or getattr(record, "next_action", None),
                    "linear_done_mutated": False,
                    "next_card_started": False,
                }
            return _controller_tick_blocked(
                "executor_result_missing_fail_closed",
                work_id=work_id,
                state=record_state,
            )
        if event_status == "same_card_continuation":
            return {
                "status": "would_continue",
                "decision": "same_card_continuation",
                "reason": "executor_result_requires_same_card_continuation",
                "work_id": work_id,
                "next_action": event.get("next_action") or getattr(record, "next_action", None),
                "linear_done_mutated": False,
                "next_card_started": False,
            }
        if event_status == "blocked":
            return _controller_tick_blocked(
                "executor_result_blocked_same_card",
                work_id=work_id,
                next_action=event.get("next_action") or getattr(record, "next_action", None),
            )
        if event_status == "executor_finished":
            evidence = _record_review_closeout_evidence(record)
            gate = evaluate_autopilot_pr_closeout_gate(
                evidence=evidence,
                repo_name=evidence.get("repo_name"),
                repo_policy=repo_policy,
                expected_repo_full_name=expected_repo_full_name,
                expected_work_id=str(work_id or ""),
            )
            if not gate.get("allowed"):
                return {
                    "status": "blocked",
                    "decision": "closeout_verification",
                    "reason": "autopilot_closeout_review_gate_blocked",
                    "work_id": work_id,
                    "closeout_gate": gate,
                    "linear_done_mutated": False,
                    "next_card_started": False,
                }
            continue
        return _controller_tick_blocked(
            "executor_event_unknown_fail_closed",
            work_id=work_id,
            executor_event=event_status,
            state=record_state,
        )

    command = AutopilotCommand(action=ACTION_DRY_RUN, raw_args="dry-run")
    dry_run = evaluate_dry_run_admission(
        command=command,
        state=state,
        linear_client=linear_client,
        work_state_store=work_state_store,
        enforce_closeout_gate=True,
        repo_policy=repo_policy,
        expected_repo_full_name=expected_repo_full_name,
    )
    if dry_run.get("status") == "would_run":
        return {
            "status": "would_select_next_card",
            "decision": "next_card_selection",
            "reason": "prior_closeout_verified_select_next_admitted_card",
            "selected_issue": dry_run.get("selected_issue"),
            "would_create_work_state": dry_run.get("would_create_work_state"),
            "would_goal_contract": dry_run.get("would_goal_contract"),
            "would_kanban_payload": dry_run.get("would_kanban_payload"),
            "linear_done_mutated": False,
            "next_card_started": False,
        }
    return _controller_tick_blocked(
        str(dry_run.get("reason") or "no_next_card_selected"),
        dry_run=dry_run,
    )


def _candidate_from_issue(
    issue: Mapping[str, Any],
    *,
    active_work_ids: set[str],
    parent_override: Optional[str] = None,
) -> dict[str, Any]:
    identifier = str(issue.get("identifier") or "").strip()
    classified = classify_linear_target(issue, identifier)
    shape = str(classified.get("shape") or "unknown")
    if parent_override and shape == "standalone":
        shape = "child"
    parent_id = _parent_identifier(issue, parent_override)
    candidate = {
        "identifier": identifier,
        "eligible": False,
        "reason": "unknown",
        "shape": shape,
        "parent_id": parent_id,
        "project": _project_name(issue),
    }
    if classified.get("status") != "ok":
        candidate["reason"] = str(classified.get("reason") or "linear_target_unavailable_fail_closed")
        return candidate

    state_name, state_type = _state_parts(issue)
    parent = _parent_payload(issue)
    parent_state = parent.get("state") if isinstance(parent, Mapping) else None
    parent_state_name = str((parent_state or {}).get("name") or "") if isinstance(parent_state, Mapping) else ""
    parent_state_type = str((parent_state or {}).get("type") or "") if isinstance(parent_state, Mapping) else ""

    if _is_terminal_state(state_name, state_type):
        candidate["reason"] = "terminal_state"
    elif state_name.lower() != "execution ready":
        candidate["reason"] = "state_not_execution_ready"
    elif parent and _is_terminal_state(parent_state_name, parent_state_type):
        candidate["reason"] = "parent_terminal"
    elif shape == "parent":
        candidate["reason"] = "parent_target_requires_child_selection"
    elif identifier in active_work_ids:
        candidate["reason"] = "active_work_state_lock"
    elif not _has_done_when(issue):
        candidate["reason"] = "missing_done_when"
    elif not _has_verification(issue):
        candidate["reason"] = "missing_verification"
    else:
        candidate["eligible"] = True
        candidate["reason"] = "eligible_execution_ready"
    return candidate


def _group_candidates(candidates: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    parents: dict[str, int] = {}
    projects: dict[str, int] = {}
    for candidate in candidates:
        parent_key = str(candidate.get("parent_id") or "standalone")
        project_key = str(candidate.get("project") or "none")
        parents[parent_key] = parents.get(parent_key, 0) + 1
        projects[project_key] = projects.get(project_key, 0) + 1
    return {"parents": parents, "projects": projects}


def _selected_issue_payload(issue: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "identifier": issue.get("identifier"),
        "title": issue.get("title") or "",
        "url": issue.get("url"),
    }


def _would_goal_contract(issue: Mapping[str, Any]) -> dict[str, Any]:
    identifier = str(issue.get("identifier") or "")
    title = str(issue.get("title") or "")
    summary = f"Execute Linear {identifier}: {title}".strip()
    try:
        contract = generate_goal_contract(issue, mode="single-card")
        return {
            "command": contract["prompt"],
            "summary": summary,
            "mode": contract["mode"],
        }
    except GoalContractError as exc:
        return {
            "command": f"/goal {summary}; verify Done when before closeout; preserve dry-run side-effect boundaries.",
            "summary": summary,
            "blocked_reason": exc.reason,
        }


def _would_work_state(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "work_id": candidate.get("identifier"),
        "owner": "hermes",
        "executor": "hermes",
        "mode": "autopilot",
        "state": "created",
        "parent_id": candidate.get("parent_id"),
    }


def _kanban_policy(repo_policy: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not isinstance(repo_policy, Mapping):
        return {}
    nested = repo_policy.get("kanban")
    return nested if isinstance(nested, Mapping) else {}


def _kanban_payload_dry_run_enabled(repo_policy: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(repo_policy, Mapping):
        return False
    nested = _kanban_policy(repo_policy)
    return any(
        _truthy_evidence(value)
        for value in (
            repo_policy.get("kanban_payload_dry_run"),
            repo_policy.get("enable_kanban_payload_dry_run"),
            nested.get("payload_dry_run"),
            nested.get("dry_run"),
        )
    )


def _kanban_native_selection_enabled(repo_policy: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(repo_policy, Mapping):
        return False
    nested = _kanban_policy(repo_policy)
    return any(
        _truthy_evidence(value)
        for value in (
            repo_policy.get("kanban_native_candidate_selection"),
            repo_policy.get("kanban_candidate_selection"),
            nested.get("native_candidate_selection"),
            nested.get("candidate_selection"),
        )
    )


def _policy_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(part).strip() for part in value if str(part).strip()]
    return []


def _approved_kanban_tenants(repo_policy: Optional[Mapping[str, Any]]) -> set[str]:
    nested = _kanban_policy(repo_policy)
    values: list[str] = []
    if isinstance(repo_policy, Mapping):
        for key in ("approved_kanban_tenants", "kanban_approved_tenants", "approved_domains"):
            values.extend(_policy_string_list(repo_policy.get(key)))
    for key in ("approved_tenants", "approved_domains", "native_approved_tenants"):
        values.extend(_policy_string_list(nested.get(key)))
    return {value.lower() for value in values if value}


def _kanban_task_value(task: Any, key: str, default: Any = None) -> Any:
    if isinstance(task, Mapping):
        return task.get(key, default)
    return getattr(task, key, default)


def _kanban_task_identifier(task: Any) -> str:
    return str(
        _kanban_task_value(task, "public_id")
        or _kanban_task_value(task, "id")
        or ""
    ).strip()


def _kanban_task_to_selected_issue(task: Any) -> dict[str, Any]:
    identifier = _kanban_task_identifier(task)
    task_id = str(_kanban_task_value(task, "id", "") or "").strip()
    payload = {
        "identifier": identifier,
        "title": str(_kanban_task_value(task, "title", "") or ""),
        "url": None,
        "source": "kanban",
        "task_id": task_id or identifier,
    }
    public_id = str(_kanban_task_value(task, "public_id", "") or "").strip()
    if public_id:
        payload["public_id"] = public_id
    return payload


def _kanban_task_candidate(
    task: Any,
    *,
    active_work_ids: set[str],
    approved_tenants: set[str],
) -> dict[str, Any]:
    identifier = _kanban_task_identifier(task)
    tenant = str(_kanban_task_value(task, "tenant", "") or "").strip()
    status = str(_kanban_task_value(task, "status", "") or "").strip().lower()
    review_phase = str(_kanban_task_value(task, "review_phase", "") or "").strip().lower()
    candidate = {
        "identifier": identifier,
        "eligible": False,
        "reason": "unknown",
        "shape": "kanban_native",
        "parent_id": None,
        "project": tenant or "none",
        "source": "kanban",
        "task_id": str(_kanban_task_value(task, "id", "") or "").strip() or identifier,
        "tenant": tenant or None,
    }
    if not identifier:
        candidate["reason"] = "missing_kanban_identifier"
    elif not tenant:
        candidate["reason"] = "missing_kanban_tenant"
    elif approved_tenants and tenant.lower() not in approved_tenants:
        candidate["reason"] = "kanban_tenant_not_approved"
    elif status != "ready":
        candidate["reason"] = "kanban_status_not_ready"
    elif review_phase in {"worker_done", "review_ready", "closed"}:
        candidate["reason"] = "kanban_review_phase_not_executable"
    elif identifier in active_work_ids or candidate["task_id"] in active_work_ids:
        candidate["reason"] = "active_work_state_lock"
    else:
        candidate["eligible"] = True
        candidate["reason"] = "eligible_kanban_ready"
    return candidate


def _kanban_goal_contract(task: Any) -> dict[str, Any]:
    identifier = _kanban_task_identifier(task)
    title = str(_kanban_task_value(task, "title", "") or "")
    summary = f"Execute Kanban {identifier}: {title}".strip()
    body = str(_kanban_task_value(task, "body", "") or "")
    command = (
        f"/goal {summary}; source=kanban-native; verify Done when before closeout; "
        "preserve review gates and do not admit a next card on budget exhaustion."
    )
    if body:
        command += f"\n\nKanban task body:\n{body}"
    return {"command": command, "summary": summary, "mode": "kanban-native"}


def _list_kanban_native_ready_tasks(tenant: Optional[str] = None) -> list[Any]:
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        kb.init_db(conn)
        return kb.list_tasks(conn, status="ready", tenant=tenant, limit=100)
    finally:
        conn.close()


def _finish_kanban_native_dry_run(
    tasks: list[Any],
    *,
    active_work_ids: set[str],
    repo_policy: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    approved_tenants = _approved_kanban_tenants(repo_policy)
    candidates = [
        _kanban_task_candidate(task, active_work_ids=active_work_ids, approved_tenants=approved_tenants)
        for task in tasks
    ]
    groups = _group_candidates(candidates)
    for task, candidate in zip(tasks, candidates):
        if candidate.get("eligible"):
            goal_contract = _kanban_goal_contract(task)
            selected = _kanban_task_to_selected_issue(task)
            work_state = _would_work_state(candidate)
            work_state.update({"mode": "kanban", "source": "kanban", "task_id": candidate.get("task_id")})
            return {
                "status": "would_run",
                "reason": "kanban_native_ready_task_selected",
                "selected_issue": selected,
                "would_create_work_state": work_state,
                "would_goal_contract": goal_contract,
                "candidates": candidates,
                "groups": groups,
                "source": "kanban",
                "side_effects": {
                    "work_state_written": False,
                    "executor_spawned": False,
                    "linear_mutated": False,
                    "kanban_task_claimed": False,
                },
            }
    return {
        "status": "blocked",
        "reason": "no_eligible_kanban_native_task",
        "selected_issue": None,
        "would_create_work_state": None,
        "would_goal_contract": None,
        "candidates": candidates,
        "groups": groups,
        "source": "kanban",
    }


def _first_policy_value(
    repo_policy: Optional[Mapping[str, Any]],
    keys: Iterable[str],
    *,
    nested_keys: Iterable[str] = (),
) -> Optional[Any]:
    nested = _kanban_policy(repo_policy)
    for key in nested_keys:
        value = nested.get(key)
        if value not in (None, ""):
            return value
    if isinstance(repo_policy, Mapping):
        for key in keys:
            value = repo_policy.get(key)
            if value not in (None, ""):
                return value
    return None


def _repo_name_from_full_name(repo_full_name: Optional[str]) -> Optional[str]:
    value = str(repo_full_name or "").strip()
    if not value:
        return None
    return value.rstrip("/").split("/")[-1] or None


def _worktree_branch_intent(issue: Mapping[str, Any], repo_policy: Optional[Mapping[str, Any]]) -> str:
    identifier = str(issue.get("identifier") or "").strip().upper()
    title = str(issue.get("title") or "").strip().lower()
    slug = _BRANCH_SAFE_RE.sub("-", title.replace(" ", "-")).strip("-/")[:48]
    fallback = f"autopilot/{identifier.lower()}" + (f"-{slug}" if slug else "")
    template = _first_policy_value(
        repo_policy,
        ("worktree_branch_template", "branch_template"),
        nested_keys=("worktree_branch_template", "branch_template"),
    )
    if not template:
        return fallback
    try:
        branch = str(template).format(identifier=identifier, issue=identifier, title_slug=slug)
    except Exception:
        return fallback
    branch = _BRANCH_SAFE_RE.sub("-", branch).strip("-/")
    return branch or fallback


def _skill_hints(repo_policy: Optional[Mapping[str, Any]]) -> list[str]:
    raw = _first_policy_value(repo_policy, ("skills",), nested_keys=("skills",))
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(raw, (list, tuple, set)):
        return [str(part).strip() for part in raw if str(part).strip()]
    return []


def _linear_comments_snapshot(issue: Mapping[str, Any]) -> list[dict[str, Any]]:
    comments = issue.get("comments")
    nodes = comments.get("nodes") if isinstance(comments, Mapping) else []
    if not isinstance(nodes, list):
        return []
    result: list[dict[str, Any]] = []
    for comment in nodes[:5]:
        if not isinstance(comment, Mapping):
            continue
        result.append(
            {
                "body": str(comment.get("body") or "")[:500],
                "created_at": comment.get("createdAt") or comment.get("created_at"),
            }
        )
    return result


def _resolve_kanban_tenant(issue: Mapping[str, Any], repo_policy: Optional[Mapping[str, Any]]) -> Optional[str]:
    explicit = _first_policy_value(repo_policy, ("tenant",), nested_keys=("tenant",))
    if explicit:
        return str(explicit).strip()
    try:
        from hermes_cli.linear_kanban_shadow import tenant_for_linear_project

        return tenant_for_linear_project(_project_name(issue))
    except Exception:
        return None


def _would_kanban_payload(
    issue: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    goal_contract: Mapping[str, Any],
    repo_policy: Optional[Mapping[str, Any]],
    expected_repo_full_name: Optional[str],
) -> dict[str, Any]:
    identifier = str(issue.get("identifier") or "").strip().upper()
    source_card = {
        "identifier": identifier,
        "title": issue.get("title") or "",
        "url": issue.get("url"),
        "state": (issue.get("state") or {}) if isinstance(issue.get("state"), Mapping) else {},
        "parent_identifier": candidate.get("parent_id"),
        "project": _project_name(issue),
    }
    repo_full_name = str(
        expected_repo_full_name
        or _first_policy_value(repo_policy, ("github_repo", "repo_full_name", "repo"))
        or ""
    ).strip()
    executor = str(
        _first_policy_value(repo_policy, ("executor",), nested_keys=("executor",))
        or ""
    ).strip()
    tenant = _resolve_kanban_tenant(issue, repo_policy)
    missing: list[str] = []
    if not isinstance(repo_policy, Mapping) or not repo_policy:
        missing.append("policy")
    if not repo_full_name:
        missing.append("repo_full_name")
    if not _has_done_when(issue):
        missing.append("done_when")
    if not executor:
        missing.append("executor")
    if not tenant:
        missing.append("tenant")
    idempotency_key = f"linear:{identifier}" if identifier else ""
    base_branch = resolve_autopilot_integration_branch(
        repo_name=_repo_name_from_full_name(repo_full_name),
        repo_policy=repo_policy,
    )
    worktree_branch = _worktree_branch_intent(issue, repo_policy)
    profile = _first_policy_value(repo_policy, ("profile",), nested_keys=("profile",))
    skills = _skill_hints(repo_policy)
    body_source_payload = {
        "source": "linear",
        "identifier": identifier,
        "url": issue.get("url"),
        "idempotency_key": idempotency_key,
        "tenant": tenant,
        "autopilot": {
            "mode": "dry_run_admission",
            "linear_is_ssot": True,
            "executor_dispatch": "forbidden_in_dry_run",
            "kanban_done_projection": "forbidden",
        },
        "repo_intent": {
            "repo_full_name": repo_full_name or None,
            "base_branch": base_branch,
            "worktree_branch": worktree_branch,
        },
        "goal_contract": {
            "summary": goal_contract.get("summary"),
            "mode": goal_contract.get("mode"),
        },
    }
    body = "\n".join(
        [
            f"Autopilot Kanban dry-run for Linear `{identifier}`.",
            "",
            "Linear remains the source of truth. This payload is preview-only: "
            "do not dispatch an executor, mutate Linear, or project Kanban Done back to Linear.",
            "",
            "```json source_payload",
            json.dumps(body_source_payload, indent=2, sort_keys=True, ensure_ascii=False),
            "```",
        ]
    )
    payload: dict[str, Any] = {
        "status": "would_create" if not missing else "blocked",
        "reason": "kanban_payload_contract_ready"
        if not missing
        else "kanban_payload_contract_missing_required_fields",
        "dry_run": True,
        "missing": missing,
        "source_card": source_card,
        "goal_contract": {
            "summary": goal_contract.get("summary"),
            "mode": goal_contract.get("mode"),
        },
        "repo_intent": {
            "repo_full_name": repo_full_name or None,
            "base_branch": base_branch,
            "worktree_branch": worktree_branch,
        },
        "execution_hints": {
            "executor": executor or None,
            "profile": str(profile).strip() if profile else None,
            "skills": skills,
        },
        "task": {
            "title": f"{identifier} — {issue.get('title') or ''}".strip(),
            "body": body,
            "tenant": tenant,
            "idempotency_key": idempotency_key,
            "workspace_kind": "worktree",
            "status": "triage",
            "assignee": None,
            "skills": skills,
        },
        "dependencies": [
            {
                "source": "linear_parent",
                "identifier": candidate.get("parent_id"),
            }
        ]
        if candidate.get("parent_id")
        else [],
        "comments": _linear_comments_snapshot(issue),
        "events": [
            {
                "kind": "autopilot_kanban_payload_dry_run",
                "payload": {
                    "source": "linear",
                    "identifier": identifier,
                    "idempotency_key": idempotency_key,
                },
            }
        ],
        "task_runs_metadata": {
            "source": "linear_autopilot",
            "linear_identifier": identifier,
            "idempotency_key": idempotency_key,
            "repo_full_name": repo_full_name or None,
            "base_branch": base_branch,
            "worktree_branch": worktree_branch,
            "executor": executor or None,
            "profile": str(profile).strip() if profile else None,
        },
        "side_effects": {
            "kanban_task_written": False,
            "executor_spawned": False,
            "linear_done_mutated": False,
            "kanban_done_projected_to_linear": False,
        },
    }
    return payload


def _finish_dry_run(
    issues: list[Mapping[str, Any]],
    *,
    active_work_ids: set[str],
    parent_override: Optional[str] = None,
    no_eligible_reason: str = "no_eligible_execution_ready_issue",
    repo_policy: Optional[Mapping[str, Any]] = None,
    expected_repo_full_name: Optional[str] = None,
) -> dict[str, Any]:
    sorted_issues = sorted(issues, key=_issue_identifier_sort_key)
    candidates = [
        _candidate_from_issue(
            issue,
            active_work_ids=active_work_ids,
            parent_override=parent_override,
        )
        for issue in sorted_issues
    ]
    groups = _group_candidates(candidates)
    for issue, candidate in zip(sorted_issues, candidates):
        if candidate.get("eligible"):
            goal_contract = _would_goal_contract(issue)
            if _kanban_payload_dry_run_enabled(repo_policy):
                kanban_payload = _would_kanban_payload(
                    issue,
                    candidate,
                    goal_contract=goal_contract,
                    repo_policy=repo_policy,
                    expected_repo_full_name=expected_repo_full_name,
                )
                if kanban_payload.get("status") != "would_create":
                    return {
                        "status": "blocked",
                        "reason": str(
                            kanban_payload.get("reason")
                            or "kanban_payload_contract_missing_required_fields"
                        ),
                        "selected_issue": _selected_issue_payload(issue),
                        "would_create_work_state": None,
                        "would_goal_contract": goal_contract,
                        "would_kanban_payload": kanban_payload,
                        "candidates": candidates,
                        "groups": groups,
                    }
            else:
                kanban_payload = None
            payload = {
                "status": "would_run",
                "reason": "dry_run_selected_execution_ready_issue",
                "selected_issue": _selected_issue_payload(issue),
                "would_create_work_state": _would_work_state(candidate),
                "would_goal_contract": goal_contract,
                "candidates": candidates,
                "groups": groups,
            }
            if kanban_payload is not None:
                payload["would_kanban_payload"] = kanban_payload
            return payload
    return {
        "status": "blocked",
        "reason": no_eligible_reason,
        "selected_issue": None,
        "would_create_work_state": None,
        "would_goal_contract": None,
        "candidates": candidates,
        "groups": groups,
    }


def evaluate_dry_run_admission(
    *,
    command: AutopilotCommand,
    state: Mapping[str, Any],
    linear_client: LinearIssueClient,
    work_state_store: Any = None,
    target_issue: Optional[Mapping[str, Any]] = None,
    target_classification: Optional[Mapping[str, Any]] = None,
    enforce_closeout_gate: bool = False,
    repo_policy: Optional[Mapping[str, Any]] = None,
    expected_repo_full_name: Optional[str] = None,
    kanban_task_provider: Optional[Callable[[], list[Any]]] = None,
) -> dict[str, Any]:
    """Produce CH-388 read-only admission proof for /autopilot dry-run."""

    if command.action != ACTION_DRY_RUN:
        return _empty_dry_run("blocked", "not_a_dry_run")

    lock_snapshot = _work_state_lock_snapshot(work_state_store)
    if not lock_snapshot.get("available"):
        return _empty_dry_run(
            "blocked",
            "work_state_unavailable_fail_closed",
            work_state_reason=lock_snapshot.get("reason"),
        )
    active_ids = lock_snapshot.get("active_work_ids")
    active_ids = active_ids if isinstance(active_ids, set) else set()
    if not command.target_id:
        if not bool(state.get("enabled")):
            return _empty_dry_run("paused", "controller_disabled_noop")
        closeout_snapshot = _autopilot_closeout_snapshot(
            work_state_store,
            repo_policy=repo_policy,
            expected_repo_full_name=expected_repo_full_name,
        )
        if not closeout_snapshot.get("available"):
            return _empty_dry_run(
                "blocked",
                "work_state_unavailable_fail_closed",
                work_state_reason=closeout_snapshot.get("reason"),
            )
        if closeout_snapshot.get("blocked_cards"):
            return _closeout_gate_blocked_dry_run(
                closeout_snapshot,
                blocking_cards=list(closeout_snapshot.get("blocked_cards") or []),
            )
        if _kanban_native_selection_enabled(repo_policy):
            try:
                tasks = (kanban_task_provider or _list_kanban_native_ready_tasks)()
            except Exception as exc:
                return _empty_dry_run(
                    "blocked",
                    "kanban_native_queue_unavailable_fail_closed",
                    kanban_reason=type(exc).__name__,
                )
            kanban_dry_run = _finish_kanban_native_dry_run(
                [task for task in tasks if task is not None],
                active_work_ids=active_ids,
                repo_policy=repo_policy,
            )
            if kanban_dry_run.get("status") == "would_run":
                return kanban_dry_run
            if not _truthy_evidence((_kanban_policy(repo_policy)).get("fallback_to_linear")):
                return kanban_dry_run
        queue = linear_client.list_execution_ready_issues(
            team_key="CH",
            state_name="Execution Ready",
            limit=100,
        )
        if queue.get("status") != "ok":
            return _empty_dry_run(
                "blocked",
                "linear_queue_unavailable_fail_closed",
                linear_reason=queue.get("reason") or queue.get("status"),
            )
        issues = queue.get("issues") if isinstance(queue.get("issues"), list) else []
        return _finish_dry_run(
            issues,
            active_work_ids=active_ids,
            repo_policy=repo_policy,
            expected_repo_full_name=expected_repo_full_name,
        )

    if enforce_closeout_gate:
        closeout_snapshot = _autopilot_closeout_snapshot(
            work_state_store,
            repo_policy=repo_policy,
            expected_repo_full_name=expected_repo_full_name,
        )
        if not closeout_snapshot.get("available"):
            return _empty_dry_run(
                "blocked",
                "work_state_unavailable_fail_closed",
                work_state_reason=closeout_snapshot.get("reason"),
            )
        blocking_cards = _closeout_snapshot_blocking_cards_for_target(
            closeout_snapshot,
            command.target_id,
        )
        if blocking_cards:
            return _closeout_gate_blocked_dry_run(
                closeout_snapshot,
                blocking_cards=blocking_cards,
            )

    target_payload = target_issue or linear_client.fetch_issue(command.target_id)
    if target_payload.get("status") != "ok":
        return _empty_dry_run(
            "blocked",
            "linear_target_unavailable_fail_closed",
            linear_reason=target_payload.get("reason") or target_payload.get("status"),
        )
    classified = target_classification or classify_linear_target(target_payload, command.target_id)
    if classified.get("shape") == "parent":
        state_name, state_type = _state_parts(target_payload)
        if _is_terminal_state(state_name, state_type):
            return _empty_dry_run("blocked", "parent_terminal")
        children = target_payload.get("children") or {}
        child_nodes = children.get("nodes") if isinstance(children, Mapping) else []
        child_issues = [child for child in child_nodes if isinstance(child, Mapping)]
        if not child_issues:
            return _empty_dry_run("blocked", "no_eligible_child")
        return _finish_dry_run(
            child_issues,
            active_work_ids=active_ids,
            parent_override=str(classified.get("identifier") or command.target_id),
            no_eligible_reason="no_eligible_child",
            repo_policy=repo_policy,
            expected_repo_full_name=expected_repo_full_name,
        )
    return _finish_dry_run(
        [target_payload],
        active_work_ids=active_ids,
        repo_policy=repo_policy,
        expected_repo_full_name=expected_repo_full_name,
    )


def _admission_decision(
    *,
    command: AutopilotCommand,
    state: Mapping[str, Any],
    linear: Optional[Mapping[str, Any]],
    dry_run: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    if command.action == ACTION_DRY_RUN and dry_run:
        return {
            "status": str(dry_run.get("status") or "blocked"),
            "reason": str(dry_run.get("reason") or "dry_run_admission_unavailable"),
            "admission_bypassed": False,
        }
    if command.action in READ_ONLY_ACTIONS:
        return {
            "status": "read_only",
            "reason": f"{command.action}_never_starts_executor",
            "admission_bypassed": False,
        }
    if command.action == ACTION_ENABLE:
        return {
            "status": "controller_intent_enabled",
            "reason": "future_automatic_starts_still_require_admission",
            "admission_bypassed": False,
        }
    if command.action == ACTION_DISABLE:
        return {
            "status": "controller_intent_disabled",
            "reason": "new_automatic_starts_prevented",
            "admission_bypassed": False,
        }
    if command.action == ACTION_ONE_SHOT:
        if not linear or linear.get("status") != "ok":
            return {
                "status": "blocked",
                "reason": "linear_target_unavailable_fail_closed",
                "admission_bypassed": False,
            }
        if linear.get("shape") == "parent":
            return {
                "status": "requires_child_selection",
                "reason": "parent_target_requires_execution_ready_child_admission",
                "admission_bypassed": False,
            }
        if not linear.get("execution_ready"):
            return {
                "status": "blocked",
                "reason": "linear_target_not_execution_ready",
                "admission_bypassed": False,
            }
        return {
            "status": "eligible_for_admission",
            "reason": "one_shot_target_requires_admission_before_executor_start",
            "admission_bypassed": False,
        }
    return {
        "status": "blocked",
        "reason": "unknown_autopilot_action_fail_closed",
        "admission_bypassed": False,
    }


def _format_bool(value: Any) -> str:
    return "ON" if bool(value) else "OFF"


def _format_result_message(decision: Mapping[str, Any]) -> str:
    command = decision.get("command") or {}
    action = command.get("action") or "unknown"
    target_id = command.get("target_id")
    state = decision.get("state") or {}
    admission = decision.get("admission") or {}
    lines = [f"/autopilot {action.replace('_', '-')}" + (f" {target_id}" if target_id else "")]
    lines.append(f"Controller intent: {_format_bool(state.get('enabled'))}")
    if action in {ACTION_STATUS, ACTION_DRY_RUN}:
        lines.append("Mode: read-only — no executor spawn, no Linear mutation.")
    if action == ACTION_ENABLE:
        lines.append("ON recorded: automatic controller intent is enabled; admission is still mandatory.")
    elif action == ACTION_DISABLE:
        lines.append("OFF recorded: new automatic starts are prevented.")
    elif action == ACTION_ONE_SHOT:
        lines.append("One-shot target classified; executor spawn is not performed by this entrypoint.")
    if target_id:
        linear = decision.get("linear") or {}
        shape = linear.get("shape") or "unknown"
        linear_status = linear.get("status") or "unknown"
        state_name = linear.get("state_name") or linear.get("reason") or "unknown"
        lines.append(f"Linear target: {target_id} ({shape}, {linear_status}, {state_name})")
    work_state = decision.get("work_state") or {}
    if target_id:
        lines.append(
            "work_state: "
            f"{work_state.get('resolution_status', 'unavailable')}"
            f"/{work_state.get('resolution_reason', work_state.get('reason', 'unknown'))}"
        )
    else:
        if work_state.get("available"):
            lines.append(
                "work_state: "
                f"{work_state.get('hermes_owned_live', 0)} Hermes live, "
                f"{work_state.get('delegated_omx_live', 0)} delegated OMX live."
            )
            closeout = work_state.get("autopilot_closeout") or {}
            lines.append(
                "autopilot review: "
                f"{closeout.get('review_ready_count', 0)} review-ready, "
                f"{closeout.get('blocked_count', 0)} blocked."
            )
            blocked_cards = closeout.get("blocked_cards") or []
            for card in blocked_cards[:3]:
                lines.append(
                    "blocked review: "
                    f"{card.get('work_id')} "
                    f"missing={','.join(card.get('missing') or []) or '-'} "
                    f"violations={','.join(card.get('violations') or []) or '-'}"
                )
        else:
            lines.append(f"work_state: {work_state.get('reason', 'unavailable')}")
    dry_run = decision.get("dry_run") or {}
    if dry_run:
        lines.append(
            "Dry-run admission: "
            f"{dry_run.get('status', 'unknown')} "
            f"({dry_run.get('reason', 'no_reason')})."
        )
        selected = dry_run.get("selected_issue") or {}
        if selected:
            lines.append(
                "Selected issue: "
                f"{selected.get('identifier')} — {selected.get('title', '')}".rstrip()
            )
        else:
            lines.append("Selected issue: none")
        would_lock = dry_run.get("would_create_work_state") or {}
        if would_lock:
            lines.append(f"Would-lock id: {would_lock.get('work_id')}")
        else:
            lines.append("Would-lock id: none")
        would_goal = dry_run.get("would_goal_contract") or {}
        if would_goal:
            lines.append(f"Would-contract: {would_goal.get('summary')}")
        else:
            lines.append("Would-contract: none")
        would_kanban = dry_run.get("would_kanban_payload") or {}
        if would_kanban:
            task = would_kanban.get("task") or {}
            lines.append(
                "Would-kanban: "
                f"{would_kanban.get('status')} "
                f"{task.get('idempotency_key') or '-'} "
                f"tenant={task.get('tenant') or '-'}"
            )
        if dry_run.get("status") == "would_run":
            lines.append("Controller would: run after admission materializes the lock/goal contract.")
        elif dry_run.get("status") == "paused":
            lines.append("Controller would: pause; OFF intent prevents queue admission.")
        else:
            lines.append("Controller would: refuse/no-op until the blocker is resolved.")
        lines.append("Dry-run side effects: work_state_written=false, executor_spawned=false, linear_mutated=false.")
    lines.append(
        "Decision: "
        f"{admission.get('status', 'unknown')} "
        f"({admission.get('reason', 'no_reason')})."
    )
    lines.append("Side effects: executor_spawned=false, linear_done_mutated=false.")
    return "\n".join(lines)


def _format_runtime_result_message(command: AutopilotCommand, materialization: Mapping[str, Any]) -> str:
    action = command.action.replace("_", "-")
    target = f" {command.target_id}" if command.target_id else ""
    lines = [f"/autopilot {action}{target}"]
    lines.append("Mode: execution — live admission must materialize lock/goal before executor kickoff.")
    lines.append(
        "Decision: "
        f"{materialization.get('status', 'unknown')} "
        f"({materialization.get('reason', 'no_reason')})."
    )
    selected = materialization.get("selected_issue") or {}
    if selected:
        lines.append(
            "Selected issue: "
            f"{selected.get('identifier')} — {selected.get('title', '')}".rstrip()
        )
    else:
        lines.append("Selected issue: none")
    work_id = materialization.get("work_id")
    owner_session_id = materialization.get("owner_session_id")
    if work_id:
        lines.append(f"work_state lock: {work_id} owner_session={owner_session_id}")
    else:
        lines.append("work_state lock: none")
    executor_session_id = materialization.get("executor_session_id")
    if executor_session_id:
        lines.append(f"Executor session: {executor_session_id}")
    else:
        lines.append("Executor session: none")
    lines.append(
        "Side effects: "
        f"work_state_written={str(bool(materialization.get('work_state_written'))).lower()}, "
        f"executor_spawned={str(bool(materialization.get('executor_spawned'))).lower()}, "
        f"linear_done_mutated={str(bool(materialization.get('linear_done_mutated'))).lower()}."
    )
    if materialization.get("status") == "started":
        lines.append(
            "Closeout guard: Linear Done remains forbidden until executor evidence, "
            "verified PR to the repo integration branch, and cleanup proof satisfy the closeout gate."
        )
    else:
        dry_run = materialization.get("dry_run") or {}
        blocker = dry_run.get("reason") or materialization.get("reason")
        candidates = dry_run.get("candidates") if isinstance(dry_run, Mapping) else None
        if candidates:
            for candidate in candidates[:5]:
                candidate_id = candidate.get("identifier") or "unknown"
                candidate_reason = candidate.get("reason") or "unknown"
                eligible = str(bool(candidate.get("eligible"))).lower()
                lines.append(
                    f"Admission candidate: {candidate_id} eligible={eligible} reason={candidate_reason}"
                )
        if blocker:
            lines.append(f"Blocked/no-op reason: {blocker}")
    return "\n".join(lines)


def run_autopilot_once(
    *,
    state_store: Optional[AutopilotStateStore] = None,
    work_state_store: Any = None,
    linear_client: Optional[LinearIssueClient] = None,
    executor_spawner: Optional[Callable[..., Any]] = None,
    actor: Optional[str] = None,
    owner_session_id: Optional[str] = None,
    target_id: Optional[str] = None,
    repo_policy: Optional[Mapping[str, Any]] = None,
    expected_repo_full_name: Optional[str] = None,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Materialize at most one admitted autopilot task.

    This is the CH-389 controller boundary below the slash-command entrypoint:
    it reuses dry-run admission, writes exactly one work_state lock, then calls a
    caller-supplied executor spawner with the generated `/goal` contract.  It
    never mutates Linear Done and never loops to a second task.
    """

    event_at = now or _utcnow()
    store = state_store or AutopilotStateStore()
    state = store.status()
    if not target_id and not bool(state.get("enabled")):
        return {
            "status": "paused",
            "reason": "controller_disabled_noop",
            "selected_issue": None,
            "work_id": None,
            "owner_session_id": owner_session_id,
            "work_state_written": False,
            "executor_spawned": False,
            "linear_done_mutated": False,
            "generated_at": event_at.isoformat(),
        }
    if executor_spawner is None:
        return {
            "status": "blocked",
            "reason": "executor_spawner_missing_fail_closed",
            "selected_issue": None,
            "work_id": None,
            "owner_session_id": owner_session_id,
            "work_state_written": False,
            "executor_spawned": False,
            "linear_done_mutated": False,
            "generated_at": event_at.isoformat(),
        }

    client = linear_client or EnvLinearIssueClient()
    command = AutopilotCommand(
        action=ACTION_DRY_RUN,
        target_id=target_id.upper() if target_id else None,
        raw_args=(f"dry-run {target_id.upper()}" if target_id else "dry-run"),
    )
    dry_run = evaluate_dry_run_admission(
        command=command,
        state=state,
        linear_client=client,
        work_state_store=work_state_store,
        enforce_closeout_gate=True,
        repo_policy=repo_policy,
        expected_repo_full_name=expected_repo_full_name,
    )
    if dry_run.get("status") != "would_run":
        return {
            "status": str(dry_run.get("status") or "blocked"),
            "reason": str(dry_run.get("reason") or "admission_refused"),
            "selected_issue": dry_run.get("selected_issue"),
            "work_id": None,
            "owner_session_id": owner_session_id,
            "dry_run": dry_run,
            "work_state_written": False,
            "executor_spawned": False,
            "linear_done_mutated": False,
            "generated_at": event_at.isoformat(),
        }

    selected_issue = dry_run.get("selected_issue") or {}
    work_id = str(selected_issue.get("identifier") or "").strip()
    if not work_id:
        return {
            "status": "blocked",
            "reason": "selected_issue_missing_identifier_fail_closed",
            "selected_issue": selected_issue,
            "work_id": None,
            "owner_session_id": owner_session_id,
            "dry_run": dry_run,
            "work_state_written": False,
            "executor_spawned": False,
            "linear_done_mutated": False,
            "generated_at": event_at.isoformat(),
        }
    owner_session = owner_session_id or f"autopilot:{work_id}"
    goal = dry_run.get("would_goal_contract") or {}
    goal_contract = str(goal.get("command") or "")
    if not goal_contract.startswith("/goal"):
        return {
            "status": "blocked",
            "reason": "goal_contract_unavailable_fail_closed",
            "selected_issue": selected_issue,
            "work_id": work_id,
            "owner_session_id": owner_session,
            "dry_run": dry_run,
            "work_state_written": False,
            "executor_spawned": False,
            "linear_done_mutated": False,
            "generated_at": event_at.isoformat(),
        }

    from gateway.work_state import WorkRecord

    record = WorkRecord(
        work_id=work_id,
        title=str(selected_issue.get("title") or work_id),
        objective=str(goal.get("summary") or f"Execute Linear {work_id}"),
        owner="hermes",
        executor="hermes",
        mode="autopilot",
        owner_session_id=owner_session,
        state="created",
        started_at=event_at,
        last_progress_at=event_at,
        next_action="Spawn selected executor from generated /goal contract",
        proof="work_state lock created before executor spawn",
        close_authority="autopilot_pr_closeout_gate",
        cleanup_required=True,
        cleanup_proof="pending_autopilot_pr_closeout_cleanup",
        review_closeout={
            "status": "pending",
            "required": True,
            "work_id": work_id,
            "required_evidence": [
                "repo_full_name",
                "task_worktree_path",
                "task_branch",
                "commit_sha",
                "pushed_branch",
                "pr_url",
                "pr_base",
                "pr_head",
                "checks_passed",
                "verification_result",
                "linear_evidence_comment_id",
                "cleanup_proof",
            ],
        },
    )
    if work_state_store is None or not hasattr(work_state_store, "upsert"):
        return {
            "status": "blocked",
            "reason": "work_state_store_missing_upsert_fail_closed",
            "selected_issue": selected_issue,
            "work_id": work_id,
            "owner_session_id": owner_session,
            "dry_run": dry_run,
            "work_state_written": False,
            "executor_spawned": False,
            "linear_done_mutated": False,
            "generated_at": event_at.isoformat(),
        }
    work_state_store.upsert(record)

    executor_dry_run = dict(dry_run)
    # CH-410's Kanban payload is a preview/admission contract only. Do not pass
    # it into the live executor-spawner path, where an integration could mistake
    # the preview for authority to create a Kanban task or dispatch from it.
    executor_dry_run.pop("would_kanban_payload", None)

    try:
        spawn_result = executor_spawner(
            goal_contract=goal_contract,
            work_id=work_id,
            owner_session_id=owner_session,
            selected_issue=selected_issue,
            actor=actor,
            dry_run=executor_dry_run,
        )
    except Exception as exc:  # fail closed and leave an auditable blocked record
        blocked_reason = f"executor_spawn_failed:{type(exc).__name__}:{exc}"
        if hasattr(work_state_store, "update_record"):
            work_state_store.update_record(
                work_id,
                owner_session,
                state="blocked",
                last_progress_at=event_at,
                blocked_reason=blocked_reason,
                proof=blocked_reason,
                usable_outcome="blocked",
                close_disposition="close",
                next_action="Executor spawn failed; operator review required before retry",
            )
        return {
            "status": "blocked",
            "reason": "executor_spawn_failed",
            "selected_issue": selected_issue,
            "work_id": work_id,
            "owner_session_id": owner_session,
            "dry_run": dry_run,
            "work_state_written": True,
            "executor_spawned": False,
            "linear_done_mutated": False,
            "error": blocked_reason,
            "generated_at": event_at.isoformat(),
        }

    spawn_payload = spawn_result if isinstance(spawn_result, Mapping) else {"result": spawn_result}
    executor_session_id = spawn_payload.get("session_id") or spawn_payload.get("executor_session_id")
    proof = str(spawn_payload.get("proof") or "executor spawned from generated /goal contract")
    repo_path = spawn_payload.get("repo_path")
    worktree_path = spawn_payload.get("worktree_path") or spawn_payload.get("task_worktree_path")
    task_branch = (
        spawn_payload.get("task_branch")
        or spawn_payload.get("branch")
        or spawn_payload.get("local_branch")
        or spawn_payload.get("remote_branch")
    )
    review_closeout = {
        **record.review_closeout,
        "status": "pending_review_artifacts",
        "repo_path": repo_path,
        "task_worktree_path": worktree_path,
        "worktree_path": worktree_path,
        "task_branch": task_branch,
    }
    review_closeout = {key: value for key, value in review_closeout.items() if value is not None}
    if hasattr(work_state_store, "update_record"):
        work_state_store.update_record(
            work_id,
            owner_session,
            state="running",
            executor_session_id=str(executor_session_id) if executor_session_id else None,
            repo_path=str(repo_path) if repo_path else None,
            worktree_path=str(worktree_path) if worktree_path else None,
            task_branch=str(task_branch) if task_branch else None,
            review_closeout=review_closeout,
            proof=proof,
            last_progress_at=event_at,
            next_action="Await executor evidence; do not mark Linear Done without verification",
        )
    return {
        "status": "started",
        "reason": "one_task_autopilot_started",
        "selected_issue": selected_issue,
        "work_id": work_id,
        "owner_session_id": owner_session,
        "executor_session_id": executor_session_id,
        "dry_run": dry_run,
        "work_state_written": True,
        "executor_spawned": True,
        "linear_done_mutated": False,
        "spawn_result": spawn_payload,
        "generated_at": event_at.isoformat(),
    }


def handle_autopilot_command(
    raw_args: str,
    *,
    actor: Optional[str] = None,
    state_store: Optional[AutopilotStateStore] = None,
    work_state_store: Any = None,
    linear_client: Optional[LinearIssueClient] = None,
    executor_spawner: Optional[Callable[..., Any]] = None,
    repo_policy: Optional[Mapping[str, Any]] = None,
    expected_repo_full_name: Optional[str] = None,
    now: Optional[datetime] = None,
) -> AutopilotResult:
    """Handle /autopilot without spawning executors or mutating Linear.

    ``executor_spawner`` is accepted only as a test seam/documented guard: this
    entrypoint must never call it in CH-385.
    """

    del executor_spawner  # CH-385 guard: parsing/classification only.
    store = state_store or AutopilotStateStore()
    try:
        command = parse_autopilot_args(raw_args)
    except AutopilotParseError as exc:
        return AutopilotResult(
            ok=False,
            command=None,
            fail_closed=True,
            message=f"Fail-closed: {exc}\nUsage: {AUTOPILOT_USAGE}",
            decision={
                "ok": False,
                "fail_closed": True,
                "reason": str(exc),
                "usage": AUTOPILOT_USAGE,
                "side_effects": {
                    "state_written": False,
                    "executor_spawned": False,
                    "linear_done_mutated": False,
                },
            },
        )

    state_written = False
    if command.action == ACTION_ENABLE:
        state = store.set_enabled(True, actor=actor, now=now)
        state_written = True
    elif command.action == ACTION_DISABLE:
        state = store.set_enabled(False, actor=actor, now=now)
        state_written = True
    else:
        state = store.status()

    linear: Optional[dict[str, Any]] = None
    target_issue: Optional[Mapping[str, Any]] = None
    client = linear_client or EnvLinearIssueClient()
    if command.target_id:
        target_issue = client.fetch_issue(command.target_id)
        linear = dict(classify_linear_target(target_issue, command.target_id))
        work_state = classify_work_state_target(command.target_id, work_state_store)
    else:
        work_state = summarize_work_state(
            work_state_store,
            repo_policy=repo_policy,
            expected_repo_full_name=expected_repo_full_name,
        )

    dry_run: Optional[dict[str, Any]] = None
    if command.action == ACTION_DRY_RUN:
        dry_run = evaluate_dry_run_admission(
            command=command,
            state=state,
            linear_client=client,
            work_state_store=work_state_store,
            target_issue=target_issue,
            target_classification=linear,
            repo_policy=repo_policy,
            expected_repo_full_name=expected_repo_full_name,
        )

    admission = _admission_decision(command=command, state=state, linear=linear, dry_run=dry_run)
    decision: dict[str, Any] = {
        "ok": True,
        "fail_closed": False,
        "command": {
            "action": command.action,
            "target_id": command.target_id,
            "read_only": command.read_only,
        },
        "state": state,
        "linear": linear,
        "work_state": work_state,
        "admission": admission,
        "side_effects": {
            "state_written": state_written,
            "executor_spawned": False,
            "linear_done_mutated": False,
        },
        "generated_at": (now or _utcnow()).isoformat(),
    }
    if dry_run is not None:
        decision["dry_run"] = dry_run
    return AutopilotResult(
        ok=True,
        command=command,
        message=_format_result_message(decision),
        decision=decision,
    )


def handle_autopilot_runtime_command(
    raw_args: str,
    *,
    actor: Optional[str] = None,
    state_store: Optional[AutopilotStateStore] = None,
    work_state_store: Any = None,
    linear_client: Optional[LinearIssueClient] = None,
    executor_spawner: Optional[Callable[..., Any]] = None,
    owner_session_id: Optional[str] = None,
    repo_policy: Optional[Mapping[str, Any]] = None,
    expected_repo_full_name: Optional[str] = None,
    now: Optional[datetime] = None,
) -> AutopilotResult:
    """Handle the live operator /autopilot path.

    Read-only/status/OFF commands keep the CH-385 fail-closed controller
    behavior.  Execution commands (``ON`` and targeted one-shot ``CH-123``)
    must traverse ``run_autopilot_once`` so the user-facing entrypoint proves
    the same path as the materialization helper: live admission -> work_state
    lock -> generated /goal contract -> executor spawner.  Linear Done remains
    forbidden here.
    """

    try:
        command = parse_autopilot_args(raw_args)
    except AutopilotParseError as exc:
        return handle_autopilot_command(
            raw_args,
            actor=actor,
            state_store=state_store,
            work_state_store=work_state_store,
            linear_client=linear_client,
            repo_policy=repo_policy,
            expected_repo_full_name=expected_repo_full_name,
            now=now,
        )

    if command.action in {ACTION_STATUS, ACTION_DRY_RUN, ACTION_DISABLE}:
        return handle_autopilot_command(
            raw_args,
            actor=actor,
            state_store=state_store,
            work_state_store=work_state_store,
            linear_client=linear_client,
            repo_policy=repo_policy,
            expected_repo_full_name=expected_repo_full_name,
            now=now,
        )

    store = state_store or AutopilotStateStore()
    state_written = False
    if command.action == ACTION_ENABLE:
        store.set_enabled(True, actor=actor, now=now)
        state_written = True

    materialization = run_autopilot_once(
        state_store=store,
        work_state_store=work_state_store,
        linear_client=linear_client,
        executor_spawner=executor_spawner,
        actor=actor,
        owner_session_id=owner_session_id,
        target_id=command.target_id,
        repo_policy=repo_policy,
        expected_repo_full_name=expected_repo_full_name,
        now=now,
    )
    decision: dict[str, Any] = {
        "ok": materialization.get("status") == "started",
        "fail_closed": materialization.get("status") != "started",
        "command": {
            "action": command.action,
            "target_id": command.target_id,
            "read_only": False,
        },
        "materialization": materialization,
        "side_effects": {
            "state_written": state_written,
            "work_state_written": bool(materialization.get("work_state_written")),
            "executor_spawned": bool(materialization.get("executor_spawned")),
            "linear_done_mutated": bool(materialization.get("linear_done_mutated")),
        },
        "generated_at": (now or _utcnow()).isoformat(),
    }
    return AutopilotResult(
        ok=bool(decision["ok"]),
        command=command,
        fail_closed=bool(decision["fail_closed"]),
        message=_format_runtime_result_message(command, materialization),
        decision=decision,
    )


def plugin_command_entrypoint(raw_args: str) -> str:
    """Plugin-compatible fail-closed slash-command entrypoint."""

    return handle_autopilot_command(raw_args).message

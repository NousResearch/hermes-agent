"""Controller-owned Kanban completion integrity (Phase 1A + bounded 1C).

Workers make claims. The controller verifies them. Worker-supplied metadata
is never authoritative merely because the worker supplied it.

This module is SQLite / modular-monolith only: no queue, workflow engine,
or generalized policy platform. Legacy tasks without a completion contract
or semantic gate keep existing behaviour.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

CONTRACT_GIT_REVISION = "git_revision"
CONTRACT_REVIEW = "review"
CONTRACT_HISTORIAN_CERTIFY = "historian_certify"

GATE_REVIEW_APPROVED = "review_approved"

APPROVAL_VERDICTS = frozenset({"APPROVED", "PASS"})
NEGATIVE_VERDICTS = frozenset({"REJECTED", "CHANGES_REQUIRED", "FAIL"})
KNOWN_VERDICTS = APPROVAL_VERDICTS | NEGATIVE_VERDICTS

STATUS_AWAITING = "awaiting_verification"
STATUS_VERIFIED = "verified"
STATUS_REJECTED = "verification_rejected"
STATUS_ERROR = "verification_error"
STATUS_RECONCILE = "needs_reconciliation"

REASON_SATISFIED = "satisfied"
REASON_UNSATISFIED = "unsatisfied"
REASON_INVALID = "invalid"
REASON_PARENT_NOT_TERMINAL = "PARENT_NOT_TERMINAL"
REASON_MISSING_TERMINAL_RESULT = "MISSING_TERMINAL_RESULT"
REASON_NARRATIVE_ONLY = "NARRATIVE_ONLY"
REASON_WRONG_ATTEMPT = "WRONG_ATTEMPT"
REASON_MISSING_ATTEMPT = "MISSING_ATTEMPT"
REASON_CONFLICTING_RESULT = "CONFLICTING_TERMINAL_RESULT"
REASON_SHA_SYNTAX = "SHA_SYNTAX"
REASON_OBJECT_MISSING = "OBJECT_MISSING"
REASON_OBJECT_NOT_COMMIT = "OBJECT_NOT_COMMIT"
REASON_WRONG_REPOSITORY = "WRONG_REPOSITORY"
REASON_WRONG_WORKTREE = "WRONG_WORKTREE"
REASON_BASE_MISMATCH = "BASE_MISMATCH"
REASON_COMMIT_COUNT = "COMMIT_COUNT"
REASON_REMOTE_MISMATCH = "REMOTE_MISMATCH"
REASON_DIRTY_WORKTREE = "DIRTY_WORKTREE"
REASON_VERDICT_MISSING = "VERDICT_MISSING"
REASON_VERDICT_MALFORMED = "VERDICT_MALFORMED"
REASON_VERDICT_NOT_APPROVAL = "VERDICT_NOT_APPROVAL"
REASON_SHA_MISMATCH = "SHA_MISMATCH"
REASON_GATE_INVALID = "GATE_INVALID"
REASON_NEEDS_RECONCILIATION = "NEEDS_RECONCILIATION"
REASON_VERIFICATION_REJECTED = "VERIFICATION_REJECTED"
REASON_VERIFICATION_ERROR = "VERIFICATION_ERROR"
REASON_HEAD_NOT_REQUIRED = "HEAD_NOT_REQUIRED"

KNOWN_DIRTY_POLICIES = frozenset({"observe", "reject"})

_GIT_TIMEOUT_SECONDS = 15
_GIT_NO_REPLACE_ENV = "GIT_NO_REPLACE_OBJECTS"


class CompletionIntegrityError(ValueError):
    """Fail-closed governance refusal with a structured reason code."""

    def __init__(self, code: str, message: str, *, details: Optional[dict] = None):
        self.code = code
        self.details = details or {}
        super().__init__(message)


@dataclass(frozen=True)
class GateEvaluation:
    state: str
    code: str
    message: str
    details: dict

    @property
    def satisfied(self) -> bool:
        return self.state == REASON_SATISFIED


@dataclass(frozen=True)
class GitVerification:
    status: str
    code: str
    message: str
    details: dict

    @property
    def ok(self) -> bool:
        return self.status == STATUS_VERIFIED


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def result_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(payload)).encode("utf-8")).hexdigest()


def parse_json_object(raw: Any, *, field: str) -> dict:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise CompletionIntegrityError(
                REASON_INVALID,
                f"{field} is not valid JSON",
                details={"field": field},
            ) from exc
        if not isinstance(parsed, dict):
            raise CompletionIntegrityError(
                REASON_INVALID,
                f"{field} must be a JSON object",
                details={"field": field},
            )
        return parsed
    raise CompletionIntegrityError(
        REASON_INVALID,
        f"{field} must be a JSON object",
        details={"field": field, "type": type(raw).__name__},
    )


def normalize_completion_contract(raw: Any) -> Optional[dict]:
    if raw is None or raw == "":
        return None
    contract = parse_json_object(raw, field="completion_contract")
    if not contract:
        return None
    schema_version = contract.get("schema_version", 1)
    ctype = contract.get("type")
    if schema_version != 1 or ctype not in {
        CONTRACT_GIT_REVISION,
        CONTRACT_REVIEW,
        CONTRACT_HISTORIAN_CERTIFY,
    }:
        raise CompletionIntegrityError(
            REASON_INVALID,
            "completion_contract is malformed or unsupported",
            details={"contract": contract},
        )
    dirty_policy = contract.get("dirty_policy") or "observe"
    if dirty_policy not in KNOWN_DIRTY_POLICIES:
        raise CompletionIntegrityError(
            REASON_INVALID,
            "completion_contract dirty_policy is unknown",
            details={"dirty_policy": dirty_policy},
        )
    expected_base = _optional_str(contract.get("expected_base"))
    max_commits = _normalize_max_commits(contract.get("max_commits"))
    if max_commits is not None and not expected_base:
        raise CompletionIntegrityError(
            REASON_INVALID,
            "max_commits without expected_base is an invalid contract combination",
            details={"max_commits": max_commits},
        )
    repository = _optional_str(contract.get("repository"))
    worktree = _optional_str(contract.get("worktree"))
    git_common_dir = _optional_str(contract.get("git_common_dir"))
    if not (repository or worktree or git_common_dir):
        raise CompletionIntegrityError(
            REASON_INVALID,
            "revision-required completion contract must establish "
            "controller-authorized repository identity",
            details={"contract": contract},
        )
    required_sha = _normalize_sha(contract.get("required_sha"))
    if ctype == CONTRACT_HISTORIAN_CERTIFY:
        if not required_sha or not FULL_SHA_RE.fullmatch(required_sha):
            raise CompletionIntegrityError(
                REASON_INVALID,
                "historian_certify requires a full required_sha",
                details={"required_sha": required_sha},
            )
    return {
        "schema_version": 1,
        "type": ctype,
        "repository": repository,
        "worktree": worktree,
        "git_common_dir": git_common_dir,
        "expected_base": expected_base,
        "max_commits": max_commits,
        "require_remote": _optional_str(contract.get("require_remote")),
        "dirty_policy": dirty_policy,
        "required_sha": required_sha,
    }


def normalize_semantic_gate(raw: Any) -> Optional[dict]:
    if raw is None or raw == "":
        return None
    gate = parse_json_object(raw, field="semantic_gate")
    if not gate:
        return None
    if gate.get("schema_version", 1) != 1 or gate.get("type") != GATE_REVIEW_APPROVED:
        raise CompletionIntegrityError(
            REASON_GATE_INVALID,
            "semantic_gate must be a typed review_approved predicate",
            details={"gate": gate},
        )
    reviewed_sha = _optional_str(gate.get("reviewed_sha"))
    if not reviewed_sha or not FULL_SHA_RE.fullmatch(reviewed_sha):
        raise CompletionIntegrityError(
            REASON_GATE_INVALID,
            "review_approved.reviewed_sha must be a full 40-char commit SHA",
            details={"gate": gate},
        )
    return {
        "schema_version": 1,
        "type": GATE_REVIEW_APPROVED,
        "reviewed_sha": reviewed_sha,
    }


def persist_json(value: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not value:
        return None
    return canonical_json(value)


def load_task_contract(row: Mapping[str, Any]) -> Optional[dict]:
    raw = None
    if "completion_contract" in row.keys():
        raw = row["completion_contract"]
    if not raw:
        return None
    try:
        return normalize_completion_contract(raw)
    except CompletionIntegrityError:
        return {"schema_version": 1, "type": "__invalid__"}


def contract_requires_revision(contract: Optional[Mapping[str, Any]]) -> bool:
    return bool(contract) and contract.get("type") in {
        CONTRACT_GIT_REVISION,
        CONTRACT_HISTORIAN_CERTIFY,
        CONTRACT_REVIEW,
    }


def mint_attempt_id(task_id: str, run_id: Any) -> str:
    return f"att_{task_id}_{int(run_id)}"


def task_governance_fields(task: Any) -> dict[str, Any]:
    """Explicit governance fields for worker/CLI/tool/API surfaces."""
    return {
        "completion_contract": getattr(task, "completion_contract", None),
        "attempt_id": getattr(task, "attempt_id", None),
        "terminal_result": getattr(task, "terminal_result", None),
        "terminal_result_hash": getattr(task, "terminal_result_hash", None),
        "verification_status": getattr(task, "verification_status", None),
        "verification_code": getattr(task, "verification_code", None),
        "verified_revision": getattr(task, "verified_revision", None),
        "verified_verdict": getattr(task, "verified_verdict", None),
        "awaiting_verification": bool(getattr(task, "awaiting_verification", False)),
        "needs_reconciliation": bool(getattr(task, "needs_reconciliation", False)),
    }


def extract_terminal_result(
    *,
    terminal_result: Any = None,
    metadata: Any = None,
    result: Any = None,
    summary: Any = None,
) -> Optional[dict]:
    raw = terminal_result
    if raw is None and isinstance(metadata, dict):
        raw = metadata.get("terminal_result")
    if raw is None:
        return None
    payload = parse_json_object(raw, field="terminal_result")
    claimed: dict[str, Any] = {
        "attempt_id": _optional_str(payload.get("attempt_id")),
        "repository": _optional_str(payload.get("repository")),
        "worktree": _optional_str(payload.get("worktree")),
        "commit_sha": _normalize_sha(payload.get("commit_sha") or payload.get("reviewed_sha")),
        "reviewed_sha": _normalize_sha(payload.get("reviewed_sha") or payload.get("commit_sha")),
        "verdict": _optional_str(payload.get("verdict")),
    }
    extra = {
        key: payload[key]
        for key in payload
        if key not in claimed
    }
    if extra:
        claimed["extra"] = extra
    if result:
        claimed["result_note"] = str(result)
    if summary:
        claimed["summary_note"] = str(summary)
    return claimed


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_max_commits(raw: Any) -> Optional[int]:
    if raw is None or raw == "":
        return None
    # Accept only an actual positive int. Do not coerce strings, floats,
    # bools, or other numeric-looking values.
    if type(raw) is int and raw >= 1:
        return raw
    raise CompletionIntegrityError(
        REASON_INVALID,
        "max_commits must be a positive integer",
        details={"max_commits": raw},
    )


def _normalize_sha(value: Any) -> Optional[str]:
    text = _optional_str(value)
    if text is None:
        return None
    return text.lower()


def _run_git(args: list[str], *, cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    argv = ["git", "-c", "core.useReplaceRefs=false", *args]
    env = os.environ.copy()
    env[_GIT_NO_REPLACE_ENV] = "1"
    return subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_SECONDS,
        check=False,
        shell=False,
        env=env,
    )


def resolve_git_common_dir(path: str) -> Optional[str]:
    try:
        proc = _run_git(["rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=path)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    resolved = (proc.stdout or "").strip()
    if not resolved:
        return None
    try:
        return str(Path(resolved).resolve())
    except OSError:
        return resolved


def resolve_head(path: str) -> Optional[str]:
    try:
        proc = _run_git(["rev-parse", "HEAD"], cwd=path)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip().lower() or None


def verify_git_revision(
    *,
    claimed_sha: Optional[str],
    contract: Mapping[str, Any],
    claimed_repository: Optional[str] = None,
    claimed_worktree: Optional[str] = None,
    require_head_match: bool = False,
) -> GitVerification:
    """Verify a claimed revision from canonical Git object storage.

    ``require_head_match`` is False by default: Historian certification must
    inspect the immutable object even when the mutable checkout has moved.
    """
    details: dict[str, Any] = {
        "claimed_sha": claimed_sha,
        "require_head_match": require_head_match,
    }
    if not claimed_sha or not FULL_SHA_RE.fullmatch(claimed_sha):
        return GitVerification(
            STATUS_REJECTED,
            REASON_SHA_SYNTAX,
            "claimed revision is not a full 40-character commit SHA",
            details,
        )

    required_sha = _normalize_sha(contract.get("required_sha"))
    if required_sha:
        details["required_sha"] = required_sha
        if claimed_sha != required_sha:
            return GitVerification(
                STATUS_REJECTED,
                REASON_SHA_MISMATCH,
                "claimed revision is not the controller-authorized required SHA",
                details,
            )

    repo = contract.get("repository") or contract.get("worktree")
    worktree = contract.get("worktree")
    if not repo:
        return GitVerification(
            STATUS_REJECTED,
            REASON_INVALID,
            "completion contract does not identify an authorized repository",
            details,
        )
    details["repository"] = repo
    details["worktree"] = worktree

    try:
        repo_common = resolve_git_common_dir(repo)
        if repo_common is None:
            return GitVerification(
                STATUS_ERROR,
                REASON_VERIFICATION_ERROR,
                "authorized repository is not a readable git directory",
                details,
            )
        details["repo_common_dir"] = repo_common

        expected_common = contract.get("git_common_dir")
        if expected_common:
            expected_resolved = str(Path(expected_common).resolve())
            details["expected_common_dir"] = expected_resolved
            if repo_common != expected_resolved:
                return GitVerification(
                    STATUS_REJECTED,
                    REASON_WRONG_REPOSITORY,
                    "claimed repository does not match the authorized git identity",
                    details,
                )

        if worktree:
            worktree_common = resolve_git_common_dir(worktree)
            details["worktree_common_dir"] = worktree_common
            if worktree_common is None or worktree_common != repo_common:
                return GitVerification(
                    STATUS_REJECTED,
                    REASON_WRONG_WORKTREE,
                    "claimed worktree is not the authorized worktree/common git directory",
                    details,
                )

        if claimed_repository:
            claimed_common = resolve_git_common_dir(claimed_repository)
            details["claimed_repository_common_dir"] = claimed_common
            if claimed_common is None or claimed_common != repo_common:
                return GitVerification(
                    STATUS_REJECTED,
                    REASON_WRONG_REPOSITORY,
                    "claimed repository identity is not the authorized repository",
                    details,
                )

        type_proc = _run_git(["cat-file", "-t", claimed_sha], cwd=repo)
        if type_proc.returncode != 0:
            return GitVerification(
                STATUS_REJECTED,
                REASON_OBJECT_MISSING,
                "claimed revision does not exist in the authorized repository",
                details,
            )
        object_type = (type_proc.stdout or "").strip()
        details["object_type"] = object_type
        if object_type != "commit":
            return GitVerification(
                STATUS_REJECTED,
                REASON_OBJECT_NOT_COMMIT,
                "claimed revision exists but is not a commit object",
                details,
            )

        expected_base = contract.get("expected_base")
        if expected_base:
            details["expected_base"] = expected_base
            if not FULL_SHA_RE.fullmatch(str(expected_base).lower()):
                return GitVerification(
                    STATUS_REJECTED,
                    REASON_BASE_MISMATCH,
                    "completion contract expected_base is not a full commit SHA",
                    details,
                )
            base = str(expected_base).lower()
            ancestor = _run_git(["merge-base", "--is-ancestor", base, claimed_sha], cwd=repo)
            if ancestor.returncode != 0:
                return GitVerification(
                    STATUS_REJECTED,
                    REASON_BASE_MISMATCH,
                    "claimed commit is not a descendant of the authorized base",
                    details,
                )
            max_commits = contract.get("max_commits")
            if max_commits is not None:
                try:
                    bound = _normalize_max_commits(max_commits)
                except CompletionIntegrityError as exc:
                    return GitVerification(
                        STATUS_REJECTED,
                        exc.code,
                        str(exc),
                        {**details, **exc.details},
                    )
                count_proc = _run_git(
                    ["rev-list", "--count", f"{base}..{claimed_sha}"],
                    cwd=repo,
                )
                if count_proc.returncode != 0:
                    return GitVerification(
                        STATUS_ERROR,
                        REASON_VERIFICATION_ERROR,
                        "could not count commits from the authorized base",
                        details,
                    )
                try:
                    count = int((count_proc.stdout or "0").strip() or "0")
                except ValueError:
                    return GitVerification(
                        STATUS_ERROR,
                        REASON_VERIFICATION_ERROR,
                        "git rev-list returned a non-integer commit count",
                        details,
                    )
                details["commit_count"] = count
                if bound is None or count < 1 or count > bound:
                    return GitVerification(
                        STATUS_REJECTED,
                        REASON_COMMIT_COUNT,
                        "claimed commit is outside the authorized commit-count bound",
                        details,
                    )

        require_remote = contract.get("require_remote")
        if require_remote:
            details["require_remote"] = require_remote
            return GitVerification(
                STATUS_REJECTED,
                REASON_REMOTE_MISMATCH,
                "mutable remote URL text is not proof of authorized lineage",
                details,
            )

        dirty_policy = contract.get("dirty_policy") or "observe"
        details["dirty_policy"] = dirty_policy
        if dirty_policy not in KNOWN_DIRTY_POLICIES:
            return GitVerification(
                STATUS_REJECTED,
                REASON_INVALID,
                "completion contract dirty_policy is unknown",
                details,
            )
        inspect_path = worktree or repo
        status_proc = _run_git(["status", "--porcelain"], cwd=inspect_path)
        if status_proc.returncode != 0:
            details["git_status_returncode"] = status_proc.returncode
            return GitVerification(
                STATUS_ERROR,
                REASON_VERIFICATION_ERROR,
                "git status failed; dirty state cannot be treated as clean",
                details,
            )
        dirty = bool((status_proc.stdout or "").strip())
        details["dirty"] = dirty
        if dirty_policy == "reject" and dirty:
            return GitVerification(
                STATUS_REJECTED,
                REASON_DIRTY_WORKTREE,
                "authorized worktree is dirty and the contract forbids dirty state",
                details,
            )

        if require_head_match:
            head = resolve_head(inspect_path)
            details["head"] = head
            if head != claimed_sha:
                return GitVerification(
                    STATUS_REJECTED,
                    REASON_SHA_MISMATCH,
                    "HEAD does not equal the claimed revision",
                    details,
                )
        else:
            details["head"] = resolve_head(inspect_path)
            details["head_policy"] = REASON_HEAD_NOT_REQUIRED

        return GitVerification(
            STATUS_VERIFIED,
            REASON_SATISFIED,
            "claimed revision verified from the authorized git object store",
            details,
        )
    except CompletionIntegrityError as exc:
        return GitVerification(STATUS_REJECTED, exc.code, str(exc), {**details, **exc.details})
    except (OSError, subprocess.TimeoutExpired) as exc:
        details["error"] = str(exc)
        return GitVerification(
            STATUS_ERROR,
            REASON_VERIFICATION_ERROR,
            "git verification failed unexpectedly",
            details,
        )


def certify_immutable_revision(
    *,
    repository: str,
    commit_sha: str,
    expected_base: Optional[str] = None,
    git_common_dir: Optional[str] = None,
    max_commits: Any = None,
    required_sha: Optional[str] = None,
    worktree: Optional[str] = None,
    dirty_policy: str = "observe",
) -> GitVerification:
    """Bounded Historian consumer: certify a reviewed SHA from object storage."""
    return verify_git_revision(
        claimed_sha=_normalize_sha(commit_sha),
        contract={
            "repository": repository,
            "worktree": worktree,
            "git_common_dir": git_common_dir,
            "expected_base": expected_base,
            "max_commits": max_commits,
            "required_sha": required_sha,
            "dirty_policy": dirty_policy or "observe",
        },
        require_head_match=False,
    )


def normalize_verdict(raw: Any) -> tuple[Optional[str], Optional[str]]:
    """Return (normalized_verdict, error_code)."""
    if raw is None or str(raw).strip() == "":
        return None, REASON_VERDICT_MISSING
    verdict = str(raw).strip().upper()
    if verdict not in KNOWN_VERDICTS:
        return None, REASON_VERDICT_MALFORMED
    return verdict, None


def evaluate_review_approved_gate(
    *,
    required_sha: str,
    parent_status: str,
    parent_verification_status: Optional[str],
    parent_verified_revision: Optional[str],
    parent_verdict: Optional[str],
    parent_needs_reconciliation: bool,
    parent_contract_type: Optional[str],
) -> GateEvaluation:
    if parent_needs_reconciliation:
        return GateEvaluation(
            REASON_UNSATISFIED,
            REASON_NEEDS_RECONCILIATION,
            "parent has a conflicting terminal result and needs reconciliation",
            {"required_sha": required_sha},
        )
    if parent_status not in {"done", "archived"}:
        return GateEvaluation(
            REASON_UNSATISFIED,
            REASON_PARENT_NOT_TERMINAL,
            "parent is not a terminal review result",
            {"parent_status": parent_status},
        )
    if parent_contract_type not in {CONTRACT_REVIEW, None} and parent_contract_type not in {
        CONTRACT_GIT_REVISION,
        CONTRACT_HISTORIAN_CERTIFY,
    }:
        # A review_approved child may depend on a review task. An invalid
        # parent contract cannot authorize descendants.
        if parent_contract_type == "__invalid__":
            return GateEvaluation(
                REASON_INVALID,
                REASON_GATE_INVALID,
                "parent completion contract is malformed",
                {},
            )
    if parent_verification_status == STATUS_ERROR:
        return GateEvaluation(
            REASON_UNSATISFIED,
            REASON_VERIFICATION_ERROR,
            "parent review evidence has a verification error",
            {},
        )
    if parent_verification_status == STATUS_REJECTED:
        return GateEvaluation(
            REASON_UNSATISFIED,
            REASON_VERIFICATION_REJECTED,
            "parent review evidence was rejected",
            {},
        )
    if parent_verification_status != STATUS_VERIFIED:
        return GateEvaluation(
            REASON_UNSATISFIED,
            REASON_UNSATISFIED,
            "parent review result has not been controller-verified",
            {"verification_status": parent_verification_status},
        )
    verdict, verdict_err = normalize_verdict(parent_verdict)
    if verdict_err:
        return GateEvaluation(
            REASON_INVALID if verdict_err == REASON_VERDICT_MALFORMED else REASON_UNSATISFIED,
            verdict_err,
            "parent review verdict is missing or unrecognized",
            {"verdict": parent_verdict},
        )
    if verdict not in APPROVAL_VERDICTS:
        return GateEvaluation(
            REASON_UNSATISFIED,
            REASON_VERDICT_NOT_APPROVAL,
            "parent review verdict is not an allowed approval",
            {"verdict": verdict},
        )
    verified = _normalize_sha(parent_verified_revision)
    if not verified or not FULL_SHA_RE.fullmatch(verified):
        return GateEvaluation(
            REASON_INVALID,
            REASON_SHA_SYNTAX,
            "parent verified revision is not a full commit SHA",
            {"verified_revision": parent_verified_revision},
        )
    if verified != required_sha:
        return GateEvaluation(
            REASON_UNSATISFIED,
            REASON_SHA_MISMATCH,
            "parent approval does not match the SHA required by the gate",
            {"required_sha": required_sha, "verified_revision": verified},
        )
    return GateEvaluation(
        REASON_SATISFIED,
        REASON_SATISFIED,
        "typed review_approved gate is satisfied",
        {"reviewed_sha": verified, "verdict": verdict},
    )

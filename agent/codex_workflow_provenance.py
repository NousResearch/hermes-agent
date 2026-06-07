import datetime as _datetime
import fnmatch
import hashlib
import json
from pathlib import Path
from typing import Any


_REPO_PRESERVE_PREFIXES = (("docs", "plans"),)
_CACHE_PARTS = {".pytest_cache", "__pycache__", ".mypy_cache", ".ruff_cache", "node_modules", "dist", "build"}


def repo_id(repo: Path) -> str:
    return hashlib.sha256(str(repo.resolve(strict=False)).encode("utf-8")).hexdigest()[:16]


def file_hash(path: Path) -> str | None:
    try:
        if not path.exists() or not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _path_class(rel_path: str) -> str:
    path = Path(rel_path)
    parts = tuple(part.lower() for part in path.parts)
    name = path.name.lower()
    if any(part in _CACHE_PARTS for part in parts):
        return "cache"
    if parts[:1] in {("tests",), ("test",)} or name.startswith("test_") or name.endswith("_test.py"):
        return "test"
    if parts[:1] in {("docs",), ("doc",)} or name in {"readme.md", "changelog.md", "contributing.md"}:
        return "docs"
    if parts[:1] in {("src",), ("lib",), ("app",), ("apps",), ("packages",), ("tools",), ("scripts",), ("gateway",), ("agent",), ("hermes_cli",)}:
        return "source"
    if any(name.endswith(ext) for ext in (".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".sh")):
        return "source"
    return "unknown"


def _event_path_class(rel_path: str, operation: str | None = None) -> str:
    path_class = _path_class(rel_path)
    if operation == "review_artifact":
        return "review_artifact"
    if path_class == "cache":
        return "generated_cache"
    return path_class


def _docs_plans_path(rel_path: str) -> bool:
    return Path(rel_path).parts[:2] in _REPO_PRESERVE_PREFIXES


def provenance_event(
    *,
    repo: Path,
    branch: str | None,
    head_sha: str | None,
    stage_id: str,
    session_id: str | None,
    actor: str,
    tool: str,
    operation: str,
    path: str,
    before_hash: str | None,
    after_hash: str | None,
    explicit_authorization: bool = False,
    authorization_reason: str = "",
) -> dict[str, Any]:
    owner_session_id = session_id if session_id else "unknown"
    return {
        "schema_version": 1,
        "event_id": hashlib.sha256(
            json.dumps(
                [str(repo.resolve(strict=False)), branch, head_sha, stage_id, session_id, operation, path, after_hash],
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:24],
        "repo_id": repo_id(repo),
        "branch": branch,
        "head_sha": head_sha,
        "stage_id": stage_id,
        "session_id": session_id or "unknown",
        "actor": actor,
        "tool": tool,
        "operation": operation,
        "path": path,
        "path_class": _event_path_class(path, operation),
        "before_hash": before_hash,
        "after_hash": after_hash,
        "owner_session_id": owner_session_id,
        "owner_policy": "current_session" if session_id else "unknown_unowned",
        "authorization": {"explicit": bool(explicit_authorization), "reason": authorization_reason},
        "timestamp": _datetime.datetime.now(_datetime.UTC).isoformat().replace("+00:00", "Z"),
    }


def _matching_event(events: list[dict[str, Any]] | None, rel_path: str) -> dict[str, Any] | None:
    if not isinstance(events, list):
        return None
    for event in reversed(events):
        if isinstance(event, dict) and event.get("path") == rel_path:
            return event
    return None


def classify_path(
    *,
    repo: Path,
    rel_path: str,
    current_session_id: str | None,
    branch: str | None,
    head_sha: str | None,
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    path_class = _path_class(rel_path)
    reasons: list[str] = []
    event = _matching_event(events, rel_path)
    current_hash = file_hash(repo / rel_path)
    owner_session_id = None
    owner_policy = "unknown_unowned"

    if path_class == "cache" and event is None:
        owner_policy = "generated_cache"
    elif not current_session_id:
        reasons.append("missing_session_id")
    elif event is None:
        reasons.append("missing_provenance")
    else:
        owner_session_id = event.get("owner_session_id") or event.get("session_id")
        event_repo_id = event.get("repo_id")
        current_repo_id = repo_id(repo)
        if not event_repo_id:
            reasons.append("missing_repo_id")
        elif event_repo_id != current_repo_id:
            reasons.append("repo_id_mismatch")
        if event.get("branch") != branch:
            reasons.append("branch_mismatch")
        if event.get("head_sha") != head_sha:
            reasons.append("head_mismatch")
        if event.get("after_hash") != current_hash:
            reasons.append("hash_mismatch")

        if reasons:
            owner_policy = "unknown_unowned"
        elif owner_session_id != current_session_id:
            owner_policy = "other_known_session"
        elif event.get("operation") == "review_artifact":
            owner_policy = "review_artifact_current_session"
        else:
            owner_policy = "current_session"

    if _docs_plans_path(rel_path) and owner_policy != "review_artifact_current_session":
        reasons.append("docs_plans_default_preserve")

    default_behavior = "preserve"
    return {
        "path": rel_path,
        "path_class": path_class,
        "owner_policy": owner_policy,
        "owner_session_id": owner_session_id,
        "default_behavior": default_behavior,
        "cleanup_allowed": False,
        "blocking_reasons": sorted(set(reasons)),
    }


def _path_allowed(rel_path: str, allowed_files: list[str], allowed_globs: list[str]) -> bool:
    return rel_path in allowed_files or any(fnmatch.fnmatchcase(rel_path, pattern) for pattern in allowed_globs)


def cleanup_decision(
    *,
    repo: Path,
    rel_path: str,
    current_session_id: str | None,
    branch: str | None,
    head_sha: str | None,
    events: list[dict[str, Any]] | None,
    allowed_files: list[str],
    allowed_globs: list[str],
    explicit_authorization: bool,
    operation: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    decision = classify_path(
        repo=repo,
        rel_path=rel_path,
        current_session_id=current_session_id,
        branch=branch,
        head_sha=head_sha,
        events=events,
    )
    reasons = list(decision["blocking_reasons"])

    if dry_run:
        reasons.append("dry_run_non_mutating")
    if operation not in {"cleanup", "delete", "overwrite"}:
        reasons.append("unsupported_operation")
    if not explicit_authorization:
        reasons.append("authorization_required")

    owner_policy = decision["owner_policy"]
    owner_ok = owner_policy in {"current_session", "review_artifact_current_session"}
    if not owner_ok:
        reasons.append("owner_policy_not_current_session")
    if not _path_allowed(rel_path, allowed_files, allowed_globs):
        reasons.append("path_not_in_allowlist")
    if _docs_plans_path(rel_path) and owner_policy != "review_artifact_current_session":
        reasons.append("docs_plans_default_preserve")

    cleanup_allowed = not reasons
    decision["cleanup_allowed"] = cleanup_allowed
    decision["blocking_reasons"] = sorted(set(reasons))
    return decision


def ownership_preview(
    *,
    repo: Path,
    dirty_paths: list[str],
    current_session_id: str | None,
    branch: str | None,
    head_sha: str | None,
    events: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    return [
        classify_path(
            repo=repo,
            rel_path=path,
            current_session_id=current_session_id,
            branch=branch,
            head_sha=head_sha,
            events=events,
        )
        for path in dirty_paths
    ]

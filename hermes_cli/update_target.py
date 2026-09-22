"""Strict identity validation and exact Git application for pinned updates.

The ordinary updater intentionally keeps its historical branch-tip behaviour.  A
pinned request enters this module instead: it admits one existing ``origin``
tracking branch, proves the reviewed commit and protocol floor, and performs one
literal fast-forward to that commit.  No stash, ZIP, branch-tip, or divergence
fallback belongs on this path; the one allowed movement is ``merge --ff-only``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from hermes_constants import get_default_hermes_root


_HEX = re.compile(r"^[0-9a-f]+$")
_INSTALL_ID = re.compile(r"^[0-9a-f]{32}$")
_PROTOCOL_PATH = "hermes_cli/update_rollout_protocol.json"
_PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class TargetRequest:
    """The exact checkout identity a pinned update request names."""

    revision: str
    install_id: str
    current_sha: str


class TargetAdmissionError(RuntimeError):
    """A pinned target failed a pre-mutation admission predicate."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(reason if not detail else f"{reason}: {detail}")


# T1/T2 named this public refusal seam ``PinnedTargetRefused``. Keep the
# spelling while the T3 admission engine uses the more precise base name.
PinnedTargetRefused = TargetAdmissionError


@dataclass(frozen=True)
class PinnedApplyResult:
    """Truthful result of the one allowed exact-target Git operation."""

    outcome: str  # ``applied`` or ``already-current``
    prior_sha: str
    post_sha: str
    branch: str
    origin: str
    protocol: int = _PROTOCOL_VERSION

    @property
    def target_sha(self) -> str:
        """Compatibility spelling used by the update receipt and fleet tests."""
        return self.post_sha


def _validate_field(name: str, value: object, length: int) -> str:
    if not isinstance(value, str) or len(value) != length or _HEX.fullmatch(value) is None:
        raise ValueError(f"invalid-{name}")
    return value


def validate_target_request(
    revision: object, install_id: object, current_sha: object
) -> TargetRequest | None:
    """Validate a complete target identity without normalizing its fields."""
    supplied = (revision is not None, install_id is not None, current_sha is not None)
    if not any(supplied):
        return None
    if not all(supplied):
        raise ValueError("incomplete-target-intent")
    return TargetRequest(
        _validate_field("revision", revision, 40),
        _validate_field("install_id", install_id, 32),
        _validate_field("current_sha", current_sha, 40),
    )


def validate_update_intent(intent: object) -> dict[str, str]:
    """Validate and copy the complete T4 identity carried through handoff."""
    if not isinstance(intent, dict):
        raise ValueError("invalid-update-intent")
    required = {"target", "install_id", "correlation_id", "prior_sha", "branch"}
    if not required.issubset(intent):
        raise ValueError("incomplete-update-intent")
    target = _validate_field("target", intent["target"], 40)
    prior = _validate_field("prior_sha", intent["prior_sha"], 40)
    install_id = intent["install_id"]
    if not isinstance(install_id, str) or _INSTALL_ID.fullmatch(install_id) is None:
        raise ValueError("invalid-install_id")
    correlation_id = intent["correlation_id"]
    if (
        not isinstance(correlation_id, str)
        or not correlation_id
        or len(correlation_id) > 128
        or any(ch.isspace() for ch in correlation_id)
    ):
        raise ValueError("invalid-correlation_id")
    branch = intent["branch"]
    if (
        not isinstance(branch, str)
        or not branch
        or branch.startswith("-")
        or ".." in branch
        or branch.endswith("/")
        or "//" in branch
        or any(ch.isspace() for ch in branch)
    ):
        raise ValueError("invalid-branch")
    return {
        "target": target,
        "install_id": install_id,
        "correlation_id": correlation_id,
        "prior_sha": prior,
        "branch": branch,
    }


def build_update_intent(request: TargetRequest, correlation_id: str, branch: str) -> dict[str, str]:
    """Create the exact five-field immutable intent used by T4 handoff."""
    checked = _require_request(request)
    return validate_update_intent({
        "target": checked.revision,
        "install_id": checked.install_id,
        "correlation_id": correlation_id,
        "prior_sha": checked.current_sha,
        "branch": branch,
    })

def _run_git(root: Path, *args: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    """Run a non-interactive Git query against *root* with no fallback transport."""
    try:
        return subprocess.run(
            ["git", *args], cwd=str(root), stdin=subprocess.DEVNULL,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, check=False,
            env={**__import__("os").environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(
            ["git", *args], 125, stdout="", stderr=str(exc)
        )


def _git_value(root: Path, *args: str) -> str | None:
    result = _run_git(root, *args)
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def current_branch(root: str | Path) -> str | None:
    """Return the checked-out branch without changing refs or using a fallback remote."""
    return _git_value(Path(root), "symbolic-ref", "--quiet", "--short", "HEAD")


def _refuse(reason: str, result: subprocess.CompletedProcess[str] | None = None) -> None:
    detail = ""
    if result is not None:
        detail = (result.stderr or result.stdout or "").strip().splitlines()[0] if (result.stderr or result.stdout) else ""
    raise TargetAdmissionError(reason, detail)


def _read_install_id() -> str | None:
    """Read the already-published identity from the authoritative Hermes home."""
    try:
        value = (get_default_hermes_root() / "install_id").read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError, UnicodeError):
        return None
    return value if _INSTALL_ID.fullmatch(value) else None


def _protocol_from_target(root: Path, revision: str) -> int:
    result = _run_git(root, "show", f"{revision}:{_PROTOCOL_PATH}", timeout=10)
    if result.returncode != 0:
        raise TargetAdmissionError("incompatible-target", "missing protocol metadata")
    raw = result.stdout
    if len(raw.encode("utf-8", "replace")) > 4096:
        raise TargetAdmissionError("incompatible-target", "protocol metadata exceeds 4 KiB")
    try:
        pairs: list[tuple[str, Any]] = []
        def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            names = [key for key, _value in items]
            if len(names) != len(set(names)):
                raise ValueError("duplicate key")
            pairs.extend(items)
            return dict(items)
        metadata = json.loads(raw.encode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise TargetAdmissionError("incompatible-target", f"invalid protocol metadata: {exc}") from exc
    if not isinstance(metadata, dict) or set(metadata) != {"protocol"}:
        raise TargetAdmissionError("incompatible-target", "protocol metadata is not exactly {protocol}")
    version = metadata.get("protocol")
    if isinstance(version, bool) or not isinstance(version, int) or version < _PROTOCOL_VERSION:
        raise TargetAdmissionError("incompatible-target", "unsupported protocol floor")
    return version


def _require_request(request: TargetRequest) -> TargetRequest:
    if not isinstance(request, TargetRequest):
        raise TargetAdmissionError("invalid-target-intent")
    try:
        checked = validate_target_request(request.revision, request.install_id, request.current_sha)
    except ValueError as exc:
        raise TargetAdmissionError(str(exc)) from exc
    if checked is None:  # pragma: no cover - guarded by the dataclass check above
        raise TargetAdmissionError("incomplete-target-intent")
    return checked


def apply_pinned_target(
    root: str | Path,
    request: TargetRequest,
    *,
    branch: str | None = None,
) -> PinnedApplyResult:
    """Admit and apply one exact target from the existing ``origin`` remote.

    Every refusal is raised before the one ``merge --ff-only <sha>`` movement.
    The origin branch may advance after review; the requested object remains the
    only object this function can apply.
    """
    root = Path(root)
    request = _require_request(request)
    if not root.exists():
        raise TargetAdmissionError("checkout-unavailable")

    origin = _git_value(root, "remote", "get-url", "origin")
    if origin is None:
        raise TargetAdmissionError("origin-required")
    current_branch = _git_value(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    if current_branch is None:
        raise TargetAdmissionError("branch-not-admitted", "detached HEAD")
    if branch is not None and branch != current_branch:
        raise TargetAdmissionError("branch-not-admitted", f"current branch is {current_branch}")
    tracking = _git_value(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    if tracking != f"origin/{current_branch}":
        raise TargetAdmissionError("branch-not-admitted", "tracking branch is not origin/<current branch>")

    status = _run_git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status.returncode != 0:
        _refuse("checkout-unverifiable", status)
    if status.stdout.strip():
        raise TargetAdmissionError("dirty-checkout")

    prior_sha = _git_value(root, "rev-parse", "HEAD")
    if prior_sha != request.current_sha:
        raise TargetAdmissionError("current-sha-mismatch")
    stored_install_id = _read_install_id()
    if stored_install_id != request.install_id:
        raise TargetAdmissionError("install-id-mismatch")

    fetched = _run_git(root, "fetch", "--no-tags", "origin", current_branch, timeout=300)
    if fetched.returncode != 0:
        _refuse("origin-unreachable", fetched)
    # A fetch cannot authorize a concurrent working-tree change.
    if _git_value(root, "rev-parse", "HEAD") != request.current_sha:
        raise TargetAdmissionError("current-sha-mismatch")

    authorized_ref = f"origin/{current_branch}"
    if _run_git(root, "merge-base", "--is-ancestor", request.current_sha, authorized_ref).returncode != 0:
        raise TargetAdmissionError("diverged-target")
    target_type = _git_value(root, "cat-file", "-t", request.revision)
    if target_type != "commit":
        raise TargetAdmissionError("target-unreachable")
    if _run_git(root, "merge-base", "--is-ancestor", request.revision, authorized_ref).returncode != 0:
        raise TargetAdmissionError("target-not-reachable (target-not-on-authorized-origin)")
    if _run_git(root, "merge-base", "--is-ancestor", request.current_sha, request.revision).returncode != 0:
        raise TargetAdmissionError("target-not-fast-forward")

    # This is deliberately before merge: a pre-protocol target may not move
    # the checkout even when the caller supplied a protocol=1 claim.
    protocol = _protocol_from_target(root, request.revision)

    if request.revision == prior_sha:
        return PinnedApplyResult("already-current", prior_sha, prior_sha, current_branch, origin, protocol)

    merged = _run_git(root, "merge", "--ff-only", request.revision)
    if merged.returncode != 0:
        _refuse("pinned-apply-refused", merged)
    post_sha = _git_value(root, "rev-parse", "HEAD")
    if post_sha != request.revision:
        raise TargetAdmissionError("post-apply-head-mismatch")
    return PinnedApplyResult("applied", prior_sha, post_sha, current_branch, origin, protocol)


def verify_pinned_post_swap(root: str | Path, request: TargetRequest) -> dict[str, str]:
    """Verify exact target HEAD and identity after handoff, without repairing either."""
    root = Path(root)
    request = _require_request(request)
    post_sha = _git_value(root, "rev-parse", "HEAD")
    if post_sha != request.revision:
        raise TargetAdmissionError("post-swap-head-mismatch")
    install_id = _read_install_id()
    if install_id != request.install_id:
        raise TargetAdmissionError("post-swap-install-id-mismatch")
    return {"post_sha": post_sha, "post_install_id": install_id}

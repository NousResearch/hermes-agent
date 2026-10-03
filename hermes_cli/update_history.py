"""Fork history assessment. Tree equivalence never authorizes rewriting a branch."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
import uuid

#: Exit 2 is the one non-zero code both desktop updaters never retry: the legacy
#: retry arms in ``scripts/desktop-update/posix.sh`` and ``windows.ps1`` exempt only
#: 0 and 2, and the PM path never retries. A fold review is deterministic, so a
#: retry would only repeat it. Both scripts match exit 2 plus
#: :data:`HISTORY_REVIEW_BANNER` on stdout and report a refusal, not a failure.
HISTORY_REVIEW_EXIT = 2

#: Printed first by :func:`guard_fork_history`; the desktop updaters key on it.
HISTORY_REVIEW_BANNER = "HERMES_UPDATE_HISTORY_REVIEW_REQUIRED"

#: The first commit of the fork's reconstructed history (2026-09-15, "fork: build
#: test platform"; its parent is upstream's). A checkout whose HEAD does not descend
#: from it carries no fork history for the guard to protect: upstream's own updater
#: tests build throwaway repositories and exercise upstream's rescue-ref path there.
FORK_ROOT_COMMIT = "5a411b1657eb8373d4f719a117f6d0aacd2b9fd7"

@dataclass(frozen=True)
class HistoryAssessment:
    target: str
    head: str | None
    target_head: str | None
    relationship: str
    same_tree: bool = False
    local_only: int | None = None
    remote_only: int | None = None
    shallow: bool = False

    def to_dict(self):
        return asdict(self)


def _git(git_cmd, cwd, *args, input=None):
    # Git's update-ref transaction protocol requires literal LF. Windows text
    # subprocess input translates those to CRLF and invalidates the commands.
    result = subprocess.run(
        [*git_cmd, *args], cwd=cwd,
        input=input.encode("utf-8") if input is not None else None,
        capture_output=True, timeout=30,
    )
    return subprocess.CompletedProcess(result.args, result.returncode,
                                       result.stdout.decode("utf-8", "replace"),
                                       result.stderr.decode("utf-8", "replace"))


def assess_history(git_cmd, cwd: Path, target: str) -> HistoryAssessment:
    """Use local refs only: safe for update --plan, which must not fetch."""
    def rev(ref):
        result = _git(git_cmd, cwd, "rev-parse", "--verify", ref + "^{commit}")
        return result.stdout.strip() if result.returncode == 0 else None

    head, tip = rev("HEAD"), rev(target)
    shallow = _git(git_cmd, cwd, "rev-parse", "--is-shallow-repository").stdout.strip() == "true"
    if not head or not tip:
        return HistoryAssessment(target, head, tip, "unknown", shallow=shallow)
    same_tree = _git(git_cmd, cwd, "diff", "--quiet", head, tip, "--").returncode == 0
    if head == tip:
        relation = "equal"
    elif _git(git_cmd, cwd, "merge-base", "--is-ancestor", head, tip).returncode == 0:
        relation = "fast_forward"
    elif _git(git_cmd, cwd, "merge-base", "--is-ancestor", tip, head).returncode == 0:
        relation = "local_ahead"
    elif shallow:
        relation = "incomplete_history"
    else:
        base = _git(git_cmd, cwd, "merge-base", head, tip)
        relation = "diverged" if base.returncode == 0 else "unrelated" if base.returncode == 1 else "unknown"
    counts = _git(git_cmd, cwd, "rev-list", "--left-right", "--count", f"{head}...{tip}")
    values = counts.stdout.split()
    left, right = (map(int, values) if counts.returncode == 0 and len(values) == 2 else (None, None))
    return HistoryAssessment(target, head, tip, relation, same_tree, left, right, shallow)


def has_fork_ancestry(git_cmd, cwd: Path) -> bool:
    """True when HEAD descends from :data:`FORK_ROOT_COMMIT`.

    ``merge-base --is-ancestor`` answers 0 (yes) / 1 (no); anything else means the
    object is unknown here, which is "no fork history" for a full clone and "cannot
    tell, keep guarding" for a shallow one.
    """
    result = _git(git_cmd, cwd, "merge-base", "--is-ancestor", FORK_ROOT_COMMIT, "HEAD")
    if result.returncode in (0, 1):
        return result.returncode == 0
    return _git(git_cmd, cwd, "rev-parse", "--is-shallow-repository").stdout.strip() == "true"


def guard_fork_history(git_cmd, cwd: Path, target: str) -> HistoryAssessment:
    """Refuse non-FF fork updates, preserving both known tips atomically first.

    Deliberately no expiry: these refs may be the only surviving attribution or
    local work after a remote fold. Never push, reset, rebase, stash or checkout.
    A checkout without fork history (:func:`has_fork_ancestry`) is not guarded.
    """
    history = assess_history(git_cmd, cwd, target)
    if history.relationship in {"equal", "fast_forward", "local_ahead"}:
        return history
    if not has_fork_ancestry(git_cmd, cwd):
        return history
    print(HISTORY_REVIEW_BANNER)
    refs = []
    prefix = "refs/hermes-update-backups/review-" + uuid.uuid4().hex
    commands = ["start"]
    for suffix, sha in (("local", history.head), ("remote", history.target_head)):
        if sha:
            ref = f"{prefix}/{suffix}"
            refs.append(ref)
            commands.append(f"create {ref} {sha}")
    if refs:
        commands.extend(["prepare", "commit"])
        result = _git(git_cmd, cwd, "update-ref", "--stdin", input="\n".join(commands) + "\n")
        if result.returncode:
            print("Update stopped: recovery refs could not be written. Checkout unchanged.")
            raise SystemExit(HISTORY_REVIEW_EXIT)
    print(f"Update stopped: fork history is {history.relationship} against {target}.")
    if history.same_tree:
        print("The tracked trees match, but ancestry differs (possibly folded commits).")
        print("Tree equivalence does not make this a fast-forward or authorize a reset.")
    print("Recovery refs: " + (", ".join(refs) if refs else "no commit tips could be resolved"))
    print("Review contributors, merge ancestry and patch equivalence on a dedicated codex/ branch.")
    print("Keep consolidation as a review series; deliver a history-preserving descendant of published main.")
    print("No checkout files or published branches were changed. --yes does not override this guard.")
    raise SystemExit(HISTORY_REVIEW_EXIT)

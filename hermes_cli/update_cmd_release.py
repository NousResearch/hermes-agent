"""Official-release (``hermes update --version``) git plumbing.

Leaf module of the update pipeline: tag grammar, the scoped canonical-repo tag fetch, commit
resolution, and transactional restore of the pre-update checkout identity. Orchestration
(``_pull_release_update`` etc.) stays in ``update_cmd.py``; shared names resolve through
``hermes_cli.update_cmd`` at call time so test monkeypatches on that namespace stay effective.
"""

import datetime
import os
import re
from pathlib import Path
import subprocess

from hermes_cli.update_cmd_stash import _AUTOSTASH_NAME_PREFIX


def _u():
    """Lazy ``hermes_cli.update_cmd`` handle (same pattern as ``_m()``): keeps imports one-way
    and keeps ``update_cmd.<name>`` monkeypatches effective."""
    from hermes_cli import update_cmd
    return update_cmd


# Public releases are date-versioned (vYYYY.M.D with an optional .N respin). Keep ``--version``
# narrower than Git's tag/refspec grammar: this installs official Hermes releases, not arbitrary
# repository objects. The regex owns the strict formatting (no zero-padding, no partial dates);
# the Gregorian check below owns whether the date exists at all.
_OFFICIAL_RELEASE_TAG_RE = re.compile(
    r"^v(?P<year>[1-9][0-9]{3})\.(?P<month>[1-9]|1[0-2])\.(?P<day>[1-9]|[12][0-9]|3[01])"
    r"(?:\.[1-9][0-9]*)?$")


def _official_release_tag(value: str) -> str:
    """Return an official release-shaped tag or raise ``ValueError``."""
    match = _OFFICIAL_RELEASE_TAG_RE.fullmatch(value)
    if not match:
        raise ValueError("expected an official release such as v2026.8.31 or v2026.8.31.2")
    try:
        datetime.date(int(match["year"]), int(match["month"]), int(match["day"]))
    except ValueError:
        raise ValueError("the release date does not exist on the Gregorian calendar") from None
    return value


def _update_tag_ref(tag: str) -> str:
    """Local ref name for an official release target."""
    return f"refs/tags/{tag}"


def _update_tag_refspec(tag: str) -> str:
    """Scoped fetch refspec for one release tag."""
    ref = _update_tag_ref(tag)
    return f"{ref}:{ref}"


def _update_peeled_tag_ref(tag: str) -> str:
    """Commit object behind a possibly annotated release tag."""
    return f"{_update_tag_ref(tag)}^{{commit}}"


def _fetch_official_release_tag(
    git_cmd: list, cwd: Path, tag: str, depth_args: tuple = ()
) -> subprocess.CompletedProcess:
    """Fetch exactly one canonical release tag, without auto-following others.

    ``--no-tags`` stops git from dragging in the full tag namespace; the forced refspec makes
    the canonical repo authoritative, so a counterfeit local tag of the same name is replaced
    rather than silently kept. The URL is pinned to ``OFFICIAL_REPO_URL`` — releases never come
    from a fork's origin.
    """
    tag = _official_release_tag(tag)
    tag_ref = _update_tag_ref(tag)
    if _u()._git_run(git_cmd, ["check-ref-format", tag_ref], cwd).returncode != 0:
        raise ValueError(f"invalid release tag: {tag}")
    return _u()._git_run(
        git_cmd,
        ["fetch", "--no-tags", *depth_args, _u().OFFICIAL_REPO_URL, f"+{_update_tag_refspec(tag)}"],
        cwd, network=True)


def _resolve_release_commit(git_cmd: list, cwd: Path, tag: str) -> "str | None":
    """Resolve ``tag`` to a commit SHA, rejecting tags of trees or blobs."""
    result = _u()._git_run(
        git_cmd, ["rev-parse", "--verify", "--quiet", _update_peeled_tag_ref(tag)], cwd)
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _release_checkout_required(*, current_branch: str, head_sha: str, release_sha: str) -> bool:
    """True unless HEAD is already detached at the requested release commit. An attached branch
    sitting at the release commit still requires a checkout: ``--version`` promises a detached
    release checkout, and staying attached would let the next branch pull silently move it."""
    return current_branch != "HEAD" or head_sha != release_sha


#: Git in-progress operation markers (``rev-parse --git-path`` names → refusal reason).
#: Presence of any of them means a merge/rebase/cherry-pick/revert/bisect is mid-flight.
#: ``sequencer`` covers the marker-less tail of a multi-commit cherry-pick/revert (a
#: conflicted pick committed manually consumes CHERRY_PICK_HEAD and leaves a clean status,
#: yet ``--continue``/``--abort`` still apply). ``--git-path`` resolves per checkout, so
#: linked-worktree layouts (markers under ``.git/worktrees/<name>/``) probe correctly.
_RELEASE_BLOCKING_GIT_OPERATIONS = (
    ("MERGE_HEAD", "a merge is in progress"),
    ("CHERRY_PICK_HEAD", "a cherry-pick is in progress"),
    ("REVERT_HEAD", "a revert is in progress"),
    ("BISECT_LOG", "a bisect is in progress"),
    ("rebase-merge", "a rebase is in progress"),
    ("rebase-apply", "a rebase or am is in progress"),
    ("sequencer", "a cherry-pick or revert sequence is in progress"),
)

_RELEASE_STATE_UNVERIFIABLE = "the repository state could not be verified"

#: The autostash producer runs ``git stash push -m <prefix><UTC %Y%m%d-%H%M%S>``, which git
#: subjects as ``On <branch or (no branch)>: <message>``. Branch names cannot contain a
#: colon, so the message is exactly what follows the first ``": "``.
_AUTOSTASH_STASH_MESSAGE_RE = re.compile(
    re.escape(_AUTOSTASH_NAME_PREFIX) + r"\d{8}-\d{6}")


def _is_hermes_updater_autostash_subject(subject: str) -> bool:
    """Whether a ``stash list --format=%s`` line is an unresolved Hermes updater autostash.

    Matches the COMPLETE generated subject grammar, never a substring: an ordinary user
    stash whose message merely contains the words (or the generated name inside a longer
    message) must not block a release update. ``WIP on`` subjects are unnamed user stashes
    by construction — the updater always passes ``-m``."""
    head, sep, message = subject.partition(": ")
    if not sep or not head.startswith("On "):
        return False
    return _AUTOSTASH_STASH_MESSAGE_RE.fullmatch(message) is not None


def _release_apply_git_state_block_reason(git_cmd: list, cwd: Path) -> "str | None":
    """Reason the exact-release APPLY path must refuse, or None when Git state is clean and
    settled. Read-only by construction; every probe failure fails closed.

    Exact release updates are exceptional transitions: rather than rely on the autostash to
    absorb whatever state the checkout is in, they require a clean index and working tree
    (untracked files included), no in-progress merge/rebase/cherry-pick/revert/bisect or
    unmerged index, and no parked Hermes update autostash entries — a leftover autostash
    signals unresolved prior update work the user must inspect first. Ordinary user stashes
    never block."""
    try:
        for git_path_name, reason in _RELEASE_BLOCKING_GIT_OPERATIONS:
            probe = _u()._git_run(git_cmd, ["rev-parse", "--git-path", git_path_name], cwd)
            if probe.returncode != 0:
                return _RELEASE_STATE_UNVERIFIABLE
            raw = probe.stdout.strip()
            if raw:
                marker = Path(raw)
                if (marker if marker.is_absolute() else cwd / marker).exists():
                    return reason
        unmerged = _u()._git_run(git_cmd, ["ls-files", "--unmerged"], cwd)
        if unmerged.returncode != 0:
            return _RELEASE_STATE_UNVERIFIABLE
        if unmerged.stdout.strip():
            return "the index has unmerged (conflicted) entries"
        status = _u()._git_run(git_cmd, ["status", "--porcelain"], cwd)
        if status.returncode != 0:
            return _RELEASE_STATE_UNVERIFIABLE
        if status.stdout.strip():
            return "the working tree or index has uncommitted changes (untracked files included)"
        stash_list = _u()._git_run(git_cmd, ["stash", "list", "--format=%s"], cwd)
        if stash_list.returncode != 0:
            return _RELEASE_STATE_UNVERIFIABLE
        if any(_is_hermes_updater_autostash_subject(line) for line in stash_list.stdout.splitlines()):
            return "one or more Hermes update autostash entries are parked in `git stash`"
    except (OSError, subprocess.SubprocessError):
        return _RELEASE_STATE_UNVERIFIABLE
    return None


def _rollback_blocked_by_concurrent_worktree_state(git_cmd: list, cwd: Path) -> bool:
    """True when the pre-rollback cleanliness re-probe finds — or cannot rule out —
    concurrent index/worktree/untracked changes. The release transaction started from a
    verified-clean tree, so ANY dirt observed here appeared while HEAD was detached at the
    release; a rollback checkout over it could erase another actor's work. Fail closed:
    the caller leaves HEAD at the release with every concurrent byte untouched."""
    try:
        status = _u()._git_run(git_cmd, ["status", "--porcelain"], cwd)
    except (OSError, subprocess.SubprocessError):
        return True
    return status.returncode != 0 or bool(status.stdout.strip())


def _restore_checkout_identity(git_cmd: list, cwd: Path, start_branch: str, start_sha: str) -> bool:
    """Restore and verify the original commit AND attached/detached identity.

    ``start_branch`` is the literal ``rev-parse --abbrev-ref`` answer: "HEAD" means the update
    began detached and must end detached; anything else re-attaches that branch at *start_sha*.

    The attached path NEVER rewrites the branch ref. The release checkout is detached, so
    ``refs/heads/<start_branch>`` still equals *start_sha* unless a concurrent git actor moved
    it — and rewinding a moved branch (the old ``reset --hard``) would orphan that actor's
    commits, with the reflog as the only recovery path. Fail closed instead: a missing, moved,
    or unverifiable ref returns False with the branch untouched.

    Rollback never uses ``--force``: the transaction started clean, so anything the
    cleanliness re-probe immediately above the checkout finds (staged, tracked, or untracked)
    is another actor's concurrent work and forces a fail-closed return with HEAD left at the
    release and the bytes untouched. An edit landing in the probe/checkout race window is
    caught by the checkout itself — non-force (plus ``--no-overwrite-ignore``) refuses the
    overwrite rather than erasing it. Checkout moves HEAD but structurally cannot rewrite the
    branch, so no atomic compare-and-swap is needed: a branch move that lands inside the
    verify/checkout window puts HEAD on the moved tip and the post-checkout verification
    below reports failure with the concurrent commit still at the branch tip.
    """
    if start_branch == "HEAD":
        if _rollback_blocked_by_concurrent_worktree_state(git_cmd, cwd):
            return False
        if _u()._git_run(
                git_cmd, ["checkout", "--detach", "--no-overwrite-ignore", start_sha],
                cwd).returncode != 0:
            return False
    else:
        branch_ref = _u()._git_run(
            git_cmd, ["rev-parse", "--verify", "--quiet", f"refs/heads/{start_branch}"], cwd)
        if branch_ref.returncode != 0 or branch_ref.stdout.strip() != start_sha:
            return False
        if _rollback_blocked_by_concurrent_worktree_state(git_cmd, cwd):
            return False
        if _u()._git_run(
                git_cmd, ["checkout", "--no-overwrite-ignore", start_branch],
                cwd).returncode != 0:
            return False

    if _u()._capture_head_sha(git_cmd, cwd) != start_sha:
        return False
    symbolic = _u()._git_run(git_cmd, ["symbolic-ref", "--quiet", "--short", "HEAD"], cwd)
    if symbolic.returncode == 0:
        actual_branch = symbolic.stdout.strip()
    elif symbolic.returncode == 1:
        # ``--quiet`` makes rc 1 the documented "HEAD is not a symbolic ref" answer — the
        # only return code that PROVES a detached checkout. 128 (or anything else) is a
        # probe failure: the identity is unverified, so the restoration is not claimed.
        actual_branch = "HEAD"
    else:
        return False
    return actual_branch == start_branch


def _release_overwrite_collision_paths(
    git_cmd: list, cwd: Path, head_sha: str, release_sha: str) -> "list[str] | None":
    """Target-added paths that already exist in the working tree, or None when git cannot
    answer.

    The clean-tree preflight already refused tracked/staged/untracked changes, so a path
    that exists here despite being absent from HEAD's tree is an ignored user file — and
    ``git checkout`` overwrites ignored files silently by default. Plumbing-only tree
    comparison over explicit SHAs with no pathspec arguments: paths are literal on both
    sides (NUL-separated output, plain filesystem existence probe — no glob expansion).
    """
    try:
        result = _u()._git_run(
            git_cmd,
            ["diff-tree", "-r", "-z", "--name-only", "--no-renames", "--diff-filter=A",
             head_sha, release_sha],
            cwd)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return sorted(
        path for path in result.stdout.split("\0")
        if path and os.path.lexists(cwd / path))

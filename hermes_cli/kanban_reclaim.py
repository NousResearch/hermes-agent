"""Reclaim disk held by finished kanban cards: their git worktrees and the
sibling ``*-pi*.log`` / ``*-spec*.md`` scratch files left next to them.

Single Python implementation — the shell guard (``disk-guard.sh``) is expected
to call ``hermes kanban reclaim`` rather than keep its own copy.

The defect that caused the leak, pinned by tests:

Eligibility in the shell copy was ``git merge-base --is-ancestor <tip>
origin/main`` and nothing else. AgentPod squash-merges, so a merged branch's tip
is never an ancestor of main and NOTHING was ever eligible — measured on the
live host, all five worktrees died at that first gate and
``reclaim_merged_worktrees`` had removed 0 dirs over its entire log. A branch
whose local tip equals its remote ref has nothing unpushed and is safe to drop,
which is the arm added here.

:func:`path_in_use` enumerates processes in Python (never ``ps | grep``, which
can match its own argv) and excludes our own pid, so a path that does not exist
is not reported "in use". Note: this is hardening, NOT a fix for an observed
bug — the shell ``path_in_use`` was measured returning correct answers
(0/100 false positives, correct positive control), and an earlier claim that it
self-matched universally was a harness artifact and is retracted.

Every refusal carries a human-readable ``reason`` — a silent no-op is exactly
how the shell guard hid for months.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

TASK_ID_RE = re.compile(r"^t_[0-9a-f]{8}$")
_EMBEDDED_TASK_ID_RE = re.compile(r"t_[0-9a-f]{8}")

#: Card states whose workspace is dead weight.
DONE_STATUSES = frozenset({"done", "archived"})

#: Sibling scratch artefacts a finished card leaves next to its worktree.
SIBLING_LOG_GLOBS = ("*-pi*.log", "*-spec*.md", "spec-t_*.md", "pi-t_*.log")

#: Globs, relative to the workspace home, that hold agent worktrees. Never a
#: literal path — the base comes from the caller (or ``Path.home()``).
WORKTREE_ROOT_GLOBS = ("*-worktrees", "*/.worktrees")

_GIT_TIMEOUT = 60
_GH_TIMEOUT = 30


@dataclass
class ReclaimCandidate:
    task_id: str
    path: Path
    branch: Optional[str]
    size_mb: int


@dataclass
class ReclaimDecision:
    path: Path
    task_id: str
    removed: bool
    reason: str


# --- Process liveness (hardening; see module docstring) ----------------------

def _run(
    argv: Sequence[str], *, timeout: int = 20, cwd: Optional[str] = None,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            list(argv), capture_output=True, text=True, timeout=timeout, check=False,
            cwd=cwd,
        )
    except (OSError, subprocess.SubprocessError):
        return subprocess.CompletedProcess(list(argv), returncode=127, stdout="", stderr="")


def _ps_holder(spath: str, ignore_pids: set[int]) -> Optional[int]:
    """First pid (other than ours) whose argv mentions ``spath``."""
    res = _run(["ps", "-Ao", "pid=,args="])
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, args = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid in ignore_pids:
            continue
        # The probe itself carries the path in its argv on some platforms.
        if args.split(" ")[:1] == ["ps"] or " -Ao pid=,args=" in args:
            continue
        if spath in args:
            return pid
    return None


def _lsof_holder(spath: str, ignore_pids: set[int]) -> Optional[int]:
    """First pid holding an open file/cwd under ``spath``.

    ``lsof +D`` exits 1 on macOS even when it prints a holder, so the OUTPUT is
    the signal, never the exit code.
    """
    res = _run(["lsof", "-w", "+D", spath], timeout=30)
    for line in res.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[0] == "COMMAND":
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        if pid in ignore_pids:
            continue
        return pid
    return None


def path_holder_pid(path: Path) -> Optional[int]:
    """Pid of a live process holding ``path`` (argv mention or open fd/cwd), else None.

    Self-matching is impossible: we never shell out to grep, and our own pid is
    excluded explicitly.
    """
    try:
        spath = str(path.resolve())
    except OSError:
        spath = str(path)
    ignore = {os.getpid(), os.getppid()}
    if not Path(spath).exists():
        # Nothing can hold a path that is not there. Checking this first is the
        # negative control that the shell guard failed.
        return None
    pid = _ps_holder(spath, ignore)
    if pid is not None:
        return pid
    return _lsof_holder(spath, ignore)


def path_in_use(path: Path) -> bool:
    """True when a live process holds ``path``. ``/nonexistent/...`` is False."""
    return path_holder_pid(path) is not None


# --- Kanban DB ---------------------------------------------------------------

def _load_cards(db_path: Optional[Path]) -> dict[str, tuple[str, Optional[int]]]:
    """``{task_id: (status, completed_at)}`` for every card on the board."""
    from hermes_cli import kanban_db as kbc

    try:
        with kbc.connect_closing(db_path=db_path) as conn:
            rows = conn.execute("SELECT id, status, completed_at FROM tasks").fetchall()
    except (sqlite3.Error, OSError, PermissionError):
        return {}
    return {r["id"]: (r["status"], r["completed_at"]) for r in rows}


def _card_refusal(
    task_id: str, cards: dict[str, tuple[str, Optional[int]]], *, now: float, min_age_seconds: int,
    unit: str, min_label: str,
) -> Optional[str]:
    """Shared card-state guard. Returns a refusal reason, or None when eligible."""
    card = cards.get(task_id)
    if card is None:
        # Deliberately stricter than the shell guard: an id we cannot find is
        # how you delete somebody's live work.
        return "unknown task"
    status, completed_at = card
    if status not in DONE_STATUSES:
        return f"task status={status}, not done"
    if not completed_at:
        return "done but no completed_at timestamp"
    age = max(0.0, now - float(completed_at))
    if age < min_age_seconds:
        divisor = 3600 if unit == "h" else 86400
        return f"completed {int(age // divisor)}{unit} ago, under the {min_label} floor"
    return None


# --- Git predicates (defect B) ------------------------------------------------

def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return _run(["git", "-C", str(repo), *args], timeout=_GIT_TIMEOUT)


def _git_out(repo: Path, *args: str) -> Optional[str]:
    res = _git(repo, *args)
    return res.stdout.strip() if res.returncode == 0 else None


def repo_root_for(path: Path) -> Optional[Path]:
    """Main checkout backing ``path`` (a linked worktree resolves to its parent repo)."""
    common = _git_out(path, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if not common:
        return None
    gitdir = Path(common)
    return gitdir.parent if gitdir.name == ".git" else gitdir


def _remote_head(repo: Path, branch: str) -> Optional[str]:
    out = _git_out(repo, "ls-remote", "--heads", "origin", branch)
    if not out:
        return None
    return out.split()[0]


#: Refs tried, in order, when looking for "main".
_MAIN_REFS = ("origin/main", "origin/master", "main", "master")


def _main_ref(repo: Path) -> Optional[str]:
    """First of :data:`_MAIN_REFS` that resolves in ``repo``, else None."""
    for ref in _MAIN_REFS:
        if _git(repo, "rev-parse", "--verify", "--quiet", ref).returncode == 0:
            return ref
    return None


def _is_ancestor_of_main(repo: Path, tip: str) -> bool:
    for ref in _MAIN_REFS:
        if _git(repo, "rev-parse", "--verify", "--quiet", ref).returncode != 0:
            continue
        if _git(repo, "merge-base", "--is-ancestor", tip, ref).returncode == 0:
            return True
    return False


def _gh_pr_json(path: Path, branch: str) -> Optional[str]:
    """Raw ``gh pr list`` JSON for ``branch``, or None when the probe failed.

    Module-level and deliberately tiny so tests can replace it wholesale — unit
    tests must never need the network or a ``gh`` binary.

    ``gh`` has NO ``-C`` flag (that is git); it resolves the repo from the cwd,
    so the worktree is passed as ``cwd=``. Measured: ``gh -C . pr list ...``
    exits 1 with "unknown shorthand flag: 'C' in -C", which would have made this
    arm silently dead on every real host while the mocked unit tests stayed green.
    """
    res = _run(
        [
            "gh", "pr", "list",
            "--head", branch, "--state", "merged", "--limit", "10",
            "--json", "number,state,mergedAt,headRefOid",
        ],
        timeout=_GH_TIMEOUT,
        cwd=str(path),
    )
    if res.returncode != 0:
        return None
    return res.stdout


def merged_pr_for_tip(path: Path, branch: str, tip: str) -> Optional[str]:
    """Human string for a MERGED PR whose head sha is exactly ``tip``, else None.

    This is the arm that discriminates squash merges: GitHub deletes the head
    ref on merge, so ``ls-remote`` goes empty and ``git cherry`` still reports
    the rewritten commits as non-equivalent — but the PR records the exact sha
    that was merged. Any failure of the probe (no ``gh``, non-zero exit, network
    error, malformed JSON, empty list, sha mismatch) returns None: a failed
    probe is never evidence of a merge.
    """
    raw = _gh_pr_json(path, branch)
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, list):
        return None
    for pr in payload:
        if not isinstance(pr, dict):
            continue
        if pr.get("state") != "MERGED":
            continue
        merged_at = pr.get("mergedAt")
        if not merged_at:
            continue
        if pr.get("headRefOid") != tip:
            continue
        return f"PR #{pr.get('number')} merged {merged_at}"
    return None


def all_commits_equivalent_on_main(repo_or_path: Path, tip: str) -> bool:
    """True when every commit on ``tip`` already has an equivalent on main.

    Offline fallback for Arm A. ``git cherry`` marks a commit ``-`` when an
    equivalent patch is upstream and ``+`` when it is not. Empty output is NOT
    proof — no commits to compare says nothing, and the ancestor check above
    already covers that case.
    """
    ref = _main_ref(repo_or_path)
    if ref is None:
        return False
    res = _git(repo_or_path, "cherry", ref, tip)
    if res.returncode != 0:
        return False
    lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
    if not lines:
        return False
    return not any(ln.lstrip().startswith("+") for ln in lines)


def branch_safety_reason(
    path: Path, repo: Optional[Path], branch: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """``(refusal, safe_reason)`` for the checkout at ``path``.

    Exactly one of the two is non-None for a decidable checkout; both are None
    when there is simply nothing branch-shaped to lose.
    """
    if repo is None:
        return None, None  # not a git checkout — nothing branch-shaped to lose
    dirty = _git_out(path, "status", "--porcelain")
    if dirty:
        # Uncommitted work always wins, even over a merged PR.
        return f"uncommitted changes in {path.name}", None
    tip = _git_out(path, "rev-parse", "HEAD")
    if tip is None:
        return None, None  # unborn branch, no commits at all
    if _is_ancestor_of_main(repo, tip):
        return None, "merged into main"
    if branch is None:
        return "detached HEAD not merged into origin/main", None
    # Defect B: a squash-merged branch is never an ancestor of main, but its
    # tip matches the remote ref, so nothing is unpushed and it is safe to drop.
    if _remote_head(repo, branch) == tip:
        return None, f"tip pushed to origin/{branch}"
    # Arm A: the PR that carried this exact tip is merged (head ref since gone).
    pr_reason = merged_pr_for_tip(path, branch, tip)
    if pr_reason:
        return None, pr_reason
    # Arm B: offline patch-equivalence against main.
    if all_commits_equivalent_on_main(path, tip):
        return None, "all commits patch-equivalent on main (git cherry)"
    return f"unpushed commits on {branch}", None


def branch_refusal(path: Path, repo: Optional[Path], branch: Optional[str]) -> Optional[str]:
    """Reason the checkout at ``path`` must be kept, or None when safe to drop."""
    return branch_safety_reason(path, repo, branch)[0]


def current_branch(path: Path) -> Optional[str]:
    name = _git_out(path, "rev-parse", "--abbrev-ref", "HEAD")
    if not name or name == "HEAD":
        return None
    return name


# --- Roots and candidates -----------------------------------------------------

def default_roots(
    *, db_path: Optional[Path] = None, workspace_home: Optional[Path] = None,
) -> list[Path]:
    """Worktree roots, derived by glob and from the board — never a literal list."""
    roots: list[Path] = []

    def add(p: Path) -> None:
        try:
            rp = p.resolve()
        except OSError:
            return
        if rp.is_dir() and rp not in roots:
            roots.append(rp)

    home = workspace_home or (Path.home() / "workspace")
    if home.is_dir():
        for pattern in WORKTREE_ROOT_GLOBS:
            for match in sorted(home.glob(pattern)):
                add(match)
    # Parents of worktrees the board itself created (board repos may live
    # anywhere, so the DB is the only honest source for those).
    from hermes_cli import kanban_db as kbc
    try:
        with kbc.connect_closing(db_path=db_path) as conn:
            rows = conn.execute(
                "SELECT workspace_path FROM tasks WHERE workspace_kind = 'worktree' "
                "AND workspace_path IS NOT NULL"
            ).fetchall()
    except (sqlite3.Error, OSError, PermissionError):
        rows = []
    for row in rows:
        wp = row["workspace_path"]
        if wp:
            add(Path(wp).parent)
    return roots


def _dir_size_mb(path: Path) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path, onerror=lambda _e: None):
        for name in filenames:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue
    return total // (1024 * 1024)


def _contained(path: Path, roots: Sequence[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if path != root:
            return True
    return False


def _resolved_roots(roots: Iterable[Path]) -> list[Path]:
    out = []
    for r in roots:
        try:
            out.append(Path(r).expanduser().resolve())
        except OSError:
            continue
    return out


# --- Removal ------------------------------------------------------------------

def _registered_worktree(repo: Path, path: Path) -> bool:
    out = _git_out(repo, "worktree", "list", "--porcelain") or ""
    target = str(path)
    for line in out.splitlines():
        if line.startswith("worktree "):
            listed = line[len("worktree "):].strip()
            if listed == target or str(Path(listed).resolve()) == target:
                return True
    return False


def _remove_path(path: Path, repo: Optional[Path]) -> tuple[bool, str]:
    """Remove ``path``; registered worktrees go through git, never ``rmtree``."""
    if repo is not None and _registered_worktree(repo, path):
        res = _git(repo, "worktree", "remove", "--force", str(path))
        _git(repo, "worktree", "prune")
        if path.exists():
            err = (res.stderr or res.stdout).strip().splitlines()
            return False, f"git worktree remove failed: {err[-1] if err else 'unknown error'}"
        return True, "removed (git worktree remove)"
    try:
        shutil.rmtree(path)
    except OSError as exc:
        return False, f"rmtree failed: {exc}"
    return True, "removed"


# --- Public entry points ------------------------------------------------------

def reclaim_done_worktrees(
    *,
    roots: Sequence[Path],
    repos: Optional[Sequence[Path]] = None,
    min_age_hours: int = 6,
    dry_run: bool = False,
    now: Optional[float] = None,
    db_path: Optional[Path] = None,
) -> list[ReclaimDecision]:
    """Remove the worktrees of done cards under ``roots``; explain every refusal."""
    now = time.time() if now is None else now
    rroots = _resolved_roots(roots)
    cards = _load_cards(db_path)
    hint_repos = [Path(r).expanduser() for r in (repos or [])]
    decisions: list[ReclaimDecision] = []

    for root in rroots:
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            name = entry.name
            if not TASK_ID_RE.match(name):
                decisions.append(ReclaimDecision(entry, name, False, "not a task-named directory"))
                continue
            decisions.append(
                _decide_worktree(
                    entry, name, cards, rroots, hint_repos,
                    now=now, min_age_hours=min_age_hours, dry_run=dry_run,
                )
            )
    return decisions


def _decide_worktree(
    entry: Path, task_id: str, cards, roots: Sequence[Path], hint_repos: Sequence[Path],
    *, now: float, min_age_hours: int, dry_run: bool,
) -> ReclaimDecision:
    refusal = _card_refusal(
        task_id, cards, now=now, min_age_seconds=min_age_hours * 3600,
        unit="h", min_label=f"{min_age_hours}h",
    )
    if refusal:
        return ReclaimDecision(entry, task_id, False, refusal)

    try:
        real = entry.resolve()
    except OSError as exc:
        return ReclaimDecision(entry, task_id, False, f"cannot resolve path: {exc}")
    if not _contained(real, roots):
        return ReclaimDecision(entry, task_id, False, "outside the worktrees root")

    holder = path_holder_pid(real)
    if holder is not None:
        return ReclaimDecision(entry, task_id, False, f"in use by pid {holder}")

    repo = repo_root_for(real)
    if repo is None:
        for hint in hint_repos:
            if _registered_worktree(hint, real):
                repo = hint
                break
    branch = current_branch(real) if repo is not None else None
    refusal, safe_reason = branch_safety_reason(real, repo, branch)
    if refusal:
        return ReclaimDecision(entry, task_id, False, refusal)
    # Why it was considered safe travels with the decision: a removal whose
    # justification is invisible is indistinguishable from a bug.
    why = f"; {safe_reason}" if safe_reason else ""

    if dry_run:
        return ReclaimDecision(
            entry, task_id, False, f"would remove ({_dir_size_mb(real)} MB, dry run{why})"
        )
    candidate = ReclaimCandidate(task_id=task_id, path=real, branch=branch, size_mb=_dir_size_mb(real))
    ok, reason = _remove_path(candidate.path, repo)
    if ok:
        suffix = f" ({safe_reason})" if safe_reason else ""
        reason = (
            f"{reason}, {candidate.size_mb} MB reclaimed from "
            f"{candidate.branch or 'detached HEAD'}{suffix}"
        )
    return ReclaimDecision(entry, task_id, ok, reason)


def reclaim_stale_sibling_logs(
    *, root: Path, min_age_days: int = 7, dry_run: bool = False,
    now: Optional[float] = None, db_path: Optional[Path] = None,
) -> list[ReclaimDecision]:
    """Move finished cards' sibling logs/specs into ``<root>/logs/``.

    Move, never delete — for some runs these files are the only record.
    """
    now = time.time() if now is None else now
    try:
        rroot = Path(root).expanduser().resolve()
    except OSError:
        return []
    cards = _load_cards(db_path)
    logs_dir = rroot / "logs"
    decisions: list[ReclaimDecision] = []

    seen: set[Path] = set()
    for pattern in SIBLING_LOG_GLOBS:
        for entry in sorted(rroot.glob(pattern)):
            if entry in seen or not entry.is_file():
                continue
            seen.add(entry)
            match = _EMBEDDED_TASK_ID_RE.search(entry.name)
            if not match:
                decisions.append(ReclaimDecision(entry, entry.name, False, "not a task-named file"))
                continue
            task_id = match.group(0)
            refusal = _card_refusal(
                task_id, cards, now=now, min_age_seconds=min_age_days * 86400,
                unit="d", min_label=f"{min_age_days}d",
            )
            if refusal:
                decisions.append(ReclaimDecision(entry, task_id, False, refusal))
                continue
            try:
                real = entry.resolve()
            except OSError as exc:
                decisions.append(ReclaimDecision(entry, task_id, False, f"cannot resolve path: {exc}"))
                continue
            if not _contained(real, [rroot]):
                decisions.append(ReclaimDecision(entry, task_id, False, "outside the worktrees root"))
                continue
            if dry_run:
                decisions.append(ReclaimDecision(entry, task_id, False, "would move to logs/ (dry run)"))
                continue
            try:
                logs_dir.mkdir(parents=True, exist_ok=True)
                dest = _free_name(logs_dir / entry.name)
                shutil.move(str(real), str(dest))
            except OSError as exc:
                decisions.append(ReclaimDecision(entry, task_id, False, f"move failed: {exc}"))
                continue
            decisions.append(ReclaimDecision(entry, task_id, True, f"moved to {dest}"))
    return decisions


def _free_name(dest: Path) -> Path:
    if not dest.exists():
        return dest
    n = 1
    while (candidate := dest.with_name(f"{dest.name}.{n}")).exists():
        n += 1
    return candidate


# --- Free-space accounting + the board record --------------------------------

#: Volume whose free space the reclaim reports. Derived from the reclaim roots
#: at call time, never a literal mount point.
def free_space_mb(path: Path) -> Optional[int]:
    """Free MB on the volume holding ``path``; ``None`` when it cannot be read.

    ``shutil.disk_usage`` is the same number ``df`` prints (statvfs f_bavail),
    so the before/after pair recorded on the card is comparable to a hand-run
    ``df -m`` and does not depend on parsing ``df`` output.
    """
    probe = Path(path).expanduser()
    while True:
        try:
            return int(shutil.disk_usage(probe).free // (1024 * 1024))
        except (OSError, ValueError):
            if probe.parent == probe:
                return None
            probe = probe.parent


def format_reclaim_report(
    decisions: Sequence[ReclaimDecision], *, before_mb: Optional[int], after_mb: Optional[int],
) -> str:
    """The body recorded on each reclaimed card: before/after free space + what went."""
    removed = [d for d in decisions if d.removed]

    def gi(mb: Optional[int]) -> str:
        return "unknown" if mb is None else f"{mb}MB ({mb / 1024:.1f}Gi)"

    delta = (
        "unknown" if before_mb is None or after_mb is None
        else f"{after_mb - before_mb:+d}MB"
    )
    lines = [
        "Automated worktree reclaim (`hermes kanban reclaim`).",
        "",
        f"- free before: {gi(before_mb)}",
        f"- free after:  {gi(after_mb)}",
        f"- delta:       {delta}",
        "",
        f"Removed {len(removed)} of {len(decisions)} candidate director(ies):",
    ]
    lines += [f"- {d.path}: {d.reason}" for d in removed] or ["- (none)"]
    return "\n".join(lines)


def record_reclaim_comments(
    decisions: Sequence[ReclaimDecision], *, before_mb: Optional[int], after_mb: Optional[int],
    db_path: Optional[Path] = None, author: str = "disk-guard",
) -> int:
    """Comment the before/after free space on every card whose worktree was removed.

    Deliverable 1 of t_a3cc342c: the evidence has to land on the board from the
    scheduled tick, not from a human running a script by hand. Returns the
    number of comments written. Never raises — a board write failing must not
    turn a successful reclaim into an error.
    """
    removed = [d for d in decisions if d.removed]
    if not removed:
        return 0
    body = format_reclaim_report(decisions, before_mb=before_mb, after_mb=after_mb)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db as kbc

    written = 0
    try:
        with kbc.connect_closing(db_path=db_path) as conn:
            for d in removed:
                if not TASK_ID_RE.match(d.task_id):
                    continue
                try:
                    kb.add_comment(conn, d.task_id, author, body)
                    written += 1
                except (sqlite3.Error, ValueError):
                    continue
    except (sqlite3.Error, OSError):
        return written
    return written


def format_decisions(decisions: Sequence[ReclaimDecision]) -> list[str]:
    """One human line per decision — refusals included, that is the whole point."""
    return [
        f"  {'removed ' if d.removed else 'kept    '} {d.path}: {d.reason}"
        for d in decisions
    ]

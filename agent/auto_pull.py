"""Fast-forward a clean default-branch checkout so sessions don't start stale.

Opt-in. Called once at coding-session prompt build, before the workspace
snapshot is baked into the cached system prompt. Never stashes, rebases,
merges, switches branches, or force-updates.

Gates, in order:

- setting enabled (global ``agent.auto_pull`` or the owning project's flag)
- git repo on its default branch with an upstream
- clean working tree, no local commits ahead, not the running Hermes checkout
- behind the upstream after a scoped fetch
- ``git merge --ff-only``

Failures are warnings. The session still starts.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from hermes_cli._subprocess_compat import bounded_probe_run, noninteractive_git_env
from utils import is_truthy_value

logger = logging.getLogger("hermes.auto_pull")

# Fetch + ff-only has to finish before the system prompt is built. Stay well
# under typical session-start budgets; skip rather than stall.
_GIT_TIMEOUT = 8.0
_DEBOUNCE_SECONDS = 60.0

# Git roots we already fetched in this process, so a second prompt-build in
# the same session doesn't pay the network round-trip again.
_recent_fetches: dict[str, float] = {}


@dataclass(frozen=True)
class AutoPullResult:
    """Outcome of one auto-pull attempt. ``action`` is pulled / skipped / failed."""

    action: str
    reason: str
    commits: int = 0
    upstream: str = ""

    @property
    def pulled(self) -> bool:
        return self.action == "pulled"

    def snapshot_line(self) -> str:
        """One workspace-snapshot line, or ``""`` when nothing changed."""
        if not self.pulled:
            return ""
        n = self.commits
        unit = "commit" if n == 1 else "commits"
        target = self.upstream or "upstream"
        return f"- Auto-pulled {n} {unit} from {target} at session start"


def reset_auto_pull_state_for_tests() -> None:
    """Drop process-local debounce so tests don't leak across cases."""
    _recent_fetches.clear()


def auto_pull_is_enabled(
    cwd: Optional[str | Path] = None,
    config: Optional[dict[str, Any]] = None,
) -> bool:
    """True when this workspace should auto-pull.

    A first-class project's ``auto_pull`` flag ORs with the global
    ``agent.auto_pull`` config. Either switch is enough. An unreadable
    projects.db names no project, so only the global switch counts.
    """
    from agent.coding_context import _agent_config_value, _resolve_cwd

    if is_truthy_value(_agent_config_value(config, "auto_pull", False, readonly=True)):
        return True
    from hermes_cli import projects_db as pdb

    try:
        with pdb.connect_closing() as conn:
            proj = pdb.project_for_path(conn, str(_resolve_cwd(cwd)))
    except (sqlite3.Error, OSError):
        logger.debug("auto-pull: projects.db unreadable", exc_info=True)
        return False
    return proj is not None and proj.auto_pull


@dataclass(frozen=True)
class _Checkout:
    """A clean default-branch checkout that passed every pre-fetch gate."""

    root: Path
    head: str
    upstream: str
    remote: str
    remote_branch: str


def maybe_auto_pull(
    cwd: Optional[str | Path] = None,
    *,
    enabled: Optional[bool] = None,
    config: Optional[dict[str, Any]] = None,
    running_root: Optional[Path] = None,
    now: Optional[float] = None,
) -> AutoPullResult:
    """Fast-forward ``cwd``'s repo when the gates pass.

    ``enabled`` short-circuits the project/config lookup when the caller
    already resolved it. ``running_root`` is the live Hermes checkout (tests
    inject a fake; production uses this source tree when it is a git repo).
    """
    if enabled is None:
        enabled = auto_pull_is_enabled(cwd, config)
    if not enabled:
        return AutoPullResult("skipped", "disabled")

    from agent.coding_context import _resolve_cwd

    gated = _pre_fetch_gate(_resolve_cwd(cwd), running_root)
    if isinstance(gated, AutoPullResult):
        return gated

    key = str(gated.root)
    clock = time.monotonic() if now is None else now
    last = _recent_fetches.get(key)
    if last is not None and (clock - last) < _DEBOUNCE_SECONDS:
        return AutoPullResult("skipped", "recently fetched")

    from hermes_cli.gitlock import clear_stale_git_locks, clear_stale_tmp_packs

    clear_stale_git_locks(gated.root)
    clear_stale_tmp_packs(gated.root)

    before = _git(gated.root, "rev-parse", "HEAD") or ""
    fetch = _git_run(gated.root, "fetch", "--quiet", gated.remote, gated.remote_branch, timeout=_GIT_TIMEOUT)
    _recent_fetches[key] = clock
    if fetch is None or fetch.returncode != 0:
        logger.warning("auto-pull fetch failed in %s", gated.root)
        return AutoPullResult("failed", "fetch failed", upstream=gated.upstream)

    # Fetch can block while the user changes this checkout. Never merge that new state.
    if not before or not _unchanged_since_gate(gated, before):
        return AutoPullResult("skipped", "checkout changed during fetch")
    return _fast_forward(gated, before)


def _pre_fetch_gate(cwd: Path, running_root: Optional[Path]) -> AutoPullResult | _Checkout:
    """The skip reason, or the checkout every local (no-network) gate accepted."""
    root = _git_root(cwd)
    if root is None:
        return AutoPullResult("skipped", "not a git repo")

    live = running_root if running_root is not None else _default_running_root()
    if live is not None and root == live:
        return AutoPullResult("skipped", "running hermes checkout")

    stamp = _git(root, "status", "--porcelain=2", "--branch")
    if stamp is None:
        return AutoPullResult("failed", "git status failed")
    branch, counts = _parse_status(stamp)
    head = branch.get("head", "")
    if not head or head == "(detached)":
        return AutoPullResult("skipped", "detached HEAD")
    upstream = branch.get("upstream", "")
    remote, remote_branch = _split_upstream(upstream)
    if not remote or not remote_branch:
        return AutoPullResult("skipped", "no upstream")
    if any(counts.values()):
        return AutoPullResult("skipped", "dirty")
    if _as_int(branch.get("ahead")) > 0:
        return AutoPullResult("skipped", "local commits")
    if not _is_default_branch(root, head, upstream):
        return AutoPullResult("skipped", "not default branch")
    return _Checkout(root, head, upstream, remote, remote_branch)


def _unchanged_since_gate(gated: _Checkout, before: str) -> bool:
    status = _git(gated.root, "status", "--porcelain=2", "--branch")
    if status is None:
        return False
    branch, counts = _parse_status(status)
    return (branch.get("head") == gated.head and branch.get("upstream") == gated.upstream
            and not any(counts.values()) and _as_int(branch.get("ahead")) <= 0
            and _git(gated.root, "rev-parse", "HEAD") == before)


def _fast_forward(gated: _Checkout, before: str) -> AutoPullResult:
    behind = _behind_count(gated.root)
    if behind <= 0:
        return AutoPullResult("skipped", "up to date", upstream=gated.upstream)

    merged = _git_run(gated.root, "merge", "--ff-only", "@{u}", timeout=_GIT_TIMEOUT)
    if merged is None or merged.returncode != 0:
        logger.warning("auto-pull ff-only failed in %s", gated.root)
        return AutoPullResult("failed", "not fast-forward", commits=behind, upstream=gated.upstream)

    after = _git(gated.root, "rev-parse", "HEAD") or ""
    commits = behind
    if after and before != after:
        counted = _git(gated.root, "rev-list", "--count", f"{before}..{after}")
        if counted and counted.isdigit():
            commits = int(counted)
    logger.info("auto-pulled %s commit(s) in %s from %s", commits, gated.root, gated.upstream)
    return AutoPullResult("pulled", "fast-forwarded", commits=commits, upstream=gated.upstream)


def _git_root(cwd: Path) -> Optional[Path]:
    try:
        current = cwd.resolve()
    except OSError:
        return None
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def _default_running_root() -> Optional[Path]:
    """The checkout this process is running from, if it is a git repo."""
    root = Path(__file__).resolve().parent.parent
    if (root / ".git").exists() or (root / "run_agent.py").is_file():
        try:
            return root.resolve()
        except OSError:
            return root
    return None


def _git(cwd: Path, *args: str, timeout: float = 2.5) -> Optional[str]:
    result = _git_run(cwd, *args, timeout=timeout)
    if result is None or result.returncode != 0:
        return None
    return (result.stdout or "").strip()


def _git_run(cwd: Path, *args: str, timeout: float = 2.5):
    return bounded_probe_run(
        ["git", "-C", str(cwd), *args],
        timeout=timeout,
        env=noninteractive_git_env(),
    )


def _parse_status(porcelain: str) -> tuple[dict[str, str], dict[str, int]]:
    branch: dict[str, str] = {}
    counts = {"staged": 0, "modified": 0, "untracked": 0, "conflicts": 0}
    for line in porcelain.splitlines():
        if line.startswith("# branch.head"):
            branch["head"] = line.split(maxsplit=2)[-1]
        elif line.startswith("# branch.upstream"):
            branch["upstream"] = line.split(maxsplit=2)[-1]
        elif line.startswith("# branch.ab"):
            parts = line.split()
            branch["ahead"], branch["behind"] = parts[2].lstrip("+"), parts[3].lstrip("-")
        elif line.startswith(("1 ", "2 ")):
            xy = line.split(maxsplit=2)[1]
            if xy[0] != ".":
                counts["staged"] += 1
            if xy[1] != ".":
                counts["modified"] += 1
        elif line.startswith("u "):
            counts["conflicts"] += 1
        elif line.startswith("? "):
            counts["untracked"] += 1
    return branch, counts


def _as_int(value: Optional[str]) -> int:
    try:
        return int(value or "0")
    except ValueError:
        return 0


def _split_upstream(upstream: str) -> tuple[str, str]:
    remote, sep, branch = upstream.partition("/")
    if not sep or not remote or not branch:
        return "", ""
    return remote, branch


def _is_default_branch(root: Path, head: str, upstream: str) -> bool:
    remote, _branch = _split_upstream(upstream)
    if remote:
        sym = _git(root, "symbolic-ref", "--quiet", f"refs/remotes/{remote}/HEAD")
        if sym:
            prefix = f"refs/remotes/{remote}/"
            default = sym[len(prefix):] if sym.startswith(prefix) else sym.rsplit("/", 1)[-1]
            if default:
                return head == default
    return head in {"main", "master"}


def _behind_count(root: Path) -> int:
    raw = _git(root, "rev-list", "--count", "HEAD..@{u}")
    return _as_int(raw)

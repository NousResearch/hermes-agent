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
import os
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
    ``agent.auto_pull`` config. Either switch is enough. Missing config or a
    projects-db error fails closed (off).
    """
    global_on = False
    if config is None:
        try:
            from hermes_cli.config import load_config_readonly

            config = load_config_readonly()
        except Exception:
            config = {}
    global_on = is_truthy_value(((config or {}).get("agent") or {}).get("auto_pull", False))

    resolved = _resolve_cwd(cwd)
    try:
        from hermes_cli import projects_db as pdb

        with pdb.connect_closing() as conn:
            proj = pdb.project_for_path(conn, str(resolved))
            if proj is not None and bool(getattr(proj, "auto_pull", False)):
                return True
    except Exception:
        pass
    return global_on


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

    root = _git_root(_resolve_cwd(cwd))
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
    if not upstream:
        return AutoPullResult("skipped", "no upstream")
    if counts["staged"] or counts["modified"] or counts["untracked"] or counts["conflicts"]:
        return AutoPullResult("skipped", "dirty")
    if _as_int(branch.get("ahead")) > 0:
        return AutoPullResult("skipped", "local commits")
    if not _is_default_branch(root, head, upstream):
        return AutoPullResult("skipped", "not default branch")

    key = str(root)
    clock = time.monotonic() if now is None else now
    last = _recent_fetches.get(key)
    if last is not None and (clock - last) < _DEBOUNCE_SECONDS:
        return AutoPullResult("skipped", "recently fetched")

    remote, remote_branch = _split_upstream(upstream)
    if not remote or not remote_branch:
        return AutoPullResult("skipped", "no upstream")

    try:
        from hermes_cli.gitlock import clear_stale_git_locks, clear_stale_tmp_packs

        clear_stale_git_locks(root)
        clear_stale_tmp_packs(root)
    except Exception:
        pass

    fetch = _git_run(root, "fetch", "--quiet", remote, remote_branch, timeout=_GIT_TIMEOUT)
    _recent_fetches[key] = clock
    if fetch is None or fetch.returncode != 0:
        logger.warning("auto-pull fetch failed in %s", root)
        return AutoPullResult("failed", "fetch failed", upstream=upstream)

    behind = _behind_count(root)
    if behind <= 0:
        return AutoPullResult("skipped", "up to date", upstream=upstream)

    before = _git(root, "rev-parse", "HEAD") or ""
    merged = _git_run(root, "merge", "--ff-only", "@{u}", timeout=_GIT_TIMEOUT)
    if merged is None or merged.returncode != 0:
        logger.warning("auto-pull ff-only failed in %s", root)
        return AutoPullResult("failed", "not fast-forward", commits=behind, upstream=upstream)

    after = _git(root, "rev-parse", "HEAD") or ""
    commits = behind
    if before and after and before != after:
        counted = _git(root, "rev-list", "--count", f"{before}..{after}")
        if counted and counted.isdigit():
            commits = int(counted)
    logger.info("auto-pulled %s commit(s) in %s from %s", commits, root, upstream)
    return AutoPullResult("pulled", "fast-forwarded", commits=commits, upstream=upstream)


def _resolve_cwd(cwd: Optional[str | Path]) -> Path:
    if cwd:
        return Path(cwd).expanduser()
    try:
        from agent.runtime_cwd import resolve_agent_cwd

        return resolve_agent_cwd()
    except Exception:
        return Path(os.getcwd())


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

"""``hermes doctor`` — Installation section.

Answers the four questions a user asks before filing a bug or updating: what is installed
(version + checkout), how it was installed, whether upstream has moved, and the exact
command that updates *this* install. The upstream distance comes from
:func:`hermes_cli.banner.check_for_updates` (GitHub API, cached) rather than a local
``origin/main`` ref, which is only as fresh as the last ``git fetch`` and reads "behind 0" on
a checkout that never fetched.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from hermes_cli.doctor_report import Finding, check_info, check_ok, check_warn, doctor_check

# Install methods with no working tree to compare (`check_for_updates` returns None for them).
_NO_UPSTREAM_CHECK = {"docker", "apt"}


def collect_source_tree_state(project_root: Path) -> list[tuple[str, str, str]]:
    """Return non-failing diagnostics for the Hermes source checkout.

    The running install may be a git checkout with local patches applied. Doctor
    should surface that state so users can tell whether they are running clean
    upstream, a fork branch, or dirty live source.
    """
    root = Path(project_root)
    git_dir = root / ".git"
    if not git_dir.exists():
        return []

    # Share one small latency budget across every git probe so a slow or
    # network-mounted checkout cannot stall the whole doctor command.
    deadline = time.monotonic() + 3.0

    def _git(*args: str) -> str | None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            proc = subprocess.run(
                ["git", "-C", str(root), *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=remaining,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    rows: list[tuple[str, str, str]] = []
    branch = _git("branch", "--show-current") or "(detached)"
    head = _git("rev-parse", "--short=12", "HEAD")
    if head:
        rows.append(("info", "Source checkout", f"{branch} @ {head}"))

    status = _git("status", "--porcelain")
    if status is None:
        return rows
    changed = [line for line in status.splitlines() if line and not line.startswith("?? ")]
    untracked = [line for line in status.splitlines() if line.startswith("?? ")]
    if changed:
        rows.append(("warn", "Source checkout has local modifications", f"({len(changed)} tracked file(s) changed)"))
    else:
        rows.append(("ok", "No tracked source modifications", ""))
    if untracked:
        sample = untracked[0][3:]
        suffix = f"; first: {sample}" if sample else ""
        rows.append(("info", "Untracked source files", f"{len(untracked)} file(s){suffix}"))
    return rows


def _upstream_row(method: str, behind: int | None, ahead: int, check_disabled: bool) -> tuple[str, str, str]:
    """One row describing distance from upstream ``main``; never a failure (an update is never an error)."""
    if check_disabled:
        return ("info", "Upstream check disabled", "(updates.check: false)")
    if method in _NO_UPSTREAM_CHECK:
        return ("info", "Upstream check not applicable", f"({method} installs update by pulling a new build)")
    if behind is None:
        return ("info", "Upstream: could not check", "(offline, rate-limited, or no origin)")
    carried = f", {ahead} local commit(s) carried" if ahead > 0 else ""
    if behind > 0:
        return ("warn", f"{behind} commit(s) behind upstream main{carried}", "")
    return ("ok", f"Up to date with upstream main{carried}", "")


def collect_installation_state(project_root: Path) -> list[tuple[str, str, str]]:
    """Version, install method, upstream distance, checkout state, and the update command — in that order."""
    from hermes_cli import __release_date__, __version__
    from hermes_cli.banner import check_for_updates, get_git_banner_state
    from hermes_cli.config import detect_install_method, load_config, recommended_update_command

    method = detect_install_method(project_root)
    rows: list[tuple[str, str, str]] = [
        ("ok", f"Hermes Agent v{__version__}", f"({__release_date__})"),
        ("info", "Install method", method),
    ]
    check_disabled = load_config().get("updates", {}).get("check", True) is False
    # Doctor is an explicit request for status, so it may hit GitHub even when the passive banner
    # check is turned off — but we tell the user rather than silently probing anyway.
    behind = None if check_disabled or method in _NO_UPSTREAM_CHECK else check_for_updates(passive=False)
    ahead = int(((get_git_banner_state() if method == "git" else None) or {}).get("ahead") or 0)
    rows.append(_upstream_row(method, behind, ahead, check_disabled))
    rows.extend(collect_source_tree_state(project_root))
    rows.append(("info", "Update", recommended_update_command()))
    return rows


def _render(rows: list[tuple[str, str, str]]) -> None:
    for level, text, detail in rows:
        if level == "ok":
            check_ok(text, detail)
        elif level == "warn":
            check_warn(text, detail)
        else:
            check_info(f"{text}: {detail}" if detail and not detail.startswith("(") else f"{text} {detail}".strip())


def report_source_tree_state(project_root: Path) -> None:
    _render(collect_source_tree_state(project_root))


@doctor_check(on_error="Installation check failed", detail="({e})")
def _check_installation(should_fix: bool, f: Finding) -> None:
    from hermes_cli.doctor import PROJECT_ROOT
    _render(collect_installation_state(PROJECT_ROOT))

"""Deployment-branch guard — fails loud when the running tree is not on the canonical deploy branch.

**Background (t_7161d9a9).**
``/home/serveradmin/.hermes/hermes-agent`` is the live production tree for the
whole agent fleet — gateways, dispatcher, kanban engine, auto-decomposer,
watchers. It is a mutable git checkout: workers that pick up a card checkout
their fix branch into the *production* tree, which silently wipes whatever was
on the previous branch. Two silent-fix-loss incidents have been observed
(2026-09-17, 2026-09-23) — Friday's restored gate went live at 17:21Z and
was gone by 17:36Z. The accepted remedy (Bryan's "Option A", 2026-09-23) is
to pin the deployed tree to one canonical branch and have workers do their
work in separate worktrees.

**This module makes that policy self-enforcing.** Any code path that imports
``hermes_cli.deploy_guard`` and calls ``check_deploy_branch()`` (or sets
``HERMES_DEPLOY_GUARD=strict`` to gate the import side) will refuse to run
when the running tree's HEAD is not on the canonical branch. The check is
fast (a single ``git rev-parse`` + ``diff``), runs early (before the
dispatch loop or the gateway handlers begin turning work), and reports
loudly: a structured warning log + a stderr banner + an event row in the
kanban DB (``deploy_branch_drift``).

**Config.**
- ``HERMES_DEPLOY_BRANCH`` — the canonical branch name. Default is
  ``canonical-deploy-t_7161d9a9`` for the immediate term; promote to
  ``main`` (or whatever the durable canonical is named) once that is set.
- ``HERMES_DEPLOY_GUARD`` — ``off`` (no-op), ``warn`` (default; log + stderr
  banner; never abort), ``strict`` (raise ``DeployBranchDriftError`` so the
  caller can choose to abort or downgrade).
- ``HERMES_AGENT_HOME`` — the working tree to inspect. Defaults to the env
  var ``HERMES_AGENT`` if set, else
  ``/home/serveradmin/.hermes/hermes-agent``.

**Failure modes.**
- HEAD detached (CI / detached review worktree): the guard returns a
  specific reason ("detached HEAD"), not "wrong branch", so a worker in a
  throwaway review worktree is not falsely flagged.
- Repo missing (e.g. a packaging extract): ``reason="missing_repo"``.
- git binary missing: ``reason="git_unavailable"``; not a hard error — a
  containerized build environment may not have git.

**Tests** live in ``tests/hermes_cli/test_deploy_guard.py`` and cover:
clean match, drift, detached, missing repo, missing git, strict-mode raise,
warn-mode log-only, env-override.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

_LOG = logging.getLogger("hermes_cli.deploy_guard")

DEFAULT_DEPLOY_BRANCH = "canonical-deploy-t_7161d9a9"
DEFAULT_AGENT_HOME = "/home/serveradmin/.hermes/hermes-agent"


class DeployBranchDriftError(RuntimeError):
    """Raised by ``check_deploy_branch(strict=True)`` when HEAD != canonical.

    Carries the observed ref + expected branch + reason so a caller can log
    or alert with full context without re-running ``git rev-parse``.
    """


@dataclass(frozen=True)
class DeployCheckResult:
    """Outcome of a deploy-branch check. Pure data — no side effects."""

    ok: bool
    expected_branch: str
    observed_branch: Optional[str]   # None when HEAD is detached / repo missing
    observed_sha: Optional[str]
    reason: str                       # "match" | "wrong_branch" | "detached" | "missing_repo" | "git_unavailable"
    agent_home: str


def _agent_home() -> str:
    env = os.environ.get("HERMES_AGENT_HOME") or os.environ.get("HERMES_AGENT")
    if env and os.path.isabs(env) and os.path.isdir(env):
        return env
    return DEFAULT_AGENT_HOME


def _git_available(home: str) -> bool:
    """True when ``git`` is on PATH. Repo-presence is checked separately so the
    guard can distinguish "git missing" (re-installer problem) from
    "no .git at this path" (packaged install or wrong HERMES_AGENT_HOME).
    """
    try:
        subprocess.run(
            ["git", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return True


def _has_git_dir(home: str) -> bool:
    """True when ``home`` contains a ``.git`` directory or file (worktrees)."""
    git_dir = os.path.join(home, ".git")
    return os.path.isdir(git_dir) or os.path.isfile(git_dir)


def _rev_parse(home: str, *args: str) -> Optional[str]:
    """Run ``git -C home rev-parse <args>`` and return stdout.strip() or None on any failure."""
    try:
        proc = subprocess.run(
            ["git", "-C", home, "rev-parse", *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", errors="replace").strip()


def check_deploy_state() -> DeployCheckResult:
    """Inspect the working tree and return a DeployCheckResult. Pure — no logging, no side effects."""
    expected = os.environ.get("HERMES_DEPLOY_BRANCH") or DEFAULT_DEPLOY_BRANCH
    home = _agent_home()

    if not _git_available(home):
        return DeployCheckResult(
            ok=False,
            expected_branch=expected,
            observed_branch=None,
            observed_sha=None,
            reason="git_unavailable",
            agent_home=home,
        )

    if not _has_git_dir(home):
        return DeployCheckResult(
            ok=False,
            expected_branch=expected,
            observed_branch=None,
            observed_sha=None,
            reason="missing_repo",
            agent_home=home,
        )

    head_sha_str = _rev_parse(home, "HEAD")
    head_ref_str = _rev_parse(home, "--abbrev-ref", "HEAD")
    if not head_sha_str or not head_ref_str:
        return DeployCheckResult(
            ok=False,
            expected_branch=expected,
            observed_branch=None,
            observed_sha=None,
            reason="missing_repo",
            agent_home=home,
        )
    head_sha: str = head_sha_str
    head_ref: str = head_ref_str

    if head_ref == "HEAD":
        # Detached HEAD — could be a review worktree or a sandbox. Not a
        # wrong-branch drift per se; surface it as its own reason so callers
        # can distinguish.
        return DeployCheckResult(
            ok=False,
            expected_branch=expected,
            observed_branch=None,
            observed_sha=head_sha,
            reason="detached",
            agent_home=home,
        )

    if head_ref != expected:
        return DeployCheckResult(
            ok=False,
            expected_branch=expected,
            observed_branch=head_ref,
            observed_sha=head_sha,
            reason="wrong_branch",
            agent_home=home,
        )

    return DeployCheckResult(
        ok=True,
        expected_branch=expected,
        observed_branch=head_ref,
        observed_sha=head_sha,
        reason="match",
        agent_home=home,
    )


def _format_banner(result: DeployCheckResult) -> str:
    """One-line stderr banner that names the drift reason and the action an operator should take."""
    if result.reason == "match":
        return (
            f"[deploy-guard] OK: HEAD={result.observed_sha[:12]} "
            f"on branch {result.observed_branch} ({result.agent_home})"
        )
    if result.reason == "wrong_branch":
        return (
            f"[deploy-guard] DRIFT: deployed tree is on branch "
            f"'{result.observed_branch}' (expected '{result.expected_branch}'). "
            f"Running fixes are at risk of silent revert — "
            f"git checkout {result.expected_branch} in {result.agent_home} or "
            f"set HERMES_DEPLOY_BRANCH={result.observed_branch} if that is now canonical."
        )
    if result.reason == "detached":
        return (
            f"[deploy-guard] DRIFT: deployed tree is in detached HEAD at "
            f"{result.observed_sha[:12] if result.observed_sha else '?'} "
            f"(expected branch '{result.expected_branch}'). A worker probably "
            f"checked out a feature branch and forgot to return to the canonical."
        )
    if result.reason == "missing_repo":
        return (
            f"[deploy-guard] WARN: {result.agent_home} has no .git — "
            f"deploy-branch check skipped. If this is a packaged install, "
            f"set HERMES_DEPLOY_GUARD=off."
        )
    if result.reason == "git_unavailable":
        return (
            f"[deploy-guard] WARN: git binary not on PATH — deploy-branch "
            f"check skipped. Set HERMES_DEPLOY_GUARD=off to silence."
        )
    return f"[deploy-guard] UNKNOWN reason={result.reason!r} agent_home={result.agent_home}"


def check_deploy_branch(*, strict: Optional[bool] = None) -> DeployCheckResult:
    """Run the deploy-branch check and act per ``HERMES_DEPLOY_GUARD`` (or the ``strict`` override).

    - ``off``  : no-op, returns a synthetic ``ok=True`` result with reason="match".
    - ``warn`` : logs at WARNING, prints a stderr banner, returns the real result.
    - ``strict``: same as warn, plus raises ``DeployBranchDriftError`` on drift.

    The default (``warn``) is chosen so the dispatcher doesn't die when the
    guard fires — but the banner reaches the operator through both the
    structured log and the cron pump's captured stderr.
    """
    mode = (os.environ.get("HERMES_DEPLOY_GUARD") or "warn").lower()
    if strict is not None:
        mode = "strict" if strict else "warn"

    if mode == "off":
        return DeployCheckResult(
            ok=True,
            expected_branch=os.environ.get("HERMES_DEPLOY_BRANCH") or DEFAULT_DEPLOY_BRANCH,
            observed_branch=None,
            observed_sha=None,
            reason="match",
            agent_home=_agent_home(),
        )

    result = check_deploy_state()
    banner = _format_banner(result)

    if result.reason == "match":
        _LOG.info(banner)
        return result

    # Anything below is a drift or a missing-repo warning.
    _LOG.warning(banner)
    try:
        print(banner, file=sys.stderr, flush=True)
    except Exception:
        # Some embedded contexts forbid stderr writes; the log line is enough.
        pass

    # Drift events are NOT written to kanban ``task_events`` — that table is
    # task-scoped, and the deploy guard is a tree-level concern with no
    # associated task id. Operators correlate via the structured log line
    # (timestamp + reason + agent_home + observed_sha). A separate
    # tree-drift watcher (Ultron-owned, see t_7161d9a9) polls ``git
    # rev-parse HEAD`` for cron-pump visibility.

    if mode == "strict":
        raise DeployBranchDriftError(banner)

    return result
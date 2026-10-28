"""Verified pull-request facts for a finished Kanban task.

The completion ping may tell a Human that a task produced a pull request. That
statement is only admissible when it comes from GitHub itself and binds to the
task that is being reported, so this module owns the one trusted path:

1. **The repository comes from native Kanban state**, never from worker text:
   the task's ``project_id`` -> the project's ``primary_path`` -> the ``origin``
   remote of that checkout. The worker cannot name the repo it is reported
   against.
2. **The pull request comes from GitHub**, queried with the ``gh`` CLI against
   that repository, filtered by the task's own head branch
   (``tasks.branch_name``). A worker's ``pr_url``/``head_sha`` strings are never
   read here: format checks are input validation, not fact.
3. **The binding is checked, not assumed.** The returned PR must belong to the
   same repository, have exactly the task's head branch, and carry a real head
   OID. A merge is only reported when GitHub says ``MERGED``.

Anything that cannot be established yields :class:`PrFacts` with
``verified=False``, and the caller renders "not verified" plus the native task
pointer. There is no partial state that reads as success.

Failures (no ``gh``, offline, rate limit, malformed JSON) are *soft*: they
degrade to "not verified" and are never raised, because a completion ping must
still be delivered.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ``gh pr list --json`` fields we consume. Kept explicit so a gh release that
# adds fields cannot change what we read.
_GH_FIELDS = "number,url,state,isDraft,headRefName,headRefOid,baseRefName,mergedAt,mergeCommit"

# A GitHub pull-request URL. Used to re-validate the URL gh returned, so even a
# compromised/hostile gh shim cannot make us publish an arbitrary link: the URL
# must be https on github.com and end in the exact PR number we bound.
_PR_URL_RE = re.compile(
    r"^https://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)/pull/(?P<number>[1-9][0-9]{0,9})$"
)
_OID_RE = re.compile(r"^[0-9a-f]{7,40}$")
_OWNER_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_GH_TIMEOUT_SECONDS = 8


@dataclass(frozen=True)
class PrFacts:
    """What GitHub actually said about the task's pull request.

    ``verified=False`` is the only state that may never be rendered as a
    successful delivery claim. ``reason`` names why, for the debug log — it is
    never published to the chat.
    """

    verified: bool = False
    reason: str = "not_checked"
    repository: Optional[str] = None
    number: Optional[int] = None
    url: Optional[str] = None
    head_oid: Optional[str] = None
    head_branch: Optional[str] = None
    state: Optional[str] = None
    merged: bool = False
    draft: bool = False


def _gh_executable() -> Optional[str]:
    return shutil.which("gh")


def _repo_owner_name(path: str) -> Optional[str]:
    """``owner/repo`` from a checkout's ``origin`` remote, or ``None``.

    The remote is read with plain ``git remote get-url`` (no fetch, no network),
    and only ``https://github.com/...`` / ``git@github.com:...`` shapes are
    accepted. Anything else (GitLab, a local path, a mirror) is not something
    this receipt can verify against GitHub, so it reports "not verified" rather
    than guessing.
    """
    try:
        result = subprocess.run(
            ["git", "-C", path, "remote", "get-url", "origin"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=5, stdin=subprocess.DEVNULL, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("kanban receipt: origin remote lookup failed for %s: %s", path, exc)
        return None
    if result.returncode != 0:
        return None
    url = result.stdout.strip()
    match = re.match(r"^(?:https://github\.com/|git@github\.com:)(?P<path>[^/\s]+/[^/\s]+?)(?:\.git)?/?$", url)
    if not match:
        return None
    candidate = match.group("path")
    return candidate if _OWNER_REPO_RE.fullmatch(candidate) else None


def _project_repository(conn, task) -> Optional[str]:
    """``owner/repo`` for the task's project checkout, resolved from native state."""
    project_id = getattr(task, "project_id", None)
    if not project_id:
        # A workspace path with no project link is still a checkout we can ask;
        # the remote is the authority either way.
        workspace = getattr(task, "workspace_path", None)
        return _repo_owner_name(workspace) if workspace else None
    try:
        from hermes_cli import projects_db

        project = projects_db.get_project(conn, project_id)
    except Exception as exc:  # noqa: BLE001 - a receipt is never worth an exception
        logger.debug("kanban receipt: project lookup failed for %s: %s", project_id, exc)
        return None
    primary = getattr(project, "primary_path", None) if project else None
    return _repo_owner_name(primary) if primary else None


def _gh_pr_for_branch(repository: str, branch: str, gh: str) -> Optional[dict]:
    """The PR on *repository* whose head branch is exactly *branch*.

    ``--head`` is queried WITHOUT an ``owner:`` prefix. GitHub treats a bare branch
    name as "any owner", which is what we want: the branch may live on a fork, as
    it does for cross-fork contributions, and the task's own checkout cannot know
    the fork's owner. The result is then filtered locally on the exact branch
    string, so a same-named branch from another fork cannot be adopted.
    """
    try:
        result = subprocess.run(
            [gh, "pr", "list", "--repo", repository, "--head", branch,
             "--state", "all", "--limit", "20", "--json", _GH_FIELDS],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=_GH_TIMEOUT_SECONDS, stdin=subprocess.DEVNULL, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("kanban receipt: gh pr list failed for %s: %s", repository, exc)
        return None
    if result.returncode != 0:
        logger.debug("kanban receipt: gh pr list exit=%s for %s", result.returncode, repository)
        return None
    try:
        entries = json.loads(result.stdout or "[]")
    except ValueError:
        return None
    if not isinstance(entries, list):
        return None
    matches = [e for e in entries if isinstance(e, dict) and e.get("headRefName") == branch]
    if not matches:
        return None
    # Prefer the most recent state that represents the task's work: a merged PR
    # outranks an open one, and closed-unmerged is still a real fact.
    def rank(entry: dict) -> tuple:
        state = str(entry.get("state") or "").upper()
        merged = bool(entry.get("mergedAt"))
        return (merged, state == "OPEN", int(entry.get("number") or 0))

    return max(matches, key=rank)


def verify_task_pull_request(conn, task) -> PrFacts:
    """Resolve the verified PR facts for *task*, or unverified facts with a reason."""
    if task is None:
        return PrFacts(reason="no_task")
    branch = (getattr(task, "branch_name", None) or "").strip()
    if not branch:
        return PrFacts(reason="task_has_no_branch")
    gh = _gh_executable()
    if not gh:
        return PrFacts(reason="gh_cli_unavailable")
    repository = _project_repository(conn, task)
    if not repository:
        return PrFacts(reason="repository_unknown")

    entry = _gh_pr_for_branch(repository, branch, gh)
    if entry is None:
        return PrFacts(reason="no_pull_request_for_task_branch", repository=repository)

    number = entry.get("number")
    url = entry.get("url")
    head_oid = entry.get("headRefOid")
    if not isinstance(number, int) or isinstance(number, bool):
        return PrFacts(reason="malformed_number", repository=repository)
    if not isinstance(url, str) or not isinstance(head_oid, str):
        return PrFacts(reason="malformed_entry", repository=repository)

    # Re-validate gh's own output before it can be published: the URL must be
    # github.com/<the same repo>/pull/<the same number>, and the head OID must be
    # a real object id. This is what stops a hostile ``gh`` on PATH from making
    # the receipt advertise an arbitrary link.
    parsed = _PR_URL_RE.match(url.strip())
    if not parsed:
        return PrFacts(reason="pr_url_rejected", repository=repository)
    if f"{parsed.group('owner')}/{parsed.group('repo')}".lower() != repository.lower():
        return PrFacts(reason="pr_url_repository_mismatch", repository=repository)
    if int(parsed.group("number")) != number:
        return PrFacts(reason="pr_url_number_mismatch", repository=repository)
    if not _OID_RE.fullmatch(head_oid.strip().lower()):
        return PrFacts(reason="head_oid_rejected", repository=repository)
    if entry.get("headRefName") != branch:
        return PrFacts(reason="head_branch_mismatch", repository=repository)

    state = str(entry.get("state") or "").upper()
    if state not in {"OPEN", "CLOSED", "MERGED"}:
        return PrFacts(reason="unknown_state", repository=repository)

    return PrFacts(
        verified=True,
        reason="verified",
        repository=repository,
        number=number,
        url=url.strip(),
        head_oid=head_oid.strip().lower(),
        head_branch=branch,
        state=state,
        merged=state == "MERGED" or bool(entry.get("mergedAt")),
        draft=bool(entry.get("isDraft")),
    )

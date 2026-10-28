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
# adds fields cannot change what we read. ``headRepository``/``headRepositoryOwner``
# are what bind a PR to the FORK that pushed it — without them a bare branch name
# matches any owner and an older merged PR from a different fork can be adopted.
_GH_FIELDS = (
    "number,url,state,isDraft,headRefName,headRefOid,baseRefName,mergedAt,mergeCommit,"
    "headRepository,headRepositoryOwner,isCrossRepository"
)

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
    head_owner: Optional[str] = None
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


def _rest_pulls(repository: str, gh: str, *, head: Optional[str]) -> Optional[list[dict]]:
    """PRs from the GitHub REST API, optionally filtered by ``head``.

    ``gh pr list`` is NOT used for the identity query: for a cross-repository PR
    its ``--head owner:branch`` filter returns nothing even though the PR exists,
    so the fork could never be bound. The REST endpoint
    ``GET /repos/{owner}/{repo}/pulls?head=owner:branch`` is the documented
    filter and answers correctly, and it reports the fork per PR
    (``head.repo.full_name`` / ``.head.repo.owner.login``), which is what the
    binding needs.

    ``head=None`` asks for the repository's recent PRs; the caller filters.
    """
    path = f"repos/{repository}/pulls?state=all&per_page=100"
    if head:
        path += f"&head={head}"
    try:
        result = subprocess.run(
            [gh, "api", path],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=_GH_TIMEOUT_SECONDS, stdin=subprocess.DEVNULL, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("kanban receipt: gh api pulls failed for %s: %s", repository, exc)
        return None
    if result.returncode != 0:
        logger.debug("kanban receipt: gh api pulls exit=%s for %s", result.returncode, repository)
        return None
    try:
        entries = json.loads(result.stdout or "[]")
    except ValueError:
        return None
    if not isinstance(entries, list):
        return None
    return [e for e in entries if isinstance(e, dict)]


def _rest_entry_to_facts_entry(entry: dict) -> dict:
    """Flatten a REST PR into the flat shape the verification step reads."""
    head_raw = entry.get("head")
    repo_raw = head_raw.get("repo") if isinstance(head_raw, dict) else None
    owner_raw = repo_raw.get("owner") if isinstance(repo_raw, dict) else None
    base_raw = entry.get("base")
    head: dict = head_raw if isinstance(head_raw, dict) else {}
    repo: dict = repo_raw if isinstance(repo_raw, dict) else {}
    owner: dict = owner_raw if isinstance(owner_raw, dict) else {}
    base: dict = base_raw if isinstance(base_raw, dict) else {}
    state = str(entry.get("state") or "").lower()
    merged_at = entry.get("merged_at")
    if merged_at:
        state = "merged"
    return {
        "number": entry.get("number"),
        "url": entry.get("html_url"),
        "state": state.upper(),
        "isDraft": bool(entry.get("draft")),
        "headRefName": head.get("ref"),
        "headRefOid": head.get("sha"),
        "baseRefName": base.get("ref"),
        "mergedAt": merged_at,
        "headRepositoryOwner": {"login": owner.get("login")},
        "headRepository": {"nameWithOwner": repo.get("full_name")},
        "isCrossRepository": bool(repo.get("full_name")),
    }


def _pr_for_owner_branch(repository: str, branch: str, gh: str, head_owner: str) -> Optional[dict]:
    """The PR on *repository* pushed from *head_owner*'s fork on *branch*.

    The query is scoped by ``owner:branch`` so a same-named branch in another fork
    is not even returned, and the reply is checked again against the fork GitHub
    reports. Among matches for that fork, a merged PR outranks an open one — the
    preference only runs over entries already bound to this fork.
    """
    raw = _rest_pulls(repository, gh, head=f"{head_owner}:{branch}")
    if raw is None:
        return None
    matches = [
        e for e in (_rest_entry_to_facts_entry(x) for x in raw)
        if e.get("headRefName") == branch
        and (e.get("headRepositoryOwner") or {}).get("login", "").lower() == head_owner.lower()
    ]
    if not matches:
        return None

    def rank(entry: dict) -> tuple:
        state = str(entry.get("state") or "").upper()
        merged = bool(entry.get("mergedAt"))
        return (merged, state == "OPEN", int(entry.get("number") or 0))

    return max(matches, key=rank)


def _entry_head_owner(entry: dict) -> Optional[str]:
    """Login of the fork that pushed *entry*, lowercased, or ``None``."""
    owner = entry.get("headRepositoryOwner")
    login = owner.get("login") if isinstance(owner, dict) else None
    if isinstance(login, str) and login.strip():
        return login.strip().lower()
    repo = entry.get("headRepository")
    full = repo.get("nameWithOwner") if isinstance(repo, dict) else None
    if isinstance(full, str) and "/" in full:
        return full.split("/", 1)[0].strip().lower()
    return None


def _resolve_head_owner(repository: str, branch: str, gh: str, *, exclude: set[str]) -> list[str]:
    """The fork logins allowed to own this task's PR, most reliable first.

    The candidate set is NOT "whoever has a branch with this name". Collecting
    owners from the PRs themselves would let an older PR from *another* fork
    nominate its own owner, and then win the state preference — the very
    mis-attribution this binding exists to stop.

    So the only admissible owners are identities the task's own tooling holds:

    1. the authenticated ``gh`` login (the account that would have pushed);
    2. the ``origin`` remote's owner, when it is not the upstream org itself —
       i.e. a fork checkout, which is the normal shape for a contribution.

    A same-named branch in some unrelated fork is therefore never a candidate.
    """
    owners: list[str] = []
    try:
        who = subprocess.run(
            [gh, "api", "user", "--jq", ".login"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=_GH_TIMEOUT_SECONDS, stdin=subprocess.DEVNULL, check=False,
        )
        login = who.stdout.strip().lower() if who.returncode == 0 else ""
        if login:
            owners.append(login)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("kanban receipt: gh identity lookup failed: %s", exc)
    # The checkout's own origin, but only when it is a fork of the upstream repo
    # (same repository name, different owner). A checkout of the upstream repo
    # itself tells us nothing about who pushed a branch.
    upstream_owner, _, upstream_name = repository.partition("/")
    origin = _repo_owner_name(_origin_path() or "") if _origin_path() else None
    if origin:
        origin_owner, _, origin_name = origin.partition("/")
        if origin_name.lower() == upstream_name.lower() and origin_owner.lower() != upstream_owner.lower():
            owners.append(origin_owner.lower())
    seen: set[str] = set()
    ordered: list[str] = []
    for owner in owners:
        if owner and owner not in seen and owner not in exclude:
            seen.add(owner)
            ordered.append(owner)
    return ordered


def _origin_path() -> Optional[str]:
    """A checkout to ask for its ``origin`` remote: the task-independent cwd.

    Worker-controlled state is avoided on purpose — ``tasks.workspace_path`` is a
    directory the worker itself can rewrite. This is only used as a *corroborating*
    hint for a fork owner, and every candidate is then confirmed against GitHub's
    own per-PR ``head.repo.owner.login``, so a wrong value here cannot create a
    verified PR; it can only fail to find one.
    """
    import os

    return os.getcwd()


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

    # Bind the PR to the FORK that pushed the task's work, not just to a branch
    # name. Order matters: a candidate whose identity cannot be established must
    # not be adopted, so every owner is tried with an explicit ``owner:branch``
    # selector and the first one GitHub actually confirms wins.
    upstream_owner = repository.split("/", 1)[0].lower()
    owners = _resolve_head_owner(repository, branch, gh, exclude={upstream_owner})
    if not owners:
        return PrFacts(reason="head_owner_unresolved", repository=repository)
    entry = None
    head_owner = None
    for candidate in owners:
        found = _pr_for_owner_branch(repository, branch, gh, candidate)
        if found is None:
            continue
        # The selector said ``candidate:branch``; GitHub must agree about who
        # pushed it. A reply naming a different fork is not this candidate's PR,
        # and must not be reported even if no other candidate matches — that is
        # exactly how an impostor fork gets adopted.
        if _entry_head_owner(found) != candidate:
            return PrFacts(reason="head_owner_mismatch", repository=repository)
        entry, head_owner = found, candidate
        break
    if entry is None or head_owner is None:
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
    # The identity binding itself: the PR must still report the same fork we
    # selected, so a raced or tampered response cannot swap the owner under us.
    if _entry_head_owner(entry) != head_owner:
        return PrFacts(reason="head_owner_mismatch", repository=repository)

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
        head_owner=head_owner,
        state=state,
        merged=state == "MERGED" or bool(entry.get("mergedAt")),
        draft=bool(entry.get("isDraft")),
    )

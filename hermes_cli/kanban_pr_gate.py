"""Re-evaluate kanban blocks whose premise is an external GitHub PR.

A worker that finishes its code and needs a sibling PR merged before it can
continue blocks with a reason like ``"merge PR #787 then unblock me"``. Nothing
in the dispatcher ever re-read that reason, so the card sat ``blocked`` until a
human board sweep noticed — measured 2026-09-21: six ``needs_input`` cards held
up to **12 hours** on gates whose referenced PRs had already merged.

This module makes that mechanical. Each dispatcher tick:

1. parse the LAST ``blocked``-family event's reason for PR references,
2. resolve each reference's state through ``gh`` (bounded + cached),
3. when EVERY referenced PR is MERGED, unblock the card, post the evidence as a
   comment, and write one ``gate_auto_resolved`` event so a sweep can count it.

Deliberate non-actions, each one a fail-safe:

* only ``needs_input`` / ``capability`` / ``dependency`` block kinds — a
  ``transient`` or un-typed block is not a "waiting on an external object" claim;
* a reason naming no PR is never touched, and burns no lookup budget;
* a bare ``#N`` with no unambiguous repo context is dropped rather than guessed;
* CLOSED-unmerged posts ONE advisory comment and never unblocks;
* any lookup failure is a no-op plus one WARN — never a page, never an unblock.

Harness safety
--------------
``gate_auto_resolved`` is a real state transition justified by real PR
evidence. A test or probe that stubs :func:`query_pr` holds a FABRICATED
oracle, and must therefore never be able to write that transition to a live
board. ``HERMES_HOME`` alone does not sandbox kanban — ``HERMES_KANBAN_DB``
outranks it (see :func:`hermes_cli.kanban_db.kanban_db_path`) — so a probe that
redirects only ``HERMES_HOME`` still resolves to production. On 2026-09-21 a
lock-timing probe did exactly that and wrote 20 ``gate_auto_resolved`` events
to the live board, falsely unblocking seven real cards.
:func:`assert_write_allowed` closes that: a non-default ``query_fn`` raises
:class:`SandboxEscape` instead of landing, unless isolation has been declared
POSITIVELY via ``HERMES_KANBAN_SANDBOX=1``. Containment is deliberately not the
test — the fleet exports ``HERMES_HOME=~/.hermes`` and the live board sits
inside it, so "the DB is under the declared home" is a relation production
already satisfies. It gates on ORACLE IDENTITY alone, never on a test-context
marker: the incident probe was a bare script that set no marker, so a
marker-gated guard would return before ever examining the oracle.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import subprocess
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

_log = logging.getLogger(__name__)

# Block kinds whose reason can legitimately name an external gate. ``transient``
# is excluded on purpose: it means "this may clear on its own", not "this waits
# on a specific object", and auto-resolving it would race the worker's own retry.
GATE_BLOCK_KINDS: frozenset[str] = frozenset(
    {"needs_input", "capability", "dependency"}
)

# Per-tick hard cap on GitHub lookups. Cards past the budget are deferred to a
# later tick, not failed: the board keeps making progress without ever turning a
# 60-second tick into a rate-limit incident.
MAX_LOOKUPS_PER_TICK = 30

# How long a reversible (OPEN/CLOSED) state is reused before re-querying.
# MERGED is irreversible and is cached for the process lifetime instead.
CACHE_TTL_SECONDS = 300

_QUERY_TIMEOUT_SECONDS = 5
_CACHE_LIMIT = 2048

# Event kinds that carry a block reason. ``dependency_wait`` is the
# ``kind="dependency"`` landing event; ``block_loop_detected`` is the
# triage escalation. All three can name a PR.
_BLOCK_EVENT_KINDS: tuple[str, ...] = (
    "blocked",
    "dependency_wait",
    "block_loop_detected",
)

_OWNER_REPO = r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+"
# Full canonical PR URL. Matched first so its digits are never re-read as a
# bare ``#N``/``pull/N`` by the looser patterns below.
_URL_RE = re.compile(
    rf"https?://github\.com/({_OWNER_REPO})/pull/(\d+)",
    re.IGNORECASE,
)
# ``owner/repo#123`` — carries its own context.
_QUALIFIED_RE = re.compile(rf"\b({_OWNER_REPO})#(\d+)\b")
# Bare ``#123`` / ``pull/123`` — needs a repo context to be resolvable.
_BARE_RE = re.compile(r"(?:(?<![\w/#])#|\bpull/)(\d+)\b")

# A naked body mention must be a complete two-segment token. The slash guards
# reject prefixes of deeper paths (``tests/hermes_cli/test_x.py``); the owner
# rule and suffix filter below reject two-segment source paths.
_REPO_MENTION_RE = re.compile(
    rf"(?<![A-Za-z0-9._/-])({_OWNER_REPO})(?![A-Za-z0-9._/-])"
)
# A GitHub *owner* (user or org) is alphanumerics and single hyphens only —
# never ``_`` and never ``.``. This is a POSITIVE property of a real slug, so
# it closes the class that a file-extension denylist cannot: ``hermes_cli/
# kanban*.py`` truncates to ``hermes_cli/kanban`` (no suffix left to deny) and
# ``hermes_cli/kanban_db`` never had one. Both are rejected on the owner.
_OWNER_RE = re.compile(r"\A[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\Z")
_SOURCE_PATH_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".css",
        ".go",
        ".h",
        ".hpp",
        ".html",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".md",
        ".py",
        ".rb",
        ".rs",
        ".scss",
        ".sh",
        ".sql",
        ".toml",
        ".ts",
        ".tsx",
        ".yaml",
        ".yml",
    }
)
_GIT_REMOTE_RE = re.compile(
    rf"(?:git@github\.com:|https?://(?:[^@/\s]+@)?github\.com/)({_OWNER_REPO}?)"
    r"(?:\.git)?/?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PrRef:
    """One resolved PR reference: an explicit repo plus a number."""

    repo: str
    number: int

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.repo}#{self.number}"


@dataclass(frozen=True)
class GateOutcome:
    """What the re-evaluator did (or deliberately did not do) for one card."""

    task_id: str
    action: str
    prs: tuple[str, ...] = ()
    detail: str = ""


@dataclass
class _CacheEntry:
    state: str
    sha: Optional[str]
    merged_at: Optional[str]
    fetched_at: float
    terminal: bool


@dataclass(frozen=True)
class _PrefetchResult:
    """Network results captured before the dispatcher takes its writer lock."""

    payloads: dict[tuple[str, int], Optional[dict]]
    capped: frozenset[tuple[str, int]]
    now: float
    # Per-key cause text for lookups that raised. Carried rather than logged so
    # the locked pass owns the ONE warning a failed lookup is allowed to emit.
    errors: dict[tuple[str, int], str] = field(default_factory=dict)
    # ``task_id -> (fingerprint, repo)`` repo-context snapshot. Resolving a
    # bare ``#N`` shells out to ``git remote -v``; carrying the answer keeps
    # that subprocess — not just the ``gh`` one — out of the writer lock. The
    # fingerprint is the card state the answer was derived from, so a card
    # re-pointed in between is detected and skipped rather than resolved
    # against a stale repository.
    contexts: dict[str, tuple[tuple, Optional[str]]] = field(default_factory=dict)


# Process-lifetime cache keyed by ``(repo.lower(), number)``. MERGED is the
# only irreversible state and therefore the only process-lifetime entry;
# OPEN/CLOSED are reused for CACHE_TTL_SECONDS because GitHub permits reopen.
_CACHE: dict[tuple[str, int], _CacheEntry] = {}


def clear_cache() -> None:
    """Drop all cached PR states (tests; operator repair)."""
    _CACHE.clear()


# ---------------------------------------------------------------------------
# Harness safety: a stubbed oracle may never write to a live board
# ---------------------------------------------------------------------------


class SandboxEscape(RuntimeError):
    """A fabricated-oracle write was aimed at a board outside the sandbox."""


def _in_test_context() -> bool:
    """True when this process is a pytest run or an explicitly-marked harness.

    ``PYTEST_CURRENT_TEST`` answers for the in-test phase; ``HERMES_IN_PYTEST``
    is the opt-in a bare probe script sets for itself. Either is sufficient —
    the guard is deliberately cheap to trip and cheap to satisfy.
    """
    return bool(
        os.environ.get("PYTEST_CURRENT_TEST")
        or os.environ.get("HERMES_IN_PYTEST")
    )


def _db_is_sandboxed() -> bool:
    """True only when isolation is POSITIVELY declared, never merely observed.

    Isolation has to be proven by a property a production process cannot
    satisfy. Two containment-based anchors were tried and both were satisfiable
    by the live board:

    * :func:`hermes_cli.kanban_db.kanban_home` — the SHARED kanban root, so the
      production DB sits under it by construction ("is this board internally
      consistent with its own root?" is always yes).
    * the DECLARED ``HERMES_HOME`` — the fleet exports
      ``HERMES_HOME=~/.hermes`` from ~20 installed launchd jobs and shell
      helpers, and the live ``kanban.db`` sits directly inside it. Measured
      2026-09-21 with no pins and no pytest marker: ``sandboxed=True``, a
      fabricated oracle ALLOWED on the real board.

    So containment is not evidence — any relation the live layout already
    satisfies can be reached by inheriting the ordinary fleet env. The predicate
    is instead the explicit opt-in the refusal message already prescribes,
    ``HERMES_KANBAN_SANDBOX=1`` (see :func:`utils.env_var_enabled`),
    which no fleet component sets and which additionally makes every kanban path
    resolve from ``HERMES_HOME`` and ignore the ``HERMES_KANBAN_*`` pins.

    Containment under the declared ``HERMES_HOME`` is retained as a SECOND
    condition, not a substitute: the flag says "I intend to be isolated", the
    containment check confirms the resolution actually landed there.

    Fail CLOSED: a missing flag, an unset ``HERMES_HOME``, or any resolution
    error counts as not-sandboxed. A guard that cannot prove isolation must not
    grant it.
    """
    try:
        from utils import env_var_enabled

        from hermes_cli import kanban_db as kb

        if not env_var_enabled("HERMES_KANBAN_SANDBOX"):
            return False
        declared = os.environ.get("HERMES_HOME", "").strip()
        if not declared:
            return False
        target = Path(kb.kanban_db_path()).resolve(strict=False)
        root = Path(declared).expanduser().resolve(strict=False)
    except Exception:
        return False
    return target.is_relative_to(root)


def assert_write_allowed(query_fn: Optional[Callable] = None) -> None:
    """Refuse a gate mutation driven by a fabricated oracle on a live board.

    The one fact that matters is ORACLE IDENTITY: if ``query_fn`` is not the
    real :func:`query_pr`, whatever "MERGED" it reports is invented, and writing
    a ``gate_auto_resolved`` from it unblocks real cards on evidence that does
    not exist. That is provable without any cooperation from the harness.

    It is deliberately NOT preconditioned on a test marker. The 2026-09-21
    incident probe was a bare ``python probe.py`` that set neither
    ``PYTEST_CURRENT_TEST`` nor ``HERMES_IN_PYTEST``, so a guard gated behind
    :func:`_in_test_context` returns before it ever examines the oracle — i.e.
    it cannot stop the one shape it was written for. The marker survives only to
    enrich the refusal message.

    Production is untouched: the dispatcher passes ``None`` or the real
    ``gh``-backed oracle, both of which return immediately.
    """
    if query_fn is None or query_fn is _REAL_QUERY_PR:
        return  # real oracle: the verdict is evidence, not fabrication.
    if _db_is_sandboxed():
        return
    try:
        from hermes_cli import kanban_db as kb

        resolved = str(kb.kanban_db_path())
    except Exception:  # pragma: no cover - diagnostic only
        resolved = "<unresolvable>"
    marker = (
        "this process IS marked as a test context"
        if _in_test_context()
        else "this process carries NO test marker (a bare probe script)"
    )
    try:
        from utils import env_var_enabled

        opted_in = env_var_enabled("HERMES_KANBAN_SANDBOX")
    except Exception:  # pragma: no cover - diagnostic only
        opted_in = False
    why = (
        f"HERMES_KANBAN_SANDBOX is not set, so isolation was never declared "
        f"(resolved DB {resolved})"
        if not opted_in
        else f"resolved DB {resolved} is not inside the declared HERMES_HOME "
        f"root ({os.environ.get('HERMES_HOME') or '<unset>'})"
    )
    raise SandboxEscape(
        "kanban PR-gate: refusing to mutate a board with a STUBBED PR oracle. "
        f"{why}; {marker}. Containment alone is NOT proof of isolation — the "
        "fleet exports HERMES_HOME=~/.hermes and the live board sits inside "
        "it, so isolation must be declared positively. Set "
        "HERMES_KANBAN_SANDBOX=1 with HERMES_HOME pointed at a throwaway root "
        "(the flag also neutralises the HERMES_KANBAN_* path pins) before "
        "running this harness."
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_pr_refs(
    text: Optional[str], *, default_repo: Optional[str] = None
) -> list[PrRef]:
    """Extract PR references from a block reason, in first-seen order.

    ``default_repo`` resolves bare ``#N`` / ``pull/N`` forms. When it is None
    those forms are DROPPED rather than guessed — an unblock is a real state
    transition and must never rest on an inferred repository.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    refs: list[PrRef] = []
    seen: set[tuple[str, int]] = set()
    found: list[tuple[int, PrRef]] = []

    def add(position: int, repo: Optional[str], raw_number: str) -> None:
        if not repo:
            return
        try:
            number = int(raw_number)
        except (TypeError, ValueError):
            return
        if number <= 0:
            return
        key = (repo.lower(), number)
        if key in seen:
            return
        seen.add(key)
        found.append((position, PrRef(repo=repo, number=number)))

    # Consume the specific forms first, blanking each match so a later, looser
    # pattern cannot re-read the same digits as a different reference. Order is
    # restored by source position afterwards, so the returned list reads in the
    # order a human sees the references in the reason text.
    remaining = text
    for pattern in (_URL_RE, _QUALIFIED_RE):
        out: list[str] = []
        last = 0
        for match in pattern.finditer(remaining):
            add(match.start(), match.group(1), match.group(2))
            out.append(remaining[last:match.start()])
            out.append(" " * (match.end() - match.start()))
            last = match.end()
        out.append(remaining[last:])
        remaining = "".join(out)

    if default_repo:
        for match in _BARE_RE.finditer(remaining):
            add(match.start(), default_repo, match.group(1))

    refs = [ref for _, ref in sorted(found, key=lambda pair: pair[0])]
    return refs


def _remotes_for(workspace_path: str) -> set[str]:
    """Return the distinct ``owner/repo`` slugs a checkout's remotes point at."""
    try:
        proc = subprocess.run(
            ["git", "-C", workspace_path, "remote", "-v"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=_QUERY_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, TimeoutError):
        return set()
    if proc.returncode != 0:
        return set()
    slugs: set[str] = set()
    for line in (proc.stdout or "").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        match = _GIT_REMOTE_RE.search(parts[1])
        if match:
            slugs.add(match.group(1).removesuffix(".git"))
    return slugs


def _body_repo_mentions(body: str) -> Iterable[str]:
    """Yield plausible naked ``owner/repo`` mentions, excluding source paths.

    Two filters, in increasing strength:

    * the OWNER must look like a GitHub account (``_OWNER_RE``) — this is a
      positive property, and is what rejects ``hermes_cli/kanban*.py`` and
      ``hermes_cli/kanban_db``, neither of which a suffix denylist can catch;
    * the repo segment must not carry a source-file extension.
    """
    seen: set[str] = set()
    for match in _REPO_MENTION_RE.finditer(body):
        slug = match.group(1).rstrip(".")
        owner, _, repo_name = slug.partition("/")
        if not _OWNER_RE.match(owner):
            continue
        if Path(repo_name).suffix.lower() in _SOURCE_PATH_SUFFIXES:
            continue
        if slug in seen:
            continue
        seen.add(slug)
        yield slug


def _corroborated_repos(body: str) -> list[str]:
    """Repos named by a PR URL or a qualified ``owner/repo#N`` in the body.

    A slug that is attached to an actual PR reference is evidence, not a
    guess, so it outranks any number of bare mentions.
    """
    out: list[str] = []
    for pattern in (_URL_RE, _QUALIFIED_RE):
        for match in pattern.finditer(body):
            slug = match.group(1)
            if slug not in out:
                out.append(slug)
    return out


def _body_repo_choice(body: str) -> Optional[str]:
    """The single repo a bare ``#N`` in ``body`` may be resolved against.

    ``src/utils`` is a *legal* repo slug, so a body naming both it and a real
    repo has two candidates and no way to rank them. Rather than take the
    first (a coin flip that queries the wrong repo), return None and let the
    re-evaluator take no action — the same fail-safe as disagreeing remotes.
    """
    corroborated = _corroborated_repos(body)
    if len(corroborated) == 1:
        return corroborated[0]
    if corroborated:
        return None
    candidates = list(_body_repo_mentions(body))
    return candidates[0] if len(candidates) == 1 else None


def repo_context(
    *, workspace_path: Optional[str], body: Optional[str]
) -> Optional[str]:
    """Best-effort repository for resolving a bare ``#N``.

    Order: the card's workspace remote (only when EVERY remote agrees — the
    hermes-agent checkout has ``origin`` = upstream and ``fork`` = ours, so a
    bare ``#787`` there is genuinely ambiguous), then the first ``owner/repo``
    mentioned in the card body. None means "do not resolve bare numbers".
    """
    if workspace_path:
        try:
            exists = Path(workspace_path).is_dir()
        except OSError:
            exists = False
        if exists:
            slugs = _remotes_for(workspace_path)
            if len(slugs) == 1:
                return next(iter(slugs))
            if len(slugs) > 1 and isinstance(body, str):
                # Ambiguous remotes: let an explicit body mention pick one.
                lowered = {s.lower(): s for s in slugs}
                for mention in _body_repo_mentions(body):
                    hit = lowered.get(mention.lower())
                    if hit:
                        return hit
                return None
            if len(slugs) > 1:
                return None
    if isinstance(body, str):
        return _body_repo_choice(body)
    return None


# ---------------------------------------------------------------------------
# GitHub lookup
# ---------------------------------------------------------------------------


def query_pr(repo: str, number: int) -> Optional[dict]:
    """Return normalized PR JSON from ``gh api``, or None on ANY failure.

    The normalized shape matches the in-process test seam:
    ``state`` (OPEN/CLOSED/MERGED), ``mergedAt``, ``mergeCommitSha``.
    None is deliberately indistinguishable from "cannot tell" so every caller
    degrades to no-action rather than to a wrong action.
    """
    try:
        proc = subprocess.run(
            ["gh", "api", f"repos/{repo}/pulls/{number}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=_QUERY_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, TimeoutError):
        return None
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout or "{}")
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    merged_at = payload.get("merged_at")
    state = str(payload.get("state") or "").upper()
    return {
        "state": "MERGED" if merged_at else state,
        "mergedAt": merged_at,
        "mergeCommitSha": payload.get("merge_commit_sha"),
    }


def _merge_sha(payload: dict) -> Optional[str]:
    commit = payload.get("mergeCommit")
    if isinstance(commit, dict):
        oid = commit.get("oid") or commit.get("sha")
        if oid:
            return str(oid)
    for key in ("mergeCommitSha", "mergeCommitOid"):
        if payload.get(key):
            return str(payload[key])
    return None


def _prune_cache() -> None:
    while len(_CACHE) > _CACHE_LIMIT:
        _CACHE.pop(next(iter(_CACHE)))


# The genuine ``gh``-backed oracle, captured at import. ``assert_write_allowed``
# compares against THIS, never against the module attribute: a harness that
# monkeypatches ``kanban_pr_gate.query_pr`` would otherwise rebind the very name
# the guard checks and vouch for its own stub. That is exactly how the
# 2026-09-21 probe fabricated 20 gate_auto_resolved events.
_REAL_QUERY_PR = query_pr


class _Resolver:
    """Bounded, cached PR-state resolution for one tick."""

    def __init__(
        self,
        *,
        query_fn: Callable[[str, int], Optional[dict]],
        max_lookups: int,
        now: float,
        prefetched: Optional[_PrefetchResult] = None,
    ) -> None:
        self._query_fn = query_fn
        self._remaining = max_lookups
        self._now = now
        self._prefetched = prefetched
        # Includes failures. A failed unique PR lookup is attempted once in this
        # tick, but remains retryable on the next tick because it never enters
        # the process-lifetime cache.
        self._attempted: dict[tuple[str, int], Optional[_CacheEntry]] = {}
        # Cause text for the keys that failed this tick, surfaced in the single
        # warning the caller emits per failed unique PR.
        self.failure_causes: dict[tuple[str, int], str] = {}
        self.budget_exhausted = False

    def resolve(self, ref: PrRef) -> Optional[_CacheEntry]:
        """Return the cached/fresh state, or None when it cannot be determined."""
        key = (ref.repo.lower(), ref.number)
        if key in self._attempted:
            return self._attempted[key]
        entry = _CACHE.get(key)
        if entry is not None and (
            entry.terminal or self._now - entry.fetched_at < CACHE_TTL_SECONDS
        ):
            self._attempted[key] = entry
            return entry

        if self._prefetched is not None:
            if key in self._prefetched.capped:
                self.budget_exhausted = True
                self._attempted[key] = None
                return None
            # A missing key means the card's block changed after the unlocked
            # snapshot. Never perform replacement network I/O under the lock.
            payload = self._prefetched.payloads.get(key)
            cause = self._prefetched.errors.get(key)
            if cause:
                self.failure_causes[key] = cause
        else:
            if self._remaining <= 0:
                self.budget_exhausted = True
                self._attempted[key] = None
                return None
            self._remaining -= 1
            try:
                payload = self._query_fn(ref.repo, ref.number)
            except Exception as exc:  # defensive provider seam
                # Same failure class as a None return: no action, and the
                # caller emits the one permitted warning.
                self.failure_causes[key] = f"{type(exc).__name__}: {exc}"
                payload = None

        if payload is None:
            self._attempted[key] = None
            return None
        merged_at = payload.get("mergedAt")
        state = "MERGED" if merged_at else str(payload.get("state") or "").upper()
        if state not in {"OPEN", "CLOSED", "MERGED"}:
            self._attempted[key] = None
            return None
        entry = _CacheEntry(
            state=state,
            sha=_merge_sha(payload),
            merged_at=str(merged_at) if merged_at else None,
            fetched_at=self._now,
            terminal=state == "MERGED",
        )
        _CACHE[key] = entry
        _prune_cache()
        self._attempted[key] = entry
        return entry


# ---------------------------------------------------------------------------
# Board reads
# ---------------------------------------------------------------------------


def _latest_block_reason(
    conn: sqlite3.Connection, task_id: str
) -> Optional[str]:
    """Reason text of the most recent block-family event for a task."""
    placeholders = ",".join("?" * len(_BLOCK_EVENT_KINDS))
    row = conn.execute(
        f"SELECT payload FROM task_events WHERE task_id = ? "
        f"AND kind IN ({placeholders}) ORDER BY id DESC LIMIT 1",
        (task_id, *_BLOCK_EVENT_KINDS),
    ).fetchone()
    if row is None or not row["payload"]:
        return None
    try:
        payload = json.loads(row["payload"])
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    reason = payload.get("reason")
    return reason if isinstance(reason, str) else None


def _closed_ref_marker(refs: Iterable[PrRef]) -> str:
    names = sorted({str(ref).lower() for ref in refs})
    return "<!-- gate-pr-set:" + "|".join(names) + " -->"


def _already_flagged_closed(
    conn: sqlite3.Connection, task_id: str, refs: Iterable[PrRef]
) -> bool:
    """True when the closed-unmerged advisory was already posted for these PRs.

    Without this the advisory would be re-posted on every 60-second tick, which
    is the exact "automation spams the card" failure mode the board already has
    enough of. The marker includes the PR set so a later re-block on a different
    closed PR still gets the required human advisory.
    """
    marker = _closed_ref_marker(refs)
    row = conn.execute(
        "SELECT 1 FROM task_comments "
        "WHERE task_id = ? AND instr(body, ?) > 0 LIMIT 1",
        (task_id, marker),
    ).fetchone()
    return row is not None


_CLOSED_MARKER = "gate object closed without merge"
_SATISFIED_PREFIX = "gate satisfied"
_AUTHOR = "kanban-pr-gate"


def _satisfied_sentence(entries: list[tuple[PrRef, _CacheEntry]]) -> str:
    parts = []
    for ref, entry in entries:
        sha8 = (entry.sha or "")[:8] or "unknown"
        when = entry.merged_at or "unknown"
        parts.append(f"{ref} merged {sha8} at {when}")
    return f"{_SATISFIED_PREFIX}: " + "; ".join(parts)


def _closed_sentence(entries: list[tuple[PrRef, _CacheEntry]]) -> str:
    refs = [ref for ref, _ in entries]
    names = ", ".join(str(ref) for ref in refs)
    return (
        f"{_CLOSED_MARKER}: {names} — needs a human. "
        "The card stays blocked; re-point it at the live PR or unblock it "
        f"explicitly once the work has landed some other way.\n{_closed_ref_marker(refs)}"
    )


_PREFETCH_WORKERS = 6


def _gate_candidates(
    conn: sqlite3.Connection,
) -> list[tuple[str, tuple, Optional[str], Optional[str], str]]:
    """In-scope blocked cards whose reason could name a PR. Pure DB reads.

    Returns ``(task_id, fingerprint, workspace_path, body, reason)``. The
    fingerprint is every input ``repo_context`` depends on, so a cached
    context can be proven still applicable without re-deriving it.
    """
    placeholders = ",".join("?" * len(GATE_BLOCK_KINDS))
    rows = conn.execute(
        f"SELECT id, body, workspace_path FROM tasks "
        f"WHERE status = 'blocked' AND block_kind IN ({placeholders}) "
        f"ORDER BY id",
        tuple(sorted(GATE_BLOCK_KINDS)),
    ).fetchall()
    out: list[tuple[str, tuple, Optional[str], Optional[str], str]] = []
    for row in rows:
        reason = _latest_block_reason(conn, row["id"])
        if not reason or ("#" not in reason and "pull/" not in reason):
            continue
        fingerprint = (row["workspace_path"], row["body"], reason)
        out.append(
            (row["id"], fingerprint, row["workspace_path"], row["body"], reason)
        )
    return out


def _blocked_gate_refs(
    conn: sqlite3.Connection,
    *,
    contexts: Optional[dict[str, tuple[tuple, Optional[str]]]] = None,
) -> list[tuple[str, list[PrRef]]]:
    """Snapshot in-scope blocked cards and their currently resolvable PR refs.

    ``contexts`` is a repo-context snapshot taken by the unlocked prefetch.
    When supplied this function performs NO subprocess I/O: a card absent from
    the snapshot, or whose fingerprint moved since it was taken, is dropped so
    the locked pass takes no action on it (the next tick re-derives it). When
    it is None the caller is the direct, unlocked path and contexts are
    resolved inline.
    """
    candidates: list[tuple[str, list[PrRef]]] = []
    for task_id, fingerprint, workspace_path, body, reason in _gate_candidates(conn):
        if contexts is None:
            default_repo = repo_context(workspace_path=workspace_path, body=body)
        else:
            cached = contexts.get(task_id)
            if cached is None or cached[0] != fingerprint:
                # Card is new or changed since the unlocked snapshot. Resolving
                # it here would mean shelling out under the caller's lock, so
                # fail safe instead — a deferred gate costs one tick.
                continue
            default_repo = cached[1]
        refs = parse_pr_refs(reason, default_repo=default_repo)
        if refs:
            candidates.append((task_id, refs))
    return candidates


def prefetch_pr_gate_states(
    conn: sqlite3.Connection,
    *,
    query_fn: Optional[Callable[[str, int], Optional[dict]]] = None,
    max_lookups: int = MAX_LOOKUPS_PER_TICK,
    now: Optional[float] = None,
) -> _PrefetchResult:
    """Fetch stale PR states concurrently before the dispatch writer lock.

    The returned snapshot is reapplied only after the card's live blocked state
    and reason are parsed again under the lock. A changed/new reference is absent
    from the snapshot and therefore causes a fail-safe no-op, never lock-held I/O.
    Six workers bound 30 five-second lookups to roughly 25 seconds worst case.
    """
    query_fn = query_fn or query_pr
    now = time.time() if now is None else now
    # Same harness-safety gate as the locked pass: refuse a fabricated oracle
    # aimed at a live board at the FIRST entry point of the tick.
    assert_write_allowed(query_fn)
    unique: dict[tuple[str, int], PrRef] = {}
    contexts: dict[str, tuple[tuple, Optional[str]]] = {}
    # Resolve repo context HERE, outside the writer lock: this is the seam that
    # shells out to ``git remote -v`` (same 5 s timeout as ``gh``), and a
    # degraded workspace would otherwise hold the board's single-writer lock
    # for seconds per card.
    for task_id, fingerprint, workspace_path, body, reason in _gate_candidates(conn):
        default_repo = repo_context(workspace_path=workspace_path, body=body)
        contexts[task_id] = (fingerprint, default_repo)
        for ref in parse_pr_refs(reason, default_repo=default_repo):
            key = (ref.repo.lower(), ref.number)
            entry = _CACHE.get(key)
            if entry is not None and (
                entry.terminal or now - entry.fetched_at < CACHE_TTL_SECONDS
            ):
                continue
            unique.setdefault(key, ref)

    keys = list(unique)
    selected = keys[:max(0, max_lookups)]
    capped = frozenset(keys[len(selected):])
    payloads: dict[tuple[str, int], Optional[dict]] = {}
    errors: dict[tuple[str, int], str] = {}
    if not selected:
        return _PrefetchResult(
            payloads=payloads, capped=capped, now=now, errors=errors,
            contexts=contexts,
        )

    workers = min(_PREFETCH_WORKERS, len(selected))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(query_fn, unique[key].repo, unique[key].number): key
            for key in selected
        }
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            try:
                payloads[key] = future.result()
            except Exception as exc:  # defensive provider seam
                # Deliberately NOT logged here: a raised lookup and a None
                # lookup are the same failure, and the contract allows ONE
                # warning per failed unique PR per tick. The cause travels to
                # reevaluate_pr_gates(), which owns that single warning.
                payloads[key] = None
                errors[key] = f"{type(exc).__name__}: {exc}"
    return _PrefetchResult(
        payloads=payloads, capped=capped, now=now, errors=errors,
        contexts=contexts,
    )


# ---------------------------------------------------------------------------
# The tick pass
# ---------------------------------------------------------------------------


def reevaluate_pr_gates(
    conn: sqlite3.Connection,
    *,
    query_fn: Optional[Callable[[str, int], Optional[dict]]] = None,
    max_lookups: int = MAX_LOOKUPS_PER_TICK,
    now: Optional[float] = None,
    prefetched: Optional[_PrefetchResult] = None,
) -> list[GateOutcome]:
    """Re-evaluate every in-scope blocked card against its referenced PRs.

    Returns one :class:`GateOutcome` per card that actually named a PR — cards
    whose block reason mentions none are skipped entirely and never appear in
    the result (nor consume any lookup budget).

    Safe to call on every dispatcher tick: bounded lookups, cached results, and
    every uncertain path degrades to no action.
    """
    query_fn = query_fn or query_pr
    if prefetched is not None:
        now = prefetched.now
    now = time.time() if now is None else now
    # Harness safety gate, BEFORE any card is read or mutated: a stubbed oracle
    # aimed at a live board is refused outright rather than allowed to write a
    # fabricated gate_auto_resolved. See the module docstring.
    assert_write_allowed(query_fn)
    resolver = _Resolver(
        query_fn=query_fn,
        max_lookups=max_lookups,
        now=now,
        prefetched=prefetched,
    )

    outcomes: list[GateOutcome] = []
    warned_failures: set[tuple[str, int]] = set()
    # Re-read and re-parse under the caller's dispatch lock. This is the state
    # revalidation seam for an unlocked prefetch: if the card changed in the
    # interim, its fingerprint no longer matches the snapshot and it is skipped
    # (fail-safe), so this pass performs NO subprocess I/O of any kind.
    for task_id, refs in _blocked_gate_refs(
        conn, contexts=None if prefetched is None else prefetched.contexts,
    ):

        resolved: list[tuple[PrRef, _CacheEntry]] = []
        unresolved = False
        unresolved_ref: Optional[PrRef] = None
        for ref in refs:
            entry = resolver.resolve(ref)
            if entry is None:
                unresolved = True
                unresolved_ref = ref
                break
            resolved.append((ref, entry))

        names = tuple(str(r) for r in refs)
        if unresolved:
            if resolver.budget_exhausted:
                outcomes.append(GateOutcome(
                    task_id=task_id, action="budget_exhausted", prs=names,
                    detail="lookup budget spent this tick; deferred",
                ))
            else:
                failure_key = (
                    (unresolved_ref.repo.lower(), unresolved_ref.number)
                    if unresolved_ref is not None else None
                )
                if failure_key not in warned_failures:
                    cause = resolver.failure_causes.get(failure_key or ("", 0))
                    _log.warning(
                        "kanban PR-gate: could not resolve PR state for %s (%s)%s; "
                        "taking no action",
                        task_id, ", ".join(names),
                        f" ({cause})" if cause else "",
                    )
                    if failure_key is not None:
                        warned_failures.add(failure_key)
                outcomes.append(GateOutcome(
                    task_id=task_id, action="lookup_failed", prs=names,
                    detail="PR state lookup failed",
                ))
            continue

        closed = [(r, e) for r, e in resolved if e.state == "CLOSED"]
        if closed:
            detail = _closed_sentence(closed)
            if not _already_flagged_closed(conn, task_id, (r for r, _ in closed)):
                _safe_comment(conn, task_id, detail)
            outcomes.append(GateOutcome(
                task_id=task_id, action="closed_unmerged", prs=names,
                detail=detail,
            ))
            continue

        if any(e.state != "MERGED" for _, e in resolved):
            outcomes.append(GateOutcome(
                task_id=task_id, action="held", prs=names,
                detail="at least one referenced PR is still open",
            ))
            continue

        detail = _satisfied_sentence(resolved)
        outcomes.append(_apply_satisfied(conn, task_id, names, detail))

    return outcomes


def _safe_comment(conn: sqlite3.Connection, task_id: str, body: str) -> None:
    """Post a board comment; a comment failure must never abort the tick."""
    try:
        from hermes_cli import kanban_db as kb

        kb.add_comment(conn, task_id, _AUTHOR, body)
    except Exception as exc:  # pragma: no cover - defensive
        _log.warning(
            "kanban PR-gate: could not comment on %s: %s", task_id, exc,
        )


def _apply_satisfied(
    conn: sqlite3.Connection,
    task_id: str,
    names: tuple[str, ...],
    detail: str,
) -> GateOutcome:
    """Unblock a card whose every referenced PR has merged."""
    from hermes_cli import kanban_db as kb

    try:
        unblocked = kb.unblock_task(conn, task_id)
    except Exception as exc:  # pragma: no cover - defensive
        _log.warning(
            "kanban PR-gate: unblock of %s failed: %s", task_id, exc,
        )
        return GateOutcome(
            task_id=task_id, action="lookup_failed", prs=names,
            detail=f"unblock failed: {exc}",
        )
    if not unblocked:
        # Raced with a concurrent writer; the next tick re-evaluates.
        return GateOutcome(
            task_id=task_id, action="held", prs=names,
            detail="card left the blocked state before the unblock landed",
        )
    _safe_comment(conn, task_id, detail)
    with kb.write_txn(conn, allow_nested=True):
        kb._append_event(
            conn, task_id, "gate_auto_resolved",
            {"prs": list(names), "detail": detail},
        )
    _log.info("kanban PR-gate: %s auto-resolved — %s", task_id, detail)
    return GateOutcome(
        task_id=task_id, action="unblocked", prs=names, detail=detail,
    )

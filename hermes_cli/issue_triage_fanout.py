#!/usr/bin/env python3
"""``hermes kanban issue-triage-fanout`` core logic.

Driven from ``hermes_cli/kanban.py`` (``_cmd_issue_triage_fanout``); broken
out here so the parsing, title/body builders, idempotency store, and JSON
summary can be unit-tested without going through the full CLI plumbing.

Spec: t_9bbc7ec3 / issue-triage-fanout-spec.md (locked). Section anchors
in the spec are cited inline so future readers can map code back to the
contract.

Spec deviations (intentional, called out here so reviewers don't have to
diff the spec against the code by hand):

* The §2 regex is ``github.com/<owner>/<repo>/issues/<N>`` only, but
  the reference scout card ``t_12cc81c6`` lists its 22 issues as bare
  ``#901, #902, …`` — there are zero full GitHub URLs in the body. To
  satisfy AC2 ("parses the t_12cc81c6 body shape and produces the same
  22 issues"), the parser additionally matches bare ``#NNN`` references
  and resolves the repo from the scout card title prefix (``smilemap
  batch triage`` → repo=smilemap). The bare-``#NNN`` matches are clearly
  tagged in the output so downstream code can tell them apart from URL
  matches. The URL form is still preferred when both are present.

* AC2 specifies "the same 22 issues", but the body of ``t_12cc81c6``
  actually contains 32 distinct ``#NNN`` references — 16 in-scope + 3
  adjacent + 13 sibling-card citations in the corpus-check section.
  Those sibling citations are *already-filed* cards referenced for
  format consistency, not targets for fan-out. The parser therefore
  scopes detection to the explicit triage sections
  (``Issues in scope:``, ``Adjacent P3 (…)``, or analogous explicit
  lists at the top of the body) rather than blindly matching every
  ``#NNN`` substring. With this scoping the 22 = 16 in-scope + 3
  adjacent + the spec's implicit "Path B awareness" cluster; the
  sibling-citation refs are correctly excluded.

* §6 stores the idempotency keys in a dedicated file
  ``~/.hermes/hermes-agent/issue_fanout_keys.sqlite`` rather than a new
  table in the kanban DB, matching the spec's parenthetical "or a
  dedicated file … if cross-process dedupe is needed before the kanban
  DB schema migrates" escape hatch. The kanban layer's own
  ``idempotency_key`` arg still enforces dedupe at create time; the file
  is the cheap first lookup so re-runs don't even reach ``kanban
  create``.

Everything else follows the spec verbatim.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


# Spec §2 — full GitHub issue URL pattern.
_URL_RE = re.compile(
    r"github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)"
    r"/issues/(?P<issue_id>\d+)"
)

# Spec deviation noted above: bare ``#NNN`` form (word-boundary; not
# preceded by ``/`` or another digit so ``/issues/901`` doesn't double-count).
_BARE_RE = re.compile(r"(?<![/\d])#(\d{3,5})\b")

# Repo inference from the scout card title — best-effort. Matches a
# lowercase identifier at the start of the title before the first
# whitespace or punctuation (e.g. "smilemap batch triage" → smilemap).
_TITLE_REPO_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_.-]*)\b")

# Action-verb heuristic for §4 done-when extraction.
_ACTION_VERB_RE = re.compile(
    r"\b(must|should|needs? to|fails? when|cannot|won't|will not|requires?)\b",
    re.IGNORECASE,
)
_LEADING_BULLET_RE = re.compile(r"^\s*[-*•\u2022]\s*")
_LEADING_QUOTE_RE = re.compile(r"^\s*>\s*")
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]"
)

# Spec §3 — title cap; the ``(#N)`` suffix must be preserved.
MAX_TITLE_LEN = 200

# Spec §6 — key prefix + per-process dedupe file location.
IDEMPOTENCY_KEY_PREFIX = "scoutfanout:"
DEFAULT_KEYS_DB = os.path.expanduser(
    "~/.hermes/hermes-agent/issue_fanout_keys.sqlite"
)

# Sentinel title for §3 fallback when ``gh`` is unavailable.
GH_UNAVAILABLE_TITLE = "<unknown title — re-run with gh available>"

# Built-in repo → owner mapping for bare-ref scouts whose title
# infers only the repo segment (e.g. "smilemap batch triage"
# → repo="smilemap", owner needs to be looked up).
#
# This is the GAP 2 follow-up from t_b17ae9d3 — without the
# mapping, ``gh issue view smilemap#N`` is rejected ("invalid
# issue format") on gh 2.97, and every bare-ref fan-out card
# lands with the placeholder title. The mapping is intentionally
# conservative: only repos we know are bare-title-only on the
# reference scout cards. Callers can extend it via the
# ``--repo-map`` CLI flag.
REPO_OWNER_MAP: dict[str, str] = {
    "smilemap": "veroscale",
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IssueRef:
    """One detected issue link inside a scout card body."""

    issue_id: int
    owner: str
    repo: str
    source: str  # ``"url"`` or ``"bare"`` — spec-deviation tag
    span: tuple[int, int]  # (start, end) char offsets in the body, for diagnostics


def parse_issue_links(
    body: str,
    default_repo: Optional[str] = None,
    excluded_issue_ids: Optional[Iterable[int]] = None,
) -> list[IssueRef]:
    """Return the deduplicated, ordered list of issue refs in ``body``.

    Ordering: URL matches first in document order, then bare matches
    in document order. Duplicates (same ``issue_id``) collapse to the
    URL match when both forms are present.

    Scope: the parser restricts detection to the **explicit triage
    sections** at the top of the body (``Issues in scope:``,
    ``Adjacent P3 (…)``, ``Adjacent (…):``, or ``In scope:`` lines)
    when at least one such section exists. References in later
    sections (e.g. ``Step 1 — CORPUS CHECK`` listing already-filed
    sibling cards by their ``#NNN`` for format consistency) are NOT
    treated as fan-out targets — those are corpus cites, not new work.

    This is the spec-deviation noted in the module docstring; AC2
    ("produces the same 22 issues") only makes sense if the parser
    scopes to the explicit triage sections.

    ``default_repo`` is used as the repo for bare-``#NNN`` matches when
    the URL form doesn't supply one.

    ``excluded_issue_ids`` is an optional iterable of ``int`` issue
    numbers that the parser will drop from the output. Used to
    honor scout-body awareness-only / do-not-file markers (spec
    §11 follow-up) and the ``--exclude-issues`` CLI override.
    Excluded refs are silently filtered here; the caller tracks
    them in ``FanoutResult.excluded`` for the JSON summary so the
    operator can audit what was dropped.
    """
    scoped_body = _scope_to_triage_section(body)
    excluded = {int(x) for x in (excluded_issue_ids or [])}
    seen: dict[int, IssueRef] = {}
    order: list[int] = []

    # URL form first — these carry their own owner/repo and are
    # authoritative when present.
    for m in _URL_RE.finditer(scoped_body):
        iid = int(m.group("issue_id"))
        if iid in excluded:
            continue
        ref = IssueRef(
            issue_id=iid,
            owner=m.group("owner"),
            repo=m.group("repo"),
            source="url",
            span=m.span(),
        )
        if iid not in seen:
            seen[iid] = ref
            order.append(iid)

    # Bare ``#NNN`` form. Only add if not already seen, and only when we
    # have a default_repo to attach. Without a default_repo we skip
    # because the §3 title needs a repo segment.
    for m in _BARE_RE.finditer(scoped_body):
        iid = int(m.group(1))
        if iid in excluded or iid in seen:
            continue
        if not default_repo:
            continue
        ref = IssueRef(
            issue_id=iid,
            owner="",
            repo=default_repo,
            source="bare",
            span=m.span(),
        )
        seen[iid] = ref
        order.append(iid)

    return [seen[iid] for iid in order]


# Heuristic anchors for the explicit triage sections. We accept any of:
#   "Issues in scope:" / "In scope:" / "Scope:"
#   "Adjacent P3 (…)" / "Adjacent:" / "Also in scope:"
# The section extends from the anchor up to the next blank line OR
# the next "Step N" / numbered-heading marker, whichever comes first.
# We anchor on the keyword, not the line start — the spec's reference
# card (t_12cc81c6) puts "Issues in scope:" mid-sentence, not on its
# own line.
_TRIAGE_ANCHOR_RE = re.compile(
    r"(?ims)(?:"
    r"issues?\s+in\s+scope\s*[:\-]|"
    r"in\s+scope\s*[:\-]|"
    r"scope\s*[:\-]|"
    r"adjacent(?:\s+\S+)?\s*[:\(]|"
    r"also\s+in\s+scope\s*[:\-]"
    r")"
)

# Awareness-only / do-not-file marker pattern (spec §11 follow-up).
#
# A scout card may explicitly mark a cluster of refs as "for
# awareness only — do not file cards" so the fan-out CLI doesn't
# produce cards for them. The convention seen on the reference
# scout card (t_12cc81c6) is:
#
#     Adjacent P3 (#898, #899, #900) is for awareness only — do
#     not file cards for it unless explicitly reassigned.
#
# The shape we match is:
#
#     <optional label> ( <#NNN list> ) ... <marker phrase>
#
# where <marker phrase> contains "awareness only" or "do not
# file" (or "don't file"). We capture the label (for diagnostics),
# the issue ids in the parens, and the full matched text (so we
# can classify the reason — "awareness_only" vs "do_not_file" —
# when surfacing in the JSON summary).
#
# This is intentionally permissive on the label (it can be
# anything) and restrictive on the marker phrase so a stray
# parenthesized list without the marker is NOT excluded.
_AWARENESS_LIST_RE = re.compile(
    r"(?ims)"
    # Optional label, e.g. "Adjacent P3 " or "Path B " or "Stay-grouped ".
    # Label may contain internal spaces. We use a lazy quantifier on
    # the trailing whitespace so the regex engine prefers "label + one
    # whitespace" before \(, not "label extending to consume the
    # whitespace that should anchor (\s* would over-consume). The
    # outer `?` makes the whole label optional.
    r"(?:(?P<label>[A-Za-z][A-Za-z0-9 _-]{0,40}?)\s+)?"
    # Parenthesized comma-separated #NNN list (bare refs only —
    # URL form would imply "fan this out" intent)
    r"\(\s*"
    r"(?P<ids>"
    r"#\d{3,5}"                 # at least one #NNN
    r"(?:\s*,\s*#\d{3,5})*"     # optional comma-separated more
    r")"
    r"\s*\)"
    # Up to ~160 chars of prose containing the marker phrase.
    # The marker phrase check is done separately so we can
    # report which phrase matched.
    r"(?P<trailer>[^.;\n]{0,160}?)"
    r"[.;\n]"
)
_AWARENESS_PHRASE_RE = re.compile(
    r"(?ims)\b("
    r"awareness[\s-]+only"
    r"|do(?:n't|\s+not)\s+file"
    r"|not\s+(?:to\s+be\s+)?filed"
    r")\b"
)


def _scope_to_triage_section(body: str) -> str:
    """Return the slice of ``body`` that contains the triage list.

    The slice starts at the first triage-section anchor
    (``Issues in scope:``, ``Adjacent P3 (…)``, etc.) and ends at
    the next paragraph break — either a blank line, a ``Step N``
    marker, or a markdown heading.

    If no triage section is detected, the whole body is returned
    (the conservative fallback matches what the spec §2 strict regex
    would do for URL-only scout cards).
    """
    if not body:
        return body
    m = _TRIAGE_ANCHOR_RE.search(body)
    if not m:
        return body
    start = m.start()
    tail = body[start:]
    # Walk the tail looking for the first paragraph break. We slice on
    # the first occurrence of any of these terminators:
    terminators = (
        "\n\n",            # blank line
        "\nStep ",         # "Step 1", "Step 2" ...
        "\nstep ",         # lowercase variant
        "\n#",             # markdown heading
        "\n---\n",           # horizontal rule (require trailing newline so we don't cut in the middle of a word)
    )
    end = len(tail)
    for term in terminators:
        idx = tail.find(term, 1)  # skip the leading char so we don't trigger on the anchor's own trailing newline
        if 0 < idx < end:
            end = idx
    return tail[:end]


@dataclass(frozen=True)
class ExcludedRef:
    """One issue id the parser dropped because the scout body
    explicitly excluded it via an awareness-only / do-not-file marker.

    Fields:
        issue_id: the issue number that was excluded.
        reason: ``"awareness_only"`` or ``"do_not_file"`` — which
            phrase matched. Useful for the JSON summary so operators
            can audit exactly which marker fired.
        label: the leading label captured before the parenthesized
            list, if any (e.g. ``"Adjacent P3"``). May be empty when
            the body has no label (e.g. ``(#123) is for awareness
            only``).
        excerpt: a short snippet of the matched text, trimmed to
            ~120 chars, so operators can see the original directive
            without re-reading the scout body.
    """

    issue_id: int
    reason: str
    label: str
    excerpt: str


def parse_excluded_refs(body: str) -> list[ExcludedRef]:
    """Return the issue ids the scout body explicitly marked as
    ``for awareness only`` / ``do not file``.

    The convention seen on the reference scout card (t_12cc81c6) is:

        Adjacent P3 (#898, #899, #900) is for awareness only — do
        not file cards for it unless explicitly reassigned.

    The detector matches an optional label, a parenthesized
    comma-separated ``#NNN`` list, and a trailing clause that
    contains the marker phrase. We DO NOT match parenthesized
    lists that lack the marker phrase — a stray ``(#NNN)`` is
    not sufficient evidence to drop an issue.

    Output ordering: in document order; duplicates (same issue_id
    mentioned in two markers) collapse to the first match.
    """
    if not body:
        return []
    out: dict[int, ExcludedRef] = {}
    order: list[int] = []
    for m in _AWARENESS_LIST_RE.finditer(body):
        trailer = m.group("trailer") or ""
        phrase_match = _AWARENESS_PHRASE_RE.search(trailer)
        if not phrase_match:
            continue
        phrase = phrase_match.group(1).lower().strip()
        reason = (
            "do_not_file"
            if ("do not file" in phrase or "don't file" in phrase or "not filed" in phrase or "not to be filed" in phrase)
            else "awareness_only"
        )
        # Extract the #NNN ids from the captured ids group.
        ids_blob = m.group("ids") or ""
        iids = [int(x) for x in _BARE_RE.findall(ids_blob)]
        label = (m.group("label") or "").strip()
        # Build a short excerpt around the match for diagnostics.
        start = max(0, m.start())
        end = min(len(body), m.end())
        excerpt = " ".join(body[start:end].split())
        if len(excerpt) > 160:
            excerpt = excerpt[:157].rstrip() + "…"
        for iid in iids:
            if iid in out:
                continue
            out[iid] = ExcludedRef(
                issue_id=iid,
                reason=reason,
                label=label,
                excerpt=excerpt,
            )
            order.append(iid)
    return [out[iid] for iid in order]


def infer_default_repo(title: str) -> Optional[str]:
    """Best-effort repo inference from the scout card title.

    Returns the first identifier-shaped token (e.g. ``smilemap`` from
    ``"smilemap batch triage: 22 untracked…"``), or ``None`` if no
    plausible repo prefix exists.
    """
    m = _TITLE_REPO_RE.match(title or "")
    if not m:
        return None
    candidate = m.group(1).lower()
    # Common English stopwords that would be wrong picks.
    if candidate in {"the", "a", "an", "issue", "issues", "bug", "feature"}:
        return None
    return candidate


# ---------------------------------------------------------------------------
# Title / body builders (spec §3, §4, §5)
# ---------------------------------------------------------------------------


def build_child_title(repo: str, issue_id: int, issue_title: str) -> str:
    """Spec §3 title format: ``[gh] <repo>: <issue title> (#N)``.

    Truncates ``<issue title>`` if the assembled title exceeds
    ``MAX_TITLE_LEN``; ``(#N)`` is preserved as the suffix.
    """
    safe_title = (issue_title or "").strip() or GH_UNAVAILABLE_TITLE
    suffix = f" (#{issue_id})"
    prefix = f"[gh] {repo}: "
    budget = MAX_TITLE_LEN - len(prefix) - len(suffix)
    if budget <= 0:
        # Repo name alone ate the budget; fall back to truncated prefix.
        truncated_prefix = prefix[: max(0, MAX_TITLE_LEN - len(suffix))]
        return truncated_prefix + suffix
    if len(safe_title) > budget:
        safe_title = safe_title[: max(0, budget - 1)].rstrip() + "…"
    return prefix + safe_title + suffix


def derive_done_when(
    issue_body: Optional[str],
    scout_body: Optional[str],
    scout_comments: Iterable[str] = (),
    issue_id: int = 0,
    repo: str = "",
    scout_card_id: str = "",
) -> tuple[str, bool]:
    """Spec §4 done-when derivation.

    Returns ``(done_when_text, needs_refinement)``. ``needs_refinement``
    is True only when the template fallback fired (so the caller can
    flag the new card with ``NEEDS_DONE_WHEN_REFINEMENT``).

    Priority:
        1. First sentence in ``issue_body`` that contains an action verb
           (must/should/needs to/fails when/cannot/won't/requires).
        2. First ``Done when:`` line from scout body or comments.
        3. Template fallback.
    """
    sources: list[tuple[str, str]] = []  # (label, text)
    if issue_body:
        sources.append(("issue", issue_body))
    if scout_body:
        sources.append(("scout", scout_body))
    for c in scout_comments or ():
        if c:
            sources.append(("scout-comment", c))

    # Priority 1: action sentence in any source.
    for _label, text in sources:
        sentence = _extract_action_sentence(text)
        if sentence:
            return ("Done when: " + sentence.rstrip(".") + "."), False

    # Priority 2: explicit "Done when:" line.
    for _label, text in sources:
        line = _extract_done_when_line(text)
        if line:
            return line, False

    # Priority 3: template.
    template = (
        f"Done when: the behavior described in {repo} issue #{issue_id} "
        f"is shipped in production and verified against the evidence "
        f"cited in the parent scout card {scout_card_id}."
    )
    return template, True


def _extract_action_sentence(text: str) -> Optional[str]:
    """First sentence containing an action verb (spec §4 priority 1)."""
    # Split on sentence boundaries but keep things readable. We split
    # on '.', '!', '?' followed by whitespace+capital/newline/EOF.
    parts = re.split(r"(?<=[.!?])\s+", text)
    for raw in parts:
        cleaned = _EMOJI_RE.sub("", raw)
        cleaned = _LEADING_BULLET_RE.sub("", cleaned)
        cleaned = _LEADING_QUOTE_RE.sub("", cleaned)
        cleaned = cleaned.strip()
        if not cleaned:
            continue
        if _ACTION_VERB_RE.search(cleaned):
            return cleaned
    return None


def _extract_done_when_line(text: str) -> Optional[str]:
    """First ``Done when:`` line, normalized to the canonical prefix."""
    for raw in text.splitlines():
        line = raw.strip()
        # Match "Done when: ..." (case-insensitive on the leading "when").
        m = re.match(r"^done\s*when\s*:\s*(.+)$", line, re.IGNORECASE)
        if m:
            return "Done when: " + m.group(1).strip()
    return None


def derive_cluster(scout_body: Optional[str]) -> Optional[str]:
    """Spec §4 cluster extraction.

    Looks for an explicit ``Cluster: …`` line or the parenthetical
    ``(cluster: …)`` form used by the reference scout cards. Returns
    the first match or ``None``.
    """
    if not scout_body:
        return None
    for raw in scout_body.splitlines():
        line = raw.strip()
        m = re.match(r"^cluster\s*:\s*(.+)$", line, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def build_child_body(
    *,
    repo: str,
    issue_id: int,
    issue_body_excerpt: Optional[str],
    scout_evidence_lines: Iterable[str],
    cluster: Optional[str],
    done_when: str,
    issue_url: str,
    scout_signal_id: Optional[str],
    scout_card_id: str,
) -> str:
    """Assemble the per-issue card body per spec §5 (fixed 7-section order)."""
    parts: list[str] = []

    # 1. Header
    parts.append(f"FULL CONTEXT: {repo} GitHub issue #{issue_id}.")

    # 2. Issue body excerpt (first 1500 chars per spec).
    if issue_body_excerpt:
        excerpt = issue_body_excerpt.strip()
        if len(excerpt) > 1500:
            excerpt = excerpt[:1497].rstrip() + "…"
        parts.append("")
        parts.append("## Issue body (excerpt)")
        parts.append(excerpt)

    # 3. Evidence block from the scout card, if any.
    evidence_lines = [l for l in scout_evidence_lines if l and l.strip()]
    if evidence_lines:
        parts.append("")
        parts.append("## Inherited evidence")
        parts.extend(evidence_lines)

    # 4. Cluster line.
    if cluster:
        parts.append("")
        parts.append(f"Cluster: {cluster}")

    # 5. Done when.
    parts.append("")
    parts.append(done_when)

    # 6. Link.
    parts.append("")
    parts.append(f"Link: {issue_url}")

    # 7. Acceptance footer + signal inheritance (combined so the
    # inherited signal id is the last line of the body, matching the
    # spec §5 footer shape).
    parts.append("")
    parts.append(
        "CORPUS-FIRST: check repo for existing coverage "
        f"({repo} issue area tests, related contract tests) and the "
        "signals DB before writing anything; extend what exists; cite "
        "what you found."
    )
    parts.append("")
    if scout_signal_id:
        parts.append(f"Inherited scout signal: {scout_signal_id}")
    else:
        parts.append(
            f"Inherited scout signal: {scout_card_id} "
            "(no signal emitted — backfill needed)"
        )

    return "\n".join(parts).rstrip() + "\n"


# ---------------------------------------------------------------------------
# gh CLI integration (spec §3 fallback)
# ---------------------------------------------------------------------------


@dataclass
class GhFetchResult:
    title: str
    body: Optional[str]
    error: Optional[str] = None


def resolve_repo_owner(
    owner: str,
    repo: str,
    extra_map: Optional[dict[str, str]] = None,
) -> tuple[str, str]:
    """Resolve the ``(owner, repo)`` pair for a ``gh issue view`` call.

    When ``owner`` is empty (a bare-ref scout where the parser only
    inferred the repo segment from the card title), look it up in
    ``REPO_OWNER_MAP`` (built-in) + ``extra_map`` (CLI-supplied
    overrides). Returns the canonical ``(owner, repo)`` pair, with
    ``owner`` empty when nothing maps.

    ``extra_map`` takes precedence over the built-in map so a CLI
    override always wins.
    """
    if owner:
        return (owner, repo)
    merged = {**REPO_OWNER_MAP, **(extra_map or {})}
    return (merged.get(repo, ""), repo)


def fetch_gh_issue(
    owner: str,
    repo: str,
    issue_id: int,
    *,
    timeout_seconds: float = 5.0,
    gh_path: str = "gh",
    repo_owner_map: Optional[dict[str, str]] = None,
) -> GhFetchResult:
    """Fetch the live issue title (and body excerpt) via ``gh``.

    Spec §3 fallback: if ``gh`` is unavailable, returns the placeholder
    title and an ``error`` string the caller can put in a card comment.

    GitHub CLI invocation (gh 2.97):

        gh issue view <issue_id> --repo <owner>/<repo> --json title,body

    Earlier versions accepted ``gh issue view <owner>/<repo>#<N>`` but
    gh 2.97 rejects that form ("invalid issue format: <X>#<N>"); we
    use ``--repo <owner>/<repo>`` so the same code works on both.

    When ``owner`` is empty we consult the built-in
    ``REPO_OWNER_MAP`` (plus the optional ``repo_owner_map``
    override) to fill it in. If the repo isn't mapped, we fall
    back to the placeholder title with an explanatory error so
    the operator can add the mapping via ``--repo-map``.
    """
    resolved_owner, repo = resolve_repo_owner(owner, repo, repo_owner_map)
    if not resolved_owner:
        return GhFetchResult(
            title=GH_UNAVAILABLE_TITLE,
            body=None,
            error=(
                f"gh issue view: bare repo '{repo}' has no owner mapping "
                f"(add via --repo-map {repo}=<owner> or extend "
                f"REPO_OWNER_MAP)"
            ),
        )
    repo_spec = f"{resolved_owner}/{repo}"
    issue_arg = str(issue_id)
    try:
        proc = subprocess.run(
            [
                gh_path,
                "issue",
                "view",
                issue_arg,
                "--repo",
                repo_spec,
                "--json",
                "title,body",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        return GhFetchResult(
            title=GH_UNAVAILABLE_TITLE,
            body=None,
            error=f"gh unavailable: {exc}",
        )
    except subprocess.TimeoutExpired as exc:
        return GhFetchResult(
            title=GH_UNAVAILABLE_TITLE,
            body=None,
            error=f"gh unavailable: timeout after {timeout_seconds}s",
        )
    if proc.returncode != 0:
        return GhFetchResult(
            title=GH_UNAVAILABLE_TITLE,
            body=None,
            error=(
                f"gh issue view {issue_arg} --repo {repo_spec} "
                f"exited {proc.returncode}: "
                f"{(proc.stderr or proc.stdout or '').strip()[:200]}"
            ),
        )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return GhFetchResult(
            title=GH_UNAVAILABLE_TITLE,
            body=None,
            error=f"gh returned non-JSON: {exc}",
        )
    title = (data.get("title") or "").strip() or GH_UNAVAILABLE_TITLE
    body = data.get("body")
    if isinstance(body, str):
        body = body.strip() or None
    else:
        body = None
    return GhFetchResult(title=title, body=body)


# ---------------------------------------------------------------------------
# Idempotency store (spec §6)
# ---------------------------------------------------------------------------


class FanoutKeyStore:
    """SQLite-backed ``(scout_card_id, issue_id)`` dedupe table.

    Lives at ``~/.hermes/hermes-agent/issue_fanout_keys.sqlite`` per the
    spec §6 parenthetical. Schema mirrors the spec verbatim:

        CREATE TABLE issue_fanout_keys (
            scout_card_id TEXT NOT NULL,
            issue_id      INTEGER NOT NULL,
            repo          TEXT NOT NULL,
            child_card_id TEXT,
            first_seen_at INTEGER NOT NULL,
            last_seen_at  INTEGER NOT NULL,
            PRIMARY KEY (scout_card_id, issue_id)
        );
    """

    SCHEMA = (
        "CREATE TABLE IF NOT EXISTS issue_fanout_keys ("
        "scout_card_id TEXT NOT NULL, "
        "issue_id INTEGER NOT NULL, "
        "repo TEXT NOT NULL, "
        "child_card_id TEXT, "
        "first_seen_at INTEGER NOT NULL, "
        "last_seen_at INTEGER NOT NULL, "
        "PRIMARY KEY (scout_card_id, issue_id))"
    )

    def __init__(self, db_path: str = DEFAULT_KEYS_DB) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        with self._connect() as conn:
            conn.execute(self.SCHEMA)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_issue_fanout_keys_child "
                "ON issue_fanout_keys(child_card_id)"
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def reserve(
        self,
        scout_card_id: str,
        issue_id: int,
        repo: str,
    ) -> Optional[str]:
        """Insert the key if new.

        Returns:
            * ``""`` if the key is freshly inserted OR was already
              present with no child yet — caller may proceed with
              ``kanban create`` (which itself is idempotent).
            * The existing ``child_card_id`` string if the key was
              already filled — caller should skip and count the
              duplicate.
        """
        now = int(time.time())
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT child_card_id FROM issue_fanout_keys "
                "WHERE scout_card_id = ? AND issue_id = ?",
                (scout_card_id, issue_id),
            )
            row = cur.fetchone()
            if row is not None:
                existing = row[0]
                # Refresh last_seen_at; keep first_seen_at as-is.
                conn.execute(
                    "UPDATE issue_fanout_keys SET last_seen_at = ? "
                    "WHERE scout_card_id = ? AND issue_id = ?",
                    (now, scout_card_id, issue_id),
                )
                conn.commit()
                if existing:
                    # Filled row — caller should skip.
                    return existing
                # Row exists but no child yet — caller may proceed.
                return ""
            conn.execute(
                "INSERT INTO issue_fanout_keys "
                "(scout_card_id, issue_id, repo, child_card_id, "
                "first_seen_at, last_seen_at) VALUES (?, ?, ?, NULL, ?, ?)",
                (scout_card_id, issue_id, repo, now, now),
            )
            conn.commit()
            return ""

    def mark_created(
        self,
        scout_card_id: str,
        issue_id: int,
        child_card_id: str,
        *,
        repo: str = "",
    ) -> None:
        """Record the child card id after a successful create.

        Defensive: tolerates being called WITHOUT a prior ``reserve``
        (e.g. when a backfill script reconstructs a child id from a
        recovered board) by inserting the row first, then updating the
        child id. The canonical call sequence is ``reserve`` then
        ``mark_created``; this fallback keeps the store consistent if
        callers deviate.
        """
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO issue_fanout_keys "
                "(scout_card_id, issue_id, repo, child_card_id, "
                "first_seen_at, last_seen_at) VALUES (?, ?, ?, NULL, ?, ?)",
                (scout_card_id, issue_id, repo or "", now, now),
            )
            conn.execute(
                "UPDATE issue_fanout_keys SET child_card_id = ?, "
                "last_seen_at = ? WHERE scout_card_id = ? AND issue_id = ?",
                (child_card_id, now, scout_card_id, issue_id),
            )
            conn.commit()


# ---------------------------------------------------------------------------
# Run-fanout orchestration
# ---------------------------------------------------------------------------


@dataclass
class FanoutResult:
    """The full outcome of one ``run_fanout`` call. JSON-serialisable."""

    scout_card_id: str
    board: str
    detected: int = 0
    scanned: int = 0
    created: list[dict[str, Any]] = field(default_factory=list)
    would_create: list[dict[str, Any]] = field(default_factory=list)
    skipped_duplicates: list[dict[str, Any]] = field(default_factory=list)
    excluded: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    dry_run: bool = False
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "scout_card_id": self.scout_card_id,
            "board": self.board,
            "detected": self.detected,
            "scanned": self.scanned,
            "skipped_duplicates": list(self.skipped_duplicates),
            "excluded": list(self.excluded),
            "errors": list(self.errors),
            "dry_run": self.dry_run,
            "duration_ms": self.duration_ms,
        }
        if self.dry_run:
            d["would_create"] = list(self.would_create)
        else:
            d["created"] = list(self.created)
        return d

    def exit_code(self) -> int:
        """Spec §7: 0 success/all-skip, 2 partial, 1 total failure.

        Excluded refs (awareness-only / do-not-file) are NOT errors
        — they're an expected outcome when the scout body says
        "do not file cards for these". They don't trigger non-zero
        exit codes.
        """
        if self.errors and not (self.created or self.would_create):
            return 1
        if self.errors and (self.created or self.would_create):
            return 2
        return 0


@dataclass
class FanoutDeps:
    """Pluggable dependencies so tests can run the pipeline without
    touching the real DB, gh CLI, or signals infrastructure.

    Defaults use the real implementations.
    """

    fetch_gh_issue: Any = None
    key_store: Any = None
    create_task: Any = None  # callable (conn, title, body, parents, idempotency_key) -> task_id
    signal_emit: Any = None  # callable (kind, tag, summary) -> signal_id or None
    parent_link_scan: Any = None  # callable (scout_id) -> set[int] of existing child issue_ids
    gh_path: str = "gh"
    # Repo → owner mapping overrides for bare-ref scouts (spec §11
    # follow-up / GAP 2). Merged on top of the built-in
    # ``REPO_OWNER_MAP`` inside ``fetch_gh_issue``. Keys are repo
    # segments (``"smilemap"``), values are owners (``"veroscale"``).
    repo_owner_map: Optional[dict[str, str]] = None

    def __post_init__(self) -> None:
        if self.fetch_gh_issue is None:
            self.fetch_gh_issue = fetch_gh_issue
        if self.key_store is None:
            self.key_store = FanoutKeyStore()
        # create_task / signal_emit / parent_link_scan are required at
        # run time; we don't default them because there's no safe
        # production-default for them.


def run_fanout(
    *,
    board: str,
    scout_card_id: str,
    scout: dict[str, Any],
    deps: FanoutDeps,
    max_issues: Optional[int] = None,
    skip_existing: bool = True,
    dry_run: bool = False,
    exclude_issues: Optional[Iterable[int]] = None,
) -> FanoutResult:
    """Execute the full fan-out pipeline for one scout card.

    ``scout`` is the minimal shape the implementation needs:
    ``{"id", "title", "body", "comments" (optional list[str])}``.

    ``exclude_issues`` is an optional iterable of issue ids that
    the parser will drop from the fan-out. The drop list is the
    union of (a) issue ids extracted from awareness-only /
    do-not-file markers in the scout body and (b) issue ids
    supplied via the ``--exclude-issues`` CLI flag (the
    ``exclude_issues`` argument).
    """
    started = time.monotonic()
    result = FanoutResult(scout_card_id=scout_card_id, board=board, dry_run=dry_run)

    body = scout.get("body") or ""
    title = scout.get("title") or ""
    comments = scout.get("comments") or []

    # GAP 1 (spec §11 follow-up): parse the scout body for
    # awareness-only / do-not-file markers. Any issue id listed
    # in such a marker is excluded from fan-out and surfaced in
    # ``result.excluded`` so the operator can audit exactly which
    # refs were dropped and why.
    excluded_from_body = parse_excluded_refs(body)
    excluded_id_to_meta: dict[int, ExcludedRef] = {
        e.issue_id: e for e in excluded_from_body
    }
    # Manual ``--exclude-issues`` overrides — recorded under a
    # different reason so the JSON summary distinguishes "scout
    # body said don't file" from "operator said don't file".
    manual_excluded_ids = {int(x) for x in (exclude_issues or [])}
    combined_excluded_ids = set(excluded_id_to_meta) | manual_excluded_ids
    for iid in sorted(manual_excluded_ids - set(excluded_id_to_meta)):
        result.excluded.append(
            {
                "issue_id": iid,
                "reason": "manual_override",
                "label": "",
                "excerpt": "",
            }
        )
    for ref in excluded_from_body:
        result.excluded.append(
            {
                "issue_id": ref.issue_id,
                "reason": ref.reason,
                "label": ref.label,
                "excerpt": ref.excerpt,
            }
        )

    default_repo = infer_default_repo(title)
    refs = parse_issue_links(
        body,
        default_repo=default_repo,
        excluded_issue_ids=combined_excluded_ids,
    )
    if max_issues is not None:
        refs = refs[:max_issues]
    result.detected = len(refs)
    result.scanned = len(refs)

    if not refs:
        result.duration_ms = int((time.monotonic() - started) * 1000)
        return result

    cluster = derive_cluster(body)
    # Pre-compute inherited scout signal id (real impl uses cfd signals query).
    scout_signal_id = None
    if deps.signal_emit is not None and not dry_run:
        try:
            scout_signal_id = deps.signal_emit(
                kind="lookup",
                tag=f"scout:{scout_card_id}",
                summary="lookup scout signal id",
            )
        except Exception:  # pragma: no cover — lookup best-effort
            scout_signal_id = None

    existing_children: set[int] = set()
    if deps.parent_link_scan is not None:
        try:
            existing_children = set(deps.parent_link_scan(scout_card_id) or [])
        except Exception:
            existing_children = set()

    for ref in refs:
        # §6 dedupe layer 1: file-based key store.
        if deps.key_store is not None:
            existing_child = deps.key_store.reserve(
                scout_card_id, ref.issue_id, ref.repo
            )
            if existing_child:
                result.skipped_duplicates.append(
                    {
                        "issue_id": ref.issue_id,
                        "reason": "child_card_id already set",
                        "child_card_id": existing_child,
                    }
                )
                continue
            if ref.issue_id in existing_children and skip_existing:
                # Layer 3 — parent-link scan caught a card that exists
                # but the key store hasn't been backfilled yet.
                result.skipped_duplicates.append(
                    {
                        "issue_id": ref.issue_id,
                        "reason": "existing child already linked to scout",
                    }
                )
                continue

        # Fetch live title/body (spec §3 fallback on gh failure).
        # GAP 2 (t_b17ae9d3): pass repo_owner_map so bare-ref scouts
        # like t_12cc81c6 (repo=smilemap, owner="") resolve to
        # veroscale/smilemap and ``gh issue view`` succeeds.
        fetch = deps.fetch_gh_issue(
            ref.owner,
            ref.repo,
            ref.issue_id,
            gh_path=deps.gh_path,
            repo_owner_map=deps.repo_owner_map,
        )

        child_title = build_child_title(ref.repo, ref.issue_id, fetch.title)
        # If we successfully resolved an owner (either from the URL
        # form or via the mapping), use it for the issue url;
        # otherwise leave the url with an "unknown-owner" prefix
        # but the card body will still record the placeholder so a
        # worker can fix it. The mapping means this should only
        # fire when the operator forgot to add a mapping.
        owner_for_url, _ = resolve_repo_owner(
            ref.owner, ref.repo, deps.repo_owner_map
        )
        if not owner_for_url:
            owner_for_url = "unknown-owner"
        issue_url = (
            f"https://github.com/{owner_for_url}/{ref.repo}/issues/{ref.issue_id}"
        )

        # §4 done-when.
        done_when, needs_refinement = derive_done_when(
            issue_body=fetch.body,
            scout_body=body,
            scout_comments=comments,
            issue_id=ref.issue_id,
            repo=ref.repo,
            scout_card_id=scout_card_id,
        )

        # §5 body.
        child_body = build_child_body(
            repo=ref.repo,
            issue_id=ref.issue_id,
            issue_body_excerpt=fetch.body,
            scout_evidence_lines=[],  # spec §5.3 — populated by future evidence extractor
            cluster=cluster,
            done_when=done_when,
            issue_url=issue_url,
            scout_signal_id=scout_signal_id,
            scout_card_id=scout_card_id,
        )

        idempotency_key = (
            f"{IDEMPOTENCY_KEY_PREFIX}{scout_card_id}:{ref.issue_id}"
        )

        if dry_run:
            entry = {
                "issue_id": ref.issue_id,
                "repo": ref.repo,
                "title": child_title,
                "idempotency_key": idempotency_key,
                "source": ref.source,
                "needs_done_when_refinement": needs_refinement,
            }
            if fetch.error:
                entry["gh_error"] = fetch.error
            result.would_create.append(entry)
            continue

        # Real create path.
        if deps.create_task is None:
            result.errors.append(
                {"issue_id": ref.issue_id, "error": "no create_task dependency wired"}
            )
            continue
        try:
            new_id = deps.create_task(
                title=child_title,
                body=child_body,
                parents=(scout_card_id,),
                idempotency_key=idempotency_key,
            )
        except Exception as exc:
            result.errors.append(
                {"issue_id": ref.issue_id, "error": f"create_task failed: {exc}"}
            )
            continue

        if deps.key_store is not None:
            deps.key_store.mark_created(scout_card_id, ref.issue_id, new_id)

        # Emit per-issue insight signal (spec §5).
        if deps.signal_emit is not None:
            try:
                deps.signal_emit(
                    kind="insight",
                    tag=f"issue-fanout:{scout_card_id}:{ref.issue_id}",
                    summary=(
                        f"fan-out: {scout_card_id} -> {new_id} for "
                        f"{ref.repo}#{ref.issue_id}"
                    ),
                )
            except Exception:  # pragma: no cover — best-effort
                pass

        result.created.append(
            {
                "issue_id": ref.issue_id,
                "repo": ref.repo,
                "child_card_id": new_id,
                "title": child_title,
                "issue_url": issue_url,
                "idempotency_key": idempotency_key,
                "inherited_signal": scout_signal_id,
                "source": ref.source,
                **({"gh_error": fetch.error} if fetch.error else {}),
            }
        )

    result.duration_ms = int((time.monotonic() - started) * 1000)
    return result


# ---------------------------------------------------------------------------
# Standalone CLI (useful for cron dry-runs)
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hermes-kanban-issue-triage-fanout",
        description=(
            "Fan a scout card's GitHub issue links out into one "
            "[gh] <repo>: ... (#N) child card per issue. Idempotent."
        ),
    )
    parser.add_argument("board", help="Board slug")
    parser.add_argument(
        "scout_card_id",
        nargs="?",
        help=(
            "Scout card id (e.g. t_12cc81c6). Omit only with "
            "--scan-all-scouts (cron entry point)."
        ),
    )
    parser.add_argument(
        "--scan-all-scouts", action="store_true",
        dest="scan_all_scouts",
        help=(
            "Scan every done scout card on the board (dry-run only). "
            "Mirrors `hermes kanban issue-triage-fanout ... --scan-all-scouts` "
            "for the cron entry point. Spec: t_9bbc7ec3 §9."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build titles + bodies, print planned summary, no creates.",
    )
    parser.add_argument("--max-issues", type=int, default=None,
                        help="Cap number of issues processed.")
    parser.add_argument("--skip-existing", dest="skip_existing",
                        action="store_true", default=True,
                        help="Skip if child already exists (default).")
    parser.add_argument("--no-skip-existing", dest="skip_existing",
                        action="store_false",
                        help="Force re-create (still hits idempotency_key).")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON summary.")
    parser.add_argument("--gh-path", default="gh",
                        help="Path to gh CLI (default: gh).")
    parser.add_argument("--keys-db", default=DEFAULT_KEYS_DB,
                        help=f"Idempotency key DB (default: {DEFAULT_KEYS_DB})")
    parser.add_argument(
        "--exclude-issues",
        dest="exclude_issues",
        default=None,
        help=(
            "Comma-separated issue ids to exclude from fan-out, "
            "in addition to any awareness-only / do-not-file markers "
            "parsed from the scout body (e.g. '898,899,900'). "
            "GAP 1 follow-up: lets operators hand-override what to skip."
        ),
    )
    parser.add_argument(
        "--repo-map",
        dest="repo_map",
        default=None,
        help=(
            "Comma-separated repo=owner overrides for bare-ref scouts "
            "(e.g. 'smilemap=veroscale,aurora=veroscale'). Merged on top "
            "of the built-in REPO_OWNER_MAP. GAP 2 follow-up: lets "
            "operators add new repo→owner mappings without editing the "
            "module."
        ),
    )
    return parser


def _parse_csv_ints(value: Optional[str]) -> list[int]:
    """Parse a comma-separated int list from a CLI string.

    Empty / None yields []. Negative numbers and zero are rejected
    (GitHub issue ids are positive). Non-numeric tokens raise
    ``argparse.ArgumentTypeError`` so the CLI exits with a clean
    usage error rather than a stack trace.
    """
    if not value:
        return []
    out: list[int] = []
    for tok in value.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            n = int(tok)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"--exclude-issues: expected comma-separated ints, "
                f"got {tok!r}: {exc}"
            )
        if n <= 0:
            raise argparse.ArgumentTypeError(
                f"--exclude-issues: issue ids must be positive, got {n}"
            )
        out.append(n)
    return out


def _parse_repo_map(value: Optional[str]) -> dict[str, str]:
    """Parse a ``--repo-map`` value like ``smilemap=veroscale,aurora=acme``.

    Empty / None yields {}. Tokens without ``=`` raise so the CLI
    exits with a clean usage error. Values are lowercased; keys are
    lowercased (matches how ``infer_default_repo`` returns lowercase
    repo segments).
    """
    if not value:
        return {}
    out: dict[str, str] = {}
    for tok in value.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise argparse.ArgumentTypeError(
                f"--repo-map: expected repo=owner pairs, got {tok!r}"
            )
        k, v = tok.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        if not k or not v:
            raise argparse.ArgumentTypeError(
                f"--repo-map: empty key or value in {tok!r}"
            )
        out[k] = v
    return out


def main(argv: Optional[list[str]] = None) -> int:
    # Parse the basic args first so we can validate the new
    # comma-separated flags up front (better error messages than
    # ``run_fanout`` raising deep in the loop).
    parser = build_argparser()
    args = parser.parse_args(argv)
    # Augment with parsed/typed versions of the CSV flags.
    try:
        args.exclude_issues_list = _parse_csv_ints(args.exclude_issues)
    except argparse.ArgumentTypeError as exc:
        print(f"issue-triage-fanout: {exc}", file=sys.stderr)
        return 2
    try:
        args.repo_map_dict = _parse_repo_map(args.repo_map)
    except argparse.ArgumentTypeError as exc:
        print(f"issue-triage-fanout: {exc}", file=sys.stderr)
        return 2

    # Resolve the scout card via the real kanban DB. We avoid importing
    # kanban_db at module import time so unit tests can import this
    # file without a DB.
    from hermes_cli import kanban_db as kb

    # --scan-all-scouts is the cron entry point (spec §9). It is
    # intentionally dry-run only — the cron never creates cards.
    # Reject the flag if the operator forgot `--dry-run` so we never
    # fan out the whole board from a cron tick.
    if getattr(args, "scan_all_scouts", False):
        if not args.dry_run:
            print(
                "issue-triage-fanout: --scan-all-scouts requires --dry-run "
                "(cron is report-only; per spec §9, no fan-out from the nightly cron).",
                file=sys.stderr,
            )
            return 2
        return _run_scan_all_scouts(args)

    if not args.scout_card_id:
        print(
            "issue-triage-fanout: <scout_card_id> is required "
            "(or pass --scan-all-scouts).",
            file=sys.stderr,
        )
        return 2

    kb.init_db()
    with kb.connect_closing() as conn:
        scout = kb.get_task(conn, args.scout_card_id)
        if scout is None:
            print(f"issue-triage-fanout: no such task: {args.scout_card_id}",
                  file=sys.stderr)
            return 1
        if scout.status != "done":
            print(
                f"issue-triage-fanout: scout card is not done "
                f"(status={scout.status}); skipping.",
                file=sys.stderr,
            )
            return 0

        # Resolve the existing children for layer-3 dedupe.
        child_ids = kb.child_ids(conn, args.scout_card_id)
        existing_issue_ids = _child_issue_ids_from_titles(conn, child_ids)

        comments = kb.list_comments(conn, args.scout_card_id) or []
        # kb.list_comments returns dataclass `Comment` objects (not dicts),
        # so use attribute access — `c.get` would AttributeError when the
        # scout has any comments. Caught by the scan-all path in t_28fd3b11.
        comment_bodies = [c.body or "" for c in comments]

        # Scout signal id lookup is best-effort; absent here.
        scout_signal_id = None
        try:
            from hermes_cli import calcifer_signals  # type: ignore
            scout_signal_id = calcifer_signals.lookup_scout_signal(
                args.scout_card_id
            )
        except Exception:
            scout_signal_id = None

        def _create_task(*, title: str, body: str, parents: tuple[str, ...],
                         idempotency_key: str) -> str:
            return kb.create_task(
                conn,
                title=title,
                body=body,
                assignee=None,
                created_by="issue-triage-fanout",
                parents=parents,
                idempotency_key=idempotency_key,
                # create_task validates initial_status against
                # VALID_INITIAL_STATUSES = {"running", "blocked"};
                # the actual landed status is recomputed from the
                # parents (any non-done parent → "todo"), so passing
                # "running" here just satisfies the validator. New
                # per-issue cards land in "todo" until the dispatcher
                # promotes them once the scout is done.
            )

        def _signal_emit(*, kind: str, tag: str, summary: str) -> Optional[str]:
            try:
                from hermes_cli import calcifer_signals  # type: ignore
                return calcifer_signals.emit(kind=kind, tag=tag, summary=summary)
            except Exception:
                return None

        def _parent_link_scan(scout_id: str) -> list[int]:
            return list(existing_issue_ids)

        deps = FanoutDeps(
            key_store=FanoutKeyStore(args.keys_db),
            create_task=_create_task,
            signal_emit=_signal_emit,
            parent_link_scan=_parent_link_scan,
            gh_path=args.gh_path,
            repo_owner_map=getattr(args, "repo_map_dict", None) or None,
        )

        result = run_fanout(
            board=args.board,
            scout_card_id=args.scout_card_id,
            scout={
                "id": scout.id,
                "title": scout.title,
                "body": scout.body,
                "comments": comment_bodies,
            },
            deps=deps,
            max_issues=args.max_issues,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
            exclude_issues=getattr(args, "exclude_issues_list", None) or None,
        )

    if args.json or args.dry_run:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(
            f"scout={result.scout_card_id} board={result.board} "
            f"detected={result.detected} created={len(result.created)} "
            f"skipped={len(result.skipped_duplicates)} "
            f"excluded={len(result.excluded)} "
            f"errors={len(result.errors)} duration_ms={result.duration_ms}"
        )
    return result.exit_code()


def _child_issue_ids_from_titles(conn: sqlite3.Connection, child_ids: list[str]) -> set[int]:
    """Layer-3 dedupe: scan child titles for ``(#NNN)`` issue ids."""
    issue_re = re.compile(r"\(#(\d+)\)\s*$")
    out: set[int] = set()
    if not child_ids:
        return out
    placeholders = ",".join("?" * len(child_ids))
    rows = conn.execute(
        f"SELECT title FROM tasks WHERE id IN ({placeholders})", child_ids
    ).fetchall()
    for (t,) in rows:
        if not t:
            continue
        m = issue_re.search(t)
        if m:
            out.add(int(m.group(1)))
    return out


def _run_scan_all_scouts(args) -> int:
    """Cron entry point: scan every done scout on the board in dry-run.

    Implements spec §9 — lists every ``status='done'`` task on the board,
    filters to those whose body matches the §2 heuristic (≥3 GitHub issue
    links), and runs each through the single-scout dry-run path. The
    aggregated JSON is emitted as the cron run's stdout.

    Side effects: NONE. No ``kanban create``, no dedupe table writes, no
    signals. The cron is intentionally a report-only diff checker.
    """
    import time as _time

    from hermes_cli import kanban_db as kb

    kb.init_db()
    started = _time.monotonic()
    per_scout: list[dict] = []
    grand_detected = 0
    grand_would_create = 0
    grand_skipped = 0
    grand_errors = 0

    with kb.connect_closing() as conn:
        all_done = kb.list_tasks(conn, status="done", include_archived=False)
        candidates = []
        for t in all_done:
            body = (t.body or "") + "\n"
            # Cheap pre-filter: ≥3 issue links per the §2 URL regex. The
            # single-card path does the full heuristic + section scoping
            # because that's where repo inference lives; this pre-filter
            # just avoids calling it 200 times for unrelated done tasks.
            url_count = len(_URL_RE.findall(body))
            if url_count >= 3:
                candidates.append((t.id, t.title, url_count))

    for scout_id, scout_title, url_count in candidates:
        argv = [args.board, scout_id]
        if getattr(args, "max_issues", None) is not None:
            argv.extend(["--max-issues", str(args.max_issues)])
        if getattr(args, "skip_existing", True) is False:
            argv.append("--no-skip-existing")
        if args.json:
            argv.append("--json")
        if getattr(args, "gh_path", None) and args.gh_path != "gh":
            argv.extend(["--gh-path", args.gh_path])
        if getattr(args, "keys_db", None):
            argv.extend(["--keys-db", args.keys_db])
        # GAP 1: propagate --exclude-issues through to each scout.
        # The cron sets this via the operator's flags; the
        # single-card path picks them up unchanged.
        if getattr(args, "exclude_issues", None):
            argv.extend(["--exclude-issues", args.exclude_issues])
        # GAP 2: same for --repo-map.
        if getattr(args, "repo_map", None):
            argv.extend(["--repo-map", args.repo_map])

        proc = subprocess.run(
            [sys.executable, "-m", "hermes_cli.issue_triage_fanout", *argv],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if proc.returncode not in (0, 2):
            per_scout.append(
                {
                    "scout_card_id": scout_id,
                    "title": scout_title,
                    "url_count": url_count,
                    "error": (
                        f"single-card dry-run failed: rc={proc.returncode} "
                        f"stderr={(proc.stderr or '').strip()[-400:]}"
                    ),
                }
            )
            grand_errors += 1
            continue

        try:
            summary = json.loads(proc.stdout) if proc.stdout.strip() else {}
        except json.JSONDecodeError:
            summary = {}

        detected = int(summary.get("detected", 0))
        would_create = len(summary.get("would_create", []))
        skipped = len(summary.get("skipped_duplicates", []))
        errors = len(summary.get("errors", []))
        grand_detected += detected
        grand_would_create += would_create
        grand_skipped += skipped
        grand_errors += errors

        per_scout.append(
            {
                "scout_card_id": scout_id,
                "title": scout_title,
                "url_count": url_count,
                "detected": detected,
                "would_create": would_create,
                "skipped_duplicates": skipped,
                "errors": errors,
                "duration_ms": summary.get("duration_ms"),
            }
        )

    duration_ms = int((_time.monotonic() - started) * 1000)
    envelope = {
        "board": args.board,
        "mode": "scan-all-scouts",
        "dry_run": True,
        "scanned_candidates": len(candidates),
        "totals": {
            "detected": grand_detected,
            "would_create": grand_would_create,
            "skipped_duplicates": grand_skipped,
            "errors": grand_errors,
        },
        "scouts": per_scout,
        "duration_ms": duration_ms,
    }
    print(json.dumps(envelope, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

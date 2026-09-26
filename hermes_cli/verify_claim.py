"""``hermes verify-claim`` - a machine-checkable landing-evidence bundle for one claim.

Executable slice of issue #111189 (evidence-carrying patches). Given an issue or PR
number, this command collects what a reviewer would otherwise collect by hand and
prints it as a JSON bundle plus a human summary:

* **is the target an issue or a PR** - one REST read (``/issues/<n>`` carries the
  ``pull_request`` key for PRs).
* **what is being claimed** - ``fixed`` (closed/completed), ``duplicate``
  (``state_reason: duplicate``), or ``fix-pr`` when the target *is* a PR.
* **did it land** - the canonical PR's merge commit, proven locally with
  ``git merge-base --is-ancestor <sha> <base>``. ``state_reason: completed`` is
  recorded as a *label* and never as landing evidence: that is exactly the
  failure mode #111189 cites (#91203, #42199, #72485 were all "completed"
  without a fix on ``main``).
* **is the repro recorded** - for fix PRs, the test names and the red-on-base ->
  green-after record parsed out of the PR body. The parser records what the body
  *attests* (``verified_by: body-text``); it does not re-run the tests. A missing
  record is reported ``missing``, never invented.

Exit codes (also carried in the bundle as ``exit_code``)::

    0  every required check passed - the claim carries verified evidence
    1  evidence missing or unverifiable (no merge commit, no repro record,
       no local clone to run the ancestry proof in)
    2  usage error: bad input, unknown target, ``gh`` unavailable
    3  refuted - a landing commit exists but is NOT an ancestor of the base ref

Network access goes through the ``gh`` CLI (same access pattern as
``hermes_cli/kanban_pr_acceptance.py``), so no HTTP client or token handling is
added here. All git work is local and read-only (plus an optional
``git fetch origin <branch>`` to resolve a SHA the clone does not have yet);
``--no-fetch`` turns that off.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA = "hermes.verify-claim/1"
DEFAULT_REPO = "NousResearch/hermes-agent"
DEFAULT_BASE_REF = "origin/main"

EXIT_OK = 0
EXIT_UNVERIFIED = 1
EXIT_USAGE = 2
EXIT_REFUTED = 3

_GH_TIMEOUT = 45.0
_GH_ATTEMPTS = 2
_GIT_TIMEOUT = 30.0
_MAX_CANDIDATES = 8

_SHA_RE = re.compile(r"[0-9a-f]{40}")
_REPO_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


class UsageError(RuntimeError):
    """Bad input, or a prerequisite (``gh``, a local clone) that is not available."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# subprocess seams (monkeypatched in tests)
# --------------------------------------------------------------------------- #

def _gh_api(endpoint: str, *, paginate: bool = False) -> Any:
    """Fetch one GitHub REST endpoint through the ``gh`` CLI.

    One bounded retry absorbs a transient network flap: a flake must not be
    reported as "unverified" evidence. Raises :class:`UsageError` on a missing
    binary, a timeout, a non-2xx response, or a non-JSON body. ``gh`` stderr is
    inspected only to label a 404 as "not found" - it is never echoed back, since
    it can carry host/auth details.
    """
    command = ["gh", "api", endpoint, "--hostname", "github.com"]
    if paginate:
        command += ["--paginate", "--slurp"]
    proc = None
    for attempt in range(_GH_ATTEMPTS):
        try:
            proc = subprocess.run(
                command, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=_GH_TIMEOUT,
            )
        except FileNotFoundError as exc:
            raise UsageError("gh CLI not found on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            if attempt + 1 < _GH_ATTEMPTS:
                time.sleep(1.5)
                continue
            raise UsageError(f"gh api timed out after {_GH_TIMEOUT:.0f}s: {endpoint}") from exc
        except OSError as exc:  # pragma: no cover - platform dependent
            raise UsageError(f"gh api could not run: {exc.__class__.__name__}") from exc
        if proc.returncode == 0:
            break
        if attempt + 1 < _GH_ATTEMPTS and "404" not in (proc.stderr or ""):
            time.sleep(1.5)
            continue
        break

    if proc is None or proc.returncode != 0:
        stderr = (proc.stderr if proc is not None else "") or ""
        label = "not found" if "404" in stderr else "request failed"
        code = proc.returncode if proc is not None else 1
        raise UsageError(f"gh api {label}: {endpoint} (gh exit {code}); check `gh auth status`")
    try:
        value = json.loads(proc.stdout or "null")
    except json.JSONDecodeError as exc:
        raise UsageError(f"gh api returned a non-JSON body for {endpoint}") from exc
    if paginate and isinstance(value, list):
        flat: list[Any] = []
        for page in value:
            flat.extend(page if isinstance(page, list) else [page])
        return flat
    return value


def _git(args: Sequence[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run a local ``git`` command. Never raises for a non-zero exit."""
    try:
        return subprocess.run(
            ["git", *args], stdin=subprocess.DEVNULL, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=_GIT_TIMEOUT, cwd=str(cwd),
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(["git", *args], 127, "", "git not found on PATH")
    except (subprocess.TimeoutExpired, OSError) as exc:
        return subprocess.CompletedProcess(["git", *args], 128, "", f"{exc.__class__.__name__}")


# --------------------------------------------------------------------------- #
# landing proof
# --------------------------------------------------------------------------- #

def verify_ancestry(
    sha: str | None, *, base_ref: str = DEFAULT_BASE_REF, cwd: Path, fetch: bool = True,
) -> dict:
    """Prove ``sha`` is reachable from ``base_ref`` in the local clone.

    ``status`` is ``verified`` (``git merge-base --is-ancestor`` exited 0),
    ``refuted`` (exit 1 - the commit exists but is not on the base branch) or
    ``unverified`` (no SHA, no clone, SHA not fetchable, base ref unknown).
    """
    command = ["git", "merge-base", "--is-ancestor", sha or "<sha>", base_ref]
    proof = {
        "kind": "is-ancestor",
        "command": command,
        "shell": " ".join(command),
        "status": "not-run",
        "sha_present_locally": None,
        "fetched": False,
        "exit_code": None,
        "detail": "",
    }
    if not sha:
        proof["detail"] = "no landing commit was identified, so there is nothing to test for ancestry"
        return proof

    cwd = Path(cwd)
    if _git(["rev-parse", "--git-dir"], cwd).returncode != 0:
        proof["status"] = "unverified"
        proof["detail"] = (
            f"not inside a git working tree ({cwd}); the is-ancestor proof needs a local clone"
        )
        return proof

    if _git(["cat-file", "-e", f"{sha}^{{commit}}"], cwd).returncode != 0:
        proof["sha_present_locally"] = False
        if fetch:
            branch = base_ref.split("/", 1)[1] if base_ref.startswith("origin/") else None
            fetch_args = ["fetch", "--quiet", "origin", branch] if branch else ["fetch", "--quiet", "origin"]
            fetched = _git(fetch_args, cwd)
            proof["fetched"] = True
            proof["fetch_exit_code"] = fetched.returncode
            if fetched.returncode != 0:
                proof["detail"] = f"`git {' '.join(fetch_args)}` failed (exit {fetched.returncode})"
        if not proof["fetched"]:
            proof["detail"] = "commit is not present in the local clone and --no-fetch was given"
        if _git(["cat-file", "-e", f"{sha}^{{commit}}"], cwd).returncode != 0:
            proof["status"] = "unverified"
            proof["detail"] = proof["detail"] or (
                f"commit {sha} is not present in the local clone; cannot prove ancestry"
            )
            return proof
        proof["sha_present_locally"] = True
    else:
        proof["sha_present_locally"] = True

    if _git(["rev-parse", "--verify", "--quiet", base_ref], cwd).returncode != 0:
        proof["status"] = "unverified"
        proof["detail"] = f"base ref {base_ref!r} does not exist locally; cannot run the ancestry proof"
        return proof

    result = _git(["merge-base", "--is-ancestor", sha, base_ref], cwd)
    proof["exit_code"] = result.returncode
    if result.returncode == 0:
        proof["status"] = "verified"
        proof["detail"] = f"{sha} is an ancestor of {base_ref}"
    elif result.returncode == 1:
        proof["status"] = "refuted"
        proof["detail"] = f"{sha} is NOT reachable from {base_ref}: the fix is not on the base branch"
    else:
        proof["status"] = "unverified"
        proof["detail"] = f"git merge-base --is-ancestor exited {result.returncode}"
    return proof


# --------------------------------------------------------------------------- #
# claim extraction
# --------------------------------------------------------------------------- #

_BODY_REF_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("duplicate", re.compile(r"(?i)\bduplicat\w*\s*(?:of|for|to)\s*:?\s*#(\d+)")),
    ("duplicate", re.compile(r"(?i)\bdupe\s+(?:of\s+)?#(\d+)")),
    ("fixed", re.compile(r"(?i)\b(?:fixed|closed|resolved|addressed)\s+(?:by|in|with)\s+(?:PR\s*)?#(\d+)")),
    ("fixed", re.compile(r"(?i)\bfix(?:es|ed)?\s+by\s+(?:PR\s*)?#(\d+)")),
)
_BODY_PR_URL = re.compile(r"(?i)https://github\.com/([\w.\-]+)/([\w.\-]+)/pull/(\d+)")
_BODY_URL_CLAIM = re.compile(r"(?i)\b(?:fixed|closed|resolved|addressed|fix(?:es|ed)?|duplicat\w*|dupe)\b")


def body_references(body: str | None) -> list[dict]:
    """``#N`` / PR-URL references in a body that read as an explicit claim ("fixed by #12").

    A bare PR URL ("Related discussion: <url>", a revert reference, a quoted
    reply) is not a claim, so URL matches only count when claim-adjacent
    wording precedes them on the same line. URL matches keep their own
    ``owner/repo`` and a distinct ``body-url-reference`` source so resolution
    can target the linked repo and ``resolve_landing`` can try timeline
    candidates first.
    """
    if not body:
        return []
    found: list[dict] = []
    seen: set[tuple] = set()
    for line_no, line in enumerate(body.splitlines(), 1):
        for kind, pattern in _BODY_REF_PATTERNS:
            for match in pattern.finditer(line):
                key = (kind, int(match.group(1)))
                if key in seen:
                    continue
                seen.add(key)
                found.append({"kind": kind, "number": key[1], "source": "body-reference",
                              "evidence": line.strip()[:200], "line": line_no})
        for match in _BODY_PR_URL.finditer(line):
            # ponytail: same-line verb check only; a claim verb on an earlier line does not qualify.
            if not _BODY_URL_CLAIM.search(line[:match.start()]):
                continue
            key = ("fixed", match.group(1).lower() + "/" + match.group(2).lower(), int(match.group(3)))
            if key in seen:
                continue
            seen.add(key)
            found.append({"kind": "fixed", "number": key[2], "repo": f"{match.group(1)}/{match.group(2)}",
                          "source": "body-url-reference",
                          "evidence": line.strip()[:200], "line": line_no})
    return found


def timeline_candidates(events: Iterable[dict], *, repo: str, self_number: int) -> list[dict]:
    """Canonical candidates from a timeline: cross-referenced PRs first, then other issues.

    Merged cross-referenced PRs come first because a merged PR is landing evidence,
    while another closed issue is only a pointer. A ``closed`` event's ``commit_id``
    (the commit GitHub recorded as closing the issue) is surfaced separately by
    :func:`close_commit`.
    """
    merged_prs: list[dict] = []
    other_prs: list[dict] = []
    issues: list[dict] = []
    seen: set[int] = set()
    for event in events or []:
        if event.get("event") != "cross-referenced":
            continue
        source = event.get("source") or {}
        item = source.get("issue") or {}
        number = item.get("number")
        if not isinstance(number, int) or number == self_number or number in seen:
            continue
        seen.add(number)
        pull = item.get("pull_request") or None
        entry = {
            "number": number,
            "kind": "pull_request" if pull else "issue",
            "title": item.get("title"),
            "url": item.get("html_url") or f"https://github.com/{repo}/issues/{number}",
            "state": item.get("state"),
            "source": "timeline-cross-reference",
            "merged_at": (pull or {}).get("merged_at"),
        }
        if pull and entry["merged_at"]:
            merged_prs.append(entry)
        elif pull:
            other_prs.append(entry)
        else:
            issues.append(entry)
    return merged_prs + other_prs + issues


def close_commit(events: Iterable[dict]) -> dict | None:
    """The commit GitHub recorded on the target's ``closed`` event, if any."""
    for event in events or []:
        if event.get("event") == "closed" and _SHA_RE.fullmatch(str(event.get("commit_id") or "")):
            return {"sha": event["commit_id"], "source": "closed-event-commit-id"}
    return None


def _issue_kind(payload: dict) -> str:
    return "pull_request" if payload.get("pull_request") else "issue"


def _target_summary(payload: dict, repo: str, kind: str) -> dict:
    number = payload.get("number")
    return {
        "number": number,
        "kind": kind,
        "title": payload.get("title"),
        "url": payload.get("html_url") or f"https://github.com/{repo}/{'pull' if kind == 'pull_request' else 'issues'}/{number}",
        "state": payload.get("state"),
        "state_reason": payload.get("state_reason"),
        # Recorded, never trusted: this is how the closer clicked, not what landed.
        "state_label_is_landing_evidence": False,
    }


def claim_kind(issue: dict) -> tuple[str, str]:
    """``(claim kind, basis)`` for an issue target.

    ``not_planned`` asserts nothing (the closer declined a fix), so it produces no
    landing requirement; ``completed`` is the label #111189 warns about and is
    verified rather than trusted.
    """
    reason = (issue.get("state_reason") or "").lower()
    if reason == "duplicate":
        return "duplicate", "state_reason"
    if reason == "not_planned":
        return "none", "state_reason=not_planned"
    if issue.get("state") == "closed":
        return "fixed", "state_reason" if issue.get("state_reason") else "closed-state"
    return "none", "open"


# --------------------------------------------------------------------------- #
# landing resolution
# --------------------------------------------------------------------------- #

def _pull_landing(pr: dict, repo: str, *, role: str) -> dict:
    """Landing candidate from a ``/pulls/<n>`` payload.

    ``merge_commit_sha`` is only landing evidence when ``merged`` is true: GitHub
    fills it with an ephemeral test-merge commit on closed-but-unmerged PRs.
    """
    number = pr.get("number")
    entry = {
        "merge_commit": None,
        "merged_at": None,
        "pull_request": {"number": number, "url": pr.get("html_url") or f"https://github.com/{repo}/pull/{number}",
                         "title": pr.get("title"), "role": role},
        "reason": "",
    }
    if pr.get("merged"):
        sha = str(pr.get("merge_commit_sha") or "")
        if _SHA_RE.fullmatch(sha):
            entry["merge_commit"] = sha
            entry["merged_at"] = pr.get("merged_at")
            entry["reason"] = f"PR #{number} is merged; merge commit {sha}"
            return entry
        entry["reason"] = f"PR #{number} is merged but the API exposed no 40-hex merge_commit_sha"
        return entry
    if pr.get("state") == "open":
        entry["reason"] = f"PR #{number} is still open; nothing has landed yet"
        return entry
    entry["reason"] = (
        f"PR #{number} is closed but was never merged - its merge_commit_sha is the ephemeral "
        "test-merge commit and is not landing evidence"
    )
    return entry


def _resolve_candidate(candidate: dict, repo: str) -> tuple[dict | None, dict | None, str]:
    """Resolve one candidate reference to a landing commit.

    Returns ``(landing, canonical, reason)``. A body reference does not know
    whether ``#N`` is a PR or an issue, so the kind is read from the API first;
    an issue candidate is followed one level down to its merged fix PR.
    """
    number = candidate["number"]
    resolved_candidate = dict(candidate)
    # A body URL names its own repo; a bare #N belongs to the target repo.
    target_repo = str(candidate.get("repo") or repo)
    if resolved_candidate.get("kind") not in {"pull_request", "issue"}:
        payload = _gh_api(f"repos/{target_repo}/issues/{number}")
        resolved_candidate["kind"] = _issue_kind(payload)
        if resolved_candidate["kind"] == "issue":
            resolved_candidate.setdefault("title", payload.get("title"))
            resolved_candidate.setdefault("state", payload.get("state"))

    if resolved_candidate["kind"] == "pull_request":
        pr = _gh_api(f"repos/{target_repo}/pulls/{number}")
        landing = _pull_landing(pr, target_repo, role="canonical")
        if landing["merge_commit"]:
            canonical = {**resolved_candidate, "kind": "pull_request",
                         "title": pr.get("title") or resolved_candidate.get("title"),
                         "state": pr.get("state"), "resolution": "merged-pr-merge-commit"}
            return landing, canonical, ""
        return None, None, f"#{number}: {landing['reason']}"

    nested = _gh_api(f"repos/{target_repo}/issues/{number}/timeline", paginate=True) or []
    for nested_candidate in timeline_candidates(nested, repo=target_repo, self_number=number):
        if nested_candidate["kind"] != "pull_request":
            continue
        pr = _gh_api(f"repos/{target_repo}/pulls/{nested_candidate['number']}")
        landing = _pull_landing(pr, target_repo, role="canonical")
        if landing["merge_commit"]:
            canonical = {**resolved_candidate, "kind": "issue", "resolution": "canonical-issue-merged-pr"}
            return landing, canonical, ""
    return None, None, f"#{number}: canonical issue has no merged fix PR"


def resolve_landing(target_number: int, target_kind: str, issue: dict, timeline: list, repo: str) -> tuple[dict, dict | None]:
    """Resolve the landing commit for the target's claim.

    Returns ``(landing, canonical)``. ``landing['merge_commit']`` is ``None``
    whenever no merged commit could be identified - in which case the caller must
    never report the claim as landed.
    """
    landing: dict = {
        "merge_commit": None, "merged_at": None, "pull_request": None,
        "evidence_kind": "merge_commit", "reason": "",
    }

    if target_kind == "pull_request":
        pr = _gh_api(f"repos/{repo}/pulls/{target_number}")
        landing.update(_pull_landing(pr, repo, role="target"))
        return landing, None

    refs = body_references(issue.get("body"))
    # URL matches read as weaker claims than verbs on #N, so timeline
    # cross-references outrank them: an incidental link to a merged PR must
    # not shadow the real canonical.
    refs, url_refs = ([c for c in refs if c.get("source") != "body-url-reference"],
                      [c for c in refs if c.get("source") == "body-url-reference"])
    # A linked PR in another repo is not landing evidence for this one: its #N is
    # that repo's number, and its merge commit can never be an ancestor of this
    # repo's base ref. Skip it rather than resolve a same-number PR here.
    foreign = [c for c in url_refs if c.get("repo") != repo]
    url_refs = [c for c in url_refs if c.get("repo") == repo]
    candidates = refs + timeline_candidates(
        timeline, repo=repo, self_number=target_number) + url_refs
    tried: list[str] = []
    for candidate in candidates[:_MAX_CANDIDATES]:
        number = candidate["number"]
        try:
            resolved, canonical, reason = _resolve_candidate(candidate, repo)
        except UsageError as exc:
            tried.append(f"#{number}: {exc}")
            continue
        if resolved:
            landing.update(resolved)
            landing["reason"] = (
                f"{resolved['reason']} (canonical of #{target_number} via {candidate['source']})"
            )
            return landing, canonical
        tried.append(reason)

    commit = close_commit(timeline)
    if commit:
        landing["merge_commit"] = commit["sha"]
        landing["reason"] = (
            f"the issue's closed event records closing commit {commit['sha']} "
            "(ancestry still has to be proven locally)"
        )
        return landing, None

    skipped = [f"#{c['number']} links to {c['repo']}, not {repo}" for c in foreign]
    landing["reason"] = (
        "no merged canonical fix PR was found"
        + (f"; tried {'; '.join(tried)}" if tried else "")
        + (f"; skipped {'; '.join(skipped)}" if skipped else "")
    )
    return landing, None


# --------------------------------------------------------------------------- #
# repro record
# --------------------------------------------------------------------------- #

_SECTION_RE = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*|\*\*\s*|__\s*)?"
    r"(evidence|test plan|tests?|how to test|how to verify|repro(?:duction)?|verification|"
    r"manual (?:test|verification)(?: steps)?)\b",
    re.I,
)
_HEADING_RE = re.compile(r"^\s{0,3}(?:#{1,6}\s+\S|\*\*\s*\S[^*]*\*\*\s*$|__\s*\S[^_]*__\s*$)")
_TEST_ID_RE = re.compile(r"[A-Za-z0-9_][\w./\\-]*test_[\w]+\.py(?:::{1,2}[\w:\[\].\-]+)?")
_RED_RE = re.compile(r"(?i)\b(?:red|fail(?:s|ed|ing)?|failure)\b")
_RED_CONTEXT_RE = re.compile(r"(?i)\b(?:on\s+base|base|main|before|pre-?fix|without\s+the\s+fix|reproduce[sd]?)\b")
_GREEN_RE = re.compile(r"(?i)\b(?:green|pass(?:es|ed|ing)?)\b")
_GREEN_CONTEXT_RE = re.compile(r"(?i)\b(?:after|post-?fix|now|with\s+the\s+fix|fixed|this\s+branch|branch|head|verified)\b")
_PASS_COUNT_RE = re.compile(r"(?i)\b\d+\s+(?:passed|passing)\b")
_ARROW_RE = re.compile(r"(?i)\b(?:red|fail\w*|failing)\b\s*(?:->|=>|-->|\u2192|\u21d2)\s*\b(?:green|pass\w*|passing)\b")
_COUNT_RE = re.compile(r"(?i)\b(\d+)\s+(passed|failed|failing)\b")


def parse_repro(body: str | None, *, source: str) -> dict:
    """Parse the repro record out of a PR body.

    Structured output only - the test identifiers and the red-on-base / green-after
    marker lines are copied verbatim out of the body. Nothing is inferred: a body
    with no section, no test identifier, and no red/green markers is ``missing``.
    ``verified_by`` is always ``body-text``; this command does not re-run tests.
    """
    repro: dict = {
        "status": "missing",
        "required": False,
        "source": source,
        "section": None,
        "tests": [],
        "red_on_base": {"observed": False, "evidence": None},
        "green_after": {"observed": False, "evidence": None},
        "evidence_lines": [],
        "counts": [],
        "verified_by": "body-text",
        "reason": "",
    }
    if not body or not body.strip():
        repro["reason"] = "no pull-request body to parse"
        return repro

    lines = body.splitlines()
    section: list[str] = []
    heading: str | None = None
    for index, line in enumerate(lines):
        if _SECTION_RE.match(line):
            heading = line.strip().strip("#").strip("*_").strip()
            for candidate in lines[index + 1:]:
                if _HEADING_RE.match(candidate):
                    break
                section.append(candidate)
            break
    if heading is None:
        # No section: still accept a labelled blue line such as "red -> green" evidence.
        section = [line for line in lines if _TEST_ID_RE.search(line) or _ARROW_RE.search(line)]
        if not section:
            repro["reason"] = "no evidence / test-plan / repro section and no test identifier in the body"
            return repro

    repro["section"] = heading
    tests: list[str] = []
    for line in section:
        for match in _TEST_ID_RE.finditer(line):
            test_id = match.group(0).rstrip(":.,;")
            if test_id and test_id not in tests:
                tests.append(test_id)
    repro["tests"] = tests[:20]

    # Marker detection runs in document order: a fail-with-base-context line before a
    # pass line is the red-on-base -> green-after record, whatever words it uses.
    red_line: str | None = None
    green_line: str | None = None
    for line in section:
        text = line.strip()
        if not text:
            continue
        if len(repro["evidence_lines"]) < 12 and (_TEST_ID_RE.search(text) or _RED_RE.search(text) or _GREEN_RE.search(text)):
            repro["evidence_lines"].append(text[:300])
        if _ARROW_RE.search(text):
            red_line = red_line or text
            green_line = green_line or text
        elif red_line is None and _RED_RE.search(text) and _RED_CONTEXT_RE.search(text):
            red_line = text
        if green_line is None and _GREEN_RE.search(text) and (
            _GREEN_CONTEXT_RE.search(text) or _PASS_COUNT_RE.search(text) or red_line is not None
        ):
            green_line = text
        for count in _COUNT_RE.finditer(text):
            record = f"{count.group(1)} {count.group(2).lower()}"
            if record not in repro["counts"]:
                repro["counts"].append(record)

    repro["red_on_base"] = {"observed": red_line is not None,
                            "evidence": red_line[:300] if red_line else None}
    repro["green_after"] = {"observed": green_line is not None,
                            "evidence": green_line[:300] if green_line else None}

    if repro["tests"]:
        repro["status"] = "present"
        repro["reason"] = f"{len(repro['tests'])} test identifier(s) recorded in the PR body"
    elif repro["red_on_base"]["observed"] and repro["green_after"]["observed"]:
        repro["status"] = "present"
        repro["reason"] = "red-on-base -> green-after recorded, but without test identifiers"
    else:
        repro["reason"] = "no test identifier and no red-on-base/green-after record in the body"
    return repro


def no_repro(reason: str, *, source: str | None = None) -> dict:
    return {
        "status": "not-applicable", "required": False, "source": source, "section": None,
        "tests": [], "red_on_base": {"observed": False, "evidence": None},
        "green_after": {"observed": False, "evidence": None}, "evidence_lines": [],
        "counts": [], "verified_by": "body-text", "reason": reason,
    }


# --------------------------------------------------------------------------- #
# bundle
# --------------------------------------------------------------------------- #

def collect_evidence(
    number: int, *, repo: str = DEFAULT_REPO, cwd: Path = Path("."),
    base_ref: str = DEFAULT_BASE_REF, fetch: bool = True, require_repro: bool = True,
) -> dict:
    """Build the evidence bundle for ``number``. Raises :class:`UsageError` if the target cannot be read."""
    cwd = Path(cwd)
    issue = _gh_api(f"repos/{repo}/issues/{number}")
    if not isinstance(issue, dict) or "number" not in issue:
        raise UsageError(f"gh api returned no issue/PR payload for #{number}")
    kind = _issue_kind(issue)
    target = _target_summary(issue, repo, kind)

    if kind == "pull_request":
        claim = {"kind": "fix-pr", "basis": "target-is-pull-request", "canonical": None}
        preref = [c for c in body_references(issue.get("body")) if c["kind"] == "fixed"]
        if preref:
            claim["references_issue"] = preref[0]["number"]
    else:
        claim_kind_name, basis = claim_kind(issue)
        claim = {"kind": claim_kind_name, "basis": basis, "canonical": None}
        refs = body_references(issue.get("body"))
        if refs and claim_kind_name in {"fixed", "duplicate"} and refs[0]["kind"] == claim_kind_name:
            claim["basis"] = "body-reference"
            claim["evidence"] = refs[0]["evidence"]

    timeline: list = []
    timeline_error: str | None = None
    if kind == "issue" and claim["kind"] == "none":
        # Nothing is claimed, so no landing evidence is required and no canonical
        # lookup is attempted: skip both the timeline read and the candidate walk.
        landing = {"merge_commit": None, "merged_at": None, "pull_request": None,
                   "evidence_kind": "none", "verdict": "not-claimed",
                   "reason": f"no fix or duplicate is asserted ({claim['basis']})"}
        canonical = None
    else:
        if kind == "issue":
            # Only an issue target needs the timeline (cross-references, close commit).
            try:
                timeline = _gh_api(f"repos/{repo}/issues/{number}/timeline", paginate=True) or []
            except UsageError as exc:
                timeline_error = str(exc)
        try:
            landing, canonical = resolve_landing(number, kind, issue, timeline, repo)
        except UsageError as exc:
            landing = {"merge_commit": None, "merged_at": None, "pull_request": None,
                       "evidence_kind": "merge_commit", "lookup_failed": True,
                       "reason": f"landing lookup failed: {exc}"}
            canonical = None
    if canonical:
        claim["canonical"] = canonical

    proof = verify_ancestry(landing.get("merge_commit"), base_ref=base_ref, cwd=cwd, fetch=fetch)
    if landing.get("merge_commit") and proof["status"] == "verified":
        landing["verdict"] = "landed"
    elif proof["status"] == "refuted":
        landing["verdict"] = "refuted"
    elif landing.get("merge_commit"):
        landing["verdict"] = "unverified"
    elif kind == "pull_request" and (issue.get("state") == "open"):
        landing["verdict"] = "not-merged"
    elif claim["kind"] == "none":
        landing["verdict"] = "not-claimed"
    else:
        landing["verdict"] = "unverified"

    repro = _repro_for(kind, issue, landing, number, repo, require_repro=require_repro)

    bundle = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "repo": repo,
        "base_ref": base_ref,
        "target": target,
        "claim": claim,
        "landing": landing,
        "proof": proof,
        "repro": repro,
        "timeline_error": timeline_error,
    }
    bundle["checks"] = _checks(bundle)
    return finalize(bundle)


def _repro_for(kind: str, issue: dict, landing: dict,
               number: int, repo: str, *, require_repro: bool = True) -> dict:
    """Repro record for the target PR, or (informational) for an issue's canonical PR.

    ``required`` is what gates the exit code: a fix PR's repro record is required by
    default (``--repro-optional`` downgrades it), while an issue target's canonical
    repro is reported but never gates - landing proof is the issue's evidence.
    """
    if kind == "pull_request":
        repro = parse_repro(issue.get("body"), source=f"pull:{number}:body")
        repro["required"] = bool(require_repro)
        return repro
    pull = landing.get("pull_request") or None
    if not pull or not pull.get("number"):
        return no_repro("target is an issue and no merged canonical PR was identified")
    try:
        pr = _gh_api(f"repos/{repo}/pulls/{pull['number']}")
    except UsageError as exc:
        return no_repro(f"canonical PR body unavailable: {exc}", source=f"pull:{pull['number']}:body")
    repro = parse_repro(pr.get("body"), source=f"pull:{pull['number']}:body")
    repro["canonical_for"] = number
    repro["required"] = False
    return repro


def _checks(bundle: dict) -> list[dict]:
    """Machine-checkable assertions. ``required`` drives the exit code."""
    target, claim, landing, proof, repro = (
        bundle["target"], bundle["claim"], bundle["landing"], bundle["proof"], bundle["repro"],
    )
    asserted = claim["kind"] in {"fixed", "duplicate", "fix-pr"}
    checks: list[dict] = []

    checks.append({
        "id": "target-resolved", "required": True, "status": "pass",
        "detail": f"{target['kind']} #{target['number']} is {target['state']}"
                  + (f" (state_reason: {target['state_reason']})" if target.get("state_reason") else ""),
    })

    if claim["kind"] == "none":
        checks.append({
            "id": "claim-identified", "required": False, "status": "na",
            "detail": f"#{target['number']} asserts nothing to verify (basis: {claim['basis']})",
        })
    else:
        checks.append({
            "id": "claim-identified", "required": True, "status": "pass",
            "detail": f"claim={claim['kind']} (basis: {claim['basis']})"
                      + (f", canonical #{claim['canonical']['number']}" if claim.get("canonical") else ", no canonical found"),
        })

    if asserted and landing["verdict"] not in {"not-merged", "not-claimed"}:
        checks.append({
            "id": "landing-commit-identified", "required": True,
            "status": "pass" if landing.get("merge_commit") else "fail",
            "detail": landing.get("merge_commit") or landing.get("reason") or "no merge commit identified",
        })
        checks.append({
            "id": "landing-commit-reachable-from-base", "required": True,
            "status": {"verified": "pass", "refuted": "fail"}.get(proof["status"], "fail"),
            "detail": f"{proof['shell']} -> {proof['status']}: {proof['detail']}",
        })
    else:
        reason = ("the target PR is not merged yet; landing is not asserted"
                  if landing["verdict"] == "not-merged" else "no landing claim to verify")
        checks.append({"id": "landing-commit-identified", "required": False, "status": "na", "detail": reason})
        checks.append({"id": "landing-commit-reachable-from-base", "required": False, "status": "na", "detail": reason})

    checks.append({
        "id": "state-label-not-used-as-landing-evidence", "required": True,
        "status": "pass" if not (landing["verdict"] == "landed" and not landing.get("merge_commit")) else "fail",
        "detail": "target.state_reason is recorded as a label only; landing is asserted only from a merge "
                  "commit reachable from the base ref",
    })

    if repro["status"] == "not-applicable":
        checks.append({"id": "repro-record", "required": False, "status": "na", "detail": repro["reason"]})
    else:
        checks.append({
            "id": "repro-record", "required": bool(repro.get("required")),
            "status": "pass" if repro["status"] == "present" else "fail",
            "detail": (f"{repro['status']}: {repro['reason']} (source: {repro['source']}, "
                       "attested by the PR body; not re-run by this command)"),
        })
    return checks


def finalize(bundle: dict) -> dict:
    """Derive ``verdict``/``ok``/``exit_code`` from the checks, and enforce the label invariant."""
    refuted = bundle["proof"]["status"] == "refuted" or bundle["landing"].get("verdict") == "refuted"
    landing = bundle["landing"]
    failed = [c for c in bundle["checks"] if c["required"] and c["status"] == "fail"]
    if bundle["landing"].get("verdict") == "landed" and not bundle["landing"].get("merge_commit"):
        # Belt-and-braces: a label alone must never reach the landed verdict.
        bundle["landing"]["verdict"] = "unverified"
        bundle["landing"]["reason"] = (
            (bundle["landing"].get("reason") or "")
            + " [invariant: landed requires a merge commit reachable from the base ref]"
        ).strip()
        refuted = False
        failed = [c for c in bundle["checks"] if c["required"] and c["status"] == "fail"]

    if refuted:
        verdict, exit_code = "refuted", EXIT_REFUTED
    elif bundle["claim"]["kind"] == "none":
        verdict, exit_code = "no-claim", EXIT_OK
    elif landing["verdict"] == "not-merged" and not failed:
        verdict, exit_code = "pending", EXIT_OK
    elif landing["verdict"] == "landed":
        verdict, exit_code = "landed", (EXIT_UNVERIFIED if failed else EXIT_OK)
    else:
        verdict, exit_code = "unverified", EXIT_UNVERIFIED

    bundle["verdict"] = verdict
    bundle["ok"] = exit_code == EXIT_OK
    bundle["exit_code"] = exit_code
    bundle["exit_codes"] = {
        "0": "evidence complete: every required check passed",
        "1": "evidence missing or unverifiable",
        "2": "usage error (bad input, unknown target, gh unavailable)",
        "3": "refuted: a landing commit exists but is not reachable from the base ref",
    }
    return bundle


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

_LABEL = {
    "landed": "LANDED", "unverified": "UNVERIFIED", "refuted": "REFUTED",
    "not-merged": "NOT MERGED", "not-claimed": "NO CLAIM", "pending": "PENDING (nothing landed yet)",
}


def render_bundle(bundle: dict) -> str:
    """Human-readable summary. The JSON bundle is the machine-checkable artifact."""
    target, claim, landing, proof, repro = (
        bundle["target"], bundle["claim"], bundle["landing"], bundle["proof"], bundle["repro"],
    )
    lines: list[str] = []
    lines.append(f"hermes verify-claim {target['number']} - {bundle['repo']}")
    lines.append("")
    state = f"{target['state']}"
    if target.get("state_reason"):
        state += f", state_reason={target['state_reason']}"
    lines.append(f"Target:  {target['kind']} #{target['number']} \"{(target.get('title') or '')[:70]}\" [{state}]")
    canonical = claim.get("canonical")
    lines.append(
        f"Claim:   {claim['kind']} (basis: {claim['basis']})"
        + (f" -> canonical {canonical['kind']} #{canonical['number']} [{canonical.get('state')}]" if canonical else "")
    )
    lines.append(f"Landing: {_LABEL.get(landing['verdict'], landing['verdict'])} - {landing.get('reason') or 'no reason recorded'}")
    if landing.get("merge_commit"):
        lines.append(f"         merge commit {landing['merge_commit']}"
                     + (f" merged {landing['merged_at']}" if landing.get("merged_at") else ""))
    lines.append(f"Proof:   {proof['shell']}")
    lines.append(f"         -> {proof['status']}: {proof['detail']}")
    if repro["status"] == "not-applicable":
        lines.append(f"Repro:   not applicable - {repro['reason']}")
    else:
        lines.append(f"Repro:   {repro['status']} (required={repro['required']}) - {repro['reason']}")
        for test in repro["tests"][:8]:
            lines.append(f"           test {test}")
        if repro["red_on_base"]["observed"]:
            lines.append(f"           red-on-base:  {repro['red_on_base']['evidence']}")
        if repro["green_after"]["observed"]:
            lines.append(f"           green-after:  {repro['green_after']['evidence']}")
    lines.append("")
    passed = sum(1 for c in bundle["checks"] if c["status"] == "pass")
    failed = [c for c in bundle["checks"] if c["status"] == "fail"]
    skipped = sum(1 for c in bundle["checks"] if c["status"] == "na")
    lines.append(f"Checks:  {passed} pass / {len(failed)} fail / {skipped} n/a")
    for check in failed:
        lines.append(f"  FAIL  {check['id']}: {check['detail']}")
    lines.append(f"Verdict: {bundle['verdict']} (exit {bundle['exit_code']})")
    lines.append("")
    lines.append("Note: state_reason is recorded as a label, never as landing evidence. "
                 "Repro records are parsed from the PR body and are not re-run here.")
    return "\n".join(lines)


def run_verify_claim_command(args) -> int:
    """CLI entry point. Returns the process exit code."""
    try:
        number = int(getattr(args, "number"))
    except (TypeError, ValueError):
        print("error: <number> must be an issue or PR number", file=sys.stderr)
        return EXIT_USAGE
    repo = getattr(args, "repo", None) or DEFAULT_REPO
    if not _REPO_RE.fullmatch(repo):
        print(f"error: --repo must be OWNER/REPO, got {repo!r}", file=sys.stderr)
        return EXIT_USAGE
    cwd = Path(getattr(args, "repo_dir", None) or ".")
    as_json = bool(getattr(args, "json", False))
    try:
        bundle = collect_evidence(
            number, repo=repo, cwd=cwd,
            base_ref=getattr(args, "base", None) or DEFAULT_BASE_REF,
            fetch=not getattr(args, "no_fetch", False),
            require_repro=not getattr(args, "repro_optional", False),
        )
    except UsageError as exc:
        if as_json:
            print(json.dumps({"schema": SCHEMA, "ok": False, "verdict": "error",
                              "error": str(exc), "exit_code": EXIT_USAGE}))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if as_json:
        print(json.dumps(bundle, indent=None if getattr(args, "compact", False) else 2))
    else:
        print(render_bundle(bundle))
    return int(bundle["exit_code"])

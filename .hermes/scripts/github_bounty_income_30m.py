#!/usr/bin/env python3
"""Deterministic GitHub bounty income scout gate.

No-agent cron script. It writes a report on every run and prints only when a
TAKE candidate needs user confirmation. Non-TAKE runs end with a wakeAgent=false
gate so Hermes stays silent.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERMES_HOME = Path(os.environ.get("HERMES_HOME", ".hermes")).resolve()
REPORT_DIR = HERMES_HOME / "reports" / "bounty-candidates"
WORKSPACE = HERMES_HOME / "bounty-workspace"
WATCHLIST_FILE = HERMES_HOME / "config" / "bounty-watchlist.json"
ALGORA_BOUNTIES_URL = "https://algora.io/bounties"
CRON_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

LANGUAGES = ("Python", "TypeScript", "Rust", "Go")
LIMIT_PER_QUERY = int(os.environ.get("BOUNTY_SCOUT_LIMIT_PER_QUERY", "18"))
ENRICH_LIMIT = int(os.environ.get("BOUNTY_SCOUT_ENRICH_LIMIT", "36"))
TAKE_THRESHOLD = int(os.environ.get("BOUNTY_SCOUT_TAKE_THRESHOLD", "70"))
PIPELINE_TIMEOUT_SECONDS = int(os.environ.get("BOUNTY_SCOUT_PIPELINE_TIMEOUT_SECONDS", "180"))
COMMAND_TIMEOUT_SECONDS = int(os.environ.get("BOUNTY_SCOUT_COMMAND_TIMEOUT_SECONDS", "30"))

FIELDS = "title,url,repository,labels,updatedAt,createdAt,commentsCount,body,number,state"
NOTIFY_CMD = [
    "osascript",
    "-e",
    'display notification "Found/Commented on Bounty" with title "Hermes Bounty Alert"',
]


def ensure_cli_path() -> None:
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    for entry in reversed(CRON_PATH.split(os.pathsep)):
        if entry not in path_entries:
            path_entries.insert(0, entry)
    os.environ["PATH"] = os.pathsep.join(entry for entry in path_entries if entry)


def load_env_file() -> None:
    env_path = HERMES_HOME / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        if key in {"GITHUB_TOKEN", "GH_TOKEN"} and key not in os.environ:
            os.environ[key] = value.strip().strip("'\"")

LABEL_QUERIES = ("bounty", "help wanted", "good first issue")
TERM_QUERIES = (
    '"$" bounty',
    '"paid issue"',
    '"paid task"',
    '"reward"',
    '"sponsor"',
    '"bounty"',
    '"USDC"',
    '"USD"',
)

NOISE_REPO_PATTERNS = (
    "awesome",
    "radar",
    "artifact",
    "bounties",
    "claim",
)
NOISE_TERMS = (
    "artifact aggregation",
    "radar",
    "claim:",
    "claim #",
    "duplicate claim",
    "social task",
    "share this",
    "retweet",
    "video submission",
    "run the miner",
    "banana bread",
)
PRIVATE_OR_UNSAFE_TERMS = (
    "exploit",
    "0day",
    "zero-day",
    "credential",
    "private key",
    "seed phrase",
    "password dump",
    "unauthorized",
    "scan mainnet",
    "private vulnerability",
)
AUTHORIZATION_TERMS = (
    "bounty",
    "reward",
    "paid",
    "sponsor",
    "grant",
    "prize",
    "payout",
)
ACCEPTANCE_TERMS = (
    "acceptance criteria",
    "requirements",
    "deliverables",
    "merged pr",
    "submit a pr",
    "pull request",
    "passes ci",
    "tests pass",
    "review",
)


@dataclass
class Candidate:
    title: str
    url: str
    repo: str
    labels: list[str]
    updated_at: str
    created_at: str
    comments_count: int
    language: str
    body: str = ""
    comments_text: str = ""
    linked_pr_count: int = 0
    state: str = ""
    number: int | None = None
    score: int = 0
    gate: str = "WATCH"
    reasons: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)


@dataclass
class WatchResult:
    target: dict[str, Any]
    status: str
    action: str
    issue_url: str
    existing_prs: list[str]
    reasons: list[str]
    next_command: str


@dataclass
class SourceResult:
    id: str
    url: str
    status: str
    highlights: list[str]
    reason: str = ""


def run_json(args: list[str], timeout: int = COMMAND_TIMEOUT_SECONDS) -> Any:
    try:
        completed = subprocess.run(
            args,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError:
        return []
    except subprocess.TimeoutExpired:
        return []
    if completed.returncode != 0:
        return []
    text = completed.stdout.strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return []


def load_watchlist() -> list[dict[str, Any]]:
    data = load_watch_config()
    items = data.get("watchlist") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def load_watch_sources() -> list[dict[str, Any]]:
    data = load_watch_config()
    sources = data.get("sources") if isinstance(data, dict) else None
    if not isinstance(sources, list):
        return [{"id": "algora-public-bounties", "url": ALGORA_BOUNTIES_URL}]
    return [source for source in sources if isinstance(source, dict)]


def load_watch_config() -> dict[str, Any]:
    if not WATCHLIST_FILE.exists():
        return {}
    try:
        data = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def strip_html(text: str) -> str:
    text = re.sub(r"<script\b.*?</script>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&amp;", "&").replace("&nbsp;", " ").replace("&#x27;", "'")
    return re.sub(r"\s+", " ", text).strip()


def inspect_sources() -> list[SourceResult]:
    results: list[SourceResult] = []
    for source in load_watch_sources():
        url = str(source.get("url") or "").strip()
        source_id = str(source.get("id") or url or "source")
        if not url:
            continue
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Hermes-Bounty-Scout/1.0"})
            with urllib.request.urlopen(request, timeout=25) as response:
                html = response.read().decode("utf-8", errors="replace")
        except Exception as exc:
            results.append(SourceResult(source_id, url, "unavailable", [], f"{type(exc).__name__}: {exc}"))
            continue

        visible = strip_html(html)
        highlights: list[str] = []
        for pattern in (
            r"Twenty \(YC S23\)\s+\$2,500\s+IMAP",
            r"Kyo\s+#390\s+\$500\s+gRPC Support",
            r"Isaac\s+#45\s+\$850\s+\[ISAAC-497\][^.]+?(?=Fund GitHub issues|Did you know|$)",
        ):
            match = re.search(pattern, visible, flags=re.I)
            if match:
                highlights.append(match.group(0).strip())
        if not highlights:
            money_matches = re.findall(r"[^.]{0,40}\$\s?[\d,]+[^.]{0,80}", visible)
            highlights = [item.strip() for item in money_matches[:5]]
        results.append(SourceResult(source_id, url, "ok", highlights[:8]))
    return results


def notify() -> None:
    try:
        subprocess.run(NOTIFY_CMD, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return


def repo_name(item: dict[str, Any]) -> str:
    repository = item.get("repository") or {}
    return repository.get("nameWithOwner") or ""


def labels_from(item: dict[str, Any]) -> list[str]:
    return [label.get("name", "") for label in item.get("labels") or [] if label.get("name")]


def candidate_from_item(item: dict[str, Any], language: str) -> Candidate | None:
    repo = repo_name(item)
    url = item.get("url") or ""
    if not repo or not url:
        return None
    number = item.get("number")
    return Candidate(
        title=item.get("title") or "",
        url=url,
        repo=repo,
        labels=labels_from(item),
        updated_at=item.get("updatedAt") or "",
        created_at=item.get("createdAt") or "",
        comments_count=int(item.get("commentsCount") or 0),
        language=language,
        body=item.get("body") or "",
        state=item.get("state") or "",
        number=int(number) if isinstance(number, int) else None,
    )


def gh_search(args: list[str], language: str) -> list[Candidate]:
    data = run_json(args)
    if not isinstance(data, list):
        return []
    candidates: list[Candidate] = []
    for item in data:
        if isinstance(item, dict):
            candidate = candidate_from_item(item, language)
            if candidate:
                candidates.append(candidate)
    return candidates


def scout(deadline: float) -> list[Candidate]:
    if not shutil.which("gh"):
        return []

    found: dict[str, Candidate] = {}
    for language in LANGUAGES:
        for label in LABEL_QUERIES:
            if time.monotonic() >= deadline:
                return list(found.values())
            args = [
                "gh",
                "search",
                "issues",
                "--label",
                label,
                "--language",
                language,
                "--state",
                "open",
                "--archived=false",
                "--sort",
                "updated",
                "--limit",
                str(LIMIT_PER_QUERY),
                "--json",
                FIELDS,
            ]
            for candidate in gh_search(args, language):
                found.setdefault(candidate.url, candidate)

        for term in TERM_QUERIES:
            if time.monotonic() >= deadline:
                return list(found.values())
            query = f"{term} is:issue is:open archived:false"
            args = [
                "gh",
                "search",
                "issues",
                query,
                "--language",
                language,
                "--state",
                "open",
                "--archived=false",
                "--match",
                "title,body,comments",
                "--sort",
                "updated",
                "--limit",
                str(LIMIT_PER_QUERY),
                "--json",
                FIELDS,
            ]
            for candidate in gh_search(args, language):
                found.setdefault(candidate.url, candidate)

    return list(found.values())


def issue_number(candidate: Candidate) -> str | None:
    if candidate.number is not None:
        return str(candidate.number)
    match = re.search(r"/issues/(\d+)", candidate.url)
    return match.group(1) if match else None


def enrich(candidate: Candidate) -> Candidate:
    number = issue_number(candidate)
    if not number:
        return candidate
    data = run_json(
        [
            "gh",
            "issue",
            "view",
            number,
            "--repo",
            candidate.repo,
            "--json",
            "title,url,state,body,labels,comments,createdAt,updatedAt,number",
        ]
    )
    if isinstance(data, dict):
        candidate.body = data.get("body") or candidate.body
        candidate.state = data.get("state") or candidate.state
        candidate.labels = labels_from(data) or candidate.labels
        candidate.created_at = data.get("createdAt") or candidate.created_at
        candidate.updated_at = data.get("updatedAt") or candidate.updated_at
        comments = data.get("comments") or []
        candidate.comments_text = "\n".join(str(comment.get("body") or "") for comment in comments)
        candidate.linked_pr_count = len(
            set(
                re.findall(
                    r"(?:/pull/|PR\s*#|pull request\s*#)(\d+)",
                    candidate.comments_text,
                    flags=re.I,
                )
            )
        )
    return candidate


def inspect_watch_target(target: dict[str, Any]) -> WatchResult:
    repo = str(target.get("repo") or "").strip()
    issue = target.get("issue")
    title = str(target.get("title") or target.get("id") or "watch target")
    configured_url = str(target.get("url") or "").strip()
    reasons = [str(target.get("notes") or "").strip()] if target.get("notes") else []
    existing_prs: list[str] = []
    issue_url = configured_url
    status = str(target.get("initial_status") or "WATCH")
    action = "WATCH"

    if repo and issue:
        issue_data = run_json(
            [
                "gh",
                "issue",
                "view",
                str(issue),
                "--repo",
                repo,
                "--json",
                "title,url,state,body,labels,comments,createdAt,updatedAt,closedByPullRequestsReferences",
            ]
        )
        if isinstance(issue_data, dict):
            issue_url = issue_data.get("url") or issue_url
            state = str(issue_data.get("state") or "?").upper()
            comments = issue_data.get("comments") or []
            closed_refs = issue_data.get("closedByPullRequestsReferences") or []
            comments_text = "\n".join(str(comment.get("body") or "") for comment in comments)
            linked = sorted(set(re.findall(r"(?:/pull/|PR\s*#|pull request\s*#)(\d+)", comments_text, re.I)))
            existing_prs.extend(f"#{number}" for number in linked[:8])
            for ref in closed_refs[:8]:
                if isinstance(ref, dict) and ref.get("url"):
                    existing_prs.append(str(ref["url"]))
            if state != "OPEN":
                action = "SKIP"
                reasons.append(f"issue state is {state}")
            elif len(comments) >= 15 or len(existing_prs) >= 3:
                action = "WATCH"
                reasons.append("crowded issue or multiple linked PR/attempt signals")
            else:
                action = "ANALYZE"
                reasons.append("open issue with manageable competition; read-only repo analysis may be useful")
            status = state

        pr_query = f"{title} repo:{repo}"
        pr_data = run_json(
            [
                "gh",
                "search",
                "prs",
                pr_query,
                "--state",
                "open",
                "--limit",
                "10",
                "--json",
                "title,url,state",
            ]
        )
        if isinstance(pr_data, list):
            for pr in pr_data[:5]:
                if isinstance(pr, dict) and pr.get("url"):
                    existing_prs.append(str(pr["url"]))
            if pr_data:
                reasons.append(f"{len(pr_data)} open PR search results for target title")
                if action == "ANALYZE":
                    action = "WATCH"
    else:
        action = "LOW_PRIORITY_WATCH" if configured_url else "SKIP"
        reasons.append("no GitHub repo/issue configured; keep as Algora/public-listing verification only")

    next_command = (
        f"gh issue view {issue} --repo {repo} --json title,url,state,labels,comments,updatedAt"
        if repo and issue
        else "Use public Algora page only; do not rely on deep links or browser bypasses."
    )
    return WatchResult(
        target=target,
        status=status,
        action=action,
        issue_url=issue_url,
        existing_prs=sorted(set(existing_prs))[:10],
        reasons=[reason for reason in reasons if reason],
        next_command=next_command,
    )


def inspect_watchlist() -> list[WatchResult]:
    return [inspect_watch_target(target) for target in load_watchlist()]


def has_amount(text: str) -> bool:
    return bool(
        re.search(
            r"(\$\s*\d+|\d[\d,.\s]*(xtm|rtc|sol|usd|usdc|usdt|dxtn|eur|gbp|inr|₹))",
            text,
            re.I,
        )
    )


def repo_noise(repo: str) -> bool:
    lower = repo.lower()
    return any(part in lower for part in NOISE_REPO_PATTERNS) and "bounty" not in lower


def score(candidate: Candidate) -> Candidate:
    labels_lower = [label.lower() for label in candidate.labels]
    text = f"{candidate.title}\n{candidate.body}\n{candidate.comments_text}\n{' '.join(labels_lower)}".lower()
    score_value = 0
    reasons: list[str] = []
    risk_flags: list[str] = []

    if repo_noise(candidate.repo):
        score_value -= 25
        risk_flags.append("possible radar/artifact/aggregation repository")

    if any(term in text for term in NOISE_TERMS):
        score_value -= 35
        risk_flags.append("noise, artifact, duplicate, or social-task signal")

    if any(term in text for term in PRIVATE_OR_UNSAFE_TERMS):
        score_value -= 45
        risk_flags.append("private or unauthorized security-sensitive request")

    if any(term in text for term in AUTHORIZATION_TERMS) or any("bounty" in label for label in labels_lower):
        score_value += 25
        reasons.append("maintainer-facing bounty/reward language")

    if any(label in {"bounty", "help wanted", "good first issue"} for label in labels_lower):
        score_value += 15
        reasons.append("useful public issue label")

    if any(label.startswith("bounty-") for label in labels_lower):
        score_value += 15
        reasons.append("bounty tier label")

    if has_amount(text):
        score_value += 25
        reasons.append("explicit amount or token")

    if any(term in text for term in ACCEPTANCE_TERMS):
        score_value += 20
        reasons.append("acceptance or submission path described")

    if "first come" in text or "first valid" in text or "merged pr" in text:
        score_value += 8
        reasons.append("award path described")

    if candidate.comments_count <= 6:
        score_value += 8
        reasons.append("not too crowded")
    elif candidate.comments_count >= 15:
        score_value -= 15
        risk_flags.append("crowded thread")
    if candidate.comments_count >= 25:
        score_value -= 20
        risk_flags.append("very crowded thread")

    if candidate.linked_pr_count >= 3:
        score_value -= 45
        risk_flags.append(f"crowded with {candidate.linked_pr_count}+ linked PRs")

    if candidate.language in {"Python", "TypeScript", "Rust", "Go"}:
        score_value += 5
        reasons.append("preferred implementation language")

    if candidate.state and candidate.state.upper() != "OPEN":
        score_value -= 60
        risk_flags.append("issue is not open")

    lacks_terms = not has_amount(text) and not any(term in text for term in AUTHORIZATION_TERMS)
    lacks_acceptance = not any(term in text for term in ACCEPTANCE_TERMS)
    if lacks_terms:
        score_value -= 20
        risk_flags.append("no clear payout or authorization terms")
    if lacks_acceptance:
        score_value -= 15
        risk_flags.append("unclear acceptance criteria")

    candidate.score = max(0, min(100, score_value))
    candidate.reasons = reasons
    candidate.risk_flags = risk_flags

    if any("private or unauthorized" in flag for flag in risk_flags):
        candidate.gate = "SKIP"
    elif candidate.linked_pr_count >= 3:
        candidate.gate = "WATCH"
    elif candidate.score >= TAKE_THRESHOLD and not lacks_terms and not lacks_acceptance:
        candidate.gate = "TAKE"
    elif candidate.score >= 50:
        candidate.gate = "WATCH"
    else:
        candidate.gate = "SKIP"
    return candidate


def maybe_clone(candidate: Candidate | None) -> Path | None:
    """Return the expected local repo path without cloning or fetching.

    Unattended bounty cron runs are read-only against GitHub and should not
    mutate local workspaces either. A TAKE is only a reminder for explicit user
    confirmation.
    """
    if not candidate or candidate.gate != "TAKE":
        return None
    return WORKSPACE / candidate.repo.replace("/", "__")


def report_header(now: str, gate: str, take: Candidate | None) -> list[str]:
    return [
        f"# GitHub Bounty Scout Report - {now}",
        "",
        "## Safety Envelope",
        "- Scout + Score + Repo Analysis Gate + Report during unattended runs.",
        "- Actual available workflow used: GitHub CLI search/issue inspection plus local deterministic scoring/reporting.",
        "- Agent skills are not loaded during this no-agent cron script; code-review/TDD/debugging skills are reserved for user-confirmed follow-up work.",
        "- Do not claim bounties, fork repositories, create branches, commit, push, comment, or open PRs.",
        "- Do not clone or fetch target repositories during unattended runs; report the next safe read-only command instead.",
        "- A TAKE means only that the user should confirm whether to proceed.",
        "- Any claiming/reserving, local code work, branch creation, committing, pushing, or PR creation requires separate explicit user instruction and must be allowed by the repository and bounty terms.",
        "- Must not run unauthorized scanning, spam comments, disclose private vulnerabilities publicly, fabricate findings, bypass platform rules, or publish payout-sensitive data.",
        "",
        "## Gate Result",
        f"- Decision: {gate}",
        f"- TAKE target: {take.url if take else 'none'}",
    ]


def render(
    candidates: list[Candidate],
    take: Candidate | None,
    cloned: Path | None,
    watch_results: list[WatchResult],
    source_results: list[SourceResult],
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gate = take.gate if take else ("WATCH" if any(c.gate == "WATCH" for c in candidates) else "SKIP")
    lines = report_header(now, gate, take)

    if source_results:
        lines.extend(["", "## Public Bounty Sources"])
        for result in source_results:
            lines.extend(
                [
                    f"- Source: `{result.id}`",
                    f"  - URL: {result.url}",
                    f"  - Status: `{result.status}`",
                    f"  - Highlights: {', '.join(result.highlights) if result.highlights else 'none'}",
                ]
            )
            if result.reason:
                lines.append(f"  - Reason: {result.reason}")

    if watch_results:
        lines.extend(["", "## Manual Watchlist"])
        for idx, result in enumerate(watch_results, 1):
            target = result.target
            skills = ", ".join(str(skill) for skill in target.get("skills", [])) or "unknown"
            existing_prs = ", ".join(result.existing_prs) if result.existing_prs else "none found by read-only check"
            lines.extend(
                [
                    f"{idx}. {target.get('title') or target.get('id')}",
                    f"   - Repo: `{target.get('repo') or 'unknown / Algora-only'}`",
                    f"   - Issue / bounty link: {result.issue_url or target.get('url') or 'unknown'}",
                    f"   - Bounty amount: `{target.get('bounty') or 'unknown'}`",
                    f"   - Current status: `{result.status}`",
                    f"   - Required skills: {skills}",
                    f"   - Estimated difficulty: `{target.get('estimated_difficulty') or 'unknown until repo analysis'}`",
                    f"   - Competition risk: `{'high' if result.existing_prs else 'medium/unknown'}`",
                    f"   - Existing PR already exists: {existing_prs}",
                    f"   - Recommended action: `{result.action}`",
                    f"   - Reason: {'; '.join(result.reasons) or 'none'}",
                    f"   - Next safe read-only command: `{result.next_command}`",
                ]
            )

    lines.extend(["", "## Ranked Candidates"])

    if not candidates:
        lines.append("- No candidates collected. Check `gh auth status`, network access, and GitHub search rate limits.")

    for idx, c in enumerate(candidates[:15], 1):
        body_excerpt = " ".join(c.body.split())[:280]
        lines.extend(
            [
                f"{idx}. [{c.title}]({c.url})",
                f"   - Repo: `{c.repo}`",
                f"   - Language: `{c.language}`",
                f"   - Labels: `{', '.join(c.labels) or 'none'}`",
                f"   - Created/Updated: `{c.created_at or '?'}` / `{c.updated_at or '?'}`",
                f"   - Comments: `{c.comments_count}`",
                f"   - Score/Gate: `{c.score}` / `{c.gate}`",
                f"   - Reasons: {', '.join(c.reasons) or 'none'}",
                f"   - Risks: {', '.join(c.risk_flags) or 'none'}",
            ]
        )
        if body_excerpt:
            lines.append(f"   - Body: {body_excerpt}")

    if take:
        likely_files = "Inspect after user confirms TAKE; start with issue-linked modules, tests, and CI config."
        lines.extend(
            [
                "",
                "## TAKE Action Plan",
                f"- Target repo: `{take.repo}`",
                f"- Issue URL: {take.url}",
                f"- Language: `{take.language}`",
                f"- Bounty/terms evidence: labels `{', '.join(take.labels) or 'none'}`; score reasons `{', '.join(take.reasons) or 'none'}`",
                f"- Legitimacy signals: open public issue, recent update `{take.updated_at or '?'}`, low linked-PR crowding `{take.linked_pr_count}`.",
                "- Smallest viable fix/contribution: after user confirmation, inspect the issue acceptance criteria and prepare the narrowest verifiable patch.",
                f"- Likely files: {likely_files}",
                f"- Suggested local workspace path after confirmation: `{cloned}`" if cloned else "- Suggested local workspace path: not available",
                f"- Next safe read-only command: `gh issue view {issue_number(take) or '<issue_id>'} --repo {take.repo} --json title,url,state,labels,comments,updatedAt`",
                "- Tests to run: project-specific unit tests, lint/typecheck, and the smallest regression test for the touched behavior.",
                "- Claim/reservation path: do not claim or reserve during the unattended run. Only do this after separate explicit user instruction and only if public issue/bounty rules permit it.",
                "- Branch/commit/push/PR path: do not create branches, commit, push, or open PRs during the unattended run. After separate explicit instruction, use the repository's contribution rules.",
                "",
                "## Validation Checklist",
                "- [ ] User confirms this TAKE target in the current thread",
                "- [ ] Re-read issue body and maintainer comments for payout and acceptance terms",
                "- [ ] Inspect issue and repository metadata with read-only commands first",
                "- [ ] Confirm any claim/reservation, branch, push, or PR action has separate explicit user instruction",
                "- [ ] Confirm those actions are allowed by public repository or bounty terms",
                "- [ ] Identify minimal files and tests",
                "- [ ] Prepare the minimal code change locally only after TAKE confirmation",
                "- [ ] Run local verification before any submission",
                "- [ ] Commit, push, or open PR/submission only after separate explicit instruction, passing verification, and rule confirmation",
                "",
                "## Rollback / Debug Path",
                f"- Remove local workspace if later created manually: `rm -rf {cloned}`" if cloned else "- No cloned workspace to remove",
                "- Re-run this script manually with `HERMES_HOME=.hermes python .hermes/scripts/github_bounty_income_30m.py`.",
                "- If GitHub returns no results, verify `gh auth status` and search rate limits.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Quiet Status",
                "- DONT_NOTIFY: No TAKE candidate met the gate on this run.",
                "- Reports are still written locally for review.",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    ensure_cli_path()
    load_env_file()
    deadline = time.monotonic() + PIPELINE_TIMEOUT_SECONDS
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACE.mkdir(parents=True, exist_ok=True)

    candidates = scout(deadline)
    seed_ranked = sorted(candidates, key=lambda c: (c.comments_count, c.updated_at), reverse=True)
    enriched: list[Candidate] = []
    for candidate in seed_ranked[:ENRICH_LIMIT]:
        if time.monotonic() >= deadline:
            break
        enriched.append(score(enrich(candidate)))

    seen = {candidate.url for candidate in enriched}
    for candidate in seed_ranked[ENRICH_LIMIT:]:
        if candidate.url not in seen:
            enriched.append(score(candidate))

    ranked = sorted(enriched, key=lambda c: c.score, reverse=True)
    take = next((candidate for candidate in ranked if candidate.gate == "TAKE"), None)
    cloned = maybe_clone(take)
    source_results = inspect_sources()
    watch_results = inspect_watchlist()
    report = render(ranked, take, cloned, watch_results, source_results)
    report_path = REPORT_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-github-bounty-income-30m.md"
    report_path.write_text(report, encoding="utf-8")

    if take:
        notify()
        print(report)
        print(json.dumps({"wakeAgent": True, "report": str(report_path), "take_count": 1, "score": take.score}))
    else:
        print('DONT_NOTIFY: No TAKE candidate met the gate on this run.')
        print(json.dumps({"wakeAgent": False, "report": str(report_path), "take_count": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

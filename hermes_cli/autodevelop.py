"""``hermes autodevelop`` — BYOK contributor loop over a maintainer-curated GitHub queue.

Resolves issues labeled for unattended work, claims them atomically, prints a
search-first one-shot prompt, optionally mirrors cards onto the local kanban
board, and stops on budget / max-issues / hard gates.

Live GitHub traffic uses ``GITHUB_TOKEN`` or ``GH_TOKEN`` (same as Skills Hub).
Mutating claim/park calls are off unless ``--commit`` is passed so dry runs stay
read-only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional
from urllib.parse import quote

import httpx

CLAIM_MARKER = "<!-- hermes-autodevelop-claim -->"
PARK_MARKER = "<!-- hermes-autodevelop-park -->"
DEFAULT_LABEL = "agent-ready"
DEFAULT_CLAIM_TTL_HOURS = 8
DEFAULT_MAX_ISSUES = 1
DEFAULT_MAX_CLAIMS_PER_USER = 2
SENSITIVE_LABEL = "autodevelop-allow-sensitive"
FORBIDDEN_TOUCH_TOKENS = (
    ".env",
    "secrets",
    "credentials",
    "/auth",
    "release",
    "nous_billing",
)
API_ROOT = "https://api.github.com"
USER_AGENT = "hermes-autodevelop"


# --------------------------------------------------------------------------------------
# Queue Contract
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class QueueContract:
    """Maintainer fields parsed from an issue body (plus label fallbacks)."""

    scope: Optional[str] = None
    touches: tuple[str, ...] = ()
    no_human_gate: Optional[bool] = None
    acceptance: tuple[str, ...] = ()


@dataclass
class QueueItem:
    number: int
    title: str
    html_url: str
    body: str
    labels: tuple[str, ...]
    assignees: tuple[str, ...]
    locked: bool
    repo: str
    contract: QueueContract
    skip_reason: Optional[str] = None

    @property
    def claimable(self) -> bool:
        return self.skip_reason is None


# --------------------------------------------------------------------------------------
# Contract Parsing And Safety
# --------------------------------------------------------------------------------------


_SCOPE_RE = re.compile(r"(?im)^\s*(?:[-*]\s*)?(?:scope|scope:)\s*[:=]?\s*(small|medium|large)\b")
_TOUCHES_RE = re.compile(r"(?im)^\s*(?:[-*]\s*)?(?:touches|touches:)\s*[:=]?\s*(.+)$")
_GATE_RE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?:no-human-gate|no_human_gate)\s*[:=]?\s*(true|false|yes|no)\b"
)
_ACCEPT_RE = re.compile(r"(?m)^\s*[-*]\s*\[\s*[ xX]?\s*\]\s+(.+)$")


def parse_queue_contract(body: str, labels: Iterable[str] = ()) -> QueueContract:
    text = body or ""
    label_list = [str(x).lower() for x in labels]
    scope_m = _SCOPE_RE.search(text)
    scope = scope_m.group(1).lower() if scope_m else None
    if scope is None:
        for token in ("scope:small", "scope:medium", "scope:large"):
            if token in label_list:
                scope = token.split(":", 1)[1]
                break
    touches: list[str] = []
    touch_m = _TOUCHES_RE.search(text)
    if touch_m:
        raw = touch_m.group(1).strip()
        touches = [p.strip() for p in raw.split(",") if p.strip()]
    gate_m = _GATE_RE.search(text)
    no_human_gate: Optional[bool] = None
    if gate_m:
        no_human_gate = gate_m.group(1).lower() in ("true", "yes")
    elif "no-human-gate" in label_list:
        no_human_gate = True
    acceptance = tuple(m.group(1).strip() for m in _ACCEPT_RE.finditer(text))
    return QueueContract(
        scope=scope,
        touches=tuple(touches),
        no_human_gate=no_human_gate,
        acceptance=acceptance,
    )


def touches_are_sensitive(touches: Iterable[str], *, allow_sensitive: bool) -> bool:
    if allow_sensitive:
        return False
    for path in touches:
        lowered = path.lower()
        if any(token in lowered for token in FORBIDDEN_TOUCH_TOKENS):
            return True
    return False


def skip_reason_for(
    *,
    locked: bool,
    assignees: Iterable[str],
    contract: QueueContract,
    labels: Iterable[str],
    include_assigned: bool,
    include_large: bool,
    include_human_gate: bool,
) -> Optional[str]:
    label_set = {str(x).lower() for x in labels}
    if locked:
        return "locked"
    if assignees and not include_assigned:
        return "assigned"
    if contract.scope == "large" and not include_large:
        return "scope:large"
    if contract.no_human_gate is False and not include_human_gate:
        return "needs-human-gate"
    allow_sensitive = SENSITIVE_LABEL in label_set
    if touches_are_sensitive(contract.touches, allow_sensitive=allow_sensitive):
        return "sensitive-paths"
    return None


# --------------------------------------------------------------------------------------
# GitHub Client
# --------------------------------------------------------------------------------------


def resolve_github_token() -> Optional[str]:
    for key in ("GITHUB_TOKEN", "GH_TOKEN"):
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    return None


class GitHubError(RuntimeError):
    """GitHub API call failed."""


class GitHubClient:
    """Minimal Issues API client. Transport is always a live httpx client unless injected."""

    def __init__(
        self,
        token: Optional[str] = None,
        *,
        transport: Optional[httpx.BaseTransport] = None,
        request: Optional[Callable[..., httpx.Response]] = None,
    ) -> None:
        self.token = token if token is not None else resolve_github_token()
        self._transport = transport
        self._request = request

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        if self._request is not None:
            return self._request(method, url, **kwargs)
        timeout = kwargs.pop("timeout", 30.0)
        with httpx.Client(timeout=timeout, transport=self._transport) as client:
            return client.request(method, url, headers=self._headers(), **kwargs)

    def list_issues(self, repo: str, *, label: str, limit: int = 30) -> list[dict[str, Any]]:
        owner, name = _split_repo(repo)
        params = {
            "state": "open",
            "labels": label,
            "per_page": str(min(max(limit, 1), 100)),
            "sort": "updated",
            "direction": "desc",
        }
        url = f"{API_ROOT}/repos/{quote(owner)}/{quote(name)}/issues"
        resp = self.request("GET", url, params=params)
        if resp.status_code >= 400:
            raise GitHubError(f"GET {url} failed: HTTP {resp.status_code}")
        rows = resp.json()
        if not isinstance(rows, list):
            raise GitHubError("unexpected issues payload")
        # Pull requests are also returned by this endpoint.
        return [row for row in rows if "pull_request" not in row][:limit]

    def get_issue(self, repo: str, number: int) -> dict[str, Any]:
        owner, name = _split_repo(repo)
        url = f"{API_ROOT}/repos/{quote(owner)}/{quote(name)}/issues/{int(number)}"
        resp = self.request("GET", url)
        if resp.status_code >= 400:
            raise GitHubError(f"GET {url} failed: HTTP {resp.status_code}")
        data = resp.json()
        if not isinstance(data, dict):
            raise GitHubError("unexpected issue payload")
        return data

    def list_comments(self, repo: str, number: int) -> list[dict[str, Any]]:
        owner, name = _split_repo(repo)
        url = f"{API_ROOT}/repos/{quote(owner)}/{quote(name)}/issues/{int(number)}/comments"
        resp = self.request("GET", url, params={"per_page": "100"})
        if resp.status_code >= 400:
            raise GitHubError(f"GET {url} failed: HTTP {resp.status_code}")
        rows = resp.json()
        return rows if isinstance(rows, list) else []

    def search_prior_art(self, repo: str, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        q = f"repo:{repo} {query}".strip()
        url = f"{API_ROOT}/search/issues"
        resp = self.request("GET", url, params={"q": q, "per_page": str(min(max(limit, 1), 10))})
        if resp.status_code >= 400:
            raise GitHubError(f"GET {url} failed: HTTP {resp.status_code}")
        data = resp.json()
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return []
        out = []
        for row in items:
            if not isinstance(row, dict):
                continue
            out.append({
                "number": row.get("number"),
                "title": row.get("title"),
                "html_url": row.get("html_url"),
                "is_pr": "pull_request" in row,
            })
        return out

    def post_comment(self, repo: str, number: int, body: str) -> dict[str, Any]:
        owner, name = _split_repo(repo)
        url = f"{API_ROOT}/repos/{quote(owner)}/{quote(name)}/issues/{int(number)}/comments"
        resp = self.request("POST", url, json={"body": body})
        if resp.status_code >= 400:
            raise GitHubError(f"POST {url} failed: HTTP {resp.status_code}")
        data = resp.json()
        return data if isinstance(data, dict) else {}

    def add_assignee(self, repo: str, number: int, login: str) -> dict[str, Any]:
        owner, name = _split_repo(repo)
        url = f"{API_ROOT}/repos/{quote(owner)}/{quote(name)}/issues/{int(number)}/assignees"
        resp = self.request("POST", url, json={"assignees": [login]})
        if resp.status_code >= 400:
            raise GitHubError(f"POST {url} failed: HTTP {resp.status_code}")
        data = resp.json()
        return data if isinstance(data, dict) else {}


def _split_repo(repo: str) -> tuple[str, str]:
    parts = (repo or "").strip().strip("/").split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("repo must be owner/name")
    return parts[0], parts[1]


def issue_from_api(repo: str, raw: dict[str, Any], **skip_kwargs: Any) -> QueueItem:
    labels = tuple(
        str(lab.get("name") if isinstance(lab, dict) else lab)
        for lab in (raw.get("labels") or [])
    )
    assignees = tuple(
        str(a.get("login") if isinstance(a, dict) else a)
        for a in (raw.get("assignees") or [])
        if a
    )
    body = str(raw.get("body") or "")
    contract = parse_queue_contract(body, labels)
    item = QueueItem(
        number=int(raw["number"]),
        title=str(raw.get("title") or ""),
        html_url=str(raw.get("html_url") or ""),
        body=body,
        labels=labels,
        assignees=assignees,
        locked=bool(raw.get("locked")),
        repo=repo,
        contract=contract,
    )
    item.skip_reason = skip_reason_for(
        locked=item.locked,
        assignees=item.assignees,
        contract=item.contract,
        labels=item.labels,
        include_assigned=bool(skip_kwargs.get("include_assigned", False)),
        include_large=bool(skip_kwargs.get("include_large", False)),
        include_human_gate=bool(skip_kwargs.get("include_human_gate", False)),
    )
    return item


# --------------------------------------------------------------------------------------
# Claims
# --------------------------------------------------------------------------------------


def claim_comment_body(*, repo: str, number: int, login: str) -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return (
        f"{CLAIM_MARKER}\n"
        f"I'll take this via `hermes autodevelop` (BYOK, contributor `{login}`).\n"
        f"Claimed `{repo}#{number}` at `{now}` UTC. Credit stays with the human contributor.\n"
    )


def park_comment_body(*, reason: str) -> str:
    return (
        f"{PARK_MARKER}\n"
        f"Parking this claim: {reason}\n"
        f"Releasing so another BYOK worker can pick it up.\n"
    )


def claim_is_fresh(comments: Iterable[dict[str, Any]], *, ttl_hours: int) -> bool:
    newest: Optional[datetime] = None
    for comment in comments:
        body = str(comment.get("body") or "")
        if CLAIM_MARKER not in body:
            continue
        created = comment.get("created_at")
        try:
            ts = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
        except ValueError:
            continue
        if newest is None or ts > newest:
            newest = ts
    if newest is None:
        return False
    age = datetime.now(timezone.utc) - newest
    return age.total_seconds() < max(ttl_hours, 1) * 3600


# --------------------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------------------


def draft_pr_command(item: QueueItem, *, draft: bool = True) -> str:
    flag = " --draft" if draft else ""
    return (
        f"gh pr create --repo {item.repo}{flag} --title {item.title!r} "
        f"--body 'Fixes #{item.number}\\n\\nCredit: human contributor (BYOK autodevelop).'"
    )


def execute_argv(prompt: str) -> list[str]:
    return [sys.executable, "-m", "hermes_cli.main", "chat", "-q", prompt, "--oneshot"]


def _git_remotes(cwd: str) -> str:
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "-C", cwd, "remote", "-v"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    return proc.stdout or ""


def bootstrap_report(cwd: str) -> dict[str, Any]:
    from pathlib import Path

    root = Path(cwd)
    missing = []
    if not (root / ".git").exists() and not (root / ".git").is_file():
        missing.append("git-checkout")
    if not (root / "AGENTS.md").is_file() and not (root / "CONTRIBUTING.md").is_file():
        missing.append("project-law")
    remotes = _git_remotes(cwd)
    if "github.com" not in remotes.lower():
        missing.append("github-remote")
    in_venv = (
        bool(os.environ.get("VIRTUAL_ENV"))
        or getattr(sys, "prefix", "") != getattr(sys, "base_prefix", "")
        or (root / ".venv").exists()
        or (root / "venv").exists()
    )
    if not in_venv:
        missing.append("venv")
    return {"cwd": str(root), "ok": not missing, "missing": missing, "remotes": remotes}


def codeowners_paths(cwd: str) -> tuple[str, ...]:
    from pathlib import Path

    root = Path(cwd)
    for rel in (".github/CODEOWNERS", "CODEOWNERS", "docs/CODEOWNERS"):
        path = root / rel
        if not path.is_file():
            continue
        found: list[str] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            token = stripped.split()[0]
            if token.startswith("/") or "/" in token or token.endswith("*"):
                found.append(token)
        return tuple(found)
    return ()


def persist_claim(repo: str, login: str, number: int) -> int:
    """Record a claim and return how many open claims this login holds on repo."""
    from pathlib import Path

    from hermes_constants import get_hermes_home

    path = Path(get_hermes_home()) / "autodevelop" / "claims.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except json.JSONDecodeError:
            data = {}
    repo_map = data.setdefault(repo, {})
    if not isinstance(repo_map, dict):
        repo_map = {}
        data[repo] = repo_map
    rec = repo_map.setdefault(login, {"open": []})
    opened = rec.setdefault("open", [])
    if number not in opened:
        opened.append(number)
    rec["updated"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return len(opened)


def open_claim_count(repo: str, login: str) -> int:
    from pathlib import Path

    from hermes_constants import get_hermes_home

    path = Path(get_hermes_home()) / "autodevelop" / "claims.json"
    if not path.is_file():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    rec = ((data.get(repo) or {}) if isinstance(data, dict) else {}).get(login) or {}
    opened = rec.get("open") if isinstance(rec, dict) else []
    return len(opened) if isinstance(opened, list) else 0


def draft_pr_argv(item: QueueItem, *, draft: bool = True) -> list[str]:
    argv = ["gh", "pr", "create", "--repo", item.repo]
    if draft:
        argv.append("--draft")
    argv.extend([
        "--title", item.title,
        "--body", f"Fixes #{item.number}\n\nCredit: human contributor (BYOK autodevelop).",
    ])
    return argv


def test_runner_argv(cwd: str) -> list[str]:
    from pathlib import Path

    script = Path(cwd) / "scripts" / "run_tests.sh"
    if script.is_file():
        return ["bash", str(script), "-q"]
    return [sys.executable, "-m", "pytest", "tests/", "-q"]


def format_prior_art(rows: Iterable[dict[str, Any]]) -> str:
    lines = []
    for row in rows:
        kind = "PR" if row.get("is_pr") else "issue"
        lines.append(f"- {kind} #{row.get('number')}: {row.get('title')} ({row.get('html_url')})")
    return "\n".join(lines) if lines else "- (none found)"


def build_oneshot_prompt(
    item: QueueItem,
    *,
    draft_pr: bool = True,
    prior_art: Iterable[dict[str, Any]] = (),
) -> str:
    owner, name = _split_repo(item.repo)
    search = item.title.replace('"', " ").strip()[:80]
    acceptance = "\n".join(f"- {row}" for row in item.contract.acceptance) or "- (none listed)"
    touches = ", ".join(item.contract.touches) or "(not specified)"
    draft_line = (
        "Open a **draft** PR linked to this issue. Do not merge. Do not force-push foreign branches."
        if draft_pr
        else "Open a PR linked to this issue. Do not merge. Do not force-push foreign branches."
    )
    art = format_prior_art(prior_art)
    return f"""You are contributing to {item.repo} as the human operator's BYOK agent.

# Task
Issue #{item.number}: {item.title}
{item.html_url}

# Project law
Read AGENTS.md and CONTRIBUTING.md first. Credit the human contributor, not the tool.
Skills vs core: do not dump third-party plugins into this repo.

# Search-first (already run; inspect before coding)
Query: `{search}` on {owner}/{name}
{art}
A matching PR is prior art, not authority. Do not open a duplicate issue or PR.
Fork PR flow only. Suggested ship command:
  {draft_pr_command(item, draft=draft_pr)}

# Queue contract
- scope: {item.contract.scope or "unspecified"}
- touches: {touches}
- no-human-gate: {item.contract.no_human_gate}
- acceptance:
{acceptance}

# Safety
- BYOK only. Never bill maintainer inference.
- {draft_line}
- Refuse secrets, release, and auth surfaces unless the issue has `{SENSITIVE_LABEL}`.
- On ambiguity, auth, or missing secrets: park with `hermes autodevelop park` and stop.

# Ship or park
Implement, run the repo's tests for the touched surface, then {draft_line.lower()}
If blocked, comment the blocker and release the claim.
"""


# --------------------------------------------------------------------------------------
# Kanban Mirror
# --------------------------------------------------------------------------------------


def kanban_idempotency_key(repo: str, number: int) -> str:
    return f"github:{repo}#{int(number)}"


def sync_item_to_kanban(item: QueueItem) -> str:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    key = kanban_idempotency_key(item.repo, item.number)
    title = f"{item.repo}#{item.number}: {item.title}"[:200]
    body = (
        f"Mirrored from {item.html_url}\n\n"
        f"{build_oneshot_prompt(item)}"
    )
    with kbc.connect_closing() as conn:
        return kb.create_task(
            conn,
            title=title,
            body=body,
            created_by="autodevelop",
            idempotency_key=key,
            initial_status="ready",
            skills=["autodevelop"],
        )


# --------------------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------------------


def apply_codeowners_skip(item: QueueItem, owners: Iterable[str]) -> None:
    if item.skip_reason or not item.contract.touches:
        return
    allow_sensitive = SENSITIVE_LABEL in {lab.lower() for lab in item.labels}
    if allow_sensitive:
        return
    for owned in owners:
        owned_l = owned.lower()
        if not any(token in owned_l for token in FORBIDDEN_TOUCH_TOKENS):
            continue
        for touch in item.contract.touches:
            if touch.lower().strip("/") in owned_l or owned_l.strip("/") in touch.lower():
                item.skip_reason = "codeowners-sensitive"
                return


def mark_in_progress_claims(
    client: GitHubClient,
    items: Iterable[QueueItem],
    *,
    ttl_hours: int,
) -> None:
    for item in items:
        if item.skip_reason:
            continue
        comments = client.list_comments(item.repo, item.number)
        if claim_is_fresh(comments, ttl_hours=ttl_hours):
            item.skip_reason = "in-progress-claim"


def resolve_queue(
    client: GitHubClient,
    repo: str,
    *,
    label: str,
    limit: int,
    include_assigned: bool = False,
    include_large: bool = False,
    include_human_gate: bool = False,
    ttl_hours: int = DEFAULT_CLAIM_TTL_HOURS,
    cwd: Optional[str] = None,
) -> list[QueueItem]:
    raws = client.list_issues(repo, label=label, limit=max(limit, 1) * 2)
    items = [
        issue_from_api(
            repo,
            raw,
            include_assigned=include_assigned,
            include_large=include_large,
            include_human_gate=include_human_gate,
        )
        for raw in raws
    ]
    apply_codeowners_skip_all = codeowners_paths(cwd or os.getcwd())
    for item in items:
        apply_codeowners_skip(item, apply_codeowners_skip_all)
    mark_in_progress_claims(client, items, ttl_hours=ttl_hours)
    return items


def claim_item(
    client: GitHubClient,
    item: QueueItem,
    *,
    login: str,
    ttl_hours: int,
    commit: bool,
    assign: bool = True,
) -> dict[str, Any]:
    comments = client.list_comments(item.repo, item.number)
    if claim_is_fresh(comments, ttl_hours=ttl_hours):
        return {"claimed": False, "reason": "fresh-claim", "number": item.number}
    body = claim_comment_body(repo=item.repo, number=item.number, login=login)
    if not commit:
        return {"claimed": False, "reason": "dry-run", "number": item.number, "comment": body}
    posted = client.post_comment(item.repo, item.number, body)
    assigned = False
    if assign and login:
        client.add_assignee(item.repo, item.number, login)
        assigned = True
    persist_claim(item.repo, login, item.number)
    return {
        "claimed": True,
        "number": item.number,
        "comment_id": posted.get("id"),
        "assigned": assigned,
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser(parent_subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = parent_subparsers.add_parser(
        "autodevelop",
        help="Drain a maintainer-curated GitHub issue queue with BYOK",
        description=(
            "Contributor-local BYOK loop: list agent-ready issues, claim one, "
            "print a search-first prompt, optionally mirror onto kanban. "
            "Never bills maintainer inference. Draft PRs by default."
        ),
    )
    sub = parser.add_subparsers(dest="autodevelop_action")

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--repo", required=True, help="owner/name of the opted-in repository")
        sp.add_argument("--label", default=DEFAULT_LABEL, help=f"Queue label (default: {DEFAULT_LABEL})")

    p_list = sub.add_parser("list", help="List claimable queue issues")
    add_common(p_list)
    p_list.add_argument("--limit", type=int, default=20)
    p_list.add_argument("--include-assigned", action="store_true")
    p_list.add_argument("--include-large", action="store_true")
    p_list.add_argument("--include-human-gate", action="store_true")
    p_list.add_argument("--json", action="store_true")

    p_prompt = sub.add_parser("prompt", help="Print the one-shot implement prompt for an issue")
    add_common(p_prompt)
    p_prompt.add_argument("number", type=int)
    p_prompt.add_argument("--no-draft-pr", action="store_true")

    p_claim = sub.add_parser("claim", help="Atomically claim an issue (comment). Dry-run unless --commit")
    add_common(p_claim)
    p_claim.add_argument("number", type=int)
    p_claim.add_argument("--login", default="", help="Human contributor login for the claim comment")
    p_claim.add_argument("--ttl-hours", type=int, default=DEFAULT_CLAIM_TTL_HOURS)
    p_claim.add_argument("--commit", action="store_true", help="POST the claim comment (off by default)")

    p_park = sub.add_parser("park", help="Release a claim with a blocker comment. Dry-run unless --commit")
    add_common(p_park)
    p_park.add_argument("number", type=int)
    p_park.add_argument("--reason", required=True)
    p_park.add_argument("--commit", action="store_true")

    p_sync = sub.add_parser("sync-kanban", help="Mirror claimable queue items onto the local kanban board")
    add_common(p_sync)
    p_sync.add_argument("--limit", type=int, default=20)
    p_sync.add_argument("--include-assigned", action="store_true")
    p_sync.add_argument("--include-large", action="store_true")
    p_sync.add_argument("--include-human-gate", action="store_true")

    p_run = sub.add_parser("run", help="Unattended list → claim → prompt loop (dry-run claims unless --commit)")
    add_common(p_run)
    p_run.add_argument("--max-issues", type=int, default=DEFAULT_MAX_ISSUES)
    p_run.add_argument("--budget", type=int, default=0, help="Max claims this invocation (0 = use --max-issues)")
    p_run.add_argument("--login", default="")
    p_run.add_argument("--ttl-hours", type=int, default=DEFAULT_CLAIM_TTL_HOURS)
    p_run.add_argument("--commit", action="store_true")
    p_run.add_argument("--include-assigned", action="store_true")
    p_run.add_argument("--include-large", action="store_true")
    p_run.add_argument("--include-human-gate", action="store_true")
    p_run.add_argument("--no-draft-pr", action="store_true")
    p_run.add_argument("--sync-kanban", action="store_true")
    p_run.add_argument(
        "--execute",
        action="store_true",
        help="After printing the prompt, run `hermes chat -q --oneshot` (contributor BYOK)",
    )
    p_run.add_argument(
        "--require-tests",
        action="store_true",
        help="After --execute, run scripts/run_tests.sh (or pytest tests/)",
    )
    p_run.add_argument(
        "--open-pr",
        action="store_true",
        help="After --commit, run `gh pr create --draft` (never implied by dry-run)",
    )

    parser.set_defaults(_autodevelop_parser=parser)
    return parser


def autodevelop_command(args: argparse.Namespace) -> int:
    action = getattr(args, "autodevelop_action", None)
    if not action:
        parser = getattr(args, "_autodevelop_parser", None)
        if parser is not None:
            parser.print_help()
            return 0
        print("usage: hermes autodevelop <list|prompt|claim|park|sync-kanban|run>", file=sys.stderr)
        return 2
    handlers = {
        "list": _cmd_list,
        "prompt": _cmd_prompt,
        "claim": _cmd_claim,
        "park": _cmd_park,
        "sync-kanban": _cmd_sync,
        "run": _cmd_run,
    }
    try:
        return handlers[action](args)
    except (GitHubError, ValueError) as exc:
        print(f"autodevelop: {exc}", file=sys.stderr)
        return 1


def _client() -> GitHubClient:
    token = resolve_github_token()
    if not token:
        raise GitHubError("GITHUB_TOKEN or GH_TOKEN is required for live GitHub access")
    return GitHubClient(token)


def _cmd_list(args: argparse.Namespace) -> int:
    items = resolve_queue(
        _client(),
        args.repo,
        label=args.label,
        limit=args.limit,
        include_assigned=args.include_assigned,
        include_large=args.include_large,
        include_human_gate=args.include_human_gate,
    )
    claimable = [i for i in items if i.claimable]
    skipped = [i for i in items if not i.claimable]
    if args.json:
        payload = {
            "repo": args.repo,
            "label": args.label,
            "claimable": [_item_json(i) for i in claimable],
            "skipped": [_item_json(i) for i in skipped],
        }
        print(json.dumps(payload, indent=2))
        return 0
    if not items:
        print(f"No open issues with label {args.label!r} on {args.repo}.")
        return 0
    print(f"Queue {args.repo} label={args.label} claimable={len(claimable)} skipped={len(skipped)}")
    for item in claimable:
        print(f"  #{item.number}\t{item.title}\t{item.html_url}")
    for item in skipped:
        print(f"  skip #{item.number}\t{item.skip_reason}\t{item.title}")
    return 0


def _item_json(item: QueueItem) -> dict[str, Any]:
    data = asdict(item)
    data["contract"] = asdict(item.contract)
    return data


def _cmd_prompt(args: argparse.Namespace) -> int:
    client = _client()
    raw = client.get_issue(args.repo, args.number)
    item = issue_from_api(
        args.repo,
        raw,
        include_assigned=True,
        include_large=True,
        include_human_gate=True,
    )
    prior = client.search_prior_art(args.repo, item.title[:80])
    print(build_oneshot_prompt(item, draft_pr=not args.no_draft_pr, prior_art=prior))
    return 0


def _cmd_claim(args: argparse.Namespace) -> int:
    client = _client()
    raw = client.get_issue(args.repo, args.number)
    item = issue_from_api(
        args.repo,
        raw,
        include_assigned=True,
        include_large=True,
        include_human_gate=True,
    )
    login = (args.login or os.environ.get("GITHUB_ACTOR") or os.environ.get("USER") or "contributor").strip()
    result = claim_item(client, item, login=login, ttl_hours=args.ttl_hours, commit=args.commit)
    print(json.dumps(result, indent=2))
    return 0 if result.get("claimed") or result.get("reason") in ("dry-run", "fresh-claim") else 1


def _cmd_park(args: argparse.Namespace) -> int:
    client = _client()
    body = park_comment_body(reason=args.reason)
    if not args.commit:
        print(json.dumps({"parked": False, "reason": "dry-run", "comment": body}, indent=2))
        return 0
    posted = client.post_comment(args.repo, args.number, body)
    print(json.dumps({"parked": True, "comment_id": posted.get("id")}, indent=2))
    return 0


def _cmd_sync(args: argparse.Namespace) -> int:
    items = [
        i
        for i in resolve_queue(
            _client(),
            args.repo,
            label=args.label,
            limit=args.limit,
            include_assigned=args.include_assigned,
            include_large=args.include_large,
            include_human_gate=args.include_human_gate,
        )
        if i.claimable
    ]
    created = []
    for item in items:
        created.append({"number": item.number, "task_id": sync_item_to_kanban(item)})
    print(json.dumps({"mirrored": created}, indent=2))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    boot = bootstrap_report(os.getcwd())
    if not boot["ok"]:
        print(f"autodevelop: bootstrap incomplete: {', '.join(boot['missing'])}", file=sys.stderr)
        return 2
    budget = args.budget if args.budget > 0 else args.max_issues
    budget = min(budget, max(args.max_issues, 1))
    login = (args.login or os.environ.get("GITHUB_ACTOR") or os.environ.get("USER") or "contributor").strip()
    already = open_claim_count(args.repo, login)
    if already >= DEFAULT_MAX_CLAIMS_PER_USER:
        print(
            f"autodevelop: rate limit — {login} already has {already} open claims on {args.repo}",
            file=sys.stderr,
        )
        return 3
    budget = min(budget, max(0, DEFAULT_MAX_CLAIMS_PER_USER - already))
    items = [
        i
        for i in resolve_queue(
            _client(),
            args.repo,
            label=args.label,
            limit=max(args.max_issues, 1) * 3,
            include_assigned=args.include_assigned,
            include_large=args.include_large,
            include_human_gate=args.include_human_gate,
            ttl_hours=args.ttl_hours,
        )
        if i.claimable
    ]
    taken = 0
    client = _client()
    for item in items:
        if taken >= budget:
            break
        result = claim_item(
            client,
            item,
            login=login,
            ttl_hours=args.ttl_hours,
            commit=args.commit,
        )
        if result.get("reason") == "fresh-claim":
            continue
        taken += 1
        if args.sync_kanban:
            result["kanban_task_id"] = sync_item_to_kanban(item)
        prior = client.search_prior_art(item.repo, item.title[:80])
        prompt = build_oneshot_prompt(item, draft_pr=not args.no_draft_pr, prior_art=prior)
        print(f"## {item.repo}#{item.number} {item.title}")
        print(json.dumps(result, indent=2))
        print()
        print(prompt)
        print()
        if args.execute:
            import subprocess

            completed = subprocess.run(execute_argv(prompt), check=False)
            if completed.returncode != 0:
                return completed.returncode
            if args.require_tests:
                tests = subprocess.run(test_runner_argv(os.getcwd()), check=False)
                if tests.returncode != 0:
                    return tests.returncode
        if args.open_pr:
            if not args.commit:
                print("autodevelop: --open-pr requires --commit", file=sys.stderr)
                return 2
            import subprocess

            opened = subprocess.run(draft_pr_argv(item, draft=not args.no_draft_pr), check=False)
            if opened.returncode != 0:
                return opened.returncode
    if taken == 0:
        print("autodevelop: queue empty or every item skipped.")
    return 0

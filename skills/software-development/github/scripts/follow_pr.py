#!/usr/bin/env python3
"""Block until a pull request has something new for the agent, print it, exit.

Run it as a background terminal job with ``notify=true``: the job's exit wakes
the session that started it (CLI, TUI, Desktop or a messaging chat), and its
output is the event digest. Re-run the same command after handling the batch;
the state file remembers what was already reported, so nothing replays.

    python follow_pr.py --pr 123 [--repo owner/name] [--interval 60] [--hours 24]
    python follow_pr.py --gitlab-mr https://gitlab.example.com/group/project/-/merge_requests/12

GitLab (gitlab.com or self-managed) is read over its REST API: ``GITLAB_TOKEN``
(a personal access token with ``read_api``; sent only to ``GITLAB_HOST``,
default gitlab.com) is needed for comments and reviews
(GitLab hides MR notes from anonymous callers) and for private projects;
without it a public MR still reports its CI verdict and merge/close. MR notes map to
comments, diff notes to inline comments, approve/unapprove/request-changes
system notes to reviews, and the MR's head pipeline to the CI verdict.

Events: new issue comments, submitted reviews, inline review comments, the CI
verdict for the PR's current head commit (once per head), and merge/close
(which ends the follow). Comments by the authenticated ``gh`` user are skipped
unless ``--include-self``, so the agent's own replies never wake it.

Exit codes: 0 events printed (or PR merged/closed) · 3 window elapsed with no
events · 2 repeated ``gh``/GitLab API failures (stderr names the error).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PENDING_CHECK_STATES = {"QUEUED", "IN_PROGRESS", "PENDING", "WAITING", "REQUESTED", "EXPECTED"}
FAILED_CONCLUSIONS = {"FAILURE", "TIMED_OUT", "CANCELLED", "STARTUP_FAILURE", "ERROR"}
# Settled but never ran: waiting on a person (GitHub fork-run approval, GitLab manual pipeline).
BLOCKED_CONCLUSIONS = {"ACTION_REQUIRED"}
BODY_LIMIT = 1500
MAX_GH_FAILURES = 5


def gh_json(args: list[str]):
    out = subprocess.run(["gh", *args], check=False, capture_output=True, text=True, encoding="utf-8",
                         timeout=120, stdin=subprocess.DEVNULL)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or f"gh {' '.join(args)} exited {out.returncode}")
    return _json_pages(out.stdout)


def _json_pages(text: str):
    """One JSON value, or the pages of ``gh api --paginate`` joined into one list.

    Older ``gh`` releases print each page as its own array (``[...][...]``) and
    ``--slurp`` does not exist before 2.48, so concatenated values are decoded here.
    """
    decoder, pos, values = json.JSONDecoder(), 0, []
    text = text.strip()
    while pos < len(text):
        value, pos = decoder.raw_decode(text, pos)
        values.append(value)
        while pos < len(text) and text[pos].isspace():
            pos += 1
    if len(values) <= 1:
        return values[0] if values else None
    return [item for page in values for item in (page if isinstance(page, list) else [page])]


def fetch_snapshot(repo: str, pr: int) -> dict:
    view = gh_json(["pr", "view", str(pr), "--repo", repo, "--json",
                    "state,url,headRefOid,comments,reviews,statusCheckRollup"])
    view["inline"] = gh_json(["api", "--paginate", f"repos/{repo}/pulls/{pr}/comments?per_page=100"])
    return view


GITLAB_RUNNING = {"created", "waiting_for_resource", "preparing", "pending", "running", "scheduled"}
GITLAB_CONCLUSION = {"success": "SUCCESS", "failed": "FAILURE", "canceled": "CANCELLED",
                     "skipped": "SKIPPED", "manual": "ACTION_REQUIRED"}
GITLAB_REVIEW_NOTES = (("requested changes", "CHANGES_REQUESTED"), ("unapproved", "UNAPPROVED"),
                       ("approved", "APPROVED"))
GITLAB_STATE = {"opened": "OPEN", "locked": "OPEN", "merged": "MERGED", "closed": "CLOSED"}


def gitlab_token(host: str) -> str:
    """``GITLAB_TOKEN`` only for the host it belongs to (``GITLAB_HOST``, default gitlab.com).

    The MR URL is user input: a link on another host must never receive the credential.
    """
    token = os.environ.get("GITLAB_TOKEN", "")
    trusted = (os.environ.get("GITLAB_HOST") or "gitlab.com").strip().lower()
    trusted = re.sub(r"^https?://", "", trusted).rstrip("/")
    return token if token and host.lower() == trusted else ""


def gitlab_get(host: str, path: str, *, paginate: bool = False):
    """GET ``https://<host>/api/v4/<path>``, authenticated only when ``gitlab_token`` allows it."""
    headers = {"Accept": "application/json"}
    if gitlab_token(host):
        headers["PRIVATE-TOKEN"] = gitlab_token(host)
    sep = "&" if "?" in path else "?"
    items, page = [], "1"
    while page:
        url = f"https://{host}/api/v4/{path}" + (f"{sep}per_page=100&page={page}" if paginate else "")
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8") or "null")
                page = resp.headers.get("X-Next-Page", "") if paginate else ""
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"GitLab GET {path}: {exc}") from exc
        if not paginate:
            return data
        items.extend(data or [])
    return items


def gitlab_snapshot(host: str, project: str, mr: int) -> dict:
    base = f"projects/{urllib.parse.quote(project, safe='')}"
    view = gitlab_get(host, f"{base}/merge_requests/{mr}")
    # GitLab serves MR notes only to signed-in callers, public projects included;
    # without GITLAB_TOKEN the follow still reports CI and merge/close.
    notes = gitlab_get(host, f"{base}/merge_requests/{mr}/notes?sort=asc", paginate=True) \
        if gitlab_token(host) else []
    return normalize_gitlab(view, notes)


def normalize_gitlab(view: dict, notes: list[dict]) -> dict:
    """GitLab MR + notes -> the GitHub snapshot shape ``collect_events`` reads.

    CI is the MR's ``head_pipeline`` only: older pipelines on the same SHA (a failed
    run that was retried) are history, not the current verdict.
    """
    sha = view.get("sha") or ""
    web = view.get("web_url", "")
    snap = {"state": GITLAB_STATE.get(view.get("state", ""), "OPEN"), "url": web, "headRefOid": sha,
            "comments": [], "reviews": [], "inline": [], "statusCheckRollup": []}
    for n in notes:
        author = {"login": (n.get("author") or {}).get("username", "")}
        body = n.get("body") or ""
        if n.get("system"):
            state = next((s for phrase, s in GITLAB_REVIEW_NOTES if body.lower().startswith(phrase)), "")
            if state:
                snap["reviews"].append({"id": f"gl{n['id']}", "author": author, "state": state, "body": ""})
            continue
        if n.get("type") == "DiffNote":
            pos = n.get("position") or {}
            snap["inline"].append({"id": f"gl{n['id']}", "user": author, "body": body,
                                   "path": pos.get("new_path") or pos.get("old_path") or "",
                                   "line": pos.get("new_line") or pos.get("old_line"),
                                   "html_url": f"{web}#note_{n['id']}"})
        else:
            snap["comments"].append({"id": f"gl{n['id']}", "author": author, "body": body,
                                     "url": f"{web}#note_{n['id']}"})
    head = view.get("head_pipeline") or {}
    if head.get("sha") not in (None, sha):
        head = {}  # a pipeline for an older head commit is not this head's verdict
    for p in [head] if head else []:
        status = p.get("status", "")
        snap["statusCheckRollup"].append({
            "__typename": "CheckRun", "name": f"pipeline {p.get('id')} ({p.get('source', '')})",
            "status": "IN_PROGRESS" if status in GITLAB_RUNNING else "COMPLETED",
            "conclusion": GITLAB_CONCLUSION.get(status, status.upper())})
    return snap


def _trim(text: str) -> str:
    text = (text or "").strip()
    return text if len(text) <= BODY_LIMIT else text[:BODY_LIMIT] + "\n...(truncated)"


def ci_verdict(rollup: list[dict]) -> tuple[str, list[str]] | None:
    """Return (passed|failed|blocked, names) once every check on the head settled, else None."""
    if not rollup:
        return None
    failed, blocked = [], []
    for check in rollup:
        if check.get("__typename") == "StatusContext" or "context" in check:
            state = (check.get("state") or "").upper()
            if state in PENDING_CHECK_STATES:
                return None
            if state in FAILED_CONCLUSIONS:
                failed.append(check.get("context") or "status")
            continue
        if (check.get("status") or "").upper() != "COMPLETED":
            return None
        conclusion = (check.get("conclusion") or "").upper()
        if conclusion in FAILED_CONCLUSIONS:
            failed.append(check.get("name") or "check")
        elif conclusion in BLOCKED_CONCLUSIONS:
            blocked.append(check.get("name") or "check")
    if failed:
        return "failed", failed
    return ("blocked", blocked) if blocked else ("passed", [])


def collect_events(snap: dict, seen: set[str], self_login: str) -> list[tuple[str, str]]:
    """New (key, markdown) events in snapshot order; ``seen`` is not mutated."""
    events: list[tuple[str, str]] = []

    def author_of(item: dict) -> str:
        return ((item.get("author") or item.get("user") or {}).get("login") or "")

    for c in snap.get("comments") or []:
        key = f"comment:{c.get('id') or c.get('url')}"
        if key not in seen and author_of(c) != self_login:
            events.append((key, f"## [comment] @{author_of(c)}\n{c.get('url', '')}\n\n{_trim(c.get('body'))}"))
    for r in snap.get("reviews") or []:
        key = f"review:{r.get('id')}"
        if key in seen or author_of(r) == self_login or (r.get("state") or "") == "PENDING":
            continue
        events.append((key, f"## [review] @{author_of(r)} {r.get('state', '')}\n\n{_trim(r.get('body'))}".rstrip()))
    for c in snap.get("inline") or []:
        key = f"inline:{c.get('id')}"
        if key not in seen and author_of(c) != self_login:
            where = f"{c.get('path', '')}:{c.get('line') or c.get('original_line') or ''}"
            events.append((key, f"## [inline] @{author_of(c)} {where}\n{c.get('html_url', '')}\n\n{_trim(c.get('body'))}"))
    sha = snap.get("headRefOid") or ""
    verdict = ci_verdict(snap.get("statusCheckRollup") or [])
    if sha and verdict and f"ci:{sha}" not in seen:
        status, names = verdict
        label = "Waiting on a person to start/approve" if status == "blocked" else "Failed"
        detail = (f"\n{label}: " + ", ".join(sorted(set(names)))) if names else ""
        events.append((f"ci:{sha}", f"## [ci] {status} on {sha[:10]}{detail}"))
    state = (snap.get("state") or "").upper()
    if state in {"MERGED", "CLOSED"}:
        events.append((f"state:{state}", f"## [state] PR {state.lower()} — follow ended\n{snap.get('url', '')}"))
    return events


def baseline_keys(snap: dict) -> set[str]:
    """Comments and reviews that already exist when following starts are history, not news.
    CI is not baselined: a verdict that already landed is exactly what the follower reports."""
    keys = {f"comment:{c.get('id') or c.get('url')}" for c in snap.get("comments") or []}
    keys |= {f"review:{r.get('id')}" for r in snap.get("reviews") or []}
    keys |= {f"inline:{c.get('id')}" for c in snap.get("inline") or []}
    return keys


def state_path(repo: str, pr: int, override: str) -> Path:
    if override:
        return Path(override)
    home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", repo)
    return home / "watcher-state" / f"pr-{slug}-{pr}.json"


def load_seen(path: Path) -> set[str] | None:
    try:
        return set(json.loads(path.read_text(encoding="utf-8-sig")).get("seen", []))
    except FileNotFoundError:
        return None


def save_seen(path: Path, seen: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"seen": sorted(seen)}), encoding="utf-8")
    tmp.replace(path)


def parse_gitlab_mr(url: str) -> tuple[str, str, int]:
    """``https://host/group/sub/project/-/merge_requests/12`` -> (host, "group/sub/project", 12)."""
    m = re.match(r"https?://([^/]+)/(.+?)/-/merge_requests/(\d+)", url.strip())
    if not m:
        raise SystemExit(f"follow_pr: not a GitLab merge request URL: {url}")
    return m.group(1), m.group(2), int(m.group(3))


def main() -> int:
    p = argparse.ArgumentParser(description="Wait for new PR/MR comments, reviews, CI verdicts.")
    p.add_argument("--pr", type=int, default=0, help="GitHub pull request number")
    p.add_argument("--repo", default="", help="owner/name (default: the current directory's repo)")
    p.add_argument("--gitlab-mr", default="", help="GitLab merge request URL (gitlab.com or self-managed) instead of --pr")
    p.add_argument("--interval", type=int, default=60, help="seconds between polls (min 20)")
    p.add_argument("--settle", type=int, default=30,
                   help="after the first new event, wait this long so a review and its inline comments arrive as one batch")
    p.add_argument("--hours", type=float, default=24.0, help="give up after this long with no events (exit 3)")
    p.add_argument("--include-self", action="store_true", help="also report comments by the authenticated user")
    p.add_argument("--state-file", default="", help="override $HERMES_HOME/watcher-state/pr-<repo>-<n>.json")
    args = p.parse_args()
    if bool(args.pr) == bool(args.gitlab_mr):
        p.error("pass exactly one of --pr or --gitlab-mr")

    for attempt in range(MAX_GH_FAILURES):
        try:
            return follow(args)
        except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            if attempt == MAX_GH_FAILURES - 1:
                print(f"follow_pr: startup failed {MAX_GH_FAILURES} times, last: {exc}", file=sys.stderr)
                return 2
            time.sleep(max(args.interval, 20))
    return 2


def follow(args: argparse.Namespace) -> int:
    if args.gitlab_mr:
        host, project, number = parse_gitlab_mr(args.gitlab_mr)
        repo = f"{host}/{project}"
        self_login = "" if args.include_self or not gitlab_token(host) else \
            gitlab_get(host, "user").get("username", "")
        snapshot = lambda: gitlab_snapshot(host, project, number)
    else:
        number = args.pr
        repo = args.repo or gh_json(["repo", "view", "--json", "nameWithOwner"])["nameWithOwner"]
        self_login = "" if args.include_self else gh_json(["api", "user"])["login"]
        snapshot = lambda: fetch_snapshot(repo, number)
    path = state_path(repo, number, args.state_file)
    interval = max(args.interval, 20)
    deadline = time.time() + args.hours * 3600

    seen = load_seen(path)
    failures = 0
    pending_since = None
    while True:
        try:
            snap = snapshot()
            failures = 0
        except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            failures += 1
            if failures >= MAX_GH_FAILURES:
                print(f"follow_pr: {failures} consecutive fetch failures, last: {exc}", file=sys.stderr)
                return 2
            time.sleep(interval)
            continue
        if seen is None:
            seen = baseline_keys(snap)
            save_seen(path, seen)
        events = collect_events(snap, seen, self_login)
        terminal = any(k.startswith("state:") for k, _ in events)
        if events and (terminal or (pending_since and time.time() - pending_since >= args.settle)):
            print(f"# {repo}#{number}: {len(events)} new event(s)\n")
            print("\n\n".join(md for _, md in events))
            save_seen(path, seen | {k for k, _ in events})
            return 0
        if events and pending_since is None:
            pending_since = time.time()
            time.sleep(min(args.settle, interval))
            continue
        if time.time() >= deadline:
            print(f"# {repo}#{number}: no new events in {args.hours:g}h")
            return 3
        time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main())

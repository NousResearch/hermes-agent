#!/usr/bin/env python3
"""Block until a pull request has something new for the agent, print it, exit.

Run it as a background terminal job with ``notify=true``: the job's exit wakes
the session that started it (CLI, TUI, Desktop or a messaging chat), and its
output is the event digest. Re-run the same command after handling the batch;
the state file remembers what was already reported, so nothing replays.

    python follow_pr.py --pr 123 [--repo owner/name] [--interval 60] [--hours 24]

Events: new issue comments, submitted reviews, inline review comments, the CI
verdict for the PR's current head commit (once per head), and merge/close
(which ends the follow). Comments by the authenticated ``gh`` user are skipped
unless ``--include-self``, so the agent's own replies never wake it.

Exit codes: 0 events printed (or PR merged/closed) · 3 window elapsed with no
events · 2 repeated ``gh`` failures (stderr names the error).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PENDING_CHECK_STATES = {"QUEUED", "IN_PROGRESS", "PENDING", "WAITING", "REQUESTED", "EXPECTED"}
FAILED_CONCLUSIONS = {"FAILURE", "TIMED_OUT", "CANCELLED", "ACTION_REQUIRED", "STARTUP_FAILURE", "ERROR"}
BODY_LIMIT = 1500
MAX_GH_FAILURES = 5


def gh_json(args: list[str]):
    out = subprocess.run(["gh", *args], check=False, capture_output=True, text=True, encoding="utf-8",
                         timeout=120, stdin=subprocess.DEVNULL)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or f"gh {' '.join(args)} exited {out.returncode}")
    return json.loads(out.stdout or "null")


def fetch_snapshot(repo: str, pr: int) -> dict:
    view = gh_json(["pr", "view", str(pr), "--repo", repo, "--json",
                    "state,url,headRefOid,comments,reviews,statusCheckRollup"])
    view["inline"] = gh_json(["api", "--paginate", f"repos/{repo}/pulls/{pr}/comments?per_page=100"])
    return view


def _trim(text: str) -> str:
    text = (text or "").strip()
    return text if len(text) <= BODY_LIMIT else text[:BODY_LIMIT] + "\n...(truncated)"


def ci_verdict(rollup: list[dict]) -> tuple[str, list[str]] | None:
    """Return (verdict, failed names) once every check on the head finished, else None."""
    if not rollup:
        return None
    failed = []
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
        if (check.get("conclusion") or "").upper() in FAILED_CONCLUSIONS:
            failed.append(check.get("name") or "check")
    return ("failed" if failed else "passed"), failed


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
        status, failed = verdict
        detail = ("\nFailed: " + ", ".join(sorted(set(failed)))) if failed else ""
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


def main() -> int:
    p = argparse.ArgumentParser(description="Wait for new PR comments, reviews, CI verdicts.")
    p.add_argument("--pr", type=int, required=True)
    p.add_argument("--repo", default="", help="owner/name (default: the current directory's repo)")
    p.add_argument("--interval", type=int, default=60, help="seconds between polls (min 20)")
    p.add_argument("--settle", type=int, default=30,
                   help="after the first new event, wait this long so a review and its inline comments arrive as one batch")
    p.add_argument("--hours", type=float, default=24.0, help="give up after this long with no events (exit 3)")
    p.add_argument("--include-self", action="store_true", help="also report comments by the authenticated gh user")
    p.add_argument("--state-file", default="", help="override $HERMES_HOME/watcher-state/pr-<repo>-<n>.json")
    args = p.parse_args()

    repo = args.repo or gh_json(["repo", "view", "--json", "nameWithOwner"])["nameWithOwner"]
    self_login = "" if args.include_self else gh_json(["api", "user"])["login"]
    path = state_path(repo, args.pr, args.state_file)
    interval = max(args.interval, 20)
    deadline = time.time() + args.hours * 3600

    seen = load_seen(path)
    failures = 0
    pending_since = None
    while True:
        try:
            snap = fetch_snapshot(repo, args.pr)
            failures = 0
        except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            failures += 1
            if failures >= MAX_GH_FAILURES:
                print(f"follow_pr: {failures} consecutive gh failures, last: {exc}", file=sys.stderr)
                return 2
            time.sleep(interval)
            continue
        if seen is None:
            seen = baseline_keys(snap)
            save_seen(path, seen)
        events = collect_events(snap, seen, self_login)
        terminal = any(k.startswith("state:") for k, _ in events)
        if events and (terminal or (pending_since and time.time() - pending_since >= args.settle)):
            print(f"# {repo}#{args.pr}: {len(events)} new event(s)\n")
            print("\n\n".join(md for _, md in events))
            save_seen(path, seen | {k for k, _ in events})
            return 0
        if events and pending_since is None:
            pending_since = time.time()
            time.sleep(min(args.settle, interval))
            continue
        if time.time() >= deadline:
            print(f"# {repo}#{args.pr}: no new events in {args.hours:g}h")
            return 3
        time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Find plugin-catalog entries whose repository is gone or whose pinned commit no longer exists.

An entry like that can no longer be installed: ``hermes plugins install`` clones the repo and checks
out the pin. The weekly ``catalog-dead-repos.yml`` workflow delists them (deletes the entry file,
per plugin-catalog/README.md; never ``removed.yaml``, which is the security blocklist) in one PR.

Only a definitive answer counts. GitHub repos are looked up in GraphQL batches, and a miss is
confirmed over REST, which follows rename and transfer redirects, so a moved repo is never listed.
Other hosts are probed with ``git ls-remote`` and only a clean "not found" counts; their pins are
not checked. Any lookup that does not answer cleanly (timeout, 5xx, rate limit) aborts the run with
nothing delisted, and so does a run that finds more than ``--max-dead`` entries: that shape is an
outage or a token problem, not one week of deleted repos.

    catalog_dead_repos.py [--catalog plugin-catalog] [--max-dead 25] [--delete] [--report out.md]

Reads ``GH_TOKEN``. ``--delete`` removes the dead entry files; ``--report`` writes the PR body.
Exit 0 = done (possibly nothing dead), 1 = refused, 2 = usage.
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
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ruamel.yaml import YAML

API = "https://api.github.com"
BATCH = 50
ATTEMPTS = 3
_GITHUB = re.compile(r"^https://github\.com/([A-Za-z0-9-]+)/([A-Za-z0-9._-]+?)(?:\.git)?/?$")
_GONE_WORDS = ("not found", "does not exist")

# (method, url, json body or None) -> (HTTP status, parsed json). 404/422 are answers, not errors.
Fetch = Callable[[str, str, "dict | None"], "tuple[int, dict]"]


@dataclass(frozen=True)
class Entry:
    path: Path
    repo: str
    sha: str


@dataclass(frozen=True)
class Dead:
    entry: Entry
    reason: str


REPO_GONE = "repository deleted"
PIN_GONE = "pinned commit no longer exists"


class LookupFailed(Exception):
    """No definitive answer; the run must not delist anything."""


def load_entries(catalog: Path) -> list[Entry]:
    yaml = YAML(typ="safe", pure=True)
    out = []
    for path in sorted(catalog.glob("*.yaml")):
        data = yaml.load(path.read_text(encoding="utf-8-sig"))
        if path.name != "removed.yaml" and isinstance(data, dict) and data.get("repo") and data.get("sha"):
            out.append(Entry(path, str(data["repo"]).strip(), str(data["sha"]).strip()))
    return out


def github_fetch(token: str) -> Fetch:
    def fetch(method: str, url: str, body: dict | None) -> tuple[int, dict]:
        req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body else None,
                                     headers={"Authorization": f"Bearer {token}",
                                              "Accept": "application/vnd.github+json",
                                              "User-Agent": "hermes-catalog-dead-repos"})
        last: Exception | None = None
        for attempt in range(ATTEMPTS):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return resp.status, json.load(resp)
            except urllib.error.HTTPError as e:
                if e.code in (404, 422):
                    return e.code, {}
                last = e
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last = e
            time.sleep(5 * (attempt + 1))
        raise LookupFailed(f"{url}: {last}")
    return fetch


def _confirm_missing(entry: Entry, owner: str, name: str, fetch: Fetch) -> Dead | None:
    """GraphQL has no repo under this name. REST follows renames; only its 404 means deleted."""
    if fetch("GET", f"{API}/repos/{owner}/{name}", None)[0] == 404:
        return Dead(entry, REPO_GONE)
    status = fetch("GET", f"{API}/repos/{owner}/{name}/commits/{entry.sha}", None)[0]
    return Dead(entry, PIN_GONE) if status in (404, 422) else None


def check_github(batch: list[tuple[Entry, str, str]], fetch: Fetch) -> list[Dead]:
    fields = " ".join(f"r{i}: repository(owner: {json.dumps(o)}, name: {json.dumps(n)}) "
                      f"{{ object(oid: {json.dumps(e.sha)}) {{ oid }} }}" for i, (e, o, n) in enumerate(batch))
    _, resp = fetch("POST", f"{API}/graphql", {"query": f"query {{ {fields} }}"})
    errors = resp.get("errors") or []
    if "data" not in resp or any(err.get("type") != "NOT_FOUND" for err in errors):
        raise LookupFailed(f"graphql: {errors[:1] or resp}")
    data = resp["data"] or {}
    dead = []
    for i, (entry, owner, name) in enumerate(batch):
        node = data.get(f"r{i}")
        if node is None:
            hit = _confirm_missing(entry, owner, name, fetch)
        else:
            hit = Dead(entry, PIN_GONE) if node.get("object") is None else None
        if hit:
            dead.append(hit)
    return dead


def check_other_host(entry: Entry) -> Dead | None:
    p = subprocess.run(["git", "ls-remote", "--", entry.repo, "HEAD"], text=True, encoding="utf-8",
                       errors="replace", capture_output=True, check=False, timeout=120, stdin=subprocess.DEVNULL,
                       env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "true"})
    if p.returncode == 0:
        return None
    if any(w in p.stderr.lower() for w in _GONE_WORDS):
        return Dead(entry, REPO_GONE)
    raise LookupFailed(f"{entry.repo}: {p.stderr.strip()[-200:]}")


def find_dead(entries: list[Entry], fetch: Fetch,
              other_host: Callable[[Entry], Dead | None] = check_other_host) -> list[Dead]:
    github = [(e, m.group(1), m.group(2)) for e in entries if (m := _GITHUB.match(e.repo))]
    dead = []
    for i in range(0, len(github), BATCH):
        dead += check_github(github[i:i + BATCH], fetch)
    dead += [d for e in entries if not _GITHUB.match(e.repo) and (d := other_host(e))]
    return dead


def report(dead: list[Dead]) -> str:
    rows = "\n".join(f"| `{d.entry.path.stem}` | {d.entry.repo} | `{d.entry.sha[:12]}` | {d.reason} |"
                     for d in dead)
    return ("These catalog entries can no longer be installed: their repository was deleted, or the commit "
            "they pin was removed from it. Delisting deletes the entry file only. Anyone who already "
            "installed one keeps it, and the author can resubmit with a live repository and pin.\n\n"
            "| Entry | Repository | Pin | Why |\n|---|---|---|---|\n" + rows + "\n\n"
            "Opened by `.github/workflows/catalog-dead-repos.yml`, which re-checks the whole catalog weekly.\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Delist plugin-catalog entries whose repo or pin is gone.")
    ap.add_argument("--catalog", type=Path, default=Path("plugin-catalog"))
    ap.add_argument("--max-dead", type=int, default=25)
    ap.add_argument("--delete", action="store_true")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args(argv)
    token = os.environ.get("GH_TOKEN", "")
    if not token:
        print("::error::GH_TOKEN is required", file=sys.stderr)
        return 2
    entries = load_entries(args.catalog)
    try:
        dead = find_dead(entries, github_fetch(token))
    except LookupFailed as e:
        print(f"::error::a lookup did not answer, delisting nothing: {e}", file=sys.stderr)
        return 1
    if len(dead) > args.max_dead:
        print(f"::error::{len(dead)} entries look dead (cap {args.max_dead}); refusing, likely an outage",
              file=sys.stderr)
        return 1
    for d in dead:
        print(f"{d.entry.path.stem}: {d.reason} ({d.entry.repo})")
        if args.delete:
            d.entry.path.unlink()
    if args.report and dead:
        args.report.write_text(report(dead), encoding="utf-8")
    print(f"{len(dead)} of {len(entries)} entries are dead")
    return 0


if __name__ == "__main__":
    sys.exit(main())

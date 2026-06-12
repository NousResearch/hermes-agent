#!/usr/bin/env python3
"""Hermes maintainer sweep: durable, proposal-first GitHub backlog reports.

This script intentionally performs no GitHub mutations. It borrows the useful
ClawSweeper/Clownfish pattern for Hermes development work: hydrate live or
fixture-backed issue/PR state, write one durable report per item, append an
audit ledger event, and produce a compact dashboard that a human or a separate
applicator can promote later.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Any

ACTION_STATE = "proposal"
RECOMMENDATION = "needs_human"
SCHEMA_VERSION = 1


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slug_repo(repo: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", repo.strip()).strip("_") or "unknown_repo"


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def snapshot_hash(item: dict[str, Any]) -> str:
    """Hash the hydrated item snapshot, excluding volatile local run metadata."""
    return hashlib.sha256(_stable_json(item).encode("utf-8")).hexdigest()[:16]


def _frontmatter(data: dict[str, Any]) -> str:
    lines = ["---"]
    for key, value in data.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif value is None:
            rendered = "null"
        else:
            rendered = json.dumps(value, ensure_ascii=False)
        lines.append(f"{key}: {rendered}")
    lines.append("---")
    return "\n".join(lines)


def _labels(item: dict[str, Any]) -> list[str]:
    labels = item.get("labels") or []
    out: list[str] = []
    for label in labels:
        if isinstance(label, str):
            out.append(label)
        elif isinstance(label, dict):
            name = label.get("name")
            if name:
                out.append(str(name))
    return out


def _author_login(item: dict[str, Any]) -> str:
    author = item.get("author") or {}
    if isinstance(author, dict):
        return str(author.get("login") or "")
    return str(author or "")


def item_kind(item: dict[str, Any]) -> str:
    explicit = item.get("kind") or item.get("type")
    if explicit:
        return str(explicit).lower()
    if "isDraft" in item or "headRefName" in item or "mergeable" in item:
        return "pr"
    return "issue"


def normalize_item(raw: dict[str, Any], default_kind: str | None = None) -> dict[str, Any]:
    item = dict(raw)
    item["kind"] = default_kind or item_kind(item)
    item["number"] = int(item.get("number") or 0)
    item["title"] = str(item.get("title") or "")
    item["url"] = str(item.get("url") or "")
    item["state"] = str(item.get("state") or "unknown").lower()
    item["labels"] = _labels(item)
    item["author_login"] = _author_login(item)
    return item


def load_items_from_file(path: pathlib.Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [normalize_item(item) for item in payload]
    if not isinstance(payload, dict):
        raise ValueError("source JSON must be a list or object with issues/prs/items")
    items: list[dict[str, Any]] = []
    for key, kind in (("issues", "issue"), ("prs", "pr"), ("pull_requests", "pr"), ("items", None)):
        values = payload.get(key) or []
        if not isinstance(values, list):
            raise ValueError(f"{key} must be a list")
        items.extend(normalize_item(item, kind) for item in values)
    return items


def _run_gh_json(args: list[str]) -> list[dict[str, Any]]:
    proc = subprocess.run(args, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"command failed: {' '.join(args)}")
    data = json.loads(proc.stdout or "[]")
    if not isinstance(data, list):
        raise ValueError(f"gh command returned non-list JSON: {' '.join(args)}")
    return data


def load_items_from_gh(repo: str, limit: int) -> list[dict[str, Any]]:
    common_issue_fields = "number,title,state,updatedAt,labels,author,url,body"
    common_pr_fields = (
        "number,title,state,updatedAt,labels,author,url,body,isDraft,headRefName,"
        "baseRefName,mergeable,reviewDecision,statusCheckRollup"
    )
    issues = _run_gh_json([
        "gh", "issue", "list", "--repo", repo, "--state", "open", "--limit", str(limit), "--json", common_issue_fields
    ])
    prs = _run_gh_json([
        "gh", "pr", "list", "--repo", repo, "--state", "open", "--limit", str(limit), "--json", common_pr_fields
    ])
    return [normalize_item(item, "issue") for item in issues] + [normalize_item(item, "pr") for item in prs]


def build_report(repo: str, item: dict[str, Any], run_id: str) -> tuple[str, dict[str, Any]]:
    h = snapshot_hash(item)
    labels = item.get("labels") or []
    gate = {
        "schema_version": SCHEMA_VERSION,
        "repo": repo,
        "kind": item["kind"],
        "number": item["number"],
        "url": item.get("url", ""),
        "snapshot_hash": h,
        "run_id": run_id,
        "action_state": ACTION_STATE,
        "recommendation": RECOMMENDATION,
        "mutation_allowed": False,
    }
    title = item.get("title") or f"{item['kind']} #{item['number']}"
    body = [
        _frontmatter(gate),
        "",
        f"# {item['kind'].upper()} #{item['number']}: {title}",
        "",
        "## Verdict",
        "",
        "- Recommendation: `needs_human`",
        "- Action state: `proposal`",
        "- Mutation allowed: `false`",
        "",
        "This report is read-only by construction. It is evidence for a future maintainer/applicator step, not permission to comment, close, push, or merge.",
        "",
        "## Snapshot",
        "",
        f"- URL: {item.get('url') or '(missing)'}",
        f"- State: `{item.get('state')}`",
        f"- Author: `{item.get('author_login') or '(unknown)'}`",
        f"- Labels: {', '.join(f'`{label}`' for label in labels) if labels else '(none)'}",
    ]
    if item.get("kind") == "pr":
        body.extend([
            f"- Base: `{item.get('baseRefName') or ''}`",
            f"- Head: `{item.get('headRefName') or ''}`",
            f"- Draft: `{item.get('isDraft')}`",
            f"- Mergeable: `{item.get('mergeable') or ''}`",
            f"- Review decision: `{item.get('reviewDecision') or ''}`",
        ])
    body.extend([
        f"- Snapshot hash: `{h}`",
        "",
        "## Required before apply",
        "",
        "1. Re-fetch live GitHub state immediately before any mutation.",
        "2. Verify this snapshot hash or explicitly record why re-review supersedes it.",
        "3. Keep LLM/worker output proposal-only; deterministic applicator owns writes.",
        "4. Require explicit allow-gates for comments, closing, branch pushes, or merges.",
        "",
        "## Raw snapshot",
        "",
        "```json",
        json.dumps(item, indent=2, sort_keys=True, ensure_ascii=False),
        "```",
        "",
    ])
    return "\n".join(body), gate


def write_outputs(repo: str, items: list[dict[str, Any]], state_dir: pathlib.Path) -> dict[str, Any]:
    run_id = _utc_now().replace(":", "").replace("-", "")
    repo_slug = _slug_repo(repo)
    base = state_dir / "repos" / repo_slug
    records = base / "items"
    records.mkdir(parents=True, exist_ok=True)
    ledger = base / "ledger.jsonl"
    gates: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda x: (x.get("kind", ""), int(x.get("number") or 0))):
        report, gate = build_report(repo, item, run_id)
        gates.append(gate)
        report_path = records / f"{item['kind']}-{item['number']}.md"
        report_path.write_text(report, encoding="utf-8")
        with ledger.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": "report_written", "at": _utc_now(), "path": str(report_path), **gate}, sort_keys=True) + "\n")
    dashboard = render_dashboard(repo, gates, run_id)
    dashboard_path = base / "dashboard.md"
    dashboard_path.write_text(dashboard, encoding="utf-8")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "repo": repo,
        "run_id": run_id,
        "state_dir": str(base),
        "dashboard": str(dashboard_path),
        "records": len(gates),
        "mutation_allowed": False,
        "action_state": ACTION_STATE,
    }
    (base / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def render_dashboard(repo: str, gates: list[dict[str, Any]], run_id: str) -> str:
    issue_count = sum(1 for gate in gates if gate["kind"] == "issue")
    pr_count = sum(1 for gate in gates if gate["kind"] == "pr")
    lines = [
        f"# Hermes Maintainer Sweep — {repo}",
        "",
        f"Run: `{run_id}`",
        "",
        "## Summary",
        "",
        f"- Records: {len(gates)}",
        f"- Issues: {issue_count}",
        f"- PRs: {pr_count}",
        "- Mutation allowed: `false`",
        "- Default recommendation: `needs_human`",
        "",
        "## Reports",
        "",
    ]
    for gate in gates:
        lines.append(
            f"- `{gate['kind']}-{gate['number']}` — `{gate['recommendation']}` — "
            f"snapshot `{gate['snapshot_hash']}` — [source]({gate['url']})"
        )
    lines.extend([
        "",
        "## Applicator contract",
        "",
        "Do not mutate GitHub from this dashboard. A separate deterministic applicator must re-fetch live state, verify maintainer authorization, branch/head/base/CI gates, and explicit allow flags before every write.",
        "",
    ])
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="GitHub repo in owner/name form")
    parser.add_argument("--state-dir", default=".hermes/maintainer", help="durable state output directory")
    parser.add_argument("--limit", type=int, default=50, help="max issues and max PRs to fetch with gh")
    parser.add_argument("--source-file", type=pathlib.Path, help="fixture JSON with issues/prs/items; skips gh")
    parser.add_argument("--json", action="store_true", help="print machine-readable summary")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if args.source_file:
        items = load_items_from_file(args.source_file)
    else:
        items = load_items_from_gh(args.repo, args.limit)
    summary = write_outputs(args.repo, items, pathlib.Path(args.state_dir))
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(f"Wrote {summary['records']} read-only reports to {summary['state_dir']}")
        print(f"Dashboard: {summary['dashboard']}")
        print("Mutation allowed: false")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

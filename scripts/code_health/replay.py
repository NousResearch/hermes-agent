"""Replay the ratchet over recently merged PRs: what would it have blocked?

    python -m scripts.code_health.replay --merged 2026-09-01..2026-09-30 --limit 300 --out <dir>
    python -m scripts.code_health.replay --manifest <dir>/manifest.json --out <dir2>

The sample is every PR merged into main inside a closed date window, in PR-number order, so the
same window always selects the same PRs (a new comment on an old PR cannot reshuffle it). Each run
writes ``<dir>/manifest.json`` (number, base, head per PR); ``--manifest`` replays exactly those
ranges, e.g. to compare two engine versions on identical input. Writes ``<dir>/replay.jsonl`` (one
row per PR) and prints per-rule totals plus how many measured PRs had at least one blocking finding;
a PR whose range cannot be resolved or measured is listed and fails the run. Use it
before promoting a rule to blocking or changing a target: a rule whose hits on merged PRs are
mostly legitimate code ships as a warning until its checker is fixed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from scripts.code_health import gitio
from scripts.code_health.compare import compare
from scripts.code_health.config import in_scope
from scripts.code_health.measure import Measurer
from scripts.code_health.report import apply_allows
from scripts.code_health.ruff_runner import resolve_ruff

_QUERY = """
query($q: String!, $after: String) {
  search(query: $q, type: ISSUE, first: 50, after: $after) {
    pageInfo { hasNextPage endCursor }
    nodes { ... on PullRequest {
      number title mergedAt
      mergeCommit { oid }
      commits(last: 100) { totalCount nodes { commit { messageHeadline } } }
    } }
  }
}"""


def merged_prs(repo: Path, window: str, limit: int) -> list[dict]:
    prs: list[dict] = []
    after = None
    query = f"repo:NousResearch/hermes-agent is:pr is:merged base:main merged:{window} sort:created-asc"
    while len(prs) < limit:
        args = ["gh", "api", "graphql", "-f", f"query={_QUERY}", "-f", f"q={query}"]
        if after:
            args += ["-f", f"after={after}"]
        out = subprocess.run(args, cwd=repo, capture_output=True, text=True, encoding="utf-8", check=True,
                             timeout=120, stdin=subprocess.DEVNULL).stdout
        data = json.loads(out)["data"]["search"]
        prs += [n for n in data["nodes"] if n.get("mergeCommit")]
        if not data["pageInfo"]["hasNextPage"]:
            break
        after = data["pageInfo"]["endCursor"]
    return sorted(prs, key=lambda pr: pr["number"])[:limit]


def build_manifest(repo: Path, window: str, limit: int) -> list[dict]:
    rows = []
    for pr in merged_prs(repo, window, limit):
        try:
            base, head = pr_range(repo, pr)
        except RuntimeError as exc:
            base, head = "", f"error: {exc}"
        rows.append({"number": pr["number"], "title": pr["title"], "base": base, "head": head})
    return rows


def pr_range(repo: Path, pr: dict) -> tuple[str, str]:
    """Base/head on main for a rebase-merged PR (its commits' subjects), else the merge parent."""
    head = pr["mergeCommit"]["oid"]
    subjects = {n["commit"]["messageHeadline"] for n in pr["commits"]["nodes"]}
    log = gitio.git(repo, "log", "--first-parent", "--format=%H%x00%s", "-n",
                    str(pr["commits"]["totalCount"] + 1), head).splitlines()
    k = 0
    for line in log:
        _, subject = line.split("\0", 1)
        if subject not in subjects:
            break
        k += 1
    return gitio.resolve_rev(repo, f"{head}~{max(k, 1)}"), head


def replay_one(repo: Path, measurer: Measurer, base: str, head: str) -> list[dict]:
    changes = gitio.changed_files(repo, base, head)
    head_paths = sorted({c.new for c in changes if c.new and in_scope(c.new)})
    base_paths = sorted({c.old for c in changes if c.old and in_scope(c.old)})
    if not head_paths:
        return []
    measurer.ctx.known_env = gitio.known_env_names(repo, base)
    base_m = measurer.measure(base, base_paths)
    head_m = measurer.measure(head, head_paths)
    findings = compare(base_m, head_m, changes)
    apply_allows(findings, head_m)
    return [asdict(f) for f in findings if f.allowed_reason is None]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="code_health.replay")
    sample = parser.add_mutually_exclusive_group(required=True)
    sample.add_argument("--merged", help="closed merge-date window, e.g. 2026-09-01..2026-09-30")
    sample.add_argument("--manifest", help="manifest.json from an earlier run: replay exactly those ranges")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    repo = gitio.repo_root(Path.cwd())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    else:
        manifest = build_manifest(repo, args.merged, args.limit)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    measurer = Measurer(repo, resolve_ruff(repo), known_env=set())
    per_rule: Counter[str] = Counter()
    prs_per_rule: Counter[str] = Counter()
    blocked = 0
    # A range that was never measured is no evidence either way: it stays out of the totals and
    # the denominator, and fails the run, so a "0 of N" verdict always means N measured PRs.
    unmeasured: list[int] = []
    with (out_dir / "replay.jsonl").open("w", encoding="utf-8") as fh:
        for row in manifest:
            try:
                if not row["base"]:
                    raise RuntimeError(row["head"] or "no base revision")
                findings = replay_one(repo, measurer, row["base"], row["head"])
            except RuntimeError as exc:
                unmeasured.append(row["number"])
                fh.write(json.dumps({**row, "error": str(exc)}) + "\n")
                continue
            fh.write(json.dumps({**row, "findings": findings}) + "\n")
            per_rule.update(f["rule"] for f in findings)
            prs_per_rule.update({f["rule"] for f in findings})
            blocked += any(f["blocking"] for f in findings)
    for rule, count in per_rule.most_common():
        print(f"{rule:<11} findings {count:>4}  PRs {prs_per_rule[rule]:>4}")
    print(f"{blocked} of {len(manifest) - len(unmeasured)} measured PRs had at least one blocking finding")
    if unmeasured:
        print(f"{len(unmeasured)} of {len(manifest)} PRs could not be measured: "
              + ", ".join(f"#{n}" for n in unmeasured) + f" (errors in {out_dir / 'replay.jsonl'})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

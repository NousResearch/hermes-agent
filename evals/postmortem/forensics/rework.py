"""Lane 5: post-open rework inventory for a large PR — reproducible part only.

    python -m evals.postmortem.forensics.rework --repo . --base <merge-base> --open <sha-at-pr-open> --head <merged-sha>

Reports (OBSERVED from git): the public-surface drop at PR open (via ``scripts/ci/check_public_surface.py``),
then every commit between the opening SHA and the merged head grouped by subject prefix
(``fix``, ``review-fix``, ``revert``, ``test``, ``docs``, ``simplify``, other) with files/insertions/deletions
and whether tests were touched. Root-cause classification of each commit (dropped symbol vs semantic
drift vs compat fallout ...) was done BY HAND in the original audit and is not reproducible here; the
counts this script prints are the mechanical inventory that classification started from.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path


def _git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--repo", default=".")
    ap.add_argument("--base", required=True, help="merge-base of the PR")
    ap.add_argument("--open", required=True, help="head SHA when the PR was opened / first reviewed")
    ap.add_argument("--head", required=True, help="final merged SHA")
    ap.add_argument("--out", default="postmortem_out")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # scripts/ci/check_public_surface.py ships with #103541; look for it in this tree, then the target repo.
    candidates = [Path(__file__).resolve().parents[3] / "scripts" / "ci" / "check_public_surface.py",
                  Path(a.repo).resolve() / "scripts" / "ci" / "check_public_surface.py"]
    checker = next((c for c in candidates if c.exists()), None)
    if checker is None:
        surface_line = "(scripts/ci/check_public_surface.py not found; merge #103541 or pass a checkout that has it)"
    else:
        surface = subprocess.run([sys.executable, str(checker), "--base", a.base, "--head", a.open, "--json", str(out / "surface_at_open.json")],
                                 cwd=a.repo, capture_output=True, text=True, encoding="utf-8", errors="replace")
        surface_line = (surface.stdout.splitlines() or [f"(check_public_surface failed: {surface.stderr.strip()[:120]})"])[0]

    log = _git(a.repo, "log", "--no-merges", "--format=%H%x00%s", f"{a.open}..{a.head}")
    groups = collections.defaultdict(list)
    for line in log.splitlines():
        if not line.strip():
            continue
        sha, subj = line.split("\x00", 1)
        m = re.match(r"^([a-z-]+)(\(|:|!)", subj)
        known = {"fix", "review-fix", "revert", "test", "docs", "simplify", "feat", "chore", "ci"}
        prefix = m.group(1) if m and m.group(1) in known else "other"
        files = _git(a.repo, "show", "--format=", "--name-only", sha).split()
        stat = _git(a.repo, "show", "--format=", "--shortstat", sha)
        ins = int((re.search(r"(\d+) insertion", stat) or [0, 0])[1]); dele = int((re.search(r"(\d+) deletion", stat) or [0, 0])[1])
        tests = [f for f in files if f.startswith("tests/") or "/tests/" in f]
        groups[prefix].append({"sha": sha[:10], "subject": subj[:120], "files": len(files), "ins": ins, "del": dele,
                               "test_files": len(tests), "src_only": len(tests) == 0 and len(files) > 0})
    summary = {g: {"commits": len(v), "files": sum(x["files"] for x in v), "ins": sum(x["ins"] for x in v), "del": sum(x["del"] for x in v),
                   "touching_no_tests": sum(1 for x in v if x["src_only"])} for g, v in groups.items()}
    report = {"surface_at_open": surface_line, "post_open_commits_by_prefix": summary, "commits": groups,
              "note": "root-cause classes were hand-labeled in the original audit; not reproduced here"}
    (out / "rework.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"[rework] surface at open: {surface_line}")
    total = sum(v["commits"] for v in summary.values())
    print(f"[rework] {total} post-open commits by prefix: " + ", ".join(f"{g} {v['commits']} ({v['touching_no_tests']} w/o tests)" for g, v in sorted(summary.items(), key=lambda kv: -kv[1]['commits'])))
    print(f"[rework] wrote {out / 'rework.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

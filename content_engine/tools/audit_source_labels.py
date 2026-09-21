"""Source-label audit — weekly scan of blog posts for source/label mismatches.

Scans src/content/blog/*.mdx and checks:
  - source: research-paper → body should contain arxiv.org link, DOI, or named paper
  - source: manual → if body references papers, flag for upgrade to research-paper

Output: JSON report with per-post status and summary counts.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional


def _extract_frontmatter(text: str) -> dict:
    """Extract YAML frontmatter as a dict (simple key: value parsing)."""
    fm: dict = {}
    if not text.startswith("---"):
        return fm
    end = text.find("---", 3)
    if end == -1:
        return fm
    yaml_block = text[3:end].strip()
    for line in yaml_block.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        fm[key] = val
    return fm


def _has_paper_reference(body: str) -> bool:
    """Check if body contains arxiv.org link, DOI, or named paper title."""
    if re.search(r"arxiv\.org", body, re.IGNORECASE):
        return True
    if re.search(r"doi:", body, re.IGNORECASE):
        return True
    if re.search(r"\b\d{4}\.\d{4,5}\b", body):  # arXiv ID pattern
        return True
    return False


def audit_post(mdx_path: Path) -> dict:
    """Audit a single MDX post for source-label consistency.

    Returns:
        {"slug": str, "source": str, "status": "ok"|"mismatch"|"upgrade",
         "issue": str|None}
    """
    text = mdx_path.read_text(encoding="utf-8", errors="replace")
    fm = _extract_frontmatter(text)
    source = fm.get("source", "manual")

    # Extract body (after frontmatter).
    body = text[text.find("---", 3) + 3:] if text.startswith("---") else text

    has_paper = _has_paper_reference(body)

    if source == "research-paper" and not has_paper:
        return {
            "slug": mdx_path.stem,
            "source": source,
            "status": "mismatch",
            "issue": "Source is research-paper but no arXiv link, DOI, or paper reference found in body",
        }
    elif source == "manual" and has_paper:
        return {
            "slug": mdx_path.stem,
            "source": source,
            "status": "upgrade",
            "issue": "Source is manual but body references papers — consider upgrading to research-paper",
        }
    return {
        "slug": mdx_path.stem,
        "source": source,
        "status": "ok",
        "issue": None,
    }


def audit_all(posts_dir: Path) -> dict:
    """Audit all MDX posts in a directory.

    Returns:
        {"total": int, "ok": int, "mismatches": int, "upgrades": int,
         "posts": [audit_post results]}
    """
    results = []
    for mdx in sorted(posts_dir.glob("*.mdx")):
        results.append(audit_post(mdx))

    ok = sum(1 for r in results if r["status"] == "ok")
    mismatches = sum(1 for r in results if r["status"] == "mismatch")
    upgrades = sum(1 for r in results if r["status"] == "upgrade")

    return {
        "total": len(results),
        "ok": ok,
        "mismatches": mismatches,
        "upgrades": upgrades,
        "posts": results,
    }


def main():
    """CLI entry: python -m tools.audit_source_labels --posts-dir <dir>"""
    import argparse
    parser = argparse.ArgumentParser(description="Audit blog source labels")
    parser.add_argument("--posts-dir", required=True, help="Path to src/content/blog")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    posts_dir = Path(args.posts_dir)
    if not posts_dir.exists():
        print(f"Error: {posts_dir} does not exist", file=sys.stderr)
        return 1

    report = audit_all(posts_dir)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Source-label audit: {report['total']} posts")
        print(f"  OK: {report['ok']}")
        print(f"  Mismatches: {report['mismatches']}")
        print(f"  Upgrade candidates: {report['upgrades']}")
        for p in report["posts"]:
            if p["status"] != "ok":
                print(f"  {p['slug']}: {p['issue']}")
    return 0 if report["mismatches"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
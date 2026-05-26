#!/usr/bin/env python3
"""Generate a repeatable Hermes skills inventory.

This is intentionally read-only. It scans the active user-local skill tree plus
repo bundled/optional skills, validates frontmatter/body basics, hashes each
SKILL.md, and reports duplicate-name/source divergence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - repo env normally has PyYAML
    yaml = None

ROOTS_DEFAULT = {
    "active_user_local": Path.home() / ".hermes" / "skills",
    "repo_builtin": Path("skills"),
    "repo_optional": Path("optional-skills"),
}

FM_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.S)


def parse_skill(path: Path, root: Path, source: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    rel = path.relative_to(root)
    category = rel.parts[0] if len(rel.parts) > 1 else "uncategorized"
    errors: list[str] = []
    meta: dict[str, Any] = {}
    body = text
    m = FM_RE.match(text)
    if not m:
        errors.append("missing_yaml_frontmatter")
    else:
        body = m.group(2)
        if yaml is None:
            errors.append("yaml_unavailable")
        else:
            try:
                loaded = yaml.safe_load(m.group(1)) or {}
                if not isinstance(loaded, dict):
                    errors.append("frontmatter_not_mapping")
                else:
                    meta = loaded
            except Exception as exc:
                errors.append(f"frontmatter_parse_error:{exc.__class__.__name__}")
    name = str(meta.get("name") or path.parent.name)
    description = str(meta.get("description") or "")
    if not name:
        errors.append("missing_name")
    if not description:
        errors.append("missing_description")
    if not body.strip():
        errors.append("empty_body")
    return {
        "source": source,
        "name": name,
        "category": category,
        "path": str(path),
        "relative_path": str(rel),
        "sha256": sha,
        "description": description,
        "tags": ((meta.get("metadata") or {}).get("hermes") or {}).get("tags", meta.get("tags", [])),
        "related_skills": ((meta.get("metadata") or {}).get("hermes") or {}).get("related_skills", meta.get("related_skills", [])),
        "errors": errors,
    }


def scan_root(source: str, root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    return [parse_skill(p, root, source) for p in sorted(root.rglob("SKILL.md"))]


def build_report(repo_root: Path) -> dict[str, Any]:
    roots = {
        name: (path if path.is_absolute() else repo_root / path)
        for name, path in ROOTS_DEFAULT.items()
    }
    skills_by_source = {source: scan_root(source, root) for source, root in roots.items()}
    all_skills = [s for skills in skills_by_source.values() for s in skills]
    active = skills_by_source["active_user_local"]
    source_skills = skills_by_source["repo_builtin"] + skills_by_source["repo_optional"]

    by_name_sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in source_skills:
        by_name_sources[s["name"]].append(s)

    duplicate_rows = []
    exact = divergent = 0
    for s in active:
        matches = by_name_sources.get(s["name"], [])
        if not matches:
            continue
        is_exact = any(m["sha256"] == s["sha256"] for m in matches)
        duplicate_rows.append({
            "name": s["name"],
            "active_path": s["path"],
            "source_paths": [m["path"] for m in matches],
            "exact": is_exact,
        })
        if is_exact:
            exact += 1
        else:
            divergent += 1

    active_name_counts = Counter(s["name"] for s in active)
    report = {
        "roots": {k: str(v) for k, v in roots.items()},
        "counts": {k: len(v) for k, v in skills_by_source.items()},
        "category_counts": {
            k: dict(sorted(Counter(s["category"] for s in v).items()))
            for k, v in skills_by_source.items()
        },
        "total_scanned": len(all_skills),
        "active_duplicate_names": {k: v for k, v in active_name_counts.items() if v > 1},
        "active_invalid_count": sum(1 for s in active if s["errors"]),
        "all_invalid_count": sum(1 for s in all_skills if s["errors"]),
        "source_duplicate_total": len(duplicate_rows),
        "source_duplicate_exact": exact,
        "source_duplicate_divergent": divergent,
        "divergent_source_duplicates": [r for r in duplicate_rows if not r["exact"]],
        "skills": all_skills,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("docs/plans/2026-05-26-skills-audit-data.json"))
    args = parser.parse_args()
    report = build_report(args.repo_root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "counts": report["counts"],
        "source_duplicate_total": report["source_duplicate_total"],
        "source_duplicate_exact": report["source_duplicate_exact"],
        "source_duplicate_divergent": report["source_duplicate_divergent"],
        "active_invalid_count": report["active_invalid_count"],
        "all_invalid_count": report["all_invalid_count"],
    }, indent=2, sort_keys=True))
    return 0 if report["active_invalid_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

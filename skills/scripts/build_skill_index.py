#!/usr/bin/env python3
"""Build skill registry index for semantic search and dependency resolution."""
import json
import yaml
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Get repo root (parent of skills dir)
REPO_ROOT = Path(__file__).parent.parent.parent
SKILLS_ROOT = REPO_ROOT / "skills"
INDEX_PATH = SKILLS_ROOT / "index-cache" / "registry.json"
INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

def parse_skill_md(skill_dir: Path) -> dict | None:
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return None
    content = skill_md.read_text(encoding="utf-8")
    # Parse frontmatter (--- delimited)
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                meta = yaml.safe_load(parts[1])
            except yaml.YAMLError:
                meta = {}
        else:
            meta = {}
    else:
        meta = {}
    if not meta:
        return None
    meta["path"] = str(skill_dir.relative_to(SKILLS_ROOT))
    meta["updated_at"] = datetime.fromtimestamp(skill_md.stat().st_mtime).isoformat()
    return meta

def main():
    registry = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "skills": [],
        "categories": {}
    }
    for cat_dir in SKILLS_ROOT.iterdir():
        if not cat_dir.is_dir() or cat_dir.name in {"index-cache", "templates"}:
            continue
        cat_skills = []
        for skill_dir in cat_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            meta = parse_skill_md(skill_dir)
            if meta:
                meta["category"] = cat_dir.name
                registry["skills"].append(meta)
                cat_skills.append(meta["name"])
        if cat_skills:
            registry["categories"][cat_dir.name] = cat_skills
    
    INDEX_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False))
    print(f"Indexed {len(registry['skills'])} skills across {len(registry['categories'])} categories")
    print(f"Written to {INDEX_PATH}")

if __name__ == "__main__":
    main()
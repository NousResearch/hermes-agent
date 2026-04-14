#!/usr/bin/env python3
"""
Batch fix skill frontmatter issues:
1. Add missing 'category' field (derived from path)
2. Add missing 'version' field (default "1.0.0")
3. Add missing 'triggers' field (derived from name/description)

Usage:
    python scripts/batch_fix_skills.py [--dry-run]
"""

import argparse
import re
import yaml
from pathlib import Path

SKILLS_DIR = Path.home() / ".hermes" / "skills"

# Required fields
REQUIRED_FIELDS = ["name", "description", "category"]
# Recommended fields to auto-fix
AUTO_FIX_FIELDS = ["version", "triggers"]


def derive_category(skill_path: Path) -> str:
    """Derive category from skill directory path.

    For paths like:
    - skills/github/code-review/SKILL.md -> "github"
    - skills/dogfood/SKILL.md -> "dogfood" (root-level skill uses its own name)
    """
    rel = skill_path.relative_to(SKILLS_DIR)
    parts = rel.parts
    if len(parts) >= 2:
        return parts[0]
    # Root-level skill (e.g., dogfood/, mmx/): use the skill directory name
    return skill_path.name


STOPWORDS = frozenset({"a", "an", "and", "or", "the", "for", "to", "of", "in", "on", "with", "by", "as", "at", "from", "via", "using", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "this", "that", "these", "those", "it", "its", "and", "or", "but", "if", "then", "else", "when", "up", "down", "out", "no", "so", "than", "too", "very", "just", "also", "now", "here", "there", "all", "each", "every", "both", "few", "more", "most", "other", "some", "such", "only", "own", "same", "than", "through", "about", "into", "over", "after", "before", "between", "under", "again", "further", "then", "once", "what", "which", "who", "whom", "where", "why", "how", "any", "each", "other", "same", "only", "own", "very", "just", "and,", "screen", "located"})


def derive_triggers(name: str, description: str) -> str:
    """Derive triggers from name and description keywords."""
    import re as _re
    
    # Extract key terms from name (replace hyphens with spaces)
    name_words = name.replace("-", " ").replace("_", " ").lower().split()
    
    # Extract words from description
    desc_lower = description.lower()
    # Remove common punctuation and split
    desc_words = _re.findall(r'\b[a-z][a-z0-9]+\b', desc_lower)
    
    # Filter stopwords and short words
    keywords = []
    seen = set()
    all_source_words = name_words + desc_words
    for w in all_source_words:
        w_lower = w.lower()
        if (w_lower not in seen 
            and w_lower not in STOPWORDS 
            and len(w_lower) > 2 
            and not w_lower.isdigit()):
            seen.add(w_lower)
            keywords.append(w)
    
    # Build triggers: skill name + up to 5 key domain words
    triggers_words = [name]
    triggers_words.extend([k for k in keywords if k.lower() != name.lower()][:5])
    return ", ".join(triggers_words)


def parse_frontmatter(content: str) -> tuple[dict, str, int, int]:
    """Parse YAML frontmatter, return (fm_dict, body, fm_start, fm_end)."""
    match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not match:
        return {}, content, -1, -1
    fm_text = match.group(1)
    fm = yaml.safe_load(fm_text) or {}
    body_start = match.end()
    return fm, content[body_start:], match.start(), match.end()


def fix_skill(skill_md: Path, dry_run: bool = True) -> dict:
    """Check and fix frontmatter for a single skill. Returns changes dict."""
    skill_dir = skill_md.parent
    rel_path = str(skill_dir.relative_to(SKILLS_DIR))
    
    content = skill_md.read_text(encoding="utf-8")
    fm, body, fm_start, fm_end = parse_frontmatter(content)
    
    if fm_start == -1:
        return {"skill": rel_path, "status": "skip", "reason": "no frontmatter"}
    
    changes = {"category": None, "version": None, "triggers": None}
    
    # 1. Fix missing 'category'
    if "category" not in fm or not fm.get("category"):
        derived_cat = derive_category(skill_dir)
        changes["category"] = derived_cat
    
    # 2. Fix missing 'version'
    if "version" not in fm or not fm.get("version"):
        changes["version"] = "1.0.0"
    
    # 3. Fix missing/empty 'triggers'
    triggers_val = fm.get("triggers") or fm.get("trigger") or ""
    if not triggers_val or not str(triggers_val).strip():
        name = fm.get("name", skill_dir.name)
        desc = fm.get("description", "")
        derived_triggers = derive_triggers(name, desc)
        changes["triggers"] = derived_triggers
    
    # Apply changes
    if dry_run:
        if any(v is not None for v in changes.values()):
            return {"skill": rel_path, "status": "would_fix", "changes": changes}
        return {"skill": rel_path, "status": "ok"}
    
    # Actually write changes
    needs_update = any(v is not None for v in changes.values())
    if not needs_update:
        return {"skill": rel_path, "status": "ok"}
    
    # Update frontmatter dict
    updated_fm = dict(fm)
    for field, value in changes.items():
        if value is not None:
            updated_fm[field] = value
    
    # Re-serialize frontmatter: use block scalar for long strings, keep simple values inline
    def represent_str(dumper, data):
        if '\n' in data or len(data) > 80:
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)
    
    yaml.add_representer(str, represent_str, Dumper=yaml.SafeDumper)
    
    # Determine field order: name, description first, then triggers, category, version, then rest
    priority_fields = ["name", "title", "description", "triggers", "category", "version"]
    ordered_fm = {}
    for f in priority_fields:
        if f in updated_fm:
            ordered_fm[f] = updated_fm.pop(f)
    ordered_fm.update(updated_fm)  # Add remaining fields
    
    fm_text = yaml.safe_dump(ordered_fm, default_flow_style=False, allow_unicode=True, sort_keys=False).strip()
    new_content = content[:fm_start] + "---\n" + fm_text + "\n---\n" + body
    
    skill_md.write_text(new_content, encoding="utf-8")
    return {"skill": rel_path, "status": "fixed", "changes": changes}


def main():
    parser = argparse.ArgumentParser(description="Batch fix skill frontmatter issues")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fixed without making changes")
    parser.add_argument("--category", action="store_true", help="Only fix missing category")
    parser.add_argument("--version", action="store_true", help="Only fix missing version")
    parser.add_argument("--triggers", action="store_true", help="Only fix missing/empty triggers")
    args = parser.parse_args()
    
    filter_fields = []
    if args.category:
        filter_fields.append("category")
    if args.version:
        filter_fields.append("version")
    if args.triggers:
        filter_fields.append("triggers")
    
    if not SKILLS_DIR.exists():
        print(f"Error: Skills directory not found: {SKILLS_DIR}")
        return 1
    
    results = {"ok": [], "would_fix": [], "fixed": [], "skip": []}
    total_changes = 0
    
    for skill_md in SKILLS_DIR.rglob("SKILL.md"):
        # Skip excluded dirs
        if any(part in {".github", ".hub", ".git"} for part in skill_md.parts):
            continue
        
        result = fix_skill(skill_md, dry_run=args.dry_run)
        status = result["status"]
        
        if status == "ok":
            results["ok"].append(result["skill"])
        elif status == "skip":
            results["skip"].append(result["skill"])
        elif status == "would_fix":
            changes = result["changes"]
            # Filter if requested
            if filter_fields:
                filtered_changes = {k: v for k, v in changes.items() if k in filter_fields and v is not None}
                if not filtered_changes:
                    results["ok"].append(result["skill"])
                    continue
                result["changes"] = filtered_changes
            results["would_fix"].append(result)
            total_changes += sum(1 for v in result["changes"].values() if v is not None)
        elif status == "fixed":
            results["fixed"].append(result)
            total_changes += sum(1 for v in result["changes"].values() if v is not None)
    
    # Print report
    mode = "DRY RUN" if args.dry_run else "FIXED"
    print(f"\n{'='*60}")
    print(f"Skill Frontmatter Fix ({mode})")
    print(f"{'='*60}")
    
    if args.dry_run:
        print(f"\nTotal skills: {len(results['ok']) + len(results['would_fix']) + len(results['skip'])}")
        print(f"  OK (no changes needed): {len(results['ok'])}")
        print(f"  Would fix: {len(results['would_fix'])}")
        print(f"  Skipped (no frontmatter): {len(results['skip'])}")
        print(f"\nTotal changes: {total_changes}")
        if filter_fields:
            print(f"Filtered to: {', '.join(filter_fields)}")
        
        print(f"\n--- Changes ---")
        for r in sorted(results["would_fix"], key=lambda x: x["skill"])[:50]:
            changes_str = ", ".join(f"{k}={v!r}" for k, v in r["changes"].items() if v is not None)
            print(f"  {r['skill']}: {changes_str}")
        if len(results["would_fix"]) > 50:
            print(f"  ... and {len(results['would_fix']) - 50} more")
    else:
        print(f"\nFixed: {len(results['fixed'])} skills")
        print(f"Total changes made: {total_changes}")
    
    print()
    return 0


if __name__ == "__main__":
    exit(main())

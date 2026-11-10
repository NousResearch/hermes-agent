"""Rebuild a skill library (names, descriptions, usage, file ages) inside a throwaway HERMES_HOME.

Input is a JSON list of ``{"rel": "category/dir", "name", "desc", "mtime", "usage": {...}|null}``
exported from a real library (``rel`` = the skill dir relative to the skills root). Rows without
``rel`` (plugin-registered skills have no file) are skipped: the index only tiers skills it can rank.
"""
import json
import os


def build_home(hermes_home: str, library_json: str) -> list[dict]:
    """Write every file-backed skill and the usage sidecar; returns the rows written."""
    with open(library_json, encoding="utf-8") as f:
        rows = [r for r in json.load(f) if r.get("rel")]
    skills = os.path.join(hermes_home, "skills")
    usage = {}
    for row in rows:
        path = os.path.join(skills, *row["rel"].split("/"), "SKILL.md")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"---\nname: {row['name']}\ndescription: {json.dumps(row['desc'])}\n---\n\n# {row['name']}\n\n"
                    f"Procedure for: {row['desc']}\n\n1. State the goal back in one line.\n2. Follow the user's request.\n")
        if row.get("mtime"):
            os.utime(path, (row["mtime"], row["mtime"]))
        if row.get("usage"):
            usage[row["name"]] = row["usage"]
    with open(os.path.join(skills, ".usage.json"), "w", encoding="utf-8") as f:
        json.dump(usage, f)
    return rows

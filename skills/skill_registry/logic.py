import os
from pathlib import Path


SKILLS_ROOT = Path(os.path.expanduser("~/.hermes/skills"))


def list_skills(skills_root: str = None) -> list[str]:
    """
    Scan the Hermes skills directory and return a list of all installed skills
    with their name and description extracted from each SKILL.md frontmatter.

    Args:
        skills_root: Optional override path to the skills directory.
                     Defaults to ~/.hermes/skills.

    Returns:
        A list of formatted strings: "[skill-name] — Description"
    """
    root = Path(skills_root) if skills_root else SKILLS_ROOT

    if not root.exists():
        return [f"Skills directory not found: {root}"]

    skill_entries = []

    # Walk all subdirectories looking for SKILL.md files
    for skill_md in sorted(root.rglob("SKILL.md")):
        skill_dir = skill_md.parent
        name = skill_dir.name
        description = _extract_description(skill_md)
        skill_entries.append(f"[{name}] — {description}")

    if not skill_entries:
        return ["No skills found."]

    return skill_entries


def _extract_description(skill_md: Path) -> str:
    """
    Parse the YAML frontmatter of a SKILL.md file and return the description field.
    Falls back to 'No description.' if not found.
    """
    try:
        content = skill_md.read_text(encoding="utf-8")
        lines = content.splitlines()

        # Frontmatter is between the first two '---' lines
        if lines[0].strip() != "---":
            return "No description."

        in_front = False
        for line in lines:
            stripped = line.strip()
            if stripped == "---":
                in_front = not in_front
                continue
            if in_front and stripped.startswith("description:"):
                return stripped.split("description:", 1)[1].strip().strip('"').strip("'")

    except Exception:
        pass

    return "No description."


if __name__ == "__main__":
    print(f"Scanning skills in: {SKILLS_ROOT}\n")
    results = list_skills()
    print(f"Found {len(results)} skill(s):\n")
    for entry in results:
        print(" ", entry)

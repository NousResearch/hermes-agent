#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Scaffold a BMad module setup skill from template.

Copies the setup-skill-template into the target directory as {code}-setup/,
then writes the generated module.yaml and module-help.csv into the assets folder
and updates the SKILL.md frontmatter with the module's identity.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scaffold a BMad module setup skill from template"
    )
    parser.add_argument(
        "--target-dir",
        required=True,
        help="Directory to create the setup skill in (the user's skills folder)",
    )
    parser.add_argument(
        "--module-code",
        required=True,
        help="Module code (2-4 letter abbreviation, e.g. 'cis')",
    )
    parser.add_argument(
        "--module-name",
        required=True,
        help="Module display name (e.g. 'Creative Intelligence Suite')",
    )
    parser.add_argument(
        "--module-yaml",
        required=True,
        help="Path to the generated module.yaml content file",
    )
    parser.add_argument(
        "--module-csv",
        required=True,
        help="Path to the generated module-help.csv content file",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print progress to stderr"
    )
    args = parser.parse_args()

    template_dir = Path(__file__).resolve().parent.parent / "assets" / "setup-skill-template"
    setup_skill_name = f"{args.module_code}-setup"
    target = Path(args.target_dir) / setup_skill_name

    if not template_dir.is_dir():
        print(
            json.dumps({"status": "error", "message": f"Template not found: {template_dir}"}),
            file=sys.stdout,
        )
        return 2

    for source_path in [args.module_yaml, args.module_csv]:
        if not Path(source_path).is_file():
            print(
                json.dumps({"status": "error", "message": f"Source file not found: {source_path}"}),
                file=sys.stdout,
            )
            return 2

    target_dir = Path(args.target_dir)
    if not target_dir.is_dir():
        print(
            json.dumps({"status": "error", "message": f"Target directory not found: {target_dir}"}),
            file=sys.stdout,
        )
        return 2

    # Remove existing setup skill if present (anti-zombie)
    if target.exists():
        if args.verbose:
            print(f"Removing existing {setup_skill_name}/", file=sys.stderr)
        shutil.rmtree(target)

    # Copy template
    if args.verbose:
        print(f"Copying template to {target}", file=sys.stderr)
    shutil.copytree(template_dir, target)

    # Update SKILL.md frontmatter placeholders
    skill_md = target / "SKILL.md"
    content = skill_md.read_text(encoding="utf-8")
    content = content.replace("{setup-skill-name}", setup_skill_name)
    content = content.replace("{module-name}", args.module_name)
    content = content.replace("{module-code}", args.module_code)
    skill_md.write_text(content, encoding="utf-8")

    # Write generated module.yaml
    yaml_content = Path(args.module_yaml).read_text(encoding="utf-8")
    (target / "assets" / "module.yaml").write_text(yaml_content, encoding="utf-8")

    # Write generated module-help.csv
    csv_content = Path(args.module_csv).read_text(encoding="utf-8")
    (target / "assets" / "module-help.csv").write_text(csv_content, encoding="utf-8")

    # Collect file list
    files_created = sorted(
        str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()
    )

    result = {
        "status": "success",
        "setup_skill": setup_skill_name,
        "location": str(target),
        "files_created": files_created,
        "files_count": len(files_created),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

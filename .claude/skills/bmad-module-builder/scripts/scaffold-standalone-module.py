#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Scaffold standalone module infrastructure into an existing skill.

Copies template files (module-setup.md, merge scripts) into the skill directory
and generates a .claude-plugin/marketplace.json for distribution. The LLM writes
module.yaml and module-help.csv directly to the skill's assets/ folder before
running this script.
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scaffold standalone module infrastructure into an existing skill"
    )
    parser.add_argument(
        "--skill-dir",
        required=True,
        help="Path to the existing skill directory (must contain SKILL.md)",
    )
    parser.add_argument(
        "--module-code",
        required=True,
        help="Module code (2-4 letter abbreviation, e.g. 'exc')",
    )
    parser.add_argument(
        "--module-name",
        required=True,
        help="Module display name (e.g. 'Excalidraw Tools')",
    )
    parser.add_argument(
        "--marketplace-dir",
        default=None,
        help="Directory to create .claude-plugin/ in (defaults to skill-dir parent)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print progress to stderr"
    )
    args = parser.parse_args()

    template_dir = (
        Path(__file__).resolve().parent.parent
        / "assets"
        / "standalone-module-template"
    )
    skill_dir = Path(args.skill_dir).resolve()
    marketplace_dir = (
        Path(args.marketplace_dir).resolve() if args.marketplace_dir else skill_dir.parent
    )

    # --- Validation ---

    if not template_dir.is_dir():
        print(
            json.dumps({"status": "error", "message": f"Template not found: {template_dir}"}),
            file=sys.stdout,
        )
        return 2

    if not skill_dir.is_dir():
        print(
            json.dumps({"status": "error", "message": f"Skill directory not found: {skill_dir}"}),
            file=sys.stdout,
        )
        return 2

    if not (skill_dir / "SKILL.md").is_file():
        print(
            json.dumps({"status": "error", "message": f"No SKILL.md found in {skill_dir}"}),
            file=sys.stdout,
        )
        return 2

    if not (skill_dir / "assets" / "module.yaml").is_file():
        print(
            json.dumps({
                "status": "error",
                "message": f"assets/module.yaml not found in {skill_dir} — the LLM must write it before running this script",
            }),
            file=sys.stdout,
        )
        return 2

    # --- Copy template files ---

    files_created: list[str] = []
    files_skipped: list[str] = []
    warnings: list[str] = []

    # 1. Copy module-setup.md to assets/ (alongside module.yaml and module-help.csv)
    assets_dir = skill_dir / "assets"
    assets_dir.mkdir(exist_ok=True)
    src_setup = template_dir / "module-setup.md"
    dst_setup = assets_dir / "module-setup.md"
    if args.verbose:
        print(f"Copying module-setup.md to {dst_setup}", file=sys.stderr)
    dst_setup.write_bytes(src_setup.read_bytes())
    files_created.append("assets/module-setup.md")

    # 2. Copy merge scripts to scripts/
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    for script_name in ("merge-config.py", "merge-help-csv.py"):
        src = template_dir / script_name
        dst = scripts_dir / script_name
        if dst.exists():
            msg = f"scripts/{script_name} already exists — skipped to avoid overwriting"
            files_skipped.append(f"scripts/{script_name}")
            warnings.append(msg)
            if args.verbose:
                print(f"SKIP: {msg}", file=sys.stderr)
        else:
            if args.verbose:
                print(f"Copying {script_name} to {dst}", file=sys.stderr)
            dst.write_bytes(src.read_bytes())
            dst.chmod(0o755)
            files_created.append(f"scripts/{script_name}")

    # 3. Generate marketplace.json
    plugin_dir = marketplace_dir / ".claude-plugin"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    marketplace_json = plugin_dir / "marketplace.json"

    # Read module.yaml for description and version
    module_yaml_path = skill_dir / "assets" / "module.yaml"
    module_description = ""
    module_version = "1.0.0"
    try:
        yaml_text = module_yaml_path.read_text(encoding="utf-8")
        for line in yaml_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("description:"):
                module_description = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            elif stripped.startswith("module_version:"):
                module_version = stripped.split(":", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass

    skill_dir_name = skill_dir.name
    marketplace_data = {
        "name": args.module_code,
        "owner": {"name": ""},
        "license": "",
        "homepage": "",
        "repository": "",
        "keywords": ["bmad"],
        "plugins": [
            {
                "name": args.module_code,
                "source": "./",
                "description": module_description,
                "version": module_version,
                "author": {"name": ""},
                "skills": [f"./{skill_dir_name}"],
            }
        ],
    }

    if args.verbose:
        print(f"Writing marketplace.json to {marketplace_json}", file=sys.stderr)
    marketplace_json.write_text(
        json.dumps(marketplace_data, indent=2) + "\n", encoding="utf-8"
    )
    files_created.append(".claude-plugin/marketplace.json")

    # --- Result ---

    result = {
        "status": "success",
        "skill_dir": str(skill_dir),
        "module_code": args.module_code,
        "files_created": files_created,
        "files_skipped": files_skipped,
        "warnings": warnings,
        "marketplace_json": str(marketplace_json),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

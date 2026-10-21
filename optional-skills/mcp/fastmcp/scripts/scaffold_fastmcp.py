#!/usr/bin/env python3
"""Copy a FastMCP starter template into a working file."""

from __future__ import annotations

import argparse
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
TEMPLATE_DIR = SKILL_DIR / "templates"
PLACEHOLDER = "__SERVER_NAME__"


def list_templates() -> list[str]:
    return sorted(path.stem for path in TEMPLATE_DIR.glob("*.py"))


def render_template(template_name: str, server_name: str) -> str:
    template_path = TEMPLATE_DIR / f"{template_name}.py"
    if not template_path.exists():
        available = ", ".join(list_templates())
        raise SystemExit(f"Unknown template '{template_name}'. Available: {available}")
    return template_path.read_text(encoding="utf-8").replace(PLACEHOLDER, server_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", help="Template name without .py suffix")
    parser.add_argument("--name", help="FastMCP server display name")
    parser.add_argument("--output", help="Destination Python file path")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    parser.add_argument("--list", action="store_true", help="List available templates and exit")
    args = parser.parse_args()

    if args.list:
        for name in list_templates():
            print(name)
        return 0

    if not args.template or not args.name or not args.output:
        parser.error("--template, --name, and --output are required unless --list is used")

    output_path = Path(args.output).expanduser()
    if output_path.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing file: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_template(args.template, args.name), encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

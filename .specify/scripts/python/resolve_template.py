#!/usr/bin/env python3
"""Resolve composed template content from the project template stack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from common import (
        TemplateResolutionError,
        get_repo_root,
        resolve_template_content,
    )
except ImportError:  # pragma: no cover - direct execution from unusual cwd
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import (
        TemplateResolutionError,
        get_repo_root,
        resolve_template_content,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("template_name")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    repo_root = get_repo_root(Path(__file__))
    try:
        content = resolve_template_content(args.template_name, repo_root)
    except TemplateResolutionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if content is None:
        print(
            f"ERROR: Could not resolve required {args.template_name} from the "
            f"template override stack for {repo_root}",
            file=sys.stderr,
        )
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "TEMPLATE_NAME": args.template_name,
                    "TEMPLATE_CONTENT": content,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    else:
        sys.stdout.write(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Check tasks prerequisites and resolve the tasks template."""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from common import (
        FeaturePaths,
        TemplateResolutionError,
        format_speckit_command,
        get_feature_paths,
        resolve_template,
        resolve_template_content,
    )
except ImportError:  # pragma: no cover - direct execution from unusual cwd
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import (
        FeaturePaths,
        TemplateResolutionError,
        format_speckit_command,
        get_feature_paths,
        resolve_template,
        resolve_template_content,
    )


def _json_line(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def _help_text(argv0: str) -> str:
    return f"""Usage: {argv0} [--json]
  --json    Output results in JSON format
  --help    Show this help message
"""


def _dir_has_entries(path: Path) -> bool:
    try:
        return path.is_dir() and any(path.iterdir())
    except OSError:
        return False


def _available_docs(paths: FeaturePaths) -> list[str]:
    docs: list[str] = []
    if paths.research.is_file():
        docs.append("research.md")
    if paths.data_model.is_file():
        docs.append("data-model.md")
    if _dir_has_entries(paths.contracts_dir):
        docs.append("contracts/")
    if paths.quickstart.is_file():
        docs.append("quickstart.md")
    return docs


def _status_marker(ok: bool) -> str:
    """Return the status glyph, downgraded to ASCII when stdout cannot encode it.

    On Windows sys.stdout falls back to the ANSI code page whenever it is not a
    console - a pipe or a file redirect, which is how agents and workflow steps
    invoke these scripts - and U+2713 is unencodable in cp1252, so printing it
    raised UnicodeEncodeError and aborted the report mid-listing.
    "[OK]"/"[FAIL]" is the ASCII rendering these markers already have in-tree:
    see Test-FileExists in scripts/powershell/common.ps1 and
    normalize_status_text in tests/parity_helpers.py.
    """
    glyph = "✓" if ok else "✗"
    try:
        glyph.encode(getattr(sys.stdout, "encoding", None) or "utf-8")
    except (LookupError, UnicodeEncodeError):
        return "[OK]" if ok else "[FAIL]"
    return glyph


def _check_file(path: Path, description: str) -> None:
    print(f"  {_status_marker(path.is_file())} {description}")


def _check_dir(path: Path, description: str) -> None:
    print(f"  {_status_marker(_dir_has_entries(path))} {description}")


def main(argv: list[str] | None = None) -> int:
    json_mode = False
    for arg in list(argv if argv is not None else sys.argv[1:]):
        if arg == "--json":
            json_mode = True
        elif arg in {"--help", "-h"}:
            sys.stdout.write(_help_text(sys.argv[0]))
            return 0
        else:
            print(f"ERROR: Unknown option '{arg}'", file=sys.stderr)
            return 1

    try:
        paths = get_feature_paths(script_file=Path(__file__))
    except SystemExit as exc:
        if exc.code == 0:
            return 0
        print("ERROR: Failed to resolve feature paths", file=sys.stderr)
        return int(exc.code) if isinstance(exc.code, int) else 1

    if not paths.impl_plan.is_file():
        print(f"ERROR: plan.md not found in {paths.feature_dir}", file=sys.stderr)
        print(
            f"Run {format_speckit_command('plan', paths.repo_root)} first to create the implementation plan.",
            file=sys.stderr,
        )
        return 1

    if not paths.feature_spec.is_file():
        print(f"ERROR: spec.md not found in {paths.feature_dir}", file=sys.stderr)
        print(
            f"Run {format_speckit_command('specify', paths.repo_root)} first to create the feature structure.",
            file=sys.stderr,
        )
        return 1

    docs = _available_docs(paths)

    try:
        tasks_template_content = resolve_template_content(
            "tasks-template", paths.repo_root
        )
    except TemplateResolutionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if tasks_template_content is None:
        print(
            "ERROR: Could not resolve required tasks-template from the template "
            f"override stack for {paths.repo_root}",
            file=sys.stderr,
        )
        print(
            "Template 'tasks-template' was not found in any supported location "
            "(overrides, presets, extensions, or shared core). Add an override at "
            ".specify/templates/overrides/tasks-template.md, or run 'specify init' "
            "/ reinstall shared infra to restore the core "
            ".specify/templates/tasks-template.md template.",
            file=sys.stderr,
        )
        return 1

    if json_mode:
        tasks_template = resolve_template("tasks-template", paths.repo_root)
        sys.stdout.write(
            _json_line(
                {
                    "FEATURE_DIR": str(paths.feature_dir),
                    "AVAILABLE_DOCS": docs,
                    "TASKS_TEMPLATE": str(tasks_template) if tasks_template else "",
                    "TASKS_TEMPLATE_CONTENT": tasks_template_content,
                }
            )
        )
    else:
        tasks_template = resolve_template("tasks-template", paths.repo_root)
        print(f"FEATURE_DIR: {paths.feature_dir}")
        print(f"TASKS_TEMPLATE: {tasks_template or 'not found'}")
        print("AVAILABLE_DOCS:")
        _check_file(paths.research, "research.md")
        _check_file(paths.data_model, "data-model.md")
        _check_dir(paths.contracts_dir, "contracts/")
        _check_file(paths.quickstart, "quickstart.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

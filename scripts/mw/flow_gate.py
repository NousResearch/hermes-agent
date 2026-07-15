#!/usr/bin/env python3
"""CLI gate for live Migrate Web flow evidence.

Commands:

  flow_gate.py status <menu> [--project-root DIR] [--rules FILE] [--json]
  flow_gate.py can-enter <STEP> <menu> [--project-root DIR] [--rules FILE] [--json]
  flow_gate.py guard-write <file_path> [--rules FILE]

Exit codes: 0 = readable/pass, 1 = cannot enter, 2 = usage/config error.
No state or cache files are created; every invocation reads evidence live.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

from flow_eval import ConfigError, can_enter, evaluate, load_rules, validate_menu

EXIT_OK = 0
EXIT_BLOCKED = 1
EXIT_ERROR = 2
DEFAULT_RULES_PATH = Path(__file__).with_name("flow-rules.yaml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flow_gate.py",
        description=(
            "Evaluate the 13-step Migrate Web flow from live evidence. "
            "Exit 0 for status/readable or enterable, 1 when blocked, 2 on error."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Show every flow step")
    status.add_argument("menu", help="Safe menu slug")
    _add_common_arguments(status)

    enter = subparsers.add_parser("can-enter", help="Check all steps before STEP")
    enter.add_argument("step", help="Target step id, for example M4")
    enter.add_argument("menu", help="Safe menu slug")
    _add_common_arguments(enter)

    guard = subparsers.add_parser(
        "guard-write", help="Check whether a file may be written in an MW project"
    )
    guard.add_argument("file_path", type=Path, help="File about to be written")
    guard.add_argument(
        "--rules",
        type=Path,
        default=None,
        help=(
            "Complete rules file (default: <project>/.work/flow-rules.yaml "
            "when present, otherwise the rules beside this script)"
        ),
    )
    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root containing evidence (default: cwd)",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=None,
        help=(
            "Complete rules file (default: <project>/.work/flow-rules.yaml "
            "when present, otherwise the rules beside this script)"
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")


def resolve_rules_path(project_root: Path, explicit: Optional[Path]) -> Path:
    """Choose one complete rules file; project override replaces the default."""
    if explicit is not None:
        path = explicit.expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            raise ConfigError(f"rules file not found: {path}")
        return path

    override = project_root / ".work" / "flow-rules.yaml"
    if override.is_file():
        return override
    if DEFAULT_RULES_PATH.is_file():
        return DEFAULT_RULES_PATH
    raise ConfigError(
        f"rules file not found: neither {override} nor {DEFAULT_RULES_PATH} exists"
    )


def _print_error(message: str, as_json: bool) -> None:
    if as_json:
        print(json.dumps({"ok": False, "error": message}, ensure_ascii=False))
    else:
        print(f"error: {message}", file=sys.stderr)


def _format_status(result: dict) -> str:
    lines = ["ขั้น | ชื่อ | สถานะ | ขาดอะไร", "--- | --- | --- | ---"]
    for step in result["steps"]:
        missing = "; ".join(step["missing"]) if step["missing"] else "-"
        lines.append(
            f"{step['id']} | {step['title']} | {step['status']} | {missing}"
        )
    current = result["current_step"] if result["current_step"] is not None else "none"
    lines.append(
        f"flow: {result['done_count']}/{result['total']} done · current={current}"
    )
    return "\n".join(lines)


def find_project_root(file_path: Path) -> Optional[Path]:
    """Return the nearest parent carrying the MW project marker."""
    target = file_path.expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    target = Path(os.path.abspath(str(target)))
    for parent in (target.parent, *target.parents):
        if (parent / ".work" / "profile.yaml").is_file():
            return parent
    return None


def _output_match(relative_path: str, pattern: str) -> Optional[str]:
    """Return the validated menu captured from one exact output pattern."""
    marker = "{menu}"
    if marker not in pattern:
        if relative_path == pattern:
            raise ConfigError(f"output path cannot identify menu: {pattern}")
        return None
    expression = "^" + re.escape(pattern).replace(re.escape(marker), "(?P<menu>[^/]+)") + "$"
    match = re.fullmatch(expression, relative_path)
    if match is None:
        return None
    return validate_menu(match.group("menu"))


def _has_user_menu_lock(project_root: Path) -> bool:
    queue = project_root / ".work" / "menu-queue.md"
    user = os.environ.get("MW_USER") or os.environ.get("USER")
    if not user:
        return False
    try:
        lines = queue.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ConfigError(f"cannot read menu queue {queue}: {exc}") from exc
    locked_by = re.compile(rf"locked_by:\s*{re.escape(user)}(?:\s|$)")
    return any(locked_by.search(line) and "released" not in line.lower() for line in lines)


def guard_write(file_path: Path, explicit_rules: Optional[Path] = None) -> int:
    """Apply flow ordering or the menu-lock policy to one prospective write."""
    target = file_path.expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    target = Path(os.path.abspath(str(target)))
    root = find_project_root(target)
    if root is None:
        return EXIT_OK

    try:
        relative = target.relative_to(root).as_posix()
    except ValueError as exc:
        raise ConfigError(f"write path escapes project root: {target}") from exc
    real_root = Path(os.path.realpath(str(root)))
    real_target = Path(os.path.realpath(str(target)))
    try:
        real_target.relative_to(real_root)
    except ValueError as exc:
        raise ConfigError(f"write path escapes project root through a symlink: {target}") from exc

    rules_path = resolve_rules_path(root, explicit_rules)
    rules = load_rules(rules_path)
    for step in rules["steps"]:
        for output in step["outputs"]:
            menu = _output_match(relative, output["path"])
            if menu is None:
                continue
            allowed, reasons = can_enter(root, menu, step["id"], rules)
            if allowed:
                return EXIT_OK
            print(
                f"ไม่อนุญาตให้เขียน {relative}: ขั้น {step['id']} ยังเข้าไม่ได้",
                file=sys.stderr,
            )
            for reason in reasons:
                print(f"- {reason}", file=sys.stderr)
            return EXIT_BLOCKED

    queue = root / ".work" / "menu-queue.md"
    if not queue.exists():
        print(
            "คำเตือน: โปรเจกต์ MW ยังไม่มี .work/menu-queue.md; "
            "อนุญาตชั่วคราวจนกว่าจะตั้งคิวเมนู",
            file=sys.stderr,
        )
        return EXIT_OK
    if _has_user_menu_lock(root):
        return EXIT_OK
    user = os.environ.get("MW_USER") or os.environ.get("USER") or "ผู้ใช้ปัจจุบัน"
    print(
        "ต้องจองเมนูใน menu-queue ก่อนเขียนไฟล์โปรเจกต์ "
        f"(ต้องมี locked_by: {user} และยังไม่มี released)",
        file=sys.stderr,
    )
    return EXIT_BLOCKED


def run(argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI and return its process exit code."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "guard-write":
            return guard_write(args.file_path, args.rules)

        root = (args.project_root or Path.cwd()).expanduser().resolve()
        rules_path = resolve_rules_path(root, args.rules)
        rules = load_rules(rules_path)
        if args.command == "status":
            result = evaluate(root, args.menu, rules)
            if args.json:
                print(json.dumps(result, ensure_ascii=False))
            else:
                print(_format_status(result))
            return EXIT_OK

        allowed, reasons = can_enter(root, args.menu, args.step, rules)
        payload = {
            "menu": args.menu,
            "step": args.step,
            "can_enter": allowed,
            "missing": reasons,
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False))
        elif allowed:
            print(f"can-enter {args.step}: YES")
        else:
            print(f"can-enter {args.step}: NO")
            for reason in reasons:
                print(f"- {reason}")
        return EXIT_OK if allowed else EXIT_BLOCKED
    except ConfigError as exc:
        _print_error(str(exc), bool(getattr(args, "json", False)))
        return EXIT_ERROR
    except (OSError, UnicodeError) as exc:
        _print_error(f"cannot evaluate flow: {exc}", bool(getattr(args, "json", False)))
        return EXIT_ERROR


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()

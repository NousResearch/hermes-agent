"""User-local Harness / Agenting Engineering Hermes plugin.

This plugin intentionally delegates implementation to the user-local helper
`~/.hermes/bin/hermes-harness` so it remains easy to roll back without touching
Hermes core code.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Sequence

HELP_TEXT = """Harness / Agenting Engineering intake

CLI:
  hermes harness template
  hermes harness new --title "My task" --workspace /path/to/repo --mode "Implement changes"
  hermes harness check /path/to/intake.md
  hermes harness prompt /path/to/intake.md

Local helper:
  hermes-harness <template|new|check|prompt> ...

Purpose:
  Move non-trivial AI-assisted coding tasks from vibe coding to sustainable
  Harness / Agenting Engineering by requiring task scope, acceptance criteria,
  risk surface, and verification evidence before implementation.
""".strip()


def _helper_path() -> Path:
    return Path.home() / ".hermes" / "bin" / "hermes-harness"


def _run_helper(argv: Sequence[str]) -> int:
    helper = _helper_path()
    if not helper.exists():
        print(f"Missing helper: {helper}")
        print("Expected script created by the harness-agenting-engineering skill.")
        return 2
    proc = subprocess.run([str(helper), *argv], check=False)
    return int(proc.returncode)


def _setup_harness_cli(parser) -> None:
    sub = parser.add_subparsers(dest="harness_action")

    template_p = sub.add_parser("template", help="Print or copy the Harness task intake template")
    template_p.add_argument("--output", "-o", default="", help="Copy template to this path instead of stdout")

    new_p = sub.add_parser("new", help="Create a new Harness task intake form")
    new_p.add_argument("--title", default="", help="Task title")
    new_p.add_argument("--workspace", default="", help="Workspace / repo path")
    new_p.add_argument("--mode", default="", help="Desired mode / permission level")
    new_p.add_argument("--out", default="", help="Output path or directory")
    new_p.add_argument("--print-path", action="store_true", help="Print only the created file path")

    check_p = sub.add_parser("check", help="Validate a filled Harness intake form")
    check_p.add_argument("file", help="Intake markdown file")

    prompt_p = sub.add_parser("prompt", help="Render a compact prompt from a filled intake form")
    prompt_p.add_argument("file", help="Intake markdown file")
    prompt_p.add_argument("--allow-incomplete", action="store_true", help="Render even if required fields are missing")
    prompt_p.add_argument("--output", "-o", default="", help="Write prompt to this file instead of stdout")
    prompt_p.add_argument("--force", action="store_true", help="Overwrite output file if it exists")

    parser.set_defaults(func=_handle_harness_cli)


def _handle_harness_cli(args) -> None:
    action = getattr(args, "harness_action", None)
    if not action:
        print(HELP_TEXT)
        raise SystemExit(0)

    argv: list[str] = [action]
    if action == "new":
        for flag in ("title", "workspace", "mode"):
            value = getattr(args, flag, "")
            if value:
                argv.extend([f"--{flag}", value])
        out_value = getattr(args, "out", "")
        if out_value:
            argv.extend(["--output", out_value])
        # The helper already prints the created path for `new`; keep
        # --print-path as a plugin-side compatibility flag without passing it
        # to the helper.
    elif action == "check":
        argv.append(getattr(args, "file"))
    elif action == "prompt":
        argv.append(getattr(args, "file"))
        if getattr(args, "allow_incomplete", False):
            argv.append("--allow-incomplete")
        output_value = getattr(args, "output", "")
        if output_value:
            argv.extend(["--output", output_value])
        if getattr(args, "force", False):
            argv.append("--force")
    elif action == "template":
        output_value = getattr(args, "output", "")
        if output_value:
            argv.extend(["--output", output_value])
    else:
        print(HELP_TEXT)
        raise SystemExit(2)
    raise SystemExit(_run_helper(argv))


ENGINEERING_KEYWORDS = re.compile(
    r"("
    r"修复|修 bug|bug|报错|异常|失败|不生效|调试|排查|"
    r"实现|开发|改代码|重构|优化|接入|集成|迁移|发布|部署|"
    r"测试|单测|pytest|CI|PR|代码审查|review|"
    r"feature|fix|debug|refactor|implement|integrat|migrat|deploy|release|test|coverage"
    r")",
    re.IGNORECASE,
)
LOW_RISK_PATTERNS = re.compile(
    r"^(怎么看|是什么|解释|总结|翻译|润色|写一段|生成文案|画|查一下|搜索|what is|explain|summarize|translate)\b",
    re.IGNORECASE,
)
HARNESS_ALREADY_PRESENT = re.compile(
    r"(Harness\s*/\s*Agenting|harness-agenting|hermes\s+harness|/intake|intake\s+form|验收标准|验证证据|risk surface)",
    re.IGNORECASE,
)

PREFLIGHT_NOTICE = """[Harness / Agenting Engineering preflight]\nThis appears to be a non-trivial engineering task. Before implementing, use the harness discipline:\n- define scope, acceptance criteria, risk surface, and rollback plan;\n- inspect the codebase before editing;\n- preserve tests / quality gates as evidence;\n- prefer reusable Skill/Rules/Plugin updates for repeated workflow.\nIf the task is underspecified, ask only for missing information that changes the implementation.\n\nOriginal user request:\n"""

STRICT_NOTICE = """[Harness / Agenting Engineering preflight: intake required]\nThis appears to be a high-risk engineering task. Ask the user to create or fill a Harness intake before implementation:\n  hermes harness new --title \"<task>\" --workspace \"<repo>\" --mode \"Implement changes\" --out /tmp/task-intake.md\n  hermes harness check /tmp/task-intake.md\nProceed only after scope, acceptance criteria, risk surface, and verification evidence are explicit.\n\nOriginal user request:\n"""


def _preflight_mode() -> str:
    env_mode = os.getenv("HERMES_HARNESS_PREFLIGHT")
    if env_mode is not None:
        return env_mode.strip().lower()

    try:
        from hermes_cli.config import cfg_get, load_config

        config_mode = cfg_get(
            load_config(),
            "harness_engineering",
            "preflight_mode",
            default="advisory",
        )
    except Exception:
        config_mode = "advisory"

    if not isinstance(config_mode, str):
        return "advisory"
    return config_mode.strip().lower() or "advisory"


def _looks_like_engineering_task(text: str) -> bool:
    compact = (text or "").strip()
    if not compact:
        return False
    if compact.startswith("/"):
        return False
    if HARNESS_ALREADY_PRESENT.search(compact):
        return False
    if LOW_RISK_PATTERNS.search(compact) and len(compact) < 220:
        return False
    return bool(ENGINEERING_KEYWORDS.search(compact))


def _handle_pre_gateway_dispatch(event: Any = None, **_: Any) -> dict[str, str] | None:
    """Soft Level-4 preflight for gateway messages.

    Modes via HERMES_HARNESS_PREFLIGHT:
      off/0/false/no  -> disabled
      advisory/warn/rewrite (default) -> prepend Harness discipline reminder
      strict -> prepend an intake-required instruction

    The hook returns only `allow` or `rewrite`; it never skips messages.
    """
    mode = _preflight_mode()
    if mode in {"", "off", "0", "false", "no", "disabled"}:
        return {"action": "allow"}
    text = getattr(event, "text", "") if event is not None else ""
    if not isinstance(text, str) or not _looks_like_engineering_task(text):
        return {"action": "allow"}
    if mode in {"strict", "require", "required"}:
        return {"action": "rewrite", "text": STRICT_NOTICE + text}
    return {"action": "rewrite", "text": PREFLIGHT_NOTICE + text}


def _handle_intake_slash(raw_args: str = "") -> str:
    raw = (raw_args or "").strip()
    if raw:
        return (
            f"/intake currently provides entry instructions only. Received: {raw}\n\n"
            f"{HELP_TEXT}"
        )
    return HELP_TEXT


def register(ctx) -> None:
    ctx.register_cli_command(
        "harness",
        help="Harness / Agenting Engineering task intake helper",
        description=HELP_TEXT,
        setup_fn=_setup_harness_cli,
        handler_fn=_handle_harness_cli,
    )
    ctx.register_command(
        "intake",
        handler=_handle_intake_slash,
        description="Show Harness / Agenting Engineering task intake instructions.",
        args_hint="[optional note]",
    )
    ctx.register_hook("pre_gateway_dispatch", _handle_pre_gateway_dispatch)

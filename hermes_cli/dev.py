"""Developer workflow commands for Hermes maintainers.

This module intentionally stays thin: it builds inspectable command plans for
linting, testing, odin sync/restart, and live smoke validation, then executes
them only when the caller has opted into mutation.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ODIN_HOST = "odin"
ODIN_USER = "hermes"
ODIN_WORKTREE = "/home/hermes/.hermes/hermes-agent-context-work"
ODIN_LIVE_TREE = "/home/hermes/.hermes/hermes-agent"
ODIN_PYTHON = "/home/hermes/.hermes/hermes-agent/venv/bin/python"
DEFAULT_VALIDATION_CHANNEL_ID = "1501008202630696981"
PYTHON = sys.executable or "python"

TOUCHED_PACKAGES = (
    "gateway/context_layers.py",
    "gateway/startup_context.py",
    "gateway/context_dump.py",
    "gateway/route_banner.py",
    "gateway/tool_policy.py",
    "gateway/validation/discord_context_smoke.py",
    "hermes_cli/dev.py",
    "tests/gateway/test_context_dump.py",
    "tests/gateway/test_startup_context.py",
    "tests/gateway/test_tool_policy.py",
    "tests/gateway/test_discord_context_smoke.py",
    "tests/hermes_cli/test_dev.py",
)
CONTEXT_TESTS = (
    "tests/gateway/test_discord_slash_auth.py",
    "tests/gateway/test_discord_slash_commands.py",
    "tests/gateway/test_discord_context_smoke.py",
    "tests/gateway/test_context_dump.py",
    "tests/gateway/test_startup_context.py",
    "tests/gateway/test_tool_policy.py",
    "tests/hermes_cli/test_dev.py",
)
RSYNC_EXCLUDES = (
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.bak",
    "*.bak.*",
    "*.message.txt",
    "*.dump.json",
    ".env",
    "venv",
    "node_modules",
)


@dataclass(frozen=True)
class CommandPlan:
    """A shell command with a human-readable label."""

    label: str
    argv: tuple[str, ...]

    def render(self) -> str:
        return " ".join(_quote_arg(arg) for arg in self.argv)


@dataclass(frozen=True)
class DumpParseResult:
    """Summary produced from a Hermes `.message.txt` context dump."""

    raw_message_count: int
    tool_schema_count: int
    metadata_keys: tuple[str, ...]
    rough_message_tokens: int
    rough_tool_tokens: int
    rough_total_tokens: int
    missing_required_sections: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "raw_message_count": self.raw_message_count,
                "tool_schema_count": self.tool_schema_count,
                "metadata_keys": list(self.metadata_keys),
                "rough_message_tokens": self.rough_message_tokens,
                "rough_tool_tokens": self.rough_tool_tokens,
                "rough_total_tokens": self.rough_total_tokens,
                "missing_required_sections": list(self.missing_required_sections),
            },
            indent=2,
            sort_keys=True,
        )


def build_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Register `hermes dev ...` subcommands."""

    parser = subparsers.add_parser(
        "dev",
        help="Developer workflow toolkit",
        description="Run Hermes maintainer checks, odin sync, live smoke tests, and context dump parsing.",
    )
    dev_sub = parser.add_subparsers(dest="dev_command")

    _add_common_dry_run(dev_sub.add_parser("status", help="Show local and odin status"))

    sync = dev_sub.add_parser("sync", help="Rsync this worktree to odin")
    sync.add_argument("--to", choices=("odin",), required=True)
    sync.add_argument(
        "--odin", action="store_true", help="Required acknowledgement for odin mutation"
    )
    sync.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and dry-run rsync without mutation",
    )

    diff = dev_sub.add_parser(
        "diff-odin", help="Show local-vs-odin tracked source differences"
    )
    diff.add_argument(
        "--odin", action="store_true", help="Required acknowledgement for odin access"
    )
    diff.add_argument(
        "--dry-run", action="store_true", help="Print planned command without executing"
    )

    _add_common_dry_run(dev_sub.add_parser("lint", help="Run ruff checks"))
    _add_common_dry_run(
        dev_sub.add_parser("typecheck", help="Run mypy on new context modules")
    )
    _add_common_dry_run(
        dev_sub.add_parser("dead-code", help="Run vulture advisory checks")
    )
    _add_common_dry_run(dev_sub.add_parser("test", help="Run focused developer tests"))

    restart = dev_sub.add_parser("restart", help="Restart hermes-gateway on odin")
    restart.add_argument(
        "--odin", action="store_true", help="Required acknowledgement for odin mutation"
    )
    restart.add_argument(
        "--dry-run", action="store_true", help="Print planned command without executing"
    )

    logs = dev_sub.add_parser("logs", help="Inspect hermes-gateway logs on odin")
    logs.add_argument(
        "--odin", action="store_true", help="Required acknowledgement for odin access"
    )
    logs.add_argument(
        "--dry-run", action="store_true", help="Print planned command without executing"
    )
    logs.add_argument("--lines", type=int, default=200)

    smoke = dev_sub.add_parser(
        "smoke", help="Run live Discord smoke validation on odin"
    )
    smoke.add_argument(
        "--odin", action="store_true", help="Required acknowledgement for odin mutation"
    )
    smoke.add_argument(
        "--dry-run", action="store_true", help="Print planned command without executing"
    )
    smoke.add_argument("--channel-id", default=None)
    smoke.add_argument("--timeout-seconds", type=int, default=60)
    smoke.add_argument("--mode", choices=("auto", "internal", "live"), default="auto")

    dump_parse = dev_sub.add_parser(
        "dump-parse", help="Parse a Hermes .message.txt context dump"
    )
    dump_parse.add_argument("path", type=Path)
    dump_parse.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )

    verify = dev_sub.add_parser(
        "verify", help="Run the ordered developer verification workflow"
    )
    verify.add_argument(
        "--odin", action="store_true", help="Include odin sync/restart/smoke/log checks"
    )
    verify.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ordered plan without executing",
    )

    parser.set_defaults(func=cmd_dev)
    return parser


def cmd_dev(args: argparse.Namespace) -> None:
    command = getattr(args, "dev_command", None)
    if command is None:
        raise SystemExit("Missing dev subcommand. Try `hermes dev --help`.")

    if command == "dump-parse":
        result = parse_context_dump(args.path)
        if args.json:
            print(result.to_json())
        else:
            print_dump_summary(result)
        if result.missing_required_sections:
            raise SystemExit(2)
        return

    if command in {"sync", "restart", "logs", "smoke", "diff-odin"} and not getattr(
        args, "odin", False
    ):
        raise SystemExit(f"`hermes dev {command}` requires --odin.")

    plans = build_plans(args)
    if not plans:
        raise SystemExit(f"No plan for dev command {command!r}.")

    for plan in plans:
        print(f"[{plan.label}] {plan.render()}")
        if getattr(args, "dry_run", False):
            continue
        subprocess.run(plan.argv, check=True)


def build_plans(args: argparse.Namespace) -> list[CommandPlan]:
    command = getattr(args, "dev_command", None)
    if command == "status":
        return _status_plans()
    if command == "sync":
        return [_rsync_plan(dry_run=getattr(args, "dry_run", False))]
    if command == "diff-odin":
        return [
            _ssh_plan(
                "diff-odin",
                "cd {worktree} && git status --short --branch && git diff --stat".format(
                    worktree=ODIN_WORKTREE
                ),
            )
        ]
    if command == "lint":
        return _lint_plans()
    if command == "typecheck":
        return _typecheck_plans()
    if command == "dead-code":
        return _dead_code_plans()
    if command == "test":
        return _test_plans()
    if command == "restart":
        return [_ssh_plan("restart gateway", "systemctl --user restart hermes-gateway")]
    if command == "logs":
        lines = max(1, int(getattr(args, "lines", 200)))
        return [
            _ssh_plan(
                "gateway logs",
                f"journalctl --user -u hermes-gateway -n {lines} --no-pager",
            )
        ]
    if command == "smoke":
        channel_id = getattr(args, "channel_id", None) or os.getenv(
            "HERMES_CONTEXT_VALIDATION_CHANNEL_ID",
            DEFAULT_VALIDATION_CHANNEL_ID,
        )
        timeout = int(getattr(args, "timeout_seconds", 60))
        mode = getattr(args, "mode", "auto")
        smoke_cmd = (
            f"cd {ODIN_WORKTREE} && "
            f"{ODIN_PYTHON} -m gateway.validation.discord_context_smoke "
            f"--channel-id {channel_id} "
            f"--timeout-seconds {timeout} "
            f"--mode {mode}"
        )
        return [_ssh_plan("discord smoke", smoke_cmd)]
    if command == "verify":
        plans: list[CommandPlan] = []
        plans.extend(_lint_plans())
        plans.extend(_typecheck_plans())
        plans.extend(_dead_code_plans())
        plans.extend(_test_plans())
        if getattr(args, "odin", False):
            plans.append(_rsync_plan(dry_run=getattr(args, "dry_run", False)))
            plans.append(
                _ssh_plan("restart gateway", "systemctl --user restart hermes-gateway")
            )
            plans.append(
                _ssh_plan(
                    "discord smoke",
                    "cd {worktree} && {python} -m gateway.validation.discord_context_smoke "
                    "--channel-id {channel_id} --timeout-seconds 60 --mode auto".format(
                        worktree=ODIN_WORKTREE,
                        python=ODIN_PYTHON,
                        channel_id=os.getenv(
                            "HERMES_CONTEXT_VALIDATION_CHANNEL_ID",
                            DEFAULT_VALIDATION_CHANNEL_ID,
                        ),
                    ),
                )
            )
            plans.append(
                _ssh_plan(
                    "gateway logs",
                    "journalctl --user -u hermes-gateway -n 200 --no-pager",
                )
            )
        return plans
    return []


def parse_context_dump(path: Path) -> DumpParseResult:
    text = path.read_text(encoding="utf-8")
    raw_messages = _extract_json_section(
        text, "## Raw API Messages", "## Tool Schemas", default=[]
    )
    tools = _extract_json_section(
        text, "## Tool Schemas", "## Debug Metadata", default=[]
    )
    metadata = _extract_json_section(text, "## Debug Metadata", None, default={})

    missing: list[str] = []
    if not isinstance(raw_messages, list) or not raw_messages:
        missing.append("raw_api_messages")
        raw_messages = []
    if not isinstance(tools, list):
        missing.append("tool_schemas")
        tools = []
    if not isinstance(metadata, dict) or not metadata:
        missing.append("debug_metadata")
        metadata = {}

    if "schema" not in metadata:
        missing.append("metadata.schema")
    if "session_key" not in metadata:
        missing.append("metadata.session_key")

    message_chars = len(json.dumps(raw_messages, ensure_ascii=False, default=str))
    tool_chars = len(json.dumps(tools, ensure_ascii=False, default=str))
    metadata_chars = len(json.dumps(metadata, ensure_ascii=False, default=str))

    return DumpParseResult(
        raw_message_count=len(raw_messages),
        tool_schema_count=len(tools),
        metadata_keys=tuple(sorted(str(k) for k in metadata.keys())),
        rough_message_tokens=_rough_tokens(message_chars),
        rough_tool_tokens=_rough_tokens(tool_chars),
        rough_total_tokens=_rough_tokens(message_chars + tool_chars + metadata_chars),
        missing_required_sections=tuple(missing),
    )


def print_dump_summary(result: DumpParseResult) -> None:
    print(f"Raw API messages: {result.raw_message_count}")
    print(f"Tool schemas: {result.tool_schema_count}")
    print(f"Rough message tokens: {result.rough_message_tokens}")
    print(f"Rough tool tokens: {result.rough_tool_tokens}")
    print(f"Rough total tokens: {result.rough_total_tokens}")
    if result.missing_required_sections:
        print(
            "Missing required sections: " + ", ".join(result.missing_required_sections)
        )
    else:
        print("Missing required sections: none")


def _add_common_dry_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run", action="store_true", help="Print planned command without executing"
    )


def _status_plans() -> list[CommandPlan]:
    return [
        CommandPlan("local status", ("git", "status", "--short", "--branch")),
        _ssh_plan(
            "odin status",
            "cd {worktree} && git status --short --branch; "
            "pgrep -af 'hermes_cli.main gateway run|gateway run' || true; "
            "systemctl --user is-active hermes-gateway || true".format(
                worktree=ODIN_WORKTREE
            ),
        ),
    ]


def _lint_plans() -> list[CommandPlan]:
    return [
        CommandPlan("ruff check", (PYTHON, "-m", "ruff", "check", *TOUCHED_PACKAGES)),
        CommandPlan(
            "ruff gateway run undefined names",
            (PYTHON, "-m", "ruff", "check", "gateway/run.py", "--select", "F821"),
        ),
        CommandPlan(
            "compile gateway runtime",
            (
                PYTHON,
                "-m",
                "py_compile",
                "gateway/run.py",
                "run_agent.py",
                "agent/prompt_builder.py",
            ),
        ),
        CommandPlan(
            "ruff format check",
            (PYTHON, "-m", "ruff", "format", "--check", *TOUCHED_PACKAGES),
        ),
    ]


def _typecheck_plans() -> list[CommandPlan]:
    paths = (
        "gateway/context_layers.py",
        "gateway/startup_context.py",
        "gateway/context_dump.py",
        "gateway/route_banner.py",
        "gateway/validation/discord_context_smoke.py",
        "hermes_cli/dev.py",
        "tests/hermes_cli/test_dev.py",
    )
    return [
        CommandPlan(
            "mypy",
            (
                PYTHON,
                "-m",
                "mypy",
                "--follow-imports=skip",
                "--ignore-missing-imports",
                *paths,
            ),
        )
    ]


def _dead_code_plans() -> list[CommandPlan]:
    return [
        CommandPlan(
            "vulture advisory",
            (
                PYTHON,
                "-m",
                "vulture",
                "gateway/context_layers.py",
                "gateway/startup_context.py",
                "gateway/context_dump.py",
                "gateway/route_banner.py",
                "gateway/tool_policy.py",
                "gateway/validation/discord_context_smoke.py",
                "hermes_cli/dev.py",
                "tests/gateway/test_context_dump.py",
                "tests/gateway/test_startup_context.py",
                "tests/gateway/test_tool_policy.py",
                "tests/gateway/test_discord_context_smoke.py",
                "tests/hermes_cli/test_dev.py",
                "--min-confidence",
                "80",
            ),
        )
    ]


def _test_plans() -> list[CommandPlan]:
    return [CommandPlan("focused pytest", (PYTHON, "-m", "pytest", *CONTEXT_TESTS))]


def _rsync_plan(*, dry_run: bool) -> CommandPlan:
    argv = [
        "rsync",
        "-az",
        "--delete",
        "--rsync-path",
        f"sudo -n -u {ODIN_USER} rsync",
    ]
    if dry_run:
        argv.append("--dry-run")
    for pattern in RSYNC_EXCLUDES:
        argv.extend(("--exclude", pattern))
    argv.extend(("./", f"{ODIN_HOST}:{ODIN_WORKTREE}/"))
    return CommandPlan("rsync to odin", tuple(argv))


def _ssh_plan(label: str, remote_command: str) -> CommandPlan:
    if _running_on_odin():
        return CommandPlan(label, ("bash", "-lc", remote_command))
    return CommandPlan(
        label,
        (
            "ssh",
            ODIN_HOST,
            f"sudo -n -u {ODIN_USER} bash -lc {json.dumps(remote_command)}",
        ),
    )


def _running_on_odin() -> bool:
    try:
        hostname = socket.gethostname().split(".", 1)[0]
    except Exception:
        hostname = ""
    return hostname == ODIN_HOST


def _extract_json_section(
    text: str, start_marker: str, end_marker: str | None, *, default: Any
) -> Any:
    start = text.find(start_marker)
    if start < 0:
        return default
    start += len(start_marker)
    end = text.find(end_marker, start) if end_marker else len(text)
    if end < 0:
        end = len(text)
    section = text[start:end].strip()
    if not section:
        return default
    return json.loads(section)


def _rough_tokens(char_count: int) -> int:
    return max(1, (char_count + 3) // 4) if char_count else 0


def _quote_arg(arg: str) -> str:
    if not arg:
        return "''"
    if all(ch.isalnum() or ch in "-_./:=@" for ch in arg):
        return arg
    return "'" + arg.replace("'", "'\"'\"'") + "'"


__all__ = [
    "CommandPlan",
    "DumpParseResult",
    "build_parser",
    "build_plans",
    "cmd_dev",
    "parse_context_dump",
]

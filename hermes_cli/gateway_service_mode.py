"""Classify persisted gateway launchers before rewriting or starting them.

Only exact command arguments written by Hermes are classified. Anything outside
the supported launcher shapes is unknown and must be preserved by refresh paths.
"""
from __future__ import annotations

import plistlib
import re
import shlex
from pathlib import Path


def _valid_profile_operand(value: str) -> bool:
    """Match the canonical CLI profile grammar before consuming an argv token."""
    from hermes_constants import PROFILE_ID_RE

    # Keep normalization and validation aligned with hermes_cli.main._scan_profile_flag.
    return PROFILE_ID_RE.fullmatch(value.strip().casefold()) is not None

def _mode_launch_module(argv: list[str], *, legacy: bool):
    """Recognize launcher syntax only; never use this as a profile identity proof."""
    import ast
    from hermes_cli import gateway
    from hermes_cli._launchers import runtime_command
    from hermes_cli.gateway_runtime_service_identity import _launch_module, _PYTHON_EXECUTABLE

    launch = _launch_module(argv, legacy=legacy)
    if launch is not None:
        return launch
    if (len(argv) < 4 or not _PYTHON_EXECUTABLE.fullmatch(Path(argv[0]).name.lower())
            or argv[1:3] != ["-I", "-c"]):
        return None
    try:
        statements = ast.parse(argv[3]).body
    except (SyntaxError, ValueError):
        return None
    for statement in statements:
        if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.BoolOp):
            continue
        values = statement.value.values
        if not values or not isinstance(values[-1], ast.Constant) or not isinstance(values[-1].value, str):
            continue
        for module in ("hermes_cli.stderr_timestamp", "hermes_cli.main"):
            expected = runtime_command(gateway.PROJECT_ROOT, module=module, python=argv[0], home=values[-1].value)
            if expected[3] == argv[3]:
                return module, argv[4:]
    return None



def _command_mode(argv: list[str], *, live: bool = False) -> str:
    """Classify a persisted launcher, or a live command using the actual run parser."""
    if not argv:
        return "unknown"
    from hermes_cli.gateway_runtime_service_identity import _launchd_command

    try:
        command, wrapped = _launchd_command(argv)
        launch = _mode_launch_module(command, legacy=not wrapped)
        if launch is not None and launch[0] == "hermes_cli.stderr_timestamp":
            args = launch[1]
            if len(args) < 4 or args[0] != "--error-log" or args[2] != "--":
                return "unknown"
            command = args[3:]
            launch = _mode_launch_module(command, legacy=not wrapped)
        if launch is not None:
            if launch[0] != "hermes_cli.main":
                return "unknown"
            args = launch[1]
        elif command[0].rsplit("/", 1)[-1].lower() in {"hermes", "hermes.exe"}:
            args = command[1:]
        else:
            return "unknown"
    except (ValueError, IndexError, OSError):
        return "unknown"

    filtered: list[str] = []
    profiles = 0
    index = 0
    while index < len(args):
        token = args[index]
        if token in {"-p", "--profile"}:
            index += 1
            if index >= len(args) or not _valid_profile_operand(args[index]):
                return "unknown"
            profiles += 1
        elif token.startswith("--profile="):
            if not _valid_profile_operand(token.partition("=")[2]):
                return "unknown"
            profiles += 1
        else:
            filtered.append(token)
        index += 1
    if profiles > 1:
        return "unknown"
    if live:
        return _live_run_mode(filtered)
    if filtered[:2] != ["gateway", "run"]:
        return "unknown"
    options = filtered[2:]
    if "--control-only" in options:
        return "control_only" if options.count("--control-only") == 1 and all(
            item in {"--control-only", "--quiet", "--external-supervisor"} for item in options
        ) else "unknown"
    if options.count("--replace") > 1:
        return "unknown"
    if all(item in {"--quiet", "--external-supervisor", "--replace"} for item in options):
        return "serving"
    return "unknown"


def _live_run_mode(argv: list[str]) -> str:
    """Parse the live CLI's complete grammar without broadening service definitions."""
    import argparse
    from hermes_cli._parser import build_top_level_parser
    from hermes_cli.subcommands.gateway import build_gateway_parser

    if any(arg in {"-h", "--help"} for arg in argv):
        return "unknown"
    parser, subparsers, _ = build_top_level_parser()
    build_gateway_parser(subparsers, cmd_gateway=None, cmd_proxy=None, cmd_gateway_enroll=None)
    gateway_parser = subparsers.choices["gateway"]
    gateway_subparsers = next(action for action in gateway_parser._actions
                              if isinstance(action, argparse._SubParsersAction))
    run_parser = gateway_subparsers.choices["run"]

    def invalid(message):
        raise ValueError(message)

    parser.error = gateway_parser.error = run_parser.error = invalid
    try:
        parsed = parser.parse_args(argv)
    except (ValueError, SystemExit):
        return "unknown"
    if getattr(parsed, "gateway_command", None) != "run":
        return "unknown"
    if parsed.control_only and (parsed.replace or parsed.force):
        return "unknown"
    return "control_only" if parsed.control_only else "serving"


def launchd_definition_mode(path: Path) -> str:
    """Classify plist ``ProgramArguments`` only when optional ``Program`` agrees with argv[0]."""
    try:
        value = plistlib.loads(path.read_bytes())
    except (OSError, ValueError, plistlib.InvalidFileException):
        return "unknown"
    argv = value.get("ProgramArguments") if isinstance(value, dict) else None
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        return "unknown"
    program = value.get("Program")
    if program is not None and (not isinstance(program, str) or not argv or program != argv[0]):
        return "unknown"
    return _command_mode(argv)


def systemd_definition_mode(path: Path) -> str:
    """Classify one literal ``ExecStart`` directive from a unit's ``[Service]`` section."""
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return "unknown"
    section = ""
    commands: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
            continue
        if section != "Service" or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key == "ExecStart":
            if value:
                commands.append(value)
            elif commands:
                commands.clear()
            else:
                return "unknown"
    if len(commands) != 1 or not commands[0] or commands[0][0] in "-+!:":
        return "unknown"
    try:
        argv = shlex.split(commands[0], posix=True)
    except ValueError:
        return "unknown"
    return _command_mode(argv)


def systemd_execstart_property_mode(value: str) -> str:
    """Classify systemd's effective ``ExecStart`` property, including drop-in overrides."""
    match = re.fullmatch(
        r"\{ path=(.*?) ; argv\[\]=(.*?) ; ignore_errors=(?:yes|no) ;(?:.*?)\}", value.strip()
    )
    if not match or any(char in match[2] for char in ("$", "\\", "{", "}")):
        return "unknown"
    try:
        argv = shlex.split(match[2])
    except ValueError:
        return "unknown"
    if not argv or match[1] != argv[0]:
        return "unknown"
    return _command_mode(argv)


def refuse_unsafe_service_action(path: Path, manager: str, action: str) -> bool:
    """Block a start/rewrite when the persisted command cannot be proven serving."""
    mode = systemd_definition_mode(path) if manager == "systemd" else launchd_definition_mode(path)
    if mode == "serving":
        return False
    print(
        f"Refusing to {action} {manager} gateway service: its persisted launcher is "
        f"{mode.replace('_', '-')} and may not be replaced or started as a serving gateway."
    )
    return True

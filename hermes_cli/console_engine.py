"""Safe Hermes Console command engine.

This module backs ``hermes console`` and is intentionally narrower than the
full Hermes CLI. It exposes a curated set of native adapters that can later be
shared by the dashboard console websocket without becoming a raw shell.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import functools
import importlib
import io
import json
import shlex
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Literal, NoReturn, Sequence
from urllib.parse import urlparse

from tools.ansi_strip import strip_ansi as _strip_ansi


ConsoleStatus = Literal["ok", "error", "confirm_required", "exit", "clear"]
ConsoleContext = Literal["local", "hosted"]
ALL_CONTEXTS: frozenset[ConsoleContext] = frozenset({"local", "hosted"})
LOCAL_CONTEXTS: frozenset[ConsoleContext] = frozenset({"local"})


class ConsoleCommandError(RuntimeError):
    """User-facing console command failure."""


@dataclass(frozen=True)
class ConsoleResult:
    status: ConsoleStatus
    output: str = ""
    command: str = ""
    confirmation_message: str = ""


@dataclass(frozen=True)
class ConsoleCommand:
    path: tuple[str, ...]
    usage: str
    summary: str
    handler: Callable[["HermesConsoleEngine", list[str]], str]
    mutating: bool = False
    confirmation: str = ""
    contexts: frozenset[ConsoleContext] = LOCAL_CONTEXTS


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:  # pragma: no cover - argparse hook
        raise ConsoleCommandError(f"{self.prog}: {message}")


def _capture_output(fn: Callable[[], object]) -> str:
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            result = fn()
            if isinstance(result, int) and result:
                raise SystemExit(result)
        except SystemExit as exc:
            code = int(exc.code or 0)
    text = stdout.getvalue() + stderr.getvalue()
    if code:
        raise ConsoleCommandError(text.strip() or f"Command exited with status {code}")
    return text.rstrip()


def _is_status_footer_rule(line: str) -> bool:
    stripped = _strip_ansi(line).strip()
    if len(stripped) < 8:
        return False
    normalized = stripped.replace("\u2500", "-")
    return set(normalized) <= {"-"}


def _strip_console_status_footer(text: str) -> str:
    lines = text.splitlines()
    while lines and not _strip_ansi(lines[-1]).strip():
        lines.pop()
    if len(lines) < 2:
        return text.rstrip()

    last = _strip_ansi(lines[-1]).strip()
    prev = _strip_ansi(lines[-2]).strip()
    if not (
        prev.startswith("Run 'hermes doctor'")
        and last.startswith("Run 'hermes setup'")
    ):
        return text.rstrip()

    lines = lines[:-2]
    while lines and not _strip_ansi(lines[-1]).strip():
        lines.pop()
    if lines and _is_status_footer_rule(lines[-1]):
        lines.pop()
    return "\n".join(lines).rstrip()


def _table_summary(summary: str, *, limit: int = 76) -> str:
    summary = " ".join(summary.split())
    if len(summary) <= limit:
        return summary
    return f"{summary[: limit - 3].rstrip()}..."


def _split_line(line: str) -> list[str]:
    try:
        return shlex.split(line, comments=False, posix=True)
    except ValueError as exc:
        raise ConsoleCommandError(f"Could not parse command: {exc}") from exc


def _contains_shell_syntax(line: str, tokens: Sequence[str]) -> bool:
    if "$(" in line or "`" in line:
        return True
    shell_tokens = {"|", "||", "&", "&&", ";", ">", ">>", "<", "<<", "2>", "2>>"}
    if any(token in shell_tokens for token in tokens):
        return True
    return any(ch in line for ch in "|<>;")


def _format_sessions(sessions: Sequence[dict]) -> str:
    if not sessions:
        return "No sessions found."
    lines = [f"{'ID':<32} {'Source':<12} {'Msgs':>5}  Title / Preview"]
    lines.append("-" * 82)
    for session in sessions:
        sid = str(session.get("id") or "")[:32]
        source = str(session.get("source") or "-")[:12]
        messages = session.get("message_count") or 0
        title = session.get("title") or session.get("preview") or ""
        title = str(title).replace("\n", " ")[:60]
        lines.append(f"{sid:<32} {source:<12} {messages:>5}  {title}")
    return "\n".join(lines)


def _format_job(job: dict, action: str) -> str:
    job_id = job.get("id") or job.get("job_id") or "?"
    name = job.get("name") or "(unnamed)"
    state = job.get("state") or ("scheduled" if job.get("enabled", True) else "paused")
    return f"{action} job: {name} ({job_id}) [{state}]"


EXPECTED_HOSTED_PATHS: tuple[tuple[str, ...], ...] = (
    ("status",),
    ("doctor",),
    ("logs",),
    ("version",),
    ("prompt-size",),
    ("insights",),
    ("security", "audit"),
    ("portal", "info"),
    ("portal", "tools"),
    ("send",),
    ("config", "show"),
    ("config", "path"),
    ("config", "env-path"),
    ("config", "check"),
    ("config", "migrate"),
    ("config", "set"),
    ("sessions", "list"),
    ("sessions", "stats"),
    ("sessions", "export"),
    ("sessions", "rename"),
    ("sessions", "optimize"),
    ("sessions", "repair"),
    ("cron", "list"),
    ("cron", "status"),
    ("cron", "create"),
    ("cron", "edit"),
    ("cron", "pause"),
    ("cron", "resume"),
    ("cron", "run"),
    ("cron", "remove"),
    ("cron", "tick"),
    ("profile",),
    ("profile", "list"),
    ("profile", "show"),
    ("profile", "info"),
    ("tools", "list"),
    ("tools", "enable"),
    ("tools", "disable"),
    ("tools", "post-setup"),
    ("skills", "browse"),
    ("skills", "search"),
    ("skills", "inspect"),
    ("skills", "list"),
    ("skills", "check"),
    ("skills", "list-modified"),
    ("skills", "diff"),
    ("skills", "install"),
    ("skills", "update"),
    ("skills", "audit"),
    ("skills", "uninstall"),
    ("skills", "reset"),
    ("skills", "opt-in"),
    ("skills", "opt-out"),
    ("skills", "repair-official"),
    ("skills", "snapshot", "export"),
    ("skills", "tap", "list"),
    ("mcp", "list"),
    ("mcp", "catalog"),
    ("mcp", "test"),
    ("mcp", "add"),
    ("mcp", "remove"),
    ("mcp", "install"),
    ("mcp", "login"),
    ("mcp", "reauth"),
    ("mcp", "configure"),
    ("mcp", "picker"),
    ("memory", "status"),
    ("auth", "list"),
    ("auth", "status"),
    ("auth", "reset"),
    ("auth", "spotify", "status"),
    ("pairing", "list"),
    ("pairing", "approve"),
    ("pairing", "revoke"),
    ("pairing", "clear-pending"),
    ("webhook", "list"),
    ("webhook", "subscribe"),
    ("webhook", "remove"),
    ("webhook", "test"),
)


def _parser_root() -> tuple[_ArgumentParser, argparse._SubParsersAction]:
    parser = _ArgumentParser(prog="hermes", add_help=False)
    subparsers = parser.add_subparsers(dest="_console_command")
    return parser, subparsers


def _subparser_actions(parser: argparse.ArgumentParser) -> list[argparse._SubParsersAction]:
    return [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]


def _choice_help(action: argparse._SubParsersAction, name: str) -> str:
    for choice in action._choices_actions:
        if getattr(choice, "dest", None) == name or getattr(choice, "metavar", None) == name:
            help_text = getattr(choice, "help", None)
            if help_text and help_text is not argparse.SUPPRESS:
                return str(help_text)
    return ""


def _clean_summary(text: str | None) -> str:
    if not text:
        return ""
    if text is argparse.SUPPRESS:
        return ""
    summary = " ".join(str(text).split())
    if not summary:
        return ""
    if summary.startswith("Run `hermes "):
        return ""
    return summary


def _summaries_from_parser(parser: argparse.ArgumentParser) -> dict[tuple[str, ...], str]:
    summaries: dict[tuple[str, ...], str] = {}

    def walk(current: argparse.ArgumentParser, path: tuple[str, ...]) -> None:
        for action in _subparser_actions(current):
            for name, child in action.choices.items():
                child_path = (*path, name)
                summary = _clean_summary(_choice_help(action, name)) or _clean_summary(
                    child.description
                )
                if summary:
                    summaries.setdefault(child_path, summary)
                walk(child, child_path)

    walk(parser, ())
    return summaries


def _noop_console_command(_args: argparse.Namespace) -> None:
    return None


# The CLI surface these helpers reflect is process-static: they import a
# subcommand module and build a throwaway argparse tree purely to extract help
# summaries. Nothing about the result changes across engine instances, but the
# dashboard opens a fresh HermesConsoleEngine per /api/console connection, so
# without memoization every reconnect re-imports + re-parses the whole surface.
# Cache by args (all hashable strings); callers only read the returned map.
@functools.lru_cache(maxsize=None)
def _extracted_summaries(
    module_name: str,
    builder_name: str,
    main_handler_name: str,
) -> dict[tuple[str, ...], str]:
    try:
        parser, subparsers = _parser_root()
        module = importlib.import_module(module_name)
        builder = getattr(module, builder_name)
        builder(subparsers, **{main_handler_name: _noop_console_command})
        return _summaries_from_parser(parser)
    except Exception:
        return {}


@functools.lru_cache(maxsize=None)
def _registered_summaries(
    root: str,
    module_name: str,
    register_name: str,
) -> dict[tuple[str, ...], str]:
    try:
        parser, subparsers = _parser_root()
        module = importlib.import_module(module_name)
        top_parser = subparsers.add_parser(root)
        register = getattr(module, register_name)
        register(top_parser)
        return _summaries_from_parser(parser)
    except Exception:
        return {}


@functools.lru_cache(maxsize=None)
def _builder_summaries(
    module_name: str,
    builder_name: str,
) -> dict[tuple[str, ...], str]:
    try:
        parser, subparsers = _parser_root()
        module = importlib.import_module(module_name)
        getattr(module, builder_name)(subparsers)
        return _summaries_from_parser(parser)
    except Exception:
        return {}


@functools.lru_cache(maxsize=None)
def _adder_summaries(module_name: str, add_name: str) -> dict[tuple[str, ...], str]:
    try:
        parser, subparsers = _parser_root()
        module = importlib.import_module(module_name)
        getattr(module, add_name)(subparsers)
        return _summaries_from_parser(parser)
    except Exception:
        return {}


def _invoke_namespace(args: argparse.Namespace) -> object:
    func = getattr(args, "func", None)
    if not callable(func):
        raise ConsoleCommandError("No handler is available for that console command.")
    return func(args)


def _set_attrs(args: argparse.Namespace, **attrs: object) -> argparse.Namespace:
    for name, value in attrs.items():
        setattr(args, name, value)
    return args


def _dispatch_extracted_subcommand(
    *,
    root: str,
    fixed: Sequence[str],
    args: Sequence[str],
    module_name: str,
    builder_name: str,
    main_handler_name: str,
    console_context: ConsoleContext,
    namespace_update: Callable[[argparse.Namespace, ConsoleContext], None] | None = None,
) -> str:
    parser, subparsers = _parser_root()
    module = importlib.import_module(module_name)
    main_module = importlib.import_module("hermes_cli.main")
    builder = getattr(module, builder_name)
    main_handler = getattr(main_module, main_handler_name)
    builder(subparsers, **{main_handler_name: main_handler})
    namespace = parser.parse_args([root, *fixed, *args])
    if namespace_update:
        namespace_update(namespace, console_context)
    return _capture_output(lambda: _invoke_namespace(namespace))


def _dispatch_registered_subcommand(
    *,
    root: str,
    fixed: Sequence[str],
    args: Sequence[str],
    module_name: str,
    register_name: str,
    handler_name: str | None = None,
    console_context: ConsoleContext,
    namespace_update: Callable[[argparse.Namespace, ConsoleContext], None] | None = None,
) -> str:
    parser, subparsers = _parser_root()
    module = importlib.import_module(module_name)
    top_parser = subparsers.add_parser(root)
    register = getattr(module, register_name)
    register(top_parser)
    if handler_name:
        top_parser.set_defaults(func=getattr(module, handler_name))
    namespace = parser.parse_args([root, *fixed, *args])
    if namespace_update:
        namespace_update(namespace, console_context)
    return _capture_output(lambda: _invoke_namespace(namespace))


def _dispatch_builder_subcommand(
    *,
    root: str,
    fixed: Sequence[str],
    args: Sequence[str],
    module_name: str,
    builder_name: str,
    main_handler_name: str,
    console_context: ConsoleContext,
    namespace_update: Callable[[argparse.Namespace, ConsoleContext], None] | None = None,
) -> str:
    parser, subparsers = _parser_root()
    module = importlib.import_module(module_name)
    main_module = importlib.import_module("hermes_cli.main")
    top_parser = getattr(module, builder_name)(subparsers)
    top_parser.set_defaults(func=getattr(main_module, main_handler_name))
    namespace = parser.parse_args([root, *fixed, *args])
    if namespace_update:
        namespace_update(namespace, console_context)
    return _capture_output(lambda: _invoke_namespace(namespace))


def _dispatch_adder_subcommand(
    *,
    root: str,
    fixed: Sequence[str],
    args: Sequence[str],
    module_name: str,
    add_name: str,
    console_context: ConsoleContext,
    namespace_update: Callable[[argparse.Namespace, ConsoleContext], None] | None = None,
) -> str:
    parser, subparsers = _parser_root()
    module = importlib.import_module(module_name)
    getattr(module, add_name)(subparsers)
    namespace = parser.parse_args([root, *fixed, *args])
    if namespace_update:
        namespace_update(namespace, console_context)
    return _capture_output(lambda: _invoke_namespace(namespace))


def _extracted_handler(
    root: str,
    fixed: Sequence[str],
    module_name: str,
    builder_name: str,
    main_handler_name: str,
    namespace_update: Callable[[argparse.Namespace, ConsoleContext], None] | None = None,
) -> Callable[["HermesConsoleEngine", list[str]], str]:
    def handler(_engine: HermesConsoleEngine, args: list[str]) -> str:
        return _dispatch_extracted_subcommand(
            root=root,
            fixed=fixed,
            args=args,
            module_name=module_name,
            builder_name=builder_name,
            main_handler_name=main_handler_name,
            console_context=_engine.context,
            namespace_update=namespace_update,
        )

    return handler


def _registered_handler(
    root: str,
    fixed: Sequence[str],
    module_name: str,
    register_name: str,
    handler_name: str | None = None,
    namespace_update: Callable[[argparse.Namespace, ConsoleContext], None] | None = None,
) -> Callable[["HermesConsoleEngine", list[str]], str]:
    def handler(_engine: HermesConsoleEngine, args: list[str]) -> str:
        return _dispatch_registered_subcommand(
            root=root,
            fixed=fixed,
            args=args,
            module_name=module_name,
            register_name=register_name,
            handler_name=handler_name,
            console_context=_engine.context,
            namespace_update=namespace_update,
        )

    return handler


def _builder_handler(
    root: str,
    fixed: Sequence[str],
    module_name: str,
    builder_name: str,
    main_handler_name: str,
    namespace_update: Callable[[argparse.Namespace, ConsoleContext], None] | None = None,
) -> Callable[["HermesConsoleEngine", list[str]], str]:
    def handler(_engine: HermesConsoleEngine, args: list[str]) -> str:
        return _dispatch_builder_subcommand(
            root=root,
            fixed=fixed,
            args=args,
            module_name=module_name,
            builder_name=builder_name,
            main_handler_name=main_handler_name,
            console_context=_engine.context,
            namespace_update=namespace_update,
        )

    return handler


def _adder_handler(
    root: str,
    fixed: Sequence[str],
    module_name: str,
    add_name: str,
    namespace_update: Callable[[argparse.Namespace, ConsoleContext], None] | None = None,
) -> Callable[["HermesConsoleEngine", list[str]], str]:
    def handler(_engine: HermesConsoleEngine, args: list[str]) -> str:
        return _dispatch_adder_subcommand(
            root=root,
            fixed=fixed,
            args=args,
            module_name=module_name,
            add_name=add_name,
            console_context=_engine.context,
            namespace_update=namespace_update,
        )

    return handler


def _register_command_family(
    engine: "HermesConsoleEngine",
    *,
    root: str,
    paths: Iterable[Sequence[str]],
    handler_factory: Callable[[Sequence[str]], Callable[["HermesConsoleEngine", list[str]], str]],
    mutating: Iterable[Sequence[str]] = (),
    hosted: Iterable[Sequence[str]] = (),
    summary: str = "",
    summaries: dict[tuple[str, ...], str] | None = None,
    confirmation: str = "",
) -> None:
    mutating_paths = {tuple(path) for path in mutating}
    hosted_paths = {tuple(path) for path in hosted}
    for child_path in paths:
        child_key = tuple(child_path)
        full_path = (root, *tuple(child_path))
        usage = " ".join(full_path)
        command_summary = summary or (summaries or {}).get(full_path) or f"Run `hermes {usage}`."
        engine.register(
            full_path,
            usage,
            command_summary,
            handler_factory(tuple(child_path)),
            mutating=child_key in mutating_paths,
            confirmation=confirmation or f"Run `hermes {usage}`?",
            contexts=ALL_CONTEXTS if child_key in hosted_paths else LOCAL_CONTEXTS,
        )


class HermesConsoleEngine:
    """Curated line-command executor for Hermes Console."""

    def __init__(self, *, output_limit: int = 20000, context: ConsoleContext = "local"):
        if context not in ALL_CONTEXTS:
            raise ValueError(f"Unknown console context: {context}")
        self.context = context
        self.output_limit = output_limit
        self.history: list[str] = []
        self.commands: dict[tuple[str, ...], ConsoleCommand] = {}
        self._register_defaults()

    def execute(self, line: str, *, confirmed: bool = False) -> ConsoleResult:
        raw_line = line.strip()
        if not raw_line:
            return ConsoleResult("ok")

        try:
            tokens = _split_line(raw_line)
            if tokens and tokens[0] == "hermes":
                tokens = tokens[1:]
            if not tokens:
                return self._help_result()

            if _contains_shell_syntax(raw_line, tokens):
                raise ConsoleCommandError(
                    "Hermes Console does not run shell syntax. Use one supported "
                    "Hermes command at a time."
                )

            builtin = self._execute_builtin(tokens)
            if builtin is not None:
                if raw_line not in {"history", "clear"}:
                    self.history.append(raw_line)
                return builtin

            command, args = self._resolve_command(tokens)
            if command.mutating and not confirmed:
                return ConsoleResult(
                    "confirm_required",
                    command=raw_line,
                    confirmation_message=command.confirmation
                    or f"Run `{command.usage}`?",
                )

            output = command.handler(self, args).rstrip()
            output = self._cap_output(output)
            self.history.append(raw_line)
            return ConsoleResult("ok", output=output, command=raw_line)
        except ConsoleCommandError as exc:
            return ConsoleResult("error", output=str(exc).strip(), command=raw_line)

    def help_text(self, subject: str | None = None) -> str:
        if subject:
            tokens = subject.split()
            command, _args = self._resolve_command(tokens)
            return f"{command.usage}\n{command.summary}"

        lines = [
            "Hermes Console",
            "",
            "Supported commands:",
        ]
        for command in sorted(self.commands.values(), key=lambda c: c.usage):
            if self.context not in command.contexts:
                continue
            marker = " *" if command.mutating else "  "
            lines.append(f"{marker} {command.usage:<32} {_table_summary(command.summary)}")
        lines.extend(
            [
                "",
                "* requires confirmation",
                "Built-ins: help, help <command>, history, clear, exit, quit",
            ]
        )
        return "\n".join(lines)

    def _register_defaults(self) -> None:
        self.register(("status",), "status", "Show Hermes component status.", _status, contexts=ALL_CONTEXTS)
        self.register(("doctor",), "doctor", "Run diagnostics without auto-fix.", _doctor, contexts=ALL_CONTEXTS)
        self.register(("logs",), "logs [name] [-n N]", "Show recent Hermes logs.", _logs, contexts=ALL_CONTEXTS)
        self.register(("sessions", "list"), "sessions list [--limit N]", "List recent sessions.", _sessions_list, contexts=ALL_CONTEXTS)
        self.register(("sessions", "stats"), "sessions stats", "Show session store statistics.", _sessions_stats, contexts=ALL_CONTEXTS)
        self.register(("config", "show"), "config show", "Show current configuration.", _config_show, contexts=ALL_CONTEXTS)
        self.register(("config", "path"), "config path", "Print config.yaml path.", _config_path, contexts=ALL_CONTEXTS)
        self.register(
            ("config", "set"),
            "config set <key> <value>",
            "Set a configuration value.",
            _config_set,
            mutating=True,
            confirmation="Update Hermes configuration?",
            contexts=ALL_CONTEXTS,
        )
        self.register(("cron", "list"), "cron list [--all]", "List scheduled jobs.", _cron_list, contexts=ALL_CONTEXTS)
        self.register(("cron", "status"), "cron status", "Show cron scheduler status.", _cron_status, contexts=ALL_CONTEXTS)
        self.register(
            ("cron", "pause"),
            "cron pause <job>",
            "Pause a scheduled job.",
            _cron_pause,
            mutating=True,
            confirmation="Pause this cron job?",
            contexts=ALL_CONTEXTS,
        )
        self.register(
            ("cron", "resume"),
            "cron resume <job>",
            "Resume a paused cron job.",
            _cron_resume,
            mutating=True,
            confirmation="Resume this cron job?",
            contexts=ALL_CONTEXTS,
        )
        self.register(
            ("cron", "run"),
            "cron run <job>",
            "Run a job on the next scheduler tick.",
            _cron_run,
            mutating=True,
            confirmation="Trigger this cron job?",
            contexts=ALL_CONTEXTS,
        )
        self._register_broad_cli_surface()

    def _register_broad_cli_surface(self) -> None:
        """Register non-admin CLI commands that are safe for Hermes Console."""

        extracted = {
            "version": (
                "hermes_cli.subcommands.version",
                "build_version_parser",
                "cmd_version",
                [()],
                set(),
            ),
            "dump": (
                "hermes_cli.subcommands.dump",
                "build_dump_parser",
                "cmd_dump",
                [()],
                set(),
            ),
            "debug": (
                "hermes_cli.subcommands.debug",
                "build_debug_parser",
                "cmd_debug",
                [("share",), ("delete",)],
                {("share",), ("delete",)},
            ),
            "prompt-size": (
                "hermes_cli.subcommands.prompt_size",
                "build_prompt_size_parser",
                "cmd_prompt_size",
                [()],
                set(),
            ),
            "insights": (
                "hermes_cli.subcommands.insights",
                "build_insights_parser",
                "cmd_insights",
                [()],
                set(),
            ),
            "security": (
                "hermes_cli.subcommands.security",
                "build_security_parser",
                "cmd_security",
                [("audit",)],
                set(),
            ),
            "backup": (
                "hermes_cli.subcommands.backup",
                "build_backup_parser",
                "cmd_backup",
                [()],
                {()},
            ),
            "import": (
                "hermes_cli.subcommands.import_cmd",
                "build_import_cmd_parser",
                "cmd_import",
                [()],
                {()},
            ),
            "config": (
                "hermes_cli.subcommands.config",
                "build_config_parser",
                "cmd_config",
                [("env-path",), ("check",)],
                set(),
            ),
            "tools": (
                "hermes_cli.subcommands.tools",
                "build_tools_parser",
                "cmd_tools",
                [("list",), ("enable",), ("disable",), ("post-setup",)],
                {("enable",), ("disable",), ("post-setup",)},
            ),
            "plugins": (
                "hermes_cli.subcommands.plugins",
                "build_plugins_parser",
                "cmd_plugins",
                [("list",), ("enable",), ("disable",), ("install",), ("update",), ("remove",)],
                {("enable",), ("disable",), ("install",), ("update",), ("remove",)},
            ),
            "skills": (
                "hermes_cli.subcommands.skills",
                "build_skills_parser",
                "cmd_skills",
                [
                    ("browse",),
                    ("search",),
                    ("inspect",),
                    ("list",),
                    ("check",),
                    ("list-modified",),
                    ("diff",),
                    ("install",),
                    ("update",),
                    ("audit",),
                    ("uninstall",),
                    ("reset",),
                    ("opt-in",),
                    ("opt-out",),
                    ("repair-official",),
                    ("snapshot", "export"),
                    ("snapshot", "import"),
                    ("tap", "list"),
                    ("tap", "add"),
                    ("tap", "remove"),
                ],
                {
                    ("install",),
                    ("update",),
                    ("audit",),
                    ("uninstall",),
                    ("reset",),
                    ("opt-in",),
                    ("opt-out",),
                    ("repair-official",),
                    ("snapshot", "export"),
                    ("snapshot", "import"),
                    ("tap", "add"),
                    ("tap", "remove"),
                },
            ),
            "mcp": (
                "hermes_cli.subcommands.mcp",
                "build_mcp_parser",
                "cmd_mcp",
                [
                    ("list",),
                    ("catalog",),
                    ("test",),
                    ("add",),
                    ("remove",),
                    ("install",),
                    ("login",),
                    ("reauth",),
                    ("configure",),
                    ("picker",),
                ],
                {
                    ("add",),
                    ("remove",),
                    ("install",),
                    ("login",),
                    ("reauth",),
                    ("configure",),
                    ("picker",),
                },
            ),
            "memory": (
                "hermes_cli.subcommands.memory",
                "build_memory_parser",
                "cmd_memory",
                [("status",), ("off",), ("reset",)],
                {("off",), ("reset",)},
            ),
            "auth": (
                "hermes_cli.subcommands.auth",
                "build_auth_parser",
                "cmd_auth",
                [
                    ("list",),
                    ("status",),
                    ("reset",),
                    ("add",),
                    ("remove",),
                    ("logout",),
                    ("spotify", "status"),
                    ("spotify", "login"),
                    ("spotify", "logout"),
                ],
                {
                    ("reset",),
                    ("add",),
                    ("remove",),
                    ("logout",),
                    ("spotify", "login"),
                    ("spotify", "logout"),
                },
            ),
            "pairing": (
                "hermes_cli.subcommands.pairing",
                "build_pairing_parser",
                "cmd_pairing",
                [("list",), ("approve",), ("revoke",), ("clear-pending",)],
                {("approve",), ("revoke",), ("clear-pending",)},
            ),
            "webhook": (
                "hermes_cli.subcommands.webhook",
                "build_webhook_parser",
                "cmd_webhook",
                [("list",), ("subscribe",), ("remove",), ("test",)],
                {("subscribe",), ("remove",)},
            ),
            "hooks": (
                "hermes_cli.subcommands.hooks",
                "build_hooks_parser",
                "cmd_hooks",
                [("list",), ("test",), ("doctor",), ("revoke",)],
                {("test",), ("doctor",), ("revoke",)},
            ),
            "slack": (
                "hermes_cli.subcommands.slack",
                "build_slack_parser",
                "cmd_slack",
                [("manifest",)],
                set(),
            ),
            "profile": (
                "hermes_cli.subcommands.profile",
                "build_profile_parser",
                "cmd_profile",
                [
                    ("list",),
                    ("show",),
                    ("info",),
                    ("create",),
                    ("use",),
                    ("describe",),
                    ("rename",),
                    ("delete",),
                    ("export",),
                    ("import",),
                    ("install",),
                    ("update",),
                ],
                {
                    ("create",),
                    ("use",),
                    ("describe",),
                    ("rename",),
                    ("delete",),
                    ("export",),
                    ("import",),
                    ("install",),
                    ("update",),
                },
            ),
            "cron": (
                "hermes_cli.subcommands.cron",
                "build_cron_parser",
                "cmd_cron",
                [("create",), ("edit",), ("remove",), ("tick",)],
                {("create",), ("edit",), ("remove",), ("tick",)},
            ),
        }

        for root, (module, builder, main_handler, paths, mutating) in extracted.items():
            summaries = _extracted_summaries(module, builder, main_handler)
            _register_command_family(
                self,
                root=root,
                paths=paths,
                mutating=mutating,
                summaries=summaries,
                handler_factory=lambda fixed, root=root, module=module, builder=builder, main_handler=main_handler: _extracted_handler(
                    root,
                    fixed,
                    module,
                    builder,
                    main_handler,
                    namespace_update=_apply_confirmed_defaults,
                ),
            )

        self.register(
            ("config", "migrate"),
            "config migrate",
            "Update config with new options.",
            _config_migrate,
            mutating=True,
            confirmation="Update Hermes configuration with missing defaults?",
        )
        self.register(
            ("sessions", "export"),
            "sessions export <output> [--source SOURCE] [--session-id ID]",
            "Export sessions to JSONL.",
            _sessions_export,
            mutating=True,
            confirmation="Export session data?",
        )
        self.register(
            ("sessions", "rename"),
            "sessions rename <session> <title>",
            "Rename a session.",
            _sessions_rename,
            mutating=True,
            confirmation="Rename this session?",
        )
        self.register(
            ("sessions", "optimize"),
            "sessions optimize",
            "Optimize the session store.",
            _sessions_optimize,
            mutating=True,
            confirmation="Optimize the session database?",
        )
        self.register(
            ("sessions", "repair"),
            "sessions repair [--check-only] [--no-backup]",
            "Repair a malformed session database schema.",
            _sessions_repair,
            mutating=True,
            confirmation="Repair the session database?",
        )

        self.register(
            ("profile",),
            "profile",
            "Show active profile status.",
            _profile_status,
        )
        self.register(
            ("send",),
            "send --to <target> <message>",
            "Send a message to a configured platform.",
            _adder_handler("send", (), "hermes_cli.send_cmd", "register_send_subparser"),
            mutating=True,
            confirmation="Send this message?",
        )

        portal_paths = [("info",), ("tools",)]
        _register_command_family(
            self,
            root="portal",
            paths=portal_paths,
            summaries=_adder_summaries("hermes_cli.portal_cli", "add_parser"),
            handler_factory=lambda fixed: _adder_handler(
                "portal",
                fixed,
                "hermes_cli.portal_cli",
                "add_parser",
            ),
        )

        _register_command_family(
            self,
            root="project",
            paths=[
                ("list",),
                ("show",),
                ("create",),
                ("add-folder",),
                ("remove-folder",),
                ("rename",),
                ("set-primary",),
                ("use",),
                ("archive",),
                ("restore",),
                ("bind-board",),
            ],
            summaries=_builder_summaries("hermes_cli.projects_cmd", "build_parser"),
            mutating=[
                ("create",),
                ("add-folder",),
                ("remove-folder",),
                ("rename",),
                ("set-primary",),
                ("use",),
                ("archive",),
                ("restore",),
                ("bind-board",),
            ],
            handler_factory=lambda fixed: _builder_handler(
                "project",
                fixed,
                "hermes_cli.projects_cmd",
                "build_parser",
                "cmd_project",
            ),
        )

        _register_command_family(
            self,
            root="kanban",
            paths=[
                ("init",),
                ("boards", "list"),
                ("boards", "create"),
                ("boards", "rm"),
                ("boards", "switch"),
                ("boards", "current"),
                ("boards", "rename"),
                ("boards", "set-workdir"),
                ("create",),
                ("list",),
                ("show",),
                ("assign",),
                ("reclaim",),
                ("reassign",),
                ("diagnose",),
                ("link",),
                ("unlink",),
                ("claim",),
                ("comment",),
                ("complete",),
                ("edit",),
                ("block",),
                ("schedule",),
                ("unblock",),
                ("promote",),
                ("archive",),
                ("stats",),
                ("runs",),
                ("heartbeat",),
                ("assignments",),
                ("context",),
            ],
            summaries=_builder_summaries("hermes_cli.kanban", "build_parser"),
            mutating=[
                ("init",),
                ("boards", "create"),
                ("boards", "rm"),
                ("boards", "switch"),
                ("boards", "rename"),
                ("boards", "set-workdir"),
                ("create",),
                ("assign",),
                ("reclaim",),
                ("reassign",),
                ("link",),
                ("unlink",),
                ("claim",),
                ("comment",),
                ("complete",),
                ("edit",),
                ("block",),
                ("schedule",),
                ("unblock",),
                ("promote",),
                ("archive",),
            ],
            handler_factory=lambda fixed: _builder_handler(
                "kanban",
                fixed,
                "hermes_cli.kanban",
                "build_parser",
                "cmd_kanban",
            ),
        )

        registered = {
            "bundles": (
                "hermes_cli.bundles",
                "register_cli",
                "bundles_command",
                [("list",), ("show",), ("create",), ("delete",), ("reload",)],
                {("create",), ("delete",), ("reload",)},
            ),
            "checkpoints": (
                "hermes_cli.checkpoints",
                "register_cli",
                None,
                [("status",), ("list",), ("prune",), ("clear",), ("clear-legacy",)],
                {("prune",), ("clear",), ("clear-legacy",)},
            ),
            "curator": (
                "hermes_cli.curator",
                "register_cli",
                None,
                [
                    ("status",),
                    ("run",),
                    ("pause",),
                    ("resume",),
                    ("pin",),
                    ("unpin",),
                    ("restore",),
                    ("list-archived",),
                    ("archive",),
                    ("prune",),
                    ("backup",),
                    ("rollback",),
                ],
                {
                    ("run",),
                    ("pause",),
                    ("resume",),
                    ("pin",),
                    ("unpin",),
                    ("restore",),
                    ("archive",),
                    ("prune",),
                    ("backup",),
                    ("rollback",),
                },
            ),
            "pets": (
                "hermes_cli.pets",
                "register_cli",
                None,
                [("list",), ("install",), ("select",), ("show",), ("off",), ("scale",), ("remove",), ("doctor",)],
                {("install",), ("select",), ("off",), ("scale",), ("remove",)},
            ),
        }
        for root, (module, register, handler_name, paths, mutating) in registered.items():
            summaries = _registered_summaries(root, module, register)
            _register_command_family(
                self,
                root=root,
                paths=paths,
                mutating=mutating,
                summaries=summaries,
                handler_factory=lambda fixed, root=root, module=module, register=register, handler_name=handler_name: _registered_handler(
                    root,
                    fixed,
                    module,
                    register,
                    handler_name=handler_name,
                    namespace_update=_apply_confirmed_defaults,
                ),
            )

        self._mark_hosted(EXPECTED_HOSTED_PATHS)

    def register(
        self,
        path: Iterable[str],
        usage: str,
        summary: str,
        handler: Callable[["HermesConsoleEngine", list[str]], str],
        *,
        mutating: bool = False,
        confirmation: str = "",
        contexts: Iterable[ConsoleContext] = LOCAL_CONTEXTS,
    ) -> None:
        key = tuple(path)
        self.commands[key] = ConsoleCommand(
            path=key,
            usage=usage,
            summary=summary,
            handler=handler,
            mutating=mutating,
            confirmation=confirmation,
            contexts=frozenset(contexts),
        )

    def _mark_hosted(self, paths: Iterable[Sequence[str]]) -> None:
        for path in paths:
            key = tuple(path)
            command = self.commands.get(key)
            if command is None:
                raise RuntimeError(f"Hosted console policy references unknown command: {' '.join(key)}")
            self.commands[key] = replace(
                command,
                contexts=command.contexts | frozenset({"hosted"}),
            )

    def _execute_builtin(self, tokens: list[str]) -> ConsoleResult | None:
        head = tokens[0]
        if head == "help":
            subject = " ".join(tokens[1:]).strip() or None
            try:
                return ConsoleResult("ok", output=self.help_text(subject))
            except ConsoleCommandError as exc:
                return ConsoleResult("error", output=str(exc))
        if head == "history":
            output = "\n".join(f"{idx + 1}: {cmd}" for idx, cmd in enumerate(self.history))
            return ConsoleResult("ok", output=output or "No history yet.")
        if head == "clear":
            return ConsoleResult("clear", output="\033[2J\033[H")
        if head in {"exit", "quit"}:
            return ConsoleResult("exit")
        return None

    def _resolve_command(self, tokens: Sequence[str]) -> tuple[ConsoleCommand, list[str]]:
        rejected = self._rejection_for(tokens)
        if rejected:
            raise ConsoleCommandError(rejected)

        for size in range(min(len(tokens), 3), 0, -1):
            key = tuple(tokens[:size])
            command = self.commands.get(key)
            if command:
                if self.context not in command.contexts:
                    raise ConsoleCommandError(
                        f"`hermes {command.usage}` is not available in "
                        f"{self.context} Hermes Console."
                    )
                self._enforce_context_policy(command, list(tokens[size:]))
                return command, list(tokens[size:])

        available = [
            " ".join(path)
            for path, command in self.commands.items()
            if self.context in command.contexts
        ]
        probe = " ".join(tokens[:2]) if len(tokens) > 1 else tokens[0]
        suggestions = difflib.get_close_matches(probe, available, n=3, cutoff=0.45)
        suffix = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ConsoleCommandError(f"Unsupported Hermes Console command: {probe}.{suffix}")

    def _enforce_context_policy(self, command: ConsoleCommand, args: list[str]) -> None:
        if self.context != "hosted":
            return
        _enforce_hosted_line_policy(command.path, args)

    def _rejection_for(self, tokens: Sequence[str]) -> str:
        first = tokens[0]
        if first.startswith("-"):
            return f"{first} is not available in Hermes Console."
        blocked_top = {
            "acp",
            "chat",
            "claw",
            "completion",
            "dashboard",
            "desktop",
            "fallback",
            "gateway",
            "gui",
            "login",
            "logout",
            "model",
            "moa",
            "oneshot",
            "postinstall",
            "proxy",
            "serve",
            "setup",
            "uninstall",
            "update",
            "whatsapp",
            "whatsapp-cloud",
        }
        if first in blocked_top:
            return f"`hermes {first}` is not available in Hermes Console."
        blocked_pairs = {
            ("config", "edit"): "`config edit` opens an editor and is not available in Hermes Console.",
            ("mcp", "serve"): "`mcp serve` starts a server and is not available in Hermes Console.",
            ("profile", "alias"): "`profile alias` creates shell wrappers and is not available in Hermes Console.",
            ("skills", "config"): "`skills config` is interactive and is not available in Hermes Console.",
            ("skills", "publish"): "`skills publish` is not available in Hermes Console.",
            ("portal", "login"): "`portal login` is interactive and is not available in Hermes Console.",
            ("portal", "open"): "`portal open` opens a browser and is not available in Hermes Console.",
            ("kanban", "tail"): "`kanban tail` streams output and is not available in Hermes Console.",
            ("kanban", "watch"): "`kanban watch` streams output and is not available in Hermes Console.",
            ("kanban", "daemon"): "`kanban daemon` starts a service and is not available in Hermes Console.",
            ("kanban", "dispatcher"): "`kanban dispatcher` starts a worker and is not available in Hermes Console.",
            ("kanban", "swarm"): "`kanban swarm` starts agent work and is not available in Hermes Console.",
            ("kanban", "decompose"): "`kanban decompose` starts agent work and is not available in Hermes Console.",
            ("kanban", "specify"): "`kanban specify` starts agent work and is not available in Hermes Console.",
            ("kanban", "gc"): "`kanban gc` is not available in Hermes Console.",
        }
        if len(tokens) >= 2:
            pair = (tokens[0], tokens[1])
            if pair in blocked_pairs:
                return blocked_pairs[pair]
        if tuple(tokens[:2]) in {("sessions", "delete"), ("sessions", "prune")}:
            return "`sessions delete` and `sessions prune` are not available in Hermes Console."
        return ""

    def _help_result(self) -> ConsoleResult:
        return ConsoleResult("ok", output=self.help_text())

    def _cap_output(self, output: str) -> str:
        if len(output) <= self.output_limit:
            return output
        omitted = len(output) - self.output_limit
        return f"{output[:self.output_limit]}\n... output truncated ({omitted} bytes omitted)"


def _expect_no_args(args: Sequence[str], usage: str) -> None:
    if args:
        raise ConsoleCommandError(f"Usage: {usage}")


HOSTED_CONFIG_ALLOWED_PREFIXES = (
    "display.",
    "ui.",
    "tts.",
    "voice.",
    "speech.",
    "sessions.",
    "cron.",
)
HOSTED_CONFIG_ALLOWED_KEYS = {
    "display.interface",
}
HOSTED_CONFIG_BLOCKED_PREFIXES = (
    "auth.",
    "dashboard.",
    "gateway.",
    "managed.",
    "model.",
    "portal.",
    "provider.",
    "providers.",
    "tool_gateway.",
    "custom_providers.",
    "mcp_servers.",
)
HOSTED_CONFIG_BLOCKED_NAMES = {
    "portal_url",
    "portal.url",
    "portal.base_url",
    "inference_url",
    "inference.url",
    "inference.base_url",
    "nous.portal_url",
    "nous.inference_url",
    "openrouter_api_key",
    "openai_api_key",
    "anthropic_api_key",
}


def _flag_present(args: Sequence[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)


def _flag_value(args: Sequence[str], flag: str) -> str | None:
    for index, arg in enumerate(args):
        if arg == flag:
            if index + 1 < len(args):
                return args[index + 1]
            return ""
        prefix = f"{flag}="
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def _hosted_config_key_allowed(key: str) -> bool:
    normalized = key.strip().lower()
    if normalized in HOSTED_CONFIG_BLOCKED_NAMES:
        return False
    if normalized.startswith(HOSTED_CONFIG_BLOCKED_PREFIXES):
        return False
    return normalized in HOSTED_CONFIG_ALLOWED_KEYS or normalized.startswith(
        HOSTED_CONFIG_ALLOWED_PREFIXES
    )


def _enforce_hosted_line_policy(path: tuple[str, ...], args: Sequence[str]) -> None:
    if path == ("config", "set"):
        key = args[0] if args else ""
        if key and not _hosted_config_key_allowed(key):
            raise ConsoleCommandError(
                f"`config set {key}` is not available in hosted Hermes Console. "
                "Use the dashboard setting for hosted account/provider changes."
            )
        return

    if path == ("mcp", "add"):
        if _flag_present(args, "--command") or _flag_present(args, "--args"):
            raise ConsoleCommandError(
                "Hosted Hermes Console does not add stdio MCP servers. "
                "Use catalog install or an HTTP/SSE URL."
            )
        if _flag_present(args, "--preset"):
            raise ConsoleCommandError(
                "Hosted Hermes Console does not add MCP presets directly. "
                "Use `mcp install <catalog-name>`."
            )
        url = _flag_value(args, "--url")
        if not url:
            raise ConsoleCommandError(
                "Hosted Hermes Console requires `mcp add` to use --url with "
                "an HTTP/SSE endpoint."
            )
        scheme = urlparse(url).scheme.lower()
        if scheme not in {"http", "https"}:
            raise ConsoleCommandError(
                "Hosted Hermes Console only accepts http:// or https:// MCP URLs."
            )
        return

    if path in {("cron", "create"), ("cron", "edit")}:
        for flag in ("--script", "--no-agent", "--workdir"):
            if _flag_present(args, flag):
                raise ConsoleCommandError(
                    f"`cron {' '.join(path[1:])} {flag}` is not available in "
                    "hosted Hermes Console."
                )


def _apply_confirmed_defaults(args: argparse.Namespace, context: ConsoleContext) -> None:
    """Skip nested prompts after the console-level confirmation has happened."""

    for attr in ("yes",):
        if hasattr(args, attr):
            setattr(args, attr, True)
    if getattr(args, "_console_command", None) == "import":
        setattr(args, "force", True)
    if getattr(args, "checkpoints_command", None) in {"clear", "clear-legacy"}:
        setattr(args, "force", True)
    if getattr(args, "plugins_action", None) == "install":
        if not getattr(args, "enable", False) and not getattr(args, "no_enable", False):
            setattr(args, "no_enable", True)
    if getattr(args, "auth_action", None) == "add":
        auth_type = getattr(args, "auth_type", None)
        if auth_type in {"api-key", "api_key"} and not getattr(args, "api_key", None):
            raise ConsoleCommandError("auth add --type api-key requires --api-key in Hermes Console.")
    if getattr(args, "import_name", None) is not None:
        # profile import has no prompt flag; leave it alone.
        return
    if getattr(args, "skills_action", None) in {
        "install",
        "reset",
        "opt-out",
        "repair-official",
    }:
        setattr(args, "yes", True)
    if getattr(args, "memory_command", None) == "reset":
        setattr(args, "yes", True)


def _status(_engine: HermesConsoleEngine, args: list[str]) -> str:
    _expect_no_args(args, "status")
    from types import SimpleNamespace

    from hermes_cli.status import show_status

    output = _capture_output(lambda: show_status(SimpleNamespace(all=False, deep=False)))
    return _strip_console_status_footer(output)


def _doctor(_engine: HermesConsoleEngine, args: list[str]) -> str:
    _expect_no_args(args, "doctor")
    from types import SimpleNamespace

    from hermes_cli.doctor import run_doctor

    return _capture_output(lambda: run_doctor(SimpleNamespace(fix=False, ack=None)))


def _logs(_engine: HermesConsoleEngine, args: list[str]) -> str:
    if "-f" in args or "--follow" in args:
        raise ConsoleCommandError("`logs -f` is not available in Hermes Console.")
    parser = _ArgumentParser(prog="logs", add_help=False)
    parser.add_argument("log_name", nargs="?", default="agent")
    parser.add_argument("-n", "--lines", type=int, default=50)
    parser.add_argument("--level")
    parser.add_argument("--session")
    parser.add_argument("--since")
    parser.add_argument("--component")
    ns = parser.parse_args(args)
    if ns.lines < 1 or ns.lines > 500:
        raise ConsoleCommandError("logs --lines must be between 1 and 500")

    from hermes_cli.logs import list_logs, tail_log

    if ns.log_name == "list":
        return _capture_output(list_logs)
    return _capture_output(
        lambda: tail_log(
            ns.log_name,
            num_lines=ns.lines,
            follow=False,
            level=ns.level,
            session=ns.session,
            since=ns.since,
            component=ns.component,
        )
    )


def _sessions_list(_engine: HermesConsoleEngine, args: list[str]) -> str:
    parser = _ArgumentParser(prog="sessions list", add_help=False)
    parser.add_argument("--limit", type=int, default=20)
    ns = parser.parse_args(args)
    if ns.limit < 1 or ns.limit > 200:
        raise ConsoleCommandError("sessions list --limit must be between 1 and 200")

    from hermes_state import SessionDB

    db = SessionDB()
    try:
        sessions = db.list_sessions_rich(
            exclude_sources=["tool"],
            limit=ns.limit,
            order_by_last_active=True,
        )
    finally:
        db.close()
    return _format_sessions(sessions)


def _sessions_stats(_engine: HermesConsoleEngine, args: list[str]) -> str:
    _expect_no_args(args, "sessions stats")
    from hermes_state import SessionDB

    db = SessionDB()
    try:
        total = db.session_count()
        listable = db.session_count(exclude_children=True, exclude_sources=["tool"])
        messages = db.message_count()
        lines = [
            f"Total sessions: {total}",
            f"Listable sessions: {listable}",
            f"Total messages: {messages}",
        ]
        for source in ["cli", "tui", "telegram", "discord", "slack", "cron"]:
            count = db.session_count(source=source)
            if count:
                lines.append(f"  {source}: {count}")
        return "\n".join(lines)
    finally:
        db.close()


def _config_show(_engine: HermesConsoleEngine, args: list[str]) -> str:
    _expect_no_args(args, "config show")
    from hermes_cli.config import show_config

    return _capture_output(show_config)


def _config_path(_engine: HermesConsoleEngine, args: list[str]) -> str:
    _expect_no_args(args, "config path")
    from hermes_cli.config import get_config_path

    return str(get_config_path())


def _config_set(_engine: HermesConsoleEngine, args: list[str]) -> str:
    if len(args) < 2:
        raise ConsoleCommandError("Usage: config set <key> <value>")
    key = args[0]
    value = " ".join(args[1:])
    from hermes_cli.config import set_config_value

    return _capture_output(lambda: set_config_value(key, value))


def _config_migrate(_engine: HermesConsoleEngine, args: list[str]) -> str:
    _expect_no_args(args, "config migrate")

    def _run() -> None:
        from hermes_cli.config import migrate_config

        results = migrate_config(interactive=False, quiet=False)
        if results.get("env_added") or results.get("config_added"):
            print("Configuration updated.")
        else:
            print("Configuration is up to date.")
        warnings = results.get("warnings") or []
        for warning in warnings:
            print(f"Warning: {warning}")

    return _capture_output(_run)


def _sessions_export(_engine: HermesConsoleEngine, args: list[str]) -> str:
    parser = _ArgumentParser(prog="sessions export", add_help=False)
    parser.add_argument("output")
    parser.add_argument("--source")
    parser.add_argument("--session-id")
    ns = parser.parse_args(args)

    def _run() -> None:
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            if ns.session_id:
                resolved_session_id = db.resolve_session_id(ns.session_id)
                if not resolved_session_id:
                    raise ConsoleCommandError(f"Session '{ns.session_id}' not found.")
                data = db.export_session(resolved_session_id)
                if not data:
                    raise ConsoleCommandError(f"Session '{ns.session_id}' not found.")
                rows = [data]
            else:
                rows = db.export_all(source=ns.source)

            lines = [json.dumps(row, ensure_ascii=False) for row in rows]
            text = "\n".join(lines)
            if text:
                text += "\n"
            if ns.output == "-":
                sys.stdout.write(text)
            else:
                Path(ns.output).expanduser().write_text(text, encoding="utf-8")
                print(f"Exported {len(rows)} session(s) to {ns.output}")
        finally:
            db.close()

    return _capture_output(_run)


def _sessions_rename(_engine: HermesConsoleEngine, args: list[str]) -> str:
    parser = _ArgumentParser(prog="sessions rename", add_help=False)
    parser.add_argument("session_id")
    parser.add_argument("title", nargs="+")
    ns = parser.parse_args(args)

    def _run() -> None:
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            resolved_session_id = db.resolve_session_id(ns.session_id)
            if not resolved_session_id:
                raise ConsoleCommandError(f"Session '{ns.session_id}' not found.")
            title = " ".join(ns.title)
            if not db.set_session_title(resolved_session_id, title):
                raise ConsoleCommandError(f"Session '{ns.session_id}' not found.")
            print(f"Session '{resolved_session_id}' renamed to: {title}")
        finally:
            db.close()

    return _capture_output(_run)


def _sessions_optimize(_engine: HermesConsoleEngine, args: list[str]) -> str:
    _expect_no_args(args, "sessions optimize")

    def _run() -> None:
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            count = db.vacuum()
            print(f"Optimized {count} FTS index(es).")
        finally:
            db.close()

    return _capture_output(_run)


def _sessions_repair(_engine: HermesConsoleEngine, args: list[str]) -> str:
    parser = _ArgumentParser(prog="sessions repair", add_help=False)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    ns = parser.parse_args(args)

    def _run() -> None:
        from hermes_state import DEFAULT_DB_PATH, _db_opens_cleanly, repair_state_db_schema

        db_path = DEFAULT_DB_PATH
        if not db_path.exists():
            print(f"No session database at {db_path} (nothing to repair).")
            return
        reason = _db_opens_cleanly(db_path)
        if reason is None:
            print(f"{db_path} opens cleanly; no repair needed.")
            return
        print(f"{db_path} does not open cleanly: {reason}")
        if ns.check_only:
            return
        report = repair_state_db_schema(db_path, backup=not ns.no_backup)
        if report.get("repaired"):
            if report.get("backup_path"):
                print(f"backup: {report['backup_path']}")
            print(f"strategy: {report.get('strategy')}")
            print("Repaired session database.")
            return
        raise ConsoleCommandError(f"Repair failed: {report.get('error')}")

    return _capture_output(_run)


def _profile_status(_engine: HermesConsoleEngine, args: list[str]) -> str:
    _expect_no_args(args, "profile")
    return _dispatch_extracted_subcommand(
        root="profile",
        fixed=(),
        args=(),
        module_name="hermes_cli.subcommands.profile",
        builder_name="build_profile_parser",
        main_handler_name="cmd_profile",
        console_context=_engine.context,
    )


def _cron_list(_engine: HermesConsoleEngine, args: list[str]) -> str:
    parser = _ArgumentParser(prog="cron list", add_help=False)
    parser.add_argument("--all", action="store_true")
    ns = parser.parse_args(args)
    from hermes_cli.cron import cron_list

    return _capture_output(lambda: cron_list(show_all=ns.all))


def _cron_status(_engine: HermesConsoleEngine, args: list[str]) -> str:
    _expect_no_args(args, "cron status")
    from hermes_cli.cron import cron_status

    return _capture_output(cron_status)


def _cron_pause(_engine: HermesConsoleEngine, args: list[str]) -> str:
    if len(args) != 1:
        raise ConsoleCommandError("Usage: cron pause <job>")
    from cron.jobs import AmbiguousJobReference, pause_job

    try:
        job = pause_job(args[0], reason="paused from hermes console")
    except AmbiguousJobReference as exc:
        raise ConsoleCommandError(str(exc)) from exc
    if not job:
        raise ConsoleCommandError(f"Job not found: {args[0]}")
    return _format_job(job, "Paused")


def _cron_resume(_engine: HermesConsoleEngine, args: list[str]) -> str:
    if len(args) != 1:
        raise ConsoleCommandError("Usage: cron resume <job>")
    from cron.jobs import AmbiguousJobReference, resume_job

    try:
        job = resume_job(args[0])
    except AmbiguousJobReference as exc:
        raise ConsoleCommandError(str(exc)) from exc
    if not job:
        raise ConsoleCommandError(f"Job not found: {args[0]}")
    return _format_job(job, "Resumed")


def _cron_run(_engine: HermesConsoleEngine, args: list[str]) -> str:
    if len(args) != 1:
        raise ConsoleCommandError("Usage: cron run <job>")
    from cron.jobs import AmbiguousJobReference, trigger_job

    try:
        job = trigger_job(args[0])
    except AmbiguousJobReference as exc:
        raise ConsoleCommandError(str(exc)) from exc
    if not job:
        raise ConsoleCommandError(f"Job not found: {args[0]}")
    return _format_job(job, "Triggered")


def run_console_repl(
    *,
    stdin=None,
    stdout=None,
    stderr=None,
    interactive: bool | None = None,
) -> int:
    """Run the local ``hermes console`` REPL."""

    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    if interactive is None:
        interactive = bool(getattr(stdin, "isatty", lambda: False)())

    engine = HermesConsoleEngine()
    if interactive:
        print("Hermes Console. Type `help` for commands, `exit` to quit.", file=stdout)

    while True:
        if interactive:
            print("hermes> ", end="", file=stdout, flush=True)
        line = stdin.readline()
        if line == "":
            if interactive:
                print(file=stdout)
            return 0

        result = engine.execute(line)
        if result.status == "confirm_required":
            if not interactive:
                print(
                    f"Confirmation required: {result.confirmation_message}",
                    file=stderr,
                )
                return 1
            print(f"{result.confirmation_message} [y/N] ", end="", file=stdout, flush=True)
            answer = stdin.readline()
            if answer.strip().lower() not in {"y", "yes"}:
                print("Cancelled.", file=stdout)
                continue
            result = engine.execute(result.command, confirmed=True)

        if result.output:
            stream = stderr if result.status == "error" else stdout
            print(result.output, file=stream)
        if result.status == "exit":
            return 0

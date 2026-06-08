"""Opt-in AI Beast orientation command hook.

This built-in hook is intentionally inert unless a caller supplies explicit
configuration with ``enabled`` and a safe local ``project_root``.  It only
handles read-only orientation commands and leaves Hermes-owned commands to the
normal gateway dispatch path.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

HERMES_OWNED_COMMANDS = frozenset({"status", "sessions"})
FORBIDDEN_COMMANDS = frozenset(
    {
        "task",
        "steer",
        "pause",
        "resume",
        "bindtopic",
        "switch",
        "open",
        "newsession",
    }
)
ORIENTATION_COMMANDS = frozenset({"whereami", "projects"})
APPROVED_BEAST_NAMESPACE_SUBCOMMANDS = frozenset(
    {
        "whereami",
        "projects",
        "sessions",
        "bindtopic",
        "task",
        "steer",
        "pause",
        "resume",
        "cancel",
        "unbindtopic",
        "move",
        "inbox",
        "open",
        "switch",
        "newsession",
    }
)


def _get_value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _normalise_command(event_type: str, context: Mapping[str, Any]) -> str:
    command = str(context.get("command") or "").strip().lower()
    if not command and event_type.startswith("command:"):
        command = event_type.split(":", 1)[1].strip().lower()
    return command.lstrip("/")


def _orientation_config(context: Mapping[str, Any]) -> Any:
    gateway_config = context.get("gateway_config")
    if gateway_config is None:
        return {}
    return _get_value(gateway_config, "ai_beast_orientation", {}) or {}


def _source_value(context: Mapping[str, Any], key: str) -> Any:
    source = context.get("source")
    if source is None:
        return None
    return _get_value(source, key)


def _inside_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False


def _safe_project_root(project_root_value: Any) -> Path | None:
    if not project_root_value:
        return None
    project_root = Path(str(project_root_value)).expanduser().resolve()
    if not project_root.exists() or not project_root.is_dir():
        return None
    return project_root


def _registry_paths(config: Any, project_root: Path) -> tuple[Path, Path]:
    registry_root_value = _get_value(
        config,
        "registry_root",
        project_root / "docs" / "interaction-layer" / "registry",
    )
    registry_root = Path(str(registry_root_value)).expanduser()
    if not registry_root.is_absolute():
        registry_root = project_root / registry_root
    registry_root = registry_root.resolve()
    if not _inside_root(registry_root, project_root):
        raise ValueError("registry_root must stay inside project_root")
    workspaces_path = (registry_root / "workspaces.json").resolve()
    bindings_path = (registry_root / "bindings.json").resolve()
    for registry_file in (workspaces_path, bindings_path):
        if not _inside_root(registry_file, registry_root):
            raise ValueError("registry files must stay inside registry_root")
        if not registry_file.is_file():
            raise ValueError("registry files must be regular files")
    return workspaces_path, bindings_path


def _module_file_inside_root(module: Any, project_root: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    return bool(module_file and _inside_root(Path(module_file), project_root))


def _import_ai_beast_modules(project_root: Path) -> tuple[Any, Any, Any]:
    root_text = str(project_root)
    original_path = list(sys.path)
    original_modules = dict(sys.modules)
    module_names = (
        "ai_beast_registry",
        "ai_beast_registry.loader",
        "ai_beast_registry.telegram_adapter",
        "ai_beast_registry.beast_namespace",
    )
    previous_modules = {name: sys.modules.get(name) for name in module_names}
    try:
        for name in module_names:
            sys.modules.pop(name, None)
        sys.path = [root_text, *(entry for entry in original_path if entry != root_text)]
        loader = importlib.import_module("ai_beast_registry.loader")
        telegram_adapter = importlib.import_module("ai_beast_registry.telegram_adapter")
        beast_namespace = importlib.import_module("ai_beast_registry.beast_namespace")
        if not (
            _module_file_inside_root(loader, project_root)
            and _module_file_inside_root(telegram_adapter, project_root)
            and _module_file_inside_root(beast_namespace, project_root)
        ):
            raise ImportError("AI Beast adapter resolved outside project_root")
    finally:
        sys.path = original_path
        for name, module in list(sys.modules.items()):
            if name not in original_modules and _module_file_inside_root(module, project_root):
                sys.modules.pop(name, None)
        for name in module_names:
            previous = previous_modules[name]
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return loader, telegram_adapter, beast_namespace


def _beast_namespace_enabled(config: Any) -> bool:
    return bool(_get_value(config, "beast_namespace_enabled", False))


def _lazy_beast_namespace_parser(project_root: Path) -> Callable[[str], Any]:
    _loader, _telegram_adapter, beast_namespace = _import_ai_beast_modules(project_root)
    parser = getattr(beast_namespace, "parse_beast_command", None)
    if not callable(parser):
        raise ImportError("AI Beast beast_namespace parser is not available")
    return parser


def _parse_value(parse: Any, key: str, default: Any = None) -> Any:
    return _get_value(parse, key, default)


def _beast_command_text(context: Mapping[str, Any]) -> str:
    raw_args = str(context.get("raw_args") or context.get("args") or "").strip()
    return f"/beast {raw_args}".strip()


def _safe_beast_command_label(
    subcommand: Any,
    args: tuple[Any, ...],
    *,
    subcommand_is_approved: bool,
) -> str:
    """Build a fail-closed label without echoing arbitrary user text."""
    if subcommand and subcommand_is_approved:
        return f"/beast {subcommand} argument_count={len(args)}"
    return f"/beast unapproved_or_unknown_subcommand argument_count={len(args)}"


def _format_beast_namespace_result(parse: Any) -> dict[str, str]:
    status = str(_parse_value(parse, "status", "unknown"))
    command_class = str(_parse_value(parse, "command_class", "unknown"))
    subcommand = _parse_value(parse, "subcommand")
    args = tuple(_parse_value(parse, "args", ()) or ())
    subcommand_is_approved = str(subcommand) in APPROVED_BEAST_NAMESPACE_SUBCOMMANDS if subcommand else False
    command_label = _safe_beast_command_label(
        subcommand,
        args,
        subcommand_is_approved=subcommand_is_approved,
    )

    if subcommand and not subcommand_is_approved:
        return {
            "decision": "deny",
            "message": (
                f"AI Beast /beast command fail-closed: {command_label} "
                "subcommand=unapproved_or_unknown is not approved for this disabled fixture path. "
                "No command behaviour executed."
            ),
        }

    if status != "recognised" or command_class.endswith("UNKNOWN") or command_class == "unknown":
        return {
            "decision": "deny",
            "message": (
                f"AI Beast /beast command fail-closed: {command_label} "
                f"status={status} class={command_class}. No command behaviour executed."
            ),
        }

    argument_count = len(args)
    if bool(_parse_value(parse, "is_read_only", False)):
        detail = "read-only metadata" if subcommand == "sessions" else "read-only context"
        message = (
            f"AI Beast classification: {command_label}\n"
            f"subcommand={subcommand} argument_count={argument_count}\n"
            f"class={command_class}\n"
            f"This command is recognised as {detail}. No state was changed."
        )
        return {"decision": "handled", "message": message}

    if bool(_parse_value(parse, "is_proposal_only", False)):
        message = (
            f"AI Beast classification: {command_label}\n"
            f"subcommand={subcommand} argument_count={argument_count}\n"
            f"class={command_class}\n"
            "This command is recognised but requires approval. No action occurred. It is approval-gated only."
        )
        return {"decision": "handled", "message": message}

    if bool(_parse_value(parse, "is_state_changing", False)):
        message = (
            f"AI Beast classification: {command_label}\n"
            f"subcommand={subcommand} argument_count={argument_count}\n"
            f"class={command_class}\n"
            "This command is recognised but requires approval. No action occurred. It was not executed."
        )
        return {"decision": "handled", "message": message}

    return {
        "decision": "deny",
        "message": (
            f"AI Beast /beast command fail-closed: {command_label} "
            f"status={status} class={command_class}. No command behaviour executed."
        ),
    }


def _lazy_orientation_adapter(config: Any, project_root: Path, context: Mapping[str, Any]) -> Callable[..., Any]:
    loader, telegram_adapter, _beast_namespace = _import_ai_beast_modules(project_root)
    workspaces_path, bindings_path = _registry_paths(config, project_root)
    registry = loader.load_registry(workspaces_path, bindings_path)

    def _adapter(*, command: str, project_root: Path, context: Mapping[str, Any]) -> str | None:
        del project_root
        return telegram_adapter.handle_telegram_orientation_command(
            f"/{command}",
            registry,
            chat_id=_source_value(context, "chat_id"),
            thread_id=_source_value(context, "thread_id"),
            bot_username=_get_value(config, "bot_username"),
        )

    return _adapter


async def _call_adapter(
    orientation_adapter: Callable[..., Any],
    *,
    command: str,
    project_root: Path,
    context: Mapping[str, Any],
) -> Any:
    result = orientation_adapter(
        command=command,
        project_root=project_root,
        context=context,
    )
    if inspect.isawaitable(result):
        result = await result
    return result


async def handle(
    event_type: str,
    context: Mapping[str, Any] | None,
    *,
    orientation_adapter: Callable[..., Any] | None = None,
    side_effects: Mapping[str, Callable[..., Any]] | None = None,
) -> dict[str, str] | None:
    """Handle a safe, explicitly enabled AI Beast orientation command.

    ``side_effects`` is accepted only as a test seam proving this hook does not
    invoke forbidden side-effect paths.  It is deliberately unused.
    """
    del side_effects

    if context is None:
        return None

    command = _normalise_command(event_type, context)
    if command in HERMES_OWNED_COMMANDS or command in FORBIDDEN_COMMANDS:
        return None

    config = _orientation_config(context)
    if not bool(_get_value(config, "enabled", False)):
        return None

    if command == "beast":
        if not _beast_namespace_enabled(config):
            return None
        project_root_value = _get_value(config, "project_root")
        project_root = _safe_project_root(project_root_value)
        if project_root is None:
            return {
                "decision": "deny",
                "message": "AI Beast orientation root is not available.",
            }
        try:
            parser = _lazy_beast_namespace_parser(project_root)
            parse = parser(_beast_command_text(context))
        except Exception:
            return {
                "decision": "deny",
                "message": "AI Beast /beast command fail-closed: parser unavailable. No command behaviour executed.",
            }
        return _format_beast_namespace_result(parse)

    if command not in ORIENTATION_COMMANDS:
        return None

    project_root_value = _get_value(config, "project_root")
    if not project_root_value:
        return {
            "decision": "deny",
            "message": "AI Beast orientation root is not available.",
        }

    project_root = _safe_project_root(project_root_value)
    if project_root is None:
        return {
            "decision": "deny",
            "message": "AI Beast orientation root is not available.",
        }

    if orientation_adapter is None:
        try:
            orientation_adapter = _lazy_orientation_adapter(config, project_root, context)
        except (ImportError, ModuleNotFoundError):
            return {
                "decision": "deny",
                "message": "AI Beast orientation adapter is not configured.",
            }
        except Exception:
            return {
                "decision": "deny",
                "message": "AI Beast orientation adapter failed safely.",
            }

    try:
        message = await _call_adapter(
            orientation_adapter,
            command=command,
            project_root=project_root,
            context=context,
        )
    except Exception:
        return {
            "decision": "deny",
            "message": "AI Beast orientation adapter failed safely.",
        }
    return {
        "decision": "handled",
        "message": str(message),
    }

"""User-defined tools from declarative YAML files.

Drop a file in ``~/.hermes/tools/<tool>.yaml`` and it becomes a first-class
tool the agent can call — no Python plugin required::

    # ~/.hermes/tools/my_search.yaml
    name: my_search
    description: "Search my internal documentation"
    command: 'curl -s "https://internal-docs/search?q=$HERMES_TOOL_ARG_QUERY"'
    parameters:
      query:
        type: string
        description: "Search query"
        required: true
    timeout: 60

Each file defines exactly one tool. Model-supplied parameters are exposed only
under the dedicated ``HERMES_TOOL_ARG_<NAME>`` namespace, so a parameter such
as ``path`` cannot overwrite inherited ``PATH`` or another execution-sensitive
variable. Values are shell-quoted into assignments inside a subshell, then the
whole command is dispatched through Hermes' ``terminal`` tool. This preserves
the configured local/container/SSH isolation and reuses its approval checks,
bounded output capture, timeout handling, and descendant-process cleanup.

The command template is user-authored and trusted. Authors should still quote
parameter expansions (for example, ``"$HERMES_TOOL_ARG_QUERY"``) to prevent
word splitting and glob expansion, and must not pass model values to ``eval``
or otherwise use them deliberately as command text.
"""

from __future__ import annotations

import logging
import re
import shlex
from pathlib import Path
from typing import Any, Callable, Mapping, Tuple

logger = logging.getLogger(__name__)

_TOOLSET = "custom"
_EMOJI = "🔧"
_DEFAULT_TIMEOUT = 60
_MAX_TIMEOUT = 600
_ALLOWED_PARAM_TYPES = {"string", "number", "integer", "boolean"}
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ARG_ENV_PREFIX = "HERMES_TOOL_ARG_"


def register(ctx) -> None:
    """Discover ``~/.hermes/tools/*.yaml`` and register each valid tool.

    A malformed file or a registration rejected by the host is logged and
    skipped so one user file cannot break agent startup.
    """
    for path in _iter_tool_files():
        try:
            name, schema, command, timeout = _load_spec(path)
        except Exception as exc:
            logger.warning("yaml_tools: skipping %s — %s", path, exc)
            continue

        # ToolRegistry.register() deliberately rejects a cross-toolset name
        # collision by returning without raising. Check that case first so
        # PluginContext cannot subsequently misattribute the pre-existing tool
        # to this plugin as successfully registered. An existing ``custom``
        # entry is allowed through: force-reloading plugins must replace the
        # prior handler/spec in the same registry.
        from tools.registry import registry

        existing = registry.get_entry(name)
        if existing is not None and existing.toolset != _TOOLSET:
            logger.warning(
                "yaml_tools: skipping %s — tool %r is already registered "
                "by toolset %r",
                path,
                name,
                existing.toolset,
            )
            continue

        parameters = schema["parameters"]
        handler = _make_handler(
            ctx.dispatch_tool,
            command,
            list(parameters["properties"]),
            timeout,
            required_names=parameters["required"],
        )
        try:
            ctx.register_tool(
                name=name,
                toolset=_TOOLSET,
                schema=schema,
                handler=handler,
                description=schema.get("description", ""),
                emoji=_EMOJI,
            )
        except Exception as exc:
            logger.warning(
                "yaml_tools: could not register tool %r from %s — %s",
                name,
                path,
                exc,
            )
        else:
            logger.debug("yaml_tools: registered custom tool %r from %s", name, path)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _tools_dir() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "tools"


def _iter_tool_files():
    directory = _tools_dir()
    if not directory.is_dir():
        return
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
            yield path


# ---------------------------------------------------------------------------
# Parsing / schema construction
# ---------------------------------------------------------------------------

def _load_spec(path: Path) -> Tuple[str, dict, str, int]:
    """Parse one tool file as ``(name, schema, command, timeout)``."""
    from utils import fast_safe_load

    raw = fast_safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("top-level YAML must be a mapping")

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("'name' is required and must be a non-empty string")
    name = name.strip()
    if not _NAME_RE.match(name):
        raise ValueError(
            f"invalid tool name {name!r}: use letters, digits and underscores, "
            "starting with a letter or underscore"
        )

    description = raw.get("description", "")
    if not isinstance(description, str):
        raise ValueError("'description' must be a string")

    command = raw.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("'command' is required and must be a non-empty string")

    timeout = _coerce_timeout(raw.get("timeout"))

    parameters = raw.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise ValueError("'parameters' must be a mapping of name -> spec")

    properties: dict = {}
    required: list = []
    env_names: dict[str, str] = {}
    for raw_pname, pspec in parameters.items():
        pname = str(raw_pname)
        if not _NAME_RE.match(pname):
            raise ValueError(
                f"invalid parameter name {pname!r}: use letters, digits and "
                "underscores, starting with a letter or underscore"
            )

        env_name = _parameter_env_name(pname)
        previous = env_names.get(env_name)
        if previous is not None:
            raise ValueError(
                f"parameters {previous!r} and {pname!r} map to the same "
                f"environment variable {env_name}; names must differ when "
                "compared case-insensitively"
            )
        env_names[env_name] = pname

        pspec = pspec or {}
        if not isinstance(pspec, dict):
            raise ValueError(f"parameter {pname!r} spec must be a mapping")
        ptype = pspec.get("type", "string")
        if ptype not in _ALLOWED_PARAM_TYPES:
            raise ValueError(
                f"parameter {pname!r} has unsupported type {ptype!r}; "
                f"allowed: {sorted(_ALLOWED_PARAM_TYPES)}"
            )

        prop: dict = {"type": ptype}
        pdesc = pspec.get("description")
        if pdesc is not None:
            prop["description"] = str(pdesc)
        enum = pspec.get("enum")
        if isinstance(enum, list) and enum:
            prop["enum"] = enum
        properties[pname] = prop
        if pspec.get("required"):
            required.append(pname)

    schema = {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }
    return name, schema, command, timeout


def _coerce_timeout(value: Any) -> int:
    if value is None:
        return _DEFAULT_TIMEOUT
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"'timeout' must be a whole number of seconds, got {value!r}")
    if seconds <= 0:
        raise ValueError("'timeout' must be a positive number of seconds")
    return min(seconds, _MAX_TIMEOUT)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _make_handler(
    dispatch_tool: Callable,
    command: str,
    param_names: list,
    timeout: int,
    *,
    required_names=(),
) -> Callable:
    """Build a handler that delegates execution to the registered terminal."""
    required_names = tuple(required_names)

    def handler(args: dict | None = None, **kwargs) -> str:
        from tools.registry import tool_error

        if args is None:
            args = {}
        if not isinstance(args, dict):
            return tool_error("Tool arguments must be an object", success=False)

        missing = [
            pname for pname in required_names
            if pname not in args or args[pname] is None
        ]
        if missing:
            return tool_error(
                f"Missing required parameter(s): {', '.join(missing)}",
                success=False,
            )

        try:
            runtime_command = _build_terminal_command(command, param_names, args)
        except ValueError as exc:
            return tool_error(str(exc), success=False)

        return dispatch_tool(
            "terminal",
            {"command": runtime_command, "timeout": timeout},
            **kwargs,
        )

    return handler


def _build_terminal_command(
    command: str,
    param_names: list,
    args: Mapping[str, Any],
) -> str:
    """Wrap a trusted template with isolated, shell-quoted parameter exports.

    Every declared parameter is assigned on every invocation. Missing optional
    values become an empty string, shadowing any same-named inherited value.
    The closing parenthesis is on its own line so a trailing comment in the
    user-authored template cannot consume it.
    """
    lines = ["("]
    for pname in param_names:
        value = args.get(pname)
        rendered = "" if value is None else _stringify(value)
        if "\x00" in rendered:
            raise ValueError(f"parameter {pname!r} contains NUL, which is not supported")
        lines.append(
            f"export {_parameter_env_name(pname)}={shlex.quote(rendered)}"
        )
    lines.extend((command, ")"))
    return "\n".join(lines)


def _parameter_env_name(parameter_name: str) -> str:
    return f"{_ARG_ENV_PREFIX}{parameter_name.upper()}"


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)

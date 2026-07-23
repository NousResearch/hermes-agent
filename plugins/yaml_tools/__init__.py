"""User-defined tools from declarative YAML files.

Drop a file in ``~/.hermes/tools/<tool>.yaml`` and it becomes a first-class
tool the agent can call — no Python plugin required::

    # ~/.hermes/tools/my_search.yaml
    name: my_search
    description: "Search my internal documentation"
    command: 'curl -s "https://internal-docs/search?q=$QUERY"'
    parameters:
      query:
        type: string
        description: "Search query"
        required: true
    timeout: 60          # optional, seconds (capped)

Each file defines exactly one tool. It is registered under the ``custom``
toolset (which auto-appears in the default tool set and can be toggled like any
other toolset).

Security — why this is injection-safe
-------------------------------------
Parameter values supplied by the model are passed to the command as
**environment variables** (both the parameter name and its upper-cased form),
never interpolated into the command string. bash does not re-parse the *result*
of a variable expansion for command substitution, so a value such as
``$(rm -rf ~)`` or ``"; rm -rf ~ ; "`` is treated as a literal string, not
executed. The command template is authored by the user (trusted); only the
argument *values* come from the model.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)

_TOOLSET = "custom"
_EMOJI = "🔧"
_DEFAULT_TIMEOUT = 60
_MAX_TIMEOUT = 600
_MAX_OUTPUT_CHARS = 100_000
_ALLOWED_PARAM_TYPES = {"string", "number", "integer", "boolean"}
# Tool and parameter names must be valid identifiers so they are safe as both
# function-call names and environment-variable names.
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def register(ctx) -> None:
    """Discover ``~/.hermes/tools/*.yaml`` and register each as a tool.

    Called once by the plugin loader. Never raises: a malformed file or a
    name collision is logged and skipped so it can't break agent startup.
    """
    for path in _iter_tool_files():
        try:
            spec = _load_spec(path)
        except Exception as exc:
            logger.warning("yaml_tools: skipping %s — %s", path, exc)
            continue
        name, schema, command, timeout = spec
        handler = _make_handler(command, list(schema["parameters"]["properties"]), timeout)
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
            # Most likely a name collision with a built-in or another YAML
            # tool. We never override built-ins, so just skip this one.
            logger.warning(
                "yaml_tools: could not register tool %r from %s — %s",
                name, path, exc,
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
    d = _tools_dir()
    if not d.is_dir():
        return
    for path in sorted(d.iterdir()):
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
            yield path


# ---------------------------------------------------------------------------
# Parsing / schema construction
# ---------------------------------------------------------------------------

def _load_spec(path: Path) -> Tuple[str, dict, str, int]:
    """Parse and validate one tool file.

    Returns ``(name, schema, command, timeout)`` or raises ``ValueError`` with
    a human-readable reason.
    """
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
    for pname, pspec in parameters.items():
        pname = str(pname)
        if not _NAME_RE.match(pname):
            raise ValueError(
                f"invalid parameter name {pname!r}: use letters, digits and "
                "underscores, starting with a letter or underscore"
            )
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

def _make_handler(command: str, param_names: list, timeout: int) -> Callable:
    def handler(args: Optional[dict] = None, **kwargs) -> str:
        from tools.registry import tool_error, tool_result

        args = args or {}
        env = os.environ.copy()
        for pname in param_names:
            value = args.get(pname)
            if value is None:
                continue
            rendered = _stringify(value)
            # Expose the value under both the exact parameter name and its
            # upper-cased form so `$query` and `$QUERY` both work in templates.
            env[pname] = rendered
            env[pname.upper()] = rendered

        bash = _find_bash()
        if bash is None:
            return tool_error(
                "bash is required to run YAML tools but was not found on PATH",
                success=False,
            )
        try:
            proc = subprocess.run(
                [bash, "-c", command],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return tool_error(f"Command timed out after {timeout}s", success=False)
        except Exception as exc:  # pragma: no cover - defensive
            return tool_error(f"Command failed to start: {exc}", success=False)

        output = (proc.stdout or "") + (proc.stderr or "")
        if len(output) > _MAX_OUTPUT_CHARS:
            output = output[:_MAX_OUTPUT_CHARS] + "\n… [output truncated]"
        if proc.returncode != 0:
            return tool_error(
                f"Command exited with status {proc.returncode}",
                success=False,
                output=output,
            )
        return tool_result(output=output, exit_code=0)

    return handler


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _find_bash() -> Optional[str]:
    import shutil
    return shutil.which("bash") or shutil.which("bash.exe")

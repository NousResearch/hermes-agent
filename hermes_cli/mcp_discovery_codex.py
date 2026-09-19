"""Translate Codex MCP entries without resolving credentials or running helpers.

Only user-level config.toml (including CODEX_HOME) is discovered. Project layers,
plugin manifests and Codex credential stores are deliberately not enumerated.
"""

from __future__ import annotations

import copy
import math
import re

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_HEADER_NAME = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
_COMMON_KEYS = {"enabled", "required", "startup_timeout_sec", "tool_timeout_sec", "enabled_tools", "disabled_tools"}
_STDIO_KEYS = {"command", "args", "env", "env_vars", "cwd"}
_HTTP_KEYS = {"url", "http_headers", "env_http_headers", "bearer_token_env_var", "auth"}
_MANUAL_SETUP = "Codex configuration uses unsupported settings; configure this server manually in Hermes."


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("Codex configuration contains an invalid string list.")
    return value


def _string_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()
    ):
        raise ValueError("Codex configuration contains an invalid string map.")
    return dict(value)


def _env_ref(name: object) -> str:
    if not isinstance(name, str) or not _ENV_NAME.fullmatch(name):
        raise ValueError("Codex configuration contains an invalid environment variable name.")
    return "${" + name + "}"


def _literal_tool_pattern(name: str) -> str:
    # Codex matches names literally; Hermes interprets fnmatch globs.
    escapes = {"[": "[[]", "*": "[*]", "?": "[?]"}
    return "".join(escapes.get(char, char) for char in name)


def normalize_codex_mcp(config: dict) -> dict:
    """Return a Hermes entry, or a credential-free explanation of incompatibility.

    Unsupported policies are NOT silently dropped (notably approval overrides,
    header helpers, remote executors and custom OAuth client configuration).
    The caller still applies the usual inline-credential and security checks.
    """
    stdio = "command" in config
    if stdio == ("url" in config):
        raise ValueError("Codex configuration must define exactly one transport (command or url).")
    if config.keys() - (_COMMON_KEYS | (_STDIO_KEYS if stdio else _HTTP_KEYS)):
        raise ValueError(_MANUAL_SETUP)
    for key in ("enabled", "required"):
        if key in config and type(config[key]) is not bool:
            raise ValueError("Codex configuration contains an invalid boolean setting.")
    if config.get("required", False):
        raise ValueError(_MANUAL_SETUP)

    transport_key = "command" if stdio else "url"
    value = config[transport_key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Codex configuration contains an invalid transport.")
    result = {transport_key: value}
    if "enabled" in config:
        result["enabled"] = config["enabled"]
    for source, target in (("startup_timeout_sec", "connect_timeout"), ("tool_timeout_sec", "timeout")):
        if source in config:
            timeout = config[source]
            try:
                valid = type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError("Codex configuration contains an invalid timeout.")
            result[target] = timeout

    if stdio:
        if "args" in config:
            result["args"] = list(_string_list(config["args"]))
        if "cwd" in config:
            if not isinstance(config["cwd"], str) or not config["cwd"].strip():
                raise ValueError("Codex configuration contains an invalid working directory.")
            result["cwd"] = config["cwd"]
        env = _string_map(config.get("env", {}))
        env_vars = config.get("env_vars", [])
        if not isinstance(env_vars, list):
            raise ValueError("Codex configuration contains an invalid environment variable list.")
        for entry in env_vars:
            if isinstance(entry, dict):
                if entry.keys() - {"name", "source"} or entry.get("source", "local") != "local":
                    raise ValueError(_MANUAL_SETUP)
                entry = entry.get("name")
            reference = _env_ref(entry)
            # Explicit env values override forwarded variables in Codex too.
            env.setdefault(entry, reference)
        if env or "env" in config:
            result["env"] = env
    else:
        headers = _string_map(config.get("http_headers", {}))
        names: set[str] = set()
        for key, value in headers.items():
            if not _HEADER_NAME.fullmatch(key) or "\r" in value or "\n" in value or key.lower() in names:
                raise ValueError("Codex configuration contains invalid or overlapping HTTP headers.")
            names.add(key.lower())
        for key, name in _string_map(config.get("env_http_headers", {})).items():
            if not _HEADER_NAME.fullmatch(key) or key.lower() in names:
                raise ValueError("Codex configuration contains invalid or overlapping HTTP headers.")
            headers[key] = _env_ref(name)
            names.add(key.lower())
        if "bearer_token_env_var" in config:
            if "authorization" in names:
                raise ValueError("Codex configuration contains overlapping authorization settings.")
            headers["Authorization"] = "Bearer " + _env_ref(config["bearer_token_env_var"])
        if headers or "http_headers" in config:
            result["headers"] = headers
        if "auth" in config:
            if config["auth"] != "oauth":
                raise ValueError(_MANUAL_SETUP)
            result["auth"] = "oauth"  # Hermes authenticates separately; no Codex tokens are read.

    denied = _string_list(config.get("disabled_tools", []))
    denied_names = set(denied)
    if "enabled_tools" in config:
        allowed = _string_list(config["enabled_tools"])
        kept = [_literal_tool_pattern(name) for name in allowed if name not in denied_names]
        # Codex applies deny AFTER allow. Hermes gives include precedence and
        # treats an empty include as no filter, so empty must explicitly deny all.
        result["tools"] = {"include": kept} if kept else {"exclude": ["*"]}
    elif denied:
        result["tools"] = {"exclude": [_literal_tool_pattern(name) for name in denied]}
    return copy.deepcopy(result)

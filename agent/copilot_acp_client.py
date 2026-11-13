"""Deprecated alias module: use :class:`agent.acp_client.ACPClient`.

Listed catalog ACP providers were built on this module at their pinned versions
(kimchi-acp-provider, kiro-acp, grok-acp, cursor-acp). They subclass ``CopilotACPClient``,
import former module helpers (``ACP_MARKER_BASE_URL``, ``_build_subprocess_env``,
``_acp_supported``, ``_model_selection_request``, ``_effective_timeout``) and patch
``subprocess.Popen`` through it. So the module keeps its whole former surface: the
Copilot-flavoured names keep their former defaults (launch env vars, the ``copilot-acp``
placeholder, the ``copilotEnablement`` model filter, the deprecated ``gh copilot`` extension
diagnosis), and the generic rest is served from :mod:`agent.acp_client`. The first access of
any of these names warns once per process. Remove no sooner than two minor releases after the
deprecation (the plugin compat contract in ``plugins/AGENTS.md``).
"""

from __future__ import annotations

import os
import shlex
# A real attribute, not served by __getattr__: plugins patch ``agent.copilot_acp_client.subprocess.Popen``
# (and ``.run``). It is the stdlib module itself, so such patches reach ACPClient's spawn and probe.
import subprocess
import warnings
from typing import Any

from agent import acp_client as _acp
from agent.acp_client import ACPClient

_warned = False
_PLACEHOLDER = "copilot-acp"
_COMMAND_ENV_VARS = ("HERMES_COPILOT_ACP_COMMAND", "COPILOT_CLI_PATH")
_ARGS_ENV_VAR = "HERMES_COPILOT_ACP_ARGS"
# Stderr fingerprint of the deprecated `gh copilot` extension: BOTH the product name AND a deprecation
# marker (the new `@github/copilot` CLI legitimately mentions "copilot-cli").
_GH_EXTENSION_REQUIRED = ("gh-copilot",)
_GH_EXTENSION_MARKERS = ("has been deprecated", "no commands will be executed")
_GH_EXTENSION_ERROR = (
    "Hermes ACP mode requires the NEW GitHub Copilot CLI (github.com/github/copilot-cli), but the binary it just "
    "spawned is the deprecated `gh copilot` extension.\n\n"
    "Install the new CLI:\n  npm install -g @github/copilot\n  # then verify with: copilot --help\n\n"
    "If `copilot` already resolves to the new CLI but you still see this,\npoint Hermes at it explicitly:\n"
    "  export HERMES_COPILOT_ACP_COMMAND=/path/to/new/copilot\n\n"
    "Alternative: use the `copilot` provider (no ACP, hits the Copilot API\ndirectly with a Copilot subscription "
    "token) via `hermes setup`.\n\nOriginal error:\n"
)


def _copilot_disabled(entry: dict[str, Any]) -> bool:
    return str((entry.get("_meta") or {}).get("copilotEnablement") or "").strip().lower() == "disabled"


def _is_gh_extension_banner(stderr_text: str) -> bool:
    lower = stderr_text.lower()
    return any(req in lower for req in _GH_EXTENSION_REQUIRED) and any(m in lower for m in _GH_EXTENSION_MARKERS)


class _DeprecatedCopilotACPClient(ACPClient):
    """ACPClient with the historical launch defaults, labels, model filter and early-exit diagnosis."""

    default_command = "copilot"
    default_args = ("--acp", "--stdio")
    command_env_vars = _COMMAND_ENV_VARS
    args_env_var = _ARGS_ENV_VAR
    marker_base_url = "acp://copilot"
    placeholder_model = _PLACEHOLDER
    label = "Copilot ACP"
    missing_command_hint = "Install GitHub Copilot CLI or set HERMES_COPILOT_ACP_COMMAND/COPILOT_CLI_PATH."
    unsupported_transport_hint = (
        "Either install a CLI that ships with --acp support (e.g. `@github/copilot` late 2025+), or set "
        "HERMES_COPILOT_ACP_COMMAND / HERMES_COPILOT_ACP_ARGS to a working pair.")

    def model_entry_disabled(self, entry: dict[str, Any]) -> bool:
        return _copilot_disabled(entry)

    def early_exit_error(self, stderr_text: str) -> str | None:
        return _GH_EXTENSION_ERROR + stderr_text if _is_gh_extension_banner(stderr_text) else None


_DeprecatedCopilotACPClient.__name__ = _DeprecatedCopilotACPClient.__qualname__ = "CopilotACPClient"


def _former_resolve_command() -> str:
    return next((v for name in _COMMAND_ENV_VARS if (v := os.getenv(name, "").strip())), "") or "copilot"


def _former_resolve_args() -> list[str]:
    return shlex.split(os.getenv(_ARGS_ENV_VAR, "").strip()) or ["--acp", "--stdio"]


def _former_enabled_id_list(entries: Any, key: str) -> list[str]:
    return _acp._enabled_id_list(entries, key, _copilot_disabled)


def _former_session_model_ids(session: dict[str, Any]) -> list[str]:
    return _acp._session_model_ids(session, _copilot_disabled)


def _former_legacy_session_model_ids(session: dict[str, Any]) -> list[str]:
    return _acp._legacy_session_model_ids(session, _copilot_disabled)


def _former_model_selection_request(session: dict[str, Any], requested_model: str) -> tuple[str, dict[str, str]] | None:
    return _acp._model_selection_request(session, requested_model, placeholder=_PLACEHOLDER, disabled=_copilot_disabled)


# Former names whose behaviour is Copilot-flavoured: served from here with their former defaults.
_COMPAT: dict[str, Any] = {
    "CopilotACPClient": _DeprecatedCopilotACPClient,
    "ACP_MARKER_BASE_URL": "acp://copilot",
    "_DEPRECATION_REQUIRED": _GH_EXTENSION_REQUIRED,
    "_DEPRECATION_MARKERS": _GH_EXTENSION_MARKERS,
    "_DEPRECATED_CLI_ERROR": _GH_EXTENSION_ERROR,
    "_is_gh_copilot_deprecation_message": _is_gh_extension_banner,
    "_resolve_command": _former_resolve_command,
    "_resolve_args": _former_resolve_args,
    "_enabled_id_list": _former_enabled_id_list,
    "_session_model_ids": _former_session_model_ids,
    "_legacy_session_model_ids": _former_legacy_session_model_ids,
    "_model_selection_request": _former_model_selection_request,
}
# Former names with unchanged behaviour: the very objects agent.acp_client uses (shared caches included).
_FORWARDED = frozenset({
    "logger", "_DEFAULT_TIMEOUT_SECONDS", "_ROLE_LABELS", "_ACP_PROBE_CACHE", "_PROMPT_PREAMBLE", "_INITIALIZE_PARAMS",
    "_acp_supported", "_resolve_home_dir", "_build_subprocess_env", "_jsonrpc_result", "_jsonrpc_error",
    "_model_config_option", "_format_messages_as_prompt", "_render_message_content", "_ensure_path_within_cwd",
    "_effective_timeout", "_fs_read_text_file", "_fs_write_text_file", "_FS_HANDLERS",
    "contextlib", "json", "logging", "queue", "re", "tempfile", "threading", "time", "deque", "Callable", "Iterator",
    "Path", "SimpleNamespace", "_completion_to_stream_chunks", "_extract_tool_calls_from_text",
    "_render_tool_bridge_sections", "get_nt_namespace_error", "get_read_block_error", "get_write_denied_error",
    "is_write_approval_required", "redact_sensitive_text", "resolve_agent_cwd",
})


def _former_attribute(name: str) -> Any:
    if name in _COMPAT:
        return _COMPAT[name]
    if name in _FORWARDED:
        return getattr(_acp, name)
    if name == "hermes_subprocess_env":
        from tools.environments.local import hermes_subprocess_env  # lazy: hermes_cli.config may still be importing

        return hermes_subprocess_env
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __getattr__(name: str) -> Any:
    global _warned
    value = _former_attribute(name)
    if not _warned:
        _warned = True
        warnings.warn(
            f"agent.copilot_acp_client.{name} is deprecated; subclass agent.acp_client.ACPClient instead.",
            DeprecationWarning, stacklevel=2)
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_COMPAT, *_FORWARDED, "hermes_subprocess_env"})

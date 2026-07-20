"""Registry of ACP-compatible coding agents Hermes can drive as backends.

Maps the ``{agent}`` part of an ``acp://{agent}`` base URL to the command
that starts that agent's ACP server on stdio. Every built-in entry uses the
official ACP adapter published by the agent's vendor or by Zed Industries —
Hermes never drives vendor CLIs through undocumented interfaces (see
issue #5257 and the compliance discussion in #3660).

Resolution order for a given agent name:
  1. ``HERMES_ACP_{NAME}_COMMAND`` env var (name uppercased, ``-`` → ``_``).
     The value is a full command line, split with :func:`shlex.split`.
  2. Legacy per-agent env vars declared on the entry (command path only —
     preserves the historical Copilot contract of
     ``HERMES_COPILOT_ACP_COMMAND`` / ``COPILOT_CLI_PATH``).
  3. The built-in registry below.

Arguments can be overridden separately with ``HERMES_ACP_{NAME}_ARGS`` (or
the entry's legacy args env var), also shlex-split. No shell is ever
involved in either path.
"""

from __future__ import annotations

import os
import shlex
import shutil
from dataclasses import dataclass


@dataclass(frozen=True)
class ACPAgentEntry:
    """A known ACP agent and how to launch its stdio server."""

    command: str
    """Executable that speaks ACP on stdin/stdout."""

    args: tuple[str, ...] = ()
    """Default arguments passed to *command*."""

    command_fallbacks: tuple[str, ...] = ()
    """Alternative executables tried (in order) when *command* is not on PATH.

    Lets an agent prefer a maintained/official bin while still resolving an
    older or vendor-alternate bin that happens to be installed instead.
    """

    env_unset: tuple[str, ...] = ()
    """Env vars stripped from the child process before launching this agent.

    Some ACP bridges self-guard against running *inside* their own parent CLI
    session (e.g. the Claude Code bridge aborts when ``CLAUDECODE`` and the
    ``CLAUDE_CODE_*`` session markers are present). Declaring them here keeps
    the generic client agent-agnostic — see ``_build_subprocess_env``.
    """

    display_name: str = ""
    """Human-readable name for logs and error messages."""

    install_hint: str = ""
    """One-line hint shown when the command is not found."""

    legacy_command_env_vars: tuple[str, ...] = ()
    """Older env vars holding the command *path only* (no args)."""

    legacy_args_env_var: str = ""
    """Older env var holding an args override (shlex-split)."""


ACP_AGENT_REGISTRY: dict[str, ACPAgentEntry] = {
    "copilot": ACPAgentEntry(
        command="copilot",
        args=("--acp", "--stdio"),
        display_name="GitHub Copilot",
        install_hint=(
            "Install GitHub Copilot CLI (npm install -g @github/copilot) or "
            "set HERMES_COPILOT_ACP_COMMAND/COPILOT_CLI_PATH."
        ),
        legacy_command_env_vars=("HERMES_COPILOT_ACP_COMMAND", "COPILOT_CLI_PATH"),
        legacy_args_env_var="HERMES_COPILOT_ACP_ARGS",
    ),
    "claude": ACPAgentEntry(
        command="claude-agent-acp",
        command_fallbacks=("claude-code-acp",),
        # The Claude Code bridge refuses to launch when it detects it is
        # running inside another Claude Code session (guard keyed off these
        # markers); strip them so Hermes can drive it from any environment.
        env_unset=("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SSE_PORT"),
        display_name="Claude Code",
        install_hint=(
            "Install the maintained adapter: "
            "npm install -g @agentclientprotocol/claude-agent-acp "
            "(the older @zed-industries/claude-code-acp bin also works). "
            "Requires Claude Code CLI credentials or ANTHROPIC_API_KEY."
        ),
    ),
    "codex": ACPAgentEntry(
        command="codex-acp",
        display_name="Codex CLI",
        install_hint=(
            "Install the official adapter: "
            "npm install -g @zed-industries/codex-acp "
            "(requires Codex CLI login or OPENAI_API_KEY)."
        ),
    ),
    "gemini": ACPAgentEntry(
        command="gemini",
        args=("--experimental-acp",),
        display_name="Gemini CLI",
        install_hint=(
            "Install Gemini CLI: npm install -g @google/gemini-cli "
            "(requires Google login or GEMINI_API_KEY)."
        ),
    ),
    "qwen": ACPAgentEntry(
        command="qwen",
        args=("--experimental-acp",),
        display_name="Qwen Code",
        install_hint=(
            "Install Qwen Code: npm install -g @qwen-code/qwen-code "
            "(requires Qwen login or an API key)."
        ),
    ),
}


def normalize_agent_name(agent_name: str) -> str:
    return str(agent_name or "").strip().lower()


def known_agents() -> tuple[str, ...]:
    return tuple(sorted(ACP_AGENT_REGISTRY))


def get_agent_entry(agent_name: str) -> ACPAgentEntry | None:
    return ACP_AGENT_REGISTRY.get(normalize_agent_name(agent_name))


def agent_display_name(agent_name: str) -> str:
    entry = get_agent_entry(agent_name)
    if entry and entry.display_name:
        return entry.display_name
    name = normalize_agent_name(agent_name)
    return name.title() if name else "ACP agent"


def agent_install_hint(agent_name: str) -> str:
    entry = get_agent_entry(agent_name)
    if entry and entry.install_hint:
        return entry.install_hint
    return (
        f"Set {_command_env_key(agent_name)} to the command that starts this "
        "agent's ACP server on stdio."
    )


def agent_env_unset(agent_name: str) -> tuple[str, ...]:
    """Env vars to strip from *agent_name*'s child process (empty if none)."""
    entry = get_agent_entry(agent_name)
    return entry.env_unset if entry else ()


def _preferred_command(entry: ACPAgentEntry) -> str:
    """First of ``command``/``command_fallbacks`` on PATH, else ``command``.

    Falling back to the primary ``command`` when nothing is installed keeps
    the not-found error message and install hint pointed at the preferred bin.
    """
    for candidate in (entry.command, *entry.command_fallbacks):
        if shutil.which(candidate):
            return candidate
    return entry.command


def _command_env_key(agent_name: str) -> str:
    return f"HERMES_ACP_{normalize_agent_name(agent_name).upper().replace('-', '_')}_COMMAND"


def _args_env_key(agent_name: str) -> str:
    return f"HERMES_ACP_{normalize_agent_name(agent_name).upper().replace('-', '_')}_ARGS"


def is_acp_agent_available(agent_name: str) -> bool:
    """True when *agent_name* resolves via env override or the registry."""
    name = normalize_agent_name(agent_name)
    if not name:
        return False
    if os.getenv(_command_env_key(name), "").strip():
        return True
    return name in ACP_AGENT_REGISTRY


def resolve_agent_launch(agent_name: str) -> tuple[str, list[str]]:
    """Return ``(command, args)`` for *agent_name*.

    Raises :class:`ValueError` for unknown agents with no env override.
    """
    name = normalize_agent_name(agent_name)
    entry = ACP_AGENT_REGISTRY.get(name)

    command = ""
    args: list[str] = []

    generic = os.getenv(_command_env_key(name), "").strip() if name else ""
    if generic:
        parts = shlex.split(generic)
        if not parts:
            raise ValueError(
                f"{_command_env_key(name)} is set but empty after parsing."
            )
        command, args = parts[0], parts[1:]
    elif entry:
        for var in entry.legacy_command_env_vars:
            value = os.getenv(var, "").strip()
            if value:
                command = value  # legacy contract: command path only
                break
        if not command:
            command = _preferred_command(entry)
        args = list(entry.args)
    else:
        raise ValueError(
            f"Unknown ACP agent '{agent_name}'. Known agents: "
            f"{', '.join(known_agents())}. For other agents set "
            f"{_command_env_key(agent_name)} to the ACP launch command."
        )

    raw_args = os.getenv(_args_env_key(name), "").strip()
    if not raw_args and entry and entry.legacy_args_env_var:
        raw_args = os.getenv(entry.legacy_args_env_var, "").strip()
    if raw_args:
        args = shlex.split(raw_args)

    return command, args

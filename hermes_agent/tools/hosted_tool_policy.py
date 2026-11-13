"""What a hosted scope may run: an allowlist of tools, and one switch for host-state features.

In a hosted scope nothing tenant-supplied may execute on this host or touch its filesystem, and no
agent may read or write state this host keeps for its profile. Two decisions enforce that. Each is
made in one function and is deny-by-default, so a new tool or feature stays off in a hosted scope
until someone shows it acts only through the agent's sandbox:

* :func:`hosted_tool_refusal` — a tool runs in a hosted scope only if it is on
  :data:`HOSTED_TOOL_ALLOWLIST`, the tools whose every effect goes through the agent's own sandbox
  (``hermes_agent.tools.environments.sandbox_scope``). Everything else (memory, recall, todo, vision, browser, web,
  MCP, connectors, catalog, delegation, skills, cron, ...) is refused before its handler. The agent
  loop asks once per call, in ``hermes_agent.agent.tool_executor._dispatch_authorized_once``: the one function every
  sequential, concurrent and batched call goes through, whatever executes it (inline agent tools,
  memory-provider and context-engine tools, the registry, connectors). The execution primitives also
  ask, for callers outside the loop: ``ToolRegistry.dispatch`` (execute_code RPC, the session kernel,
  the MCP bridge, plugin ``dispatch_tool``), the inline-executor table and connector dispatch.
* :func:`host_feature_enabled` — the agent features outside tool dispatch that read or write this
  host's state for the profile, or run here (:data:`HOST_FEATURES`). Every one is off in a hosted
  scope; each site asks this one function. They come back per agent through storage ports.

A local install (no scope, or a local one) is untouched: neither set is consulted, and every
function here answers "allowed".
"""

from __future__ import annotations

from typing import Optional

# Tools whose every effect goes through the agent's ScopedEnvironment and nothing on this host:
#   terminal        commands run by env.execute in the sandbox; background spawns via spawn_via_env,
#                   their records stamped with the sandbox key.
#   process_manage  acts only on the bound agent's own records (sandbox_visible); polls via its env.
#   execute_code    always the remote kernel / per-call script in the sandbox, never the host kernel;
#                   its RPC tool calls re-enter ToolRegistry.dispatch, so they meet this allowlist too.
#   read_file, write_file, patch, search_files
#                   ShellFileOperations over the sandbox env: no host fast path, a lexical path policy,
#                   no host document extraction; an oversized result spills into the sandbox.
HOSTED_TOOL_ALLOWLIST = frozenset({
    "terminal", "process_manage", "execute_code", "read_file", "write_file", "patch", "search_files",
})

# Agent features outside tool dispatch that read or write this host's profile state, or run here.
HOST_FEATURES = frozenset({
    "context_files",              # AGENTS.md / CLAUDE.md / .cursorrules / SOUL.md in the system prompt
    "subdirectory_hints",         # AGENTS.md found beside the paths a tool call names
    "context_references",         # @file / @diff / @url expansion of the user's message
    "workspace_snapshot",         # git and project facts for the system prompt
    "backend_probe",              # a throwaway environment built to describe the backend
    "builtin_memory",             # MEMORY.md / USER.md and external memory providers
    "session_recall",             # the profile's state.db opened for recall
    "background_review",          # the post-turn memory/skill review fork
    "trajectory_export",          # trajectory JSONL in the working directory
    "file_mutation_bookkeeping",  # host stat of a failed write's target
    "checkpoints",                # shadow-git checkpoints and rollback of the working directory
    "media_processing",           # images/video decoded, converted and cached on this host
    "skills",                     # the profile's skill files: index, preload, auto-load, slash commands
    "skill_inline_shell",         # !`cmd` blocks in a skill body
    "cron_scripts",               # a cron job's pre-run script
    "shell_hooks",                # operator shell hooks: a host process fed the event payload
    "tool_host_artifacts",        # files an allowed tool keeps here about a call: terminal output
                                  # spill, saved blocked scripts, the verification ledger
    "mcp_stdio",                  # a profile MCP server started as a host process
    "media_delivery",             # a reply attachment named by host path
    "transcription",              # audio read and speech engines run here
    "delegation",                 # child agents started outside a tool call (/review)
})


class HostedToolRefused(PermissionError):
    """A hosted scope asked for something that would run or read on this host."""


def _hosted() -> bool:
    from hermes_agent.tools.environments.sandbox_scope import hosted_scope
    return hosted_scope() is not None


def _message(what: str, why: str) -> str:
    return (f"{what} is not available to a hosted agent: {why}. A hosted agent's tools are "
            f"{', '.join(sorted(HOSTED_TOOL_ALLOWLIST))}, all run in its own sandbox.")


def hosted_tool_refusal(tool_name: str) -> Optional[str]:
    """Why a hosted scope may not run *tool_name*, or ``None``. Always ``None`` outside a hosted scope,
    where the allowlist is not consulted."""
    if not _hosted() or tool_name in HOSTED_TOOL_ALLOWLIST:
        return None
    return _message(tool_name, "only tools that act solely through the agent's sandbox are enabled")


def host_feature_enabled(feature: str) -> bool:
    """Whether *feature* (one of :data:`HOST_FEATURES`) may use this host's state now: always outside
    a hosted scope, never inside one."""
    if feature not in HOST_FEATURES:
        raise KeyError(f"unknown host feature {feature!r}")
    return not _hosted()


def host_feature_refusal(feature: str, what: str) -> Optional[str]:
    """:func:`host_feature_enabled` as a message for *what*: the refusal in a hosted scope, else ``None``."""
    if host_feature_enabled(feature):
        return None
    return _message(what, f"it uses this host's {feature.replace('_', ' ')}")


def require_host_feature(feature: str, what: str) -> None:
    """Raise :class:`HostedToolRefused` when *feature* is off (a hosted scope); a no-op otherwise."""
    if refusal := host_feature_refusal(feature, what):
        raise HostedToolRefused(refusal)

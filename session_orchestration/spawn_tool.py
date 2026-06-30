"""
session_orchestration/spawn_tool.py — Thin LLM tool wrapping spawn_session.

Registered with the tool registry at module-import time.  The gateway's
Discord platform branch imports this module to activate the tool (see the
``if platform_key == "discord":`` block in ``gateway/run.py::_run_agent``).

Tool schema
-----------
Input parameters: ``{repo: str, prompt: str, agent?: str, z_command?: str}``

Resolution order
----------------
1. Call ``build_repo_registry(cfg.repos)`` and ``registry.resolve(repo)``.
   - ``UnresolvedRepo`` → return an error string asking the user to supply a
     full path.  ``spawn_session`` is NOT called in this branch.
2. Resolve ``agent`` to ``resolved.default_agent or DEFAULT_AGENT`` ("omp")
   when the caller omits it.
3. Build a ``SpawnRequest`` (workdir filled from ``resolved.path``) and call
   ``spawn_session(request)``, returning the same reply-string shape that
   ``handle_spawn_command`` would produce.

Testability
-----------
The public ``spawn_tool_handler`` accepts keyword-injectable ``cfg``,
``_repo_registry``, and ``_spawn_fn`` so unit tests can drive it with a fake
registry and assert ``spawn_session`` is/is not called — without launching
tmux.  The registered handler calls it with all defaults (None → live).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from session_orchestration.repo_registry import (
    DEFAULT_AGENT,
    RepoRegistry,
    UnresolvedRepo,
    build_repo_registry,
)

if TYPE_CHECKING:
    from session_orchestration.config import SessionOrchestrationConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schema (OpenAI function-calling format)
# ---------------------------------------------------------------------------

SPAWN_TOOL_SCHEMA: Dict[str, Any] = {
    "name": "session_spawn",
    "description": (
        "Spawn a managed coding-agent session in a background tmux pane. "
        "Hermes relays the agent's questions to this thread and routes your "
        "replies back. Supply the repo name (alias or basename such as "
        "'hermes-agent') or an absolute path. The agent defaults to 'omp' "
        "when not specified. If the user has not said which repo to run on, "
        "do NOT guess and do NOT call this tool with a placeholder — ask the "
        "user which repo (a name or an absolute path) first, then spawn."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": (
                    "Repo name (alias, basename, or absolute path) that "
                    "identifies the working directory for the new session."
                ),
            },
            "prompt": {
                "type": "string",
                "description": "Initial task prompt to deliver to the agent.",
            },
            "agent": {
                "type": "string",
                "description": (
                    "Agent to use: 'omp', 'claude', 'claude-code', etc. "
                    "Defaults to 'omp' when omitted."
                ),
            },
            "z_command": {
                "type": "string",
                "description": (
                    "Optional z-harness slash command to prepend to the "
                    "prompt, e.g. '/z-plan'."
                ),
            },
        },
        "required": ["repo", "prompt"],
    },
}

# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def spawn_tool_handler(
    args: Dict[str, Any],
    *,
    cfg: Optional["SessionOrchestrationConfig"] = None,
    _repo_registry: Optional[RepoRegistry] = None,
    _spawn_fn: Optional[Callable] = None,
) -> str:
    """Handle an LLM ``session_spawn`` tool call.

    Parameters
    ----------
    args:
        Tool arguments from the model: ``repo``, ``prompt``, and optionally
        ``agent`` and ``z_command``.
    cfg:
        Injectable ``SessionOrchestrationConfig``.  When ``None`` (default),
        loaded from the live Hermes config via
        ``load_session_orchestration_config()``.
    _repo_registry:
        Injectable ``RepoRegistry``.  When ``None`` (default), built from
        ``cfg.repos`` via ``build_repo_registry()``.  Provide a fake
        registry in unit tests to avoid filesystem scans.
    _spawn_fn:
        Injectable replacement for ``spawn_session``.  When ``None``
        (default), the real ``spawn_session`` from
        ``session_orchestration.spawn`` is used.  Pass a mock in tests to
        assert call arguments without launching tmux.

    Returns
    -------
    str
        Human-readable reply suitable for posting to Discord.  On success
        this matches the format ``handle_spawn_command`` would produce.
        On resolution failure an error string is returned and ``_spawn_fn``
        is NOT called.
    """
    # ---- Resolve live config when not injected ---------------------------
    if cfg is None:
        from session_orchestration.config import load_session_orchestration_config
        cfg = load_session_orchestration_config()

    # ---- Build repo registry when not injected ---------------------------
    if _repo_registry is None:
        _repo_registry = build_repo_registry(cfg.repos)

    # ---- Resolve live spawn_session when not injected --------------------
    if _spawn_fn is None:
        from session_orchestration.spawn import spawn_session as _spawn_fn

    # ---- Extract arguments -----------------------------------------------
    repo: str = str(args.get("repo") or "").strip()
    prompt: str = str(args.get("prompt") or "").strip()
    agent_arg: str = str(args.get("agent") or "").strip()
    z_command: Optional[str] = str(args.get("z_command") or "").strip() or None

    if not repo:
        return (
            "Which repo should I run this on? Tell me a repo name "
            "(alias or basename like 'hermes-agent') or an absolute path, "
            "and I'll spawn the session."
        )
    if not prompt:
        return (
            "What should the session work on? Give me a task prompt and "
            "I'll spawn the session."
        )

    # ---- Resolve repo → workdir ------------------------------------------
    resolution = _repo_registry.resolve(repo)
    if isinstance(resolution, UnresolvedRepo):
        return (
            f"I couldn't find a repo matching '{resolution.name}'. "
            "Which repo should I use? Give me an absolute path "
            "(e.g. '/home/user/dev/myrepo') or configure an alias in "
            "`session_orchestration.repos`."
        )

    workdir: str = resolution.path
    agent: str = agent_arg or resolution.default_agent or DEFAULT_AGENT

    # ---- Build SpawnRequest and spawn ------------------------------------
    from session_orchestration.spawn import SpawnRequest, SpawnResult

    request = SpawnRequest(
        prompt=prompt,
        agent=agent,
        workdir=workdir,
        z_command=z_command,
    )

    try:
        result: SpawnResult = _spawn_fn(request)
    except Exception as exc:
        logger.exception(
            "session_spawn: spawn_session failed agent=%s workdir=%s", agent, workdir
        )
        return f"Spawn failed: {exc}"

    # ---- Build reply (mirrors handle_spawn_command reply shape) ----------
    lines = [
        f"Spawned **{agent}** session.",
        f"Task ID: `{result.task_id}`",
        f"Session: `{result.session_name}`",
        f"Workdir: `{workdir}`",
    ]
    if result.thread_id:
        lines.append(f"Project thread: <#{result.thread_id}>")
    else:
        lines.append("(No project thread — reply will route by task_id)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# check_fn — tool is invisible when session_orchestration is disabled
# ---------------------------------------------------------------------------


def _check_enabled() -> bool:
    """Return True when session_orchestration is enabled in Hermes config."""
    try:
        from session_orchestration.config import is_enabled
        return is_enabled()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Self-registration with the tool registry
#
# Importing this module (triggered by the gateway's Discord platform branch
# in _run_agent) registers the tool as toolset "session_spawn".  That toolset
# name is appended to ``enabled_toolsets`` for the Discord platform, making
# it available to the LLM on every @mention turn.
# ---------------------------------------------------------------------------

from tools.registry import registry  # noqa: E402 — after helper definitions

registry.register(
    name="session_spawn",
    toolset="session_spawn",
    schema=SPAWN_TOOL_SCHEMA,
    handler=spawn_tool_handler,
    check_fn=_check_enabled,
    description=SPAWN_TOOL_SCHEMA["description"],
    emoji="🚀",
)

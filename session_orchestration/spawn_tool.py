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
# Discord thread wiring
# ---------------------------------------------------------------------------
#
# The LLM ``session_spawn`` tool has no ``event`` in scope the way the
# ``/so-spawn`` command handler does, so it reads the current turn's
# chat/thread ids from the gateway *session context* (``contextvars`` bound
# per-turn by ``GatewayRunner._set_session_env``).
#
# An ``@Hermès`` turn already opens a Discord *thread* — the "Hermès window"
# the user is looking at.  We ADOPT that thread as the session's
# ``discord_thread_id`` so the agent's questions and the feed's action pointer
# land in the window the user is already in, instead of a second thread.  Only
# when a turn is in a plain channel (no thread) do we mint one, bridging to the
# live Discord adapter's ``create_handoff_thread`` on the gateway event loop.
#
# Everything here is best-effort: any failure resolves to ``(None, None)`` and
# the spawn proceeds threadless (the prior behavior), never blocking a spawn.


def _resolve_discord_thread_context():
    """Return ``(thread_target, thread_creator)`` for the live Discord turn.

    ``thread_creator(thread_target, name) -> Optional[str]`` yields the id
    ``spawn_session`` records as ``discord_thread_id``:

    - **In a thread (normal case).**  Adopt the Hermès-window thread the turn
      is already in.  ``thread_creator`` is a no-op returning that thread id;
      nothing is created, and no live adapter/loop is required.
    - **Plain channel (fallback).**  Mint a thread under the channel via the
      live adapter's async ``create_handoff_thread`` on the gateway loop.

    Returns ``(None, None)`` off Discord, with no chat id, or (channel case)
    when the live gateway/adapter/loop is unavailable — the caller then spawns
    without a thread.
    """
    try:
        from gateway.session_context import get_session_env
        platform = (get_session_env("HERMES_SESSION_PLATFORM", "") or "").strip().lower()
        if platform != "discord":
            return None, None
        chat_id = (get_session_env("HERMES_SESSION_CHAT_ID", "") or "").strip()
        thread_id = (get_session_env("HERMES_SESSION_THREAD_ID", "") or "").strip()
    except Exception:
        return None, None

    if not chat_id:
        return None, None

    # Normal case: the turn is already in the Hermès-window thread. Adopt it so
    # the agent's Q&A + the feed pointer land where the user already is. No
    # thread creation, so this path doesn't need the live adapter/loop.
    if thread_id:
        return thread_id, (lambda _target, _name: thread_id)

    # Fallback: plain-channel turn — mint a thread under the channel so the
    # session still has a home. Needs the live adapter + gateway event loop.
    try:
        from gateway.run import _gateway_runner_ref
        runner = _gateway_runner_ref()
    except Exception:
        runner = None
    if runner is None:
        return None, None

    try:
        from gateway.config import Platform
        adapter = runner.adapters.get(Platform.DISCORD)
    except Exception:
        adapter = None
    if adapter is None:
        return None, None

    loop = getattr(runner, "_gateway_loop", None)
    if loop is None:
        return None, None

    def _thread_creator(target: str, name: str) -> Optional[str]:
        import asyncio
        try:
            return asyncio.run_coroutine_threadsafe(
                adapter.create_handoff_thread(target, name), loop
            ).result(timeout=15)
        except Exception as exc:
            logger.warning("session_spawn: create_handoff_thread failed: %s", exc)
            return None

    return chat_id, _thread_creator


def _resolve_discord_user_id() -> Optional[str]:
    """Return the requesting Discord user id for the current turn, or ``None``.

    Reads ``HERMES_SESSION_USER_ID`` from the gateway session context, gated on
    a Discord platform (the id is only meaningful as a Discord mention target).
    Persisted on the spawned row so ``feed.push_turn_change`` can @-mention the
    user when the session needs input.
    """
    try:
        from gateway.session_context import get_session_env
        platform = (get_session_env("HERMES_SESSION_PLATFORM", "") or "").strip().lower()
        if platform != "discord":
            return None
        return (get_session_env("HERMES_SESSION_USER_ID", "") or "").strip() or None
    except Exception:
        return None


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
    **_injected: Any,
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

    # ---- Resolve the live Discord thread context -------------------------
    # Best-effort: gives the spawned session a real project thread so the feed
    # can link <#thread> and text replies route back into the agent. Resolves
    # to (None, None) off Discord / when the gateway isn't live, in which case
    # the spawn proceeds threadless exactly as before.
    parent_chat_id, thread_creator = _resolve_discord_thread_context()

    # ---- Capture the requesting Discord user (for @-mention on attention) -
    # Read from the per-turn gateway session context, same mechanism as the
    # thread resolution above. Without this the spawned row has no user id and
    # the WAITING_USER thread notice can't ping anyone.
    discord_user_id = _resolve_discord_user_id()

    # ---- Build SpawnRequest and spawn ------------------------------------
    from session_orchestration.spawn import SpawnRequest, SpawnResult

    request = SpawnRequest(
        prompt=prompt,
        agent=agent,
        workdir=workdir,
        z_command=z_command,
        parent_chat_id=parent_chat_id,
        discord_user_id=discord_user_id,
    )

    try:
        result: SpawnResult = _spawn_fn(request, thread_creator=thread_creator)
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

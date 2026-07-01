"""
session_orchestration/spawn.py — Discord @hermes spawn flow.

Responsibilities
----------------
1. Parse a spawn request {prompt, agent, z_command, workdir} from a Discord event.
2. Derive a deterministic slug/hash session name from (agent, workdir, prompt).
3. Select the right ``AgentAdapter`` by agent name.
4. Call ``adapter.launch()`` → ``SessionHandle``.
5. Write the initial registry row (source="spawn") via ``registry.upsert()``.
   Spawn is the initial row creator — it owns the new task_id and calls
   ``upsert()`` directly (before the cron watcher can track it).  The watcher
   then becomes the sole mutator going forward.  This matches T002's design:
   spawn creates the row; adopt (T014) uses ``enqueue_intent``.
6. Create or attach a Discord project thread (via the platform adapter's
   ``create_handoff_thread``).  Records the thread_id on the registry row.
7. Seed the first prompt via ``SessionRelay.send_message()``.

Session name derivation
-----------------------
The tmux session name and task_id are deterministic from (agent, workdir, prompt):

    prefix = hermes-{agent[:6]}-{workdir_hash[:5]}-{prompt_hash[:5]}

This is reproducible and collision-resistant enough for O(hundreds) of sessions.
The full task_id is the session_handle.session_id (UUID) from the adapter.

Adapter selection
-----------------
``get_adapter(agent_name)`` maps the string from the Discord command to a
concrete adapter instance.  Supported names: "claude", "claude-code",
"omp", "omp-adapter".  Unknown names raise ``UnknownAgentError``.

Configuration
-------------
Gated on ``session_orchestration.enabled`` in Hermes config.  When disabled,
``handle_spawn_command`` returns an error string immediately without any
side effects.

Thread routing
--------------
``SpawnResult.thread_id`` is the Discord thread id attached to this task.  The
drive loop (T012) routes replies in that thread to the relay.  When
``create_handoff_thread`` fails or is unavailable, ``thread_id`` is ``None``
and replies must be routed by ``task_id`` directly.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from session_orchestration.adapters.base import AgentAdapter
from session_orchestration.registry import (
    SessionOrchestrationRegistry,
    canonical_repo_id,
)
from session_orchestration.relay import SessionRelay
from session_orchestration.types import SessionHandle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class UnknownAgentError(ValueError):
    """Raised when the requested agent name is not registered."""


class SpawnDisabledError(RuntimeError):
    """Raised when session_orchestration.enabled is False."""


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

#: Canonical name → adapter class.  Imported lazily so that a broken
#: binary (e.g. omp not installed) does not crash the gateway at import time.
_ADAPTER_CLASSES: dict[str, str] = {
    "claude": "session_orchestration.adapters.claude_code.ClaudeCodeAdapter",
    "claude-code": "session_orchestration.adapters.claude_code.ClaudeCodeAdapter",
    "omp": "session_orchestration.adapters.omp.OmpAdapter",
    "omp-adapter": "session_orchestration.adapters.omp.OmpAdapter",
}


def get_adapter(agent_name: str) -> AgentAdapter:
    """Return a fresh adapter instance for ``agent_name``.

    Parameters
    ----------
    agent_name:
        Case-insensitive agent identifier.  Supported: "claude",
        "claude-code", "omp", "omp-adapter".

    Raises
    ------
    UnknownAgentError
        When ``agent_name`` is not in the registry.
    """
    key = agent_name.lower().strip()
    class_path = _ADAPTER_CLASSES.get(key)
    if class_path is None:
        known = ", ".join(sorted(_ADAPTER_CLASSES))
        raise UnknownAgentError(
            f"Unknown agent {agent_name!r}. Known adapters: {known}"
        )
    module_path, class_name = class_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


# ---------------------------------------------------------------------------
# Session name derivation
# ---------------------------------------------------------------------------


def derive_session_name(agent: str, workdir: str, prompt: str) -> str:
    """Return a deterministic tmux session name for this (agent, workdir, prompt).

    Format: ``hermes-{agent[:6]}-{workdir_hash[:5]}-{prompt_hash[:5]}``

    The name is stable across restarts for identical inputs, collision-
    resistant at O(hundreds) of sessions, and tmux-safe (only alphanumeric
    and hyphens, max ~32 chars).
    """
    safe_agent = "".join(c if c.isalnum() else "" for c in agent.lower())[:6] or "agent"
    wd_hash = hashlib.sha256(workdir.encode()).hexdigest()[:5]
    pr_hash = hashlib.sha256(prompt.encode()).hexdigest()[:5]
    return f"hermes-{safe_agent}-{wd_hash}-{pr_hash}"


# ---------------------------------------------------------------------------
# SpawnRequest / SpawnResult
# ---------------------------------------------------------------------------


@dataclass
class SpawnRequest:
    """Parsed Discord spawn request.

    Fields
    ------
    prompt:
        The initial prompt to deliver to the agent.
    agent:
        Agent name ("claude", "omp", …).
    workdir:
        Absolute path to the working directory.
    z_command:
        Optional z-harness slash command prefix (e.g. ``"/z-plan"``) to
        prepend to the prompt.
    discord_user_id:
        The Discord user id who issued the command (stored on the row for
        T012 thread routing and for attribution).
    feed_channel_id:
        The Discord channel id to post feed messages to (from config).
    parent_chat_id:
        The Discord channel id under which to create a project thread.
        Typically the same as ``feed_channel_id`` or a dedicated projects
        channel.
    """

    prompt: str
    agent: str
    workdir: str
    z_command: Optional[str] = None
    discord_user_id: Optional[str] = None
    feed_channel_id: Optional[str] = None
    parent_chat_id: Optional[str] = None


@dataclass
class SpawnResult:
    """Result of a successful spawn.

    Fields
    ------
    task_id:
        The registry row primary key (= ``handle.session_id``).
    handle:
        The ``SessionHandle`` returned by ``adapter.launch()``.
    session_name:
        The tmux session name (deterministic slug).
    thread_id:
        The Discord thread id attached to this task, or ``None`` if thread
        creation failed / was not available.
    """

    task_id: str
    handle: SessionHandle
    session_name: str
    thread_id: Optional[str]


# ---------------------------------------------------------------------------
# Core spawn orchestration
# ---------------------------------------------------------------------------


def spawn_session(
    request: SpawnRequest,
    *,
    adapter: Optional[AgentAdapter] = None,
    registry: Optional[SessionOrchestrationRegistry] = None,
    relay: Optional[SessionRelay] = None,
    thread_creator=None,
) -> SpawnResult:
    """Orchestrate a complete spawn: launch → registry row → thread → seed.

    Parameters
    ----------
    request:
        Parsed spawn request.
    adapter:
        Override the adapter (injected in tests).  When ``None``, resolved
        via ``get_adapter(request.agent)``.
    registry:
        Override the registry (injected in tests).  When ``None``, a default
        ``SessionOrchestrationRegistry()`` is created.
    relay:
        Override the relay (injected in tests).  When ``None``, a
        ``SessionRelay`` is built from the resolved registry and adapter.
    thread_creator:
        Callable ``(parent_chat_id: str, name: str) -> Optional[str]``.
        Should create a Discord thread and return its id, or ``None`` on
        failure.  Injected in tests; in production this is the platform
        adapter's ``create_handoff_thread``.

    Returns
    -------
    SpawnResult
        Contains the task_id, handle, session_name, and thread_id.

    Raises
    ------
    UnknownAgentError
        When ``request.agent`` is not in the adapter registry and no
        ``adapter`` override was passed.
    """
    # Step 1 — Resolve adapter
    _adapter = adapter if adapter is not None else get_adapter(request.agent)

    # Step 2 — Derive deterministic session name
    session_name = derive_session_name(request.agent, request.workdir, request.prompt)

    # Step 3 — Build effective prompt (prepend z_command if given)
    effective_prompt = request.prompt
    if request.z_command:
        effective_prompt = f"{request.z_command}\n{request.prompt}"

    # Step 4 — Launch the adapter (tmux session)
    logger.info(
        "spawn: launching adapter=%s session=%s workdir=%s",
        type(_adapter).__name__,
        session_name,
        request.workdir,
    )
    handle: SessionHandle = _adapter.launch(request.workdir, effective_prompt)

    # Step 5 — Write the initial registry row (spawn is the row creator)
    _registry = registry if registry is not None else SessionOrchestrationRegistry()

    task_id = handle.session_id
    repo = canonical_repo_id(workdir=request.workdir)

    _registry.upsert(
        task_id,
        agent=request.agent,
        source="spawn",
        run_id=None,  # no z-harness run_id at spawn time; ingest correlates later
        repo=repo,
        tmux_session=handle.tmux_session,
        workdir=request.workdir,
        state="RUNNING",
        last_output_ts=handle.launch_ts.timestamp(),
        # Persist the requesting user so attention notices can @-mention them.
        # Previously dropped here even though SpawnRequest carried it, leaving
        # every row's discord_user_id empty and the WAITING_USER ping silent.
        discord_user_id=request.discord_user_id,
    )
    logger.info("spawn: registry row created task_id=%s repo=%s", task_id, repo)

    # Step 6 — Create/attach a Discord project thread
    thread_id: Optional[str] = None
    if thread_creator is not None and request.parent_chat_id:
        try:
            thread_name = f"{request.agent}: {request.prompt[:40]}"
            thread_id = thread_creator(request.parent_chat_id, thread_name)
            if thread_id:
                # Record the thread on the registry row so the drive loop can route
                _registry.upsert(
                    task_id,
                    agent=request.agent,
                    source="spawn",
                    discord_thread_id=thread_id,
                )
                logger.info(
                    "spawn: project thread created thread_id=%s task_id=%s",
                    thread_id,
                    task_id,
                )
        except Exception as exc:
            logger.warning(
                "spawn: thread creation failed for task_id=%s: %s",
                task_id,
                exc,
            )
            # Non-fatal: proceed without a thread

    # Step 7 — Seed the first prompt via relay
    _relay = relay
    if _relay is None:
        _relay = SessionRelay(_registry, _adapter)

    _relay.send_message(task_id, handle, effective_prompt)
    logger.info("spawn: first prompt seeded task_id=%s", task_id)

    return SpawnResult(
        task_id=task_id,
        handle=handle,
        session_name=session_name,
        thread_id=thread_id,
    )


# ---------------------------------------------------------------------------
# Gateway command handler
# ---------------------------------------------------------------------------


def parse_spawn_args(args_text: str) -> dict[str, str]:
    """Parse ``/so-spawn`` argument text into keyword fields.

    Expects arguments in the form::

        agent=<name> workdir=<path> [z_command=<cmd>] <prompt...>

    Returns a dict with keys: ``agent``, ``workdir``, ``prompt``
    (and optionally ``z_command``).  Returns an empty dict on parse failure.

    The prompt is everything after the last recognised ``key=value`` token.
    Any remaining text after the last ``key=value`` pair is the prompt.
    """
    import shlex
    result: dict[str, str] = {}
    prompt_parts: list[str] = []

    try:
        tokens = shlex.split(args_text)
    except ValueError:
        tokens = args_text.split()

    _KNOWN_KEYS = {"agent", "workdir", "z_command", "z-command"}
    in_prompt = False

    for token in tokens:
        if not in_prompt and "=" in token:
            k, _, v = token.partition("=")
            key = k.strip().lower().replace("-", "_")
            if key in _KNOWN_KEYS:
                result[key] = v.strip()
                continue
        # Once we hit a non-key=value token, the rest is the prompt
        in_prompt = True
        prompt_parts.append(token)

    if prompt_parts:
        result["prompt"] = " ".join(prompt_parts)

    return result


async def handle_spawn_command(
    event,
    args_text: str,
    *,
    config=None,
    platform_adapter=None,
    registry: Optional[SessionOrchestrationRegistry] = None,
    relay: Optional[SessionRelay] = None,
    _adapter_override: Optional[AgentAdapter] = None,
) -> str:
    """Handle a ``/so-spawn`` gateway command.

    Parameters
    ----------
    event:
        The gateway ``MessageEvent`` (used to extract user_id, chat_id).
    args_text:
        Everything after ``/so-spawn`` in the message.
    config:
        The Hermes config dict (used to gate on
        ``session_orchestration.enabled`` and read ``feed_channel_id``).
    platform_adapter:
        The platform adapter for the originating platform (used to call
        ``create_handoff_thread``).  May be ``None`` in tests.
    registry:
        Override the registry (injected in tests).
    relay:
        Override the relay (injected in tests).
    _adapter_override:
        Override the agent adapter (injected in tests).

    Returns
    -------
    str
        A human-readable reply suitable for posting to Discord.
    """
    # Gate on session_orchestration.enabled
    try:
        from hermes_cli.config import cfg_get
        so_enabled = cfg_get(config, "session_orchestration", "enabled", default=False)
    except Exception:
        so_enabled = False

    if not so_enabled:
        return (
            "Session orchestration is disabled. "
            "Set `session_orchestration.enabled: true` in Hermes config to enable."
        )

    # Parse args
    parsed = parse_spawn_args(args_text)
    agent = parsed.get("agent", "").strip()
    workdir = parsed.get("workdir", "").strip()
    prompt = parsed.get("prompt", "").strip()
    z_command = parsed.get("z_command", "").strip() or None

    missing = []
    if not agent:
        missing.append("agent=<name>")
    if not workdir:
        missing.append("workdir=<path>")
    if not prompt:
        missing.append("<prompt>")
    if missing:
        return (
            f"Usage: /so-spawn agent=<name> workdir=<path> [z_command=<cmd>] <prompt>\n"
            f"Missing: {', '.join(missing)}\n"
            f"Supported agents: {', '.join(sorted(_ADAPTER_CLASSES))}"
        )

    # Extract caller metadata
    source = getattr(event, "source", None)
    discord_user_id = str(getattr(source, "user_id", "") or "") or None
    parent_chat_id = str(getattr(source, "chat_id", "") or "") or None

    try:
        from hermes_cli.config import cfg_get
        feed_channel_id = cfg_get(config, "session_orchestration", "feed_channel_id", default=None)
    except Exception:
        feed_channel_id = None

    # Build thread_creator from platform adapter
    thread_creator = None
    if platform_adapter is not None and parent_chat_id:
        _pa = platform_adapter

        async def _create_thread_async(chat_id: str, name: str) -> Optional[str]:
            import asyncio
            try:
                return await _pa.create_handoff_thread(chat_id, name)
            except Exception as exc:
                logger.warning("spawn: create_handoff_thread failed: %s", exc)
                return None

        def _sync_thread_creator(chat_id: str, name: str) -> Optional[str]:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    future = asyncio.run_coroutine_threadsafe(
                        _create_thread_async(chat_id, name), loop
                    )
                    return future.result(timeout=10)
                return loop.run_until_complete(_create_thread_async(chat_id, name))
            except Exception as exc:
                logger.warning("spawn: thread create wrapper failed: %s", exc)
                return None

        thread_creator = _sync_thread_creator

    request = SpawnRequest(
        prompt=prompt,
        agent=agent,
        workdir=workdir,
        z_command=z_command,
        discord_user_id=discord_user_id,
        feed_channel_id=feed_channel_id,
        parent_chat_id=parent_chat_id,
    )

    try:
        result = spawn_session(
            request,
            adapter=_adapter_override,
            registry=registry,
            relay=relay,
            thread_creator=thread_creator,
        )
    except UnknownAgentError as exc:
        return f"Unknown agent: {exc}"
    except Exception as exc:
        logger.exception("spawn: unexpected error for agent=%s workdir=%s", agent, workdir)
        return f"Spawn failed: {exc}"

    # Build reply
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
# /so-stop and /so-restart command handlers
# ---------------------------------------------------------------------------


def _parse_task_id(args_text: str) -> Optional[str]:
    """Extract ``task_id=<value>`` from ``args_text``, or return ``None``."""
    for token in args_text.split():
        if token.startswith("task_id="):
            value = token.partition("=")[2].strip()
            return value if value else None
    return None


def handle_stop_command(
    event,
    args_text: str,
    *,
    config=None,
    registry: Optional[SessionOrchestrationRegistry] = None,
) -> str:
    """Handle a ``/so-stop`` gateway command.

    Parses a ``task_id=<id>`` argument from ``args_text``, enqueues a
    terminate intent (``restart=False``), and returns a confirmation string.
    Returns an error string (does not raise) when ``task_id`` is absent.

    Parameters
    ----------
    event:
        The gateway ``MessageEvent`` (present for interface symmetry with
        ``handle_spawn_command``; not used directly).
    args_text:
        Everything after ``/so-stop`` in the message.
    config:
        Hermes config dict (reserved for future gating; not used today).
    registry:
        Override the registry (injected in tests).  When ``None``, a
        default ``SessionOrchestrationRegistry()`` is created.
    """
    task_id = _parse_task_id(args_text)
    if not task_id:
        return (
            "Usage: /so-stop task_id=<id>\n"
            "Missing required argument: task_id"
        )

    _registry = registry if registry is not None else SessionOrchestrationRegistry()
    _registry.enqueue_terminate(task_id, restart=False)
    logger.info("so-stop: enqueued terminate task_id=%s restart=False", task_id)
    return f"Stop requested for task `{task_id}`. The watcher will kill the session at its next tick."


def handle_restart_command(
    event,
    args_text: str,
    *,
    config=None,
    registry: Optional[SessionOrchestrationRegistry] = None,
) -> str:
    """Handle a ``/so-restart`` gateway command.

    Parses a ``task_id=<id>`` argument from ``args_text``, enqueues a
    terminate intent (``restart=True``), and returns a confirmation string.
    Returns an error string (does not raise) when ``task_id`` is absent.

    Parameters
    ----------
    event:
        The gateway ``MessageEvent`` (present for interface symmetry with
        ``handle_spawn_command``; not used directly).
    args_text:
        Everything after ``/so-restart`` in the message.
    config:
        Hermes config dict (reserved for future gating; not used today).
    registry:
        Override the registry (injected in tests).  When ``None``, a
        default ``SessionOrchestrationRegistry()`` is created.
    """
    task_id = _parse_task_id(args_text)
    if not task_id:
        return (
            "Usage: /so-restart task_id=<id>\n"
            "Missing required argument: task_id"
        )

    _registry = registry if registry is not None else SessionOrchestrationRegistry()
    _registry.enqueue_terminate(task_id, restart=True)
    logger.info("so-restart: enqueued terminate task_id=%s restart=True", task_id)
    return (
        f"Restart requested for task `{task_id}`. "
        "The watcher will kill and re-spawn the session at its next tick."
    )

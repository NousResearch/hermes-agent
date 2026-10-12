"""Messaging adapter for native plugin session commands.

The host owns source authorization, routing, profile scope and foreground
identity. Read requests never initialize a route or inspect a live agent.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from agent.i18n import t

logger = logging.getLogger("gateway.run")


def _resolve_profile_home():
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()).resolve()


def _load_command_config():
    from hermes_cli.session_commands import load_session_command_config

    config = load_session_command_config()
    if not isinstance(config, dict):
        raise ValueError("could not read global configuration")
    return config


def _run_handler(handler, request, *, agent, session_db, session_id,
                 ensure_session, profile_home, config):
    from hermes_cli.session_commands import make_session_context, run_session_command

    context = make_session_context(
        agent=agent,
        session_db=session_db,
        session_id=session_id,
        ensure_session=ensure_session,
        config_loader=lambda: config,
        profile_home=profile_home,
    )
    return run_session_command(handler, request, context)


def _run_read_request(handler, request, session_id, profile_home, config):
    from hermes_state import SessionDB

    path = Path(profile_home) / "state.db"
    session_db = None
    try:
        if session_id and path.exists():
            session_db = SessionDB(db_path=path, read_only=True)
        return _run_handler(
            handler, request, agent=None, session_db=session_db,
            session_id=session_id, ensure_session=None,
            profile_home=profile_home, config=config,
        )
    finally:
        if session_db is not None:
            session_db.close()


def _cached_agent(runner, session_key):
    cache = getattr(runner, "_agent_cache", None)
    if cache is None:
        return None
    lock = getattr(runner, "_agent_cache_lock", None)
    if lock is None:
        raise ValueError("agent cache is not available")
    with lock:
        cached = cache.get(session_key)
    return (cached[0] if cached else None) if isinstance(cached, tuple) else cached


def _validate_database(session_db, profile_home):
    path = getattr(session_db, "db_path", None)
    if not path:
        raise ValueError("session database is not available")
    if Path(path).resolve() != (Path(profile_home) / "state.db").resolve():
        raise ValueError("session database profile mismatch; refusing to mutate")


def _live_ensure(agent, session_db, session_id, profile_home):
    def ensure_session():
        if str(getattr(agent, "session_id", "") or "") != session_id:
            raise ValueError("live session identity mismatch; refusing to mutate")
        if getattr(agent, "_session_db", None) is not session_db:
            raise ValueError("live session database changed; refusing to mutate")
        _validate_database(session_db, profile_home)
        ensure = getattr(agent, "_ensure_db_session", None)
        if not callable(ensure):
            raise ValueError("canonical session ensure callback is not available")
        ensure()
        if str(getattr(agent, "session_id", "") or "") != session_id:
            raise ValueError("live session identity changed; refusing to mutate")
    return ensure_session


def _cold_ensure(session_db, session_id, source):
    def ensure_session():
        if session_db.get_session(session_id) is not None:
            return
        session_db.create_session(
            session_id=session_id,
            source=source.platform.value if source.platform else "unknown",
            user_id=source.user_id,
            chat_id=source.chat_id,
            chat_type=source.chat_type,
            thread_id=source.thread_id,
        )
    return ensure_session


def _run_write_request(runner, source, handler, request, session_key,
                       session_id, profile_home, config):
    agent = _cached_agent(runner, session_key)
    if agent is not None:
        if str(getattr(agent, "session_id", "") or "") != session_id:
            raise ValueError("live session identity mismatch; refusing to mutate")
        session_db = getattr(agent, "_session_db", None)
        _validate_database(session_db, profile_home)
        return _run_handler(
            handler, request, agent=agent, session_db=session_db,
            session_id=session_id,
            ensure_session=_live_ensure(agent, session_db, session_id, profile_home),
            profile_home=profile_home, config=config,
        )

    owned_db = None
    try:
        if getattr(runner.config, "multiplex_profiles", False):
            from hermes_state import SessionDB
            owned_db = SessionDB(db_path=Path(profile_home) / "state.db")
            session_db = owned_db
        else:
            session_db = getattr(getattr(runner, "_session_db", None), "_db", None)
        _validate_database(session_db, profile_home)
        return _run_handler(
            handler, request, agent=None, session_db=session_db,
            session_id=session_id, ensure_session=_cold_ensure(session_db, session_id, source),
            profile_home=profile_home, config=config,
        )
    finally:
        if owned_db is not None:
            owned_db.close()


async def dispatch_gateway_plugin_command(runner, event, source, command):
    """Dispatch a plugin slash command with the host's authorization and context."""
    # Underscores normalize to hyphens so Telegram's underscored autocomplete form
    # matches plugin commands registered with hyphens.
    if command:
        session_handler = False
        try:
            from hermes_cli.plugins import get_plugin_command_handler
            from hermes_cli.session_commands import is_session_command
            from gateway.session import build_session_context

            plugin_handler = get_plugin_command_handler(command.replace("_", "-"))
            if plugin_handler:
                session_handler = is_session_command(plugin_handler)
                denied = runner._check_slash_access(source, command.replace("_", "-"))
                if denied is not None:
                    return True, denied, command
                if session_handler:
                    result = await dispatch_gateway_session_command(runner, event, plugin_handler)
                    return True, str(result) if result else None, command
                # This dispatch precedes the agent-turn path's HERMES_SESSION_* binding.
                # No session entry exists yet, so derive its key from the source. Sync
                # handlers run on the gateway pool with ContextVars, avoiding loop I/O
                # that can starve the liveness watchdog (#108698, #105279).
                plugin_context = build_session_context(source, runner.config)
                plugin_context.session_key = runner._session_key_for_source(source)
                user_args = event.get_command_args().strip()
                with runner._session_env_scope(plugin_context):
                    if asyncio.iscoroutinefunction(plugin_handler):
                        result = await plugin_handler(user_args)
                    else:
                        result = await runner._run_in_executor_with_context(plugin_handler, user_args)
                        if asyncio.iscoroutine(result):
                            result = await result
                return True, str(result) if result else None, command
        except Exception as exc:
            logger.warning("Plugin command dispatch failed: %s", exc, exc_info=True)
            if session_handler:
                return True, f"Session command error: {exc}", command
    return False, None, command


async def dispatch_gateway_session_command(runner, event, handler):
    """Run one recognized session command; errors are replies, never model input."""
    from hermes_cli.session_commands import prepare_session_command
    from gateway.session import build_session_context

    try:
        request = await runner._run_in_executor_with_context(
            prepare_session_command, handler, event.get_command_args().strip())
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        logger.exception("Plugin session command parsing failed")
        return f"Session command error: {exc}"

    try:
        source = event.source
        session_key = runner._session_key_for_source(source)
        if runner._is_session_running(session_key):
            return t("gateway.busy.slash_rejected", command=handler.name)
        async with runner._async_profile_scope_for_source(source):
            profile_home = await runner._run_in_executor_with_context(_resolve_profile_home)
            # Validate configuration before a write route can create its canonical row.
            config = await runner._run_in_executor_with_context(_load_command_config)
            if request.write:
                entry = await runner.async_session_store.get_or_create_session(source)
                session_id = str(getattr(entry, "session_id", "") or "")
                if not session_id.strip():
                    raise ValueError("a non-empty exact session ID is required")
            else:
                session_id = str(await runner.async_session_store.peek_session_id(
                    session_key, peek_only=True) or "")

            plugin_context = build_session_context(source, runner.config)
            plugin_context.session_key = session_key
            plugin_context.session_id = session_id
            with runner._session_env_scope(plugin_context):
                if request.write:
                    return await runner._run_in_executor_with_context(
                        _run_write_request, runner, source, handler, request, session_key,
                        session_id, profile_home, config)
                return await runner._run_in_executor_with_context(
                    _run_read_request, handler, request, session_id, profile_home, config)
    except Exception as exc:
        logger.exception("Plugin session command execution failed")
        return f"Session command error: {exc}"

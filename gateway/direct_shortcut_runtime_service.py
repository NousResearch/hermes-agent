"""Shared direct-shortcut runtime orchestration helpers for the gateway."""

from __future__ import annotations

from typing import Any, Iterable

from gateway.direct_control_router import DirectControlRouter
from gateway.direct_shortcuts import run_direct_shortcut_handlers
from gateway.session import build_session_context


def prime_session_env_for_direct_shortcuts(runner: Any, source: Any) -> None:
    """Populate session env so tool-backed direct shortcuts can run off the main path."""
    if not source:
        return
    session_entry = runner.session_store.get_or_create_session(source)
    admin_user_ids = runner._configured_admin_user_ids(source.platform)
    is_admin_user = runner._is_admin_user(source) if admin_user_ids else None
    context = build_session_context(
        source,
        runner.config,
        session_entry,
        admin_user_ids=admin_user_ids,
        is_admin_user=is_admin_user,
    )
    runner._set_session_env(context)


def get_direct_control_router(
    runner: Any,
    *,
    router_cls=DirectControlRouter,
) -> Any:
    router = getattr(runner, "_direct_control_router", None)
    if router is None:
        router = router_cls(runner)
        runner._direct_control_router = router
    return router


def try_handle_direct_gateway_shortcuts(
    runner: Any,
    event: Any,
    *,
    prepare_session_env: bool = False,
    conversation_history: Iterable[dict[str, Any]] | None = None,
    logger=None,
    session_env_primer=prime_session_env_for_direct_shortcuts,
    handler_runner=run_direct_shortcut_handlers,
) -> str | None:
    source = getattr(event, "source", None)
    if prepare_session_env and source is not None:
        try:
            session_env_primer(runner, source)
        except Exception as exc:
            if logger is not None:
                logger.debug("Failed to prime direct-shortcut session env: %s", exc)
    return handler_runner(
        runner,
        event,
        conversation_history=conversation_history,
        logger=logger,
    )

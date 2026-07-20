"""Production-path unit tests for direct_shortcut_runtime_service (WIRED)."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from gateway.config import Platform
from gateway.session import SessionSource


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.QQ_NAPCAT,
        user_id="179033731",
        user_name="發發發",
        chat_id="179033731",
        chat_type="dm",
    )


def test_prime_session_env_for_direct_shortcuts_builds_context_and_sets_env():
    from gateway.direct_shortcut_runtime_service import prime_session_env_for_direct_shortcuts

    source = _make_source()
    session_entry = SimpleNamespace(
        session_key="sess-key",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    set_env = MagicMock()
    config = SimpleNamespace(
        get_connected_platforms=lambda: [Platform.QQ_NAPCAT],
        get_home_channel=lambda platform: None,
        get_session_isolation=lambda platform: (True, False),
    )
    runner = SimpleNamespace(
        session_store=SimpleNamespace(get_or_create_session=lambda current_source: session_entry),
        _configured_admin_user_ids=lambda platform: ["179033731"],
        _is_admin_user=lambda current_source: True,
        config=config,
        _set_session_env=set_env,
    )

    prime_session_env_for_direct_shortcuts(runner, source)

    set_env.assert_called_once()
    context = set_env.call_args.args[0]
    assert context.source == source
    assert context.admin_user_ids == ["179033731"]
    assert context.is_admin_user is True
    assert context.session_key == "sess-key"
    assert context.session_id == "sess-1"


def test_get_direct_control_router_caches_router_instance():
    from gateway.direct_shortcut_runtime_service import get_direct_control_router

    created = []

    class FakeRouter:
        def __init__(self, owner):
            created.append(owner)

    runner = SimpleNamespace()

    first = get_direct_control_router(runner, router_cls=FakeRouter)
    second = get_direct_control_router(runner, router_cls=FakeRouter)

    assert first is second
    assert created == [runner]


def test_try_handle_direct_gateway_shortcuts_primes_env_then_runs_handlers():
    from gateway.direct_shortcut_runtime_service import try_handle_direct_gateway_shortcuts

    source = _make_source()
    event = SimpleNamespace(source=source)
    prime = MagicMock()
    runner = SimpleNamespace()
    logger = MagicMock()

    result = try_handle_direct_gateway_shortcuts(
        runner,
        event,
        prepare_session_env=True,
        conversation_history=[{"role": "user", "content": "在吗"}],
        logger=logger,
        session_env_primer=prime,
        handler_runner=lambda current_runner, current_event, **kwargs: "handled",
    )

    assert result == "handled"
    prime.assert_called_once_with(runner, source)


def test_try_handle_direct_gateway_shortcuts_ignores_prime_error_and_continues():
    from gateway.direct_shortcut_runtime_service import try_handle_direct_gateway_shortcuts

    source = _make_source()
    event = SimpleNamespace(source=source)
    runner = SimpleNamespace()
    prime = MagicMock(side_effect=RuntimeError("boom"))
    logger = MagicMock()

    result = try_handle_direct_gateway_shortcuts(
        runner,
        event,
        prepare_session_env=True,
        conversation_history=None,
        logger=logger,
        session_env_primer=prime,
        handler_runner=lambda current_runner, current_event, **kwargs: "handled-anyway",
    )

    assert result == "handled-anyway"
    logger.debug.assert_called_once()

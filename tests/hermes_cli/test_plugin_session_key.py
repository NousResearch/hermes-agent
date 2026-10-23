"""A plugin captures a task-local route for later, explicitly targeted delivery."""
import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor

from gateway.session_context import scoped_session_key
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def context():
    return PluginContext(PluginManifest(name="route-observer"), PluginManager())


def test_session_key_is_empty_outside_a_bound_conversation(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_KEY", "stale-other-conversation")
    monkeypatch.setenv("HERMES_SESSION_ID", "database-id-is-not-a-route")
    assert context().session_key() == ""


def test_nested_session_keys_restore_the_calling_conversation():
    ctx = context()
    with scoped_session_key("profile-one/telegram/topic-1"):
        assert ctx.session_key() == "profile-one/telegram/topic-1"
        with scoped_session_key("profile-two/desktop/tab-2"):
            assert ctx.session_key() == "profile-two/desktop/tab-2"
        assert ctx.session_key() == "profile-one/telegram/topic-1"
    assert ctx.session_key() == ""


def test_concurrent_conversations_capture_distinct_routes():
    ctx = context()

    async def run():
        ready = asyncio.Event()
        entered = 0

        async def capture(key):
            nonlocal entered
            with scoped_session_key(key):
                entered += 1
                if entered == 2:
                    ready.set()
                await ready.wait()
                return ctx.session_key()

        return await asyncio.gather(capture("first-profile/session"), capture("second-profile/session"))

    assert asyncio.run(run()) == ["first-profile/session", "second-profile/session"]


def test_a_captured_route_survives_turn_exit_without_borrowing_another_route():
    ctx = context()
    with scoped_session_key("original-chat"):
        key = ctx.session_key()
        copied = contextvars.copy_context()
    with scoped_session_key("another-chat"), ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(copied.run, ctx.session_key).result(timeout=5) == key
        assert executor.submit(ctx.session_key).result(timeout=5) == ""
        assert ctx.session_key() == "another-chat"


def test_async_delivery_capability_uses_the_calling_context():
    from gateway.session_context import clear_session_vars, set_session_vars
    ctx = context()
    tokens = set_session_vars(platform="api_server", session_key="api-route", async_delivery=False)
    try:
        assert ctx.session_key() == "api-route"
        assert ctx.session_supports_async_delivery() is False
    finally:
        clear_session_vars(tokens)
    tokens = set_session_vars(platform="desktop", session_key="desktop-route", async_delivery=True)
    try:
        assert ctx.session_supports_async_delivery() is True
    finally:
        clear_session_vars(tokens)


def test_finite_turns_do_not_offer_async_delivery_even_on_a_cli_surface():
    from gateway.session_context import clear_session_vars, set_session_vars
    from gateway.session_finite import finite_turn_scope
    tokens = set_session_vars(platform="cli", session_key="finite-route", async_delivery=True)
    try:
        with finite_turn_scope(True):
            assert context().session_supports_async_delivery() is False
    finally:
        clear_session_vars(tokens)

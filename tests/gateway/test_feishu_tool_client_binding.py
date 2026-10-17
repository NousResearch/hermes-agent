"""Tests for Feishu adapter tool-client binding lifecycle.

Verifies the adapter publishes its client into the profile-qualified binding
registry only after a successful connect, and that teardown is
generation-owned (compare-and-remove, per-profile process-wide generation
allocation) so a stale adapter cannot clear a newer adapter's binding —
including a replacement adapter instance for the same profile.

Also witnesses the current-main fatal-WS-thread supervisor (#73779): the
supervisor rebuilds ``self._client`` after a fatal WebSocket thread exit, so
publication semantics must cover supervisor-owned client replacement (not
just the initial connect) under the same profile scope.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import tools.feishu_client_binding as binding

_ADAPTER = "plugins.platforms.feishu.adapter"


def _make_adapter():
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = object.__new__(FeishuAdapter)
    adapter._client = object()
    adapter._tool_binding_generation = 0
    adapter._tool_binding_profile_key = None
    # Attributes the connect paths read before reaching the mocked steps.
    adapter._domain_name = "lark"
    adapter._app_id = "app-id"
    adapter._app_secret = "app-secret"
    adapter._loop = None
    return adapter


def test_publish_tool_clients_stamps_generation():
    adapter = _make_adapter()
    with patch(
        "tools.feishu_client_binding.publish", return_value=5
    ) as mock_publish, patch(
        "tools.feishu_client_binding.active_profile_key",
        return_value="/profiles/a",
    ):
        adapter._publish_tool_clients()
    mock_publish.assert_called_once_with(adapter._client, "/profiles/a")
    assert adapter._tool_binding_generation == 5
    assert adapter._tool_binding_profile_key == "/profiles/a"


def test_reconnect_increments_generation():
    """The registry-allocated generation grows across publications for the
    same profile, so a reconnect supersedes the previous binding."""
    binding.clear_all()
    adapter = _make_adapter()
    adapter._publish_tool_clients()
    first = adapter._tool_binding_generation
    adapter._publish_tool_clients()
    assert adapter._tool_binding_generation > first
    assert binding.resolve(adapter._tool_binding_profile_key) is adapter._client
    binding.clear_all()


def test_unpublish_uses_own_generation():
    adapter = _make_adapter()
    adapter._tool_binding_generation = 7
    adapter._tool_binding_profile_key = "/profiles/a"
    with patch("tools.feishu_client_binding.unpublish") as mock_unpublish:
        adapter._unpublish_tool_clients()
    mock_unpublish.assert_called_once_with(7, "/profiles/a")
    assert adapter._tool_binding_profile_key is None


def test_unpublish_without_publication_is_noop():
    adapter = _make_adapter()
    with patch("tools.feishu_client_binding.unpublish") as mock_unpublish:
        adapter._unpublish_tool_clients()
    mock_unpublish.assert_not_called()


# -- connect failure/success witnesses --------------------------------------
#
# A failed connect must never expose a tool client: both connect paths build
# the lark client early (``_prepare_client``), then run several more steps
# (bot hydration, transport setup) before publishing. These tests fail a
# connect in the middle — after ``self._client`` exists — and assert the
# registry stays empty, so a regression that publishes before the final step
# fails here instead of exposing a client with no live adapter behind it.


def _fail_connect_midway(adapter, connect_coro_factory):
    """Run a connect coroutine whose bot hydration fails after the lark
    client is built; return the raised exception."""
    async def _run():
        adapter._loop = asyncio.get_running_loop()
        with patch(f"{_ADAPTER}._sdk_domain", MagicMock(return_value=object())), \
             patch.object(adapter, "_build_lark_client", return_value=adapter._client), \
             patch.object(adapter, "_build_event_handler", return_value=object()), \
             patch.object(
                 adapter,
                 "_hydrate_bot_identity",
                 AsyncMock(side_effect=RuntimeError("hydrate failed")),
             ):
            await connect_coro_factory(adapter)

    with pytest.raises(RuntimeError, match="hydrate failed"):
        asyncio.run(_run())


def test_failed_websocket_connect_never_publishes():
    binding.clear_all()
    adapter = _make_adapter()
    with patch(f"{_ADAPTER}.FEISHU_WEBSOCKET_AVAILABLE", True):
        _fail_connect_midway(
            adapter,
            lambda a: a._connect_websocket(),
        )
    # self._client exists, but no binding is discoverable...
    assert adapter._client is not None
    assert binding.resolve() is None
    # ...and the adapter knows it published nothing.
    assert adapter._tool_binding_profile_key is None
    assert adapter._tool_binding_generation == 0
    binding.clear_all()


def test_failed_webhook_connect_never_publishes():
    binding.clear_all()
    adapter = _make_adapter()
    with patch(f"{_ADAPTER}.FEISHU_WEBHOOK_AVAILABLE", True):
        _fail_connect_midway(
            adapter,
            lambda a: a._connect_webhook(),
        )
    assert adapter._client is not None
    assert binding.resolve() is None
    assert adapter._tool_binding_profile_key is None
    assert adapter._tool_binding_generation == 0
    binding.clear_all()


def _run_successful_websocket_connect(adapter):
    """Run the REAL ``_connect_websocket`` with every transport step mocked
    to success; returns after the WS thread future completed."""
    client = adapter._client

    async def _run():
        adapter._loop = asyncio.get_running_loop()
        with patch(f"{_ADAPTER}.FEISHU_WEBSOCKET_AVAILABLE", True), \
             patch(f"{_ADAPTER}.lark", MagicMock()), \
             patch(f"{_ADAPTER}.FeishuWSClient", MagicMock()), \
             patch(f"{_ADAPTER}._run_official_feishu_ws_client", lambda *a, **k: None), \
             patch(f"{_ADAPTER}._sdk_domain", MagicMock(return_value=object())), \
             patch.object(adapter, "_build_lark_client", return_value=client), \
             patch.object(adapter, "_build_event_handler", return_value=object()), \
             patch.object(adapter, "_hydrate_bot_identity", AsyncMock()):
            await adapter._connect_websocket()
            await adapter._ws_future
        adapter._shutdown_sdk_executor()

    asyncio.run(_run())


def test_successful_websocket_connect_publishes():
    """Positive control: with every step succeeding through publication, the
    binding IS discoverable — proving the failure tests above are not
    vacuously passing."""
    binding.clear_all()
    adapter = _make_adapter()
    client = adapter._client
    _run_successful_websocket_connect(adapter)
    assert binding.resolve() is client
    binding.clear_all()


def test_stale_adapter_teardown_does_not_clear_newer_binding():
    """Adapter A (old generation) disconnects after B (newer generation)
    published — including across adapter *instances* for the same profile,
    which is where a per-adapter generation counter would collide.

    The compare-and-remove unpublish must leave B's binding intact.
    """
    binding.clear_all()
    adapter_a = _make_adapter()
    adapter_b = _make_adapter()

    adapter_a._publish_tool_clients()  # A1 published first
    adapter_b._publish_tool_clients()  # replacement A2 supersedes it
    assert adapter_b._tool_binding_generation != adapter_a._tool_binding_generation

    adapter_a._unpublish_tool_clients()  # A1's stale teardown
    assert binding.resolve(adapter_b._tool_binding_profile_key) is adapter_b._client
    binding.clear_all()


def test_matching_teardown_clears_own_binding():
    binding.clear_all()
    adapter = _make_adapter()
    adapter._publish_tool_clients()
    adapter._unpublish_tool_clients()
    assert binding.resolve() is None
    binding.clear_all()


# -- supervisor-owned replacement witnesses (#73779) -------------------------
#
# The fatal-WS-thread supervisor rebuilds ``self._client`` via the real
# ``_connect_websocket`` after a fatal thread exit. Publication semantics
# must therefore cover the supervisor path too: a successful restart
# advances ONLY the owning profile's binding/generation, and a failed
# restart keeps the previous good binding discoverable.


def test_supervisor_replacement_republishes_for_owning_profile_only():
    """Kill the WS thread future, run ``_supervise_websocket_thread()``, and
    prove the replacement client advances only the owning profile's
    binding/generation — the sibling profile's binding stays untouched."""
    binding.clear_all()
    adapter = _make_adapter()
    adapter._publish_tool_clients()  # the initial connect's binding
    profile_key = adapter._tool_binding_profile_key
    initial_gen = adapter._tool_binding_generation

    sibling_client = object()
    binding.publish(sibling_client, profile_key="/profiles/sibling")

    replacement_client = object()

    def _stop_supervisor_after_restart():
        # Bot hydration runs inside the restart's connect path; once the
        # supervisor has proven it can restart, end its loop.
        adapter._running = False

    async def _run():
        loop = asyncio.get_running_loop()
        adapter._loop = loop  # the restart path runs on the real adapter loop
        dead = loop.create_future()
        dead.set_exception(RuntimeError("fatal ws thread exit"))
        adapter._ws_future = dead
        adapter._running = True
        adapter._ws_client = object()  # link object still present → supervisor restarts
        adapter._ws_restart_backoff = 0.0
        with patch(f"{_ADAPTER}.FEISHU_WEBSOCKET_AVAILABLE", True), \
             patch(f"{_ADAPTER}.lark", MagicMock()), \
             patch(f"{_ADAPTER}.FeishuWSClient", MagicMock()), \
             patch(f"{_ADAPTER}._run_official_feishu_ws_client", lambda *a, **k: None), \
             patch(f"{_ADAPTER}._sdk_domain", MagicMock(return_value=object())), \
             patch.object(adapter, "_write_runtime_status_safe"), \
             patch.object(adapter, "_build_lark_client", return_value=replacement_client), \
             patch.object(adapter, "_build_event_handler", return_value=object()), \
             patch.object(
                 adapter,
                 "_hydrate_bot_identity",
                 AsyncMock(side_effect=_stop_supervisor_after_restart),
             ):
            await asyncio.wait_for(
                adapter._supervise_websocket_thread(), timeout=10.0
            )
        adapter._shutdown_sdk_executor()

    asyncio.run(_run())
    # The restart funneled through the real _connect_websocket, which
    # re-published at its successful end: newer generation, replacement
    # client — and only for the owning profile.
    assert adapter._tool_binding_generation > initial_gen
    assert binding.resolve(profile_key) is replacement_client
    assert binding.resolve(profile_key="/profiles/sibling") is sibling_client
    binding.clear_all()


def test_failed_supervisor_restart_keeps_previous_binding():
    """A supervisor restart that fails mid-way must not retract anything:
    the previous good binding stays discoverable for DM tool calls until a
    restart actually succeeds (no regression of #73779's failed-restart
    behavior)."""
    binding.clear_all()
    adapter = _make_adapter()
    adapter._publish_tool_clients()
    profile_key = adapter._tool_binding_profile_key
    gen_before = adapter._tool_binding_generation
    original_client = adapter._client

    restarts = {"count": 0}

    def _fail_restart():
        restarts["count"] += 1
        adapter._ws_client = None  # exit the supervisor loop after this failure
        raise RuntimeError("restart failed")

    async def _run():
        loop = asyncio.get_running_loop()
        adapter._loop = loop  # the restart path runs on the real adapter loop
        dead = loop.create_future()
        dead.set_exception(RuntimeError("fatal ws thread exit"))
        adapter._ws_future = dead
        adapter._running = True
        adapter._ws_client = object()
        adapter._ws_restart_backoff = 0.0
        with patch(f"{_ADAPTER}.FEISHU_WEBSOCKET_AVAILABLE", True), \
             patch(f"{_ADAPTER}._sdk_domain", MagicMock(return_value=object())), \
             patch.object(adapter, "_build_lark_client", return_value=object()), \
             patch.object(adapter, "_build_event_handler", return_value=object()), \
             patch.object(adapter, "_write_runtime_status_safe"), \
             patch.object(adapter, "_hydrate_bot_identity", AsyncMock(side_effect=_fail_restart)):
            await asyncio.wait_for(
                adapter._supervise_websocket_thread(), timeout=10.0
            )
        adapter._shutdown_sdk_executor()

    asyncio.run(_run())
    assert restarts["count"] == 1
    # Failed restart: generation unchanged, previous binding intact.
    assert adapter._tool_binding_generation == gen_before
    assert binding.resolve(profile_key) is original_client
    binding.clear_all()


# -- disconnect() robustness witness -----------------------------------------
#
# A mid-teardown failure (webhook runner cleanup raising, a hard cancellation
# at an await point) must never leave this adapter's credential-bearing client
# discoverable in the registry: the binding is retracted as the FIRST step of
# disconnect(), before any await.


def _prepare_disconnect_state(adapter):
    adapter._running = True
    adapter._ws_supervisor = None
    adapter._app_lock_identity = None
    adapter._seen_message_ids = {}
    adapter._pending_text_batch_tasks = {}
    adapter._pending_media_batch_tasks = {}
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_counts = {}
    adapter._pending_media_batches = {}
    adapter._ws_client = None
    adapter._ws_thread_loop = None
    adapter._ws_future = None
    adapter._webhook_runner = None
    adapter._loop = None
    adapter._event_handler = None


def test_disconnect_unpublishes_even_when_teardown_raises():
    binding.clear_all()
    adapter = _make_adapter()
    adapter._publish_tool_clients()
    profile_key = adapter._tool_binding_profile_key
    assert binding.resolve(profile_key) is adapter._client
    _prepare_disconnect_state(adapter)

    async def _run():
        with patch.object(
            adapter,
            "_stop_webhook_server",
            AsyncMock(side_effect=RuntimeError("webhook cleanup exploded")),
        ):
            await adapter.disconnect()

    with pytest.raises(RuntimeError, match="webhook cleanup exploded"):
        asyncio.run(_run())
    # Teardown blew up mid-way, yet the binding is already gone.
    assert binding.resolve(profile_key) is None
    binding.clear_all()


def test_disconnect_unpublishes_on_clean_teardown():
    binding.clear_all()
    adapter = _make_adapter()
    adapter._publish_tool_clients()
    profile_key = adapter._tool_binding_profile_key
    _prepare_disconnect_state(adapter)

    async def _run():
        with patch.object(adapter, "_shutdown_sdk_executor"), \
             patch.object(adapter, "_persist_seen_message_ids"), \
             patch.object(adapter, "_mark_disconnected"):
            await adapter.disconnect()

    asyncio.run(_run())
    assert binding.resolve(profile_key) is None
    # The authority boundary is the registry retraction above, not nilling
    # the adapter's own reference: a send already past its ``_running`` check
    # keeps the old client and fails closed at the network layer, instead of
    # raising AttributeError on ``None.im`` mid-teardown.
    assert adapter._client is not None
    binding.clear_all()


def test_disconnect_retracts_the_connect_time_profile_not_the_teardown_scope(tmp_path):
    """disconnect() may run under a different contextvar scope than connect()
    (e.g. a multiplex shutdown driven from the launch profile). The retraction
    must use the key captured at publish time — re-deriving
    ``active_profile_key()`` at teardown would aim at whatever profile owns
    the disconnecting thread, leaving the owning profile's credential client
    discoverable in the registry exactly when the teardown-time profile has
    no matching generation."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    binding.clear_all()
    home_a = tmp_path / "profiles" / "a"
    home_b = tmp_path / "profiles" / "b"
    home_a.mkdir(parents=True)
    home_b.mkdir(parents=True)

    adapter = _make_adapter()
    token_a = set_hermes_home_override(str(home_a))
    try:
        adapter._publish_tool_clients()
    finally:
        reset_hermes_home_override(token_a)
    key_a = adapter._tool_binding_profile_key
    assert binding.resolve(key_a) is adapter._client

    # Teardown runs under profile B's scope, with a sibling binding there:
    # a mutated retraction that re-derives the active key would aim at B,
    # fail B's generation compare (or find nothing), and leave A published.
    token_b = set_hermes_home_override(str(home_b))
    try:
        sibling = object()
        binding.publish(sibling)
        _prepare_disconnect_state(adapter)

        async def _run():
            with patch.object(adapter, "_shutdown_sdk_executor"), \
                 patch.object(adapter, "_persist_seen_message_ids"), \
                 patch.object(adapter, "_mark_disconnected"):
                await adapter.disconnect()

        asyncio.run(_run())
        key_b = binding.active_profile_key()
        assert binding.resolve(key_b) is sibling
    finally:
        reset_hermes_home_override(token_b)

    assert binding.resolve(key_a) is None
    binding.clear_all()

"""Dispatch-outcome reporting for plugin-triggered gateway turns.

``inject_message()`` returns ``True`` as soon as the dispatch coroutine is
scheduled, which is indistinguishable from an unknown session, a rotated
session, revoked authorization, or a missing adapter -- all of which answer
``True`` and then silently drop the message. These tests pin the optional
``await_dispatch`` path that reports what dispatch actually decided.
"""

import asyncio
import concurrent.futures
import threading
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource
from hermes_cli.plugins import (
    GatewayInjectionResult,
    PluginContext,
    PluginManager,
    PluginManifest,
)

SESSION_KEY = "agent:main:telegram:dm:42"


def _entry(*, origin=True) -> SessionEntry:
    source = None
    if origin:
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="42",
            chat_type="dm",
            user_id="42",
            user_name="tester",
        )
    now = datetime.now()
    return SessionEntry(
        session_key=SESSION_KEY,
        session_id="session-42",
        created_at=now,
        updated_at=now,
        origin=source,
        platform=Platform.TELEGRAM,
    )


def _runner(entry, adapter=None, *, loop=None) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.session_store = SimpleNamespace()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        lookup_by_session_key=AsyncMock(return_value=entry),
    )
    runner.adapters = {Platform.TELEGRAM: adapter} if adapter else {}
    runner._profile_adapters = {}
    runner._running = True
    runner._draining = False
    runner._background_tasks = set()
    runner._is_user_authorized = MagicMock(return_value=True)
    runner._gateway_loop = loop
    return runner


def _context(manager: PluginManager) -> PluginContext:
    return PluginContext(
        PluginManifest(name="wake-plugin", key="wake-plugin", source="user"),
        manager,
    )


def _grant_injection(tmp_path, monkeypatch) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({
            "plugins": {"entries": {"wake-plugin": {"allow_gateway_injection": True}}}
        })
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))


# ---------------------------------------------------------------------------
# Dispatch-level outcomes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_returns_adopted_with_stored_session_identity():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    entry = _entry()
    runner = _runner(entry, adapter)

    result = await runner._dispatch_plugin_message_injection(
        session_key=SESSION_KEY,
        content="read the ledger",
        plugin_id="wake-plugin",
        correlation_id="evt-1",
    )

    assert isinstance(result, GatewayInjectionResult)
    assert result.accepted is True
    assert result.reason == "adopted"
    assert result.session_id == entry.session_id
    assert result.session_key == SESSION_KEY
    assert result.correlation_id == "evt-1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entry", "with_adapter", "expected"),
    [
        (None, True, "unknown_session"),
        (_entry(origin=False), True, "unknown_session"),
        (_entry(), False, "no_adapter"),
    ],
)
async def test_dispatch_names_each_unroutable_reason(entry, with_adapter, expected):
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(entry, adapter if with_adapter else None)

    result = await runner._dispatch_plugin_message_injection(
        session_key=SESSION_KEY,
        content="wake up",
        plugin_id="wake-plugin",
    )

    assert result.accepted is False
    assert result.reason == expected
    assert result.session_key == SESSION_KEY
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_names_unauthorized_separately_from_unknown_session():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(_entry(), adapter)
    runner._is_user_authorized = MagicMock(return_value=False)

    result = await runner._dispatch_plugin_message_injection(
        session_key=SESSION_KEY,
        content="wake up",
        plugin_id="wake-plugin",
    )

    assert result.accepted is False
    assert result.reason == "unauthorized"
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_names_unauthorized_when_the_check_itself_raises():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(_entry(), adapter)
    runner._is_user_authorized = MagicMock(side_effect=RuntimeError("allowlist down"))

    result = await runner._dispatch_plugin_message_injection(
        session_key=SESSION_KEY,
        content="wake up",
        plugin_id="wake-plugin",
    )

    assert result.accepted is False
    assert result.reason == "unauthorized"


@pytest.mark.asyncio
async def test_dispatch_names_gateway_draining():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(_entry(), adapter)
    runner._draining = True

    result = await runner._dispatch_plugin_message_injection(
        session_key=SESSION_KEY,
        content="wake up",
        plugin_id="wake-plugin",
    )

    assert result.accepted is False
    assert result.reason == "gateway_draining"


@pytest.mark.asyncio
async def test_correlation_id_is_stamped_on_the_dispatched_event():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    entry = _entry()
    runner = _runner(entry, adapter)

    await runner._dispatch_plugin_message_injection(
        session_key=SESSION_KEY,
        content="wake up",
        plugin_id="wake-plugin",
        correlation_id="evt-7",
    )

    event = adapter.handle_message.await_args.args[0]
    assert event.metadata["hermes_plugin_injection_id"] == "evt-7"
    assert event.metadata["gateway_session_id"] == entry.session_id
    assert event.metadata["gateway_session_strict"] is True


@pytest.mark.asyncio
async def test_correlation_id_absent_keeps_the_existing_metadata_shape():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    entry = _entry()
    runner = _runner(entry, adapter)

    await runner._dispatch_plugin_message_injection(
        session_key=SESSION_KEY,
        content="wake up",
        plugin_id="wake-plugin",
    )

    event = adapter.handle_message.await_args.args[0]
    assert event.metadata == {
        "hermes_plugin_id": "wake-plugin",
        "hermes_plugin_injection": True,
        "gateway_session_key": SESSION_KEY,
        "gateway_session_id": entry.session_id,
        "gateway_session_strict": True,
    }


@pytest.mark.asyncio
async def test_correlation_id_never_becomes_a_route():
    """A caller-supplied tag must not displace the host-resolved origin."""
    adapter = SimpleNamespace(handle_message=AsyncMock())
    entry = _entry()
    runner = _runner(entry, adapter)

    await runner._dispatch_plugin_message_injection(
        session_key=SESSION_KEY,
        content="wake up",
        plugin_id="wake-plugin",
        correlation_id="telegram:dm:999",
    )

    event = adapter.handle_message.await_args.args[0]
    assert event.source == entry.origin
    assert event.source is not entry.origin
    assert event.source.chat_id == "42"


# ---------------------------------------------------------------------------
# Scheduling-level contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_scheduling_still_returns_a_plain_bool():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    loop = asyncio.get_running_loop()
    runner = _runner(_entry(), adapter, loop=loop)

    scheduled = runner._schedule_plugin_message_injection(
        session_key=SESSION_KEY,
        content="wake up",
        plugin_id="wake-plugin",
    )

    assert scheduled is True
    await asyncio.gather(*list(runner._background_tasks), return_exceptions=True)


@pytest.mark.asyncio
async def test_await_dispatch_returns_an_awaitable_result_handle():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    entry = _entry()
    loop = asyncio.get_running_loop()
    runner = _runner(entry, adapter, loop=loop)

    handle = runner._schedule_plugin_message_injection(
        session_key=SESSION_KEY,
        content="wake up",
        plugin_id="wake-plugin",
        await_dispatch=True,
        correlation_id="evt-9",
    )
    result = await handle

    assert result.accepted is True
    assert result.reason == "adopted"
    assert result.session_id == entry.session_id
    assert result.correlation_id == "evt-9"


@pytest.mark.asyncio
async def test_await_dispatch_reports_a_dead_gateway_without_scheduling():
    runner = _runner(_entry(), SimpleNamespace(handle_message=AsyncMock()), loop=None)

    result = await runner._schedule_plugin_message_injection(
        session_key=SESSION_KEY,
        content="wake up",
        plugin_id="wake-plugin",
        await_dispatch=True,
    )

    assert result.accepted is False
    assert result.reason == "gateway_draining"


@pytest.mark.asyncio
async def test_await_dispatch_reports_internal_dispatch_failure():
    adapter = SimpleNamespace(
        handle_message=AsyncMock(side_effect=RuntimeError("adapter exploded"))
    )
    loop = asyncio.get_running_loop()
    runner = _runner(_entry(), adapter, loop=loop)

    result = await runner._schedule_plugin_message_injection(
        session_key=SESSION_KEY,
        content="wake up",
        plugin_id="wake-plugin",
        await_dispatch=True,
    )

    assert result.accepted is False
    assert result.reason == "internal_error"


@pytest.mark.asyncio
async def test_cancel_after_adapter_entry_cannot_claim_refusal_or_safe_retry():
    """Once the adapter holds the event, non-delivery is no longer provable.

    The adapter records its side effect before its next cancellation point, so a
    cancel landing here may have interrupted a wake that already happened.
    Reporting that as a retry-safe refusal is how the same wake gets delivered
    twice.
    """
    delivered: list = []
    entered = asyncio.Event()

    async def _blocking(event):
        delivered.append(event)
        entered.set()
        await asyncio.Event().wait()

    adapter = SimpleNamespace(handle_message=_blocking)
    loop = asyncio.get_running_loop()
    runner = _runner(_entry(), adapter, loop=loop)

    handle = runner._schedule_plugin_message_injection(
        session_key=SESSION_KEY,
        content="wake up",
        plugin_id="wake-plugin",
        await_dispatch=True,
    )
    await entered.wait()
    handle.cancel()
    result = await handle

    assert delivered, "the adapter side effect ran before cancellation"
    assert result.accepted is False
    assert result.reason == "cancelled"
    assert result.settled is False
    assert result.indeterminate is True
    assert result.refused is False
    assert result.safe_to_retry is False


@pytest.mark.asyncio
async def test_cancel_before_adapter_entry_is_a_settled_retry_safe_refusal():
    """Cancellation that provably wins the race is still a terminal refusal."""
    looked_up = asyncio.Event()
    hold = asyncio.Event()
    adapter = SimpleNamespace(handle_message=AsyncMock())
    loop = asyncio.get_running_loop()
    runner = _runner(_entry(), adapter, loop=loop)

    async def _blocked_lookup(_session_key):
        looked_up.set()
        await hold.wait()
        return _entry()

    runner._async_session_store.lookup_by_session_key = _blocked_lookup

    handle = runner._schedule_plugin_message_injection(
        session_key=SESSION_KEY,
        content="wake up",
        plugin_id="wake-plugin",
        await_dispatch=True,
    )
    await looked_up.wait()
    assert handle.cancel() is True
    result = await handle

    adapter.handle_message.assert_not_awaited()
    assert result.accepted is False
    assert result.reason == "cancelled"
    assert result.settled is True
    assert result.refused is True
    assert result.safe_to_retry is True


@pytest.mark.asyncio
async def test_adapter_raising_after_a_side_effect_preserves_unknown_delivery():
    """An adapter can deliver and then blow up; that is not a proven refusal."""
    delivered: list = []

    async def _explode(event):
        delivered.append(event)
        raise RuntimeError("adapter exploded after its side effect")

    adapter = SimpleNamespace(handle_message=_explode)
    loop = asyncio.get_running_loop()
    runner = _runner(_entry(), adapter, loop=loop)

    result = await runner._schedule_plugin_message_injection(
        session_key=SESSION_KEY,
        content="wake up",
        plugin_id="wake-plugin",
        await_dispatch=True,
    )

    assert delivered, "the adapter side effect ran before the exception"
    assert result.accepted is False
    assert result.reason == "internal_error"
    assert result.settled is False
    assert result.indeterminate is True
    assert result.refused is False
    assert result.safe_to_retry is False


def test_await_dispatch_result_is_blocking_readable_from_another_thread():
    """The webhook edge answering an HTTP request runs off the gateway loop."""
    adapter = SimpleNamespace(handle_message=AsyncMock())
    entry = _entry()
    ready = threading.Event()
    holder: dict = {}

    async def _serve():
        loop = asyncio.get_running_loop()
        runner = _runner(entry, adapter, loop=loop)
        runner._install_plugin_message_injector = None  # unused here
        holder["runner"] = runner
        ready.set()
        await asyncio.sleep(1.5)

    thread = threading.Thread(
        target=lambda: asyncio.run(_serve()),
        daemon=True,
    )
    thread.start()
    ready.wait(5)
    runner = holder["runner"]

    handle = runner._schedule_plugin_message_injection(
        session_key=SESSION_KEY,
        content="wake up",
        plugin_id="wake-plugin",
        await_dispatch=True,
        correlation_id="evt-thread",
    )
    result = handle.result(timeout=5)
    thread.join(timeout=5)

    assert result.accepted is True
    assert result.reason == "adopted"
    assert result.correlation_id == "evt-thread"


def test_blocking_result_reports_timeout_rather_than_raising():
    entry = _entry()
    ready = threading.Event()
    holder: dict = {}

    async def _slow(_event):
        await asyncio.sleep(10)

    async def _serve():
        loop = asyncio.get_running_loop()
        holder["runner"] = _runner(
            entry, SimpleNamespace(handle_message=_slow), loop=loop
        )
        ready.set()
        await asyncio.sleep(2)

    thread = threading.Thread(target=lambda: asyncio.run(_serve()), daemon=True)
    thread.start()
    ready.wait(5)

    handle = holder["runner"]._schedule_plugin_message_injection(
        session_key=SESSION_KEY,
        content="wake up",
        plugin_id="wake-plugin",
        await_dispatch=True,
    )
    result = handle.result(timeout=0.2)
    thread.join(timeout=5)

    assert result.accepted is False
    assert result.reason == "timeout"


# ---------------------------------------------------------------------------
# PluginContext surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_await_dispatch_surfaces_adoption(tmp_path, monkeypatch):
    _grant_injection(tmp_path, monkeypatch)
    adapter = SimpleNamespace(handle_message=AsyncMock())
    entry = _entry()
    loop = asyncio.get_running_loop()
    runner = _runner(entry, adapter, loop=loop)
    manager = PluginManager()

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        runner._install_plugin_message_injector()
        context = _context(manager)
        result = await context.inject_message(
            "read the ledger",
            session_key=SESSION_KEY,
            await_dispatch=True,
            correlation_id="evt-ctx",
        )

    assert result.accepted is True
    assert result.reason == "adopted"
    assert result.session_id == entry.session_id
    assert result.correlation_id == "evt-ctx"


@pytest.mark.asyncio
async def test_context_default_return_value_is_unchanged(tmp_path, monkeypatch):
    _grant_injection(tmp_path, monkeypatch)
    adapter = SimpleNamespace(handle_message=AsyncMock())
    loop = asyncio.get_running_loop()
    runner = _runner(_entry(), adapter, loop=loop)
    manager = PluginManager()

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        runner._install_plugin_message_injector()
        scheduled = _context(manager).inject_message(
            "read the ledger",
            session_key=SESSION_KEY,
        )

    assert scheduled is True
    await asyncio.gather(*list(runner._background_tasks), return_exceptions=True)


@pytest.mark.asyncio
async def test_context_await_dispatch_fails_closed_without_the_config_grant(
    tmp_path,
    monkeypatch,
):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(yaml.safe_dump({}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    manager = PluginManager()

    result = await _context(manager).inject_message(
        "read the ledger",
        session_key=SESSION_KEY,
        await_dispatch=True,
    )

    assert result.accepted is False
    assert result.reason == "injection_denied"


@pytest.mark.asyncio
async def test_context_await_dispatch_fails_closed_without_a_live_gateway(
    tmp_path,
    monkeypatch,
):
    _grant_injection(tmp_path, monkeypatch)
    result = await _context(PluginManager()).inject_message(
        "read the ledger",
        session_key=SESSION_KEY,
        await_dispatch=True,
    )

    assert result.accepted is False
    assert result.reason == "no_gateway"


@pytest.mark.asyncio
async def test_context_await_dispatch_requires_a_session_key(tmp_path, monkeypatch):
    _grant_injection(tmp_path, monkeypatch)
    result = await _context(PluginManager()).inject_message(
        "read the ledger",
        await_dispatch=True,
    )

    assert result.accepted is False
    assert result.reason == "invalid_request"


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["", "x" * 129, "line\nbreak", "tab\tstop"])
async def test_context_rejects_an_unbounded_correlation_id(tmp_path, monkeypatch, bad):
    _grant_injection(tmp_path, monkeypatch)
    result = await _context(PluginManager()).inject_message(
        "read the ledger",
        session_key=SESSION_KEY,
        await_dispatch=True,
        correlation_id=bad,
    )

    assert result.accepted is False
    assert result.reason == "invalid_request"


@pytest.mark.asyncio
async def test_context_tolerates_an_injector_without_the_new_kwargs(
    tmp_path,
    monkeypatch,
):
    """A host that registered the pre-existing injector signature must not crash."""
    _grant_injection(tmp_path, monkeypatch)
    manager = PluginManager()

    def _legacy_injector(*, session_key, content, plugin_id):
        return True

    manager.set_gateway_message_injector(object(), _legacy_injector)
    context = _context(manager)

    assert context.inject_message("hi", session_key=SESSION_KEY) is True

    result = await context.inject_message(
        "hi",
        session_key=SESSION_KEY,
        await_dispatch=True,
    )
    assert result.accepted is False
    assert result.reason == "unsupported"


def test_result_is_falsy_when_not_adopted():
    assert not GatewayInjectionResult(False, "unknown_session")
    assert GatewayInjectionResult(True, "adopted")


def test_resolved_handle_is_readable_without_a_loop():
    from hermes_cli.plugins import GatewayInjectionHandle

    handle = GatewayInjectionHandle.resolved(
        GatewayInjectionResult(False, "no_gateway", session_key=SESSION_KEY)
    )
    assert handle.result(timeout=0).reason == "no_gateway"
    assert isinstance(handle._future, concurrent.futures.Future)


# ---------------------------------------------------------------------------
# Settlement: timeout is indeterminate, not a refusal
# ---------------------------------------------------------------------------


def test_blocking_result_timeout_is_indeterminate_and_reconciles_to_adoption():
    """A read that gives up must not be readable as a denial.

    The caller stops waiting; the host does not stop working. The same handle is
    the reconciliation path to the eventual outcome -- re-injecting instead would
    deliver the wake twice.
    """
    entry = _entry()
    ready = threading.Event()
    release = threading.Event()
    reached_adapter = threading.Event()
    holder: dict = {}

    async def _blocked(_event):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, release.wait, 5)
        reached_adapter.set()

    async def _serve():
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()
        holder["runner"] = _runner(
            entry, SimpleNamespace(handle_message=_blocked), loop=loop
        )
        holder["stop"] = lambda: loop.call_soon_threadsafe(stop.set)
        ready.set()
        await stop.wait()

    thread = threading.Thread(target=lambda: asyncio.run(_serve()), daemon=True)
    thread.start()
    assert ready.wait(5)

    handle = holder["runner"]._schedule_plugin_message_injection(
        session_key=SESSION_KEY,
        content="wake up",
        plugin_id="wake-plugin",
        await_dispatch=True,
        correlation_id="evt-indeterminate",
    )
    try:
        pending = handle.result(timeout=0.2)

        # Falsy, but explicitly NOT a decided outcome.
        assert bool(pending) is False
        assert pending.reason == "timeout"
        assert pending.settled is False
        assert pending.indeterminate is True
        assert pending.refused is False
        # The whole point: a caller must not answer "failed, retry" here.
        assert pending.safe_to_retry is False
        assert handle.settled is False
        assert not reached_adapter.is_set()

        # The dispatch was never cancelled by the timeout; it still lands.
        release.set()
        final = handle.result(timeout=5)
    finally:
        release.set()
        holder["stop"]()
        thread.join(timeout=5)

    assert reached_adapter.is_set()
    assert final.accepted is True
    assert final.reason == "adopted"
    assert final.settled is True
    assert final.correlation_id == "evt-indeterminate"


def test_terminal_refusals_stay_terminal():
    """Every non-timeout outcome remains a settled fact callers can act on."""
    refusal = GatewayInjectionResult(False, "unknown_session")
    assert refusal.settled is True
    assert refusal.refused is True
    assert refusal.indeterminate is False
    assert refusal.safe_to_retry is True

    adopted = GatewayInjectionResult(True, "adopted")
    assert adopted.refused is False
    assert adopted.safe_to_retry is False

    # A dispatch that blew up may already have reached the adapter.
    crashed = GatewayInjectionResult(False, "internal_error")
    assert crashed.refused is True
    assert crashed.safe_to_retry is False


@pytest.mark.asyncio
async def test_correlation_id_is_metadata_not_idempotent_admission():
    """The tag identifies an event; it never deduplicates one.

    Nothing on this path stores correlation ids, so two injections carrying the
    same id both reach the session. Retry safety comes from
    ``GatewayInjectionResult.safe_to_retry``, not from the tag.
    """
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(_entry(), adapter)

    for _ in range(2):
        result = await runner._dispatch_plugin_message_injection(
            session_key=SESSION_KEY,
            content="wake up",
            plugin_id="wake-plugin",
            correlation_id="evt-duplicate",
        )
        assert result.reason == "adopted"

    assert adapter.handle_message.await_count == 2
    ids = {
        call.args[0].metadata["hermes_plugin_injection_id"]
        for call in adapter.handle_message.await_args_list
    }
    assert ids == {"evt-duplicate"}


def test_every_declared_reason_is_classified():
    """No reason may be silently unclassified as settled or indeterminate."""
    from hermes_cli.plugin_injection import (
        DELIVERY_SENSITIVE_REASONS,
        GATEWAY_INJECTION_REASONS,
        INDETERMINATE_REASONS,
    )

    assert INDETERMINATE_REASONS <= set(GATEWAY_INJECTION_REASONS)
    # Only an unfinished observation is unconditionally indeterminate.
    assert INDETERMINATE_REASONS == {"timeout"}
    # These two settle only when the host can prove the adapter never got the
    # event; after that boundary they become delivery-unknown.
    assert DELIVERY_SENSITIVE_REASONS <= set(GATEWAY_INJECTION_REASONS)
    assert DELIVERY_SENSITIVE_REASONS == {"cancelled", "internal_error"}
    assert not (DELIVERY_SENSITIVE_REASONS & INDETERMINATE_REASONS)
    for reason in GATEWAY_INJECTION_REASONS:
        result = GatewayInjectionResult(
            reason in ("adopted", "cli_queued"),
            reason,
            settled=reason not in INDETERMINATE_REASONS,
        )
        assert result.settled is not result.indeterminate

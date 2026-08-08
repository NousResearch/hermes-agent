"""Regression test: native Discord lane rename must not receive relay-only kwargs.

The gateway's _rename_discord_auto_thread_for_session_title used to pass
prefer_connector_created / parent_chat_id on EVERY lane. The native Discord
plugin adapter's rename_thread only accepts only_if_current_name, so every
native rename raised TypeError, was swallowed by the generic except, and the
thread was silently never renamed (no "rename result" log line either).

Only the relay connector adapter accepts the connector kwargs; the native
lane must call rename_thread with only the kwargs that adapter supports.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest

from gateway.config import Platform


def _mk_native_stub():
    """Stub runner whose _adapter_for_source returns the NATIVE adapter."""
    from gateway.run import GatewayRunner

    class _Stub:
        _is_relay_discord_channel_lane = GatewayRunner._is_relay_discord_channel_lane
        _relay_auto_thread_info = GatewayRunner._relay_auto_thread_info
        _is_discord_auto_thread_lane = GatewayRunner._is_discord_auto_thread_lane
        _sanitize_discord_thread_title = GatewayRunner._sanitize_discord_thread_title
        _rename_discord_auto_thread_for_session_title = (
            GatewayRunner._rename_discord_auto_thread_for_session_title
        )

        def __init__(self, adapter):
            self.adapters = {Platform.DISCORD: adapter}

        def _adapter_for_source(self, source):
            return self.adapters.get(source.platform)

    return _Stub


def _native_thread_source() -> SimpleNamespace:
    """A source carrying the native auto-thread markers (as the bundled
    Discord plugin adapter stamps on the SessionSource at ingest)."""
    return SimpleNamespace(
        platform=Platform.DISCORD,
        chat_id="th-native",
        chat_type="thread",
        thread_id="th-native",
        auto_thread_created=True,
        auto_thread_initial_name="Initial Words",
        delivered_via_upstream_relay=False,
    )


class _NativeAdapter:
    """Signature mirror of the bundled Discord plugin adapter's rename_thread:

    async def rename_thread(self, thread_id, name, *, only_if_current_name=None)
    """

    def __init__(self):
        self.calls: list = []

    async def rename_thread(
        self,
        thread_id: str,
        name: str,
        *,
        only_if_current_name: Optional[str] = None,
    ) -> bool:
        self.calls.append(
            {"thread_id": thread_id, "name": name, "only_if_current_name": only_if_current_name}
        )
        return True


@pytest.mark.asyncio
async def test_native_lane_rename_no_connector_kwargs():
    """The gateway's rename lane must not send relay-only kwargs to the native
    adapter — that TypeError silently killed every native thread rename."""
    adapter = _NativeAdapter()
    runner = _mk_native_stub()(adapter)
    src = _native_thread_source()

    await runner._rename_discord_auto_thread_for_session_title(
        src, "sess-native", "Real Session Title"
    )

    # The rename must have been issued with the native-only signature.
    assert len(adapter.calls) == 1
    call = adapter.calls[0]
    assert call["thread_id"] == "th-native"
    assert call["name"] == "Real Session Title"
    # The connector-owned guard must NOT be smuggled onto the native lane;
    # the initial-name string guard is the native adapter's contract.
    assert call["only_if_current_name"] == "Initial Words"


@pytest.mark.asyncio
async def test_native_lane_rename_survives_plugin_signature_strictness():
    """Pre-fix, passing prefer_connector_created to a signature-strict adapter
    raised TypeError -> rename never happened. This pins the contract so the
    gateway call stays compatible with the plugin adapter's signature."""
    adapter = _NativeAdapter()
    runner = _mk_native_stub()(adapter)
    src = _native_thread_source()

    # Would raise TypeError pre-fix (unexpected keyword argument
    # 'prefer_connector_created'); must complete cleanly post-fix.
    await runner._rename_discord_auto_thread_for_session_title(
        src, "sess2", "Native rename still works"
    )
    assert [c["name"] for c in adapter.calls] == ["Native rename still works"]


@pytest.mark.asyncio
async def test_relay_lane_still_uses_connector_guard():
    """Sanity: the relay lane keeps its connector-owned guard behavior —
    the fix must not have degraded it."""
    from tests.gateway.relay.test_relay_threads import _adapter, _mk_runner_stub, _relay_channel_source

    adapter, stub_conn = _adapter()
    renames: list = []

    async def rename_thread(
        thread_id,
        name,
        *,
        only_if_current_name=None,
        prefer_connector_created=False,
        parent_chat_id=None,
    ):
        renames.append((thread_id, name, prefer_connector_created, parent_chat_id))
        return True

    adapter.rename_thread = rename_thread  # type: ignore[method-assign]
    runner = _mk_runner_stub()(adapter)
    src = _relay_channel_source()
    adapter._auto_thread_by_chat["chan-parent"] = ("th-9", "Initial words")

    await runner._rename_discord_auto_thread_for_session_title(
        src, "sess1", "Relay title"
    )
    assert renames == [("th-9", "Relay title", True, "chan-parent")]

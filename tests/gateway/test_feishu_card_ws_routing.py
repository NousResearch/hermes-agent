"""Regression test for the Feishu WS CARD-frame routing fix.

lark-oapi (<= 1.7.x) silently drops CARD-type websocket frames in
``lark_oapi.ws.client.Client._handle_data_frame``
(``elif message_type == MessageType.CARD: return``), so interactive card
button clicks never reach the adapter.  ``_run_official_feishu_ws_client``
installs a class-level patch that routes CARD frames through the same
validation-free dispatch as EVENT frames and replies back to Feishu.

Requires the (optional) lark-oapi SDK; skipped when not installed.
"""

import asyncio
import types

import pytest

pytest.importorskip("lark_oapi")

import lark_oapi.ws.client as ws_client_mod
from lark_oapi.ws.client import MessageType

from plugins.platforms.feishu.adapter import _run_official_feishu_ws_client


class _NopClient:
    """WS client whose start() does nothing — we only want the patch installed."""

    def start(self):
        return None


def _adapter_stub():
    return types.SimpleNamespace(
        _ws_thread_loop=None,
        _ws_reconnect_nonce=None,
        _ws_reconnect_interval=None,
        _ws_ping_interval=None,
        _ws_ping_timeout=None,
    )


def _install_patch():
    """Install the CARD fix on the real SDK class; return the original handler."""
    original = ws_client_mod.Client._handle_data_frame
    _run_official_feishu_ws_client(_NopClient(), _adapter_stub())
    return original


def test_handle_data_frame_is_patched():
    """The fix must replace the SDK's frame handler at class level."""
    original = _install_patch()
    try:
        patched = ws_client_mod.Client._handle_data_frame
        assert patched is not original
        assert getattr(patched, "__name__", "") == "_patched_handle_data_frame"
    finally:
        ws_client_mod.Client._handle_data_frame = original


def test_patch_skipped_when_client_class_missing():
    """Multiplex-isolation tests inject a fake ws module without a Client
    class; the installer must not blow up in that case."""
    import sys

    original_module = sys.modules.get("lark_oapi.ws.client")
    fake = types.ModuleType("lark_oapi.ws.client")
    fake.loop = None
    fake.websockets = types.SimpleNamespace(connect=lambda: None)
    try:
        sys.modules["lark_oapi.ws.client"] = fake
        # The installer's guard must skip cleanly (no Client attribute).
        _run_official_feishu_ws_client(_NopClient(), _adapter_stub())
    finally:
        if original_module is not None:
            sys.modules["lark_oapi.ws.client"] = original_module
        else:
            sys.modules.pop("lark_oapi.ws.client", None)


def test_card_frame_reaches_event_handler_and_replies():
    """A CARD frame must be dispatched to the event handler and answered."""

    original = _install_patch()
    patched_handler = ws_client_mod.Client._handle_data_frame
    try:

        async def _exercise():
            from lark_oapi.ws.const import (
                HEADER_MESSAGE_ID,
                HEADER_SEQ,
                HEADER_SUM,
                HEADER_TRACE_ID,
                HEADER_TYPE,
            )
            from lark_oapi.ws.pb.pbbp2_pb2 import Frame

            delivered = []

            def _fake_do(payload):
                # The SDK's dispatch method is synchronous.
                delivered.append(payload)
                return None

            written = []

            async def _fake_write(data):
                written.append(data)

            # proto2 requires SeqID/LogID/service/method before serialization.
            frame = Frame(SeqID=1, LogID=1, service=1, method=1)
            entries = {
                HEADER_MESSAGE_ID: "msg-1",
                HEADER_TRACE_ID: "trace-1",
                HEADER_SUM: "1",
                HEADER_SEQ: "0",
                HEADER_TYPE: MessageType.CARD.value,
            }
            for key, value in entries.items():
                entry = frame.headers.add()
                entry.key = key
                entry.value = value
            frame.payload = b"the-card-payload"

            client = object.__new__(ws_client_mod.Client)
            client._event_handler = types.SimpleNamespace(_do_without_validation=_fake_do)
            client._combine = lambda *a, **kwargs: None
            client._fmt_log = lambda *a, **kwargs: ""
            client._write_message = _fake_write

            await patched_handler(client, frame)

            # The CARD payload must have been routed to the event handler
            # (the SDK dropped it before reaching this point).
            assert delivered == [b"the-card-payload"]
            # A reply must have been written back to Feishu...
            assert len(written) == 1
            assert written[0]
            # ...and the BIZ_RT header added, exactly like the EVENT path.
            header_keys = [entry.key for entry in frame.headers]
            from lark_oapi.ws.const import HEADER_BIZ_RT

            assert HEADER_BIZ_RT in header_keys

        asyncio.run(_exercise())
    finally:
        ws_client_mod.Client._handle_data_frame = original
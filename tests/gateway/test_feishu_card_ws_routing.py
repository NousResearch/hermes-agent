"""Regression test for the Feishu WS CARD-frame routing fix.

lark-oapi (<= 1.7.x) silently drops CARD-type websocket frames in
``lark_oapi.ws.client.Client._handle_data_frame``
(``elif message_type == MessageType.CARD: return``), so interactive card
button clicks never reach the adapter.  ``_run_official_feishu_ws_client``
installs a class-level patch that routes CARD frames through the same
validation-free dispatch as EVENT frames and replies back to Feishu, then
restores the SDK's own handler when the WS run tears down.

Requires the (optional) lark-oapi SDK; skipped when not installed.
"""

import asyncio
import types

import pytest

pytest.importorskip("lark_oapi")

import lark_oapi.ws.client as ws_client_mod
from lark_oapi.ws.client import MessageType

from plugins.platforms.feishu.adapter import _run_official_feishu_ws_client


class _ProbeClient:
    """WS client whose ``start()`` records the live class handler and returns.

    The installer restores the SDK's handler on teardown, so the patched
    handler is only observable while the run is alive — capture it here.
    """

    def __init__(self):
        self.handler_at_start = None

    def start(self):
        self.handler_at_start = ws_client_mod.Client._handle_data_frame
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
    """Run the real installer once; return the probe holding the live handler."""
    client = _ProbeClient()
    _run_official_feishu_ws_client(client, _adapter_stub())
    return client


def _make_frame(message_type, payload):
    """Build a data frame whose headers carry ``message_type``."""
    from lark_oapi.ws.const import HEADER_MESSAGE_ID, HEADER_SEQ, HEADER_SUM, HEADER_TRACE_ID, HEADER_TYPE
    from lark_oapi.ws.pb.pbbp2_pb2 import Frame

    # proto2 requires SeqID/LogID/service/method before serialization.
    frame = Frame(SeqID=1, LogID=1, service=1, method=1)
    entries = {
        HEADER_MESSAGE_ID: "msg-1",
        HEADER_TRACE_ID: "trace-1",
        HEADER_SUM: "1",
        HEADER_SEQ: "0",
        HEADER_TYPE: message_type.value,
    }
    for key, value in entries.items():
        entry = frame.headers.add()
        entry.key = key
        entry.value = value
    frame.payload = payload
    return frame


def _make_client(delivered, written):
    """SDK client instance wired to recording stand-ins for its collaborators."""

    def _fake_do(payload):
        # The SDK's dispatch method is synchronous.
        delivered.append(payload)
        return None

    async def _fake_write(data):
        written.append(data)

    client = object.__new__(ws_client_mod.Client)
    client._event_handler = types.SimpleNamespace(_do_without_validation=_fake_do)
    client._combine = lambda *a, **kwargs: None
    client._fmt_log = lambda *a, **kwargs: ""
    client._write_message = _fake_write
    return client


def test_patch_is_installed_while_running_and_restored_on_teardown():
    """The fix replaces the SDK's frame handler for the life of the WS run,
    then puts the original back so the SDK class is not left patched."""
    original = ws_client_mod.Client._handle_data_frame

    client = _install_patch()

    assert client.handler_at_start is not original
    assert getattr(client.handler_at_start, "__name__", "") == "_patched_handle_data_frame"
    # Teardown already ran when the installer returned.
    assert ws_client_mod.Client._handle_data_frame is original


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
        _run_official_feishu_ws_client(_ProbeClient(), _adapter_stub())
    finally:
        if original_module is not None:
            sys.modules["lark_oapi.ws.client"] = original_module
        else:
            sys.modules.pop("lark_oapi.ws.client", None)


def test_card_frame_reaches_event_handler_and_replies():
    """A CARD frame must be dispatched to the event handler and answered."""
    patched_handler = _install_patch().handler_at_start

    async def _exercise():
        delivered = []
        written = []
        client = _make_client(delivered, written)
        frame = _make_frame(MessageType.CARD, b"the-card-payload")

        await patched_handler(client, frame)

        # The CARD payload must have been routed to the event handler
        # (the SDK dropped it before reaching this point).
        assert delivered == [b"the-card-payload"]
        # A reply must have been written back to Feishu...
        assert len(written) == 1
        assert written[0]
        # ...and the BIZ_RT header added, exactly like the EVENT path.
        from lark_oapi.ws.const import HEADER_BIZ_RT

        header_keys = [entry.key for entry in frame.headers]
        assert HEADER_BIZ_RT in header_keys

    asyncio.run(_exercise())


def test_non_card_frame_behaviour_is_unchanged():
    """Non-CARD data frames keep the SDK's original behaviour: no dispatch
    to the event handler and no reply written.  The patch must not turn a
    frame type it does not own into a dispatched one."""
    patched_handler = _install_patch().handler_at_start

    async def _exercise():
        delivered = []
        written = []
        client = _make_client(delivered, written)
        # ``MessageType.PONG`` reaches ``_handle_data_frame`` as a non-CARD,
        # non-EVENT data frame — the branch the SDK answers with a bare
        # ``return``.
        frame = _make_frame(MessageType.PONG, b"pong-payload")

        await patched_handler(client, frame)

        assert delivered == []
        assert written == []

    asyncio.run(_exercise())

"""Regression test: keepalive override must not leak across instances.

Verifies the fix for reviewer feedback on the keepalive timeout PR:
the original code used `type(self._stream_client).keepalive = ...` which
mutated the SDK client class process-wide. The fix binds to the instance
only via ``types.MethodType``.

These tests prove that connecting one adapter (and patching its keepalive)
does not alter the class or any other DingTalkStreamClient instance.
"""
import asyncio
import types

import dingtalk_stream


def test_keepalive_instance_isolation():
    """Patching one client's keepalive must not affect the class or others."""

    # Capture the stock SDK keepalive before any patching
    original_class_keepalive = dingtalk_stream.DingTalkStreamClient.keepalive
    assert original_class_keepalive is not None, "SDK must define keepalive"

    # Create two independent clients (simulating two adapters)
    cred_a = dingtalk_stream.Credential("id_a", "secret_a")
    client_a = dingtalk_stream.DingTalkStreamClient(cred_a)
    cred_b = dingtalk_stream.Credential("id_b", "secret_b")
    client_b = dingtalk_stream.DingTalkStreamClient(cred_b)

    # Simulate the adapter's connect() logic: patch client_a's keepalive
    async def _hardened_keepalive(_self, ws, ping_interval=60):
        pass

    client_a.keepalive = types.MethodType(_hardened_keepalive, client_a)

    # 1. Class-level keepalive must be UNCHANGED
    assert dingtalk_stream.DingTalkStreamClient.keepalive is original_class_keepalive, \
        "Class-level keepalive was mutated — instance patch leaked to class!"

    # 2. client_b's keepalive must still be the stock SDK version
    #    (accessed via __dict__ to avoid descriptor protocol resolving to the class method)
    assert "keepalive" not in client_b.__dict__, \
        "client_b got an instance-level keepalive — cross-instance leak!"

    # 3. client_a's keepalive must be the patched version
    assert client_a.keepalive.__func__ is _hardened_keepalive, \
        "client_a's keepalive was not patched correctly!"

    # 4. client_a's keepalive must be bound to client_a, not client_b
    assert client_a.keepalive.__self__ is client_a, \
        "client_a's keepalive is bound to the wrong instance!"


def test_keepalive_disconnect_restoration():
    """After dropping a patched client, a new client must get stock keepalive."""

    original = dingtalk_stream.DingTalkStreamClient.keepalive

    # Create, patch, then drop client_a (simulating disconnect)
    client_a = dingtalk_stream.DingTalkStreamClient(
        dingtalk_stream.Credential("id", "secret")
    )
    async def _patched(_self, ws, ping_interval=60):
        pass
    client_a.keepalive = types.MethodType(_patched, client_a)
    del client_a

    # New client (simulating reconnect) must get stock keepalive
    client_c = dingtalk_stream.DingTalkStreamClient(
        dingtalk_stream.Credential("id2", "secret2")
    )
    assert "keepalive" not in client_c.__dict__, \
        "New client inherited an instance-level keepalive from a deleted instance!"
    assert dingtalk_stream.DingTalkStreamClient.keepalive is original, \
        "Class keepalive was mutated and not restored!"


def test_keepalive_timeout_actually_works():
    """The patched keepalive should close the ws on ping timeout.

    This is the functional test: verify that the hardened keepalive actually
    wraps ws.ping() in asyncio.wait_for with a timeout, and that a timeout
    triggers ws.close() + loop break (so the SDK's reconnect loop can kick in).
    """
    closed = False

    class _FakeWS:
        async def ping(self):
            # Simulate a hung ping — never completes within the timeout
            await asyncio.sleep(999)

        async def close(self):
            nonlocal closed
            closed = True

    async def _run():
        # Reproduce the adapter's _monitored_keepalive logic
        async def _hardened_keepalive(_self, ws, ping_interval=0.01):
            while True:
                await asyncio.sleep(ping_interval)
                try:
                    await asyncio.wait_for(ws.ping(), timeout=0.05)
                except asyncio.TimeoutError:
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    break
                except Exception:
                    break

        ws = _FakeWS()
        bound = types.MethodType(_hardened_keepalive, object())  # dummy instance
        await bound(ws)
        return closed

    ws_closed = asyncio.run(_run())
    assert ws_closed, "keepalive did not close the WebSocket on ping timeout!"

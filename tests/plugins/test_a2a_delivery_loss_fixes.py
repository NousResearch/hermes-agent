"""A2A delivery-loss regression coverage: eight focused scenarios.

Covers:
1. Regression: two in-process adapters, A→B with a 2s client timeout while B's
   handler sleeps 10s — B's reply must reach A's session via push, exactly once.
2. Round-trip: a push whose peer answers inside the HTTP response re-enters the
   pushing session; the peer must NOT also push it out-of-band.
3. Patience: a reply resolving after patience+margin takes the push path and
   skips the socket write.
4. Probe race: EOF after a successful write is NOT a rescue — the post-write
   re-probe is gone; a clean close-after-read and a dead-before-read are the
   same MSG_PEEK signal, so nothing pushes after a successful write.
5. Sender stamping: _send_task with no live adapter still stamps
   {agentId, name, url, timeout}; the recipient refines to the real peer.
6. Self-loop guard: unresolvable loopback + unmarked reply → audit
   push_dropped, success=False.
7. Inbound dedupe: the same wire message (contextId, messageId)
   inside _INBOUND_DEDUPE_WINDOW is dropped — _prepare_task rejects it
   without dispatching — accepted again once outside the window, and the
   seen map stays bounded at _INBOUND_DEDUPE_MAX.
8. Rescue flag: the _push_reply_after_client_gone rescue pushes with
   want_reply=True so the peer's answer round-trips; failed
   tasks push nothing.
"""
from __future__ import annotations

import asyncio
import socket
import threading
import time
import urllib.error
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from gateway.platforms.base import ProcessingOutcome
from plugins.platforms.a2a import protocol, security, tools


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _bare_adapter():
    from gateway.config import PlatformConfig
    from plugins.platforms.a2a.adapter import A2AAdapter
    return A2AAdapter(PlatformConfig(enabled=True))


class _ADAPTERS_CLEARED:
    """Temporarily clear the module-level live-adapter registry."""

    def __enter__(self):
        import plugins.platforms.a2a.adapter as mod
        self._mod = mod
        with mod._ADAPTERS_GUARD:
            self._saved = dict(mod._ADAPTERS)
            mod._ADAPTERS.clear()
        return self

    def __exit__(self, *exc):
        with self._mod._ADAPTERS_GUARD:
            self._mod._ADAPTERS.update(self._saved)
        return False


def _patch_persistence(monkeypatch):
    """Keep best-effort write-through maps out of the real HERMES_HOME."""
    from plugins.platforms.a2a import adapter as adapter_mod
    monkeypatch.setattr(adapter_mod, "_persist_context_peers", lambda peers: None)
    monkeypatch.setattr(adapter_mod, "_persist_context_sessions", lambda sessions: None)


def _live_adapter(monkeypatch, handler):
    """Build a connected in-process adapter on a fresh port with a fake
    session handler; the gateway loop runs on a background thread."""
    from plugins.platforms.a2a.adapter import A2AAdapter

    monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
    adapter = _bare_adapter()
    adapter.host = "127.0.0.1"
    adapter.port = _free_port()
    adapter.handle_message = handler  # type: ignore[method-assign]
    adapter._message_handler = object()
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    asyncio.run_coroutine_threadsafe(adapter.connect(), loop).result(timeout=10)
    return adapter, loop, thread


def _stop_adapter(adapter, loop, thread):
    # Preserve primary test failure: disconnect/thread errors must not swallow
    # the test's own assertion, but they are real lifecycle failures and must
    # be surfaced (not silently passed). Thread termination is asserted.
    disconnect_error = None
    try:
        try:
            fut = asyncio.run_coroutine_threadsafe(adapter.disconnect(), loop)
            fut.result(timeout=10)
        except Exception as e:
            disconnect_error = e
            import warnings
            warnings.warn(f"A2A adapter disconnect failed: {e}", RuntimeWarning)
    finally:
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass
        thread.join(timeout=5)
        if thread.is_alive():
            raise AssertionError("A2A adapter thread failed to terminate within 5s — potential deadlock or leaked loop")
        # Close the loop and clear policy to avoid ResourceWarning under -W error.
        # Cleanup is defensive but must not suppress warnings; cancellation/close
        # must run even if disconnect_error occurred, without hiding the primary
        # test failure (which is outside this helper).
        try:
            try:
                pending = asyncio.all_tasks(loop)  # type: ignore[arg-type]
            except RuntimeError:
                pending = set()
            for t in list(pending):
                t.cancel()
            if pending:
                try:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except Exception:
                    pass
        finally:
            if not loop.is_closed():
                try:
                    loop.close()
                except Exception:
                    pass
            try:
                if asyncio.get_event_loop_policy().get_event_loop() is loop:
                    asyncio.set_event_loop(None)
            except RuntimeError:
                try:
                    asyncio.set_event_loop(None)
                except Exception:
                    pass
            except Exception:
                try:
                    asyncio.set_event_loop(None)
                except Exception:
                    pass


# ═════════════════════════════════════════════════════════════════════════════
# 1. Regression — the lost push (cases 1/3 shape)
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
def test_lost_push_regression_two_adapters(monkeypatch, tmp_path):
    """A→B with a 2s client timeout; B's handler sleeps 10s. B's reply must
    arrive at A's session on the SAME contextId via push, exactly once (no
    socket + push double delivery). Pre-fix this reply vanished into the
    half-closed socket or was eaten by the push client."""
    from plugins.platforms.a2a import adapter as adapter_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _patch_persistence(monkeypatch)
    # Probe every 1s so the EOF (client closed at 2s) is seen well before
    # the reply resolves at 10s — deterministic probe-death path.
    monkeypatch.setattr(adapter_mod, "_SSE_KEEPALIVE", 1.0)

    received_a: list[str] = []
    sent_b: list[str] = []

    async def handle_a(event):
        received_a.append(event.text)
        # Simulate the gateway runner: processing ended without a reply
        # send, so the pending task resolves (empty reply) and A's HTTP
        # response is written — unblocking B's push POST.
        await adapter_a.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    async def handle_b(event):
        # The slow turn that outlives the caller's 2s client patience.
        await asyncio.sleep(10)
        await adapter_b.send(event.source.chat_id, "B_REPLY_4148", metadata={"notify": True})
        sent_b.append("replied")

    adapter_a, loop_a, thread_a = _live_adapter(monkeypatch, handle_a)
    adapter_b, loop_b, thread_b = _live_adapter(monkeypatch, handle_b)
    try:
        ctx = "ctx-regress-1"
        # A calls B with a 2s client timeout — the reply (10s) cannot ride
        # this connection; urllib gives up at 2s and CLOSES.
        with pytest.raises((urllib.error.URLError, TimeoutError)):
            tools._send_task(
                "b", {"url": f"http://127.0.0.1:{adapter_b.port}", "auth": {}, "timeout": 2},
                "please take your time", ctx,
            )

        # B's server must have dropped the stale waiter (probe EOF at ~3s).
        deadline = time.time() + 8
        while time.time() < deadline:
            with adapter_b._pending_lock:
                if not adapter_b._pending_order:
                    break
            time.sleep(0.2)
        with adapter_b._pending_lock:
            assert adapter_b._pending_order == {}, "stale waiter not dropped"

        # The reply (ready at 10s) must reach A's session via push, once.
        deadline = time.time() + 25
        while time.time() < deadline and not received_a:
            time.sleep(0.2)
        assert len(received_a) == 1, f"expected exactly one delivery, got {received_a}"
        assert "B_REPLY_4148" in received_a[0]

        # No double delivery: nothing else lands afterwards.
        time.sleep(1.5)
        assert len(received_a) == 1
        assert sent_b == ["replied"]
    finally:
        _stop_adapter(adapter_a, loop_a, thread_a)
        _stop_adapter(adapter_b, loop_b, thread_b)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Round-trip push
# ═════════════════════════════════════════════════════════════════════════════


def test_push_round_trip_surfaces_peer_reply(monkeypatch, tmp_path):
    """A pushes to B with want_reply=True; B's session answers inside the
    push's HTTP response (previously the reply was discarded here). The
    reply must re-enter A's session on the same contextId — and B must NOT
    also push it out-of-band (its send resolved a waiter)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        tools, "_load_config",
        lambda: {"a2a_agents": {"bob": {"url": "http://127.0.0.1:8805"}}},
    )
    monkeypatch.setattr(tools, "_fetch_card", lambda *a, **k: None)

    adapter = _bare_adapter()
    try:
        adapter.host, adapter.port = "127.0.0.1", 9901
        adapter._context_peers["ctx-rt"] = "bob"
        looped: list[tuple[str, str, str]] = []
        monkeypatch.setattr(
            adapter, "_push_loopback_in_process",
            lambda cid, peer, text, **kw: looped.append((cid, peer, text)),
        )
        pushed: list[dict] = []

        def fake_post(url, body, headers, timeout, **kw):
            pushed.append(body)
            return protocol.jsonrpc_result(
                body["id"],
                protocol.send_message_response(protocol.build_task(
                    "task-b", "ctx-rt", protocol.STATE_COMPLETED, "VERDICT_OK",
                )),
            )

        monkeypatch.setattr(tools, "_http_post_json", fake_post)

        adapter._push_out_of_band("ctx-rt", "round one findings", want_reply=True)
        assert len(pushed) == 1
        # The verdict re-entered A's session as an inbound message on the
        # same contextId (loopback in-process path).
        assert looped == [("ctx-rt", "bob", "VERDICT_OK")]

        # B's side: a waiter existed, so B's send resolves it and must NOT
        # push the reply out-of-band.
        adapter_b = _bare_adapter()
        try:
            fut = adapter_b._add_pending("task-b", "ctx-rt")
            pushes_b: list = []
            monkeypatch.setattr(
                adapter_b, "_push_out_of_band",
                lambda *a, **k: pushes_b.append(a),
            )

            async def run():
                res = await adapter_b.send("ctx-rt", "VERDICT_OK", metadata={"notify": True})
                assert res.success is True
                assert fut.result(timeout=0) == (protocol.STATE_COMPLETED, "VERDICT_OK")

            asyncio.run(run())
            assert pushes_b == []
        finally:
            adapter_b._unregister_adapter()
    finally:
        adapter._unregister_adapter()


# ═════════════════════════════════════════════════════════════════════════════
# 3. Patience unit — deterministic dead-client handling
# ═════════════════════════════════════════════════════════════════════════════


def test_patience_exceeded_pushes_and_skips_socket_write(monkeypatch, tmp_path):
    """Reply resolves after patience+margin → the push path is taken and the
    socket write is skipped entirely (_rpc_message_send returns None). The
    client that stays connected but would discard is invisible to any probe —
    patience is the deterministic backstop."""
    from plugins.platforms.a2a import adapter as adapter_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(adapter_mod, "_PATIENCE_MARGIN", 0.2)
    monkeypatch.setattr(adapter_mod, "_SSE_KEEPALIVE", 0.05)

    adapter = _bare_adapter()
    try:
        pushed: list[tuple[str, str, bool]] = []
        monkeypatch.setattr(
            adapter, "_push_out_of_band",
            lambda cid, text, want_reply=False: pushed.append((cid, text, want_reply)),
        )
        audited: list[tuple[str, str, str]] = []
        monkeypatch.setattr(
            security, "audit",
            lambda direction, peer, task_id, summary, context_id=None: audited.append(
                (direction, peer, summary)
            ),
        )

        fut: Future = Future()
        pending = {
            "task_id": "task-p", "context_id": "ctx-p", "peer": "alice",
            "future": fut, "created_iso": protocol.now_iso(), "started": time.time(),
        }
        monkeypatch.setattr(
            adapter, "_prepare_task",
            lambda params, peer, agent=None: (None, pending),
        )
        monkeypatch.setattr(
            tools, "_load_config",
            lambda: {"a2a_agents": {"alice": {"url": "http://localhost:8805"}}},
        )
        params = {
            "message": protocol.text_message(
                protocol.ROLE_USER, "slow turn", context_id="ctx-p",
                sender={
                    "agentId": "alice", "name": "alice",
                    "url": "http://127.0.0.1:8805", "timeout": 0.3,
                },
            ),
        }
        result_holder: dict = {}

        def run_rpc():
            result_holder["result"] = adapter._rpc_message_send("req-p", params, "alice")

        thread = threading.Thread(target=run_rpc, daemon=True)
        thread.start()
        # Wait past patience (0.3) + margin (0.2), then resolve the reply.
        time.sleep(0.7)
        fut.set_result((protocol.STATE_COMPLETED, "LATE_REPLY"))
        thread.join(timeout=5)
        assert not thread.is_alive()

        # Socket write skipped entirely; the reply was pushed with the
        # round-trip flag (a session reply wants the peer's answer back).
        assert result_holder.get("result") is None
        assert pushed == [("ctx-p", "LATE_REPLY", True)]
        assert any("patience exceeded" in s for _d, _p, s in audited), audited
    finally:
        adapter._unregister_adapter()


# ═════════════════════════════════════════════════════════════════════════════
# 4. Probe race — EOF after a successful write is not a rescue
# ═════════════════════════════════════════════════════════════════════════════


def test_post_write_eof_is_not_a_rescue(monkeypatch):
    """Client closes; the reply resolves before the next probe tick; the
    write goes into the closed socket without raising.

    Corrected mechanism (verified against live traffic): there is NO post-write
    MSG_PEEK probe — a client that reads the response and closes cleanly
    (urllib ``Connection: close``) surfaces as EOF too, so probing after the
    write would double-deliver every clean exchange and ping-pong between
    two gateways. A successful write therefore never triggers a rescue; the
    probe-race window is covered by the broad ``OSError`` write catch (RST-
    dead clients) and by the deterministic patience rule.

    Residual race window (accepted limitation): a client that FIN-closes
    between the last probe tick (5s ``_SSE_KEEPALIVE``) and the reply write
    can still silently lose the write — the kernel buffer accepts it, so no
    exception fires and no patience trigger arms. If at-least-once delivery
    is ever required, the deterministic closure is an application-level ACK
    (ack-loss → rescue push → inbound dedupe absorbs the duplicate), never
    another socket probe."""
    from plugins.platforms.a2a.adapter import A2ARequestHandler

    adapter = _bare_adapter()
    try:
        pushed: list[tuple[str, object]] = []
        monkeypatch.setattr(
            adapter, "_push_reply_after_client_gone",
            lambda req_id, result: pushed.append((req_id, result)),
        )
        written: list[dict] = []
        probes: list[bool] = []

        def fake_rpc(req_id, params, peer, agent=None, v1_response=False, client_alive=None):
            task = protocol.build_task("t-1", "ctx-r", protocol.STATE_COMPLETED, "race reply")
            return protocol.jsonrpc_result(req_id, protocol.send_message_response(task))

        monkeypatch.setattr(adapter, "_rpc_message_send", fake_rpc)
        handler = SimpleNamespace(adapter=adapter)

        def fake_json(code, payload):
            written.append(payload)

        handler._json = fake_json  # type: ignore[attr-defined]
        # The client is alive at the pre-write probe (returns True) but
        # would be EOF-dead after the write — indistinguishable from a
        # clean close-after-read, so no rescue may fire.  The pre-write
        # probe allows the write to proceed.
        handler._a2a_client_alive = lambda: True  # type: ignore[attr-defined]

        A2ARequestHandler._handle_send(
            handler, "req-r", {"message": {}}, "alice", agent=None, is_v1=True,
        )
        assert len(written) == 1  # the write happened normally
        assert pushed == []  # EOF after a successful write is NOT a rescue
        # No post-write probe runs (design decision: post-write probes
        # cause double-delivery on clean exchanges).
    finally:
        adapter._unregister_adapter()


def test_write_failure_pushes_and_skips_reprobe(monkeypatch):
    """A write that raises (OSError — broad catch) pushes once and
    does not double-push via the post-write re-probe."""
    from plugins.platforms.a2a.adapter import A2ARequestHandler

    adapter = _bare_adapter()
    try:
        pushed: list = []
        monkeypatch.setattr(
            adapter, "_push_reply_after_client_gone",
            lambda req_id, result: pushed.append((req_id, result)),
        )

        def fake_rpc(req_id, params, peer, agent=None, v1_response=False, client_alive=None):
            task = protocol.build_task("t-1", "ctx-r", protocol.STATE_COMPLETED, "late reply")
            return protocol.jsonrpc_result(req_id, protocol.send_message_response(task))

        monkeypatch.setattr(adapter, "_rpc_message_send", fake_rpc)
        handler = SimpleNamespace(adapter=adapter)
        probes: list[bool] = []

        def fake_json(code, payload):
            raise BrokenPipeError("client gone")

        handler._json = fake_json  # type: ignore[attr-defined]
        handler._a2a_client_alive = lambda: probes.append(True) or True  # type: ignore[attr-defined]

        A2ARequestHandler._handle_send(
            handler, "req-w", {"message": {}}, "alice", agent=None, is_v1=True,
        )
        assert len(pushed) == 1
        # The pre-write probe ran once (client alive → write proceeds),
        # then the write failed with BrokenPipeError → rescue fired.
        # No POST-write re-probe runs (design decision: post-write
        # probes cause double-delivery on clean exchanges).
        assert len(probes) == 1  # exactly one pre-write probe
    finally:
        adapter._unregister_adapter()


# ═════════════════════════════════════════════════════════════════════════════
# 5. Sender stamping
# ═════════════════════════════════════════════════════════════════════════════


def test_send_task_stamps_sender_with_timeout_no_live_adapter(monkeypatch, tmp_path):
    """_send_task with NO live adapter must still stamp the full sender block
    {agentId, name, url, timeout} (config/env fallback), and the
    recipient must refine the loopback identity to the real peer."""
    from plugins.platforms.a2a.adapter import A2AAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("A2A_AGENT_NAME", "helper-agent")
    monkeypatch.setenv("A2A_PORT", "8806")
    monkeypatch.setattr(tools, "_fetch_card", lambda *a, **k: None)
    posted: dict = {}

    def fake_post(url, body, headers, timeout, **kw):
        posted["body"] = body
        return protocol.jsonrpc_result(
            body["id"],
            protocol.build_task("task-1", "ctx-s", protocol.STATE_COMPLETED, "ok"),
        )

    monkeypatch.setattr(tools, "_http_post_json", fake_post)

    with _ADAPTERS_CLEARED():
        reply, _ctx, _state = tools._send_task(
            "bob", {"url": "http://127.0.0.1:8805", "auth": {}, "timeout": 7},
            "hello bob", "ctx-s",
        )
        assert reply == "ok"
        sent = posted["body"]["params"]["message"]
        assert sent["metadata"]["a2a.sender"] == {
            "agentId": "helper-agent",
            "name": "helper-agent",
            "url": "http://127.0.0.1:8806",
            "timeout": 7,
        }

    # Recipient side: the stamped sender makes the port-less loopback
    # identity refinable to the helper's real endpoint.
    adapter = _bare_adapter()
    try:
        monkeypatch.setattr(tools, "_load_config", lambda: {})
        refined = adapter._refine_peer_identity("ip:127.0.0.1", {"message": sent}, "ctx-s")
        assert refined == "http://127.0.0.1:8806"
    finally:
        adapter._unregister_adapter()


# ═════════════════════════════════════════════════════════════════════════════
# 6. Self-loop guard
# ═════════════════════════════════════════════════════════════════════════════


def test_unresolvable_loopback_reply_is_loud_failure(monkeypatch, tmp_path):
    """An unmarked session reply whose peer is an unresolvable loopback
    identity must NOT return success=True: audit push_dropped, warning, and
    SendResult(success=False) so the notifier rewinds."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _patch_persistence(monkeypatch)

    adapter = _bare_adapter()
    try:
        adapter.host, adapter.port = "127.0.0.1", 9901
        adapter._context_peers["ctx-loop"] = "ip:127.0.0.1"
        audited: list[tuple[str, str, str, str]] = []
        monkeypatch.setattr(
            security, "audit",
            lambda direction, peer, task_id, summary, context_id=None: audited.append(
                (direction, peer, summary, context_id)
            ),
        )

        async def run():
            return await adapter.send("ctx-loop", "THE_REPLY", metadata={"notify": True})

        res = asyncio.run(run())
        assert res.success is False
        assert "resolvable" in res.error
        assert ("push_dropped", "ip:127.0.0.1", "peer identity not resolvable", "ctx-loop") in audited

        # The reply push path also refuses the own-gateway loopback fallback
        # (kept only for a2a_push notifications) — loud failure, no in-process
        # self-delivery that would ping-pong.
        looped: list = []
        monkeypatch.setattr(adapter, "_push_loopback_in_process", lambda *a, **k: looped.append(a))
        audited.clear()
        adapter._push_out_of_band("ctx-loop", "THE_REPLY", want_reply=True)
        assert looped == []
        assert audited and audited[0][0] == "push_dropped"

        # The kanban-notifier path (a2a_push) still uses the loopback
        # fallback — unchanged fire-and-forget self-delivery.
        looped.clear()
        monkeypatch.setattr(tools, "_load_config", lambda: {"a2a_agents": {}})
        adapter._push_out_of_band("ctx-loop", "NOTIFICATION", want_reply=False)
        assert looped, "a2a_push notifications must keep the loopback fallback"
    finally:
        adapter._unregister_adapter()


# ═════════════════════════════════════════════════════════════════════════════
# 7. Inbound dedupe (_is_duplicate_inbound)
# ═════════════════════════════════════════════════════════════════════════════


def test_inbound_dedupe_drops_repeat_within_window():
    """The same wire message (contextId, messageId) seen again inside
    _INBOUND_DEDUPE_WINDOW is a duplicate; a distinct messageId on the same
    context — or the same messageId on a distinct context — is not.
    Asserts against the module constant so the test tracks implementation
    intent instead of freezing the literal window value."""
    from plugins.platforms.a2a import adapter as adapter_mod

    adapter = _bare_adapter()
    try:
        assert adapter._is_duplicate_inbound("ctx-f", "m-1") is False
        assert adapter._is_duplicate_inbound("ctx-f", "m-1") is True
        assert adapter._is_duplicate_inbound("ctx-f", "m-2") is False
        assert adapter._is_duplicate_inbound("ctx-other", "m-1") is False
    finally:
        adapter._unregister_adapter()


def test_inbound_dedupe_expired_entry_is_accepted_again():
    """Same (contextId, messageId) seen again AFTER the window has elapsed
    is accepted — and the fresh sighting re-arms the window. The entry's
    timestamp is aged past _INBOUND_DEDUPE_WINDOW directly instead of
    sleeping through a real 60s window."""
    from plugins.platforms.a2a import adapter as adapter_mod

    adapter = _bare_adapter()
    try:
        adapter._is_duplicate_inbound("ctx-f", "m-1")
        assert adapter._is_duplicate_inbound("ctx-f", "m-1") is True
        with adapter._inbound_seen_lock:
            adapter._inbound_seen[("ctx-f", "m-1")] = (
                time.time() - adapter_mod._INBOUND_DEDUPE_WINDOW - 1.0
            )
        assert adapter._is_duplicate_inbound("ctx-f", "m-1") is False
        assert adapter._is_duplicate_inbound("ctx-f", "m-1") is True
    finally:
        adapter._unregister_adapter()


def test_inbound_dedupe_seen_map_stays_bounded():
    """The seen map is capped at _INBOUND_DEDUPE_MAX: once full, the oldest
    entries are evicted so a long-running gateway cannot grow it without
    limit. Eviction runs on the call AFTER an insert crosses the cap (the
    insert happens after eviction), so the steady-state invariant is
    len <= MAX + 1 — never 3x the cap despite 3x distinct messages."""
    from plugins.platforms.a2a import adapter as adapter_mod

    adapter = _bare_adapter()
    try:
        cap = adapter_mod._INBOUND_DEDUPE_MAX
        for i in range(cap * 3):
            adapter._is_duplicate_inbound(f"ctx-{i}", f"m-{i}")
        with adapter._inbound_seen_lock:
            assert len(adapter._inbound_seen) <= cap + 1
            assert ("ctx-0", "m-0") not in adapter._inbound_seen  # oldest evicted
    finally:
        adapter._unregister_adapter()


def test_inbound_dedupe_rejects_duplicate_at_prepare_task(monkeypatch, tmp_path):
    """End-to-end at the call site: a repeat (contextId, messageId) inside
    the window is REJECTED by _prepare_task — not dispatched, not
    persisted, not audited. The first sighting flows through the normal
    dispatch path (the bare adapter is 'not ready', which still proves the
    message was accepted past the dedupe guard); a NEW messageId on the
    same context is accepted again."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _patch_persistence(monkeypatch)
    persisted: list[tuple[str, str]] = []
    monkeypatch.setattr(
        protocol, "persist_message",
        lambda cid, role, text, tid: persisted.append((cid, role)),
    )
    audited: list[str] = []
    monkeypatch.setattr(
        security, "audit",
        lambda direction, peer, task_id, summary, context_id=None: audited.append(
            direction
        ),
    )

    adapter = _bare_adapter()
    try:
        agent = {"slug": "a2a", "tenant": "t", "local": True}
        msg = protocol.text_message(protocol.ROLE_USER, "hello", context_id="ctx-dedupe")
        msg["messageId"] = "wire-42"  # peer-stamped id — identical on both copies
        params = {"message": msg}

        terminal, pending = adapter._prepare_task(params, "alice", agent=agent)
        assert pending is None
        assert terminal is not None  # dispatched: bare adapter is 'not ready'
        # Dispatched (only failed because the bare adapter has no loop).
        assert terminal["status"]["state"] == protocol.STATE_FAILED
        assert persisted == [("ctx-dedupe", "user")]
        assert audited == ["inbound"]

        # The identical wire message again within the window: rejected,
        # and nothing was dispatched, persisted, or audited a second time.
        terminal2, pending2 = adapter._prepare_task(params, "alice", agent=agent)
        assert pending2 is None
        assert terminal2 is not None
        assert terminal2["status"]["state"] == protocol.STATE_REJECTED
        assert "Duplicate message." in protocol.extract_text(terminal2["status"]["message"])
        assert persisted == [("ctx-dedupe", "user")]
        assert audited == ["inbound"]

        # A NEW messageId on the same context is accepted again.
        msg2 = protocol.text_message(protocol.ROLE_USER, "second turn", context_id="ctx-dedupe")
        msg2["messageId"] = "wire-43"
        terminal3, _ = adapter._prepare_task({"message": msg2}, "alice", agent=agent)
        assert terminal3 is not None
        assert terminal3["status"]["state"] == protocol.STATE_FAILED
        assert len(persisted) == 2
    finally:
        adapter._unregister_adapter()


# ═════════════════════════════════════════════════════════════════════════════
# 8. Rescue push flag
# ═════════════════════════════════════════════════════════════════════════════


def test_rescue_push_after_client_gone_uses_want_reply(monkeypatch):
    """The client-gone rescue pushes with want_reply=True so the
    peer's answer to the pushed reply re-enters our session from the push's
    HTTP response instead of being discarded. The write-failure tests above
    stub _push_reply_after_client_gone, so they pin that the rescue FIRES
    but not its flag — this asserts the flag on the real implementation.
    Failed tasks carry no reply text and must not push at all."""
    adapter = _bare_adapter()
    try:
        pushed: list[tuple[str, str, bool]] = []
        monkeypatch.setattr(
            adapter, "_push_out_of_band",
            lambda cid, text, want_reply=False: pushed.append((cid, text, want_reply)),
        )

        completed = protocol.jsonrpc_result(
            "req-1",
            protocol.send_message_response(protocol.build_task(
                "task-r", "ctx-r", protocol.STATE_COMPLETED, "RESCUE_REPLY",
            )),
        )
        adapter._push_reply_after_client_gone("req-1", completed)
        assert pushed == [("ctx-r", "RESCUE_REPLY", True)]

        pushed.clear()
        failed = protocol.jsonrpc_result(
            "req-2",
            protocol.send_message_response(protocol.build_task(
                "task-r2", "ctx-r2", protocol.STATE_FAILED, "no reply text",
            )),
        )
        adapter._push_reply_after_client_gone("req-2", failed)
        assert pushed == []
    finally:
        adapter._unregister_adapter()

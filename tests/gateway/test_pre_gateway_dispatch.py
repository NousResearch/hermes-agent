"""Tests for the pre_gateway_dispatch plugin hook.

The hook allows plugins to intercept incoming messages before auth and
agent dispatch. It runs in _handle_message and acts on returned action
dicts: {"action": "skip"|"rewrite"|"allow"}.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


def _clear_auth_env(monkeypatch) -> None:
    for key in (
        "TELEGRAM_ALLOWED_USERS",
        "WHATSAPP_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "WHATSAPP_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_event(text: str = "hello", platform: Platform = Platform.WHATSAPP) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_id="m1",
        source=SessionSource(
            platform=platform,
            user_id="15551234567@s.whatsapp.net",
            chat_id="15551234567@s.whatsapp.net",
            user_name="tester",
            chat_type="dm",
        ),
    )


def _make_runner(platform: Platform):
    from gateway.run import GatewayRunner

    config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True)},
    )
    runner = object.__new__(GatewayRunner)
    runner.config = config
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {platform: adapter}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    runner.session_store = MagicMock()
    runner._running_agents = {}
    runner._update_prompt_pending = {}
    return runner, adapter


@pytest.mark.asyncio
async def test_internal_events_bypass_hook(monkeypatch):
    """Internal events (event.internal=True) skip the plugin hook entirely."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

    called = {"count": 0}

    def _fake_hook(name, **kwargs):
        called["count"] += 1
        return [{"action": "skip"}]

    async def _capture(event, source, _quick_key, _run_generation):
        return "ok"

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    runner._handle_message_with_agent = _capture

    event = _make_event("hi")
    event.internal = True

    # Even though the hook would say skip, internal events bypass it.
    await runner._handle_message(event)
    assert called["count"] == 0

@pytest.mark.asyncio
async def test_hook_fires_without_session_store_attribute(monkeypatch):
    """A runner missing session_store still delivers the event to plugins.

    Regression: the hook kwargs read ``self.session_store`` directly, so a
    partially-initialized runner raised AttributeError inside the dispatch
    try-block — the hook never fired, and every message logged
    "pre_gateway_dispatch invocation failed: 'GatewayRunner' object has no
    attribute 'session_store'". Plugins must receive the event (with
    session_store=None) instead.
    """
    _clear_auth_env(monkeypatch)

    seen = {}

    async def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            seen["session_store"] = kwargs.get("session_store", "MISSING")
            return [{"action": "skip", "reason": "plugin-handled"}]
        return []

    # The inbound path awaits the hook, so the seam is the async entry point.
    monkeypatch.setattr("hermes_cli.plugins.ainvoke_hook", _fake_hook)

    runner, adapter = _make_runner(Platform.WHATSAPP)
    del runner.session_store

    result = await runner._handle_message(_make_event("hi"))
    assert result is None
    # Hook actually fired (skip short-circuited before auth) with a None store.
    assert seen == {"session_store": None}
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_async_hook_callback_is_awaited_on_the_gateway_loop(monkeypatch):
    """An ``async def`` pre_gateway_dispatch callback is awaited on the gateway's own loop.

    Regression: the inbound path called the sync ``invoke_hook``, which (since #109196) runs an
    async callback on a helper thread while the calling loop blocks in ``done.wait()``. A
    callback that awaits anything scheduled on the gateway loop could never complete, and every
    message stalled the loop for the callback's whole duration. Here the callback waits for a
    sibling task on the same loop to release it; that is only possible if the hook is awaited
    in place.
    """
    import asyncio

    _clear_auth_env(monkeypatch)
    gate = asyncio.Event()

    async def _hook(name, **kwargs):
        assert name == "pre_gateway_dispatch"
        await gate.wait()
        return [{"action": "skip", "reason": "gated"}]

    monkeypatch.setattr("hermes_cli.plugins.ainvoke_hook", _hook)

    async def _release():
        await asyncio.sleep(0)
        gate.set()

    runner, adapter = _make_runner(Platform.WHATSAPP)
    asyncio.create_task(_release())
    result = await asyncio.wait_for(runner._handle_message(_make_event("hi")), timeout=5)
    assert result is None
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_rewritten_message_keeps_its_callback_provenance_and_is_admitted(monkeypatch):
    """A rewrite builds a new event. Under ``gateway.multiplex_profiles`` an event without its
    adapter callback's provenance is refused ``invalid_params`` (#139045: every message to a profile
    with a rewriting plugin failed), so the rewritten event must carry the callback's provenance and
    be admitted with the rewritten text."""
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.session_ingress_context import capture_provenance, native_callback, register_transport_home
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import list_session_admissions
    from plugins.platforms.discord.adapter import DiscordAdapter

    async def rewrite(name, **_kwargs):
        return [{"action": "rewrite", "text": "rewritten"}] if name == "pre_gateway_dispatch" else []

    monkeypatch.setattr("hermes_cli.lifecycle.ainvoke_hook", rewrite)
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "42")
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fixture-token", typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="owner")
    runner._wire_adapter_handlers(adapter)
    # The multiplexed admission check that refused the reporter (thanks @unsupportedpastels).
    runner.config.multiplex_profiles = True
    runner.config._runtime_profile_homes = [("default", get_hermes_home())]
    register_transport_home(runner, None, get_hermes_home())
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    human = MessageEvent(text="original", source=source, message_id="h")

    with native_callback(runner, human, get_hermes_home()):
        expected = capture_provenance(runner, human)
        event = await runner._hm_pre_gateway_dispatch_hook(human, source)
        receipt = await authority.admit_native(event)

    payload = list_session_admissions(authority.db, session_id=receipt.ref.session_id, pending_only=False)[-1]["payload"]
    assert expected is not None and event is not human
    assert (payload["text"], payload["native_text_v1"].get("provenance")) == ("rewritten", expected)


@pytest.mark.asyncio
async def test_hook_fires_once_at_messaging_ingress_and_never_when_an_admission_executes(monkeypatch):
    """PLG-5: executing an admitted row (a local CLI/TUI/Desktop turn, or a messaging turn's first
    run or restart replay) re-enters ``_handle_message``. The hook already ran at ingress (main has
    no other firing point), so it must not run again there, and a "skip" never drops admitted input."""
    import gateway.session_authorities as authorities
    import gateway.session_ingress as ingress

    _clear_auth_env(monkeypatch)
    fired = []

    async def hook(name, **kwargs):
        if name != "pre_gateway_dispatch":
            return []
        fired.append(kwargs["event"].text)
        return [{"action": "skip"}] if kwargs["event"].text == "drop me" else []

    async def admit(_authority, event):
        admitted.append(event.text)
        return "admitted"

    admitted = []
    monkeypatch.setattr("hermes_cli.lifecycle.ainvoke_hook", hook)
    monkeypatch.setattr(authorities, "active_authority", lambda _runner: object())
    monkeypatch.setattr(ingress, "admit_message", admit)
    runner, adapter = _make_runner(Platform.WHATSAPP)
    runner._is_user_authorized_for_source = lambda _s: True
    runner._admit_bot_message_for_source = lambda _s: True
    runner._delivery_adapter_for = runner._intake_adapter_for = lambda _s: adapter
    runner._rescue_orphaned_overflow = lambda _k, _a: None
    runner._enqueue_fifo = MagicMock()
    runner._handle_message_with_agent = AsyncMock(return_value="agent-ok")
    runner._run_post_turn_hooks = AsyncMock()

    assert await runner._handle_message(_make_event("drop me")) is None
    assert await runner._handle_message(_make_event("keep me")) == "admitted"
    assert admitted == ["keep me"] and fired == ["drop me", "keep me"]

    token = ingress.executing_admission.set(True)
    try:
        assert await runner._handle_message(_make_event("drop me")) == "agent-ok"
    finally:
        ingress.executing_admission.reset(token)
    assert fired == ["drop me", "keep me"]


@pytest.mark.asyncio
async def test_webhook_delivery_fires_the_hook_once_at_ingress_before_commit():
    """PLG-5: on main the webhook adapter's ``handle_message`` fired the hook once per delivery,
    before anything ran. Receipt-only ingress (``admit_producer``) must keep that: a skip commits
    nothing, a rewrite is what gets committed and executed, the real drain's execution does not
    fire it again, and a provider retry of a committed delivery is not offered again."""
    import hashlib
    import hmac
    import json

    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from gateway.platforms.webhook import WebhookAdapter
    from hermes_cli.plugins import get_plugin_manager
    from tests.gateway.fixtures.webhook_route_authority import mount_authority
    from tests.waiting import async_wait_until

    fired = []

    def hook(event, **_kwargs):
        fired.append(event.message_id)
        if "drop" in event.text:
            return {"action": "skip"}
        return {"action": "rewrite", "text": "rewritten"} if "rewrite" in event.text else None

    manager = get_plugin_manager()
    manager._discovered = True
    manager._hooks.setdefault("pre_gateway_dispatch", []).append(hook)
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"secret": "owned-secret", "routes": {
        "fixture": {"prompt": "{text}"}}}))
    adapter._deliver_to = AsyncMock()
    app = web.Application()
    mount_authority(app, adapter)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)

    async def post(client, text, delivery_id):
        body = json.dumps({"text": text}).encode()
        signature = "sha256=" + hmac.new(b"owned-secret", body, hashlib.sha256).hexdigest()
        return await client.post("/webhooks/fixture", data=body, headers={
            "X-GitHub-Delivery": delivery_id, "X-Hub-Signature-256": signature})

    async with TestClient(TestServer(app)) as client:
        runner = adapter._message_handler.__self__
        authority = runner.session_authority
        del authority._schedule  # this test needs the real drain to execute the admitted row
        runner._handle_message_with_agent = AsyncMock(return_value="agent-ok")
        assert (await post(client, "drop this", "d-drop")).status == 202
        assert fired == ["d-drop"] and not authority.db._read_all("SELECT 1 FROM session_admissions")

        assert (await post(client, "keep this", "d-keep")).status == 202
        assert (await post(client, "please rewrite", "d-rewrite")).status == 202
        def settled():
            rows = authority.db._read_all("SELECT request_id, status, payload_json FROM session_admissions")
            return rows if len(rows) == 2 and all(row["status"] == "terminal" for row in rows) else None

        rows = await async_wait_until(settled, what="both admitted deliveries executed")
        assert sorted((row["request_id"], row["status"], json.loads(row["payload_json"])["text"]) for row in rows) == [
            ("d-keep", "terminal", "keep this"), ("d-rewrite", "terminal", "rewritten")]
        executed = sorted(call.args[0].text for call in runner._handle_message_with_agent.await_args_list)
        assert executed == ["keep this", "rewritten"]
        assert (await post(client, "keep this", "d-keep")).status in (200, 202)
    assert fired == ["d-drop", "d-keep", "d-rewrite"]


def _rewriting_hook(fired):
    from hermes_cli.plugins import get_plugin_manager

    def hook(event, **_kwargs):
        fired.append(event.message_id)
        return {"action": "rewrite", "text": "rewritten"} if "rewrite" in event.text else None

    manager = get_plugin_manager()
    manager._discovered = True
    manager._hooks.setdefault("pre_gateway_dispatch", []).append(hook)


def _webhook_app(prompt="{text}"):
    from aiohttp import web

    from gateway.platforms.webhook import WebhookAdapter
    from tests.gateway.fixtures.webhook_route_authority import mount_authority

    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"secret": "owned-secret", "routes": {
        "fixture": {"prompt": prompt}}}))
    adapter._deliver_to = AsyncMock()
    app = web.Application()
    mount_authority(app, adapter)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return adapter, app


async def _post_webhook(client, text, delivery_id):
    import hashlib
    import hmac
    import json

    body = json.dumps(text if isinstance(text, dict) else {"text": text}).encode()
    signature = "sha256=" + hmac.new(b"owned-secret", body, hashlib.sha256).hexdigest()
    return await client.post("/webhooks/fixture", data=body, headers={
        "X-GitHub-Delivery": delivery_id, "X-Hub-Signature-256": signature})


@pytest.mark.asyncio
async def test_a_resend_whose_raw_payload_changed_conflicts_even_when_its_prompt_did_not():
    """The hook sees the parsed provider payload (``raw_message``), not only the rendered prompt. A
    resend under the same delivery id whose prompt renders the same but whose raw field the hook
    rewrote from changed is refused: no duplicate receipt, no second hook call, no second row. The
    exact resend keeps its receipt, and so does a delivery rewritten to empty text."""
    import json

    from aiohttp.test_utils import TestClient, TestServer
    from hermes_cli.plugins import get_plugin_manager

    fired = []

    def hook(event, **_kwargs):
        fired.append(event.raw_message["detail"])
        return {"action": "rewrite", "text": event.raw_message["detail"].replace("blank", "")}

    manager = get_plugin_manager()
    manager._discovered = True
    manager._hooks.setdefault("pre_gateway_dispatch", []).append(hook)
    adapter, app = _webhook_app(prompt="{action}")
    async with TestClient(TestServer(app)) as client:
        authority = adapter._message_handler.__self__.session_authority
        for delivery_id, detail in (("d-1", "handle one"), ("d-2", "blank")):
            assert (await _post_webhook(client, {"action": "opened", "detail": detail}, delivery_id)).status == 202
            retry = await _post_webhook(client, {"action": "opened", "detail": detail}, delivery_id)
            assert (retry.status, (await retry.json())["status"]) == (200, "duplicate")
        changed = await _post_webhook(client, {"action": "opened", "detail": "handle two"}, "d-1")
        assert changed.status == 503
        rows = authority.db._read_all("SELECT request_id, payload_json FROM session_admissions")
    assert sorted((row["request_id"], json.loads(row["payload_json"])["text"]) for row in rows) == [
        ("d-1", "handle one"), ("d-2", "")]
    assert fired == ["handle one", "blank"]


@pytest.mark.asyncio
async def test_rewritten_webhook_delivery_retries_to_its_receipt_and_a_changed_one_is_refused():
    """The committed payload of a rewritten delivery differs from what the provider resends. An exact
    retry still gets the duplicate receipt (not 503 forever), without offering it to the hook again;
    a changed payload under the same delivery id is still refused."""
    import json

    from aiohttp.test_utils import TestClient, TestServer

    fired = []
    _rewriting_hook(fired)
    adapter, app = _webhook_app()
    async with TestClient(TestServer(app)) as client:
        authority = adapter._message_handler.__self__.session_authority
        assert (await _post_webhook(client, "please rewrite", "d-1")).status == 202
        retry = await _post_webhook(client, "please rewrite", "d-1")
        assert (retry.status, (await retry.json())["status"]) == (200, "duplicate")
        assert (await _post_webhook(client, "please rewrite it twice", "d-1")).status == 503
        rows = authority.db._read_all("SELECT payload_json FROM session_admissions")
    assert [json.loads(row["payload_json"])["text"] for row in rows] == ["rewritten"]
    assert fired == ["d-1"]


@pytest.mark.asyncio
async def test_a_webhook_row_committed_before_the_fingerprint_retries_exactly_and_runs_without_the_hook():
    """Upgrade: a row committed before ingress fingerprints (every main-era row: the original payload,
    no hook at ingress) keeps exact retry comparison, and when it executes after the upgrade the hook
    does not fire: the committed text runs as admitted."""
    import json

    from aiohttp.test_utils import TestClient, TestServer

    from gateway.session_contract import SessionRef
    from tests.waiting import async_wait_until

    fired = []
    adapter, app = _webhook_app()
    async with TestClient(TestServer(app)) as client:
        runner = adapter._message_handler.__self__
        authority = runner.session_authority
        runner._handle_message_with_agent = AsyncMock(return_value="agent-ok")
        assert (await _post_webhook(client, "please rewrite", "d-old")).status == 202
        _rewriting_hook(fired)  # the plugin arrives with the upgrade
        retry = await _post_webhook(client, "please rewrite", "d-old")
        assert (retry.status, (await retry.json())["status"]) == (200, "duplicate")
        assert (await _post_webhook(client, "please rewrite it twice", "d-old")).status == 503
        [row] = authority.db._read_all("SELECT target_session_id, payload_json FROM session_admissions")
        assert "ingress_sha256" not in json.loads(row["payload_json"])["native_text_v1"]

        del authority._schedule  # the queued row now runs on the real drain
        authority._schedule(SessionRef(authority.profile_id, row["target_session_id"]))
        await async_wait_until(lambda: authority.db._read_all("SELECT 1 FROM session_admissions WHERE status='terminal'"),
                               what="the queued row executed")
        executed = [call.args[0].text for call in runner._handle_message_with_agent.await_args_list]
    assert executed == ["please rewrite"]
    assert fired == []


@pytest.mark.asyncio
async def test_relay_resend_of_a_rewritten_delivery_resolves_its_admission(tmp_path, monkeypatch):
    """A relay re-send of a rewritten delivery resolves the existing admission (same receipt, no second
    hook call, one row) instead of being acknowledged as a permanent ``admission_conflict``."""
    import asyncio
    import json

    websockets = pytest.importorskip("websockets")
    import gateway.platforms.webhook_ingress as ingress
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CapabilityDescriptor
    from gateway.relay.ws_transport import WebSocketRelayTransport
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result
    from tests.waiting import async_wait_until

    outgoing, ready, turns, outcomes, fired = [], asyncio.Queue(), [], [], []
    descriptor = CapabilityDescriptor(1, "telegram", "Fixture", 4096, False, True, False, "plain", "chars")

    async def connector(ws):
        await ws.recv()  # hello
        await ws.send(json.dumps({"type": "descriptor", "descriptor": json.loads(descriptor.to_json())}) + "\n")
        await ready.put(ws)
        async for line in ws:
            frame = json.loads(line)
            outgoing.append(frame)
            if frame["type"] == "outbound":
                await ws.send(json.dumps({"type": "outbound_result", "requestId": frame["requestId"],
                                          "result": {"success": True, "message_id": "sent"}}) + "\n")

    async def turn(event, *_args, **_kwargs):
        turns.append(event.text)
        execution_result.get()["result"] = {"final_response": "ok", "messages": []}
        return "ok"

    admit = ingress.admit_producer

    async def observed(adapter, event):
        try:
            receipt = await admit(adapter, event)
        except Exception as exc:
            outcomes.append(getattr(exc, "reason", repr(exc)))
            raise
        outcomes.append(receipt.admission_id)
        return receipt

    monkeypatch.setattr(ingress, "admit_producer", observed)
    _rewriting_hook(fired)
    async with websockets.serve(connector, "127.0.0.1", 0) as relay:
        transport = WebSocketRelayTransport(
            f"ws://127.0.0.1:{relay.sockets[0].getsockname()[1]}/relay", "telegram", "fixture-bot",
            gateway_id="fixture-gateway", upgrade_secret="fixture-secret", reconnect=True, reconnect_backoff_s=.05)
        runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
        monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
        authority = await initialize_session_authority(runner, profile_id="default", instance_id="relay-retry")
        adapter = RelayAdapter(PlatformConfig(enabled=True), descriptor, transport)
        adapter.gateway_runner = runner
        runner.adapters[Platform.RELAY] = adapter
        runner._wire_adapter_handlers(adapter)
        try:
            assert await adapter.connect()
            ws = await asyncio.wait_for(ready.get(), 5)
            event = {"text": "please rewrite", "message_id": "r1",
                     "source": {"platform": "telegram", "chat_id": "relay-chat", "chat_type": "dm",
                                "user_id": "relay-user"}}
            acked = lambda: {f.get("bufferId") for f in outgoing if f.get("type") == "inbound_ack"}
            for buffer_id in ("buf-1", "buf-2"):  # the connector re-sends after a lost ACK
                await ws.send(json.dumps({"type": "inbound", "event": event, "bufferId": buffer_id}) + "\n")
                await async_wait_until(lambda: buffer_id in acked() and turns, timeout=20,
                                       what=f"{buffer_id} acknowledged after the turn ran")
            rows = authority.db._read_all("SELECT admission_id, payload_json FROM session_admissions")
        finally:
            await adapter.disconnect()
            authority.db.close()
    assert [json.loads(row["payload_json"])["text"] for row in rows] == ["rewritten"]
    assert outcomes == [rows[0]["admission_id"]] * 2
    assert (turns, fired) == (["rewritten"], ["r1"])


@pytest.mark.asyncio
async def test_relay_resend_of_a_rewritten_media_delivery_reuses_its_admission_and_changed_bytes_conflict(
        tmp_path, monkeypatch):
    """A rewritten relay delivery with media, still queued: an exact re-send (same bytes, re-downloaded
    under a fresh staging name) resolves to the original admission and releases its freshly retained
    copy; a re-send whose bytes changed conflicts and retains nothing."""
    import asyncio
    import json
    import threading
    from http.server import ThreadingHTTPServer
    from pathlib import Path

    websockets = pytest.importorskip("websockets")
    import gateway.platforms.webhook_ingress as ingress
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CapabilityDescriptor
    from gateway.relay.media import RelayMediaClient
    from gateway.relay.ws_transport import WebSocketRelayTransport
    from gateway.run import GatewayRunner
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.session_ingress_media import _media_root
    from tests.gateway.test_native_media_redelivery import PNG, _Media
    from tests.waiting import async_wait_until

    outgoing, ready, outcomes, fired = [], asyncio.Queue(), [], []
    descriptor = CapabilityDescriptor(1, "telegram", "Fixture", 4096, False, True, False, "plain", "chars")
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)  # the row stays queued, bytes held
    media = ThreadingHTTPServer(("127.0.0.1", 0), _Media)
    threading.Thread(target=media.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{media.server_port}/photo.png"

    async def connector(ws):
        await ws.recv()  # hello
        await ws.send(json.dumps({"type": "descriptor", "descriptor": json.loads(descriptor.to_json())}) + "\n")
        await ready.put(ws)
        async for line in ws:
            frame = json.loads(line)
            outgoing.append(frame)
            if frame["type"] == "outbound":
                await ws.send(json.dumps({"type": "outbound_result", "requestId": frame["requestId"],
                                          "result": {"success": True, "message_id": "sent"}}) + "\n")

    admit = ingress.admit_producer

    async def observed(adapter, event):
        try:
            receipt = await admit(adapter, event)
        except Exception as exc:
            outcomes.append(getattr(exc, "reason", repr(exc)))
            raise
        outcomes.append(receipt.admission_id)
        return receipt

    monkeypatch.setattr(ingress, "admit_producer", observed)
    _rewriting_hook(fired)
    retained = lambda: sorted(_media_root().rglob("*.png"))
    async with websockets.serve(connector, "127.0.0.1", 0) as relay:
        transport = WebSocketRelayTransport(
            f"ws://127.0.0.1:{relay.sockets[0].getsockname()[1]}/relay", "telegram", "fixture-bot",
            gateway_id="fixture-gateway", upgrade_secret="fixture-secret", reconnect=True, reconnect_backoff_s=.05)
        runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
        authority = await initialize_session_authority(runner, profile_id="default", instance_id="relay-media")
        adapter = RelayAdapter(PlatformConfig(enabled=True), descriptor, transport)
        adapter._media_client = RelayMediaClient(f"http://127.0.0.1:{media.server_port}", None, None)
        adapter.gateway_runner = runner
        runner.adapters[Platform.RELAY] = adapter
        runner._wire_adapter_handlers(adapter)
        acked = lambda: {f.get("bufferId") for f in outgoing if f.get("type") == "inbound_ack"}
        try:
            assert await adapter.connect()
            ws = await asyncio.wait_for(ready.get(), 5)
            event = {"text": "please rewrite", "message_id": "r1", "message_type": "photo",
                     "media_urls": [url], "media": [{"url": url, "mime": "image/png"}],
                     "source": {"platform": "telegram", "chat_id": "relay-chat", "chat_type": "dm",
                                "user_id": "relay-user"}}
            for buffer_id in ("buf-1", "buf-2"):  # the connector re-sends after a lost ACK
                await ws.send(json.dumps({"type": "inbound", "event": event, "bufferId": buffer_id}) + "\n")
                await async_wait_until(lambda: buffer_id in acked() and len(outcomes) == int(buffer_id[-1]),
                                       timeout=20, what=f"{buffer_id} acknowledged")
            kept = retained()
            _Media.body = PNG + b"changed"
            await ws.send(json.dumps({"type": "inbound", "event": event, "bufferId": "buf-3"}) + "\n")
            await async_wait_until(lambda: "buf-3" in acked(), timeout=20, what="the changed re-send acknowledged")
            after_conflict = retained()
            rows = authority.db._read_all("SELECT admission_id, payload_json FROM session_admissions")
        finally:
            _Media.body = PNG
            media.shutdown()
            media.server_close()
            await adapter.disconnect()
            authority.db.close()
    [row] = rows
    committed = json.loads(row["payload_json"])
    assert committed["text"] == "rewritten" and len(committed["native_text_v1"]["media"]) == 1
    assert outcomes == [row["admission_id"]] * 2 + ["admission_conflict"]
    assert kept == after_conflict == [Path(committed["native_text_v1"]["media"][0]["path"])], (
        "a re-send's fresh copy must be released, the original's kept")
    assert fired == ["r1"]


@pytest.mark.asyncio
async def test_an_identical_webhook_delivery_waits_for_the_running_hook_and_gets_its_receipt(monkeypatch):
    """Overlapping identical deliveries (a provider retry while the first request is still in the hook)
    are coordinated by delivery identity: the second waits, rechecks the ledger, and resolves to the
    committed receipt, so the hook and its external actions run once and one row is committed."""
    import asyncio
    import json

    from aiohttp.test_utils import TestClient, TestServer

    import gateway.platforms.webhook_ingress as ingress
    from hermes_cli.plugins import get_plugin_manager
    from tests.waiting import async_wait_until

    fired, release, entered = [], asyncio.Event(), []

    async def hook(event, **_kwargs):
        fired.append(event.message_id)
        await release.wait()
        return {"action": "rewrite", "text": "rewritten"}

    manager = get_plugin_manager()
    manager._discovered = True
    manager._hooks.setdefault("pre_gateway_dispatch", []).append(hook)
    admit = ingress.admit_producer

    async def observed(adapter, event):
        entered.append(event.message_id)
        return await admit(adapter, event)

    monkeypatch.setattr(ingress, "admit_producer", observed)
    adapter, app = _webhook_app()
    async with TestClient(TestServer(app)) as client:
        authority = adapter._message_handler.__self__.session_authority
        first = asyncio.create_task(_post_webhook(client, "please rewrite", "d-1"))
        await async_wait_until(lambda: fired, what="the first delivery in the hook")
        second = asyncio.create_task(_post_webhook(client, "please rewrite", "d-1"))
        parked = lambda: any(waiting > 1 for _lock, waiting in getattr(ingress, "_deliveries", {}).values())
        await async_wait_until(lambda: len(entered) == 2 and (len(fired) > 1 or parked()),
                               what="the identical delivery parked or in the hook")
        release.set()
        first, second = await asyncio.gather(first, second)
        assert first.status == 202
        assert (second.status, (await second.json())["status"]) == (200, "duplicate")
        rows = authority.db._read_all("SELECT payload_json FROM session_admissions")
    assert [json.loads(row["payload_json"])["text"] for row in rows] == ["rewritten"]
    assert fired == ["d-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("hook_name, verdict, outcome", [
    ("pre_gateway_dispatch", {"action": "skip"}, None),
    ("post_gateway_admission", {"action": "handled"}, None),
    ("pre_gateway_dispatch", {"action": "rewrite", "text": "/status"}, "invalid_params"),
    ("pre_gateway_dispatch", "drain", "runtime_draining"),
])
async def test_a_relay_media_delivery_no_admission_owns_retains_no_bytes(
        tmp_path, monkeypatch, hook_name, verdict, outcome):
    """Preparing a producer delivery copies its media into the retained store before the hooks run. A
    delivery a plugin skips or consumes, or whose admission is refused after the hook (a command-shaped
    rewrite, a drain that began during the hook), is never admitted: that copy is released."""
    import asyncio
    import json
    import threading
    from http.server import ThreadingHTTPServer

    websockets = pytest.importorskip("websockets")
    import gateway.platforms.webhook_ingress as ingress
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CapabilityDescriptor
    from gateway.relay.media import RelayMediaClient
    from gateway.relay.ws_transport import WebSocketRelayTransport
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_ingress_media import _media_root
    from hermes_cli.plugins import get_plugin_manager
    from tests.gateway.test_native_media_redelivery import _Media
    from tests.waiting import async_wait_until

    outgoing, ready, fired, outcomes = [], asyncio.Queue(), [], []
    descriptor = CapabilityDescriptor(1, "telegram", "Fixture", 4096, False, True, False, "plain", "chars")
    media = ThreadingHTTPServer(("127.0.0.1", 0), _Media)
    threading.Thread(target=media.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{media.server_port}/photo.png"

    def hook(**kwargs):
        fired.append(hook_name)
        if verdict == "drain":
            runner._draining = True  # the owner began draining while the hook ran
            return None
        return verdict

    manager = get_plugin_manager()
    manager._discovered = True
    manager._hooks.setdefault(hook_name, []).append(hook)
    admit = ingress.admit_producer

    async def observed(adapter, event):
        try:
            outcomes.append(await admit(adapter, event))
        except Exception as exc:
            outcomes.append(getattr(exc, "reason", repr(exc)))
            raise

    monkeypatch.setattr(ingress, "admit_producer", observed)

    async def connector(ws):
        await ws.recv()  # hello
        await ws.send(json.dumps({"type": "descriptor", "descriptor": json.loads(descriptor.to_json())}) + "\n")
        await ready.put(ws)
        async for line in ws:
            outgoing.append(json.loads(line))

    async with websockets.serve(connector, "127.0.0.1", 0) as relay:
        transport = WebSocketRelayTransport(
            f"ws://127.0.0.1:{relay.sockets[0].getsockname()[1]}/relay", "telegram", "fixture-bot",
            gateway_id="fixture-gateway", upgrade_secret="fixture-secret", reconnect=True, reconnect_backoff_s=.05)
        runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
        authority = await initialize_session_authority(runner, profile_id="default", instance_id="relay-skip")
        adapter = RelayAdapter(PlatformConfig(enabled=True), descriptor, transport)
        adapter._media_client = RelayMediaClient(f"http://127.0.0.1:{media.server_port}", None, None)
        adapter.gateway_runner = runner
        runner.adapters[Platform.RELAY] = adapter
        runner._wire_adapter_handlers(adapter)
        try:
            assert await adapter.connect()
            ws = await asyncio.wait_for(ready.get(), 5)
            event = {"text": "look", "message_id": "r1", "message_type": "photo",
                     "media_urls": [url], "media": [{"url": url, "mime": "image/png"}],
                     "source": {"platform": "telegram", "chat_id": "relay-chat", "chat_type": "dm",
                                "user_id": "relay-user"}}
            await ws.send(json.dumps({"type": "inbound", "event": event, "bufferId": "buf-1"}) + "\n")
            await async_wait_until(lambda: outcomes, timeout=20, what="the delivery decided")
            rows = authority.db._read_all("SELECT 1 FROM session_admissions")
        finally:
            runner._draining = False
            media.shutdown()
            media.server_close()
            await adapter.disconnect()
            authority.db.close()
    assert (fired, outcomes, rows) == ([hook_name], [outcome], [])
    assert not list(_media_root().rglob("*.png")), "a delivery no admission owns left its retained copy"


def test_relay_binding_moves_to_a_text_rewrite_but_never_to_a_new_source():
    """The relay binding gets the same treatment as the native one, under the same guard: it follows
    a text-only rewrite, and is never carried to an event whose source object was swapped."""
    from copy import copy
    from dataclasses import replace

    from gateway.session_ingress_context import _relay_callback, carry_callback, relay_callback

    transport, event = object(), _make_event("a")
    with relay_callback(transport, event):
        rewritten = replace(event, text="b")
        carry_callback(event, rewritten)
        assert _relay_callback.get() == (transport, rewritten)
        carry_callback(rewritten, replace(rewritten, source=copy(rewritten.source), text="c"))
        assert _relay_callback.get()[1] is rewritten
    assert _relay_callback.get() is None

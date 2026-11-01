"""``post_gateway_admission``: a fail-open plugin hook that may consume an admitted message (#129958).

Drives the real ``GatewayRunner._handle_message`` pipeline and the real plugin dispatch: callbacks
are registered on the per-home ``PluginManager`` exactly as ``ctx.register_hook`` stores them.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import _profile_runtime_scope
from gateway.session import SessionSource
from hermes_constants import get_hermes_home

HOOK = "post_gateway_admission"


def _register(callback) -> None:
    """Register *callback* on the ACTIVE home's plugin manager (what a loaded plugin does)."""
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    manager._discovered = True  # no on-disk plugins in the test home; skip discovery
    manager._hooks.setdefault(HOOK, []).append(callback)


def _runner(*, multiplex=False):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.WHATSAPP: PlatformConfig(enabled=True)}, multiplex_profiles=multiplex,
    )
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.pairing_store = MagicMock()
    runner.session_store = MagicMock()
    runner._is_user_authorized_for_source = lambda _s: True
    runner._admit_bot_message_for_source = lambda _s: True
    runner._delivery_adapter_for = lambda _s: adapter
    runner._intake_adapter_for = lambda _s: adapter
    runner._rescue_orphaned_overflow = lambda _k, _a: None
    runner._enqueue_fifo = MagicMock()
    runner._handle_message_with_agent = AsyncMock(return_value="agent-ok")
    runner._run_post_turn_hooks = AsyncMock()
    return runner


def _event(text="queue my report", profile=None, *, media_urls=None, media_types=None,
           media_text_inlined=None, media_filenames=None):
    source = SessionSource(
        platform=Platform.WHATSAPP, user_id="15551234567@s.whatsapp.net",
        chat_id="15551234567@s.whatsapp.net", user_name="tester", chat_type="dm", profile=profile,
    )
    event = MessageEvent(
        text=text, message_id="m1", source=source,
        media_urls=media_urls or [], media_types=media_types or [],
        media_text_inlined=media_text_inlined or [],
    )
    # Keep these tests compatible with older MessageEvent versions; adapters may add the optional
    # field before it becomes a declared dataclass field.
    if media_filenames is not None:
        event.media_filenames = media_filenames
    return event


@pytest.fixture(autouse=True)
def _allow_all(monkeypatch):
    for key in ("WHATSAPP_ALLOWED_USERS", "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")


@pytest.mark.asyncio
async def test_handled_consumes_the_turn_and_its_reply_is_delivered():
    seen = []

    def consume(**kwargs):
        seen.append(kwargs)
        return {"action": "handled", "reply": "Queued - I'll follow up."}

    _register(consume)
    runner = _runner()

    assert await runner._handle_message(_event()) == "Queued - I'll follow up."
    # The session slot is released on the consume path: the next message is admitted, not queued.
    assert await runner._handle_message(_event("another")) == "Queued - I'll follow up."
    runner._handle_message_with_agent.assert_not_awaited()
    # Snapshot-only payload: message + routing facts, never live runner/store handles.
    first = seen[0]
    assert first["text"] == "queue my report" and first["message_id"] == "m1"
    assert first["platform"] == "whatsapp"
    assert first["source"]["chat_id"] == "15551234567@s.whatsapp.net" and first["session_key"]
    assert "gateway" not in first and "session_store" not in first


@pytest.mark.asyncio
async def test_authorized_consumer_receives_attachment_metadata_only_after_auth(tmp_path):
    seen = []
    _register(lambda **kwargs: seen.append(kwargs) or {"action": "handled"})
    media = tmp_path / "synthetic.jpg"
    media.write_bytes(b"synthetic-only")

    denied = _runner()
    denied._is_user_authorized_for_source = lambda _source: False
    denied._get_unauthorized_dm_behavior = lambda *_args, **_kwargs: "ignore"
    denied._hm_report_ignored_dm = AsyncMock()
    denied_event = _event(
        media_urls=[str(media)], media_types=["image/jpeg"], media_text_inlined=[False],
    )
    assert await denied._handle_message(denied_event) is None
    assert seen == []

    runner = _runner()
    event = _event(
        media_urls=[str(media)], media_types=["image/jpeg"], media_text_inlined=[False],
        media_filenames=["synthetic-original.jpg"],
    )
    assert await runner._handle_message(event) is None
    assert seen[0]["media_urls"] == [str(media)]
    assert seen[0]["media_types"] == ["image/jpeg"]
    assert seen[0]["media_text_inlined"] == [False]
    assert seen[0]["media_filenames"] == ["synthetic-original.jpg"]


@pytest.mark.asyncio
async def test_hook_aligns_missing_attachment_metadata_as_isolated_copies(tmp_path):
    received = []

    def consume(**kwargs):
        received.append({
            "media_urls": list(kwargs["media_urls"]),
            "media_types": list(kwargs["media_types"]),
            "media_text_inlined": list(kwargs["media_text_inlined"]),
            "media_filenames": list(kwargs["media_filenames"]),
        })
        for key in ("media_urls", "media_types", "media_text_inlined", "media_filenames"):
            kwargs[key].clear()
        return {"action": "handled"}

    _register(consume)
    urls = [str(tmp_path / "synthetic-photo.jpg"), str(tmp_path / "synthetic-audio.ogg")]
    event = _event(media_urls=urls)

    assert await _runner()._handle_message(event) is None
    assert received == [{
        "media_urls": urls,
        "media_types": ["", ""],
        "media_text_inlined": [None, None],
        "media_filenames": [None, None],
    }]
    assert event.media_urls == urls
    assert event.media_types == []
    assert event.media_text_inlined == []
    assert not hasattr(event, "media_filenames")


@pytest.mark.asyncio
async def test_hook_passes_platform_filenames_as_aligned_copies(tmp_path):
    received = []

    def consume(**kwargs):
        received.append({
            "media_urls": list(kwargs["media_urls"]),
            "media_types": list(kwargs["media_types"]),
            "media_text_inlined": list(kwargs["media_text_inlined"]),
            "media_filenames": list(kwargs["media_filenames"]),
        })
        kwargs["media_urls"].clear()
        kwargs["media_types"].clear()
        kwargs["media_text_inlined"].clear()
        kwargs["media_filenames"].clear()
        return {"action": "handled"}

    _register(consume)
    urls = [str(tmp_path / "cached-a.bin"), str(tmp_path / "cached-b.bin")]
    original_names = ["", "camera original name.jpg", "surplus.pdf"]
    event = _event(
        media_urls=urls, media_types=["image/jpeg"], media_text_inlined=[False],
        media_filenames=original_names,
    )

    assert await _runner()._handle_message(event) is None
    assert received == [{
        "media_urls": urls,
        "media_types": ["image/jpeg", ""],
        "media_text_inlined": [False, None],
        "media_filenames": [None, "camera original name.jpg"],
    }]
    assert event.media_urls == urls
    assert event.media_types == ["image/jpeg"]
    assert event.media_text_inlined == [False]
    assert event.media_filenames == original_names


@pytest.mark.asyncio
async def test_narrow_legacy_callback_ignores_additive_attachment_fields():
    seen = []

    def legacy(session_key, platform, source, message_id, text):
        seen.append((session_key, platform, source, message_id, text))
        return {"action": "handled"}

    _register(legacy)
    event = _event(media_urls=["C:/synthetic/cache/item.bin"], media_types=["application/octet-stream"])

    assert await _runner()._handle_message(event) is None
    assert len(seen) == 1
    assert seen[0][1:] == (
        "whatsapp", event.source.to_dict(), "m1", "queue my report",
    )

@pytest.mark.asyncio
async def test_a_raising_consumer_fails_open_and_the_message_reaches_the_agent():
    def broken(**_kwargs):
        raise RuntimeError("plugin bug")

    _register(broken)
    runner = _runner()

    assert await runner._handle_message(_event()) == "agent-ok"
    # Every following message on the profile keeps flowing too.
    assert await runner._handle_message(_event("second")) == "agent-ok"
    assert runner._handle_message_with_agent.await_count == 2


@pytest.mark.asyncio
async def test_consumer_fires_in_the_routed_profile_scope_only(tmp_path):
    launch, routed = tmp_path / "launch", tmp_path / "profiles" / "beta"
    launch.mkdir()
    routed.mkdir(parents=True)
    fired = []

    with _profile_runtime_scope(launch):
        _register(lambda **_k: fired.append(("launch", get_hermes_home())))
    with _profile_runtime_scope(routed):
        _register(lambda **_k: fired.append(("beta", get_hermes_home())) or {"action": "handled"})

    runner = _runner(multiplex=True)
    runner._resolve_profile_home_for_source = lambda _source: routed
    # The receiving bot's handler binds ITS home (auth reads its .env); this chat routes to beta.
    with _profile_runtime_scope(launch):
        assert await runner._handle_message(_event(profile="beta")) is None
        assert get_hermes_home() == launch  # scope restored after the hook

    assert fired == [("beta", routed)]
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_under_an_authority_the_hook_fires_once_at_ingress_never_at_execution(monkeypatch):
    """A consumed message is never committed, and an admitted row's execution (first run or
    restart replay) never offers the same message to the plugin again."""
    import gateway.session_authorities as authorities
    import gateway.session_ingress as ingress

    fired = []
    _register(lambda **kwargs: fired.append(kwargs["text"]) or (
        {"action": "handled", "reply": "consumed"} if kwargs["text"] == "eat me" else None))
    runner = _runner()
    admitted = []

    async def _admit(_authority, event):
        admitted.append(event.text)
        return "admitted"

    monkeypatch.setattr(authorities, "active_authority", lambda _runner: SimpleNamespace(route_epoch=lambda _route: 0))
    monkeypatch.setattr(ingress, "admit_message", _admit)

    assert await runner._handle_message(_event("eat me")) == "consumed"
    assert await runner._handle_message(_event("keep me")) == "admitted"
    assert admitted == ["keep me"] and fired == ["eat me", "keep me"]

    token = ingress.executing_admission.set(True)
    try:
        assert await runner._handle_message(_event("keep me")) == "agent-ok"
    finally:
        ingress.executing_admission.reset(token)
    assert fired == ["eat me", "keep me"]


@pytest.mark.asyncio
async def test_webhook_deliveries_fire_the_hook_once_per_admission_never_on_a_provider_retry():
    """Webhook/relay traffic enters through ``webhook_ingress.admit_producer``, not
    ``_handle_message``: plugins still see each delivery once, a consumed one is never committed
    (its reply goes to the route's destination), and a provider retry of a committed delivery
    is not offered again."""
    import hashlib
    import hmac
    import json

    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from gateway.platforms.webhook import WebhookAdapter
    from tests.gateway.fixtures.webhook_route_authority import mount_authority

    fired = []
    _register(lambda **kwargs: fired.append(kwargs["message_id"]) or (
        {"action": "handled", "reply": "plugin took it"} if "eat" in kwargs["text"] else None))
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"secret": "owned-secret", "routes": {
        "fixture": {"prompt": "{text}"}}}))
    sent = []

    async def _sink(delivery, chat_id, content):
        sent.append(content)

    adapter._deliver_to = _sink
    app = web.Application()
    mount_authority(app, adapter)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)

    async def _post(client, text, delivery_id):
        body = json.dumps({"text": text}).encode()
        signature = "sha256=" + hmac.new(b"owned-secret", body, hashlib.sha256).hexdigest()
        return await client.post("/webhooks/fixture", data=body, headers={
            "X-GitHub-Delivery": delivery_id, "X-Hub-Signature-256": signature})

    async with TestClient(TestServer(app)) as client:
        db = adapter._message_handler.__self__.session_authority.db
        assert (await _post(client, "eat this", "d-eat")).status == 202
        assert sent == ["plugin took it"] and not db._read_all("SELECT 1 FROM session_admissions")
        assert (await _post(client, "keep this", "d-keep")).status == 202
        assert (await _post(client, "keep this", "d-keep")).status in (200, 202)
        rows = db._read_all("SELECT request_id FROM session_admissions")
    assert [row["request_id"] for row in rows] == ["d-keep"]
    assert fired == ["d-eat", "d-keep"]

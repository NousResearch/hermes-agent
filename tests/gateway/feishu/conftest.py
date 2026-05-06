"""Pytest fixtures + helpers for Feishu adapter contract tests.

These fixtures provide stable entry/exit points so contract tests can
exercise the SDK-backed FeishuAdapter without needing a live channel.
The helpers wrap the current internal wiring; tests only see the stable
surface.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@dataclass
class CapturedSend:
    """One outbound HTTP call to lark API, captured at the lark_oapi.Client boundary."""
    endpoint: str                    # e.g. "im.v1.message.create" or "im.v1.message.reply"
    body: Dict[str, Any]             # request body as parsed dict
    extra: Dict[str, Any] = field(default_factory=dict)  # query params, receive_id_type, etc.


@dataclass
class AdapterHarness:
    """Wraps a FeishuAdapter with capture lists + helpers."""
    adapter: Any                     # FeishuAdapter
    captured_inbound: List[Any]      # MessageEvent list (from message_handler)
    captured_sends: List[CapturedSend]
    bot_open_id: str = "ou_hermes_bot"
    bot_user_id: str = "u_hermes_bot"
    media_pipeline: Any = None
    media_pipeline_loop: Any = None


def _build_platform_config(extra_overrides: Optional[Dict[str, Any]] = None) -> Any:
    """Build a PlatformConfig with safe defaults for tests."""
    from gateway.config import PlatformConfig

    extra = {
        "app_id": "cli_test_app",
        "app_secret": "secret_test",
        "domain": "feishu",
        "connection_mode": "websocket",
        "encrypt_key": "",
        "verification_token": "",
        "bot_open_id": "ou_hermes_bot",
        "bot_user_id": "u_hermes_bot",
        "bot_name": "HermesBot",
        "group_policy": "open",
        "allowed_group_users": "",
        "admins": [],
        "group_rules": {},
        "default_group_policy": "",
        "dedup_cache_size": 256,
        "text_batch_delay_seconds": 0.0,
        "text_batch_split_delay_seconds": 0.0,
        "text_batch_max_messages": 1,
        "text_batch_max_chars": 4000,
        "media_batch_delay_seconds": 0.0,
        "webhook_host": "127.0.0.1",
        "webhook_port": 8765,
        "webhook_path": "/feishu/webhook",
    }
    if extra_overrides:
        extra.update(extra_overrides)

    return PlatformConfig(
        enabled=True,
        extra=extra,
    )


def _build_capturing_lark_client(captured_sends: List[CapturedSend]) -> Mock:
    """Mock lark_oapi.Client that records every API call into captured_sends.

    Both adapter and SDK layers ultimately call into ``lark_oapi.Client``
    for HTTP, so this single mock covers all outbound capture.
    """
    client = Mock(name="lark_client")

    def _capture(endpoint: str):
        def _execute(request):
            body = {}
            if hasattr(request, "request_body") and request.request_body is not None:
                rb = request.request_body
                for attr in ("receive_id", "msg_type", "content", "uuid", "file_type", "file_name", "duration"):
                    if hasattr(rb, attr):
                        body[attr] = getattr(rb, attr)
            extra = {}
            for attr in ("receive_id_type", "message_id", "user_id_type"):
                if hasattr(request, attr):
                    extra[attr] = getattr(request, attr)
            captured_sends.append(CapturedSend(endpoint=endpoint, body=body, extra=extra))
            if endpoint == "im.v1.file.create":
                data = SimpleNamespace(file_key=f"file_test_{len(captured_sends)}")
            elif endpoint == "im.v1.image.create":
                data = SimpleNamespace(image_key=f"img_test_{len(captured_sends)}")
            else:
                data = SimpleNamespace(message_id=f"om_test_{len(captured_sends)}")
            return SimpleNamespace(success=lambda: True, code=0, msg="", data=data)
        return _execute

    # Wire the most common paths; tests mock more specifically when needed.
    # Production code calls `client.im.v1.message.create(request)` directly
    # (NOT `.create.execute(request)`), so we must wire side_effect on the
    # callable itself. We also set `.execute.side_effect` for any code path
    # that uses the chained-builder form.
    def _wire(path: str, endpoint: str) -> None:
        # Resolve nested attribute path like "im.v1.message.create" on client.
        node = client
        parts = path.split(".")
        parent = None
        for part in parts:
            parent = node
            node = getattr(node, part)
        capture_fn = _capture(endpoint)
        async def async_capture(request):
            return capture_fn(request)
        node.side_effect = capture_fn
        node.acreate = AsyncMock(side_effect=async_capture)
        node.execute = Mock(side_effect=capture_fn)
        if parts[-1] == "create" and parent is not None:
            parent.acreate = AsyncMock(side_effect=async_capture)

    _wire("im.v1.message.create", "im.v1.message.create")
    _wire("im.v1.message.reply", "im.v1.message.reply")
    _wire("im.v1.message.patch", "im.v1.message.patch")
    _wire("im.v1.message.delete", "im.v1.message.delete")
    _wire("im.v1.image.create", "im.v1.image.create")
    _wire("im.v1.file.create", "im.v1.file.create")
    client.im.v1.chat.get.execute = Mock(return_value=SimpleNamespace(
        success=lambda: True, code=0, msg="",
        data=SimpleNamespace(name="Test Chat", chat_mode="group"),
    ))

    def _fake_resource_request(req, *args, **kwargs):
        return SimpleNamespace(
            raw=SimpleNamespace(content=b"\xff\xd8\xff\xe0fake_jpeg"),  # JPEG magic bytes
            headers={"Content-Type": "image/jpeg", "Content-Disposition": 'attachment; filename="img.jpg"'},
        )

    client.request = Mock(side_effect=_fake_resource_request)
    return client


@pytest.fixture
def adapter_harness(tmp_path) -> AdapterHarness:
    """Create a FeishuAdapter wired for contract testing.

    - HTTP layer mocked via _build_capturing_lark_client
    - Bot identity pre-hydrated (skipping the /bot/v3/info probe)
    - Persistent dedup file redirected to tmp_path (test isolation)
    - Message handler captures inbound MessageEvent objects
    """
    from gateway.platforms.feishu import FeishuAdapter

    config = _build_platform_config()
    adapter = FeishuAdapter(config)

    # Redirect dedup persistence to tmp_path. _dedup_state_path is set in
    # FeishuAdapter.__init__ to ~/.hermes/...; we override after construction
    # since there's no config knob for it.
    adapter._dedup_state_path = tmp_path / "feishu_seen_message_ids.json"

    # The harness skips ``connect()``, so wire JsonFileDedupStore here
    # directly. ``dispatch_inbound_event`` drives the same dedup path the
    # SDK ``Deduper.check_and_mark`` would in production.
    from gateway.platforms.feishu import JsonFileDedupStore
    adapter._dedup_store = JsonFileDedupStore(
        path=adapter._dedup_state_path,
        max_entries=256,
        account_id=adapter._app_id,
    )

    captured_inbound: List[Any] = []
    captured_sends: List[CapturedSend] = []

    async def capture_handler(event):
        captured_inbound.append(event)

    adapter.set_message_handler(capture_handler)
    adapter._client = _build_capturing_lark_client(captured_sends)

    # Pre-hydrate bot identity so policy gate / self-sent filter work without
    # going through the /bot/v3/info probe.
    adapter._bot_open_id = "ou_hermes_bot"
    adapter._bot_user_id = "u_hermes_bot"
    adapter._bot_name = "HermesBot"

    # NOTE: PlatformConfig.extra["group_policy"] is currently ignored — the
    # adapter reads FEISHU_GROUP_POLICY env var directly (defaults to
    # "allowlist"). For tests we want "open" so group fixtures can flow
    # through without env var setup. Override the runtime attribute.
    adapter._group_policy = "open"
    adapter._default_group_policy = "open"
    # Force the admission gate to "all" so contract tests that exercise
    # peer-bot senders (e.g. ``test_other_app_sender_is_not_filtered``)
    # flow through. Tests that target ``FEISHU_ALLOW_BOTS`` semantics
    # explicitly override this attribute.
    adapter._allow_bots = "all"

    # Mock _channel.send so send() works without a live SDK connection.
    # send() passes bare strings to channel.send; the mock renders them via
    # SDK native mode to produce the same wire body production would. Dict
    # inputs (post/text/image/etc) pass through directly without re-rendering.
    mock_channel = MagicMock(name="feishu_channel")
    mock_channel.client = adapter._client

    async def _mock_channel_send(chat_id: str, message, opts=None):
        """Pass-through capture mock for FeishuChannel.send.

        Bare strings render via SDK native mode to match the production
        wire body. Dict inputs pass through as-is.
        """
        from lark_oapi.channel.outbound.routing import infer_receive_id_type
        from lark_oapi.channel.outbound.markdown import markdown_to_post_ast
        receive_id_type = (opts or {}).get("receive_id_type") or infer_receive_id_type(chat_id)

        if isinstance(message, str):
            # send() path: use SDK native mode to build the wire body.
            post_ast = markdown_to_post_ast(message, tag_md_mode='native')
            msg_type = "post"
            content_str = json.dumps(post_ast, ensure_ascii=False)
        elif isinstance(message, dict):
            # Future media tasks may still pass dict shapes
            if "post" in message:
                msg_type = "post"
                content_str = json.dumps(message["post"], ensure_ascii=False)
            elif "text" in message:
                msg_type = "text"
                content_str = json.dumps({"text": message["text"]}, ensure_ascii=False)
            elif "markdown" in message:
                post_ast = markdown_to_post_ast(message["markdown"], tag_md_mode='native')
                msg_type = "post"
                content_str = json.dumps(post_ast, ensure_ascii=False)
            elif "image" in message:
                if "caption" in message:
                    # Image + caption: mock SDK _materialize — post with img tag appended.
                    cap_ast = markdown_to_post_ast(message["caption"], tag_md_mode='native')
                    rows = cap_ast.get("zh_cn", {}).get("content", [])
                    if rows:
                        rows[-1].append({"tag": "img", "image_key": "img_test_mock"})
                    else:
                        rows = [[{"tag": "img", "image_key": "img_test_mock"}]]
                    msg_type = "post"
                    content_str = json.dumps({"zh_cn": {"title": "", "content": rows}}, ensure_ascii=False)
                else:
                    msg_type = "image"
                    content_str = json.dumps({"image_key": "img_test_mock"}, ensure_ascii=False)
            elif "audio" in message:
                source = message["audio"].get("source") if isinstance(message["audio"], dict) else None
                file_key = source if isinstance(source, str) and source.startswith("file_") else "file_test_audio"
                if "caption" in message:
                    cap_ast = markdown_to_post_ast(message["caption"], tag_md_mode='native')
                    rows = cap_ast.get("zh_cn", {}).get("content", [])
                    if rows:
                        rows[-1].append({"tag": "audio", "file_key": file_key})
                    else:
                        rows = [[{"tag": "audio", "file_key": file_key}]]
                    msg_type = "post"
                    content_str = json.dumps({"zh_cn": {"title": "", "content": rows}}, ensure_ascii=False)
                else:
                    msg_type = "audio"
                    content_str = json.dumps({"file_key": file_key}, ensure_ascii=False)
            elif "video" in message:
                source = message["video"].get("source") if isinstance(message["video"], dict) else None
                file_key = source if isinstance(source, str) and source.startswith("file_") else "file_test_video"
                if "caption" in message:
                    cap_ast = markdown_to_post_ast(message["caption"], tag_md_mode='native')
                    rows = cap_ast.get("zh_cn", {}).get("content", [])
                    if rows:
                        rows[-1].append({"tag": "video", "file_key": file_key})
                    else:
                        rows = [[{"tag": "video", "file_key": file_key}]]
                    msg_type = "post"
                    content_str = json.dumps({"zh_cn": {"title": "", "content": rows}}, ensure_ascii=False)
                else:
                    msg_type = "video"
                    content_str = json.dumps({"file_key": file_key}, ensure_ascii=False)
            elif "file" in message:
                if "caption" in message:
                    cap_ast = markdown_to_post_ast(message["caption"], tag_md_mode='native')
                    rows = cap_ast.get("zh_cn", {}).get("content", [])
                    if rows:
                        rows[-1].append({"tag": "file", "file_key": "file_test_doc"})
                    else:
                        rows = [[{"tag": "file", "file_key": "file_test_doc"}]]
                    msg_type = "post"
                    content_str = json.dumps({"zh_cn": {"title": "", "content": rows}}, ensure_ascii=False)
                else:
                    msg_type = "file"
                    content_str = json.dumps({"file_key": "file_test_doc"}, ensure_ascii=False)
            elif "card" in message:
                msg_type = "interactive"
                content_str = json.dumps(message["card"], ensure_ascii=False)
            else:
                raise AssertionError(f"Mock channel.send: unrecognized dict input keys: {list(message.keys())}")
        else:
            raise AssertionError(f"Mock channel.send: unsupported input type {type(message).__name__}")

        captured_sends.append(CapturedSend(
            endpoint="im.v1.message.create",
            body={
                "receive_id": chat_id,
                "msg_type": msg_type,
                "content": content_str,
            },
            extra={"receive_id_type": receive_id_type},
        ))
        return SimpleNamespace(
            success=True,
            message_id=f"om_test_{len(captured_sends)}",
            error=None,
        )

    mock_channel.send = _mock_channel_send

    async def _mock_channel_edit_message(message_id: str, message):
        """Mirror what SDK does: coerce to text/post, capture wire body."""
        from lark_oapi.channel.outbound.markdown import markdown_to_post_ast
        if isinstance(message, str):
            post_ast = markdown_to_post_ast(message, tag_md_mode='native')
            msg_type = "post"
            content = json.dumps(post_ast, ensure_ascii=False)
        elif isinstance(message, dict):
            if "text" in message:
                msg_type = "text"
                content = json.dumps({"text": message["text"]}, ensure_ascii=False)
            elif "post" in message:
                msg_type = "post"
                content = json.dumps(message["post"], ensure_ascii=False)
            elif "markdown" in message:
                post_ast = markdown_to_post_ast(message["markdown"], tag_md_mode='native')
                msg_type = "post"
                content = json.dumps(post_ast, ensure_ascii=False)
            else:
                raise AssertionError(f"edit_message mock: unsupported dict {list(message.keys())}")
        else:
            raise AssertionError(f"edit_message mock: type {type(message).__name__}")

        captured_sends.append(CapturedSend(
            endpoint="im.v1.message.patch",
            body={"message_id": message_id, "msg_type": msg_type, "content": content},
            extra={},
        ))
        return SimpleNamespace(success=True, message_id=message_id, error=None)

    mock_channel.edit_message = _mock_channel_edit_message

    async def _mock_channel_update_card(message_id: str, card: Dict[str, Any]):
        captured_sends.append(CapturedSend(
            endpoint="im.v1.message.card.update",
            body={"message_id": message_id, "card": card},
            extra={},
        ))
        return SimpleNamespace(success=True, message_id=message_id, error=None)

    mock_channel.update_card = _mock_channel_update_card

    async def _mock_channel_add_reaction(message_id: str, emoji_type: str):
        captured_sends.append(CapturedSend(
            endpoint="im.v1.message.reaction.create",
            body={"message_id": message_id, "reaction_type": {"emoji_type": emoji_type}},
            extra={},
        ))
        # SDK SendResult shape; reaction_id under raw["data"]
        return SimpleNamespace(
            success=True,
            message_id=message_id,
            error=None,
            raw={"code": 0, "msg": "ok", "data": {"reaction_id": f"reaction_{len(captured_sends)}"}},
        )

    async def _mock_channel_remove_reaction(message_id: str, reaction_id: str):
        captured_sends.append(CapturedSend(
            endpoint="im.v1.message.reaction.delete",
            body={"message_id": message_id, "reaction_id": reaction_id},
            extra={},
        ))
        return SimpleNamespace(success=True, message_id=message_id, error=None, raw=None)

    mock_channel.add_reaction = _mock_channel_add_reaction
    mock_channel.remove_reaction = _mock_channel_remove_reaction

    async def _mock_channel_get_chat_info(chat_id: str):
        # Infer chat_type from chat_id prefix so _map_chat_type produces the correct
        # Hermes "type" ("dm" for p2p chats, "group" for group chats).
        # Mirrors what the old client.im.v1.chat.get mock returned: the old mock had
        # chat_mode="group" but no chat_type, so getattr(..., "chat_type", "") was ""
        # which _map_chat_type maps to "dm". For group chats we want "group".
        inferred_chat_type = "p2p" if str(chat_id).startswith("p2p") else "group"
        return SimpleNamespace(chat_id=chat_id, name="Test Chat", chat_type=inferred_chat_type)

    mock_channel.get_chat_info = _mock_channel_get_chat_info

    async def _mock_channel_download_resource_to_file(
        file_key, *, resource_type, message_id, dest_dir, file_name=None,
    ):
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        suffixes = {
            "image": ".jpg",
            "audio": ".opus",
            "video": ".mp4",
            "file": ".bin",
        }
        name = str(file_name or file_key or f"resource_{message_id}")
        if "/" in name:
            name = name.replace("/", "_")
        if not Path(name).suffix:
            name += suffixes.get(resource_type, ".bin")
        path = dest / name
        path.write_bytes(b"mock feishu resource")
        return path

    mock_channel.download_resource_to_file = _mock_channel_download_resource_to_file

    async def _mock_channel_fetch_message(message_id):
        return SimpleNamespace(
            message_id=message_id,
            sender=SimpleNamespace(open_id="ou_hermes_bot", user_id="u_hermes_bot"),
        )

    mock_channel.fetch_message = _mock_channel_fetch_message

    # `to_message_event` reads bot identity from channel.bot_identity
    mock_channel.bot_identity = SimpleNamespace(
        open_id="ou_hermes_bot", user_id="u_hermes_bot", name="HermesBot",
    )

    adapter._channel = mock_channel

    return AdapterHarness(
        adapter=adapter,
        captured_inbound=captured_inbound,
        captured_sends=captured_sends,
    )


def load_fixture(category: str, name: str) -> Dict[str, Any]:
    """Load a JSON fixture from tests/gateway/feishu/fixtures/<category>/<name>."""
    path = FIXTURES_DIR / category / name
    return json.loads(path.read_text())


def load_text_fixture(category: str, name: str) -> str:
    """Load a text fixture (e.g. *.input.txt)."""
    path = FIXTURES_DIR / category / name
    return path.read_text()


async def dispatch_inbound_event(
    harness: AdapterHarness,
    raw_event: Dict[str, Any],
    *,
    drain: bool = True,
) -> None:
    """Feed a raw Feishu event into the adapter, mimicking what the SDK
    dispatcher delivers.

    Synthesizes an SDK :class:`InboundMessage` directly from the legacy
    event payload and calls ``adapter._on_sdk_message(msg)`` — the same
    entry point SDK ``channel.on("message", ...)`` registers in production.

    The SDK pipeline applies dedup / stale / policy / mention / lock /
    batch / queue filtering before ``_on_sdk_message`` fires in production.
    Some contract tests assert policy-gate behavior (admin bypass /
    blocklist / admin_only / self-sent filter / require-mention) — to keep
    those tests' intent visible while still exercising the SDK-backed
    handler, the harness re-implements a minimal mock policy gate below.
    """
    adapter = harness.adapter

    msg = _build_sdk_inbound_message(adapter, raw_event)
    if msg is None:
        # Filtered by mock policy gate (self-sent / blocked / no-mention).
        return

    # SDK owns dedup in production via Deduper.check_and_mark backed by
    # the injected JsonFileDedupStore. For the mock harness we drive the
    # same JsonFileDedupStore directly so the persistence contract test
    # exercises the cross-restart path. If no store is wired (most
    # adapter harnesses skip ``connect()``), fall through and let the
    # test handler collect the dispatch.
    store = getattr(adapter, "_dedup_store", None)
    if store is not None:
        from lark_oapi.channel.normalize.dedup import (
            make_event_key,
            make_message_key,
        )
        account_id = getattr(adapter, "_app_id", "") or ""
        event_key = make_event_key(account_id, msg.id)
        msg_key = make_message_key(account_id, msg.id)
        if store.seen(event_key) or store.seen(msg_key):
            return
        store.mark(event_key, ttl_seconds=int(60 * 60))
        store.mark(msg_key, ttl_seconds=int(60 * 60))

    if await _dispatch_via_mock_media_pipeline(harness, msg, drain=drain):
        return

    await adapter._on_sdk_message(msg)
    if drain:
        await _drain_adapter_tasks()


async def _drain_adapter_tasks() -> None:
    # Yield enough scheduler ticks for media batching (when configured) and
    # any spawned ``_process_message_background`` tasks to drain.
    for _ in range(20):
        await asyncio.sleep(0.05)


def _get_mock_media_pipeline(harness: AdapterHarness) -> Any:
    from lark_oapi.channel.safety.media_pipeline import (
        MediaBatchConfig,
        MediaPipelineManager,
    )

    loop = asyncio.get_running_loop()
    settings = harness.adapter._settings
    if (
        harness.media_pipeline is not None
        and harness.media_pipeline_loop is loop
    ):
        return harness.media_pipeline

    harness.media_pipeline_loop = loop
    harness.media_pipeline = MediaPipelineManager(
        MediaBatchConfig(
            enabled=bool(settings.media_batch_delay_seconds > 0),
            delay_ms=int(settings.media_batch_delay_seconds * 1000),
            max_items=9,
        ),
        loop,
    )
    return harness.media_pipeline


async def _dispatch_via_mock_media_pipeline(
    harness: AdapterHarness,
    msg: Any,
    *,
    drain: bool,
) -> bool:
    pipeline = _get_mock_media_pipeline(harness)
    if not pipeline.enabled:
        return False

    if pipeline.is_compatible(msg):
        await pipeline.push(msg, harness.adapter._on_sdk_message)
        if drain:
            delay = max(harness.adapter._settings.media_batch_delay_seconds, 0.0)
            await asyncio.sleep(delay + 0.05)
            await _drain_adapter_tasks()
        return True

    await pipeline.flush_incompatible_for(msg)
    return False


def _build_sdk_inbound_message(adapter: Any, raw_event: Dict[str, Any]) -> Any:
    """Translate a legacy fixture event dict into an SDK ``InboundMessage``.

    Returns ``None`` when the harness's mock policy gate would have dropped
    the event (so it never reaches ``_on_sdk_message``). The mock gate
    re-implements enough of the SDK pipeline's policy filtering to keep
    each contract test's intent visible:

      * self-sent (``sender_type=='app'`` + bot open_id) → drop
      * group chat + group_policy=disabled / blocklist hit / admin_only
        without admin → drop
      * group chat with default ``allowlist`` policy and no @bot mention →
        drop
      * fixture-driven mention map / chat_type from the wire.
    """
    from lark_oapi.channel.types import (
        Conversation,
        Identity,
        InboundMessage,
        Mention,
        ReplyRef,
        ResourceDescriptor,
        TextContent,
        ImageContent,
        FileContent,
        AudioContent,
        MediaContent,
        PostContent,
        InteractiveContent,
        ShareChatContent,
        MergeForwardContent,
        UnknownContent,
    )

    event = raw_event.get("event", raw_event)
    sender = event.get("sender", {})
    sender_id_obj = sender.get("sender_id", {}) if isinstance(sender, dict) else {}
    sender_type = sender.get("sender_type", "user") if isinstance(sender, dict) else "user"
    message = event.get("message", {}) if isinstance(event, dict) else {}

    sender_open_id = sender_id_obj.get("open_id", "") or ""
    sender_user_id = sender_id_obj.get("user_id", "") or ""
    sender_union_id = sender_id_obj.get("union_id", "") or ""

    chat_id = str(message.get("chat_id", "") or "")
    chat_type_str = str(message.get("chat_type", "") or "")
    if chat_type_str not in ("p2p", "group", "topic"):
        chat_type_str = "p2p" if chat_id.startswith("p2p") else "group"

    message_type_str = str(message.get("message_type", "text") or "text")
    raw_content = message.get("content", "") or ""

    # Mock SDK policy gate
    bot_open_id = adapter._bot_open_id or "ou_hermes_bot"
    mentions_raw = message.get("mentions") or []

    # --- self-sent filter ---
    if sender_type == "app" and sender_open_id == bot_open_id:
        return None

    # --- group policy gate (mock) ---
    if chat_type_str == "group":
        from gateway.platforms.feishu import FeishuGroupRule
        admins = getattr(adapter._settings, "admins", frozenset()) or frozenset()
        is_admin = sender_open_id in admins
        rule = (getattr(adapter._settings, "group_rules", {}) or {}).get(chat_id)
        # Per-chat rule
        if isinstance(rule, FeishuGroupRule):
            if rule.policy == "disabled" and not is_admin:
                return None
            if rule.policy == "admin_only" and not is_admin:
                return None
            if rule.policy == "blacklist" and sender_open_id in (rule.blacklist or set()):
                return None
            if rule.policy == "allowlist" and not is_admin:
                allow = rule.allowlist or set()
                if sender_open_id not in allow:
                    return None
        else:
            # Use _default_group_policy / _group_policy from the harness.
            group_policy = (
                getattr(adapter, "_default_group_policy", "")
                or getattr(adapter, "_group_policy", "")
                or "open"
            )
            if group_policy == "disabled" and not is_admin:
                return None
            if group_policy == "allowlist" and not is_admin:
                # require @bot mention to opt in
                mentions_bot = any(
                    (m.get("id") or {}).get("open_id", "") == bot_open_id
                    for m in mentions_raw
                )
                if not mentions_bot:
                    return None

    # --- build SDK Mention list ---
    mentions: list = []
    for m in mentions_raw:
        m_open_id = (m.get("id") or {}).get("open_id", "") or ""
        mentions.append(Mention(
            key=m.get("key", "") or "",
            open_id=m_open_id,
            user_id=(m.get("id") or {}).get("user_id", "") or None,
            name=m.get("name", "") or None,
            is_bot=(m_open_id == bot_open_id),
            union_id=(m.get("id") or {}).get("union_id", "") or None,
            tenant_key=m.get("tenant_key", "") or None,
        ))

    # --- map message_type → MessageContent variant ---
    content: Any
    content_text: str = ""
    resources: list = []
    try:
        parsed = json.loads(raw_content) if isinstance(raw_content, str) and raw_content else {}
    except Exception:
        parsed = {}

    if message_type_str == "text":
        text_val = str(parsed.get("text", "") or "")
        # Resolve @_user_N placeholders against mentions to mirror SDK pipeline
        rendered = text_val
        for m in mentions:
            if m.key and m.name:
                rendered = rendered.replace(m.key, f"@{m.name}")
        content = TextContent(text=text_val, raw=parsed if isinstance(parsed, dict) else {})
        content_text = rendered
    elif message_type_str == "image":
        image_key = str(parsed.get("image_key", "") or "")
        content = ImageContent(image_key=image_key, raw=parsed if isinstance(parsed, dict) else {})
        if image_key:
            resources.append(ResourceDescriptor(type="image", file_key=image_key))
        # Locked-down legacy projection: empty text for media-only messages.
    elif message_type_str == "file":
        file_key = str(parsed.get("file_key", "") or "")
        file_name = str(parsed.get("file_name", "") or "") or None
        content = FileContent(file_key=file_key, file_name=file_name, raw=parsed)
        if file_key:
            resources.append(ResourceDescriptor(
                type="file", file_key=file_key, file_name=file_name,
            ))
    elif message_type_str == "audio":
        file_key = str(parsed.get("file_key", "") or "")
        content = AudioContent(file_key=file_key, raw=parsed)
        if file_key:
            resources.append(ResourceDescriptor(type="audio", file_key=file_key))
    elif message_type_str == "media":
        file_key = str(parsed.get("file_key", "") or "")
        image_key = str(parsed.get("image_key", "") or "") or None
        content = MediaContent(file_key=file_key, image_key=image_key, raw=parsed)
        if file_key:
            resources.append(ResourceDescriptor(
                type="video", file_key=file_key, cover_image_key=image_key,
            ))
    elif message_type_str == "post":
        post_dict = parsed if isinstance(parsed, dict) else {}
        content = PostContent(post=post_dict, raw=post_dict)
        # Walk both shapes: top-level {title, content} OR {locale: {title, content}}.
        flat_parts: list = []
        title_str = post_dict.get("title") if isinstance(post_dict.get("title"), str) else ""
        if title_str:
            flat_parts.append(title_str)
        rows_iter = []
        if isinstance(post_dict.get("content"), list):
            rows_iter = post_dict.get("content") or []
        else:
            for locale in post_dict.values():
                if isinstance(locale, dict) and isinstance(locale.get("content"), list):
                    if not title_str and isinstance(locale.get("title"), str):
                        flat_parts.append(locale.get("title"))
                    rows_iter = locale.get("content") or []
                    break
        row_strs: list = []
        for row in rows_iter or []:
            row_buf: list = []
            for item in row or []:
                if not isinstance(item, dict):
                    continue
                tag = item.get("tag")
                if tag == "text":
                    row_buf.append(str(item.get("text", "")))
                elif tag == "img":
                    ik = item.get("image_key", "")
                    if ik:
                        resources.append(ResourceDescriptor(type="image", file_key=ik))
                        row_buf.append("[Image]")
                elif tag == "a":
                    row_buf.append(str(item.get("text", "") or item.get("href", "")))
                elif tag == "at":
                    row_buf.append("@" + str(item.get("user_name", item.get("user_id", ""))))
            row_strs.append("".join(row_buf))
        if row_strs:
            flat_parts.append("\n".join(row_strs))
        content_text = "\n".join(p for p in flat_parts if p)
    elif message_type_str == "interactive":
        card = parsed if isinstance(parsed, dict) else {}
        content = InteractiveContent(card=card, raw=card)
        # Best-effort flat-text rendering (locked-down golden expects header
        # title + element text + buttons listed under "Actions:")
        flat_parts = []
        header = card.get("header", {}) if isinstance(card, dict) else {}
        title = (header.get("title") or {}).get("content") if isinstance(header, dict) else ""
        if title:
            flat_parts.append(str(title))
        button_labels = []
        for el in (card.get("elements") or []) if isinstance(card, dict) else []:
            if not isinstance(el, dict):
                continue
            tag = el.get("tag")
            if tag == "div":
                txt = (el.get("text") or {}).get("content")
                if txt:
                    flat_parts.append(str(txt))
            elif tag == "action":
                for a in el.get("actions") or []:
                    btn = (a.get("text") or {}).get("content") if isinstance(a, dict) else ""
                    if btn:
                        button_labels.append(str(btn))
                        flat_parts.append(str(btn))
        if button_labels:
            flat_parts.append("Actions: " + ", ".join(button_labels))
        content_text = "\n".join(flat_parts)
    elif message_type_str == "share_chat":
        chat_share = parsed if isinstance(parsed, dict) else {}
        content = ShareChatContent(chat_id=str(chat_share.get("chat_id", "") or ""), raw=chat_share)
        # Locked-down legacy projection: "Shared chat: <name>\nChat ID: <id>"
        cn = chat_share.get("chat_name") or ""
        cid = chat_share.get("chat_id") or ""
        if cn or cid:
            content_text = f"Shared chat: {cn}\nChat ID: {cid}"
    elif message_type_str == "merge_forward":
        mf = parsed if isinstance(parsed, dict) else {}
        content = MergeForwardContent(raw=mf)
        # Locked-down legacy projection: just the title
        title = mf.get("title") or ""
        if title:
            content_text = str(title)
    else:
        content = UnknownContent(message_type=message_type_str, raw=parsed if isinstance(parsed, dict) else {})

    parent_id = message.get("parent_id") or ""
    root_id = message.get("root_id") or ""
    reply: Optional[Any] = None
    if parent_id and parent_id != root_id:
        reply = ReplyRef(message_id=parent_id)

    create_time_raw = message.get("create_time") or "0"
    try:
        create_time_int = int(create_time_raw)
    except Exception:
        create_time_int = 0

    return InboundMessage(
        id=str(message.get("message_id", "om_unknown")),
        create_time=create_time_int,
        conversation=Conversation(
            chat_id=chat_id,
            chat_type=chat_type_str,
            thread_id=message.get("thread_id"),
        ),
        sender=Identity(
            open_id=sender_open_id,
            user_id=sender_user_id or None,
            union_id=sender_union_id or None,
            display_name=None,
            is_bot=(sender_type == "app"),
        ),
        mentions=mentions,
        mentioned_all=False,
        reply=reply,
        content=content,
        raw=message if isinstance(message, dict) else {},
        content_text=content_text,
        resources=resources,
        raw_content_type=message_type_str,
    )

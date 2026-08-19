"""Tests for the OneBot 11 platform adapter (NapCat / Lagrange / LLOneBot).

Covers reply splitting at sentence boundaries, CQ-code parsing, mention
gating, DM/group policies, outbound segment-array payloads, text-image
rendering, and a live reverse-WS round trip against a fake NapCat client.
"""

import asyncio
import base64
import json
from unittest.mock import patch

import pytest

from gateway.config import Platform, PlatformConfig
from plugins.platforms.onebot.adapter import (
    MAX_MESSAGE_LENGTH,
    OneBotAdapter,
    render_text_image,
)
from plugins.platforms.onebot.onebot_utils import (
    DEFAULT_SPLIT_LENGTH,
    _split_reply,
)


# ---------------------------------------------------------------------------
# _split_reply
# ---------------------------------------------------------------------------


def test_split_reply_short_message_unchanged() -> None:
    assert _split_reply("短消息。", 100) == ["短消息。"]


def test_split_reply_breaks_at_sentence_boundaries() -> None:
    text = "第一句完整的话。第二句完整的话！第三句问号？" * 8
    parts = _split_reply(text, 100)
    assert len(parts) > 1
    for part in parts:
        assert 0 < len(part) <= 100
        # Every non-final chunk must end on a sentence boundary.
        assert part[-1] in "。！？!?；;\n"


def test_split_reply_hard_cut_without_boundaries() -> None:
    text = "x" * 250
    parts = _split_reply(text, 100)
    assert [len(p) for p in parts] == [100, 100, 50]


def test_split_reply_respects_explicit_newlines() -> None:
    # Newlines are sentence boundaries: a >limit text full of newlines
    # breaks at the newlines (each line is short).
    text = ("行" * 30 + "\n") * 5  # 155 chars, newline every 31 chars
    parts = _split_reply(text, 100)
    assert len(parts) >= 2
    for part in parts[:-1]:
        assert part.endswith("\n")
    assert all(len(p) <= 100 for p in parts)


# ---------------------------------------------------------------------------
# Text-image rendering
# ---------------------------------------------------------------------------

_DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


@pytest.mark.skipif(
    not __import__("os").path.exists(_DEJAVU),
    reason="DejaVu font not available in this environment",
)
def test_render_text_image_produces_png(monkeypatch) -> None:
    from PIL import Image
    import io

    import plugins.platforms.onebot.onebot_utils as ou

    monkeypatch.setattr(ou, "_TEXT_IMAGE_FALLBACK_FONTS", [_DEJAVU])
    png = render_text_image("Hello world! " * 20)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    img = Image.open(io.BytesIO(png))
    assert img.size[0] == ou._TEXT_IMAGE_WIDTH
    assert img.size[1] > 0


def test_render_text_image_preserves_newlines(monkeypatch) -> None:
    import io

    from PIL import Image

    import plugins.platforms.onebot.onebot_utils as ou

    if not __import__("os").path.exists(_DEJAVU):
        pytest.skip("DejaVu font not available")
    monkeypatch.setattr(ou, "_TEXT_IMAGE_FALLBACK_FONTS", [_DEJAVU])
    single = render_text_image("line1\nline2\nline3")
    joined = render_text_image("line1line2line3")
    img1 = Image.open(io.BytesIO(single))
    img2 = Image.open(io.BytesIO(joined))
    # Three explicit lines need more height than one joined paragraph.
    assert img1.size[1] > img2.size[1]


# ---------------------------------------------------------------------------
# Adapter behavior
# ---------------------------------------------------------------------------


def _make_adapter(**extra) -> OneBotAdapter:
    return OneBotAdapter(PlatformConfig(enabled=True, extra=extra or {}))


def test_adapter_has_max_message_length() -> None:
    assert MAX_MESSAGE_LENGTH == 4000
    assert OneBotAdapter.MAX_MESSAGE_LENGTH == MAX_MESSAGE_LENGTH


def test_cq_parse_at_and_face() -> None:
    adapter = _make_adapter()
    raw = "[CQ:at,qq=12345] 你好 [CQ:face,id=0]"
    text, media, media_types = asyncio.run(adapter._parse_content(raw))
    assert text == "@12345 你好 😊"
    assert media == []
    assert media_types == []


def test_cq_parse_at_all_and_reply() -> None:
    adapter = _make_adapter()
    raw = "[CQ:reply,id=99][CQ:at,qq=all] 注意"
    text, _, _ = asyncio.run(adapter._parse_content(raw))
    assert text == "@全体成员 注意"


def test_cq_parse_image_no_url_falls_back() -> None:
    adapter = _make_adapter()
    raw = "看图 [CQ:image,file=abc.jpg]"
    text, media, media_types = asyncio.run(adapter._parse_content(raw))
    assert text == "看图 [图片]"
    assert media == []
    assert media_types == []


def test_cq_parse_record_no_url_falls_back() -> None:
    adapter = _make_adapter()
    raw = "听这个 [CQ:record,file=abc.silk]"
    text, media, media_types = asyncio.run(adapter._parse_content(raw))
    assert text == "听这个 [语音]"
    assert media == []
    assert media_types == []


def test_shrink_image_downscales_large_image(tmp_path) -> None:
    from PIL import Image

    adapter = _make_adapter(image_max_size=1536)
    big = tmp_path / "big.jpg"
    Image.new("RGB", (3000, 2000), "white").save(big)
    out = adapter._shrink_image(big)
    assert out is not None
    with Image.open(out) as img:
        assert max(img.size) <= 1536
    # Aspect ratio preserved.
    assert img.size == (1536, 1024)


def test_shrink_image_skips_small_image(tmp_path) -> None:
    from PIL import Image

    adapter = _make_adapter(image_max_size=1536)
    small = tmp_path / "small.png"
    Image.new("RGB", (800, 600), "white").save(small)
    assert adapter._shrink_image(small) is None


def test_shrink_image_disabled_with_zero(tmp_path) -> None:
    from PIL import Image

    adapter = _make_adapter(image_max_size=0)
    big = tmp_path / "big.png"
    Image.new("RGB", (3000, 2000), "white").save(big)
    assert adapter._shrink_image(big) is None


def test_is_mentioned() -> None:
    adapter = _make_adapter()
    adapter._self_id = "123456789"
    assert adapter._is_mentioned("[CQ:at,qq=123456789] 嗨")
    assert adapter._is_mentioned("带回复 [CQ:reply,id=5]")
    assert not adapter._is_mentioned("没 @ 的消息")


def test_is_mentioned_fails_closed_without_self_id() -> None:
    adapter = _make_adapter()
    adapter._self_id = None
    assert not adapter._is_mentioned("随便说点什么")


def test_dm_policy_allowlist() -> None:
    adapter = _make_adapter(dm_policy="allowlist", allow_from=["10001"])
    assert adapter._dm_allowed("10001")
    assert not adapter._dm_allowed("99999")


def test_dm_policy_disabled() -> None:
    adapter = _make_adapter(dm_policy="disabled")
    assert not adapter._dm_allowed("10001")


def test_group_policy_allowlist() -> None:
    adapter = _make_adapter(group_policy="allowlist", group_allow_from=["888888"])
    assert adapter._group_allowed("888888")
    assert not adapter._group_allowed("777777")


# ---------------------------------------------------------------------------
# Markdown stripping
# ---------------------------------------------------------------------------


def test_strip_markdown_inline() -> None:
    from plugins.platforms.onebot.onebot_utils import strip_markdown

    assert strip_markdown("**加粗** 和 *斜体* 和 `代码`") == "加粗 和 斜体 和 代码"
    assert strip_markdown("[链接](https://example.com)") == "链接（https://example.com）"
    assert strip_markdown("~~删除线~~") == "删除线"


def test_strip_markdown_blocks() -> None:
    from plugins.platforms.onebot.onebot_utils import strip_markdown

    text = "## 标题\n\n- 项目一\n- 项目二\n\n1. 第一\n2. 第二\n\n> 引用"
    out = strip_markdown(text)
    assert "【标题】" in out
    assert "• 项目一" in out
    assert "1. 第一" in out
    assert "「引用」" in out


def test_strip_markdown_code_block() -> None:
    from plugins.platforms.onebot.onebot_utils import strip_markdown

    text = "```python\nprint('hi')\n```\n结尾"
    out = strip_markdown(text)
    assert "┌─[python]─" in out
    assert "│ print('hi')" in out
    assert "结尾" in out


def test_send_strips_markdown_before_delivery() -> None:
    adapter = _make_adapter(text_image_threshold=0)
    ws = _FakeWS(adapter)
    adapter._ws = ws
    result = asyncio.run(adapter.send("private:1", "**你好** `世界`"))
    assert result.success
    text = ws.sent[0]["params"]["message"][0]["data"]["text"]
    assert text == "你好 世界"


# ---------------------------------------------------------------------------
# Outbound send() — fake WebSocket with echo replies
# ---------------------------------------------------------------------------


class _FakeWS:
    def __init__(self, adapter: OneBotAdapter) -> None:
        self.adapter = adapter
        self.sent: list[dict] = []
        self._next_id = 1

    async def send_str(self, payload: str) -> None:
        data = json.loads(payload)
        self.sent.append(data)
        fut = self.adapter._pending_actions.get(data.get("echo"))
        if fut is not None and not fut.done():
            fut.set_result(
                {
                    "status": "ok",
                    "retcode": 0,
                    "echo": data.get("echo"),
                    "data": {"message_id": self._next_id},
                }
            )
            self._next_id += 1


def test_send_uses_segment_array_without_reply() -> None:
    adapter = _make_adapter()
    ws = _FakeWS(adapter)
    adapter._ws = ws
    result = asyncio.run(adapter.send("private:123456789", "你好"))
    assert result.success
    payload = ws.sent[0]
    assert payload["action"] == "send_msg"
    assert payload["params"]["user_id"] == 123456789
    assert payload["params"]["message"] == [
        {"type": "text", "data": {"text": "你好"}}
    ]
    # User asked for no reply-quoting: never emit a reply segment.
    assert all(seg["type"] != "reply" for seg in payload["params"]["message"])


def test_send_splits_long_text_into_multiple_messages() -> None:
    # Disable the text-image path so we exercise the chunking logic.
    adapter = _make_adapter(split_length=50, text_image_threshold=0)
    ws = _FakeWS(adapter)
    adapter._ws = ws
    long_text = "第一句。第二句。" * 20  # 160 chars, sentence boundaries
    result = asyncio.run(adapter.send("group:888888", long_text))
    assert result.success
    assert len(ws.sent) > 1
    for payload in ws.sent:
        assert payload["params"]["group_id"] == 888888
        segs = payload["params"]["message"]
        assert segs and segs[0]["type"] == "text"
        assert len(segs[0]["data"]["text"]) <= 50


def test_send_long_content_uses_text_image(monkeypatch) -> None:
    adapter = _make_adapter(text_image_threshold=50)
    ws = _FakeWS(adapter)
    adapter._ws = ws

    def fake_render(text: str, title: str = None) -> bytes:
        return b"\x89PNG\r\n\x1a\n" + b"0" * 64

    monkeypatch.setattr(
        "plugins.platforms.onebot.adapter.render_text_image", fake_render
    )
    result = asyncio.run(adapter.send("private:1", "很长" * 30))
    assert result.success
    assert len(ws.sent) == 1
    segs = ws.sent[0]["params"]["message"]
    assert segs[0]["type"] == "image"
    assert segs[0]["data"]["file"].startswith("base64://")


def test_send_attaches_media_to_final_chunk() -> None:
    # 80 chars: >50 (splits) but <150 (no text image).
    adapter = _make_adapter(split_length=50)
    ws = _FakeWS(adapter)
    adapter._ws = ws
    long_text = "第一句。第二句。" * 10
    result = asyncio.run(
        adapter.send(
            "private:1",
            long_text,
            metadata={"media_files": ["/nonexistent/img.png"]},
        )
    )
    # Missing local file is skipped gracefully; text still delivered.
    assert result.success
    assert len(ws.sent) > 1
    for payload in ws.sent:
        assert all(seg["type"] == "text" for seg in payload["params"]["message"])


def test_send_fails_fast_when_disconnected() -> None:
    adapter = _make_adapter()
    adapter._ws = None
    result = asyncio.run(adapter.send("private:1", "你好"))
    assert not result.success
    assert result.retryable


def test_send_typing_private_chat() -> None:
    adapter = _make_adapter()
    ws = _FakeWS(adapter)
    adapter._ws = ws
    asyncio.run(adapter.send_typing("private:123456789"))
    assert len(ws.sent) == 1
    payload = ws.sent[0]
    assert payload["action"] == "set_input_status"
    assert payload["params"] == {"user_id": "123456789", "event_type": 1}


def test_send_typing_group_chat_is_noop() -> None:
    adapter = _make_adapter()
    ws = _FakeWS(adapter)
    adapter._ws = ws
    asyncio.run(adapter.send_typing("group:888888"))
    assert ws.sent == []


def test_stop_typing_private_chat() -> None:
    adapter = _make_adapter()
    ws = _FakeWS(adapter)
    adapter._ws = ws
    asyncio.run(adapter.stop_typing("private:123456789"))
    assert len(ws.sent) == 1
    assert ws.sent[0]["params"] == {"user_id": "123456789", "event_type": 0}


# ---------------------------------------------------------------------------
# Reverse-WS round trip against a fake NapCat client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reverse_ws_round_trip(monkeypatch) -> None:
    import socket

    import aiohttp

    # The live gateway (if running) holds the per-mode platform lock; tests
    # must bypass it.
    monkeypatch.setattr(
        OneBotAdapter, "_acquire_platform_lock", lambda self, *a, **k: True
    )

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    adapter = _make_adapter(host="127.0.0.1", port=port, admin_users=[10001])
    assert await adapter.connect()
    try:
        received = []
        adapter._message_handler = lambda event: received.append(event) or _noop()

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(f"ws://127.0.0.1:{port}/ws") as ws:
                # Heartbeat meta event → learn self id.
                await ws.send_str(
                    json.dumps(
                        {
                            "post_type": "meta_event",
                            "meta_event_type": "heartbeat",
                            "self_id": 123456789,
                        }
                    )
                )
                await asyncio.sleep(0.05)
                assert adapter._self_id == "123456789"

                # Private message → dispatched as DM event.
                await ws.send_str(
                    json.dumps(
                        {
                            "post_type": "message",
                            "message_type": "private",
                            "user_id": 10001,
                            "self_id": 123456789,
                            "message_id": 111,
                            "raw_message": "你好[CQ:face,id=0]",
                            "sender": {"user_id": 10001, "nickname": "测试员"},
                        }
                    )
                )
                await asyncio.sleep(0.15)
                assert received, "private message should be dispatched"
                ev = received[-1]
                assert ev.text == "你好😊"
                assert ev.source.chat_id == "private:10001"
                assert ev.source.chat_type == "dm"

                # Group message without @ → ignored under require_mention.
                before = len(received)
                await ws.send_str(
                    json.dumps(
                        {
                            "post_type": "message",
                            "message_type": "group",
                            "group_id": 888888,
                            "user_id": 10002,
                            "self_id": 123456789,
                            "message_id": 222,
                            "raw_message": "没 @ 的消息",
                            "sender": {"user_id": 10002, "nickname": "群友"},
                        }
                    )
                )
                await asyncio.sleep(0.15)
                assert len(received) == before

                # Group message with @ → dispatched.
                await ws.send_str(
                    json.dumps(
                        {
                            "post_type": "message",
                            "message_type": "group",
                            "group_id": 888888,
                            "user_id": 10002,
                            "self_id": 123456789,
                            "message_id": 333,
                            "raw_message": "[CQ:at,qq=123456789] 在吗",
                            "sender": {
                                "user_id": 10002,
                                "nickname": "群友",
                                "card": "卡",
                            },
                        }
                    )
                )
                await asyncio.sleep(0.15)
                assert len(received) == before + 1
                ev = received[-1]
                assert ev.text.endswith("@123456789 在吗")
                assert ev.source.chat_id == "group:888888"
                assert ev.source.chat_type == "group"
                assert ev.source.user_name == "卡"  # group card preferred
    finally:
        await adapter.disconnect()


async def _noop() -> None:
    return None


# ---------------------------------------------------------------------------
# Reply quoting (get_msg -> text + media) and loop-message merge+retract
# ---------------------------------------------------------------------------


def test_quote_reply_fetches_original_text_and_image(monkeypatch) -> None:
    """引用消息时通过 get_msg 取回原文文本 + 图片（reply 段路径）。"""
    adapter = _make_adapter(admin_users=[123456789])
    ws = _FakeWS(adapter)
    adapter._ws = ws
    captured: list = []

    async def fake_handle_message(ev):
        captured.append(ev)

    adapter.handle_message = fake_handle_message  # type: ignore[method-assign]

    async def fake_get_msg(action, params, timeout=30.0):
        assert action == "get_msg"
        assert params["message_id"] == 12345
        return {
            "message": [
                {"type": "text", "data": {"text": "被引用的卡片内容"}},
                {
                    "type": "image",
                    "data": {"url": "https://fake.cdn/img.png", "file": "x.png"},
                },
            ]
        }

    monkeypatch.setattr(adapter, "_call_action", fake_get_msg)

    async def run():
        await adapter._process_message(
            {
                "message_type": "private",
                "user_id": 123456789,
                "message": [
                    {"type": "reply", "data": {"id": 12345}},
                    {"type": "text", "data": {"text": "你看看这个"}},
                ],
                "self_id": 123456789,
            }
        )

    asyncio.run(run())
    assert captured, "no event captured"
    ev = captured[0]
    # 文本拼了 [引用] 前缀 + 原消息文本
    assert "[引用]" in ev.text
    assert "被引用的卡片内容" in ev.text
    # 图片在文本里以 [图片] 占位（下载失败时降级保留占位，不阻塞消息）
    assert "[图片]" in ev.text


def test_loop_merge_buffers_interim_then_forwards_and_retracts(monkeypatch) -> None:
    """interim 消息缓冲，final 到达时合并转发 + 撤回（群聊）。"""
    adapter = _make_adapter()
    ws = _FakeWS(adapter)
    adapter._ws = ws
    adapter._self_id = "123456789"
    chat = "group:123456789"

    async def run():
        # 2 条 interim 中间评论
        await adapter.send(chat, "中间评论一", metadata={"interim": True})
        await adapter.send(chat, "中间评论二", metadata={"interim": True})
        assert len(adapter._loop_buffer.get(chat, [])) == 2
        # final 消息触发结算
        await adapter.send(chat, "最终回复内容", metadata={"notify": True})

    asyncio.run(run())
    actions = [p["action"] for p in ws.sent]
    assert actions.count("send_forward_msg") == 1
    assert actions.count("delete_msg") == 2
    fwd = next(p for p in ws.sent if p["action"] == "send_forward_msg")
    assert fwd["params"]["group_id"] == 123456789
    assert len(fwd["params"]["messages"]) == 2
    assert adapter._loop_buffer.get(chat) is None


def test_loop_merge_private_uses_send_private_forward_msg(monkeypatch) -> None:
    """私聊场景用 send_private_forward_msg。"""
    adapter = _make_adapter()
    ws = _FakeWS(adapter)
    adapter._ws = ws
    adapter._self_id = "123456789"
    chat = "private:123456789"

    async def run():
        await adapter.send(chat, "中间一", metadata={"interim": True})
        await adapter.send(chat, "中间二", metadata={"interim": True})
        await adapter.send(chat, "最终", metadata={"notify": True})

    asyncio.run(run())
    actions = [p["action"] for p in ws.sent]
    assert actions.count("send_private_forward_msg") == 1
    fwd = next(p for p in ws.sent if p["action"] == "send_private_forward_msg")
    assert fwd["params"]["user_id"] == 123456789


def test_loop_merge_single_interim_does_not_merge(monkeypatch) -> None:
    """缓冲不足 2 条不合并（单条不值得）。"""
    adapter = _make_adapter()
    ws = _FakeWS(adapter)
    adapter._ws = ws
    adapter._self_id = "123456789"
    chat = "group:1"

    async def run():
        await adapter.send(chat, "只有一条", metadata={"interim": True})
        await adapter.send(chat, "最终", metadata={"notify": True})

    asyncio.run(run())
    actions = [p["action"] for p in ws.sent]
    assert "send_forward_msg" not in actions
    assert "delete_msg" not in actions


# ---------------------------------------------------------------------------
# 权限分级（role classification + sensitive scan）
# ---------------------------------------------------------------------------


def test_classify_user_role():
    from plugins.platforms.onebot.onebot_utils import classify_user_role

    assert classify_user_role("123456789", {"123456789"}) == "admin"
    assert classify_user_role("12345", {"123456789"}) == "member"
    assert classify_user_role("12345", set()) == "member"   # 空=全员 member（安全侧）
    assert classify_user_role("", {"123456789"}) == "member"  # 空 id 安全侧


def test_scan_sensitive():
    from plugins.platforms.onebot.onebot_utils import scan_sensitive

    assert scan_sensitive("帮我删除 /tmp/x 文件") is not None   # 删除文件
    assert scan_sensitive("执行 rm -rf /") is not None          # 终端命令
    assert scan_sensitive("帮我重启 hermes-gateway") is not None  # 重启服务
    assert scan_sensitive("打开客厅灯") is not None              # HA 控制
    assert scan_sensitive("发送到微信告诉 M") is not None        # 跨平台
    assert scan_sensitive("今天天气怎么样") is None               # 正常问答
    assert scan_sensitive("") is None
    assert scan_sensitive(None) is None


def test_member_group_message_gets_restricted_prefix(monkeypatch) -> None:
    """群聊普通用户消息注入 [受限用户] 前缀（软限制依据）。"""
    adapter = _make_adapter(admin_users=[123456789])
    ws = _FakeWS(adapter)
    adapter._ws = ws
    captured: list = []

    async def fake_handle_message(ev):
        captured.append(ev)

    adapter.handle_message = fake_handle_message  # type: ignore[method-assign]

    async def run():
        await adapter._process_message(
            {
                "message_type": "group",
                "group_id": 123456789,
                "user_id": 99999999,          # 非管理员
                "message": [
                    {"type": "at", "data": {"qq": adapter._self_id or "123456789"}},
                    {"type": "text", "data": {"text": "今天天气怎么样"}},
                ],
                "self_id": 123456789,
            }
        )

    asyncio.run(run())
    assert captured, "member group message should be dispatched"
    assert captured[0].text.startswith("[受限用户:仅问答]")


def test_member_dm_rejected(monkeypatch) -> None:
    """普通用户私聊直接丢弃（pairing 入口已关，事件不构造）。"""
    adapter = _make_adapter(admin_users=[123456789])
    ws = _FakeWS(adapter)
    adapter._ws = ws
    captured: list = []

    async def fake_handle_message(ev):
        captured.append(ev)

    adapter.handle_message = fake_handle_message  # type: ignore[method-assign]

    async def run():
        await adapter._process_message(
            {
                "message_type": "private",
                "user_id": 99999999,
                "message": [{"type": "text", "data": {"text": "你好"}}],
                "self_id": 123456789,
            }
        )

    asyncio.run(run())
    assert not captured, "non-admin DM must be dropped"


def test_member_slash_command_blocked(monkeypatch) -> None:
    """普通用户斜杠命令（/help /new 等）直接丢弃。"""
    adapter = _make_adapter(admin_users=[123456789])
    ws = _FakeWS(adapter)
    adapter._ws = ws
    captured: list = []

    async def fake_handle_message(ev):
        captured.append(ev)

    adapter.handle_message = fake_handle_message  # type: ignore[method-assign]

    async def run():
        await adapter._process_message(
            {
                "message_type": "group",
                "group_id": 123456789,
                "user_id": 99999999,
                "message": [
                    {"type": "at", "data": {"qq": "123456789"}},
                    {"type": "text", "data": {"text": "/help"}},
                ],
                "self_id": 123456789,
            }
        )

    asyncio.run(run())
    assert not captured, "member slash command must be dropped"


def test_member_path_text_not_blocked(monkeypatch) -> None:
    """普通用户含路径的文本（/tmp/x 等）不误伤（命令名含 / 即非命令）。"""
    adapter = _make_adapter(admin_users=[123456789])
    ws = _FakeWS(adapter)
    adapter._ws = ws
    captured: list = []

    async def fake_handle_message(ev):
        captured.append(ev)

    adapter.handle_message = fake_handle_message  # type: ignore[method-assign]

    async def run():
        await adapter._process_message(
            {
                "message_type": "group",
                "group_id": 123456789,
                "user_id": 99999999,
                "message": [
                    {"type": "at", "data": {"qq": "123456789"}},
                    {"type": "text", "data": {"text": "看看 /tmp/x 里的内容"}},
                ],
                "self_id": 123456789,
            }
        )

    asyncio.run(run())
    assert captured, "path text is not a slash command, must pass through"
    assert captured[0].text.startswith("[受限用户:仅问答]")


def test_admin_group_message_no_prefix(monkeypatch) -> None:
    """管理员群聊消息不注入受限标记，斜杠命令放行。"""
    adapter = _make_adapter(admin_users=[123456789])
    ws = _FakeWS(adapter)
    adapter._ws = ws
    captured: list = []

    async def fake_handle_message(ev):
        captured.append(ev)

    adapter.handle_message = fake_handle_message  # type: ignore[method-assign]

    async def run():
        await adapter._process_message(
            {
                "message_type": "group",
                "group_id": 123456789,
                "user_id": 123456789,
                "message": [
                    {"type": "at", "data": {"qq": "123456789"}},
                    {"type": "text", "data": {"text": "/new"}},
                ],
                "self_id": 123456789,
            }
        )

    asyncio.run(run())
    assert captured, "admin slash command must be dispatched"
    assert not captured[0].text.startswith("[受限用户")

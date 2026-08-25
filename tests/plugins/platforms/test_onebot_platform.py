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
    """interim 缓冲 + final 结算：小结卡渲染失败 → 回退合并转发 + 撤回（群聊）。"""
    import plugins.platforms.onebot.adapter as adapter_mod

    def boom(text, title=None):
        raise RuntimeError("render unavailable (fallback path)")

    monkeypatch.setattr(adapter_mod, "render_text_image", boom)
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
    """私聊场景用 send_private_forward_msg（小结卡渲染失败回退时）。"""
    import plugins.platforms.onebot.adapter as adapter_mod

    def boom(text, title=None):
        raise RuntimeError("render unavailable (fallback path)")

    monkeypatch.setattr(adapter_mod, "render_text_image", boom)
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


def test_loop_merge_summary_card_preferred(monkeypatch) -> None:
    """#3 回移：小结卡渲染成功 → 发图片卡 + 撤回原 interim，不走合并转发。"""
    import plugins.platforms.onebot.adapter as adapter_mod

    called = {}

    def fake_render(text, title=None):
        called["text"] = text
        called["title"] = title
        return b"\x89PNG\r\n\x1a\nfakepng"

    monkeypatch.setattr(adapter_mod, "render_text_image", fake_render)
    adapter = _make_adapter()
    ws = _FakeWS(adapter)
    adapter._ws = ws
    adapter._self_id = "123456789"
    chat = "group:123456789"

    async def run():
        await adapter.send(chat, "第一步", metadata={"interim": True})
        await adapter.send(chat, "第二步", metadata={"interim": True})
        await adapter.send(chat, "最终", metadata={"notify": True})

    asyncio.run(run())
    actions = [p["action"] for p in ws.sent]
    assert "send_forward_msg" not in actions
    # 小结卡以图片形式发送（send_msg 带 image segment）+ 撤回 2 条 interim
    assert actions.count("delete_msg") == 2
    img_msgs = [
        p for p in ws.sent if p["action"] == "send_msg" and "image" in str(p["params"].get("message"))
    ]
    assert img_msgs, "expected a summary card image message"
    assert called.get("title") == "本轮进展"
    assert "第一步" in called.get("text", "")


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


def test_auto_recall_interim_after_timeout(monkeypatch) -> None:
    """#2 回移：interim 超时（90s 语义，测试用 50ms）未结算 → 单独撤回并清缓冲。"""
    adapter = _make_adapter(interim_recall_seconds=0.05)
    ws = _FakeWS(adapter)
    adapter._ws = ws
    chat = "private:123456789"

    async def run():
        await adapter.send(chat, "中间评论", metadata={"interim": True})
        assert len(adapter._loop_buffer.get(chat, [])) == 1
        # 不触发 final，等超时
        await asyncio.sleep(0.25)

    asyncio.run(run())
    actions = [p["action"] for p in ws.sent]
    assert actions.count("delete_msg") == 1
    assert adapter._loop_buffer.get(chat) in (None, [])


def test_auto_recall_interim_cancelled_by_final(monkeypatch) -> None:
    """#2：final 结算后超时任务到点不重复撤回（条目已不在缓冲）。"""
    import plugins.platforms.onebot.adapter as adapter_mod

    def boom(text, title=None):
        raise RuntimeError("render unavailable (fallback path)")

    monkeypatch.setattr(adapter_mod, "render_text_image", boom)
    adapter = _make_adapter(interim_recall_seconds=0.05)
    ws = _FakeWS(adapter)
    adapter._ws = ws
    adapter._self_id = "123456789"
    chat = "group:123456789"

    async def run():
        await adapter.send(chat, "中间", metadata={"interim": True})
        await adapter.send(chat, "最终", metadata={"notify": True})
        await asyncio.sleep(0.25)

    asyncio.run(run())
    actions = [p["action"] for p in ws.sent]
    # 只有 final 结算那一次的撤回（2 条中的 1 条缓冲不足不合并？——1 条 interim
    # 不结算，final 后 _pending_recalls 为空 → 不撤回；超时任务因条目已随
    # _merge_loop_buffer pop 清空而自愈，不应再发 delete_msg）
    assert actions.count("delete_msg") == 0


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


# ---------------------------------------------------------------------------
# _standalone_send (out-of-process cron delivery via OneBot HTTP API)
# ---------------------------------------------------------------------------


class _FakeOneBotHttp:
    """Minimal OneBot HTTP server that records incoming action payloads."""

    def __init__(self) -> None:
        from aiohttp import web

        self.requests: list = []
        self._app = web.Application()
        self._app.router.add_post("/{action}", self._handle)
        self._runner = None
        self.port = 0

    async def _handle(self, request):
        action = request.match_info["action"]
        body = await request.json()
        self.requests.append((action, body))
        return _json_response({"status": "ok", "retcode": 0, "data": None})

    async def start(self) -> None:
        from aiohttp import web

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()


def _json_response(data):
    from aiohttp import web

    return web.json_response(data)


def test_standalone_send_text_and_voice_with_mount() -> None:
    """文本 + (path, is_voice) 元组媒体：voice_mount 映射 + record 段。"""
    from plugins.platforms.onebot.adapter import _standalone_send

    server = _FakeOneBotHttp()

    async def run():
        await server.start()
        try:
            pconfig = PlatformConfig(
                enabled=True,
                extra={
                    "http_url": f"http://127.0.0.1:{server.port}",
                    "voice_mount": {
                        "host": "/data/audio",
                        "container": "/app/napcat/audio",
                    },
                },
            )
            result = await _standalone_send(
                pconfig,
                "private:123456789",
                "定时提醒",
                media_files=[("/data/audio/remind.silk", True)],
            )
        finally:
            await server.stop()

    asyncio.run(run())
    assert server.requests, "expected at least one OneBot HTTP request"
    actions = [a for a, _ in server.requests]
    assert actions == ["send_private_msg", "send_private_msg"]
    text_payload = server.requests[0][1]
    assert text_payload["user_id"] == 123456789
    assert text_payload["message"] == "定时提醒"
    media_payload = server.requests[1][1]
    assert media_payload["message"] == "[CQ:record,file=/app/napcat/audio/remind.silk]"


def test_standalone_send_group_target_and_backslash_normalization() -> None:
    """群目标走 send_group_msg；Windows 反斜杠路径经 voice_mount 规范化。"""
    from plugins.platforms.onebot.adapter import _standalone_send

    server = _FakeOneBotHttp()

    async def run():
        await server.start()
        try:
            pconfig = PlatformConfig(
                enabled=True,
                extra={
                    "http_url": f"http://127.0.0.1:{server.port}",
                    "voice_mount": {
                        "host": "C:\\data\\audio",
                        "container": "/app/napcat/audio",
                    },
                },
            )
            result = await _standalone_send(
                pconfig,
                "group:88888",
                "",
                media_files=[("C:\\data\\audio\\clip.silk", True)],
            )
        finally:
            await server.stop()

    asyncio.run(run())
    assert server.requests[0][0] == "send_group_msg"
    payload = server.requests[0][1]
    assert payload["group_id"] == 88888
    assert payload["message"] == "[CQ:record,file=/app/napcat/audio/clip.silk]"


def test_standalone_send_missing_http_url_returns_error() -> None:
    from plugins.platforms.onebot.adapter import _standalone_send

    pconfig = PlatformConfig(enabled=True, extra={})

    async def run():
        return await _standalone_send(pconfig, "private:123456789", "hi")

    result = asyncio.run(run())
    assert "ONEBOT_HTTP_URL not configured" in result["error"]


# ---------------------------------------------------------------------------
# Ported #5: inbound file dual-channel receive (CDN direct link / get_file)
# ---------------------------------------------------------------------------


def _file_process(adapter, monkeypatch, call_actions, download_results=None):
    """跑一条带文件段的消息，返回捕获的 MessageEvent。"""

    async def fake_call(action, params, timeout=30.0):
        return call_actions.get(action, {})

    monkeypatch.setattr(adapter, "_call_action", fake_call)

    if download_results is not None:
        async def fake_download(url, safe_name, max_bytes):
            return download_results.get(url)

        monkeypatch.setattr(adapter, "_download_file_bytes", fake_download)

    captured: list = []

    async def fake_handle_message(ev):
        captured.append(ev)

    adapter.handle_message = fake_handle_message  # type: ignore[method-assign]

    async def run():
        await adapter._process_message(
            {
                "message_type": "private",
                "user_id": 123456789,
                "message": [
                    {
                        "type": "file",
                        "data": {
                            "file_id": "fid-abc-123",
                            "file": "plan.md",
                            "name": "",
                        },
                    }
                ],
                "self_id": 123456789,
            }
        )

    asyncio.run(run())
    assert captured
    return captured[0]


def test_inbound_file_direct_link_downloads_and_annotates_path(monkeypatch) -> None:
    """get_private_file_url 直链 → 下载成功 → [文件:本地路径]。"""
    adapter = _make_adapter(admin_users=[123456789])
    ev = _file_process(
        adapter,
        monkeypatch,
        call_actions={"get_private_file_url": {"url": "https://cdn.qq.com/plan.md"}},
        download_results={"https://cdn.qq.com/plan.md": "/tmp/hermes_onebot/plan.md"},
    )
    assert "[文件:/tmp/hermes_onebot/plan.md]" in ev.text
    assert not ev.media_urls, "file 不进入 media_urls（靠文本注解）"


def test_inbound_file_falls_back_to_get_file_base64(monkeypatch) -> None:
    """直链失败（异常）→ get_file base64 载荷 → [文件:本地路径]。"""
    adapter = _make_adapter(admin_users=[123456789])

    async def fake_call(action, params, timeout=30.0):
        if action == "get_private_file_url":
            raise RuntimeError("private file url unsupported (group chat)")
        if action == "get_file":
            return {"base64": "aGVsbG8=", "file_size": 5}
        return {}

    monkeypatch.setattr(adapter, "_call_action", fake_call)
    captured: list = []

    async def fake_handle_message(ev):
        captured.append(ev)

    adapter.handle_message = fake_handle_message  # type: ignore[method-assign]

    async def run():
        await adapter._process_message(
            {
                "message_type": "private",
                "user_id": 123456789,
                "message": [
                    {"type": "file", "data": {"file_id": "fid", "file": "x.bin"}}
                ],
                "self_id": 123456789,
            }
        )

    asyncio.run(run())
    assert "[文件:" in captured[0].text
    assert "/tmp/hermes_onebot/" in captured[0].text


def test_inbound_file_download_failure_keeps_name(monkeypatch) -> None:
    """双通道全失败 → 保留 [文件:名] 注解，消息不阻塞。"""
    adapter = _make_adapter(admin_users=[123456789])
    ev = _file_process(
        adapter,
        monkeypatch,
        call_actions={"get_private_file_url": {"url": "https://cdn/nope.md"}, "get_file": {}},
        download_results={},  # 全部失败
    )
    assert "[文件:plan.md]" in ev.text


def test_inbound_file_size_limit_skips(monkeypatch) -> None:
    """get_file 声明的 file_size 超限 → 不下载，显示文件名。"""
    adapter = _make_adapter(admin_users=[123456789], max_inbound_file_bytes=100)
    ev = _file_process(
        adapter,
        monkeypatch,
        call_actions={
            # 直链先失败
            "get_private_file_url": {},
            "get_file": {"file_size": 999999, "base64": "AA=="},
        },
    )
    assert "[文件:plan.md]" in ev.text


def test_cq_string_file_download_annotates_path(monkeypatch) -> None:
    """CQ 字符串路径：直链成功 → [文件:本地路径]。"""
    adapter = _make_adapter(admin_users=[123456789])
    captured: list = []

    async def fake_call(action, params, timeout=30.0):
        return {"url": "https://cdn.qq.com/a.pdf"} if action == "get_private_file_url" else {}

    async def fake_download(url, safe_name, max_bytes):
        return "/tmp/hermes_onebot/a.pdf"

    monkeypatch.setattr(adapter, "_call_action", fake_call)
    monkeypatch.setattr(adapter, "_download_file_bytes", fake_download)

    async def fake_handle_message(ev):
        captured.append(ev)

    adapter.handle_message = fake_handle_message  # type: ignore[method-assign]

    async def run():
        await adapter._process_message(
            {
                "message_type": "private",
                "user_id": 123456789,
                "raw_message": "[CQ:file,file=a.pdf,file_id=fid-9]",
                "message": "[CQ:file,file=a.pdf,file_id=fid-9]",
                "self_id": 123456789,
            }
        )

    asyncio.run(run())
    assert "[文件:/tmp/hermes_onebot/a.pdf]" in captured[0].text


# ---------------------------------------------------------------------------
# Ported #1: model tools (qq_send_* / qq_napcat_api / qq_group_history)
# ---------------------------------------------------------------------------


def test_tools_resolve_chat_explicit_and_session(monkeypatch) -> None:
    from plugins.platforms.onebot import tools

    assert tools._resolve_chat("group:88888") == "group:88888"
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "onebot")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "private:123456789")
    assert tools._resolve_chat(None) == "private:123456789"
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    try:
        tools._resolve_chat(None)
        assert False, "expected ValueError without chat context"
    except ValueError:
        pass


def test_tools_send_image_builds_media_request(monkeypatch) -> None:
    from plugins.platforms.onebot import tools

    calls = []

    def fake_http(method, path, payload=None):
        calls.append((method, path, payload))
        return {"ok": True, "message_id": "m1"}

    monkeypatch.setattr(tools, "_http", fake_http)
    out = tools.qq_send_image(
        {"sources": ["/tmp/a.png", "https://x/y.jpg"], "chat_id": "private:123456789"}
    )
    assert "已发送" in out
    assert calls[0][0] == "POST"
    assert calls[0][1] == "/api/send_media"
    assert calls[0][2]["kind"] == "image"
    assert calls[0][2]["sources"] == ["/tmp/a.png", "https://x/y.jpg"]
    # 超过 9 张拒绝
    too_many = tools.qq_send_image(
        {"sources": [f"/tmp/{i}.png" for i in range(10)]}
    )
    assert "9" in too_many


def test_tools_napcat_api_whitelist_guard(monkeypatch) -> None:
    from plugins.platforms.onebot import tools

    calls = []

    def fake_http(method, path, payload=None):
        calls.append(path)
        return {"ok": True, "data": {"count": 1}}

    monkeypatch.setattr(tools, "_http", fake_http)
    # 白名单外直接拒绝，不发请求
    out = tools.qq_napcat_api({"action": "set_group_kick"})
    assert "不在白名单" in out
    assert not calls
    # 白名单内正常代理
    out = tools.qq_napcat_api({"action": "get_group_member_list", "params": {"group_id": 88888}})
    assert "api/napcat" in calls[0]
    assert '"count": 1' in out


def test_tools_group_history_url() -> None:
    from plugins.platforms.onebot import tools

    import urllib.parse

    calls = []

    def fake_http(method, path, payload=None):
        calls.append(path)
        return {"ok": True, "data": []}

    import plugins.platforms.onebot.tools as tools_mod

    tools_mod._http = fake_http
    tools.qq_group_history({"group_id": "88888", "count": 30})
    qs = urllib.parse.parse_qs(calls[0].split("?", 1)[1])
    assert qs["group_id"] == ["88888"]
    assert qs["count"] == ["30"]


async def _start_api_server(adapter):
    """起一个只挂 onebot API 路由的本地服务，返回 (runner, port)。"""
    from aiohttp import web

    app = web.Application()
    app.router.add_get("/api/napcat", adapter._handle_napcat_api)
    app.router.add_post("/api/send_media", adapter._handle_send_media)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


def test_api_napcat_whitelist_rejects() -> None:
    """非白名单 action → 403。"""
    from aiohttp import ClientSession

    adapter = _make_adapter()
    adapter._call_action = lambda *a, **kw: {}  # 不应被调用

    async def run():
        runner, port = await _start_api_server(adapter)
        try:
            async with ClientSession() as sess:
                async with sess.get(f"http://127.0.0.1:{port}/api/napcat?action=set_group_kick") as resp:
                    assert resp.status == 403
        finally:
            await runner.cleanup()

    asyncio.run(run())


def test_api_napcat_whitelisted_action() -> None:
    """白名单 action → 透传 _call_action 结果。"""
    from aiohttp import ClientSession

    adapter = _make_adapter()

    async def fake_call(action, params, timeout=30.0):
        assert action == "get_group_member_list"
        assert params == {"group_id": 88888}
        return [{"user_id": 123456789, "nickname": "M"}]

    adapter._call_action = fake_call

    async def run():
        runner, port = await _start_api_server(adapter)
        try:
            async with ClientSession() as sess:
                async with sess.get(
                    f"http://127.0.0.1:{port}/api/napcat?action=get_group_member_list&params=%7B%22group_id%22%3A88888%7D"
                ) as resp:
                    assert resp.status == 200
                    body = await resp.json()
                    assert body["status"] == "ok"
                    assert body["data"][0]["nickname"] == "M"
        finally:
            await runner.cleanup()

    asyncio.run(run())


def test_api_send_media_forward_group() -> None:
    """群合并转发 → send_forward_msg。"""
    from aiohttp import ClientSession

    adapter = _make_adapter()
    adapter._self_id = "123456789"

    async def fake_call(action, params, timeout=30.0):
        assert action == "send_forward_msg"
        assert params["group_id"] == 88888
        assert params["messages"][0]["name"] == "某人"
        return {"message_id": 777}

    adapter._call_action = fake_call

    async def run():
        runner, port = await _start_api_server(adapter)
        try:
            async with ClientSession() as sess:
                async with sess.post(
                    f"http://127.0.0.1:{port}/api/send_media",
                    json={
                        "chat_id": "group:88888",
                        "kind": "forward",
                        "nodes": [{"name": "某人", "content": "第一段"}],
                    },
                ) as resp:
                    assert resp.status == 200
                    body = await resp.json()
                    assert body["status"] == "ok"
        finally:
            await runner.cleanup()

    asyncio.run(run())


def test_api_send_media_bad_kind() -> None:
    from aiohttp import ClientSession

    adapter = _make_adapter()

    async def run():
        runner, port = await _start_api_server(adapter)
        try:
            async with ClientSession() as sess:
                async with sess.post(
                    f"http://127.0.0.1:{port}/api/send_media",
                    json={"chat_id": "private:1", "kind": "hack"},
                ) as resp:
                    assert resp.status == 400
        finally:
            await runner.cleanup()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Ported #7: t2i ink check (font-family self test)
# ---------------------------------------------------------------------------


def test_ink_check_reports_font_chain() -> None:
    """墨水自检：返回链状态，CJK 字体存在时 ok=True。"""
    from plugins.platforms.onebot.t2i_render import ink_check

    result = ink_check()
    assert "ok" in result
    assert "loaded" in result
    # 本机装了 Noto CJK → 应通过
    assert result["cjk"] is True
    assert result["ok"] is True


# ---------------------------------------------------------------------------
# Ported #6: local slash commands (/id /ver /mode /ocr)
# ---------------------------------------------------------------------------


def _command_process(adapter, ws, message_text):
    """向 _process_message 投递一条 admin 文本命令；返回捕获事件数。"""
    captured: list = []

    async def fake_handle_message(ev):
        captured.append(ev)

    adapter.handle_message = fake_handle_message  # type: ignore[method-assign]

    async def run():
        await adapter._process_message(
            {
                "message_type": "private",
                "user_id": 123456789,
                "raw_message": message_text,
                "message": message_text,
                "self_id": 123456789,
            }
        )

    asyncio.run(run())
    return captured


def test_local_command_id_and_ver(monkeypatch) -> None:
    """/id /ver 由 adapter 处理回复，事件不构造。"""
    adapter = _make_adapter(admin_users=[123456789])
    ws = _FakeWS(adapter)
    adapter._ws = ws

    captured = _command_process(adapter, ws, "/id")
    assert not captured, "local command must not reach the agent"
    texts = [
        "".join(s.get("data", {}).get("text", "") for s in p["params"]["message"])
        for p in ws.sent
        if p["action"] == "send_msg"
    ]
    assert any("chat_id: private:123456789" in t for t in texts)

    ws.sent.clear()
    captured = _command_process(adapter, ws, "/ver")
    assert not captured
    texts = [
        "".join(s.get("data", {}).get("text", "") for s in p["params"]["message"])
        for p in ws.sent
        if p["action"] == "send_msg"
    ]
    assert any("onebot-plugin v1.0.0" in t for t in texts)


def test_local_command_mode_instant_disables_loop_merge(monkeypatch) -> None:
    """/mode instant：per-chat 关闭 loop 合并，interim 逐条即时不缓冲。"""
    adapter = _make_adapter(admin_users=[123456789])
    ws = _FakeWS(adapter)
    adapter._ws = ws
    adapter._self_id = "123456789"
    chat = "private:123456789"

    _command_process(adapter, ws, "/mode instant")
    assert adapter._chat_interim_overrides.get(chat) is False

    async def run():
        await adapter.send(chat, "中间一", metadata={"interim": True})
        await adapter.send(chat, "中间二", metadata={"interim": True})
        await adapter.send(chat, "最终", metadata={"notify": True})

    asyncio.run(run())
    assert not adapter._loop_buffer.get(chat), "instant 模式不应缓冲 interim"
    actions = [p["action"] for p in ws.sent]
    assert "send_forward_msg" not in actions
    assert "delete_msg" not in actions


def test_local_command_ocr_uses_last_inbound_image(monkeypatch) -> None:
    """/ocr：用最近入站图片调 ocr_image，文本结果回发。"""
    adapter = _make_adapter(admin_users=[123456789])
    ws = _FakeWS(adapter)
    adapter._ws = ws
    sent_calls: list = []

    # 1) 先收一张图片消息（mock 下载），记录 _last_image_path
    async def fake_resolve_image(url, file):
        return "/tmp/hermes_onebot/ocr.png"

    async def fake_call(action, params, timeout=30.0):
        sent_calls.append((action, params))
        if action == "ocr_image":
            assert params["image"].startswith("base64://")
            return {"texts": [{"text": "第一行"}, {"text": "第二行"}]}
        return {}

    monkeypatch.setattr(adapter, "_resolve_image", fake_resolve_image)

    async def fake_base64(path, max_bytes=None):
        return "aGVsbG8="

    monkeypatch.setattr(adapter, "_file_to_base64", fake_base64)
    monkeypatch.setattr(adapter, "_call_action", fake_call)
    captured: list = []

    async def fake_handle_message(ev):
        captured.append(ev)

    adapter.handle_message = fake_handle_message  # type: ignore[method-assign]

    async def run():
        await adapter._process_message(
            {
                "message_type": "private",
                "user_id": 123456789,
                "message": [{"type": "image", "data": {"url": "https://cdn/x.png", "file": "x.png"}}],
                "self_id": 123456789,
            }
        )
        await adapter._process_message(
            {
                "message_type": "private",
                "user_id": 123456789,
                "raw_message": "/ocr",
                "message": "/ocr",
                "self_id": 123456789,
            }
        )

    asyncio.run(run())
    assert adapter._last_image_path.get("private:123456789") == "/tmp/hermes_onebot/ocr.png"
    texts = [
        "".join(s.get("data", {}).get("text", "") for s in p[1]["message"])
        for p in sent_calls
        if p[0] == "send_msg"
    ]
    assert any("OCR 结果" in t and "第一行" in t for t in texts)

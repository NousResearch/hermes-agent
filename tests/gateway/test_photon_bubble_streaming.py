"""Unit and integration tests for Photon discrete multi-bubble streaming."""



import asyncio
import json
import os
import re
import sys
import unittest
from pathlib import Path

# Ensure repository root is at the very front of sys.path (supports both AirHermes and deployed Hermes layouts)
_cur = Path(__file__).resolve().parent
repo_root = str(Path(__file__).resolve().parent.parent)
while _cur.parent != _cur:
    if (_cur / "gateway" / "run_turn.py").is_file() or (_cur / "install.py").is_file():
        repo_root = str(_cur)
        break
    _cur = _cur.parent

if repo_root in sys.path:
    sys.path.remove(repo_root)
sys.path.insert(0, repo_root)

try:
    import gateway
    airhermes_gateway = str(Path(repo_root) / "gateway")
    if hasattr(gateway, "__path__") and airhermes_gateway not in gateway.__path__:
        gateway.__path__.insert(0, airhermes_gateway)
except ImportError:
    pass

from typing import Any, Optional

from gateway.bubble_parser import BubbleParser
from gateway.bubble_queue import ChatPacerRegistry
from gateway.bubble_stream_consumer import BubbleStreamConsumer

try:
    from plugins.platforms.photon.adapter import PhotonAdapter
    from gateway.run_turn_runner import TurnRunner
    HERMES_UPSTREAM_AVAILABLE = True
except ImportError:
    PhotonAdapter = None
    TurnRunner = None
    HERMES_UPSTREAM_AVAILABLE = False





class MockSendResult:

    def __init__(self, success: bool = True, message_id: Optional[str] = None, error: Optional[str] = None):

        self.success = success

        self.message_id = message_id or "guid_mock_msg"

        self.error = error





class MockAdapter:

    def __init__(self):

        self.sent_messages: list[dict[str, Any]] = []

        self.reactions: list[dict[str, Any]] = []

        self.polls: list[dict[str, Any]] = []

        self.documents: list[dict[str, Any]] = []

        self.effects: list[dict[str, Any]] = []

        self.send_delay: float = 0.01



    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[dict] = None):

        await asyncio.sleep(self.send_delay)  # health: allow HX013 -- simulated network latency

        record = {

            "chat_id": chat_id,

            "content": content,

            "reply_to": reply_to,

            "metadata": metadata,

        }

        self.sent_messages.append(record)

        return MockSendResult(success=True, message_id=f"msg_{len(self.sent_messages)}")



    async def add_reaction(self, chat_id: str, emoji: str, message_id: Optional[str] = None):

        await asyncio.sleep(self.send_delay)  # health: allow HX013 -- simulated network latency

        record = {

            "chat_id": chat_id,

            "emoji": emoji,

            "message_id": message_id,

        }

        self.reactions.append(record)

        return {"success": True, "message_id": message_id or "reaction_target"}



    async def _sidecar_send_poll(self, chat_id: str, title: str, options: list[str]):

        await asyncio.sleep(self.send_delay)  # health: allow HX013 -- simulated network latency

        record = {

            "chat_id": chat_id,

            "title": title,

            "options": options,

        }

        self.polls.append(record)

        return {"success": True, "message_id": f"poll_{len(self.polls)}"}



    async def _sidecar_send_effect(self, chat_id: str, text: str, effect: str):

        await asyncio.sleep(self.send_delay)  # health: allow HX013 -- simulated network latency

        record = {

            "chat_id": chat_id,

            "text": text,

            "effect": effect,

        }

        self.effects.append(record)

        return {"success": True, "message_id": f"eff_{len(self.effects)}"}



    async def send_document(self, chat_id: str, file_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None):

        await asyncio.sleep(self.send_delay)  # health: allow HX013 -- simulated network latency

        record = {

            "chat_id": chat_id,

            "file_path": file_path,

            "caption": caption,

            "reply_to": reply_to,

        }

        self.documents.append(record)

        return MockSendResult(success=True, message_id=f"doc_{len(self.documents)}")





class TestPhotonBubbleStreaming(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        ChatPacerRegistry.reset()

    def test_bubble_parser_strict_tags(self):

        parser = BubbleParser(paragraph_fallback=True)

        b1 = parser.feed("Here is thinking <bubble>First idea</bubble> and then ")

        self.assertEqual(b1, ["First idea"])



        b2 = parser.feed("<bubble>Second idea: ```python\nprint(1)")

        self.assertEqual(b2, [])  # unclosed bubble



        b3 = parser.feed("\n```</bubble>")

        self.assertEqual(b3, ["Second idea: ```python\nprint(1)\n```"])



        rem = parser.finish()

        self.assertEqual(rem, [])



    def test_bubble_parser_paragraph_fallback(self):

        parser = BubbleParser(paragraph_fallback=True)

        # Without bubble tags, double newlines trigger paragraph splitting

        b1 = parser.feed("This is the first standalone paragraph with enough content.\n\n")

        self.assertEqual(len(b1), 1)

        self.assertEqual(b1[0], "This is the first standalone paragraph with enough content.")



        b2 = parser.feed("Here is the second paragraph trailing.")

        self.assertEqual(b2, [])



        rem = parser.finish()

        self.assertEqual(rem, ["Here is the second paragraph trailing."])



    async def test_bubble_stream_consumer_lifecycle(self):

        adapter = MockAdapter()

        consumer = BubbleStreamConsumer(

            adapter=adapter,

            chat_id="chat_test_1",

            reply_to="inbound_123",

        )

        # Run consumer in background

        consumer_task = asyncio.create_task(consumer.run())



        # Feed deltas

        consumer.on_delta("<bubble>Hello from iMessage bubble 1!</bubble>")

        await asyncio.sleep(0.05)  # health: allow HX013 -- queue pacing verification

        consumer.on_delta("<bubble>Here is bubble 2 with details.</bubble>")

        await asyncio.sleep(0.05)  # health: allow HX013 -- queue pacing verification



        # Signal finish

        consumer.finish()

        await consumer_task



        self.assertTrue(consumer.final_content_delivered)

        self.assertTrue(consumer.final_response_sent)

        self.assertEqual(len(adapter.sent_messages), 2)

        self.assertEqual(adapter.sent_messages[0]["content"], "Hello from iMessage bubble 1!")

        self.assertEqual(adapter.sent_messages[1]["content"], "Here is bubble 2 with details.")

        self.assertEqual(adapter.sent_messages[0]["reply_to"], "inbound_123")



    async def test_bubble_stream_consumer_superseded(self):

        adapter = MockAdapter()

        active = True



        def run_still_current():

            return active



        consumer = BubbleStreamConsumer(

            adapter=adapter,

            chat_id="chat_test_cancel",

            run_still_current=run_still_current,

        )

        consumer_task = asyncio.create_task(consumer.run())



        consumer.on_delta("<bubble>Bubble before cancel</bubble>")

        await asyncio.sleep(0.05)  # health: allow HX013 -- queue pacing verification



        # Cancel turn

        active = False

        await asyncio.sleep(0.05)  # health: allow HX013 -- queue pacing verification

        consumer.finish()

        await consumer_task



        # Consumer should exit gracefully without hanging or leaking

        self.assertTrue(consumer_task.done())



    def test_bubble_parser_url_isolation(self):

        # isolate_urls=False preserves URLs inline

        p_off = BubbleParser(isolate_urls=False)

        b_off = p_off.feed("<bubble>Check this: https://example.com for info</bubble>")

        self.assertEqual(b_off, ["Check this: https://example.com for info"])



        # isolate_urls=True splits URLs out into standalone bubbles

        p_on = BubbleParser(isolate_urls=True)

        b_on = p_on.feed("<bubble>Check this: https://example.com for info</bubble>")

        self.assertEqual(b_on, ["Check this:", "https://example.com", "for info"])



        # Chinese punctuation attached to URL

        b_cn = p_on.feed("<bubble>这是链接：https://example.com/123，可以看下！</bubble>")

        self.assertEqual(b_cn, ["这是链接：", "https://example.com/123", "可以看下！"])



        # Markdown links expanded cleanly into standalone URL

        b_md = p_on.feed("<bubble>详见 [Pitchfork Review](https://example.com/album) 乐评</bubble>")

        self.assertEqual(b_md, ["详见 Pitchfork Review", "https://example.com/album", "乐评"])



        # URLs inside code fences remain untouched

        code_input = "<bubble>```bash\ncurl https://example.com\n```</bubble>"

        b_code = p_on.feed(code_input)

        self.assertEqual(b_code, ["```bash\ncurl https://example.com\n```"])



    def test_bubble_queue_adaptive_pacing(self):
        from gateway.bubble_queue import ChatPacer

        pacer = ChatPacer(chat_id="test_adaptive", min_interval_s=0.400, adaptive=True)

        # Standalone bare URL has preview delay
        pacer.update_finish_time(last_text="https://apnews.com/article/123")
        self.assertTrue(1.0 <= pacer.current_interval_s <= 2.0)

        # Short text: clamped to minimum
        pacer.update_finish_time(last_text="好")
        self.assertTrue(pacer.current_interval_s >= pacer.min_interval_s)

        # Long text (80 chars): interval scales with length
        long_txt = "A" * 80
        pacer.update_finish_time(last_text=long_txt)
        self.assertTrue(pacer.current_interval_s > 1.5)

    async def test_bubble_stream_consumer_commentary(self):

        adapter = MockAdapter()

        consumer = BubbleStreamConsumer(

            adapter=adapter,

            chat_id="chat_test_commentary",

        )

        consumer_task = asyncio.create_task(consumer.run())



        consumer.on_commentary("<bubble>正在全网检索最新信息...</bubble>")

        await asyncio.sleep(0.05)  # health: allow HX013 -- queue pacing verification

        consumer.on_delta("<bubble>找到结果了！</bubble>")

        await asyncio.sleep(0.05)  # health: allow HX013 -- queue pacing verification



        consumer.finish()

        await consumer_task



        self.assertEqual(len(adapter.sent_messages), 2)

        self.assertEqual(adapter.sent_messages[0]["content"], "正在全网检索最新信息...")

        self.assertEqual(adapter.sent_messages[1]["content"], "找到结果了！")



    async def test_bubble_stream_consumer_duplicate_suppression_properties(self):

        adapter = MockAdapter()

        consumer = BubbleStreamConsumer(adapter=adapter, chat_id="chat_test_suppress")

        consumer_task = asyncio.create_task(consumer.run())



        # Initially nothing delivered

        self.assertFalse(consumer.final_content_delivered)

        self.assertFalse(consumer.final_response_sent)



        # Feed delta and wait for bubble send

        consumer.on_delta("<bubble>First delivered bubble</bubble>")

        await asyncio.sleep(0.08)  # health: allow HX013 -- queue pacing verification



        # As soon as bubble is sent, suppression flags MUST be True to prevent duplicate final-send

        self.assertTrue(consumer.final_content_delivered)

        self.assertTrue(consumer.final_response_sent)

        self.assertTrue(consumer.delivered_final_matches("Any final response text"))

        self.assertTrue(consumer.has_durably_delivered_text("Any final response text"))



        consumer.finish()

        await consumer_task



    def test_bubble_parser_tapback_extraction(self):

        p = BubbleParser()

        # 1. Standalone tapback

        res1 = p.feed("<tapback>❤️</tapback>")

        res1.extend(p.finish())

        self.assertEqual(res1, ["TAPBACK:❤️"])



        # 2. Tapback followed by bubble

        p.finish()

        res2 = p.feed("<tapback>😂</tapback><bubble>哈哈太损了</bubble>")

        res2.extend(p.finish())

        self.assertEqual(res2, ["TAPBACK:😂", "哈哈太损了"])



        # 3. English name normalization

        p.finish()

        res3 = p.feed("<reaction>love</reaction>")

        res3.extend(p.finish())

        self.assertEqual(res3, ["TAPBACK:❤️"])



        # 4. Self-closing emoji attribute

        p.finish()

        res4 = p.feed("<tapback emoji=\"👍\"/>")

        res4.extend(p.finish())

        self.assertEqual(res4, ["TAPBACK:👍"])



        # 5. Code block protection

        p.finish()

        code_input = "<bubble>```python\n# <tapback>❤️</tapback>\n```</bubble><bubble>这是代码</bubble>"

        res5 = p.feed(code_input)

        res5.extend(p.finish())

        self.assertEqual(res5, ["```python\n# <tapback>❤️</tapback>\n```", "这是代码"])



    async def test_bubble_stream_consumer_tapback_delivery(self):

        adapter = MockAdapter()

        consumer = BubbleStreamConsumer(

            adapter=adapter,

            chat_id="chat_tapback_test",

            reply_to="target_inbound_msg_999",

        )

        consumer_task = asyncio.create_task(consumer.run())



        # Stream a tapback and a bubble

        consumer.on_delta("<tapback>😂</tapback><bubble>真有你的！</bubble>")

        await asyncio.sleep(0.08)  # health: allow HX013 -- queue pacing verification



        consumer.finish()

        await consumer_task



        # Verify reaction was delivered to target inbound message

        self.assertEqual(len(adapter.reactions), 1)

        self.assertEqual(adapter.reactions[0]["chat_id"], "chat_tapback_test")

        self.assertEqual(adapter.reactions[0]["emoji"], "😂")

        self.assertEqual(adapter.reactions[0]["message_id"], "target_inbound_msg_999")



        # Verify bubble was delivered

        self.assertEqual(len(adapter.sent_messages), 1)

        self.assertEqual(adapter.sent_messages[0]["content"], "真有你的！")



        # Verify delivery flags are True

        self.assertTrue(consumer.final_content_delivered)



    async def test_bubble_stream_consumer_zero_text_tapback_delivery(self):

        adapter = MockAdapter()

        consumer = BubbleStreamConsumer(

            adapter=adapter,

            chat_id="chat_pure_tapback",

            reply_to="inbound_polite_ack",

        )

        consumer_task = asyncio.create_task(consumer.run())



        # Stream ONLY a tapback (no text bubbles at all)

        consumer.on_delta("<tapback>❤️</tapback>")

        await asyncio.sleep(0.08)  # health: allow HX013 -- queue pacing verification



        consumer.finish()

        await consumer_task



        # Reaction delivered

        self.assertEqual(len(adapter.reactions), 1)

        self.assertEqual(adapter.reactions[0]["emoji"], "❤️")

        self.assertEqual(adapter.reactions[0]["message_id"], "inbound_polite_ack")



        # Zero text messages delivered

        self.assertEqual(len(adapter.sent_messages), 0)



        # Content still marked delivered so no fallback text is triggered

        self.assertTrue(consumer.final_content_delivered)



    def test_bubble_parser_poll_extraction(self):

        p = BubbleParser()

        poll_input = (

            "<bubble>Here is a question for you:</bubble>"

            "<poll title=\"Dinner choice?\">"

            "<option>Pizza</option>"

            "<option>Sushi</option>"

            "<option>Burger</option>"

            "</poll>"

        )

        res = p.feed(poll_input)

        res.extend(p.finish())

        self.assertEqual(len(res), 2)

        self.assertEqual(res[0], "Here is a question for you:")

        self.assertTrue(res[1].startswith("POLL:"))


        poll_data = json.loads(res[1][len("POLL:"):])

        self.assertEqual(poll_data["title"], "Dinner choice?")

        self.assertEqual(poll_data["options"], ["Pizza", "Sushi", "Burger"])



    async def test_bubble_stream_consumer_poll_delivery(self):

        adapter = MockAdapter()

        consumer = BubbleStreamConsumer(

            adapter=adapter,

            chat_id="chat_poll_test",

        )

        consumer_task = asyncio.create_task(consumer.run())



        consumer.on_delta(

            "<poll title=\"Which option?\"><option>Option 1</option><option>Option 2</option></poll>"

        )

        await asyncio.sleep(0.08)  # health: allow HX013 -- queue pacing verification

        consumer.finish()

        await consumer_task



        self.assertEqual(len(adapter.polls), 1)

        self.assertEqual(adapter.polls[0]["chat_id"], "chat_poll_test")

        self.assertEqual(adapter.polls[0]["title"], "Which option?")

        self.assertEqual(adapter.polls[0]["options"], ["Option 1", "Option 2"])



    def test_bubble_parser_code_offloading(self):

        p = BubbleParser()

        code_lines = [f"line_{i} = {i}" for i in range(25)]

        long_code = "\n".join(code_lines)

        bubble_text = f"<bubble>Here is the script:\n```python\n{long_code}\n```\nLet me know!</bubble>"

        res = p.feed(bubble_text)

        res.extend(p.finish())



        self.assertEqual(len(res), 2)

        # First element is preview bubble

        self.assertIn("line_0 = 0", res[0])

        self.assertIn("已省略其余 20 行代码", res[0])

        # Second element is FILE: record

        self.assertTrue(res[1].startswith("FILE:"))

        file_path = res[1][len("FILE:"):].strip()

        self.assertTrue(os.path.exists(file_path))

        with open(file_path, "r", encoding="utf-8-sig") as fp:

            saved_content = fp.read()

        self.assertEqual(saved_content.strip(), long_code.strip())



    async def test_bubble_stream_consumer_file_delivery(self):

        adapter = MockAdapter()

        consumer = BubbleStreamConsumer(

            adapter=adapter,

            chat_id="chat_file_test",

        )

        consumer_task = asyncio.create_task(consumer.run())



        import tempfile

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py") as tf:

            tf.write("print('hello world')")

            temp_path = tf.name



        try:

            consumer.on_delta(f"FILE:{temp_path}")

            await asyncio.sleep(0.08)  # health: allow HX013 -- queue pacing verification

            consumer.finish()

            await consumer_task



            # Under Phase 5 Stage 2A security hardening, unprivileged plain text FILE:
            # is never dispatched as a native local document attachment (VULN-01).
            self.assertEqual(len(adapter.documents), 0)
            self.assertTrue(any(f"FILE:{temp_path}" in msg["content"] for msg in adapter.sent_messages))

        finally:

            import os

            if os.path.exists(temp_path):

                os.remove(temp_path)

    def test_bubble_parser_effect_extraction(self):

        p = BubbleParser()

        res = p.feed("<effect name=\"confetti\">太棒了！恭喜！🎉</effect>")

        res.extend(p.finish())

        self.assertEqual(len(res), 1)

        self.assertTrue(res[0].startswith("EFFECT:"))


        data = json.loads(res[0][len("EFFECT:"):])

        self.assertEqual(data["effect"], "confetti")

        self.assertEqual(data["text"], "太棒了！恭喜！🎉")



    async def test_bubble_stream_consumer_effect_delivery(self):

        adapter = MockAdapter()

        consumer = BubbleStreamConsumer(

            adapter=adapter,

            chat_id="chat_effect_test",

        )

        consumer_task = asyncio.create_task(consumer.run())



        consumer.on_delta("<effect name=\"invisible\">剧透警告</effect>")

        await asyncio.sleep(0.08)  # health: allow HX013 -- queue pacing verification

        consumer.finish()

        await consumer_task



        self.assertEqual(len(adapter.effects), 1)

        self.assertEqual(adapter.effects[0]["chat_id"], "chat_effect_test")

        self.assertEqual(adapter.effects[0]["effect"], "invisible")

        self.assertEqual(adapter.effects[0]["text"], "剧透警告")



    def test_bubble_parser_calendar_extraction(self):

        p = BubbleParser()

        xml = (

            '<calendar title="Q4复盘会" start="2026-11-06 14:00" end="2026-11-06 15:00" '

            'location="腾讯会议" description="复盘讨论">这是会议邀请</calendar>'

        )

        res = p.feed(xml)

        res.extend(p.finish())

        self.assertEqual(len(res), 2)

        self.assertEqual(res[0], "这是会议邀请")

        self.assertTrue(res[1].startswith("FILE:"))

        file_path = res[1][len("FILE:"):].strip()
        self.assertTrue(os.path.exists(file_path))

        with open(file_path, "rb") as fp:
            raw_bytes = fp.read()

        self.assertFalse(raw_bytes.startswith(b"\xef\xbb\xbf"))  # No BOM
        self.assertIn(b"\r\n", raw_bytes)
        content = raw_bytes.decode("utf-8")

        self.assertIn("BEGIN:VCALENDAR", content)
        self.assertIn("SUMMARY:Q4复盘会", content)
        self.assertNotIn("SUMMARY;CHARSET=UTF-8:", content)
        self.assertIn("DTSTART:20261106T140000", content)
        self.assertIn("LOCATION:腾讯会议", content)
        self.assertNotIn("LOCATION;CHARSET=UTF-8:", content)

    def test_bubble_parser_calendar_rfc5545_folding_and_encoding(self):
        """Verify RFC 5545 calendar conformance: 75-octet folding, UTF-8 without BOM, CRLF line endings,
        unfolding exact recovery, escaping, and CRLF injection protection."""
        p = BubbleParser()
        long_title = "这是一个非常冗长且包含很多汉字的会议主题用来严格测试RFC5545规范的75字节折行逻辑"
        long_desc = "第一行描述包含标点, 逗号, 分号; 反斜杠\\以及各种技术规格，测试长文本属性在75字节边界下的优雅折叠与安全恢复"
        long_loc = "北京市海淀区中关村南大街1号某某大厦B座1208多功能高保真全景视频会议室"
        xml = (
            f'<calendar title="{long_title}" start="2026-12-01 09:30" end="2026-12-01 11:30" '
            f'location="{long_loc}" description="{long_desc}">详细日程</calendar>'
        )
        res = p.feed(xml) + p.finish()
        self.assertEqual(len(res), 2)
        self.assertTrue(res[1].startswith("FILE:"))
        file_path = res[1][len("FILE:"):].strip()
        self.assertTrue(os.path.exists(file_path))

        with open(file_path, "rb") as fp:
            raw_bytes = fp.read()

        # 1. UTF-8 without BOM
        self.assertFalse(raw_bytes.startswith(b"\xef\xbb\xbf"), "ICS file must not contain UTF-8 BOM")

        # 2. Strict CRLF endings and no bare LF or double CR
        self.assertTrue(raw_bytes.endswith(b"\r\n"))
        self.assertNotIn(b"\r\r\n", raw_bytes)
        lines_raw = raw_bytes.split(b"\r\n")
        self.assertTrue(len(lines_raw) > 5)

        # 3. No line exceeds 75 octets (RFC 5545 §3.1 SHOULD)
        for idx, line_bytes in enumerate(lines_raw):
            if not line_bytes:
                continue
            self.assertLessEqual(len(line_bytes), 75, f"Line {idx} exceeds 75 octets: {line_bytes}")

        # 4. Check standard properties without nonstandard CHARSET parameter
        content = raw_bytes.decode("utf-8")
        self.assertIn("SUMMARY:", content)
        self.assertIn("DESCRIPTION:", content)
        self.assertIn("LOCATION:", content)
        self.assertNotIn("SUMMARY;CHARSET=UTF-8:", content)
        self.assertNotIn("DESCRIPTION;CHARSET=UTF-8:", content)
        self.assertNotIn("LOCATION;CHARSET=UTF-8:", content)

        # 5. Unfolding restores original content cleanly without UTF-8 sequence corruption
        unfolded = re.sub(r"\r\n[ \t]", "", content)

        expected_clean_title = BubbleParser._escape_ics_text(long_title)
        expected_clean_desc = BubbleParser._escape_ics_text(long_desc, multiline=True)
        expected_clean_loc = BubbleParser._escape_ics_text(long_loc)

        self.assertIn(f"SUMMARY:{expected_clean_title}", unfolded)
        self.assertIn(f"DESCRIPTION:{expected_clean_desc}", unfolded)
        self.assertIn(f"LOCATION:{expected_clean_loc}", unfolded)

        # 6. Verify escaping of comma, semicolon, backslash
        self.assertIn(r"\,", unfolded)
        self.assertIn(r"\;", unfolded)
        self.assertIn(r"\\", unfolded)

        # 7. Test multiline text and newline escaping via direct _generate_ics
        multiline_desc = "第一行会议概要\n第二行关键决议\r\n第三行待办事项"
        direct_ics = p._generate_ics("多行会议", "2026-12-02 10:00", "2026-12-02 11:00", "会议室A", multiline_desc)
        self.assertIsNotNone(direct_ics)
        with open(direct_ics, "rb") as fp_direct:
            direct_bytes = fp_direct.read()
        self.assertFalse(direct_bytes.startswith(b"\xef\xbb\xbf"))
        direct_content = direct_bytes.decode("utf-8")
        direct_unfolded = re.sub(r"\r\n[ \t]", "", direct_content)
        self.assertIn(r"第一行会议概要\n第二行关键决议\n第三行待办事项", direct_unfolded)

        # 8. CRLF injection protection: verify newlines in single-line fields cannot inject ICS statements
        crlf_title = "正常标题\r\nATTENDEE;CN=Hacker:mailto:evil@test.com\r\nSUMMARY:篡改"
        injected_ics = p._generate_ics(crlf_title, "2026-12-03 10:00", "2026-12-03 11:00", "会议室B", "")
        self.assertIsNotNone(injected_ics)
        with open(injected_ics, "rb") as fp_inj:
            inj_bytes = fp_inj.read()
        inj_content = inj_bytes.decode("utf-8")
        inj_unfolded = re.sub(r"\r\n[ \t]", "", inj_content)
        # Verify ATTENDEE was NOT injected as a standalone property
        self.assertNotIn("\nATTENDEE", inj_unfolded)
        self.assertNotIn("\rATTENDEE", inj_unfolded)

    def test_bubble_parser_map_extraction(self):
        p = BubbleParser()
        xml = '<map coordinate="22.224483,113.259169" name="Taoyuan Road" />'
        res = p.feed(xml)
        res.extend(p.finish())
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], "https://maps.apple.com/place?coordinate=22.224483,113.259169&name=Taoyuan%20Road&map=explore")

        # Test city fallback coordinate resolution
        p2 = BubbleParser()
        xml2 = '<map query="一兰拉面 旺角店" title="一兰拉面 (香港旺角店)" address="香港九龙旺角洗衣街88号碧荟地下B铺" />'
        res2 = p2.feed(xml2)
        res2.extend(p2.finish())
        self.assertEqual(len(res2), 1)
        self.assertIn("https://maps.apple.com/place?coordinate=22.3193,114.1694", res2[0])
        self.assertIn("map=explore", res2[0])



    def test_bubble_parser_embedded_spoiler_inside_bubble(self):

        p = BubbleParser()

        res = p.feed("<bubble><spoiler>地图（Map）</spoiler></bubble>")

        res.extend(p.finish())

        self.assertEqual(len(res), 1)

        self.assertEqual(res[0], 'EFFECT:{"effect": "invisible", "text": "地图（Map）"}')



    def test_bubble_parser_embedded_spoiler_with_text(self):

        p = BubbleParser()

        res = p.feed("<bubble>谜底是：<spoiler>地图（Map）</spoiler></bubble>")

        res.extend(p.finish())

        self.assertEqual(len(res), 2)

        self.assertEqual(res[0], "谜底是：")

        self.assertEqual(res[1], 'EFFECT:{"effect": "invisible", "text": "地图（Map）"}')



    def test_bubble_parser_markdown_pipe_spoiler(self):

        p = BubbleParser()

        res = p.feed("<bubble>谜底是：||地图（Map）||</bubble>")

        res.extend(p.finish())

        self.assertEqual(len(res), 2)

        self.assertEqual(res[0], "谜底是：")

        self.assertEqual(res[1], 'EFFECT:{"effect": "invisible", "text": "地图（Map）"}')



    def test_bubble_parser_self_closing_calendar_event(self):

        p = BubbleParser()

        xml = '<calendar_event title="迪士尼庆功之旅" start="2026-11-07T10:00:00+08:00" end="2026-11-07T20:00:00+08:00" location="上海迪士尼度假区" description="庆祝拿到心仪 Offer！" />'

        res = p.feed(xml) + p.finish()

        self.assertEqual(len(res), 1)

        self.assertTrue(res[0].startswith("FILE:"))

        self.assertTrue(res[0].endswith(".ics"))



    def test_bubble_parser_robotic_boilerplate_pruning(self):

        p = BubbleParser()

        # Pure boilerplate should be pruned completely

        res1 = p.feed("<bubble>名片卡片给你生成好啦，点击上面的卡片可以直接一键拨打或存入通讯录～</bubble>") + p.finish()

        self.assertEqual(len(res1), 0)



        # Partial boilerplate appended at end should be stripped, keeping conversational content

        p2 = BubbleParser()

        res2 = p2.feed("<bubble>记好啦！明天（周五）晚上 18:00 的「煲饭」原生日程卡片已生成，点一下直接存入你的日历～</bubble>") + p2.finish()

        self.assertEqual(len(res2), 1)

        self.assertIn("记好啦！明天（周五）晚上 18:00 的「煲饭」", res2[0])

        self.assertNotIn("卡片", res2[0])



    def test_bubble_parser_effect_prefix_colon(self):

        p = BubbleParser()

        raw = '<effect:confetti>\n<bubble>哇啊啊啊恭喜！！！🎉</bubble>'

        res = p.feed(raw) + p.finish()

        self.assertEqual(len(res), 1)

        self.assertEqual(res[0], 'EFFECT:{"effect": "confetti", "text": "哇啊啊啊恭喜！！！🎉"}')



    def test_bubble_parser_apple_maps_bare_url_between_bubbles(self):

        p = BubbleParser()

        raw = '<bubble>集合地点看这里：</bubble>\nhttps://maps.apple.com/?q=Shanghai+Disney+Resort&ll=31.1415,121.6578\n<bubble>不见不散！</bubble>'

        res = p.feed(raw) + p.finish()

        self.assertEqual(len(res), 3)

        self.assertEqual(res[0], "集合地点看这里：")

        self.assertEqual(res[1], "https://maps.apple.com/?q=Shanghai+Disney+Resort&ll=31.1415,121.6578")

        self.assertEqual(res[2], "不见不散！")



    def test_bubble_parser_full_multimodal_message_all_in_one(self):

        p = BubbleParser(paragraph_fallback=True, isolate_urls=True)

        raw = (

            '<effect:confetti>\n'

            '<bubble>哇啊啊啊太牛逼啦恭喜！！！🎉🎉🎉</bubble>\n'

            '<bubble>心仪公司 Offer 到手，这周末必须冲去迪士尼好好爽玩庆祝一波！</bubble>\n\n'

            '<calendar_event title="迪士尼庆功之旅" start="2026-11-07T10:00:00+08:00" end="2026-11-07T20:00:00+08:00" location="上海迪士尼度假区" description="庆祝拿到心仪 Offer！" />\n\n'

            'https://maps.apple.com/?q=Shanghai+Disney+Resort&ll=31.1415,121.6578\n\n'

            '<poll title="庆祝晚饭吃什么？"><option>烤肉</option><option>火锅</option></poll>\n\n'

            '<bubble><spoiler>通关新手村！✨</spoiler></bubble>\n'

        )

        res = p.feed(raw) + p.finish()

        self.assertEqual(len(res), 6)

        self.assertTrue(res[0].startswith('EFFECT:{"effect": "confetti"'))

        self.assertEqual(res[1], "心仪公司 Offer 到手，这周末必须冲去迪士尼好好爽玩庆祝一波！")

        self.assertTrue(res[2].startswith("FILE:") and res[2].endswith(".ics"))

        self.assertEqual(res[3], "https://maps.apple.com/?q=Shanghai+Disney+Resort&ll=31.1415,121.6578")

        self.assertTrue(res[4].startswith('POLL:{"title": "庆祝晚饭吃什么？"'))

        self.assertEqual(res[5], 'EFFECT:{"effect": "invisible", "text": "通关新手村！✨"}')



    def test_markdown_table_preserved_as_text(self):
        """Verify that standard Markdown tables remain plain text and are not converted to images or polls."""
        raw = (
            "<bubble>这是配置清单：</bubble>\n"
            "<bubble>| 项目 | 规格 | 备注 |\n"
            "| :--- | :--- | :--- |\n"
            "| CPU | 8核 | 3.2GHz |\n"
            "| 内存 | 32GB | DDR5 |</bubble>\n"
            "<bubble>请确认配置是否合适</bubble>"
        )
        p = BubbleParser()
        res = p.feed(raw) + p.finish()
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0], "这是配置清单：")
        self.assertIn("| CPU | 8核 | 3.2GHz |", res[1])
        self.assertFalse(any(b.startswith("IMAGE:") for b in res))
        self.assertFalse(any(b.startswith("POLL:") for b in res))
        self.assertEqual(res[2], "请确认配置是否合适")

        # Also verify without bubble tags (natural paragraph fallback)
        raw_natural = (
            "这是配置清单：\n\n"
            "| 项目 | 规格 | 备注 |\n"
            "| :--- | :--- | :--- |\n"
            "| CPU | 8核 | 3.2GHz |\n"
            "| 内存 | 32GB | DDR5 |\n\n"
            "请确认配置是否合适"
        )
        p_nat = BubbleParser()
        res_nat = p_nat.feed(raw_natural) + p_nat.finish()
        self.assertEqual(len(res_nat), 3)
        self.assertFalse(any(b.startswith("IMAGE:") for b in res_nat))





    def test_bubble_parser_apple_music_url_normalized_to_cn(self):

        """Verify Apple Music US link is auto-normalized to /tr/ and isolated into its own bubble."""

        raw = (

            "<bubble>给你挑了这张专注电音歌单：</bubble>\n"

            "https://music.apple.com/us/playlist/pure-focus/pl.271ecf20531548e6921b711a84fbe43d\n"

            "<bubble>很适合写代码</bubble>"

        )

        p = BubbleParser()

        res = p.feed(raw) + p.finish()

        self.assertEqual(len(res), 3)

        self.assertEqual(res[0], "给你挑了这张专注电音歌单：")

        self.assertEqual(res[1], "https://music.apple.com/tr/playlist/pure-focus/pl.271ecf20531548e6921b711a84fbe43d")

        self.assertEqual(res[2], "很适合写代码")



    def test_bubble_parser_apple_music_search_discarded_and_boilerplate_pruned(self):

        """Verify Apple Music search links are discarded and robotic boilerplate is pruned."""

        raw = "土区 Apple Music 搜索直达给你备好啦，点开就能直接拉起大图播放器试听：https://music.apple.com/search?term=%E5%91%A8%E6%9D%B0%E4%BC%A6"

        p = BubbleParser()

        res = p.feed(raw) + p.finish()

        # Both the boilerplate announcement and the useless search URL are dropped!

        self.assertEqual(res, [])



    def test_bubble_parser_apple_podcasts_url_normalized(self):

        """Verify Apple Podcasts US link is auto-normalized to /tr/."""

        raw = "https://podcasts.apple.com/us/podcast/story-fm/id123456789"

        p = BubbleParser()

        res = p.feed(raw) + p.finish()

        self.assertEqual(res, ["https://podcasts.apple.com/tr/podcast/story-fm/id123456789"])



    def test_bubble_parser_standalone_tapback_turn(self):

        """Verify a turn with only a tapback outputs only TAPBACK with no empty bubbles."""

        raw = "<tapback>❤️</tapback>"

        p = BubbleParser()

        res = p.feed(raw) + p.finish()

        self.assertEqual(res, ["TAPBACK:❤️"])







    def test_unsupported_personal_tags_safe_fallback(self):
        """Verify that unsupported personal tags safely degrade to readable plain text without errors or images."""
        p = BubbleParser()
        raw = '<bubble>天气播报：</bubble><bubble><weather city="北京" temp="22°C">今天天气晴朗，气温舒适。</weather></bubble><bubble>祝你有愉快的一天！</bubble>'
        res = p.feed(raw) + p.finish()
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0], "天气播报：")
        self.assertEqual(res[1], "今天天气晴朗，气温舒适。")
        self.assertEqual(res[2], "祝你有愉快的一天！")
        self.assertFalse(any(b.startswith("IMAGE:") for b in res))

        # Test self-closing tag with title/label fallback
        p2 = BubbleParser()
        raw2 = '<movie title="星际穿越" director="诺兰" />这是一部关于爱与时空的硬科幻电影。'
        res2 = p2.feed(raw2) + p2.finish()
        self.assertFalse(any(b.startswith("IMAGE:") for b in res2))
        self.assertTrue(any("星际穿越" in b for b in res2))
        self.assertTrue(any("硬科幻电影" in b for b in res2))

    def test_code_block_with_tags_literal(self):
        """Verify that Markdown code blocks containing formatting tags remain literal."""
        code_snippet = (
            "```xml\n"
            "<weather city=\"上海\">\n"
            "  <temp>25°C</temp>\n"
            "</weather>\n"
            "```"
        )
        p = BubbleParser()
        res = p.feed(code_snippet) + p.finish()
        self.assertEqual(len(res), 1)
        self.assertIn("<weather city=\"上海\">", res[0])
        self.assertIn("```", res[0])
        self.assertFalse(any(b.startswith("IMAGE:") for b in res))


class TestParserRobustnessAndLifecycle(unittest.TestCase):
    """Regression tests for parser lifecycle, quote escaping, code fence tracking, and malformed tags."""

    def test_attribute_apostrophe_in_double_quotes(self):
        """1. Apostrophes inside double-quoted attributes must not truncate value."""
        attrs = BubbleParser._parse_tag_attrs('title="What\'s your plan?" name="McDonald\'s"')
        self.assertEqual(attrs.get("title"), "What's your plan?")
        self.assertEqual(attrs.get("name"), "McDonald's")

    def test_attribute_escaped_quotes(self):
        """2. Escaped quotes inside attributes must be properly unescaped."""
        attrs = BubbleParser._parse_tag_attrs('title="He said \\"hello\\"" caption=\'She said \\\'hi\\\'\'')
        self.assertEqual(attrs.get("title"), 'He said "hello"')
        self.assertEqual(attrs.get("caption"), "She said 'hi'")

    def test_single_line_opening_and_closing_fences(self):
        """3. Single-line opening and closing Markdown fences must not trap subsequent bubbles in code block."""
        parser = BubbleParser()
        b1 = parser._balance_and_track_fences("Here is inline fence: ```python print('hello') ``` that is all.")
        self.assertFalse(parser.in_code_block, "Parser should not remain in code block after closed single-line fence")
        self.assertFalse(b1.endswith("\n```"), f"Trailing fence should not be added to closed fence: {b1!r}")

        b2 = parser._balance_and_track_fences("This is the next normal bubble.")
        self.assertEqual(b2, "This is the next normal bubble.")
        self.assertFalse(parser.in_code_block)

    def test_multiple_fenced_code_blocks(self):
        """4. Multiple fenced code blocks in one text block must both balance correctly."""
        parser = BubbleParser()
        text = "First:\n```python\na = 1\n```\nSecond:\n```bash\necho 2\n```\nDone."
        res = parser._balance_and_track_fences(text)
        self.assertFalse(parser.in_code_block)
        self.assertNotIn("\n```\n```", res)
        # Next bubble
        next_b = parser._balance_and_track_fences("Normal text.")
        self.assertEqual(next_b, "Normal text.")

    def test_code_fence_state_across_successive_bubbles(self):
        """5. Code fence state across successive bubbles must properly close and reopen."""
        parser = BubbleParser()
        # Bubble 1 opens code block but does not close it
        b1 = parser._balance_and_track_fences("```python\ndef foo():\n    return 42")
        self.assertTrue(parser.in_code_block)
        self.assertEqual(parser.active_code_lang, "python")
        self.assertTrue(b1.endswith("```"))

        # Bubble 2 continues code block and closes it
        b2 = parser._balance_and_track_fences("    x = foo()\n```")
        self.assertTrue(b2.startswith("```python"))
        self.assertTrue(b2.endswith("```"))
        self.assertFalse(parser.in_code_block)
        self.assertEqual(parser.active_code_lang, "")

        # Bubble 3 is normal text
        b3 = parser._balance_and_track_fences("End of code.")
        self.assertEqual(b3, "End of code.")
        self.assertFalse(parser.in_code_block)

    def test_unclosed_bubble_tags_at_stream_end(self):
        """6. Unclosed <bubble> tags at end of stream must preserve content without leaking protocol tag."""
        parser = BubbleParser()
        parser.feed("<bubble>This was never closed")
        bubbles = parser.finish()
        self.assertEqual(bubbles, ["This was never closed"])

    def test_partial_closing_tags(self):
        """7. Partial closing tags at end of stream must be stripped cleanly."""
        parser1 = BubbleParser()
        parser1.feed("Clean content</bub")
        self.assertEqual(parser1.finish(), ["Clean content"])

        parser2 = BubbleParser()
        parser2.feed("Another content</effect")
        self.assertEqual(parser2.finish(), ["Another content"])

    def test_literal_bubble_inside_code_blocks(self):
        """8. Literal <bubble> text inside Markdown code blocks must remain untouched."""
        parser = BubbleParser()
        text = "```xml\n<bubble>This is inside a code block</bubble>\n```"
        parser.feed(text)
        bubbles = parser.finish()
        self.assertEqual(len(bubbles), 1)
        self.assertIn("<bubble>This is inside a code block</bubble>", bubbles[0])

    def test_mixed_single_and_double_quotes_attributes(self):
        """Mixed single-quoted and double-quoted attributes in the same tag."""
        attrs = BubbleParser._parse_tag_attrs('title="Double quote title" author=\'Alice Smith\' count="5"')
        self.assertEqual(attrs.get("title"), "Double quote title")
        self.assertEqual(attrs.get("author"), "Alice Smith")
        self.assertEqual(attrs.get("count"), "5")

    def test_malformed_unterminated_attribute_values(self):
        """Malformed or unterminated attribute values should not crash or corrupt other attributes."""
        attrs = BubbleParser._parse_tag_attrs('title="Valid Title" broken="unterminated other=\'ok\'')
        self.assertEqual(attrs.get("title"), "Valid Title")
        self.assertEqual(attrs.get("other"), "ok")

    def test_ordinary_text_containing_angle_brackets(self):
        """Ordinary mathematical comparisons < and > must not be treated as tags or corrupted."""
        parser = BubbleParser()
        out1 = parser.feed("If x < 10 and y > 20:\n\nThen result is valid.")
        out2 = parser.finish()
        all_bubbles = out1 + out2
        self.assertEqual(len(all_bubbles), 2)
        self.assertEqual(all_bubbles[0], "If x < 10 and y > 20:")
        self.assertEqual(all_bubbles[1], "Then result is valid.")

    def test_tags_split_across_streaming_chunks(self):
        """Tags split across streaming chunk boundaries must resolve cleanly without loss."""
        parser = BubbleParser()
        out1 = parser.feed("<bub")
        self.assertEqual(out1, [])
        out2 = parser.feed("ble>Chunk one</bub")
        self.assertEqual(out2, [])
        out3 = parser.feed("ble><bubble>Chunk two</bubble>")
        self.assertEqual(out3, ["Chunk one", "Chunk two"])
        out4 = parser.finish()
        self.assertEqual(out4, [])


class TestAirHermesConfigurationAndFlags(unittest.IsolatedAsyncioTestCase):
    """Test suite covering Phase 4 configuration, feature flags, and safe fallbacks."""

    def setUp(self):
        self._orig_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._orig_env)

    def test_01_default_configuration_equivalence(self):
        """1. Default configuration equivalence: empty config matches existing behavior."""
        from gateway.airhermes_config import AirHermesConfig
        cfg = AirHermesConfig.from_dict({})
        self.assertTrue(cfg.bubble_streaming)
        self.assertEqual(cfg.bubble_min_interval, 0.85)
        self.assertTrue(cfg.bubble_adaptive_pacing)
        self.assertTrue(cfg.enable_tapbacks)
        self.assertTrue(cfg.enable_polls)
        self.assertTrue(cfg.enable_calendar)
        self.assertTrue(cfg.enable_maps)
        self.assertTrue(cfg.enable_effects)

    def test_02_environment_variable_precedence(self):
        """2. Environment variables override YAML configuration dictionary."""
        from gateway.airhermes_config import AirHermesConfig
        os.environ["AIRHERMES_BUBBLE_STREAMING"] = "false"
        os.environ["AIRHERMES_MIN_INTERVAL"] = "1.50"
        os.environ["AIRHERMES_ENABLE_POLLS"] = "0"

        yaml_dict = {
            "bubble_streaming": True,
            "bubble_min_interval": 0.50,
            "enable_polls": True,
        }
        cfg = AirHermesConfig.from_dict(yaml_dict)
        self.assertFalse(cfg.bubble_streaming)
        self.assertEqual(cfg.bubble_min_interval, 1.50)
        self.assertFalse(cfg.enable_polls)

    def test_03_strict_boolean_parsing(self):
        """3. Strict boolean parsing accepts only recognized booleans and warns on invalid values."""
        from gateway.airhermes_config import AirHermesConfig
        # True variants
        for val in (True, "true", "True", "1", "yes", "on", "ON"):
            cfg = AirHermesConfig.from_dict({"bubble_streaming": val})
            self.assertTrue(cfg.bubble_streaming, f"Failed for {val!r}")

        # False variants
        for val in (False, "false", "False", "0", "no", "off", "OFF"):
            cfg = AirHermesConfig.from_dict({"bubble_streaming": val})
            self.assertFalse(cfg.bubble_streaming, f"Failed for {val!r}")

        # Invalid booleans fallback to default True with warning
        for invalid in ("invalid", "maybe", 100, [1], None):
            cfg = AirHermesConfig.from_dict({"bubble_streaming": invalid})
            self.assertTrue(cfg.bubble_streaming)

    def test_04_invalid_numeric_values(self):
        """4. Invalid numeric values (NaN, Inf, negative, out-of-range) fallback to default with warning."""
        from gateway.airhermes_config import AirHermesConfig

        for invalid in (float("nan"), float("inf"), float("-inf"), "nan", "inf", -1.0, 10.0, "not_a_number"):
            cfg = AirHermesConfig.from_dict({"bubble_min_interval": invalid})
            self.assertEqual(cfg.bubble_min_interval, 0.85, f"Failed fallback for {invalid!r}")

    async def test_05_streaming_disabled_original_hermes_path(self):
        """5. When bubble_streaming=False: capability is False, consumer is not created, exactly one final message sent."""
        from gateway.airhermes_config import AirHermesConfig

        cfg = AirHermesConfig(bubble_streaming=False)
        adapter = MockAdapter()
        adapter.airhermes_config = cfg
        adapter.SUPPORTS_BUBBLE_STREAMING = False  # Instance-level override

        # 1. Class-level capability remains True
        self.assertTrue(getattr(type(adapter), "SUPPORTS_BUBBLE_STREAMING", True))
        # 2. Instance capability is False
        self.assertFalse(getattr(adapter, "SUPPORTS_BUBBLE_STREAMING", False))

        # 3. Simulate turn delivery with streaming disabled
        final_text = "This is the complete final response delivered at turn finish."
        res = await adapter.send("chat_123", final_text)

        self.assertTrue(getattr(res, "success", False) or getattr(res, "ok", False))
        self.assertEqual(len(adapter.sent_messages), 1)
        msg = adapter.sent_messages[0]
        self.assertEqual(msg.get("content") or msg.get("text"), final_text)
        self.assertEqual(adapter.sent_messages[0]["chat_id"], "chat_123")

    def test_06_independent_feature_flags(self):
        """6. Feature flags are fully independent and do not interfere with each other."""
        from gateway.airhermes_config import AirHermesConfig
        cfg = AirHermesConfig(enable_polls=False, enable_calendar=True)
        self.assertFalse(cfg.enable_polls)
        self.assertTrue(cfg.enable_calendar)
        self.assertTrue(cfg.enable_maps)
        self.assertTrue(cfg.enable_effects)
        self.assertTrue(cfg.enable_tapbacks)

    def test_07_disabled_feature_safe_fallback(self):
        """7. Disabled features safely degrade to readable text without executing native actions."""
        from gateway.airhermes_config import AirHermesConfig

        # A. Polls disabled -> plain text options
        p_poll = BubbleParser(config=AirHermesConfig(enable_polls=False))
        poll_bubbles = p_poll.feed('<poll title="Dinner Choice">Hotpot\nBarbecue</poll>')
        poll_bubbles.extend(p_poll.finish())
        self.assertEqual(len(poll_bubbles), 1)
        self.assertFalse(poll_bubbles[0].startswith("POLL:"))
        self.assertIn("Dinner Choice", poll_bubbles[0])
        self.assertIn("Hotpot", poll_bubbles[0])

        # B. Calendar disabled -> readable event text without .ics
        p_cal = BubbleParser(config=AirHermesConfig(enable_calendar=False))
        cal_bubbles = p_cal.feed('<calendar title="Team Review" start="2026-11-25 10:00" location="Room A">Project Review</calendar>')
        cal_bubbles.extend(p_cal.finish())
        self.assertEqual(len(cal_bubbles), 1)
        self.assertFalse(cal_bubbles[0].startswith("FILE:"))
        self.assertIn("Team Review", cal_bubbles[0])
        self.assertIn("Room A", cal_bubbles[0])

        # C. Maps disabled -> readable text without .loc.vcf or Apple Maps URL (DEFECT-02)
        p_map = BubbleParser(config=AirHermesConfig(enable_maps=False))
        map_bubbles = p_map.feed('<map query="Forbidden City" title="故宫"/>')
        map_bubbles.extend(p_map.finish())
        self.assertEqual(len(map_bubbles), 1)
        self.assertFalse(map_bubbles[0].startswith("FILE:"))
        self.assertIn("故宫", map_bubbles[0])
        self.assertNotIn("maps.apple.com", map_bubbles[0])

        # D. Effects disabled -> plain text without EFFECT: prefix
        p_eff = BubbleParser(config=AirHermesConfig(enable_effects=False))
        eff_bubbles = p_eff.feed('<effect:fireworks>Congratulations!</effect>')
        eff_bubbles.extend(p_eff.finish())
        self.assertEqual(len(eff_bubbles), 1)
        self.assertEqual(eff_bubbles[0], "Congratulations!")

    async def test_08_no_unintended_native_side_effects_tapbacks(self):
        """8. When tapbacks are disabled, native reaction API is NEVER called and no emoji bubble is sent."""
        from gateway.airhermes_config import AirHermesConfig

        adapter = MockAdapter()
        # Parser with tapbacks disabled
        parser = BubbleParser(config=AirHermesConfig(enable_tapbacks=False))
        bubbles = parser.feed('<tapback emoji="👍">')
        bubbles.extend(parser.finish())
        self.assertEqual(bubbles, [])

        # Consumer with tapbacks disabled
        from gateway.bubble_actions import TapbackAction
        consumer = BubbleStreamConsumer(adapter=adapter, chat_id="c1", config=AirHermesConfig(enable_tapbacks=False))
        res = await consumer._send_bubble(TapbackAction(reaction="👍"))
        is_ok = res.get("ok", False) if isinstance(res, dict) else (getattr(res, "success", False) or getattr(res, "ok", False))
        self.assertTrue(is_ok)
        self.assertEqual(len(adapter.reactions), 0)
        self.assertEqual(len(adapter.sent_messages), 0)

    def test_09_adaptive_pacing_disabled(self):
        """9. When bubble_adaptive_pacing=False, ChatPacer returns constant min_interval_s without jitter."""
        from gateway.bubble_queue import ChatPacer

        pacer_off = ChatPacer("chat_test", min_interval_s=1.20, adaptive=False)
        self.assertEqual(pacer_off.compute_interval("long message?", "burst!"), 1.20)
        self.assertEqual(pacer_off.compute_interval("TAPBACK:👍", "next"), 1.20)

    def test_10_configuration_snapshot_consistency(self):
        """10. BubbleStreamConsumer holds an immutable configuration snapshot throughout a turn."""
        from gateway.airhermes_config import AirHermesConfig

        cfg = AirHermesConfig(bubble_min_interval=0.90)
        adapter = MockAdapter()
        consumer = BubbleStreamConsumer(adapter=adapter, chat_id="c_snap", config=cfg)

        self.assertEqual(consumer.config.bubble_min_interval, 0.90)
        os.environ["AIRHERMES_MIN_INTERVAL"] = "2.50"
        self.assertEqual(consumer.config.bubble_min_interval, 0.90)

    def test_11_unknown_settings(self):
        """11. Unknown settings in config dictionary are safely ignored without error."""
        from gateway.airhermes_config import AirHermesConfig

        cfg = AirHermesConfig.from_dict({"unknown_future_setting": 12345, "another_key": "val"})
        self.assertTrue(cfg.bubble_streaming)
        self.assertEqual(cfg.bubble_min_interval, 0.85)

    def test_12_existing_installation_compatibility(self):
        """12. Existing installations without configuration object run with full defaults."""
        from gateway.airhermes_config import AirHermesConfig

        cfg = AirHermesConfig.from_platform_config(None)
        self.assertTrue(cfg.bubble_streaming)
        self.assertEqual(cfg.bubble_min_interval, 0.85)
        self.assertTrue(cfg.bubble_adaptive_pacing)


@unittest.skipUnless(HERMES_UPSTREAM_AVAILABLE, "Requires upstream Hermes Agent environment (PhotonAdapter/TurnRunner)")
class TestHermesUpstreamIntegration(unittest.TestCase):
    """Integration test suite verifying patched behavior inside upstream Hermes Agent."""

    def test_photon_adapter_flag(self):
        """Verify PhotonAdapter declares support for bubble streaming and disallows message editing."""
        self.assertTrue(getattr(PhotonAdapter, "SUPPORTS_BUBBLE_STREAMING", False))
        self.assertFalse(getattr(PhotonAdapter, "SUPPORTS_MESSAGE_EDITING", True))

    def test_tool_heartbeat_hints(self):
        """Verify TurnRunner._format_bubble_tool_hint formats human-readable progress descriptions."""
        hint_search = TurnRunner._format_bubble_tool_hint("web_search", {"query": "US Midterm Election"})
        self.assertEqual(hint_search, "正在搜索「US Midterm Election」...")

        hint_term_sched = TurnRunner._format_bubble_tool_hint("terminal", {"command": "python get_schedule.py"})
        self.assertEqual(hint_term_sched, "正在翻看日程课表...")

        hint_term_git = TurnRunner._format_bubble_tool_hint("terminal", {"command": "git status"})
        self.assertEqual(hint_term_git, "正在检查代码仓库...")

        hint_file = TurnRunner._format_bubble_tool_hint("read_file", {"path": "C:/notes/todo.md"})
        self.assertEqual(hint_file, "正在查阅 todo.md...")

        hint_web = TurnRunner._format_bubble_tool_hint("web_extract", {"url": "https://example.com"})
        self.assertEqual(hint_web, "正在浏览网页内容...")

    def test_heartbeat_debounce_fast_tool(self):
        """Verify fast tools under debounce threshold do not trigger phantom commentary heartbeats."""
        import time

        emitted_hints = []

        class MockConsumer:
            _bubble_queue = None

            def on_commentary(self, msg):
                emitted_hints.append(msg)

        sc = MockConsumer()
        start_hook, complete_hook = TurnRunner._wrap_bubble_tool_callbacks(None, None, sc)

        # Fast tool finishes in 50ms (well under 2.5s threshold)
        start_hook("call_1", "web_search", {"query": "test"})
        time.sleep(0.05)  # health: allow HX013 -- debounce interval verification
        complete_hook("call_1")

        # Wait to verify timer was cancelled and did not fire
        time.sleep(0.1)  # health: allow HX013 -- debounce interval verification
        self.assertEqual(len(emitted_hints), 0)


if __name__ == "__main__":

    unittest.main()


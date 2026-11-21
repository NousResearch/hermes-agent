"""Unit and integration tests for Photon discrete multi-bubble streaming."""



import asyncio
import json
import os
import unittest

from typing import Any, Dict, List, Optional



from gateway.bubble_parser import BubbleParser

from gateway.bubble_queue import BubbleDeliveryState, ChatPacerRegistry, QueueState

from gateway.bubble_stream_consumer import BubbleStreamConsumer

from plugins.platforms.photon.adapter import PhotonAdapter





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

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import tempfile
        cls._test_dir = tempfile.mkdtemp(prefix="hermes_test_cards_")
        cls._dummy_files = {}
        for kind in ("weather", "movie", "word", "sheet"):
            p = os.path.join(cls._test_dir, f"{kind}_test_retina.png")
            with open(p, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 1024)
            cls._dummy_files[kind] = p

        def _mock_card(card_type: str, attrs: dict, body: str = "") -> str:
            return cls._dummy_files.get(card_type, cls._dummy_files["weather"])

        def _mock_sheet(body: str, title: str = "", subtitle: str = "") -> str:
            return cls._dummy_files["sheet"]

        cls._orig_render_card = BubbleParser._render_card_image
        cls._orig_render_sheet = BubbleParser._render_sheet_image
        BubbleParser._render_card_image = staticmethod(_mock_card)
        BubbleParser._render_sheet_image = staticmethod(_mock_sheet)

    @classmethod
    def tearDownClass(cls):
        BubbleParser._render_card_image = cls._orig_render_card
        BubbleParser._render_sheet_image = cls._orig_render_sheet
        import shutil
        if hasattr(cls, "_test_dir") and os.path.exists(cls._test_dir):
            shutil.rmtree(cls._test_dir, ignore_errors=True)
        super().tearDownClass()

    async def asyncSetUp(self):

        ChatPacerRegistry.reset()



    def test_photon_adapter_flag(self):

        self.assertTrue(getattr(PhotonAdapter, "SUPPORTS_BUBBLE_STREAMING", False))

        self.assertFalse(getattr(PhotonAdapter, "SUPPORTS_MESSAGE_EDITING", True))



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



    def test_tool_heartbeat_hints(self):

        from gateway.run_turn_runner import TurnRunner

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

        import json

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



            self.assertEqual(len(adapter.documents), 1)

            self.assertEqual(adapter.documents[0]["chat_id"], "chat_file_test")

            self.assertEqual(adapter.documents[0]["file_path"], temp_path)

        finally:

            import os

            if os.path.exists(temp_path):

                os.remove(temp_path)



    def test_heartbeat_debounce_fast_tool(self):

        from gateway.run_turn_runner import TurnRunner

        import time



        emitted_hints = []

        class MockConsumer:

            _bubble_queue = None

            def on_commentary(self, msg):

                emitted_hints.append(msg)



        sc = MockConsumer()

        start_hook, complete_hook = TurnRunner._wrap_bubble_tool_callbacks(None, None, sc)



        # Fast tool finishes in 50ms (well under 2.5s)

        start_hook("call_1", "web_search", {"query": "test"})

        time.sleep(0.05)  # health: allow HX013 -- debounce interval verification

        complete_hook("call_1")



        # Wait longer to verify timer was cancelled and did not fire

        time.sleep(0.1)  # health: allow HX013 -- debounce interval verification

        self.assertEqual(len(emitted_hints), 0)



    def test_bubble_parser_effect_extraction(self):

        p = BubbleParser()

        res = p.feed("<effect name=\"confetti\">太棒了！恭喜！🎉</effect>")

        res.extend(p.finish())

        self.assertEqual(len(res), 1)

        self.assertTrue(res[0].startswith("EFFECT:"))

        import json

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

        with open(file_path, "r", encoding="utf-8-sig") as fp:

            content = fp.read()

        self.assertIn("BEGIN:VCALENDAR", content)
        self.assertIn("SUMMARY;CHARSET=UTF-8:Q4复盘会", content)
        self.assertIn("DTSTART:20261106T140000", content)
        self.assertIn("LOCATION;CHARSET=UTF-8:腾讯会议", content)

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



    def test_bubble_parser_auto_poll_guardrail_on_cheat_sheet(self):

        """Verify that when a cheat_sheet is rendered without an explicit poll, the guardrail synthesizes one."""

        raw = (

            "<bubble>重庆周边自驾给你挑了2个方向：</bubble>\n"

            "<cheat_sheet title=\"重庆自驾路线全对比\">\n"

            "| 路线主题 | 亮点 |\n"

            "| :--- | :--- |\n"

            "| **路线A：武隆仙女山** | 高山草场与天坑 |\n"

            "| **路线B：三峡夔门绝壁** | 峡谷天路 |\n"

            "</cheat_sheet>\n"

            "<bubble>你大概想自驾几天？想去哪条线我把具体的途径导航和避坑节点发你～</bubble>"

        )

        p = BubbleParser()

        res = p.feed(raw) + p.finish()

        self.assertEqual(len(res), 4)

        self.assertEqual(res[0], "重庆周边自驾给你挑了2个方向：")

        self.assertTrue(res[1].startswith("IMAGE:"))

        self.assertEqual(res[2], "你大概想自驾几天？想去哪条线我把具体的途径导航和避坑节点发你～")

        self.assertTrue(res[3].startswith("POLL:"))

        poll_data = json.loads(res[3][len("POLL:"):])

        self.assertEqual(poll_data["title"], "想先看哪条路线的行程细节？")

        self.assertEqual(poll_data["options"], ["路线A：武隆仙女山", "路线B：三峡夔门绝壁"])



    def test_bubble_parser_auto_poll_guardrail_skipped_when_poll_exists(self):

        """Verify that when an explicit poll is already emitted, auto-poll does not duplicate."""

        raw = (

            "<cheat_sheet title=\"自驾对比\">\n"

            "| 路线 | 亮点 |\n"

            "| :--- | :--- |\n"

            "| 路线A | 亮点A |\n"

            "| 路线B | 亮点B |\n"

            "</cheat_sheet>\n"

            "<poll title=\"你更想选哪个？\"><option>路线A</option><option>路线B</option></poll>"

        )

        p = BubbleParser()

        res = p.feed(raw) + p.finish()

        # IMAGE: and POLL:

        self.assertEqual(len(res), 2)

        self.assertTrue(res[0].startswith("IMAGE:"))

        self.assertTrue(res[1].startswith("POLL:"))

        poll_data = json.loads(res[1][len("POLL:"):])

        self.assertEqual(poll_data["title"], "你更想选哪个？")



    def test_bubble_parser_auto_poll_guardrail_skipped_on_sequential_itinerary(self):

        """Verify that sequential/itinerary tables (第一站, 14:30) do NOT synthesize a poll."""

        raw = (

            "<bubble>路线A的自驾规划如下：</bubble>\n"

            "<cheat_sheet title=\"路线A：停靠规划与精选打卡点\">\n"

            "| 阶段与时间 | 停靠节点 | 核心体验 |\n"

            "| :--- | :--- | :--- |\n"

            "| **第一站：14:30 - 15:30** | 格力海岸 | 海边散步 |\n"

            "| **第二站：15:45 - 17:00** | 日月贝 | 贝壳建筑 |\n"

            "</cheat_sheet>\n"

            "<bubble>出发集合点给你备好啦</bubble>"

        )

        p = BubbleParser()

        res = p.feed(raw) + p.finish()

        # Should only have bubble, IMAGE, bubble -- NO POLL!

        self.assertEqual(len(res), 3)

        self.assertEqual(res[0], "路线A的自驾规划如下：")

        self.assertTrue(res[1].startswith("IMAGE:"))

        self.assertEqual(res[2], "出发集合点给你备好啦")





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







    def test_bubble_parser_weather_tag_streaming(self):
        """Verify <weather> tag renders retina capsule image bubble during streaming."""
        chunks = [
            "<bubble>今天北京降温了哈</bubble>",
            '<weather location="北京 · 朝阳" condition="rainy" temp="17°" feel="14°C" high_low="11° ~ 19°C" pop="85%" advice="出门带伞加件风衣" />',
            "<bubble>出门记得保暖~</bubble>",
        ]
        p = BubbleParser()
        res = []
        for c in chunks:
            res.extend(p.feed(c))
        res.extend(p.finish())

        self.assertEqual(len(res), 3)
        self.assertEqual(res[0], "今天北京降温了哈")
        self.assertTrue(res[1].startswith("IMAGE:"))
        self.assertIn("weather_", res[1])
        self.assertIn(".png", res[1])
        self.assertEqual(res[2], "出门记得保暖~")

    def test_bubble_parser_movie_tag_streaming(self):
        """Verify <movie> tag renders retina review card during streaming."""
        chunks = [
            "<bubble>强烈推荐这部诺兰神作</bubble>",
            '<movie title="奥本海默" year="2023" douban="8.8" imdb="8.9" director="克里斯托弗·诺兰" style="classic" quote="“普罗米修斯盗取了天火...”" />',
            "<bubble>看完后劲特别大</bubble>",
        ]
        p = BubbleParser()
        res = []
        for c in chunks:
            res.extend(p.feed(c))
        res.extend(p.finish())

        self.assertEqual(len(res), 3)
        self.assertEqual(res[0], "强烈推荐这部诺兰神作")
        self.assertTrue(res[1].startswith("IMAGE:"))
        self.assertIn("movie_", res[1])
        self.assertIn(".png", res[1])
        self.assertEqual(res[2], "看完后劲特别大")

    def test_bubble_parser_word_tag_streaming(self):
        """Verify <word> tag renders signature Klein Blue retina card during streaming."""
        chunks = [
            "<bubble>这个词的语感非常美</bubble>",
            '<word word="serendipity" phonetic="英 [ˌserənˈdɪpəti]" pos="n. 名词" meaning="不期而遇的惊喜与美好收获" roots="Serendip + -ity" />',
            "<bubble>在文学作品里经常出现</bubble>",
        ]
        p = BubbleParser()
        res = []
        for c in chunks:
            res.extend(p.feed(c))
        res.extend(p.finish())

        self.assertEqual(len(res), 3)
        self.assertEqual(res[0], "这个词的语感非常美")
        self.assertTrue(res[1].startswith("IMAGE:"))
        self.assertIn("word_", res[1])
        self.assertIn(".png", res[1])
        self.assertEqual(res[2], "在文学作品里经常出现")

    def test_bubble_parser_visual_cards_boilerplate_pruned(self):
        """Verify robotic announcements for weather, movie, and word cards are pruned."""
        p = BubbleParser()
        raw = '天气胶囊给你生成好啦：<weather location="北京" condition="sunny" temp="22°" />'
        res = p.feed(raw) + p.finish()
        self.assertEqual(len(res), 1)
        self.assertTrue(res[0].startswith("IMAGE:"))

        p2 = BubbleParser()
        raw2 = '🎬 电影卡片给你备好啦：<movie title="星际穿越" />'
        res2 = p2.feed(raw2) + p2.finish()
        self.assertEqual(len(res2), 1)
        self.assertTrue(res2[0].startswith("IMAGE:"))

        p3 = BubbleParser()
        raw3 = '单词卡片已生成！<word word="epiphany" />'
        res3 = p3.feed(raw3) + p3.finish()
        self.assertEqual(len(res3), 1)
        self.assertTrue(res3[0].startswith("IMAGE:"))


    def test_bubble_parser_weather_duplicate_cheat_sheet_suppressed_batch(self):
        """Verify that when a response contains both <weather> and a weather <cheat_sheet>, only the single weather card is emitted."""
        raw_msg = (
            '<weather location="珠海·斗门" condition="rainy" condition_text="阴天伴零星微雨" temp="25°" feel="22°C" high_low="25° ~ 26°C" pop="25%" humidity="81%" wind="偏东风 5-6级" />\n\n'
            '<cheat_sheet title="珠海·斗门 今日天气概览">\n'
            '| 指标 | 详情 | 说明 |\n'
            '| :--- | :--- | :--- |\n'
            '| **实时气温 / 体感** | **25°C / 22°C** | 偏东海风 5~6 级 |\n'
            '| **降水概率** | 25% | 零星小雨 |\n'
            '</cheat_sheet>\n\n'
            '今天主要是阴天加吹大风，凉飕飕的很舒服～'
        )
        p = BubbleParser()
        bubbles = p.feed(raw_msg) + p.finish()
        image_bubbles = [b for b in bubbles if b.startswith("IMAGE:")]
        self.assertEqual(len(image_bubbles), 1, f"Expected exactly 1 image bubble, got {len(image_bubbles)}")
        self.assertIn("weather_", image_bubbles[0])
        # Verify text chat bubble is preserved
        text_bubbles = [b for b in bubbles if not b.startswith("IMAGE:")]
        self.assertTrue(any("今天主要是阴天" in b for b in text_bubbles))

    def test_bubble_parser_weather_duplicate_cheat_sheet_suppressed_streaming(self):
        """Verify that streaming input with both <weather> and <cheat_sheet> suppresses the duplicate cheat sheet image and poll."""
        chunks = [
            '<weather location="珠海·斗门" condition="rainy" condition_text="阴天伴零星微雨" temp="25°" feel="22°C" />',
            '\n\n<cheat_sheet title="珠海·斗门 今日天气概览">\n',
            '| 指标 | 详情 |\n| :--- | :--- |\n| 实时气温 | 25°C |\n',
            '</cheat_sheet>\n\n',
            '今天主要是阴天加吹大风。',
        ]
        p = BubbleParser()
        res = []
        for c in chunks:
            res.extend(p.feed(c))
        res.extend(p.finish())

        image_bubbles = [b for b in res if b.startswith("IMAGE:")]
        poll_bubbles = [b for b in res if b.startswith("POLL:")]
        self.assertEqual(len(image_bubbles), 1, f"Expected exactly 1 image bubble in stream, got {len(image_bubbles)}")
        self.assertIn("weather_", image_bubbles[0])
        self.assertEqual(len(poll_bubbles), 0, "No auto-poll should be emitted for suppressed weather cheat sheet")
        self.assertTrue(any("今天主要是阴天" in b for b in res))

    def test_bubble_parser_non_weather_cheat_sheet_preserved(self):
        """Verify that a legitimate non-weather cheat_sheet (e.g. smartphone specs) is NOT suppressed."""
        raw_msg = (
            '<cheat_sheet title="旗舰手机参数对比">\n'
            '| 品牌 | 芯片 | 屏幕 |\n'
            '| :--- | :--- | :--- |\n'
            '| Apple | A18 Pro | 6.3 OLED |\n'
            '| Google | Tensor G4 | 6.3 OLED |\n'
            '</cheat_sheet>\n\n'
            '两款机型综合素质都非常出色。'
        )
        p = BubbleParser()
        bubbles = p.feed(raw_msg) + p.finish()
        image_bubbles = [b for b in bubbles if b.startswith("IMAGE:")]
        self.assertEqual(len(image_bubbles), 1)
        self.assertIn("sheet_", image_bubbles[0])


    def test_bubble_parser_paragraph_fallback_after_visual_tag(self):
        """Verify that multiple paragraphs following a visual tag are split into separate bubbles rather than dumped into a single block."""
        raw_msg = (
            '<movie title="末代皇帝" original_title="The Last Emperor" />\n\n'
            '史诗级神作，任何对近代史感兴趣的人绝对必看。\n\n'
            '贝托鲁奇镜头下的故宫没有现代横店的塑料感，光影、红墙都带着沉甸甸的历史苍凉感。\n\n'
            '挑个悠闲的下午静下心来看最过瘾！'
        )
        p = BubbleParser()
        bubbles = p.feed(raw_msg) + p.finish()

        image_bubbles = [b for b in bubbles if b.startswith("IMAGE:")]
        text_bubbles = [b for b in bubbles if not b.startswith("IMAGE:")]

        self.assertEqual(len(image_bubbles), 1)
        self.assertEqual(len(text_bubbles), 3, f"Expected 3 separate paragraph bubbles, got {len(text_bubbles)}: {text_bubbles}")
        self.assertIn("史诗级神作", text_bubbles[0])
        self.assertIn("贝托鲁奇镜头", text_bubbles[1])
        self.assertIn("挑个悠闲的下午", text_bubbles[2])


if __name__ == "__main__":

    unittest.main()


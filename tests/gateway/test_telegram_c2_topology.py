from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from gateway.platforms.base import SendResult
from gateway.telegram_topology import (
    operator_target_for_command,
    topic_contract_for_source,
    validate_telegram_topology,
)


CHAT_ID = "-1004298945366"


def valid_topology() -> dict:
    return {
        "version": 1,
        "platform": "telegram",
        "chat_id": CHAT_ID,
        "topics": {
            "operator": {"title": "💬 Оператор", "thread_id": "2654"},
            "briefings": {
                "title": "☀️ Брифінги",
                "thread_id": "2657",
                "contract": "BRIEFING MODE: concise daily status only.",
            },
            "alerts": {"title": "🔔 Алерти", "thread_id": "2661"},
            "input": {"title": "🎙 Вхід", "thread_id": "2664"},
            "reviews": {"title": "📊 Огляди", "thread_id": "2668"},
            "content": {"title": "✍️ Контент", "thread_id": "2671"},
            "learning": {"title": "📚 Навчання", "thread_id": "2674"},
        },
        "excluded": {"general": [None, "1"], "temporary": ["766"]},
    }


def write_topology(home: Path, data: dict) -> None:
    import yaml

    (home / "telegram_topology.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def source(thread_id):
    return SimpleNamespace(
        platform=SimpleNamespace(value="telegram"),
        chat_id=CHAT_ID,
        thread_id=thread_id,
    )


class TelegramC2TopologyTests(unittest.TestCase):
    def test_valid_topology_loads(self):
        topo = validate_telegram_topology(valid_topology())
        operator = topo.topic("operator")
        self.assertIsNotNone(operator)
        self.assertEqual(len(topo.topics), 7)
        self.assertEqual(operator.thread_id, "2654")
        self.assertEqual(len({topic.thread_id for topic in topo.topics.values()}), 7)
        self.assertEqual(
            topo.topic("briefings").contract,
            "BRIEFING MODE: concise daily status only.",
        )

    def test_topic_contract_matches_exact_telegram_chat_and_thread(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            write_topology(home, valid_topology())
            self.assertEqual(
                topic_contract_for_source(source("2657"), home=home),
                "BRIEFING MODE: concise daily status only.",
            )

    def test_topic_contract_ignores_general_unknown_and_other_chat(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            write_topology(home, valid_topology())
            self.assertIsNone(topic_contract_for_source(source("1"), home=home))
            self.assertIsNone(topic_contract_for_source(source("9999"), home=home))
            other = source("2657")
            other.chat_id = "-1000000000000"
            self.assertIsNone(topic_contract_for_source(other, home=home))

    def test_missing_topic_fails(self):
        data = valid_topology()
        data["topics"].pop("learning")
        with self.assertRaisesRegex(ValueError, "missing required"):
            validate_telegram_topology(data)

    def test_duplicate_id_fails(self):
        data = valid_topology()
        data["topics"]["learning"]["thread_id"] = "2654"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_telegram_topology(data)

    def test_excluded_id_in_main_topic_fails(self):
        data = valid_topology()
        data["topics"]["learning"]["thread_id"] = "766"
        with self.assertRaisesRegex(ValueError, "excluded"):
            validate_telegram_topology(data)

    def test_briefings_slash_source_maps_to_operator(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            write_topology(home, valid_topology())
            self.assertEqual(
                operator_target_for_command(source("2657"), home=home),
                {"chat_id": CHAT_ID, "thread_id": "2654"},
            )

    def test_operator_slash_source_no_redirect(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            write_topology(home, valid_topology())
            self.assertIsNone(operator_target_for_command(source("2654"), home=home))

    def test_reviews_status_source_maps_to_operator(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            write_topology(home, valid_topology())
            self.assertEqual(
                operator_target_for_command(source("2668"), home=home),
                {"chat_id": CHAT_ID, "thread_id": "2654"},
            )

    def test_root_general_temporary_no_redirect(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            write_topology(home, valid_topology())
            for thread_id in (None, "1", "766"):
                self.assertIsNone(operator_target_for_command(source(thread_id), home=home))

    def test_non_telegram_no_redirect(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            write_topology(home, valid_topology())
            non_telegram = SimpleNamespace(
                platform=SimpleNamespace(value="discord"),
                chat_id=CHAT_ID,
                thread_id="2657",
            )
            self.assertIsNone(operator_target_for_command(non_telegram, home=home))

    def test_non_command_unknown_unauthorized_and_hook_handled_do_not_mark_before_guards(self):
        run_py = Path("gateway/run.py").read_text(encoding="utf-8")
        marker = 'event.metadata["telegram_final_response_target"] = target'
        self.assertIn(marker, run_py)
        marker_pos = run_py.index(marker)
        self.assertLess(run_py.index("_denied = self._check_slash_access(source, canonical)"), marker_pos)
        self.assertLess(run_py.index('if decision == "deny":'), marker_pos)
        self.assertLess(run_py.index('if decision == "handled":'), marker_pos)
        self.assertGreater(run_py.index('if canonical == "new":'), marker_pos)
        self.assertIn("if command and canonical and is_gateway_known_command(canonical):", run_py)

    def test_command_route_logging_is_metadata_only_statically(self):
        run_py = Path("gateway/run.py").read_text(encoding="utf-8")
        block = run_py[
            run_py.index("telegram_command_route command=%s"):
            run_py.index('if canonical == "new":')
        ]
        self.assertIn("canonical", block)
        self.assertIn("source_thread", block)
        self.assertIn("final_thread", block)
        self.assertIn("redirect_applied", block)
        for forbidden in ("event.text", "raw_args", "user_name", "username", "token"):
            self.assertNotIn(forbidden, block.lower())

    def test_active_session_status_bypass_honors_final_target_statically(self):
        base_py = Path("gateway/platforms/base.py").read_text(encoding="utf-8")
        block = base_py[
            base_py.index("Command '/%s' bypassing active-session guard"):
            base_py.index("# Clarify reply bypass")
        ]
        self.assertIn('"telegram_final_response_target"', block)
        self.assertIn('_final_thread_meta["thread_id"] = _target_thread', block)
        send_block = block[block.index("_r = await self._send_with_retry") :]
        self.assertIn("chat_id=_final_chat_id", send_block)
        self.assertIn("reply_to=_final_reply_anchor", send_block)
        self.assertIn('_final_thread_meta.pop("telegram_reply_to_message_id", None)', block)
        self.assertNotIn("chat_id=event.source.chat_id", send_block)

    def test_telegram_final_send_logging_is_metadata_only_statically(self):
        adapter_py = Path("plugins/platforms/telegram/adapter.py").read_text(encoding="utf-8")
        self.assertIn("telegram_final_send target_chat=%s target_thread=%s", adapter_py)
        for sentinel in (
            "PRIVATE_TEXT_SENTINEL_DO_NOT_LOG",
            "123456:ABCDEF_TOKEN_SENTINEL_DO_NOT_LOG",
        ):
            self.assertNotIn(sentinel, adapter_py)
        log_pos = adapter_py.find("telegram_final_send")
        window = adapter_py[log_pos:log_pos + 1200].lower()
        for forbidden in ("caption", "raw_update", "username", "token"):
            self.assertNotIn(forbidden, window)

    def test_handler_exception_error_remains_source_topic_statically(self):
        base_py = Path("gateway/platforms/base.py").read_text(encoding="utf-8")
        bg_start = base_py.index("async def _process_message_background")
        final_pos = base_py.index("_final_target =", bg_start)
        first_exception_after_final = base_py.index("except Exception as e:", final_pos)
        error_block = base_py[first_exception_after_final:base_py.index("finally:", first_exception_after_final)]
        self.assertIn("event.source.chat_id", error_block)
        self.assertNotIn("_final_chat_id", error_block)

    def test_text_and_attachments_use_one_final_target_statically(self):
        base_py = Path("gateway/platforms/base.py").read_text(encoding="utf-8")
        bg_start = base_py.index("async def _process_message_background")
        block = base_py[
            base_py.index("_final_target =", bg_start):
            base_py.index("# Determine overall success for the processing hook", bg_start)
        ]
        self.assertIn('_final_thread_metadata["thread_id"] = _target_thread', block)
        self.assertIn('_final_thread_metadata.pop("telegram_reply_to_message_id", None)', block)
        self.assertIn('_final_thread_metadata.pop("reply_to_message_id", None)', block)
        self.assertIn("reply_to=_final_reply_anchor", block)
        for snippet in (
            "chat_id=_final_chat_id,\n                            audio_path=_tts_path",
            "chat_id=_final_chat_id,\n                        content=text_content",
            "chat_id=_final_chat_id,\n                            images=images",
            "chat_id=_final_chat_id,\n                            images=_batch",
            "chat_id=_final_chat_id,\n                                audio_path=media_path",
            "chat_id=_final_chat_id,\n                                video_path=media_path",
            "chat_id=_final_chat_id,\n                                file_path=media_path",
            "chat_id=_final_chat_id,\n                                video_path=file_path",
            "chat_id=_final_chat_id,\n                                file_path=file_path",
            "chat_id=_final_chat_id,\n                            message_id=result.message_id",
        ):
            self.assertIn(snippet, block)
        self.assertNotIn("chat_id=event.source.chat_id", block)

    def test_tts_images_voice_video_document_thread_metadata_statically(self):
        base_py = Path("gateway/platforms/base.py").read_text(encoding="utf-8")
        bg_start = base_py.index("async def _process_message_background")
        block = base_py[
            base_py.index("_final_target =", bg_start):
            base_py.index("# Determine overall success for the processing hook", bg_start)
        ]
        self.assertGreaterEqual(block.count("metadata=_final_thread_metadata"), 8)
        self.assertIn("send_voice(\n                                chat_id=_final_chat_id", block)
        self.assertIn("send_video(\n                                chat_id=_final_chat_id", block)
        self.assertIn("send_document(\n                                chat_id=_final_chat_id", block)
        self.assertIn("send_multiple_images(\n                            chat_id=_final_chat_id", block)

    def test_plain_text_flag_absent_preserves_rich_markdown_paths_statically(self):
        adapter_py = Path("plugins/platforms/telegram/adapter.py").read_text(encoding="utf-8")
        self.assertIn('plain_text_only = bool((metadata or {}).get("telegram_plain_text_only"))', adapter_py)
        self.assertIn("if not plain_text_only and self._should_attempt_rich", adapter_py)
        self.assertIn("parse_mode=ParseMode.MARKDOWN_V2", adapter_py)


class TelegramPlainTextBehaviorTests(unittest.IsolatedAsyncioTestCase):
    async def test_plain_text_flag_absent_calls_rich_path(self):
        from plugins.platforms.telegram.adapter import TelegramAdapter

        adapter = object.__new__(TelegramAdapter)
        adapter._bot = object()
        adapter._name = "telegram"
        adapter._send_path_degraded = False
        called = {"rich": False}
        adapter._should_attempt_rich = lambda content, metadata=None: True

        async def rich(chat_id, content, reply_to, metadata):
            called["rich"] = True
            self.assertEqual(metadata, {"thread_id": "2654"})
            return SendResult(success=True, message_id="r1")

        adapter._try_send_rich = rich
        result = await adapter.send(CHAT_ID, "**hello**", metadata={"thread_id": "2654"})
        self.assertTrue(result.success)
        self.assertTrue(called["rich"])

    async def test_plain_text_flag_true_no_rich_no_markdown_parse_mode_none(self):
        from plugins.platforms.telegram.adapter import TelegramAdapter

        adapter = object.__new__(TelegramAdapter)
        adapter._name = "telegram"
        adapter._send_path_degraded = False
        adapter._reply_to_mode = "first"
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter._should_attempt_rich = lambda content, metadata=None: self.fail("rich should not be attempted")
        adapter._try_send_rich = lambda *a, **kw: self.fail("rich should not be called")
        adapter.format_message = lambda content: self.fail("Markdown formatting should not be used")
        adapter.truncate_message = lambda content, max_len, len_fn=None: [content]
        adapter._metadata_thread_id = lambda metadata: metadata.get("thread_id")
        adapter._message_thread_id_for_send = lambda thread_id: int(thread_id) if thread_id else None
        adapter._metadata_reply_to_message_id = lambda metadata: metadata.get("telegram_reply_to_message_id")
        adapter._is_private_dm_topic_send = lambda chat_id, thread_id, metadata: False
        adapter._should_thread_reply = lambda reply_to, chunk_index: bool(reply_to)
        adapter._thread_kwargs_for_send = (
            lambda chat_id, thread_id, metadata, **kw: {"message_thread_id": int(thread_id)} if thread_id else {}
        )
        adapter._link_preview_kwargs = lambda: {}
        adapter._notification_kwargs = lambda metadata: {}
        adapter._looks_like_connect_timeout = lambda e: False
        adapter._looks_like_pool_timeout = lambda e: False
        adapter._is_thread_not_found_error = lambda e: False
        adapter._prune_stale_dm_topic_binding = lambda *a, **kw: None

        calls = []

        async def send_message(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(message_id=42)

        adapter._bot = SimpleNamespace(send_message=send_message)
        result = await adapter.send(
            CHAT_ID,
            "**hello**",
            metadata={"thread_id": "2654", "telegram_plain_text_only": True},
        )
        self.assertTrue(result.success)
        self.assertEqual(calls[0]["parse_mode"], None)
        self.assertEqual(calls[0]["message_thread_id"], 2654)
        self.assertEqual(calls[0]["text"], "**hello**")


if __name__ == "__main__":
    unittest.main(verbosity=2)

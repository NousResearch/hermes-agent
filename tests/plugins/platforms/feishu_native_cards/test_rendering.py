"""从原项目迁移的行为契约，保留原布局与状态断言。"""

import unittest


class RenderingContractTests(unittest.TestCase):
    def test_cardkit_cards_expose_clear_streaming_and_final_states(self):
        from plugins.platforms.feishu.native_cards import CardRenderer as FeishuAdapter

        initial = FeishuAdapter._build_cardkit_initial_card()
        self.assertNotIn("header", initial)
        self.assertEqual(initial["config"]["summary"]["content"], "思考中...")
        self.assertEqual(
            initial["body"]["elements"][0]["content"],
            "思考中...",
        )

        streaming = FeishuAdapter._compose_unified_stream_content(
            "正在整理结果",
            {
                "__hermes_stream_status": {
                    "running": ["⏳ Tool running: `web_search`"],
                    "done": ["✅ Tool done: `read_file`"],
                    "calls": [
                        {
                            "id": "call_1",
                            "round": 1,
                            "name": "read_file",
                            "status": "done",
                            "duration": 0.2,
                        },
                        {
                            "id": "call_2",
                            "round": 2,
                            "name": "web_search",
                            "status": "running",
                        },
                    ],
                }
            },
        )
        self.assertNotIn("**执行进度**", streaming)
        self.assertNotIn("**回复内容**", streaming)
        self.assertIn("**工具调用 · 第 1 轮**", streaming)
        self.assertIn("✅ `read_file` · 0.2s", streaming)
        self.assertIn("**工具调用 · 第 2 轮**", streaming)
        self.assertIn("⏳ `web_search`", streaming)
        self.assertIn("正在整理结果", streaming)

        compact_final = FeishuAdapter._compose_unified_stream_content(
            "最终答案",
            {
                "__hermes_stream_status": {
                    "running": [],
                    "done": ["✅ Tool done: `read_file`"],
                    "calls": [
                        {
                            "id": "call_1",
                            "round": 1,
                            "name": "read_file",
                            "status": "done",
                            "duration": 0.2,
                        },
                        {
                            "id": "call_2",
                            "round": 2,
                            "name": "web_search",
                            "status": "done",
                            "duration": 0.3,
                        },
                    ],
                }
            },
            finalize=True,
        )
        self.assertIn("🔧 工具调用：2 项（已收起）", compact_final)
        self.assertNotIn("`read_file`", compact_final)
        self.assertNotIn("`web_search`", compact_final)
        self.assertIn("最终答案", compact_final)

        compatible_final_with_footer = FeishuAdapter._compose_unified_stream_content(
            "最终答案",
            {
                "__hermes_stream_model": "gpt-5.6-terra-900k",
                "__hermes_stream_elapsed_seconds": 2.25,
            },
            finalize=True,
        )
        self.assertIn("🤖 gpt-5.6-terra-900k", compatible_final_with_footer)
        self.assertIn("⏱ 2.2s", compatible_final_with_footer)

        final = FeishuAdapter._build_final_cardkit_card(
            "最终答案",
            tool_status={
                "running": [],
                "done": ["✅ Tool done: `read_file`"],
                "calls": [
                    {
                        "id": "call_1",
                        "round": 1,
                        "name": "read_file",
                        "status": "done",
                        "duration": 0.2,
                    }
                ],
            },
            elapsed_seconds=1.25,
            model_name="test-model",
        )
        self.assertNotIn("header", final)
        self.assertNotIn("summary", final["config"])
        self.assertFalse(final["config"]["streaming_mode"])
        self.assertEqual(final["body"]["elements"][0]["content"], "最终答案")
        tool_trace = final["body"]["elements"][1]
        self.assertEqual(tool_trace["tag"], "collapsible_panel")
        self.assertFalse(tool_trace["expanded"])
        self.assertIn(
            "工具调用记录（1）",
            tool_trace["header"]["title"]["content"],
        )
        self.assertIn("**第 1 轮**", tool_trace["elements"][0]["content"])
        self.assertIn(
            "✅ `read_file` · 0.2s",
            tool_trace["elements"][0]["content"],
        )
        self.assertEqual(final["body"]["elements"][-2]["tag"], "hr")
        footer = final["body"]["elements"][-1]
        self.assertEqual(footer["text_size"], "notation")
        self.assertIn("🤖 test-model", footer["content"])
        self.assertIn("⏱ 1.2s", footer["content"])
        self.assertIn("🔧 1 tools", footer["content"])
        terminal = FeishuAdapter._build_cardkit_terminal_notice("已降级")
        self.assertFalse(terminal["config"]["streaming_mode"])

        def _element_ids(value):
            if isinstance(value, dict):
                if value.get("element_id"):
                    yield value["element_id"]
                for child in value.values():
                    yield from _element_ids(child)
            elif isinstance(value, list):
                for child in value:
                    yield from _element_ids(child)

        element_ids = [
            *_element_ids(initial),
            *_element_ids(final),
            *_element_ids(terminal),
        ]
        self.assertTrue(element_ids)
        self.assertTrue(
            all(
                len(element_id) <= 20
                and element_id[0].isalpha()
                and element_id.replace("_", "").isalnum()
                for element_id in element_ids
            )
        )

    def test_cardkit_renders_moa_progress_and_separate_final_trace(self):
        from plugins.platforms.feishu.native_cards import CardRenderer as FeishuAdapter

        status = {
            "running": ["⏳ Tool running: `read_file`"],
            "done": [],
            "calls": [
                {
                    "id": "call_1",
                    "round": 1,
                    "name": "read_file",
                    "status": "running",
                }
            ],
            "orchestration": {
                "kind": "moa",
                "phase": "degraded_aggregating",
                "refs_done": 2,
                "refs_total": 2,
                "references": [
                    {"label": "deepseek", "status": "done"},
                    {"label": "grok", "status": "failed"},
                ],
                "aggregator": "gpt-5.6-terra",
            },
        }

        streaming = FeishuAdapter._compose_unified_stream_content(
            "正在形成答案",
            {"__hermes_stream_status": status},
        )
        self.assertIn("部分参考模型不可用，正在降级综合", streaming)
        self.assertIn("✅ `deepseek`", streaming)
        self.assertIn("⚠️ `grok`", streaming)
        self.assertIn("🧩 正在综合：`gpt-5.6-terra`", streaming)
        self.assertIn("工具调用 · 第 1 轮", streaming)

        status["orchestration"]["phase"] = "degraded"
        status["calls"][0]["status"] = "done"
        status["running"] = []
        status["done"] = ["✅ Tool done: `read_file`"]
        final = FeishuAdapter._build_final_cardkit_card(
            "最终答案",
            tool_status=status,
        )
        panels = [
            item for item in final["body"]["elements"] if item.get("tag") == "collapsible_panel"
        ]
        self.assertEqual(
            [item["element_id"] for item in panels],
            ["moa_trace", "tool_trace"],
        )
        self.assertIn(
            "MoA 协作记录（2）",
            panels[0]["header"]["title"]["content"],
        )
        self.assertIn(
            "部分参考模型不可用，已降级综合",
            panels[0]["elements"][0]["content"],
        )
        self.assertIn(
            "工具调用记录（1）",
            panels[1]["header"]["title"]["content"],
        )

"""Tests for gateway.platforms.feishu_card."""
from __future__ import annotations
import pytest
from gateway.platforms.feishu_card import format_token_count

class TestFormatTokenCount:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (0, "0"),
            (48, "48"),
            (999, "999"),
            (1000, "1.0k"),
            (1100, "1.1k"),
            (11100, "11.1k"),
            (999999, "1000.0k"),
            (1000000, "1.0M"),
            (3400000, "3.4M"),
        ],
    )
    def test_format_token_count(self, value, expected):
        assert format_token_count(value) == expected

    def test_format_token_count_negative_returns_zero(self):
        assert format_token_count(-5) == "0"


from gateway.platforms.feishu_card import get_tool_display

class TestGetToolDisplay:
    @pytest.mark.parametrize(
        "tool_name, expected",
        [
            ("Read", "Read · 阅读文件"),
            ("read_file", "Read · 阅读文件"),
            ("Bash", "Bash · 执行命令"),
            ("terminal", "Bash · 执行命令"),
            ("Edit", "Edit · 改代码"),
            ("edit_file", "Edit · 改代码"),
            ("Write", "Write · 写文件"),
            ("write_file", "Write · 写文件"),
            ("MultiEdit", "MultiEdit · 批量改代码"),
            ("Grep", "Grep · 搜索代码"),
            ("Glob", "Glob · 查找文件"),
            ("WebFetch", "WebFetch · 抓取网页"),
            ("web_fetch", "WebFetch · 抓取网页"),
            ("WebSearch", "WebSearch · 搜索网络"),
            ("web_search", "WebSearch · 搜索网络"),
            ("Task", "Agent · 派出子任务"),
            ("Agent", "Agent · 派出子任务"),
            ("TodoWrite", "TodoWrite · 更新任务"),
            ("unknown_tool_xyz", "unknown_tool_xyz"),
        ],
    )
    def test_get_tool_display(self, tool_name, expected):
        assert get_tool_display(tool_name) == expected


from gateway.platforms.feishu_card import parse_markdown_tables

class TestParseMarkdownTables:
    def test_no_table_returns_single_text(self):
        result = parse_markdown_tables("just some text")
        assert result == [("text", "just some text")]

    def test_simple_table(self):
        md = "before\n\n| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\nafter"
        result = parse_markdown_tables(md)
        assert len(result) == 3
        assert result[0] == ("text", "before")
        assert result[1][0] == "table"
        table = result[1][1]
        assert table["columns"] == [
            {"name": "col_0", "display_name": "A"},
            {"name": "col_1", "display_name": "B"},
        ]
        assert table["rows"] == [
            {"col_0": "1", "col_1": "2"},
            {"col_0": "3", "col_1": "4"},
        ]
        assert result[2] == ("text", "after")

    def test_empty_input(self):
        result = parse_markdown_tables("")
        assert result == [("text", "")]

    def test_table_only(self):
        md = "| X |\n|---|\n| 1 |"
        result = parse_markdown_tables(md)
        assert len(result) == 1
        assert result[0][0] == "table"

    def test_multiple_tables(self):
        md = "| A |\n|---|\n| 1 |\n\nmiddle\n\n| B |\n|---|\n| 2 |"
        result = parse_markdown_tables(md)
        assert len(result) == 3
        assert result[0][0] == "table"
        assert result[1] == ("text", "middle")
        assert result[2][0] == "table"


import json
from gateway.platforms.feishu_card import build_card_json

class TestBuildCardJson:
    def test_simple_text(self):
        card = build_card_json(content="hello world")
        assert card["config"]["wide_screen_mode"] is True
        assert card["config"]["update_multi"] is True
        assert len(card["elements"]) == 1
        assert card["elements"][0]["tag"] == "markdown"
        assert card["elements"][0]["content"] == "hello world"
        assert "header" not in card

    def test_with_footer(self):
        card = build_card_json(
            content="response text",
            footer_line="📊 ↑48 | ↓11.1k | $0.01 | ⏳12s | 🧠gpt-5.5",
            status_text="✅ 回复完毕",
        )
        elements = card["elements"]
        assert elements[0]["tag"] == "markdown"
        assert elements[0]["content"] == "response text"
        assert elements[1]["tag"] == "hr"
        assert elements[2]["tag"] == "note"
        assert "📊" in elements[2]["elements"][0]["content"]
        assert elements[3]["tag"] == "note"
        assert "✅" in elements[3]["elements"][0]["content"]

    def test_with_tool_status(self):
        card = build_card_json(
            content="some text",
            tool_status="⏳ Bash · 执行命令...",
        )
        elements = card["elements"]
        assert len(elements) == 1
        assert "⏳ Bash · 执行命令..." in elements[0]["content"]

    def test_with_table_content(self):
        md = "intro\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nend"
        card = build_card_json(content=md)
        tags = [e["tag"] for e in card["elements"]]
        assert "table" in tags
        assert tags.count("markdown") == 2

    def test_ack_card(self):
        card = build_card_json(content="⏳ 正在思考...")
        assert card["elements"][0]["content"] == "⏳ 正在思考..."

    def test_error_card(self):
        card = build_card_json(
            content="❌ 处理出错，请重试",
            status_text="❌ 出错",
        )
        assert card["elements"][-1]["tag"] == "note"


import subprocess
from unittest.mock import patch, MagicMock
from gateway.platforms.feishu_card import detect_git_context

class TestDetectGitContext:
    def test_in_git_repo(self, tmp_path):
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "checkout", "-b", "feat/test"],
            cwd=str(tmp_path), capture_output=True,
        )
        result = detect_git_context(str(tmp_path))
        assert result == f"{tmp_path.name}:feat/test"

    def test_not_a_git_repo(self, tmp_path):
        result = detect_git_context(str(tmp_path))
        assert result == ""

    def test_empty_cwd(self):
        result = detect_git_context("")
        assert result == ""

    @patch("gateway.platforms.feishu_card.subprocess.run")
    def test_timeout_returns_empty(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=1)
        result = detect_git_context("/some/path")
        assert result == ""


from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


class TestFeishuCardSendPatch:
    """Test _send_card and _patch_card methods on FeishuAdapter."""

    @pytest.fixture
    def adapter(self):
        from gateway.platforms.feishu import FeishuAdapter

        mock_adapter = MagicMock(spec=FeishuAdapter)
        mock_adapter._client = MagicMock()
        mock_adapter._feishu_send_with_retry = AsyncMock(
            return_value=MagicMock(
                code=0,
                data=MagicMock(message_id="msg_123"),
            )
        )
        mock_adapter._finalize_send_result = MagicMock(
            return_value=SimpleNamespace(success=True, message_id="msg_123", error=None),
        )
        mock_adapter._send_card = FeishuAdapter._send_card.__get__(mock_adapter)
        return mock_adapter

    @pytest.mark.asyncio
    async def test_send_card_sends_interactive_type(self, adapter):
        result = await adapter._send_card(
            chat_id="oc_123",
            card={"config": {}, "elements": []},
            reply_to="msg_orig",
            metadata=None,
        )
        assert result.success
        adapter._feishu_send_with_retry.assert_called_once()
        call_kwargs = adapter._feishu_send_with_retry.call_args[1]
        assert call_kwargs["msg_type"] == "interactive"
        assert call_kwargs["chat_id"] == "oc_123"
        assert call_kwargs["reply_to"] == "msg_orig"

    @pytest.mark.asyncio
    async def test_send_card_returns_message_id(self, adapter):
        result = await adapter._send_card(
            chat_id="oc_123",
            card={"config": {}, "elements": []},
        )
        assert result.message_id == "msg_123"


class TestFeishuSendCardIntegration:
    @pytest.fixture
    def mock_adapter(self):
        from gateway.platforms.feishu import FeishuAdapter

        adapter = MagicMock(spec=FeishuAdapter)
        adapter._client = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 65535
        adapter.format_message = MagicMock(side_effect=lambda x: x.strip())
        adapter.truncate_message = MagicMock(side_effect=lambda x, limit: [x])
        adapter._send_card = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_card_1", error=None),
        )
        adapter._feishu_send_with_retry = AsyncMock(
            return_value=MagicMock(code=0, data=MagicMock(message_id="msg_text_1")),
        )
        adapter._finalize_send_result = MagicMock(
            return_value=SimpleNamespace(success=True, message_id="msg_text_1", error=None),
        )
        adapter._build_outbound_payload = MagicMock(
            return_value=("text", '{"text":"hello"}'),
        )
        adapter._response_succeeded = MagicMock(return_value=True)
        adapter.send = FeishuAdapter.send.__get__(adapter)
        adapter._card_mode_enabled = True
        return adapter

    @pytest.mark.asyncio
    async def test_send_wraps_in_card(self, mock_adapter):
        result = await mock_adapter.send(
            chat_id="oc_123",
            content="hello world",
        )
        assert result.success
        mock_adapter._send_card.assert_called_once()
        card_arg = mock_adapter._send_card.call_args[1]["card"]
        assert card_arg["config"]["wide_screen_mode"] is True
        assert card_arg["elements"][0]["content"] == "hello world"

    @pytest.mark.asyncio
    async def test_send_falls_back_on_card_failure(self, mock_adapter):
        mock_adapter._send_card = AsyncMock(
            return_value=SimpleNamespace(success=False, message_id=None, error="card failed"),
        )
        result = await mock_adapter.send(
            chat_id="oc_123",
            content="hello world",
        )
        assert result.success
        mock_adapter._feishu_send_with_retry.assert_called()


class TestFeishuEditCardIntegration:
    @pytest.fixture
    def mock_adapter(self):
        from gateway.platforms.feishu import FeishuAdapter

        adapter = MagicMock(spec=FeishuAdapter)
        adapter._client = MagicMock()
        adapter.format_message = MagicMock(side_effect=lambda x: x.strip())
        adapter._patch_card = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_123", error=None),
        )
        adapter._build_outbound_payload = MagicMock(
            return_value=("text", '{"text":"hello"}'),
        )
        adapter._build_update_message_body = MagicMock()
        adapter._build_update_message_request = MagicMock()
        adapter._finalize_send_result = MagicMock(
            return_value=SimpleNamespace(success=True, message_id="msg_123", error=None),
        )
        adapter.edit_message = FeishuAdapter.edit_message.__get__(adapter)
        adapter._card_mode_enabled = True
        return adapter

    @pytest.mark.asyncio
    async def test_edit_wraps_in_card(self, mock_adapter):
        result = await mock_adapter.edit_message(
            chat_id="oc_123",
            message_id="msg_123",
            content="updated text",
        )
        assert result.success
        mock_adapter._patch_card.assert_called_once()
        card_arg = mock_adapter._patch_card.call_args[1]["card"]
        assert card_arg["elements"][0]["content"] == "updated text"

    @pytest.mark.asyncio
    async def test_edit_falls_back_on_patch_failure(self, mock_adapter):
        mock_adapter._patch_card = AsyncMock(
            return_value=SimpleNamespace(success=False, message_id=None, error="patch failed"),
        )
        result = await mock_adapter.edit_message(
            chat_id="oc_123",
            message_id="msg_123",
            content="updated text",
        )
        assert result.success


from gateway.platforms.feishu_card import build_card_footer_line

class TestBuildCardFooterLine:
    def test_full_footer(self):
        result = build_card_footer_line(
            input_tokens=48,
            output_tokens=11100,
            cache_tokens=3400000,
            cost_usd=2.9475,
            git_context="nine:feat/xxx",
            elapsed_seconds=34.2,
            model="openai/gpt-5.5",
        )
        assert result == "📊 ↑48 | ↓11.1k | cache:3.4M | $2.9475 | @nine:feat/xxx | ⏳34s | 🧠gpt-5.5"

    def test_no_cache(self):
        result = build_card_footer_line(
            input_tokens=1200,
            output_tokens=5600,
            cache_tokens=0,
            cost_usd=0.015,
            git_context="nine:main",
            elapsed_seconds=12.0,
            model="anthropic/claude-opus-4",
        )
        assert result == "📊 ↑1.2k | ↓5.6k | $0.0150 | @nine:main | ⏳12s | 🧠claude-opus-4"

    def test_no_git(self):
        result = build_card_footer_line(
            input_tokens=100,
            output_tokens=200,
            cache_tokens=0,
            cost_usd=0.001,
            git_context="",
            elapsed_seconds=5.0,
            model="gpt-5.5",
        )
        assert result == "📊 ↑100 | ↓200 | $0.0010 | ⏳5s | 🧠gpt-5.5"


class TestRunFooterMetrics:
    """Verify run.py passes the right fields to build_footer_line."""

    def test_footer_receives_extended_fields(self):
        """Conceptual test — verify the new kwargs exist in format_runtime_footer signature."""
        import inspect
        from gateway.runtime_footer import format_runtime_footer
        sig = inspect.signature(format_runtime_footer)
        new_params = {"input_tokens", "output_tokens", "cache_tokens", "cost_usd", "elapsed_seconds", "git_context"}
        actual_params = set(sig.parameters.keys())
        assert new_params.issubset(actual_params), f"Missing: {new_params - actual_params}"


class TestFeishuEditMetadata:
    """Test that edit_message accepts metadata for finalize footer."""

    @pytest.fixture
    def mock_adapter(self):
        from gateway.platforms.feishu import FeishuAdapter

        adapter = MagicMock(spec=FeishuAdapter)
        adapter._client = MagicMock()
        adapter.format_message = MagicMock(side_effect=lambda x: x.strip())
        adapter._patch_card = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_123", error=None),
        )
        adapter._build_outbound_payload = MagicMock(
            return_value=("text", '{"text":"text"}'),
        )
        adapter._build_update_message_body = MagicMock()
        adapter._build_update_message_request = MagicMock()
        adapter._finalize_send_result = MagicMock(
            return_value=SimpleNamespace(success=True, message_id="msg_123", error=None),
        )
        adapter.edit_message = FeishuAdapter.edit_message.__get__(adapter)
        adapter._card_mode_enabled = True
        return adapter

    @pytest.mark.asyncio
    async def test_finalize_with_footer_metadata(self, mock_adapter):
        result = await mock_adapter.edit_message(
            chat_id="oc_123",
            message_id="msg_123",
            content="final response",
            finalize=True,
            metadata={
                "footer_line": "📊 ↑48 | ↓11.1k | $0.01 | ⏳12s | 🧠gpt-5.5",
                "status_text": "✅ 回复完毕",
            },
        )
        assert result.success
        card = mock_adapter._patch_card.call_args[1]["card"]
        tags = [e["tag"] for e in card["elements"]]
        assert "hr" in tags
        assert "note" in tags

    @pytest.mark.asyncio
    async def test_edit_without_metadata_still_works(self, mock_adapter):
        """Backward compat: edit_message without metadata kwarg."""
        result = await mock_adapter.edit_message(
            chat_id="oc_123",
            message_id="msg_123",
            content="streaming text",
        )
        assert result.success


class TestCardLifecycle:
    """Integration test for the full card lifecycle: ACK → streaming → tool → final."""

    def test_ack_card_structure(self):
        """ACK card has only a thinking indicator."""
        card = build_card_json(content="⏳ 正在思考...")
        assert len(card["elements"]) == 1
        assert card["elements"][0]["tag"] == "markdown"
        assert "正在思考" in card["elements"][0]["content"]

    def test_streaming_card_structure(self):
        """Streaming card has accumulated text, no footer."""
        card = build_card_json(content="partial response so far...")
        assert len(card["elements"]) == 1
        assert card["elements"][0]["tag"] == "markdown"

    def test_tool_status_card_structure(self):
        """Tool status card appends ephemeral status to accumulated text."""
        card = build_card_json(
            content="I'll check the file.",
            tool_status="⏳ Read · 阅读文件...",
        )
        assert len(card["elements"]) == 1
        md = card["elements"][0]["content"]
        assert "I'll check the file." in md
        assert "⏳ Read · 阅读文件..." in md

    def test_heartbeat_card_structure(self):
        """Heartbeat card shows generating indicator with time and tool."""
        card = build_card_json(
            content="",
            tool_status="📝 生成中 · 15s · Read · 阅读文件",
        )
        assert "📝 生成中" in card["elements"][0]["content"]

    def test_final_success_card_structure(self):
        """Final card has content + hr + footer note + status note."""
        card = build_card_json(
            content="Here is your answer.",
            footer_line="📊 ↑48 | ↓11.1k | $0.01 | ⏳12s | 🧠gpt-5.5",
            status_text="✅ 回复完毕",
        )
        tags = [e["tag"] for e in card["elements"]]
        assert tags == ["markdown", "hr", "note", "note"]

    def test_final_error_card_structure(self):
        """Error card shows error message with status."""
        card = build_card_json(
            content="❌ 处理出错，请重试",
            status_text="❌ 出错",
        )
        tags = [e["tag"] for e in card["elements"]]
        assert tags == ["markdown", "note"]

    def test_final_card_with_table(self):
        """Final card with table uses native table element."""
        md = "Results:\n\n| Name | Score |\n|---|---|\n| Alice | 95 |\n| Bob | 87 |"
        card = build_card_json(
            content=md,
            footer_line="📊 ↑100 | ↓200 | $0.01 | ⏳5s | 🧠claude",
            status_text="✅ 回复完毕",
        )
        tags = [e["tag"] for e in card["elements"]]
        assert "table" in tags
        assert "hr" in tags
        assert tags.count("note") == 2

    def test_card_json_is_valid_for_feishu(self):
        """Card JSON can be serialized to valid JSON string."""
        card = build_card_json(
            content="test",
            footer_line="📊 ↑1 | ↓2",
            status_text="✅",
        )
        payload = json.dumps(card, ensure_ascii=False)
        parsed = json.loads(payload)
        assert parsed["config"]["wide_screen_mode"] is True
        assert parsed["config"]["update_multi"] is True

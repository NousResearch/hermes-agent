"""Signal timestamp edits: capability flags, send/edit message ids, and the edit-timestamp chain."""

from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base_edit_targets import (
    adapter_declines_progress_edits,
    adapter_refuses_token_streaming,
    adapter_supports_streaming_edits,
)
from tests.gateway.test_signal import _make_signal_adapter, _stub_rpc


def _rpc_sequence(*responses):
    """An ``_rpc`` stub answering *responses* in order and capturing each call."""
    replies = iter(responses)
    captured = []

    async def mock_rpc(method, params, rpc_id=None, **kwargs):
        captured.append({"method": method, "params": dict(params)})
        return next(replies)

    return mock_rpc, captured


def _editable(monkeypatch, rpc):
    adapter = _make_signal_adapter(monkeypatch)
    adapter._rpc = rpc
    adapter._stop_typing_indicator = AsyncMock()
    return adapter


class TestSignalEditCapabilities:
    """Signal allows explicit and opt-in progress edits, never token streaming."""

    def test_signal_declares_explicit_and_progress_edit_capabilities(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)

        assert adapter.SUPPORTS_MESSAGE_EDITING is True
        assert adapter.SUPPORTS_STREAMING_EDITS is False
        assert adapter.SUPPORTS_PROGRESS_EDITS is True
        assert adapter.EDIT_RESULT_ID_IS_NEXT_TARGET is True

    def test_gateway_capability_gates_use_the_narrow_flags(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)

        assert adapter_supports_streaming_edits(adapter) is False
        assert adapter_declines_progress_edits(adapter) is False
        assert adapter_refuses_token_streaming(adapter) is True

    def test_capability_defaults(self):
        class EditableAdapter:
            SUPPORTS_MESSAGE_EDITING = True

        class NonEditableAdapter:
            SUPPORTS_MESSAGE_EDITING = False

        class ProgressOffAdapter:
            SUPPORTS_PROGRESS_EDITS = False

        assert adapter_supports_streaming_edits(EditableAdapter()) is True
        assert adapter_supports_streaming_edits(NonEditableAdapter()) is False
        # Progress edits are declined only explicitly; SUPPORTS_MESSAGE_EDITING does not decide it.
        assert adapter_declines_progress_edits(EditableAdapter()) is False
        assert adapter_declines_progress_edits(NonEditableAdapter()) is False
        assert adapter_declines_progress_edits(ProgressOffAdapter()) is True
        # A legacy non-editable adapter is not an explicit streaming refusal (proxy keeps its fallback).
        assert adapter_refuses_token_streaming(NonEditableAdapter()) is False

    def test_signal_tool_progress_is_off_by_default_and_opt_in_per_platform(self):
        from gateway.display_config import resolve_display_setting
        from hermes_cli.plugins import discover_plugins

        discover_plugins()  # the "low" tier ships on Signal's PlatformEntry (display_tier)
        opted_in = {"display": {"platforms": {"signal": {"tool_progress": "all"}}}}

        assert resolve_display_setting({}, "signal", "tool_progress") == "off"
        assert resolve_display_setting(opted_in, "signal", "tool_progress") == "all"


class TestSignalSendMessageId:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("rpc_result", [{}, "ok"])
    async def test_send_without_a_timestamp_returns_no_message_id(self, monkeypatch, rpc_result):
        mock_rpc, _ = _stub_rpc(rpc_result)
        adapter = _editable(monkeypatch, mock_rpc)

        result = await adapter.send(chat_id="+155****4567", content="hello")

        assert result.success is True
        assert result.message_id is None

    @pytest.mark.asyncio
    async def test_chunked_send_returns_the_last_chunk_timestamp(self, monkeypatch):
        mock_rpc, captured = _rpc_sequence({"timestamp": 1712345678001}, {"timestamp": 1712345678002})
        adapter = _editable(monkeypatch, mock_rpc)

        result = await adapter.send(chat_id="+155****4567", content="x" * (adapter.MAX_MESSAGE_LENGTH + 500))

        assert len(captured) == 2
        assert result.message_id == "1712345678002"


class TestSignalEditMessage:
    @pytest.mark.asyncio
    async def test_dm_edit_sends_edit_timestamp_with_native_formatting(self, monkeypatch):
        mock_rpc, captured = _stub_rpc({"timestamp": 1712345679000})
        adapter = _editable(monkeypatch, mock_rpc)

        result = await adapter.edit_message(
            chat_id="+155****4567", message_id="1712345678000", content="edited **hello**", finalize=True,
        )

        assert result.success is True
        assert result.message_id == "1712345679000"
        assert captured[0]["method"] == "send"
        params = captured[0]["params"]
        assert params["editTimestamp"] == 1712345678000
        assert params["message"] == "edited hello"
        assert params["recipient"] == ["+155****4567"]
        assert params["textStyle"].endswith(":BOLD")

    @pytest.mark.asyncio
    async def test_group_edit_routes_by_group_id(self, monkeypatch):
        mock_rpc, captured = _stub_rpc({"timestamp": 1234567999})
        adapter = _editable(monkeypatch, mock_rpc)

        result = await adapter.edit_message(
            chat_id="group:BASE64_GROUP_ID", message_id="1234567890", content="group update",
        )

        assert result.success is True
        assert result.message_id == "1234567999"
        assert captured == [{
            "method": "send",
            "params": {
                "account": "+15551234567",
                "message": "group update",
                "editTimestamp": 1234567890,
                "groupId": "BASE64_GROUP_ID",
            },
        }]

    @pytest.mark.asyncio
    async def test_successive_edits_chain_through_fresh_timestamps(self, monkeypatch):
        mock_rpc, captured = _rpc_sequence({"timestamp": 1712345679000}, {"timestamp": 1712345680000})
        adapter = _editable(monkeypatch, mock_rpc)

        first = await adapter.edit_message(chat_id="+155****4567", message_id="1712345678000", content="first")
        second = await adapter.edit_message(chat_id="+155****4567", message_id=first.message_id, content="second")

        assert (first.message_id, second.message_id) == ("1712345679000", "1712345680000")
        assert [call["params"]["editTimestamp"] for call in captured] == [1712345678000, 1712345679000]

    @pytest.mark.asyncio
    async def test_edit_without_a_fresh_timestamp_fails_closed(self, monkeypatch):
        """A successful RPC that omits the next edit handle must not report success."""
        mock_rpc, _ = _stub_rpc({})
        adapter = _editable(monkeypatch, mock_rpc)

        result = await adapter.edit_message(chat_id="+155****4567", message_id="1712345678000", content="edited")

        assert result.success is False
        assert result.message_id is None
        assert "fresh timestamp" in result.error

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("message_id", "error_part"), [("", "message_id"), ("not-a-timestamp", "numeric")])
    async def test_edit_requires_a_numeric_timestamp_handle(self, monkeypatch, message_id, error_part):
        mock_rpc, captured = _stub_rpc({"timestamp": 1712345679000})
        adapter = _editable(monkeypatch, mock_rpc)

        result = await adapter.edit_message(chat_id="+155****4567", message_id=message_id, content="edited")

        assert result.success is False
        assert error_part in result.error
        assert captured == []

"""The compaction heartbeat (#98371) must be classified like COMPACTION_STATUS.

Chat platforms suppress routine compression chatter unless
``compression.progress_notices`` is enabled; a heartbeat that slipped past
that gate would post a bubble per tick to Telegram/Discord. The TUI gateway,
by contrast, must keep receiving it so idle-turn watchdogs see progress.
"""

from types import SimpleNamespace
from unittest.mock import patch

from agent.conversation_compression import (
    COMPACTION_HEARTBEAT_STATUS,
    COMPACTION_STATUS,
    is_compaction_progress_status,
)
from gateway.run import _prepare_gateway_status_message


def _telegram():
    return SimpleNamespace(value="telegram")


def test_heartbeat_is_compaction_progress_for_tui_retagging():
    assert is_compaction_progress_status(COMPACTION_HEARTBEAT_STATUS)


def test_heartbeat_suppressed_on_chat_platforms_by_default():
    with patch("gateway.run._gateway_compression_progress_notices_enabled", return_value=False):
        assert _prepare_gateway_status_message(_telegram(), "lifecycle", COMPACTION_STATUS) is None
        assert _prepare_gateway_status_message(_telegram(), "lifecycle", COMPACTION_HEARTBEAT_STATUS) is None


def test_heartbeat_passes_when_progress_notices_enabled():
    with patch("gateway.run._gateway_compression_progress_notices_enabled", return_value=True):
        assert _prepare_gateway_status_message(_telegram(), "lifecycle", COMPACTION_HEARTBEAT_STATUS) == COMPACTION_HEARTBEAT_STATUS

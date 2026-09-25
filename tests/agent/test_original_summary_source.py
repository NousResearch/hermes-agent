from types import SimpleNamespace

from agent.agent_init import _parse_compression_config
from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY, ContextCompressor, SUMMARY_PREFIX, _MERGED_SUMMARY_DELIMITER,
)
from hermes_cli.config import DEFAULT_CONFIG


def _agent():
    return SimpleNamespace(model="m", provider="openrouter", api_mode="chat_completions", quiet_mode=True)


def _summary(label):
    return {
        "role": "user",
        "content": f"[CONTEXT COMPACTION — CHECKPOINT]\n{label}",
        COMPRESSED_SUMMARY_METADATA_KEY: True,
    }


def test_original_source_config_is_opt_in_and_bounded():
    default = _parse_compression_config(_agent(), {})
    assert default.summary_source == DEFAULT_CONFIG["compression"]["summary_source"] == "previous"
    assert default.summary_source_windows == 2

    parsed = _parse_compression_config(
        _agent(), {"compression": {"summary_source": "original", "summary_source_windows": 99}}
    )
    assert parsed.summary_source == "original"
    assert parsed.summary_source_windows == 8


def test_original_prompt_does_not_use_previous_summary():
    compressor = ContextCompressor(model="test", quiet_mode=True, summary_source="original")
    compressor._previous_summary = "THIS MUST NOT BE FED AS A ROLLING SUMMARY"
    prompt = compressor._build_summary_prompt(
        "RAW ORIGINAL TURN", 200, None, "", True,
        frozen_summary="FROZEN OLDER CHECKPOINT", original_source=True,
    )
    assert "PREVIOUS SUMMARY:" not in prompt
    assert "FROZEN HISTORICAL SUMMARY (REFERENCE ONLY):" in prompt
    assert "RAW ORIGINAL TURN" in prompt
    assert "THIS MUST NOT BE FED AS A ROLLING SUMMARY" not in prompt


def test_original_source_reads_recent_raw_windows_and_freezes_older_checkpoint():
    class FakeSessionDB:
        def get_messages_as_conversation(self, session_id, **kwargs):
            assert session_id == "session-1"
            assert kwargs["include_ancestors"] is True
            assert kwargs["include_inactive"] is True
            assert kwargs["include_summary_markers"] is True
            return [
                {"role": "user", "content": "raw-window-0"},
                _summary("checkpoint-1"),
                {"role": "assistant", "content": "raw-window-1"},
                _summary("checkpoint-2"),
                {"role": "user", "content": "raw-window-2"},
                _summary("checkpoint-3"),
                {"role": "assistant", "content": "live-tail"},
            ]

    compressor = ContextCompressor(
        model="test", quiet_mode=True, summary_source="original", summary_source_windows=2,
    )
    compressor.bind_session_state(FakeSessionDB(), "session-1")
    source = compressor._load_original_summary_source([{"role": "user", "content": "current-window"}])
    assert source is not None
    turns, frozen = source
    assert [turn["content"] for turn in turns] == ["raw-window-2", "current-window"]
    assert "checkpoint-2" in frozen
    assert "checkpoint-1" not in frozen
    assert "checkpoint-3" not in frozen


def test_original_source_keeps_live_content_from_merged_summary_carriers():
    merged = {
        "role": "user",
        "content": f"carried-tail\n\n{_MERGED_SUMMARY_DELIMITER}\n\n{SUMMARY_PREFIX}\ncheckpoint-2",
        COMPRESSED_SUMMARY_METADATA_KEY: True,
    }

    class FakeSessionDB:
        def get_messages_as_conversation(self, session_id, **kwargs):
            return [
                {"role": "user", "content": "raw-window-1"},
                merged,
                {"role": "assistant", "content": "live"},
            ]

    compressor = ContextCompressor(
        model="test", quiet_mode=True, summary_source="original", summary_source_windows=2,
    )
    compressor.bind_session_state(FakeSessionDB(), "session-1")
    turns, frozen = compressor._load_original_summary_source(
        [{"role": "user", "content": "current-window"}]
    )
    assert [turn["content"] for turn in turns] == ["raw-window-1", "carried-tail", "current-window"]
    assert frozen == ""
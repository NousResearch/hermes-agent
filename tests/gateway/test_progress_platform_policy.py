"""Platform progress policy regressions."""
import importlib
import pytest
from gateway.config import Platform, StreamingConfig
from gateway.session import SessionSource
from tests.gateway.test_run_progress_topics import CommentaryAgent, ProgressCaptureAdapter, _make_runner, _run_with_agent


class TestSlackReplyInThreadProgressRouting:
    """#18859: reply_in_thread=false must stop progress from creating threads."""

    def test_slack_reply_in_thread_false_drops_synthetic_thread(self):
        from gateway.run import _resolve_progress_thread_id

        # source.thread_id == event ts is the adapter's synthetic
        # session-keying thread for top-level messages — not a real thread.
        assert _resolve_progress_thread_id(
            Platform.SLACK,
            source_thread_id="1700000000.000100",
            event_message_id="1700000000.000100",
            reply_in_thread=False,
        ) is None

    def test_buzz_uses_event_message_id_as_progress_thread(self):
        """Buzz has no native thread_id; progress must reply-to the trigger."""
        from gateway.run import _resolve_progress_thread_id

        assert _resolve_progress_thread_id(
            "buzz",
            source_thread_id=None,
            event_message_id="evt-trigger-001",
            reply_in_thread=True,
        ) == "evt-trigger-001"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["turn_runner", "proxy"])
async def test_per_platform_streaming_does_not_override_global_disabled(monkeypatch, tmp_path, path):
    """Regression for #53697: display.platforms.telegram.streaming=True must not
    re-enable gateway streaming when streaming.enabled=False (global master switch).

    Covers both gate sites: TurnRunner.want_stream_deltas (normal agent path) and
    GatewayTurnMixin._proxy_stream_consumer (proxy / native-streaming path)."""
    config_data = {
        "display": {
            "platforms": {
                "telegram": {"streaming": True},
            },
            "interim_assistant_messages": False,
        },
        "streaming": {"enabled": False},
    }
    if path == "proxy":
        adapter = ProgressCaptureAdapter(platform=Platform.TELEGRAM)
        runner = _make_runner(adapter)
        runner.config.streaming = StreamingConfig.from_dict(config_data["streaming"])
        gateway_run = importlib.import_module("gateway.run")
        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: config_data)
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001", chat_type="dm")
        assert runner._proxy_stream_consumer(source, None, None, lambda: True) is None
        return

    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CommentaryAgent,
        session_id="sess-per-platform-streaming-global-off",
        config_data=config_data,
        platform=Platform.TELEGRAM,
    )

    assert result.get("already_sent") is not True
    assert adapter.edits == []


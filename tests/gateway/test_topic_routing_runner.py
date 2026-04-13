from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def test_runner_build_reply_metadata_for_telegram_topic():
    runner = GatewayRunner(GatewayConfig())
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="413",
        chat_topic="舆情监测",
    )

    assert runner._build_reply_metadata(source) == {
        "thread_id": "413",
        "topic_name": "舆情监测",
        "topic_boundary": "strict",
    }


def test_runner_build_reply_metadata_for_watcher_thread_only():
    runner = GatewayRunner(GatewayConfig())

    assert runner._build_reply_metadata(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        thread_id="413",
    ) == {
        "thread_id": "413",
        "topic_boundary": "strict",
    }

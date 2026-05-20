import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource
from gateway.zero_token_mirror import (
    canonical_mirror_session_key,
    load_mirror_pairs,
    mirror_assistant_message,
    mirror_user_message,
    resolve_gateway_session_key,
    targets_for_source,
)


CONFIG = {
    "gateway": {
        "deterministic_mirrors": {
            "enabled": True,
            "pairs": [
                {
                    "name": "pm",
                    "endpoints": [
                        {"platform": "telegram", "chat_id": "2091982351"},
                        {
                            "platform": "slack",
                            "chat_id": "C0B47RGM7NZ",
                            "thread_id": "1779007573.641769",
                        },
                    ],
                }
            ],
        }
    }
}

WILDCARD_CONFIG = {
    "gateway": {
        "deterministic_mirrors": {
            "enabled": True,
            "pairs": [
                {
                    "name": "pm",
                    "endpoints": [
                        {"platform": "telegram", "chat_id": "2091982351"},
                        {
                            "platform": "slack",
                            "chat_id": "C0B47RGM7NZ",
                            "thread_id": "*",
                        },
                    ],
                }
            ],
        }
    }
}


class FakeAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, content, metadata=None):
        self.sent.append((chat_id, content, metadata))


class FakeGateway:
    def __init__(self):
        self.adapters = {Platform.TELEGRAM: FakeAdapter(), Platform.SLACK: FakeAdapter()}

    @staticmethod
    def _load_config():
        return CONFIG


def test_targets_for_thread_pinned_slack_source():
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C0B47RGM7NZ",
        thread_id="1779007573.641769",
        user_id="U1",
        user_name="Leo",
    )

    targets = targets_for_source(CONFIG, source)

    assert [(t.platform, t.chat_id, t.thread_id) for t in targets] == [
        ("telegram", "2091982351", None)
    ]


def test_wrong_slack_thread_does_not_mirror():
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C0B47RGM7NZ",
        thread_id="other-thread",
    )

    assert targets_for_source(CONFIG, source) == ()


def test_paired_endpoints_share_one_canonical_session_key():
    slack_source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C0B47RGM7NZ",
        thread_id="1779007573.641769",
        chat_type="group",
        user_id="U1",
    )
    telegram_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="2091982351",
        chat_type="dm",
        user_id="2091982351",
    )

    assert canonical_mirror_session_key(CONFIG, slack_source) == "agent:main:mirror:pm"
    assert canonical_mirror_session_key(CONFIG, telegram_source) == "agent:main:mirror:pm"


def test_unpaired_thread_keeps_native_session_key_path():
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C0B47RGM7NZ",
        thread_id="other-thread",
    )

    assert canonical_mirror_session_key(CONFIG, source) is None


def test_wildcard_slack_threads_mirror_but_keep_distinct_session_keys():
    first = SessionSource(platform=Platform.SLACK, chat_id="C0B47RGM7NZ", thread_id="thread-a")
    second = SessionSource(platform=Platform.SLACK, chat_id="C0B47RGM7NZ", thread_id="thread-b")
    root = SessionSource(platform=Platform.SLACK, chat_id="C0B47RGM7NZ")
    telegram = SessionSource(platform=Platform.TELEGRAM, chat_id="2091982351")

    assert [(t.platform, t.chat_id, t.thread_id) for t in targets_for_source(WILDCARD_CONFIG, first)] == [
        ("telegram", "2091982351", None)
    ]
    assert targets_for_source(WILDCARD_CONFIG, root) == ()
    assert canonical_mirror_session_key(WILDCARD_CONFIG, root) is None
    assert targets_for_source(WILDCARD_CONFIG, telegram) == ()
    assert canonical_mirror_session_key(WILDCARD_CONFIG, telegram) is None
    assert canonical_mirror_session_key(WILDCARD_CONFIG, first) == (
        "agent:main:mirror:pm:slack_C0B47RGM7NZ_thread-a"
    )
    assert canonical_mirror_session_key(WILDCARD_CONFIG, second) == (
        "agent:main:mirror:pm:slack_C0B47RGM7NZ_thread-b"
    )



def test_string_false_disables_mirror_directions():
    config = {
        "gateway": {
            "deterministic_mirrors": {
                "enabled": "true",
                "pairs": [
                    {
                        "name": "pm",
                        "mirror_user_messages": "false",
                        "mirror_assistant_messages": "0",
                        "endpoints": [
                            {"platform": "telegram", "chat_id": "2091982351"},
                            {"platform": "slack", "chat_id": "C0B47RGM7NZ"},
                        ],
                    }
                ],
            }
        }
    }
    pair = load_mirror_pairs(config)[0]

    assert pair.mirror_user_messages is False
    assert pair.mirror_assistant_messages is False
    assert targets_for_source(config, SessionSource(platform=Platform.TELEGRAM, chat_id="2091982351")) == ()
    assert targets_for_source(
        config,
        SessionSource(platform=Platform.TELEGRAM, chat_id="2091982351"),
        assistant=True,
    ) == ()


def test_resolve_gateway_session_key_uses_same_canonical_key_as_session_store():
    slack_source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C0B47RGM7NZ",
        thread_id="1779007573.641769",
        chat_type="group",
        user_id="U1",
    )
    telegram_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="2091982351",
        chat_type="dm",
        user_id="2091982351",
    )

    assert resolve_gateway_session_key(CONFIG, slack_source) == "agent:main:mirror:pm"
    assert resolve_gateway_session_key(CONFIG, telegram_source) == "agent:main:mirror:pm"


def test_resolve_gateway_session_key_falls_back_to_native_key_when_unpaired():
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C0B47RGM7NZ",
        thread_id="other-thread",
        chat_type="group",
        user_id="U1",
    )

    assert resolve_gateway_session_key(
        CONFIG,
        source,
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    ) == "agent:main:slack:group:C0B47RGM7NZ:other-thread"


def test_create_adapter_propagates_deterministic_mirrors():
    from gateway.run import GatewayRunner

    mirror_cfg = CONFIG["gateway"]["deterministic_mirrors"]
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(deterministic_mirrors=mirror_cfg)
    platform_config = PlatformConfig(enabled=True, extra={})

    runner._create_adapter(Platform.LOCAL, platform_config)

    assert platform_config.extra["deterministic_mirrors"] == mirror_cfg


@pytest.mark.asyncio
async def test_user_mirror_sends_without_agent_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    gateway = FakeGateway()
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C0B47RGM7NZ",
        thread_id="1779007573.641769",
        user_name="Leo",
    )

    await mirror_user_message(gateway, source, "세팅")

    sent = gateway.adapters[Platform.TELEGRAM].sent
    assert len(sent) == 1
    assert sent[0][0] == "2091982351"
    assert "사용자 메시지 미러" in sent[0][1]
    assert "세션: 00001 세팅" in sent[0][1]
    assert "작성자: Leo" in sent[0][1]
    assert "chat=C0B47RGM7NZ / thread=1779007573.641769" in sent[0][1]
    assert "LLM 재입력 없음" in sent[0][1]
    assert "세팅" in sent[0][1]
    assert sent[0][2]["mirror"] is True


@pytest.mark.asyncio
async def test_assistant_mirror_sends_to_slack_thread(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    gateway = FakeGateway()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="2091982351")

    await mirror_assistant_message(gateway, source, "완료했습니다")

    sent = gateway.adapters[Platform.SLACK].sent
    assert len(sent) == 1
    assert sent[0][0] == "C0B47RGM7NZ"
    assert sent[0][2]["thread_id"] == "1779007573.641769"
    assert "Leo 답변 미러" in sent[0][1]
    assert "세션: 00001 Telegram 세션" in sent[0][1]
    assert "chat=2091982351 / thread=main" in sent[0][1]
    assert "답변 1회 생성" in sent[0][1]


@pytest.mark.asyncio
async def test_assistant_reuses_user_created_mirror_label(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    gateway = FakeGateway()

    slack_source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C0B47RGM7NZ",
        thread_id="1779007573.641769",
        user_name="Leo",
    )
    await mirror_user_message(gateway, slack_source, "오픈클로 미러링 설계")

    telegram_source = SessionSource(platform=Platform.TELEGRAM, chat_id="2091982351")
    await mirror_assistant_message(gateway, telegram_source, "확인했습니다")

    sent = gateway.adapters[Platform.SLACK].sent
    assert len(sent) == 1
    assert "세션: 00001 오픈클로 미러링 설" in sent[0][1]

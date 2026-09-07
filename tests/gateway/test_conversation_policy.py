import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource
from gateway.conversation import conversation_send
from plugins.groupchat.policy import requires_buffered_delivery
from plugins.groupchat.relevance import IntelligentReactionGate


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("text,mentioned,dropped", [
    ("abcd", False, True), ("abcd", True, False),
    ("abcdWXYZ", False, False), ("abcdXYZ", False, True),
    ("abcdWXYZ!", False, False), ("abcd abcd x", False, True),
    ("ABCD", False, False), ("# English keywords", False, False),
    ("[.*]", False, True), ("something else", False, False),
])
async def test_literal_phrase_majority_and_mention_exemption(text, mentioned, dropped, tmp_path):
    adapter = Adapter(Platform.MATRIX, {"relevance": {
        "enabled": True, "system_patterns": [], "multiline_patterns": [],
        "literal_phrases": ["# English keywords", "abcd", "[.*]"]}})
    gate = adapter.conversation_policy().relevance
    gate._throttled_evaluate = AsyncMock(return_value=(5, "Relevant test message."))
    message = event(Platform.MATRIX, text=text)
    message.metadata["conversation_mentioned"] = mentioned
    await adapter.handle_message(message)
    assert bool(adapter.delivered) is not dropped
    if dropped:
        record = json.loads((tmp_path / "logs/matrix-relevance-decisions.jsonl").read_text().splitlines()[-1])
        assert record["reason_code"] == "literal_phrase_majority"
    await adapter.disconnect()


def test_comments_are_preserved_and_not_validated_as_regex():
    from plugins.groupchat.config import validate_settings
    from plugins.groupchat.pingpong_guard import decide
    values = ["# English keywords [", "^done$", "# German keywords", "^erledigt$"]
    settings = validate_settings({"pingpong_guard": {"silence_patterns": values}})
    assert settings["pingpong_guard"]["silence_patterns"] == values
    assert decide("done", "", {}, 1, values)["pattern_index"] == 2
    assert decide("# English keywords [", "", {}, 1, values)["decision"] == "send"


def test_old_literal_setting_is_migrated_once():
    from plugins.groupchat.config import validate_settings
    settings = validate_settings({"relevance": {"interrupt_notice": "old notice"}})
    assert settings["relevance"]["literal_phrases"] == ["old notice"]
    assert "interrupt_notice" not in settings["relevance"]


@pytest.mark.parametrize("reply", ["verstanden", "understood", "danke", "thank you", "done", "(remains silent)"])
def test_bilingual_guard_defaults(reply):
    from plugins.groupchat.pingpong_guard import decide
    decision = decide(reply, "", {}, 60)
    assert decision["reason_code"] == "pattern_match"
    assert decision["decision"] == "suppress"
    assert decision["pattern_index"] > 0


@pytest.mark.parametrize("context", [
    "Please reply exactly with ok.",
    "Please say ok.",
    "Bitte antworte exakt mit ok.",
    "Sag bitte ok.",
])
def test_requested_pattern_reply_is_sent(context):
    from plugins.groupchat.pingpong_guard import decide
    decision = decide("ok", context, {}, 60)
    assert decision["decision"] == "send"
    assert decision["reason_code"] == "explicit_response_request"


@pytest.mark.parametrize("context", [
    "Charlotte is working on this.",
    "Please do not reply.",
    "Bitte nicht antworten.",
])
def test_unrequested_or_negated_pattern_reply_is_suppressed(context):
    from plugins.groupchat.pingpong_guard import decide
    decision = decide("ok", context, {}, 60)
    assert decision["decision"] == "suppress"
    assert decision["reason_code"] == "pattern_match"


@pytest.mark.anyio
async def test_outbound_decision_log_is_private_and_explains_rule(tmp_path):
    import stat
    adapter = Adapter(Platform.MATRIX, {"pingpong_guard": {"enabled": True}})
    await adapter.send("room", "understood")
    path = tmp_path / "logs/matrix-groupchat-outbound.jsonl"
    record = json.loads(path.read_text().splitlines()[-1])
    assert record["reason_code"] == "pattern_match"
    assert record["decision"] == "suppress"
    assert record["room_id"] == "room"
    assert len(record["pattern_sha256"]) == 64
    assert "understood" not in path.read_text()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    await adapter.disconnect()


def test_audit_rotation_and_allowlist(tmp_path, monkeypatch):
    from plugins.groupchat import audit
    monkeypatch.setattr(audit, "MAX_BYTES", 1)
    audit.record_outbound(tmp_path, "matrix", "room", "", {"decision": "suppress", "reason_code": "pattern_match", "text": "private", "token": "secret"})
    audit.record_outbound(tmp_path, "matrix", "room", "", {"decision": "send", "reason_code": "length_threshold"})
    current = tmp_path / "logs/matrix-groupchat-outbound.jsonl"
    old = current.with_suffix(".jsonl.1")
    assert old.exists()
    assert "private" not in old.read_text() and "secret" not in old.read_text()
    assert json.loads(current.read_text())["decision"] == "send"


def test_host_suppressed_output_is_audited_without_message_body(tmp_path):
    adapter = Adapter(Platform.MATRIX, {"pingpong_guard": {"enabled": True}})
    message = event(Platform.MATRIX, text="question")
    adapter.conversation_middleware().output_suppressed(
        message, "NO_REPLY", "host_intentional_silence"
    )
    record = json.loads(
        (tmp_path / "logs/matrix-groupchat-outbound.jsonl").read_text().splitlines()[-1]
    )
    assert record["decision"] == "suppress"
    assert record["reason_code"] == "host_intentional_silence"
    assert "NO_REPLY" not in json.dumps(record)


@pytest.mark.anyio
async def test_inbound_audit_records_matching_rule_without_message(tmp_path):
    adapter = Adapter(Platform.MATRIX, {"relevance": {"enabled": True, "system_patterns": ["^private-pattern"]}})
    await adapter.handle_message(event(Platform.MATRIX, text="private-pattern secret payload"))
    path = tmp_path / "logs/matrix-relevance-decisions.jsonl"
    record = json.loads(path.read_text().splitlines()[-1])
    assert record["reason_code"] == "system_message_before_routing"
    assert record["pattern_index"] == 1
    assert record["pattern_collection"] == "system_patterns"
    assert "secret payload" not in path.read_text()
    await adapter.disconnect()


@pytest.mark.anyio
async def test_configured_pingpong_patterns_reach_worker():
    adapter = Adapter(settings={"pingpong_guard": {
        "enabled": True, "min_chars": 1, "silence_patterns": [r"custom-block"]}})
    assert (await adapter.send("room", "A CUSTOM-BLOCK inside a reply")).success
    assert not adapter.sent
    assert (await adapter.send("room", "ok")).success
    assert adapter.sent == [("room", "ok")]
    await adapter.disconnect()


@pytest.mark.anyio
async def test_relevance_pattern_lists_replace_defaults_and_preserve_commas():
    adapter = Adapter(Platform.MATRIX, {"relevance": {
        "enabled": True, "system_patterns": [r"^a{2,3}$"],
        "multiline_patterns": [], "interrupt_notice": ""}})
    gate = adapter.conversation_policy().relevance
    assert gate._is_system_message_one_liner("aaa")
    assert not gate._is_system_message_one_liner("123")
    assert not gate._is_system_message_multi_line("📚 anything\n📚 else")
    assert not gate._is_interrupt_notice_majority("anything", gate._interrupt_notice)
    await adapter.disconnect()


def test_empty_lists_and_defaults_are_distinct():
    from plugins.groupchat.config import validate_settings
    defaults = validate_settings({})
    assert defaults["relevance"]["system_patterns"]
    assert defaults["pingpong_guard"]["silence_patterns"]
    empty = validate_settings({"relevance": {"system_patterns": [], "multiline_patterns": [], "interrupt_notice": ""},
                               "pingpong_guard": {"silence_patterns": []}})
    assert empty["relevance"]["system_patterns"] == []
    assert empty["pingpong_guard"]["silence_patterns"] == []


def test_legacy_score_one_delay_is_removed_from_settings():
    from plugins.groupchat.config import validate_settings

    settings = validate_settings({
        "relevance": {
            "score_delays": {5: 0, 4: 3, 3: 10, 2: 30, 1: 120}
        }
    })
    assert settings["relevance"]["score_delays"] == {
        5: 0,
        4: 3,
        3: 10,
        2: 30,
    }


def test_invalid_pattern_is_validation_error():
    from plugins.groupchat.config import validate_settings
    with pytest.raises(ValueError, match="Invalid system pattern"):
        validate_settings({"relevance": {"system_patterns": ["["]}})


def test_worker_cannot_reopen_profile_credentials(tmp_path, monkeypatch):
    from plugins.groupchat.models import credential
    monkeypatch.setenv("GROUPCHAT_CREDENTIALS_RESOLVED", "1")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    (tmp_path / ".env").write_text("MISTRAL_API_KEY=denied-test-key\n")
    assert credential("MISTRAL_API_KEY") == ""


@pytest.mark.anyio
async def test_canonical_slack_scope_matches_outbound():
    adapter = Adapter(settings={"pingpong_guard": {"enabled": True}})
    message = event(thread="thread-a")
    message.source.scope_id = "workspace-a"
    await adapter.handle_message(message)
    scoped = adapter.conversation_policy().for_conversation(
        "same-room", {"scope_id": "workspace-a", "thread_id": "thread-a"})
    assert scoped.last_inbound["same-room"] == message.text
    await adapter.disconnect()


@pytest.mark.anyio
async def test_native_matrix_send_runs_discovered_guard():
    from plugins.platforms.matrix.adapter import MatrixAdapter
    from unittest.mock import MagicMock
    adapter = MatrixAdapter(PlatformConfig(extra={"groupchat": {
        "enabled": True, "relevance": {"enabled": False},
        "pingpong_guard": {"enabled": True}}}))
    adapter._client = MagicMock()
    adapter._client.send_message_event = AsyncMock(return_value="$sent")
    result = await adapter.send("!room:example.org", "*(bleibt still)*")
    assert result.success
    adapter._client.send_message_event.assert_not_awaited()
    assert adapter.conversation_middleware().handlers[0].guard_enabled
    await adapter.cancel_background_tasks()


@pytest.fixture(autouse=True)
def plugin_discovery(tmp_path, monkeypatch):
    from hermes_cli import plugins
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text('plugins:\n  enabled: [groupchat]\n')
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    manager.discover_and_load()
    assert manager.has_middleware("gateway_conversation")
    yield manager


class Adapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.SLACK, settings=None):
        options = {"enabled": bool(settings), "relevance": {"enabled": False},
                   "pingpong_guard": {"enabled": False}, **(settings or {})}
        super().__init__(PlatformConfig(extra={"groupchat": options}), platform)
        self.delivered = []
        self.sent = []
        self.fail_send = False
        self.set_authorization_check(lambda *args, **kwargs: True)

    async def connect(self, **kwargs):
        return True

    async def get_chat_info(self, chat_id):
        return {"name": chat_id, "type": "group"}

    def conversation_policy(self):
        return self.conversation_middleware().handlers[0]

    async def disconnect(self):
        await self.cancel_background_tasks()

    async def _handle_message_after_conversation(self, event):
        self.delivered.append(event)

    @conversation_send
    async def send(self, chat_id, content, reply_to=None, metadata=None):
        if self.fail_send:
            return SendResult(success=False, error="temporary", retryable=True)
        self.sent.append((chat_id, content))
        return SendResult(success=True, message_id=str(len(self.sent)))


def event(platform=Platform.SLACK, *, thread=None, text="Please review", command=False):
    return MessageEvent(
        text=text, message_type=MessageType.COMMAND if command else MessageType.TEXT,
        source=SessionSource(platform=platform, chat_id="same-room", chat_type="group",
                             user_id="sender", user_name="Human", thread_id=thread),
        message_id="event-1", metadata={"conversation_mentioned": True},
    )


@pytest.mark.anyio
@pytest.mark.parametrize("platform", [Platform.MATRIX, Platform.SLACK, Platform.TELEGRAM])
async def test_normalized_ingress_without_transport_sdk(platform, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = Adapter(platform, {"relevance": {"enabled": True}, "pingpong_guard": {"enabled": False}})
    await adapter.handle_message(event(platform))
    assert len(adapter.delivered) == 1
    assert adapter.delivered[0].text.startswith("Please review")
    assert "Relevance assessment" in adapter.delivered[0].text
    await adapter.disconnect()


@pytest.mark.anyio
async def test_plain_peer_name_is_dropped_before_scoring_or_buffering(tmp_path):
    adapter = Adapter(Platform.MATRIX, {
        "relevance": {"enabled": True},
        "pingpong_guard": {"enabled": False},
    })
    gate = adapter.conversation_policy().relevance
    gate._peer_names = ["charlotte", "charlotte ai"]
    gate._peer_name_pattern = gate._names_pattern(gate._peer_names)
    gate._throttled_evaluate = AsyncMock(
        side_effect=AssertionError("peer-targeted message reached scorer")
    )

    message = event(
        Platform.MATRIX,
        text='charlotte, bitte antworte exakt mit "OK"',
    )
    message.metadata["conversation_mentioned"] = False
    await adapter.handle_message(message)

    peer_reply = event(Platform.MATRIX, text="OK")
    peer_reply.message_id = "peer-reply"
    peer_reply.reply_to_message_id = message.message_id
    peer_reply.source.user_id = "@charlotte_ai:example.test"
    peer_reply.metadata["conversation_mentioned"] = False
    await adapter.handle_message(peer_reply)

    assert adapter.delivered == []
    assert gate._pending == {}
    assert [entry[1] for entry in gate._room_transcript["same-room"][-2:]] == [
        message.text,
        "OK",
    ]
    records = [
        json.loads(line)
        for line in (tmp_path / "logs/matrix-relevance-decisions.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [record["reason_code"] for record in records[-2:]] == [
        "plain_name_addressed_to_peer",
        "reply_to_peer_targeted_message",
    ]
    assert all(record["explicitly_addressed_elsewhere"] for record in records[-2:])
    await adapter.disconnect()


@pytest.mark.anyio
async def test_peer_thread_stays_passive_until_next_open_message(tmp_path):
    adapter = Adapter(Platform.MATRIX, {
        "relevance": {"enabled": True},
        "pingpong_guard": {"enabled": True},
    })
    gate = adapter.conversation_policy().relevance
    gate._peer_names = ["charlotte"]
    gate._peer_name_pattern = gate._names_pattern(gate._peer_names)

    request = event(Platform.MATRIX, text="Charlotte, please assess this proposal")
    request.metadata["conversation_mentioned"] = False
    await adapter.handle_message(request)

    gate._throttled_evaluate = AsyncMock(
        side_effect=AssertionError("explicit peer thread reached scorer")
    )
    peer_reply = event(
        Platform.MATRIX,
        text=(
            "The proposal has a concrete compatibility risk in its storage "
            "format that should be considered before implementation."
        ),
    )
    peer_reply.message_id = "substantive-peer-reply"
    peer_reply.reply_to_message_id = request.message_id
    peer_reply.source.user_id = "@charlotte_ai:example.test"
    peer_reply.metadata["conversation_mentioned"] = False
    await adapter.handle_message(peer_reply)

    assert adapter.delivered == []
    assert len(gate._passive_context["same-room"]) == 2

    gate._throttled_evaluate = AsyncMock(
        return_value=(5, "The latest open message invites a useful contribution.")
    )
    open_message = event(
        Platform.MATRIX,
        text="Could advertising be affected by that compatibility risk?",
    )
    open_message.message_id = "open-message"
    open_message.metadata["conversation_mentioned"] = False
    await adapter.handle_message(open_message)

    assert len(adapter.delivered) == 1
    assert adapter.delivered[0].text.startswith("Could advertising")
    assert "compatibility risk" in (adapter.delivered[0].channel_context or "")
    assert "same-room" not in gate._passive_context
    gate._throttled_evaluate.assert_awaited_once()
    await adapter.disconnect()


@pytest.mark.anyio
async def test_each_latest_message_is_scored_separately_with_recent_context(tmp_path):
    adapter = Adapter(Platform.MATRIX, {
        "relevance": {"enabled": True},
        "pingpong_guard": {"enabled": False},
    })
    gate = adapter.conversation_policy().relevance
    calls = []

    async def score(_room, text, transcript):
        calls.append((text, transcript))
        return (5, "Now relevant.") if text == "Advertising implications?" else (1, "Passive context.")

    gate._throttled_evaluate = score
    for index, text in enumerate((
        "Social campaign draft one",
        "Social campaign revision two",
        "Advertising implications?",
    )):
        message = event(Platform.MATRIX, text=text)
        message.message_id = f"latest-{index}"
        message.metadata["conversation_mentioned"] = False
        await adapter.handle_message(message)

    assert [text for text, _ in calls] == [
        "Social campaign draft one",
        "Social campaign revision two",
        "Advertising implications?",
    ]
    assert "Social campaign draft one" in calls[1][1]
    assert "Social campaign revision two" not in calls[1][1]
    assert "Social campaign draft one" in calls[2][1]
    assert "Social campaign revision two" in calls[2][1]
    assert len(adapter.delivered) == 1
    assert adapter.delivered[0].text.startswith("Advertising implications?")
    assert "Social campaign draft one" in (adapter.delivered[0].channel_context or "")
    assert "same-room" not in gate._passive_context
    await adapter.disconnect()


@pytest.mark.anyio
async def test_passive_context_survives_restart_and_is_consumed_after_dispatch(
    tmp_path
):
    settings = {
        "relevance": {"enabled": True},
        "pingpong_guard": {"enabled": False},
    }
    first = Adapter(Platform.MATRIX, settings)
    first_gate = first.conversation_policy().relevance
    first_gate._throttled_evaluate = AsyncMock(return_value=(1, "Passive context."))
    context_event = event(Platform.MATRIX, text="Long-lived social campaign detail")
    context_event.metadata["conversation_mentioned"] = False
    await first.handle_message(context_event)

    state_file = tmp_path / "groupchat" / "passive-context.matrix.json"
    assert state_file.exists()
    assert state_file.stat().st_mode & 0o777 == 0o600
    assert state_file.parent.stat().st_mode & 0o777 == 0o700
    persisted = json.loads(state_file.read_text())
    assert persisted["rooms"]["same-room"][0]["text"] == context_event.text
    await first.disconnect()

    second = Adapter(Platform.MATRIX, settings)
    second_gate = second.conversation_policy().relevance
    assert second_gate._passive_context["same-room"][0].text == context_event.text
    second_gate._throttled_evaluate = AsyncMock(return_value=(5, "Relevant now."))
    trigger = event(Platform.MATRIX, text="What are the advertising implications?")
    trigger.message_id = "restart-trigger"
    trigger.metadata["conversation_mentioned"] = False
    await second.handle_message(trigger)

    assert len(second.delivered) == 1
    assert context_event.text in (second.delivered[0].channel_context or "")
    assert not state_file.exists()
    await second.disconnect()


@pytest.mark.anyio
async def test_plain_own_name_is_an_immediate_direct_address(tmp_path):
    adapter = Adapter(Platform.MATRIX, {
        "relevance": {"enabled": True},
        "pingpong_guard": {"enabled": False},
    })
    gate = adapter.conversation_policy().relevance
    gate._name_pattern_for_room = lambda _room: gate._names_pattern(["charlotte"])
    gate._throttled_evaluate = AsyncMock(
        side_effect=AssertionError("direct name reached scorer")
    )

    message = event(Platform.MATRIX, text="Charlotte?")
    message.metadata["conversation_mentioned"] = False
    await adapter.handle_message(message)

    assert len(adapter.delivered) == 1
    assert adapter.delivered[0].text.startswith("Charlotte?")
    assert gate._pending == {}
    await adapter.disconnect()


def test_agent_name_matching_uses_word_boundaries():
    pattern = IntelligentReactionGate._names_pattern(
        ["lena", "charlotte ai", "charlotte_ai"]
    )
    assert pattern.search("Lena, bitte übernehmen.")
    assert pattern.search("charlotte ai?")
    assert pattern.search("@charlotte_ai:example.test bitte prüfen")
    assert not pattern.search("Elena, bitte übernehmen.")
    assert not pattern.search("charlotte_airport")


def test_peer_names_come_only_from_running_groupchat_participants(
    tmp_path, monkeypatch
):
    peer_home = tmp_path / "profiles" / "charlotte_weiss"
    peer_home.mkdir(parents=True)
    peer_home.joinpath("config.yaml").write_text(json.dumps({
        "plugins": {"enabled": ["groupchat"]},
        "platforms": {"matrix": {"enabled": True, "require_mention": False}},
        "groupchat": {
            "enabled": True,
            "platforms": ["matrix"],
            "relevance": {"enabled": True},
        },
    }))
    dormant_home = tmp_path / "profiles" / "dormant_agent"
    dormant_home.mkdir()
    dormant_home.joinpath("config.yaml").write_text(peer_home.joinpath("config.yaml").read_text())
    monkeypatch.setattr(
        "hermes_cli.profiles.list_profiles",
        lambda: [
            SimpleNamespace(
                name="charlotte_weiss", path=peer_home,
                gateway_running=True, display_name="Charlotte W.",
            ),
            SimpleNamespace(
                name="dormant_agent", path=dormant_home,
                gateway_running=False, display_name="",
            ),
        ],
    )

    adapter = Adapter(Platform.MATRIX, {"relevance": {"enabled": True}})
    names = adapter.conversation_policy().relevance._peer_names

    assert "charlotte" in names
    assert "charlotte weiss" in names
    assert "Charlotte W." in names
    assert "dormant" not in names


@pytest.mark.anyio
async def test_disabled_policy_is_identity(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = Adapter()
    msg = event()
    await adapter.handle_message(msg)
    assert adapter.delivered == [msg]
    assert adapter.conversation_policy().relevance is None
    assert (await adapter.send("room", "ok")).success
    assert adapter.sent == [("room", "ok")]


@pytest.mark.anyio
async def test_commands_bypass_gate(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = Adapter(Platform.MATRIX, {"relevance": {"enabled": True}, "pingpong_guard": {"enabled": False}})
    gate = adapter.conversation_policy().relevance
    gate.process = AsyncMock(side_effect=AssertionError("control command reached scorer"))
    msg = event(Platform.MATRIX, text="/stop", command=True)
    await adapter.handle_message(msg)
    assert adapter.delivered == [msg]


@pytest.mark.anyio
async def test_denied_sender_does_not_reach_scorer_or_context(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = Adapter(Platform.MATRIX, {"relevance": {"enabled": True}})
    adapter.set_authorization_check(lambda *args, **kwargs: False)
    policy = adapter.conversation_policy()
    policy.relevance.process = AsyncMock(side_effect=AssertionError("unauthorized scoring"))
    await adapter.handle_message(event(Platform.MATRIX))
    assert policy.last_inbound == {}
    assert policy.relevance._pending == {}


@pytest.mark.anyio
async def test_scopes_isolate_threads_workspaces_and_platforms(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = Adapter(settings={"relevance": {"enabled": True}})
    root = adapter.conversation_policy()
    first = root.for_conversation("same-room", {"slack_team_id": "team-a"}, "1")
    other_thread = root.for_conversation("same-room", {"slack_team_id": "team-a"}, "2")
    other_workspace = root.for_conversation("same-room", {"slack_team_id": "team-b"}, "1")
    first.last_inbound["same-room"] = "private"
    assert not other_thread.last_inbound
    assert not other_workspace.last_inbound
    assert first.relevance._context_file != other_thread.relevance._context_file
    assert first.relevance._context_file != other_workspace.relevance._context_file
    assert first.relevance._room_transcript is other_thread.relevance._room_transcript
    assert first.relevance._room_transcript is not other_workspace.relevance._room_transcript
    assert root.for_conversation("same-room", {"slack_team_id": "team-a", "thread_id": "1"}) is first
    telegram = Adapter(Platform.TELEGRAM, {"relevance": {"enabled": True}})
    assert telegram.conversation_policy().relevance._context_file != root.relevance._context_file
    await adapter.disconnect()
    await telegram.disconnect()


@pytest.mark.anyio
async def test_thread_lanes_share_clean_room_context_without_merging_queues(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = Adapter(Platform.MATRIX, {"relevance": {"enabled": True}})
    root = adapter.conversation_policy()
    first = root.for_conversation("same-room", {}, "thread-1")
    second = root.for_conversation("same-room", {}, "thread-2")

    assert first.relevance._room_transcript is second.relevance._room_transcript
    assert first.relevance._passive_context is second.relevance._passive_context
    assert (
        first.relevance._peer_targeted_event_ids
        is second.relevance._peer_targeted_event_ids
    )
    assert first.relevance._pending is not second.relevance._pending

    first.relevance._record_transcript(
        "same-room",
        sender="Moritz",
        text="Charlotte, please inspect your installed skills.",
        timestamp=1.0,
        event_id="earlier",
    )
    second.relevance._throttled_evaluate = AsyncMock(
        return_value=(5, "Elliptical follow-up for this agent.")
    )
    follow_up = event(
        Platform.MATRIX,
        thread="thread-2",
        text="Bastian, you too.",
    )
    follow_up.metadata["conversation_mentioned"] = False
    await adapter.handle_message(follow_up)

    assert len(adapter.delivered) == 1
    assert "Charlotte, please inspect your installed skills." in (
        adapter.delivered[0].channel_context or ""
    )
    assert adapter.delivered[0].source.thread_id == "thread-2"
    await adapter.disconnect()


@pytest.mark.anyio
async def test_room_typing_cancels_voice_claims_in_every_thread_lane(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = Adapter(Platform.MATRIX, {"relevance": {"enabled": True}})
    root = adapter.conversation_policy()
    first = root.for_conversation("same-room", {}, "voice-thread-1")
    second = root.for_conversation("same-room", {}, "voice-thread-2")

    tasks = []
    for index, policy in enumerate((first, second), start=1):
        task = asyncio.create_task(asyncio.sleep(3600))
        tasks.append(task)
        voice_event = event(
            Platform.MATRIX,
            thread=f"voice-thread-{index}",
            text="[Voice message]",
        )
        voice_event.message_id = f"voice-{index}"
        policy.relevance._voice_pending["same-room"] = {
            "task": task,
            "event": voice_event,
        }

    root.typing("same-room")
    await asyncio.sleep(0)

    assert all(task.cancelled() for task in tasks)
    assert all(
        "same-room" not in policy.relevance._voice_pending
        for policy in (first, second)
    )
    await adapter.disconnect()


@pytest.mark.anyio
async def test_other_agent_progress_never_reaches_buffer_or_room_context(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = Adapter(Platform.MATRIX, {"relevance": {
        "enabled": True,
        "system_patterns": [],
        "multiline_patterns": [],
        "literal_phrases": [],
    }})
    gate = adapter.conversation_policy().relevance
    gate._own_user_id = lambda: "@this_agent:example.test"
    gate._throttled_evaluate = AsyncMock(return_value=(3, "Wait briefly."))

    request = event(Platform.MATRIX, text="Please inspect your skills")
    request.metadata["conversation_mentioned"] = False
    await adapter.handle_message(request)
    progress = event(Platform.MATRIX, text="📚 Reading skill relevance-context")
    progress.user_id = "@other_agent:example.test"
    progress.source.user_id = progress.user_id
    progress.metadata["conversation_mentioned"] = False
    await adapter.handle_message(progress)

    pending = gate._pending["same-room"].events
    assert [queued.text for queued, _ in pending] == ["Please inspect your skills"]
    assert all(
        "Reading skill" not in text
        for _, text, _, _ in gate._room_transcript.get("same-room", [])
    )
    await adapter.disconnect()


@pytest.mark.anyio
async def test_real_guard_worker_suppresses_patterns(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = Adapter(settings={"pingpong_guard": {"enabled": True}})
    assert (await adapter.send("room", "*(bleibt still)*")).success
    assert adapter.sent == []
    text = "This is a substantive reply with enough information to exceed sixty characters."
    assert (await adapter.send("room", text)).success
    assert adapter.sent == [("room", text)]


@pytest.mark.anyio
async def test_worker_failure_fails_open(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = Adapter(settings={"pingpong_guard": {"enabled": True}})
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(side_effect=OSError("unavailable")))
    assert (await adapter.send("room", "hello")).success
    assert adapter.sent == [("room", "hello")]


@pytest.mark.anyio
async def test_failed_send_can_retry_and_success_is_deduplicated(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = Adapter(settings={"relevance": {"enabled": True}})
    adapter.fail_send = True
    assert not (await adapter.send("room", "result")).success
    adapter.fail_send = False
    assert (await adapter.send("room", "result")).message_id == "1"
    assert (await adapter.send("room", "result")).message_id is None
    assert adapter.sent == [("room", "result")]
    gate = adapter.conversation_policy().for_conversation("room").relevance
    assert gate._own_message_ids["room"] == {"1"}
    await adapter.disconnect()


@pytest.mark.anyio
async def test_shutdown_cancels_delayed_flush(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = Adapter(Platform.MATRIX, {"relevance": {"enabled": True}, "pingpong_guard": {"enabled": False}})
    gate = adapter.conversation_policy().relevance
    task = gate._spawn(gate._delayed_flush("room", 3600, "later"))
    await adapter.disconnect()
    assert task.done()
    assert gate._closed
    assert adapter.delivered == []


def test_legacy_matrix_settings_do_not_leak_to_other_platforms(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MATRIX_INTELLIGENT_REACTION", "true")
    monkeypatch.setenv("MATRIX_INTELLIGENT_REACTION_TYPING_DELAY", "99")
    adapter = Adapter(settings={"relevance": {"enabled": True, "typing_delay": 2}})
    assert adapter.conversation_policy().relevance._typing_delay == 2
    assert not adapter.conversation_policy().guard_enabled
    assert Adapter().conversation_policy().relevance is None


def test_yaml_configuration_real_load_and_streaming_gate(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {"groupchat": {"enabled": True, "platforms": ["slack"],
        "relevance": {"enabled": True}, "pingpong_guard": {"enabled": True}
    }}
    (tmp_path / "config.yaml").write_text(json.dumps(config))
    adapter = Adapter()
    adapter.config.extra = {}
    assert adapter.conversation_policy().guard_enabled
    assert adapter.conversation_policy().relevance is not None
    assert requires_buffered_delivery(config, "slack")
    assert not requires_buffered_delivery(config, "matrix")
    assert not requires_buffered_delivery(config, "telegram")


@pytest.mark.anyio
async def test_one_filter_model_selection_drives_both_filters(monkeypatch, tmp_path):
    import io
    import urllib.request
    from plugins.groupchat.pingpong_guard import should_suppress
    calls = []
    async def inline_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    def response(request, **kwargs):
        payload = json.loads(request.data)
        calls.append(payload["model"])
        content = "SUPPRESS" if payload["max_tokens"] == 5 else '{"score": 4, "rationale": "Relevant"}'
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode())

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(urllib.request, "urlopen", response)
    # This test exercises shared model selection, not asyncio's executor.
    # Running inline also avoids a Python 3.13/AnyIO executor-shutdown hang in
    # minimal system-Python environments.
    monkeypatch.setattr(asyncio, "to_thread", inline_to_thread)
    model = {"provider": "openrouter", "model": "chosen/model", "fallbacks": []}
    adapter = Adapter(settings={"filter_model": model, "relevance": {"enabled": True}})
    gate = adapter.conversation_policy().relevance
    assert (await gate._call_model_chain("score this"))
    assert should_suppress("a vague acknowledgement", "context", model)
    assert calls == ["chosen/model"] * 2
    await adapter.disconnect()


def test_model_failure_tries_configured_fallback(monkeypatch):
    from plugins.groupchat import models
    calls = []
    def complete(choice, prompt, max_tokens):
        calls.append(choice["model"])
        if choice["model"] == "primary":
            raise RuntimeError("unavailable")
        return "SEND"
    monkeypatch.setattr(models, "complete", complete)
    raw, audit = models.run_chain({"provider": "mistral", "model": "primary",
        "fallbacks": [{"provider": "openrouter", "model": "backup"}]}, "check")
    assert raw == "SEND" and audit["fallback_used"]
    assert calls == ["primary", "backup"]


def test_migration_copies_behavior_not_credentials_and_is_idempotent():
    from plugins.groupchat.config import migrate_legacy
    original = {"model": {"provider": "openai-codex", "default": "profile-model"}}
    migrated = migrate_legacy(original, {"MATRIX_INTELLIGENT_REACTION": "true",
        "MATRIX_INTELLIGENT_REACTION_TYPING_DELAY": "7", "OPENROUTER_API_KEY": "NEVER-COPY"})
    assert migrated["groupchat"]["relevance"]["typing_delay"] == 7
    assert migrated["groupchat"]["enabled"]
    assert migrated["plugins"]["enabled"] == ["groupchat"]
    assert "groupchat" not in original
    assert "NEVER-COPY" not in json.dumps(migrated)
    assert migrate_legacy(migrated, {}) == migrated


def test_ui_rejects_invalid_models_and_limits():
    from plugins.groupchat.config import validate_settings
    with pytest.raises(ValueError):
        validate_settings({"filter_model": {"provider": "unsupported"}})
    with pytest.raises(ValueError):
        validate_settings({"pingpong_guard": {"min_chars": -1}})
    with pytest.raises(ValueError):
        validate_settings({"relevance": {"score_delays": {5: 0}}})


def test_active_secret_scope_cannot_borrow_default_credentials(monkeypatch):
    from plugins.groupchat.models import credential
    monkeypatch.setenv("OPENROUTER_API_KEY", "default-profile-key")
    monkeypatch.setattr("agent.secret_scope.current_secret_scope", lambda: {})
    assert credential("OPENROUTER_API_KEY") == ""


@pytest.mark.anyio
async def test_explicit_groupchat_off_wins_over_legacy_matrix_flag(monkeypatch):
    monkeypatch.setenv("MATRIX_INTELLIGENT_REACTION", "true")
    adapter = Adapter(Platform.MATRIX, {"enabled": False})
    policy = adapter.conversation_policy()
    assert policy.relevance is None and not policy.guard_enabled
    await adapter.handle_message(event(Platform.MATRIX))
    assert "Relevance assessment" not in adapter.delivered[0].text

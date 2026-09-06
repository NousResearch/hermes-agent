import asyncio
import json
import logging
from asyncio import get_running_loop
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from gateway.becky_loop_summarizer import (
    LoopSummary,
    SummaryUnavailable,
    _ConversationTooLarge,
    _SummaryValidationError,
)
from gateway import becky_loops
from gateway.becky_loop_reply import ReplyUnavailable
from gateway.becky_loops import BeckyLoopsBridgeServer, BeckyLoopsConfig


SOURCE_REF = "loop_" + "A" * 43
REVISION = "sha256:" + "a" * 64
NEW_REVISION = "sha256:" + "b" * 64
IDEMPOTENCY_KEY = "8c9c8217-cc0f-463d-a430-173f1802edb2"
SECOND_IDEMPOTENCY_KEY = "f42862c7-55d7-423e-bc4f-b00b8538f0c5"


class FakeStore:
    def __init__(self) -> None:
        self.rows = [
            {
                "source_ref": SOURCE_REF,
                "title": "Basement planning",
                "source_state": "active",
                "revision": REVISION,
                "message_count": 3,
                "created_at": datetime(2026, 8, 13, 20, 0, tzinfo=UTC),
                "updated_at": datetime(2026, 8, 13, 20, 3, tzinfo=UTC),
                "session_id": "session-1",
                "thread_id": "20197",
            }
        ]
        self.transcripts = {
            "session-1": [
                {
                    "role": "user",
                    "content": "We decided to use the smaller layout.",
                    "timestamp": 1_755_104_400.0,
                },
                {
                    "role": "assistant",
                    "content": "I will prepare the final plan next.",
                    "timestamp": 1_755_104_460.0,
                },
            ]
        }

    def list_topics(self, chat_id: str) -> list[dict]:
        assert chat_id == "123456789"
        return list(self.rows)

    def get_topic(self, source_ref: str) -> dict | None:
        return next((row for row in self.rows if row["source_ref"] == source_ref), None)

    def transcript(self, session_id: str) -> list[dict]:
        return list(self.transcripts.get(session_id, []))


class CloseableStore(FakeStore):
    def __init__(self) -> None:
        super().__init__()
        self.ended_topics: list[tuple[str, str, str]] = []

    def end_topic_session(self, *, chat_id: str, thread_id: str, reason: str) -> bool:
        self.ended_topics.append((chat_id, thread_id, reason))
        return True


class PersistingShortcutStore(FakeStore):
    def __init__(self) -> None:
        super().__init__()
        self.shortcut_topics: list[dict[str, str]] = []
        self.shortcut_answers: list[dict[str, str]] = []

    def record_shortcut_topic(
        self, *, title: str, text: str, topic_id: str, message_id: str
    ) -> str:
        self.shortcut_topics.append({
            "title": title,
            "text": text,
            "topic_id": topic_id,
            "message_id": message_id,
        })
        return "shortcut-session"

    def record_shortcut_answer(
        self, *, session_id: str, text: str, message_id: str
    ) -> None:
        self.shortcut_answers.append({
            "session_id": session_id,
            "text": text,
            "message_id": message_id,
        })


class RacingStore(FakeStore):
    def revision_for_topic(self, row: dict, transcript: list[dict]) -> str:
        del row, transcript
        return "sha256:" + "b" * 64


class SequencedRevisionStore(FakeStore):
    def __init__(self, revisions: list[str | Exception]) -> None:
        super().__init__()
        self.revisions = list(revisions)

    def revision_for_topic(self, row: dict, transcript: list[dict]) -> str:
        del row, transcript
        if len(self.revisions) > 1:
            revision = self.revisions.pop(0)
        else:
            revision = self.revisions[0]
        if isinstance(revision, Exception):
            raise revision
        return revision


def loop_summary() -> LoopSummary:
    return LoopSummary(
        summary="Use the smaller layout.",
        decisions=["The smaller layout was approved."],
        unresolved_items=["Confirm the installation date."],
        next_action="Prepare the final plan.",
        waiting_on="becky",
        key_events=[
            {
                "occurred_at": "2026-08-13T20:00:00+00:00",
                "text": "The smaller layout was chosen.",
            }
        ],
        final_outcome=None,
    )


class FakeSummarizer:
    def __init__(
        self,
        *,
        result: LoopSummary | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.result = result or loop_summary()
        self.failure = failure
        self.calls: list[dict[str, Any]] = []

    async def summarize(
        self,
        *,
        row: dict[str, Any],
        transcript: list[dict[str, Any]],
        deadline: float,
    ) -> LoopSummary:
        self.calls.append({"row": row, "transcript": transcript, "deadline": deadline})
        if self.failure is not None:
            raise self.failure
        return self.result


class FakeTopicSender:
    def __init__(self, outcomes: list[object] | None = None) -> None:
        self.outcomes = list(outcomes or [])
        self.calls: list[dict[str, Any]] = []
        self.next_message_id = 100

    async def send_topic(
        self,
        *,
        chat_id: str,
        thread_id: str,
        text: str,
        reply_to_message_id: str | None,
    ) -> object:
        self.calls.append({
            "chat_id": chat_id,
            "thread_id": thread_id,
            "text": text,
            "reply_to_message_id": reply_to_message_id,
        })
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        self.next_message_id += 1
        return becky_loops.TopicSendReceipt(message_id=str(self.next_message_id))


class FakeTopicController:
    method = "bot_api_private_topic"

    def __init__(self, outcomes: list[object] | None = None) -> None:
        self.outcomes = list(outcomes or [])
        self.calls: list[dict[str, str]] = []
        self.is_connected = True
        self.supports_close = True

    async def close_topic(self, *, chat_id: str, thread_id: str) -> datetime:
        self.calls.append({"chat_id": chat_id, "thread_id": thread_id})
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return datetime(2026, 8, 14, 12, 0, tzinfo=UTC)


class BlockingTopicSender(FakeTopicSender):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def send_topic(self, **kwargs: Any) -> object:
        self.calls.append(dict(kwargs))
        if len(self.calls) == 1:
            self.started.set()
            await self.release.wait()
        return becky_loops.TopicSendReceipt(message_id=str(400 + len(self.calls)))


class FakeReplyGenerator:
    def __init__(self, outcomes: list[object] | None = None) -> None:
        self.outcomes = list(outcomes or ["Start with the insulation quote."])
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        *,
        row: dict[str, Any],
        transcript: list[dict[str, Any]],
        comment: str,
        deadline: float,
    ) -> str:
        self.calls.append({
            "row": row,
            "transcript": transcript,
            "comment": comment,
            "deadline": deadline,
        })
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return str(outcome)


class FakeAgentDispatcher:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def dispatch(
        self,
        *,
        chat_id: str,
        thread_id: str,
        session_id: str,
        text: str,
        reply_to_message_id: str,
    ) -> None:
        self.calls.append({
            "chat_id": chat_id,
            "thread_id": thread_id,
            "session_id": session_id,
            "text": text,
            "reply_to_message_id": reply_to_message_id,
        })


class RevisionMutatingReplyGenerator(FakeReplyGenerator):
    def __init__(self, store: FakeStore) -> None:
        super().__init__(["Retained normalized answer."])
        self.store = store

    async def generate(self, **kwargs: Any) -> str:
        answer = await super().generate(**kwargs)
        self.store.rows[0]["revision"] = NEW_REVISION
        return answer


class BlockingReplyGenerator(FakeReplyGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()

    async def generate(self, **kwargs: Any) -> str:
        self.calls.append(dict(kwargs))
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class FakeTelegramAdapter:
    def __init__(self, outcomes: list[object] | None = None) -> None:
        self.outcomes = list(outcomes or [])
        self.calls: list[dict[str, Any]] = []

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> object:
        self.calls.append({
            "chat_id": chat_id,
            "content": content,
            "reply_to": reply_to,
            "metadata": metadata,
        })
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return SimpleNamespace(
            success=True,
            message_id=str(900 + len(self.calls)),
            raw_response={"thread_fallback": False},
        )


class RevisionMutatingTopicSender(FakeTopicSender):
    def __init__(self, store: FakeStore) -> None:
        super().__init__()
        self.store = store

    async def send_topic(self, **kwargs: Any) -> object:
        receipt = await super().send_topic(**kwargs)
        if len(self.calls) == 1:
            self.store.rows[0]["revision"] = NEW_REVISION
        return receipt


class ProjectionDB:
    def __init__(self) -> None:
        self.rows = [
            {
                "id": "root",
                "source": "telegram",
                "chat_id": "123456789",
                "thread_id": "root-thread",
                "parent_session_id": None,
                "model_config": "{}",
                "title": "Root",
                "started_at": 1_755_104_400.0,
                "last_active": 1_755_104_460.0,
                "message_count": 1,
                "ended_at": None,
            },
            {
                "id": "branch",
                "source": "telegram",
                "chat_id": "123456789",
                "thread_id": "branch-thread",
                "parent_session_id": "root",
                "model_config": "{}",
                "title": "Branch",
                "started_at": 1_755_104_400.0,
                "last_active": 1_755_104_460.0,
                "message_count": 1,
                "ended_at": None,
            },
            {
                "id": "delegate",
                "source": "telegram",
                "chat_id": "123456789",
                "thread_id": "delegate-thread",
                "parent_session_id": None,
                "model_config": '{"_delegate_from": "root"}',
                "title": "Delegate",
                "started_at": 1_755_104_400.0,
                "last_active": 1_755_104_460.0,
                "message_count": 1,
                "ended_at": None,
            },
        ]

    def list_sessions_rich(self, **kwargs: object) -> list[dict]:
        del kwargs
        return list(self.rows)

    def get_messages(
        self, session_id: str, include_inactive: bool = False
    ) -> list[dict]:
        del session_id, include_inactive
        return [{"role": "user", "content": "hello", "timestamp": 1_755_104_400.0}]


class InactiveMessageProjectionDB(ProjectionDB):
    def get_messages(
        self, session_id: str, include_inactive: bool = False
    ) -> list[dict]:
        del session_id
        if include_inactive:
            return [
                {
                    "role": "user",
                    "content": "compacted user message",
                    "platform_message_id": "221",
                    "timestamp": 1_755_104_400.0,
                },
                {
                    "role": "assistant",
                    "content": "compacted assistant message",
                    "platform_message_id": "222",
                    "timestamp": 1_755_104_460.0,
                },
            ]
        return [{"role": "user", "content": "active", "timestamp": 1_755_104_500.0}]


class ShortcutProjectionDB:
    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, object]] = {}
        self.messages: dict[str, list[dict[str, object]]] = {}

    def get_session(self, session_id: str) -> dict[str, object] | None:
        return self.sessions.get(session_id)

    def create_session(self, session_id: str, source: str, **kwargs: object) -> str:
        self.sessions[session_id] = {"id": session_id, "source": source, **kwargs}
        self.messages.setdefault(session_id, [])
        return session_id

    def set_session_title(self, session_id: str, title: str) -> bool:
        self.sessions[session_id]["title"] = title
        return True

    def get_messages(
        self, session_id: str, include_inactive: bool = False
    ) -> list[dict[str, object]]:
        del include_inactive
        return list(self.messages.get(session_id, []))

    def append_message(self, session_id: str, role: str, **kwargs: object) -> int:
        self.messages.setdefault(session_id, []).append({"role": role, **kwargs})
        return len(self.messages[session_id])


class GatewayShortcutSessionStore:
    def __init__(self, db: ShortcutProjectionDB) -> None:
        self.db = db
        self.source = None
        self.entry = None

    def get_or_create_session(self, source):
        self.source = source
        if self.entry is None:
            session_id = "gateway-session"
            session_key = (
                f"agent:main:telegram:group:{source.chat_id}:{source.thread_id}"
            )
            self.db.create_session(
                session_id=session_id,
                source="telegram",
                session_key=session_key,
                chat_id=source.chat_id,
                chat_type=source.chat_type,
                thread_id=source.thread_id,
            )
            self.entry = SimpleNamespace(
                session_id=session_id,
                session_key=session_key,
            )
        return self.entry

    def append_to_transcript(self, session_id: str, message: dict[str, object]) -> None:
        self.db.append_message(
            session_id,
            str(message.get("role") or "unknown"),
            content=message.get("content"),
            platform_message_id=message.get("message_id"),
            timestamp=message.get("timestamp"),
        )


def config(*, port: int = 0, topic_reply: str = "unavailable") -> BeckyLoopsConfig:
    return BeckyLoopsConfig(
        enabled=True,
        chat_id="123456789",
        token="t" * 64,
        port=port,
        topic_control="unavailable",
        topic_reply=topic_reply,
    )


def control_config(*, port: int = 0) -> BeckyLoopsConfig:
    return BeckyLoopsConfig(
        enabled=True,
        chat_id="123456789",
        token="t" * 64,
        port=port,
        topic_control="bot_api_private_topic",
        topic_reply="unavailable",
    )


def reply_params(
    *,
    text: str = "Can you clarify the next step?",
    idempotency_key: str = IDEMPOTENCY_KEY,
    revision: str = REVISION,
) -> dict[str, str]:
    return {
        "source_ref": SOURCE_REF,
        "expected_revision": revision,
        "text": text,
        "idempotency_key": idempotency_key,
    }


def retry_params(
    *,
    idempotency_key: str = IDEMPOTENCY_KEY,
    revision: str = REVISION,
) -> dict[str, str]:
    return {
        "source_ref": SOURCE_REF,
        "expected_revision": revision,
        "idempotency_key": idempotency_key,
    }


def new_topic_reply_params(
    *,
    title: str = "Trip planning",
    text: str = "Please help me plan this.",
    topic_id: str = "42",
    message_id: str = "101",
    idempotency_key: str = IDEMPOTENCY_KEY,
) -> dict[str, str]:
    return {
        "title": title,
        "text": text,
        "topic_id": topic_id,
        "message_id": message_id,
        "idempotency_key": idempotency_key,
        "auto_close_policy": "simple_calendar_todoist_success",
    }


def reply_server(
    *,
    store: FakeStore | None = None,
    sender: FakeTopicSender | None = None,
    generator: FakeReplyGenerator | None = None,
    agent_dispatcher: FakeAgentDispatcher | None = None,
) -> BeckyLoopsBridgeServer:
    return BeckyLoopsBridgeServer(
        config=config(topic_reply="bot_api_private_topic"),
        store=store or FakeStore(),
        summarizer=FakeSummarizer(),
        topic_sender=sender or FakeTopicSender(),
        reply_generator=generator or FakeReplyGenerator(),
        agent_dispatcher=agent_dispatcher,
    )


def _write_bridge_config(
    path: Any,
    *,
    enabled: Any = True,
    proven_topic_reply: Any = True,
    telegram_chat_id: Any = "123456789",
    telegram_topic_id: Any = "20197",
    include_platform_topic: bool = False,
    platform_chat_id: Any = "123456789",
    platform_topic_id: Any = "20197",
) -> Any:
    platform = ""
    if include_platform_topic:
        platform = (
            "platforms:\n"
            "  telegram:\n"
            "    extra:\n"
            "      dm_topics:\n"
            f"        - chat_id: {platform_chat_id}\n"
            "          topics:\n"
            "            - name: Disposable proof\n"
            f"              thread_id: {platform_topic_id}\n"
        )
    topic_selector = (
        f"    telegram_topic_id: {json.dumps(telegram_topic_id)}\n"
        if telegram_topic_id is not None
        else ""
    )
    contents = (
        f"{platform}"
        "gateway:\n"
        "  becky_loops:\n"
        f"    enabled: {json.dumps(enabled)}\n"
        f"    proven_topic_reply: {json.dumps(proven_topic_reply)}\n"
        f"    telegram_chat_id: {json.dumps(telegram_chat_id)}\n"
        f"{topic_selector}"
        "    port: 9120\n"
    )
    path.write_text(contents, encoding="utf-8")
    return path


def test_load_config_requires_exact_dual_reply_proof_and_configured_topic(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_bridge_config(tmp_path / "config.yaml", include_platform_topic=True)
    monkeypatch.setenv("HERMES_BECKY_LOOPS_TOKEN", "t" * 64)
    monkeypatch.setenv("HERMES_BECKY_LOOPS_PROVEN_TOPIC_REPLY", "1")

    loaded = becky_loops.load_becky_loops_config(path)

    assert loaded is not None
    assert loaded.topic_reply == "bot_api_private_topic"
    assert loaded.chat_id == "123456789"


def test_load_config_requires_separate_close_control_proof(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_bridge_config(tmp_path / "config.yaml", include_platform_topic=True)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "    proven_topic_reply: true\n",
            "    proven_topic_reply: true\n    proven_topic_control: true\n",
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_BECKY_LOOPS_TOKEN", "t" * 64)
    monkeypatch.setenv("HERMES_BECKY_LOOPS_PROVEN_TOPIC_REPLY", "1")
    monkeypatch.setenv("HERMES_BECKY_LOOPS_PROVEN_TOPIC_CONTROL", "1")

    loaded = becky_loops.load_becky_loops_config(path)

    assert loaded is not None
    assert loaded.topic_control == "bot_api_private_topic"


@pytest.mark.parametrize(
    ("env_value", "proven_value", "chat_id", "topic_id", "platform_topic"),
    [
        (None, True, "123456789", "20197", False),
        ("0", True, "123456789", "20197", False),
        ("true", True, "123456789", "20197", False),
        ("1", False, "123456789", "20197", False),
        ("1", "true", "123456789", "20197", False),
        ("1", True, "not-a-chat", "20197", False),
        ("1", True, "123456789", "not-a-topic", False),
        ("1", True, "123456789", None, False),
    ],
)
def test_load_config_keeps_reply_unavailable_for_missing_or_invalid_proof(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    env_value: str | None,
    proven_value: Any,
    chat_id: Any,
    topic_id: Any,
    platform_topic: bool,
) -> None:
    path = _write_bridge_config(
        tmp_path / "config.yaml",
        proven_topic_reply=proven_value,
        telegram_chat_id=chat_id,
        telegram_topic_id=topic_id,
        include_platform_topic=platform_topic,
    )
    monkeypatch.setenv("HERMES_BECKY_LOOPS_TOKEN", "t" * 64)
    if env_value is None:
        monkeypatch.delenv("HERMES_BECKY_LOOPS_PROVEN_TOPIC_REPLY", raising=False)
    else:
        monkeypatch.setenv("HERMES_BECKY_LOOPS_PROVEN_TOPIC_REPLY", env_value)

    loaded = becky_loops.load_becky_loops_config(path)

    assert loaded is not None
    assert loaded.topic_reply == "unavailable"


def test_load_config_accepts_topic_from_telegram_dm_topics_when_explicit_id_absent(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_bridge_config(
        tmp_path / "config.yaml",
        telegram_topic_id=None,
        include_platform_topic=True,
    )
    monkeypatch.setenv("HERMES_BECKY_LOOPS_TOKEN", "t" * 64)
    monkeypatch.setenv("HERMES_BECKY_LOOPS_PROVEN_TOPIC_REPLY", "1")

    loaded = becky_loops.load_becky_loops_config(path)

    assert loaded is not None
    assert loaded.topic_reply == "bot_api_private_topic"


@pytest.mark.parametrize(
    ("chat_id", "topic_id", "platform_chat_id", "platform_topic_id"),
    [
        ("123456789", "20198", "123456789", "20197"),
        ("987654321", "20197", "123456789", "20197"),
    ],
)
def test_load_config_rejects_explicit_topic_not_bound_to_configured_chat(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    chat_id: str,
    topic_id: str,
    platform_chat_id: str,
    platform_topic_id: str,
) -> None:
    path = _write_bridge_config(
        tmp_path / "config.yaml",
        telegram_chat_id=chat_id,
        telegram_topic_id=topic_id,
        include_platform_topic=True,
        platform_chat_id=platform_chat_id,
        platform_topic_id=platform_topic_id,
    )
    monkeypatch.setenv("HERMES_BECKY_LOOPS_TOKEN", "t" * 64)
    monkeypatch.setenv("HERMES_BECKY_LOOPS_PROVEN_TOPIC_REPLY", "1")

    loaded = becky_loops.load_becky_loops_config(path)

    assert loaded is not None
    assert loaded.topic_reply == "unavailable"


def test_load_config_requires_a_real_configured_dm_topic_for_explicit_proof(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_bridge_config(
        tmp_path / "config.yaml",
        telegram_topic_id="20197",
        include_platform_topic=False,
    )
    monkeypatch.setenv("HERMES_BECKY_LOOPS_TOKEN", "t" * 64)
    monkeypatch.setenv("HERMES_BECKY_LOOPS_PROVEN_TOPIC_REPLY", "1")

    loaded = becky_loops.load_becky_loops_config(path)

    assert loaded is not None
    assert loaded.topic_reply == "unavailable"


def test_load_config_rejects_conflicting_explicit_topic_aliases(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_bridge_config(
        tmp_path / "config.yaml",
        telegram_topic_id="20197",
        include_platform_topic=True,
    )
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "    port: 9120\n", '    telegram_thread_id: "20198"\n    port: 9120\n'
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_BECKY_LOOPS_TOKEN", "t" * 64)
    monkeypatch.setenv("HERMES_BECKY_LOOPS_PROVEN_TOPIC_REPLY", "1")

    loaded = becky_loops.load_becky_loops_config(path)

    assert loaded is not None
    assert loaded.topic_reply == "unavailable"


def test_load_config_rejects_invalid_explicit_topic_alias(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_bridge_config(
        tmp_path / "config.yaml",
        telegram_topic_id="not-a-topic",
        include_platform_topic=True,
    )
    monkeypatch.setenv("HERMES_BECKY_LOOPS_TOKEN", "t" * 64)
    monkeypatch.setenv("HERMES_BECKY_LOOPS_PROVEN_TOPIC_REPLY", "1")

    loaded = becky_loops.load_becky_loops_config(path)

    assert loaded is not None
    assert loaded.topic_reply == "unavailable"


async def rpc(ws, request_id: int, method: str, params: dict) -> dict:
    await ws.send(
        json.dumps({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        })
    )
    return json.loads(await ws.recv())


@pytest.mark.asyncio
async def test_bridge_auth_ready_capabilities_and_list() -> None:
    server = BeckyLoopsBridgeServer(
        config=config(), store=FakeStore(), summarizer=FakeSummarizer()
    )
    await server.start()
    try:
        async with connect(
            f"ws://127.0.0.1:{server.bound_port}/api/ws?token={'t' * 64}"
        ) as ws:
            assert json.loads(await ws.recv()) == {
                "jsonrpc": "2.0",
                "method": "event",
                "params": {"type": "gateway.ready", "payload": {"skin": {}}},
            }
            capabilities = await rpc(ws, 1, "becky.loops.capabilities", {})
            assert capabilities["result"] == {
                "schema_version": "2",
                "summary_schema_version": "1",
                "methods": [
                    "list",
                    "summarize",
                    "close",
                    "reopen",
                    "reply",
                    "reply_retry",
                ],
                "topic_control": "unavailable",
                "topic_reply": "unavailable",
                "same_topic_reopen": False,
                "new_session_fallback": True,
                "max_request_bytes": 65_536,
                "max_response_bytes": 262_144,
            }
            listed = await rpc(ws, 2, "becky.loops.list", {})
            assert listed["result"]["loops"][0]["source_ref"] == SOURCE_REF
            assert "session_id" not in listed["result"]["loops"][0]
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_bridge_advertises_reply_only_with_proven_injected_sender() -> None:
    sender = FakeTopicSender()
    generator = FakeReplyGenerator()
    server = reply_server(sender=sender, generator=generator)

    capabilities = await server._method("becky.loops.capabilities", {})

    assert capabilities["topic_control"] == "unavailable"
    assert capabilities["topic_reply"] == "bot_api_private_topic"


@pytest.mark.asyncio
async def test_bridge_answers_a_new_topic_without_exposing_answer_text() -> None:
    sender = FakeTopicSender()
    generator = FakeReplyGenerator(["Use the smaller layout."])
    server = reply_server(sender=sender, generator=generator)

    response = await server._dispatch(json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "becky.loops.answer_new_topic",
        "params": new_topic_reply_params(),
    }))

    assert response == {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"schema_version": "1", "answer_state": "answered"},
    }
    assert sender.calls == [
        {
            "chat_id": "123456789",
            "thread_id": "42",
            "text": "Use the smaller layout.",
            "reply_to_message_id": "101",
        }
    ]
    assert generator.calls[0]["row"]["title"] == "Trip planning"
    assert generator.calls[0]["transcript"][0]["content"] == (
        "Please help me plan this."
    )


@pytest.mark.asyncio
async def test_new_topic_hands_off_to_real_agent_when_session_is_available() -> None:
    store = PersistingShortcutStore()
    sender = FakeTopicSender()
    generator = FakeReplyGenerator(["This fallback must not run."])
    dispatch_calls: list[dict[str, Any]] = []

    async def dispatcher(**kwargs: Any) -> None:
        dispatch_calls.append(dict(kwargs))

    server = reply_server(
        store=store,
        sender=sender,
        generator=generator,
        agent_dispatcher=dispatcher,  # type: ignore[arg-type]
    )

    response = await server._dispatch(json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "becky.loops.answer_new_topic",
        "params": new_topic_reply_params(
            title="Daily Storage Check",
            text="Create a daily storage check.",
            topic_id="44",
            message_id="104",
        ),
    }))

    assert response["result"] == {
        "schema_version": "1",
        "answer_state": "answer_pending",
    }
    assert sender.calls == []
    assert generator.calls == []
    assert dispatch_calls == [
        {
            "chat_id": "123456789",
            "thread_id": "44",
            "session_id": "shortcut-session",
            "text": "Create a daily storage check.",
            "reply_to_message_id": "104",
            "auto_close_policy": "simple_calendar_todoist_success",
            "new_topic": True,
        }
    ]


@pytest.mark.asyncio
async def test_new_topic_answer_persists_exchange_for_loop_projection() -> None:
    store = PersistingShortcutStore()
    sender = FakeTopicSender()
    generator = FakeReplyGenerator(["The audit needs an export first."])
    server = reply_server(store=store, sender=sender, generator=generator)

    response = await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "becky.loops.answer_new_topic",
            "params": new_topic_reply_params(
                title="Energy Audit Request",
                text="Please audit the last week's energy usage.",
                topic_id="44",
                message_id="104",
            ),
        })
    )

    assert response["result"] == {
        "schema_version": "1",
        "answer_state": "answered",
    }
    assert store.shortcut_topics == [
        {
            "title": "Energy Audit Request",
            "text": "Please audit the last week's energy usage.",
            "topic_id": "44",
            "message_id": "104",
        }
    ]
    assert store.shortcut_answers == [
        {
            "session_id": "shortcut-session",
            "text": "The audit needs an export first.",
            "message_id": "101",
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params",
    [
        {**new_topic_reply_params(), "extra": True},
        {
            key: value
            for key, value in new_topic_reply_params().items()
            if key != "text"
        },
        {**new_topic_reply_params(), "topic_id": "0"},
        {**new_topic_reply_params(), "message_id": "not-a-number"},
        {**new_topic_reply_params(), "auto_close_policy": "anything"},
        {**new_topic_reply_params(), "title": ""},
        {**new_topic_reply_params(), "text": " "},
        {**new_topic_reply_params(), "idempotency_key": "not-a-uuid"},
    ],
)
async def test_bridge_new_topic_answer_requires_exact_bounded_params(
    params: dict[str, object],
) -> None:
    sender = FakeTopicSender()
    generator = FakeReplyGenerator()
    server = reply_server(sender=sender, generator=generator)

    response = await server._dispatch(json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "becky.loops.answer_new_topic",
        "params": params,
    }))

    assert response["error"] == {"code": -32600, "message": "protocol"}
    assert sender.calls == []
    assert generator.calls == []


@pytest.mark.asyncio
async def test_bridge_new_topic_answer_replays_same_key_without_sending_twice() -> None:
    sender = FakeTopicSender()
    generator = FakeReplyGenerator(["Use the smaller layout."])
    server = reply_server(sender=sender, generator=generator)
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "becky.loops.answer_new_topic",
        "params": new_topic_reply_params(),
    }

    first = await server._dispatch(json.dumps(request))
    second = await server._dispatch(json.dumps({**request, "id": 2}))

    assert first["result"] == second["result"]
    assert len(sender.calls) == 1
    assert len(generator.calls) == 1


@pytest.mark.asyncio
async def test_bridge_new_topic_answer_failure_is_safe_and_does_not_send() -> None:
    sender = FakeTopicSender()
    generator = FakeReplyGenerator([RuntimeError("provider secret")])
    server = reply_server(sender=sender, generator=generator)

    response = await server._dispatch(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "becky.loops.answer_new_topic",
                "params": new_topic_reply_params(),
            }
        )
    )

    assert response["result"] == {
        "schema_version": "1",
        "answer_state": "answer_unavailable",
    }
    assert sender.calls == []


@pytest.mark.asyncio
async def test_bridge_new_topic_answer_rejects_idempotency_conflict() -> None:
    server = reply_server()
    first = new_topic_reply_params()
    await server._dispatch(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "becky.loops.answer_new_topic",
                "params": first,
            }
        )
    )

    response = await server._dispatch(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "becky.loops.answer_new_topic",
                "params": {**first, "topic_id": "43"},
            }
        )
    )

    assert response["error"] == {
        "code": -32000,
        "message": "idempotency_conflict",
    }


@pytest.mark.asyncio
async def test_bridge_keeps_reply_unavailable_when_sender_is_missing() -> None:
    server = BeckyLoopsBridgeServer(
        config=config(topic_reply="bot_api_private_topic"),
        store=FakeStore(),
        summarizer=FakeSummarizer(),
        reply_generator=FakeReplyGenerator(),
    )

    capabilities = await server._method("becky.loops.capabilities", {})
    response = await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "becky.loops.reply",
            "params": reply_params(),
        })
    )

    assert capabilities["topic_reply"] == "unavailable"
    assert response == {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {"code": -32000, "message": "topic_reply_unavailable"},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params",
    [
        {**reply_params(), "extra": True},
        {key: value for key, value in reply_params().items() if key != "text"},
        {**reply_params(), "source_ref": "loop_bad"},
        {**reply_params(), "expected_revision": "SHA256:" + "a" * 64},
        {**reply_params(), "text": ""},
        {**reply_params(), "text": " "},
        {**reply_params(), "text": "x" * 5_001},
        {**reply_params(), "idempotency_key": "not-a-uuid"},
    ],
)
async def test_bridge_reply_requires_exact_bounded_params(
    params: dict[str, object],
) -> None:
    sender = FakeTopicSender()
    generator = FakeReplyGenerator()
    server = reply_server(sender=sender, generator=generator)

    response = await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "becky.loops.reply",
            "params": params,
        })
    )

    assert response["error"] == {"code": -32600, "message": "protocol"}
    assert sender.calls == []
    assert generator.calls == []


@pytest.mark.asyncio
async def test_bridge_accepts_a_five_thousand_character_comment() -> None:
    sender = FakeTopicSender()
    generator = FakeReplyGenerator()
    server = reply_server(sender=sender, generator=generator)

    response = await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "becky.loops.reply",
            "params": reply_params(text="x" * 5_000),
        })
    )

    assert response["result"]["answer_state"] == "answered"
    assert generator.calls[0]["comment"] == "x" * 5_000


@pytest.mark.asyncio
async def test_bridge_dispatches_app_reply_into_the_real_agent_session() -> None:
    sender = FakeTopicSender()
    generator = FakeReplyGenerator()
    dispatcher = FakeAgentDispatcher()
    server = reply_server(
        sender=sender,
        generator=generator,
        agent_dispatcher=dispatcher,
    )

    response = await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "becky.loops.reply",
            "params": reply_params(text="Please turn on the porch light."),
        })
    )

    assert response["result"]["answer_state"] == "answer_pending"
    assert sender.calls == [
        {
            "chat_id": "123456789",
            "thread_id": "20197",
            "text": "Please turn on the porch light.",
            "reply_to_message_id": None,
        }
    ]
    assert dispatcher.calls == [
        {
            "chat_id": "123456789",
            "thread_id": "20197",
            "session_id": "session-1",
            "text": "Please turn on the porch light.",
            "reply_to_message_id": "101",
        }
    ]


@pytest.mark.asyncio
async def test_bridge_dispatches_plain_callable_agent_callback() -> None:
    sender = FakeTopicSender()
    generator = FakeReplyGenerator()
    calls: list[dict[str, Any]] = []

    async def dispatch(**kwargs: Any) -> None:
        calls.append(dict(kwargs))

    server = reply_server(
        sender=sender,
        generator=generator,
        agent_dispatcher=dispatch,
    )

    response = await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "becky.loops.reply",
            "params": reply_params(text="Please summarize the latest update."),
        })
    )

    assert response["result"]["answer_state"] == "answer_pending"
    assert calls == [
        {
            "chat_id": "123456789",
            "thread_id": "20197",
            "session_id": "session-1",
            "text": "Please summarize the latest update.",
            "reply_to_message_id": "101",
        }
    ]
    assert generator.calls == []


@pytest.mark.asyncio
async def test_bridge_does_not_dispatch_the_same_pending_reply_twice() -> None:
    dispatcher = FakeAgentDispatcher()
    server = reply_server(agent_dispatcher=dispatcher)
    request = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "becky.loops.reply",
        "params": reply_params(),
    })

    first = await server._dispatch(request)
    second = await server._dispatch(json.dumps({**json.loads(request), "id": 2}))

    assert first["result"]["answer_state"] == "answer_pending"
    assert second["result"]["answer_state"] == "answer_pending"
    assert len(dispatcher.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params",
    [
        {**retry_params(), "extra": True},
        {key: value for key, value in retry_params().items() if key != "source_ref"},
        {**retry_params(), "expected_revision": "sha256:ABC"},
        {**retry_params(), "idempotency_key": "not-a-uuid"},
    ],
)
async def test_bridge_reply_retry_requires_exact_params(
    params: dict[str, object],
) -> None:
    server = reply_server()

    response = await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "becky.loops.reply_retry",
            "params": params,
        })
    )

    assert response["error"] == {"code": -32600, "message": "protocol"}


@pytest.mark.asyncio
async def test_bridge_rechecks_revision_before_reply_send_or_generation() -> None:
    sender = FakeTopicSender()
    generator = FakeReplyGenerator()
    server = reply_server(store=RacingStore(), sender=sender, generator=generator)

    response = await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "becky.loops.reply",
            "params": reply_params(),
        })
    )

    assert response["error"] == {"code": -32000, "message": "revision_conflict"}
    assert sender.calls == []
    assert generator.calls == []


@pytest.mark.asyncio
async def test_bridge_rechecks_revision_immediately_before_comment_send() -> None:
    store = SequencedRevisionStore([REVISION, NEW_REVISION])
    sender = FakeTopicSender()
    generator = FakeReplyGenerator()
    server = reply_server(store=store, sender=sender, generator=generator)

    response = await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "becky.loops.reply",
            "params": reply_params(),
        })
    )

    assert response["error"] == {"code": -32000, "message": "revision_conflict"}
    assert sender.calls == []
    assert generator.calls == []


@pytest.mark.asyncio
async def test_bridge_discards_unsent_attempt_when_revision_recheck_crashes() -> None:
    store = SequencedRevisionStore([REVISION, RuntimeError("db unavailable"), REVISION])
    sender = FakeTopicSender()
    generator = FakeReplyGenerator()
    server = reply_server(store=store, sender=sender, generator=generator)
    request = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "becky.loops.reply",
        "params": reply_params(),
    })

    response = await server._dispatch(request)
    retry = await server._dispatch(request)

    assert response["error"] == {"code": -32600, "message": "protocol"}
    assert retry["result"]["answer_state"] == "answered"
    assert len(sender.calls) == 2
    assert len(generator.calls) == 1


@pytest.mark.asyncio
async def test_bridge_rechecks_revision_after_comment_before_generation() -> None:
    store = FakeStore()
    sender = RevisionMutatingTopicSender(store)
    generator = FakeReplyGenerator()
    server = reply_server(store=store, sender=sender, generator=generator)
    request = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "becky.loops.reply",
        "params": reply_params(),
    })

    response = await server._dispatch(request)
    replay = await server._dispatch(request)

    assert response["error"] == {"code": -32000, "message": "revision_conflict"}
    assert replay["result"]["answer_state"] == "answer_unavailable"
    assert sender.calls[0]["text"] == "Can you clarify the next step?"
    assert len(sender.calls) == 1
    assert generator.calls == []


@pytest.mark.asyncio
async def test_bridge_replays_safe_state_when_post_comment_recheck_crashes() -> None:
    store = SequencedRevisionStore([REVISION, REVISION, RuntimeError("db unavailable")])
    sender = FakeTopicSender()
    generator = FakeReplyGenerator()
    server = reply_server(store=store, sender=sender, generator=generator)
    request = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "becky.loops.reply",
        "params": reply_params(),
    })

    response = await server._dispatch(request)
    replay = await server._dispatch(request)

    assert response["error"] == {"code": -32600, "message": "protocol"}
    assert replay["result"]["answer_state"] == "answer_unavailable"
    assert len(sender.calls) == 1
    assert generator.calls == []


@pytest.mark.asyncio
async def test_bridge_rechecks_revision_after_generation_before_answer_send() -> None:
    store = FakeStore()
    sender = FakeTopicSender()
    generator = RevisionMutatingReplyGenerator(store)
    server = reply_server(store=store, sender=sender, generator=generator)
    request = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "becky.loops.reply",
        "params": reply_params(),
    })

    response = await server._dispatch(request)
    replay = await server._dispatch(request)

    assert response["error"] == {"code": -32000, "message": "revision_conflict"}
    assert replay["result"]["answer_state"] == "answer_pending"
    assert replay["result"]["answer"] == "Retained normalized answer."
    assert len(sender.calls) == 1
    assert len(generator.calls) == 1


@pytest.mark.asyncio
async def test_bridge_sends_comment_then_answer_with_internal_anchor() -> None:
    sender = FakeTopicSender()
    generator = FakeReplyGenerator(["Start with the insulation quote."])
    server = reply_server(sender=sender, generator=generator)

    response = await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "becky.loops.reply",
            "params": reply_params(),
        })
    )

    assert response["result"] == {
        "schema_version": "1",
        "source_ref": SOURCE_REF,
        "revision": REVISION,
        "comment_sent_at": response["result"]["comment_sent_at"],
        "answer": "Start with the insulation quote.",
        "answer_generated_at": response["result"]["answer_generated_at"],
        "answer_sent_at": response["result"]["answer_sent_at"],
        "answer_state": "answered",
    }
    assert sender.calls == [
        {
            "chat_id": "123456789",
            "thread_id": "20197",
            "text": "Can you clarify the next step?",
            "reply_to_message_id": None,
        },
        {
            "chat_id": "123456789",
            "thread_id": "20197",
            "text": "Start with the insulation quote.",
            "reply_to_message_id": "101",
        },
    ]
    assert len(generator.calls) == 1
    assert generator.calls[0]["comment"] == "Can you clarify the next step?"
    assert "message_id" not in json.dumps(response)
    assert "thread_id" not in json.dumps(response)


@pytest.mark.asyncio
async def test_telegram_topic_sender_adds_labels_and_exact_private_topic_metadata() -> (
    None
):
    adapter = FakeTelegramAdapter()
    sender = becky_loops.TelegramTopicSender(adapter)

    comment = await sender.send_topic(
        chat_id="123456789",
        thread_id="20197",
        text="Can you clarify the next step?",
        reply_to_message_id=None,
    )
    answer = await sender.send_topic(
        chat_id="123456789",
        thread_id="20197",
        text="Start with the insulation quote.",
        reply_to_message_id=comment.message_id,
    )

    assert answer.message_id == "902"
    assert adapter.calls == [
        {
            "chat_id": "123456789",
            "content": "Cory via Becky: Can you clarify the next step?",
            "reply_to": None,
            "metadata": {
                "thread_id": "20197",
                "direct_messages_topic_id": "20197",
                "notify": True,
            },
        },
        {
            "chat_id": "123456789",
            "content": "Becky: Start with the insulation quote.",
            "reply_to": "901",
            "metadata": {
                "thread_id": "20197",
                "direct_messages_topic_id": "20197",
                "notify": True,
            },
        },
    ]


@pytest.mark.asyncio
async def test_telegram_topic_sender_routes_forum_group_topics_with_message_thread_id() -> (
    None
):
    adapter = FakeTelegramAdapter()
    sender = becky_loops.TelegramTopicSender(adapter)

    await sender.send_topic(
        chat_id="-1004476874933",
        thread_id="3964",
        text="Can you clarify the next step?",
        reply_to_message_id=None,
    )

    assert adapter.calls == [
        {
            "chat_id": "-1004476874933",
            "content": "Cory via Becky: Can you clarify the next step?",
            "reply_to": None,
            "metadata": {
                "thread_id": "3964",
                "notify": True,
            },
        }
    ]


@pytest.mark.asyncio
async def test_telegram_topic_sender_anchors_to_the_last_long_message_chunk() -> None:
    outcome = SimpleNamespace(
        success=True,
        message_id="901",
        raw_response={
            "thread_fallback": False,
            "message_ids": ["901", "902"],
        },
    )
    sender = becky_loops.TelegramTopicSender(FakeTelegramAdapter([outcome]))

    receipt = await sender.send_topic(
        chat_id="123456789",
        thread_id="20197",
        text="x" * 5_000,
        reply_to_message_id=None,
    )

    assert receipt.message_id == "902"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        SimpleNamespace(success=False, message_id="private-id", error="secret"),
        SimpleNamespace(success=True, message_id=None, raw_response={}),
        SimpleNamespace(
            success=True,
            message_id="private-id",
            raw_response={"thread_fallback": True},
        ),
    ],
)
async def test_telegram_topic_sender_fails_closed_without_returning_adapter_details(
    outcome: object,
) -> None:
    sender = becky_loops.TelegramTopicSender(FakeTelegramAdapter([outcome]))

    with pytest.raises(Exception) as caught:
        await sender.send_topic(
            chat_id="123456789",
            thread_id="20197",
            text="Comment",
            reply_to_message_id=None,
        )

    assert "private-id" not in str(caught.value)
    assert "secret" not in str(caught.value)


@pytest.mark.asyncio
async def test_disconnected_telegram_adapter_disables_capability_and_sends_nothing() -> (
    None
):
    adapter = FakeTelegramAdapter()
    adapter.is_connected = False
    sender = becky_loops.TelegramTopicSender(adapter)
    server = BeckyLoopsBridgeServer(
        config=config(topic_reply="bot_api_private_topic"),
        store=FakeStore(),
        summarizer=FakeSummarizer(),
        topic_sender=sender,
        reply_generator=FakeReplyGenerator(),
    )

    capabilities = await server._method("becky.loops.capabilities", {})

    assert capabilities["topic_reply"] == "unavailable"
    with pytest.raises(Exception):
        await sender.send_topic(
            chat_id="123456789",
            thread_id="20197",
            text="Comment",
            reply_to_message_id=None,
        )
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_bridge_duplicate_reply_replays_result_without_external_calls() -> None:
    sender = FakeTopicSender()
    generator = FakeReplyGenerator()
    server = reply_server(sender=sender, generator=generator)
    request = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "becky.loops.reply",
        "params": reply_params(),
    })

    first = await server._dispatch(request)
    second = await server._dispatch(request)

    assert second == first
    assert len(sender.calls) == 2
    assert len(generator.calls) == 1


@pytest.mark.asyncio
async def test_bridge_duplicate_reply_replays_after_transcript_revision_changes() -> (
    None
):
    store = FakeStore()
    sender = FakeTopicSender()
    generator = FakeReplyGenerator()
    server = reply_server(store=store, sender=sender, generator=generator)
    request = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "becky.loops.reply",
        "params": reply_params(),
    })
    first = await server._dispatch(request)
    store.rows[0]["revision"] = "sha256:" + "b" * 64

    replay = await server._dispatch(request)

    assert replay == first
    assert len(sender.calls) == 2
    assert len(generator.calls) == 1


@pytest.mark.asyncio
async def test_bridge_rejects_same_key_with_different_reply_payload() -> None:
    sender = FakeTopicSender()
    server = reply_server(sender=sender)
    await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "becky.loops.reply",
            "params": reply_params(),
        })
    )

    response = await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "becky.loops.reply",
            "params": reply_params(text="Different comment"),
        })
    )

    assert response["error"] == {"code": -32000, "message": "idempotency_conflict"}
    assert len(sender.calls) == 2


@pytest.mark.asyncio
async def test_bridge_retries_generated_answer_without_resending_comment() -> None:
    sender = FakeTopicSender([
        becky_loops.TopicSendReceipt(message_id="301"),
        RuntimeError("answer delivery failed"),
        becky_loops.TopicSendReceipt(message_id="302"),
    ])
    generator = FakeReplyGenerator(["Retained normalized answer."])
    server = reply_server(sender=sender, generator=generator)

    first = await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "becky.loops.reply",
            "params": reply_params(),
        })
    )
    retried = await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "becky.loops.reply_retry",
            "params": retry_params(),
        })
    )

    assert first["result"]["answer_state"] == "answer_pending"
    assert first["result"]["answer"] == "Retained normalized answer."
    assert first["result"]["answer_sent_at"] is None
    assert retried["result"]["answer_state"] == "answered"
    assert [call["text"] for call in sender.calls] == [
        "Can you clarify the next step?",
        "Retained normalized answer.",
        "Retained normalized answer.",
    ]
    assert [call["reply_to_message_id"] for call in sender.calls] == [
        None,
        "301",
        "301",
    ]
    assert len(generator.calls) == 1


@pytest.mark.asyncio
async def test_bridge_rechecks_revision_before_retry_send_or_generation() -> None:
    store = FakeStore()
    sender = FakeTopicSender([
        becky_loops.TopicSendReceipt(message_id="301"),
        RuntimeError("answer delivery failed"),
    ])
    generator = FakeReplyGenerator(["Retained normalized answer."])
    server = reply_server(store=store, sender=sender, generator=generator)
    await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "becky.loops.reply",
            "params": reply_params(),
        })
    )
    store.rows[0]["revision"] = "sha256:" + "b" * 64

    response = await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "becky.loops.reply_retry",
            "params": retry_params(),
        })
    )

    assert response["error"] == {"code": -32000, "message": "revision_conflict"}
    assert len(sender.calls) == 2
    assert len(generator.calls) == 1


@pytest.mark.asyncio
async def test_bridge_retries_generation_without_resending_comment() -> None:
    sender = FakeTopicSender()
    generator = FakeReplyGenerator([
        ReplyUnavailable(),
        "Generated after retry.",
    ])
    server = reply_server(sender=sender, generator=generator)

    first = await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "becky.loops.reply",
            "params": reply_params(),
        })
    )
    retried = await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "becky.loops.reply_retry",
            "params": retry_params(),
        })
    )

    assert first["result"]["answer_state"] == "answer_unavailable"
    assert first["result"]["answer"] is None
    assert retried["result"]["answer_state"] == "answered"
    assert [call["text"] for call in sender.calls] == [
        "Can you clarify the next step?",
        "Generated after retry.",
    ]
    assert len(generator.calls) == 2


@pytest.mark.asyncio
async def test_bridge_generation_cancellation_replays_answer_unavailable() -> None:
    sender = FakeTopicSender()
    generator = BlockingReplyGenerator()
    server = reply_server(sender=sender, generator=generator)
    request = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "becky.loops.reply",
        "params": reply_params(),
    })
    first_task = asyncio.create_task(server._dispatch(request))
    await generator.started.wait()
    first_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_task

    replay = await server._dispatch(request)

    assert replay["result"]["answer_state"] == "answer_unavailable"
    assert replay["result"]["answer"] is None
    assert len(sender.calls) == 1
    assert len(generator.calls) == 1


@pytest.mark.asyncio
async def test_bridge_comment_send_failure_is_safe_and_idempotent() -> None:
    private_error = "Telegram message 987654 failed in topic 20197"
    sender = FakeTopicSender([RuntimeError(private_error)])
    generator = FakeReplyGenerator()
    server = reply_server(sender=sender, generator=generator)
    request = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "becky.loops.reply",
        "params": reply_params(),
    })

    first = await server._dispatch(request)
    second = await server._dispatch(request)

    assert (
        first
        == second
        == {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32000, "message": "reply_send_failed"},
        }
    )
    assert private_error not in json.dumps(first)
    assert len(sender.calls) == 1
    assert generator.calls == []


@pytest.mark.asyncio
async def test_bridge_unknown_retry_after_restart_fails_closed() -> None:
    sender = FakeTopicSender()
    generator = FakeReplyGenerator()
    restarted_server = reply_server(sender=sender, generator=generator)

    response = await restarted_server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "becky.loops.reply_retry",
            "params": retry_params(),
        })
    )

    assert response == {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {"code": -32000, "message": "reply_retry_unavailable"},
    }
    assert sender.calls == []
    assert generator.calls == []


@pytest.mark.asyncio
async def test_bridge_attempts_expire_and_retry_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(becky_loops, "_REPLY_ATTEMPT_TTL_SECONDS", 0.0)
    server = reply_server()
    await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "becky.loops.reply",
            "params": reply_params(),
        })
    )

    response = await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "becky.loops.reply_retry",
            "params": retry_params(),
        })
    )

    assert response["error"] == {
        "code": -32000,
        "message": "reply_retry_unavailable",
    }


@pytest.mark.asyncio
async def test_bridge_attempt_map_rejects_new_key_without_evicting_stored_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(becky_loops, "_MAX_REPLY_ATTEMPTS", 1)
    sender = FakeTopicSender()
    server = reply_server(sender=sender)
    first_request = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "becky.loops.reply",
        "params": reply_params(),
    })
    first = await server._dispatch(first_request)
    second = await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "becky.loops.reply",
            "params": reply_params(idempotency_key=SECOND_IDEMPOTENCY_KEY),
        })
    )
    replay = await server._dispatch(first_request)

    assert second["error"] == {
        "code": -32000,
        "message": "topic_reply_unavailable",
    }
    assert replay == first
    assert len(sender.calls) == 2


@pytest.mark.asyncio
async def test_bridge_attempt_bound_never_evicts_an_in_flight_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(becky_loops, "_MAX_REPLY_ATTEMPTS", 1)
    sender = BlockingTopicSender()
    server = reply_server(sender=sender)
    first_task = asyncio.create_task(
        server._dispatch(
            json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "becky.loops.reply",
                "params": reply_params(),
            })
        )
    )
    await sender.started.wait()

    try:
        second = await server._dispatch(
            json.dumps({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "becky.loops.reply",
                "params": reply_params(idempotency_key=SECOND_IDEMPOTENCY_KEY),
            })
        )
    finally:
        sender.release.set()
    first = await first_task
    replay = await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "becky.loops.reply",
            "params": reply_params(),
        })
    )

    assert second["error"] == {
        "code": -32000,
        "message": "topic_reply_unavailable",
    }
    assert first == replay
    assert len(sender.calls) == 2


@pytest.mark.asyncio
async def test_bridge_rejects_wrong_token_before_accepting_socket() -> None:
    server = BeckyLoopsBridgeServer(
        config=config(), store=FakeStore(), summarizer=FakeSummarizer()
    )
    await server.start()
    try:
        with pytest.raises(InvalidStatus) as caught:
            async with connect(
                f"ws://127.0.0.1:{server.bound_port}/api/ws?token={'x' * 64}"
            ):
                pass
        assert caught.value.response.status_code == 401
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_bridge_rejects_duplicate_or_extra_token_query_values() -> None:
    server = BeckyLoopsBridgeServer(
        config=config(), store=FakeStore(), summarizer=FakeSummarizer()
    )
    await server.start()
    try:
        with pytest.raises(InvalidStatus) as caught:
            async with connect(
                f"ws://127.0.0.1:{server.bound_port}/api/ws?token={'t' * 64}&token=wrong"
            ):
                pass
        assert caught.value.response.status_code == 401
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_bridge_returns_structured_protocol_errors() -> None:
    server = BeckyLoopsBridgeServer(
        config=config(), store=FakeStore(), summarizer=FakeSummarizer()
    )
    malformed = await server._dispatch("not-json")
    assert malformed == {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": -32600, "message": "protocol"},
    }
    boolean_id = await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": True,
            "method": "becky.loops.list",
            "params": {},
        })
    )
    assert boolean_id["id"] is None


def test_public_index_uses_force_redaction_and_title_fallback() -> None:
    store = FakeStore()
    store.rows[0]["title"] = "123456789"
    server = BeckyLoopsBridgeServer(
        config=config(), store=store, summarizer=FakeSummarizer()
    )
    index = server._public_index(store.rows[0])
    assert index["title"] == "Telegram loop"


def test_public_index_emits_a_private_forum_topic_deep_link() -> None:
    store = FakeStore()
    server = BeckyLoopsBridgeServer(
        config=BeckyLoopsConfig(
            enabled=True,
            chat_id="-1004476874933",
            token="t" * 64,
            port=0,
            topic_control="unavailable",
            topic_reply="unavailable",
        ),
        store=store,
        summarizer=FakeSummarizer(),
    )

    index = server._public_index(store.rows[0])

    assert index["telegram_url"] == "https://t.me/c/4476874933/20197"


def test_public_index_targets_the_latest_topic_message_when_available() -> None:
    store = FakeStore()
    store.transcripts["session-1"] = [
        {
            "role": "user",
            "content": "Earlier",
            "platform_message_id": "20201",
            "timestamp": 1_755_104_400.0,
        },
        {
            "role": "assistant",
            "content": "Latest",
            "platform_message_id": "20214",
            "timestamp": 1_755_104_460.0,
        },
    ]
    server = BeckyLoopsBridgeServer(
        config=BeckyLoopsConfig(
            enabled=True,
            chat_id="-1004476874933",
            token="t" * 64,
            port=0,
            topic_control="unavailable",
            topic_reply="unavailable",
        ),
        store=store,
        summarizer=FakeSummarizer(),
    )

    index = server._public_index(store.rows[0])

    assert index["telegram_url"] == "https://t.me/c/4476874933/20197/20214?single"


def test_session_store_targets_latest_message_after_compaction() -> None:
    from gateway.becky_loops import SessionDBBeckyLoopsStore

    db = InactiveMessageProjectionDB()
    db.rows[0].update({
        "chat_id": "-1004476874933",
        "thread_id": "20197",
        "id": "compacted-session",
    })
    store = SessionDBBeckyLoopsStore(db)

    row = store.list_topics("-1004476874933")[0]

    assert row["telegram_url"] == "https://t.me/c/4476874933/20197/222?single"


@pytest.mark.asyncio
async def test_list_projects_latest_safe_becky_response_without_tool_payloads() -> None:
    store = FakeStore()
    store.transcripts["session-1"] = [
        {
            "role": "user",
            "content": "Can you check the reading?",
            "timestamp": 1_755_104_400.0,
        },
        {
            "role": "assistant",
            "content": '{"output":"raw tool data", "exit_code":0}',
            "timestamp": 1_755_104_450.0,
        },
        {
            "role": "assistant",
            "content": "The reading is ready to review.",
            "timestamp": 1_755_104_460.0,
        },
    ]
    server = BeckyLoopsBridgeServer(
        config=config(), store=store, summarizer=FakeSummarizer()
    )

    result = await server._method("becky.loops.list", {})

    item = result["loops"][0]
    assert item["last_becky_response"] == "The reading is ready to review."
    assert item["last_becky_response_at"] == "2025-08-13T17:01:00+00:00"
    assert "session_id" not in item
    assert "raw tool data" not in json.dumps(item)


@pytest.mark.asyncio
async def test_bridge_never_advertises_unproven_topic_control_or_identifiers() -> None:
    store = FakeStore()
    store.rows[0]["thread_id"] = "thread-9"
    store.rows[0]["title"] = "Topic session-1 thread-9 123456789"
    store.transcripts["session-1"][0]["content"] = (
        "Decided to use session-1 in thread-9 for chat 123456789."
    )
    bridge_config = BeckyLoopsConfig(
        enabled=True,
        chat_id="123456789",
        token="t" * 64,
        port=0,
        topic_control="bot_api_private_topic",
    )
    unsafe_summary = LoopSummary(
        summary="Session session-1 in thread-9 for chat 123456789 needs review.",
        decisions=["Use session-1 in thread-9."],
        unresolved_items=["Does 123456789 need review?"],
        next_action="Review session-1.",
        waiting_on="becky",
        key_events=[
            {
                "occurred_at": "2026-08-13T20:00:00+00:00",
                "text": "thread-9 was selected for 123456789.",
            }
        ],
        final_outcome=None,
    )
    server = BeckyLoopsBridgeServer(
        config=bridge_config,
        store=store,
        summarizer=FakeSummarizer(result=unsafe_summary),
    )
    await server.start()
    try:
        async with connect(
            f"ws://127.0.0.1:{server.bound_port}/api/ws?token={'t' * 64}"
        ) as ws:
            await ws.recv()
            capabilities = await rpc(ws, 1, "becky.loops.capabilities", {})
            assert capabilities["result"]["topic_control"] == "unavailable"
            listed = await rpc(ws, 2, "becky.loops.list", {})
            assert "session-1" not in listed["result"]["loops"][0]["title"]
            summary = await rpc(
                ws,
                3,
                "becky.loops.summarize",
                {
                    "source_ref": SOURCE_REF,
                    "expected_revision": REVISION,
                    "force": False,
                },
            )
            encoded = json.dumps(summary)
            assert "session-1" not in encoded
            assert "thread-9" not in encoded
            assert "123456789" not in encoded
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_bridge_summarize_delegates_to_injected_summarizer() -> None:
    store = FakeStore()
    store.transcripts["session-1"][0]["content"] = "RAW TRANSCRIPT MUST NOT RETURN"
    summarizer = FakeSummarizer()
    server = BeckyLoopsBridgeServer(config=config(), store=store, summarizer=summarizer)
    await server.start()
    try:
        async with connect(
            f"ws://127.0.0.1:{server.bound_port}/api/ws?token={'t' * 64}"
        ) as ws:
            await ws.recv()
            listed = await rpc(ws, 1, "becky.loops.list", {})
            current_revision = listed["result"]["loops"][0]["revision"]
            summary = await rpc(
                ws,
                2,
                "becky.loops.summarize",
                {
                    "source_ref": SOURCE_REF,
                    "expected_revision": current_revision,
                    "force": False,
                },
            )
            assert summary["result"]["source_ref"] == SOURCE_REF
            assert summary["result"] == {
                "schema_version": "1",
                "source_ref": SOURCE_REF,
                "revision": REVISION,
                "generated_at": summary["result"]["generated_at"],
                "summary": "Use the smaller layout.",
                "decisions": ["The smaller layout was approved."],
                "unresolved_items": ["Confirm the installation date."],
                "next_action": "Prepare the final plan.",
                "waiting_on": "becky",
                "key_events": [
                    {
                        "occurred_at": "2026-08-13T20:00:00+00:00",
                        "text": "The smaller layout was chosen.",
                    }
                ],
                "final_outcome": None,
            }
            assert len(summarizer.calls) == 1
            call = summarizer.calls[0]
            assert call["row"] == store.rows[0]
            assert call["transcript"] == store.transcripts["session-1"]
            assert call["deadline"] > get_running_loop().time()
            assert "RAW TRANSCRIPT MUST NOT RETURN" not in json.dumps(summary)

            conflict = await rpc(
                ws,
                3,
                "becky.loops.summarize",
                {
                    "source_ref": SOURCE_REF,
                    "expected_revision": "sha256:" + "b" * 64,
                    "force": False,
                },
            )
            assert conflict == {
                "jsonrpc": "2.0",
                "id": 3,
                "error": {"code": -32000, "message": "revision_conflict"},
            }
    finally:
        await server.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "code"),
    [
        (_ConversationTooLarge(), "conversation_too_large"),
        (_SummaryValidationError(), "summary_invalid"),
        (SummaryUnavailable(), "summary_timeout"),
    ],
)
async def test_bridge_maps_summary_failures_without_transcript_fallback(
    failure: Exception, code: str
) -> None:
    store = FakeStore()
    store.transcripts["session-1"][0]["content"] = "RAW TRANSCRIPT MUST NOT RETURN"
    server = BeckyLoopsBridgeServer(
        config=config(), store=store, summarizer=FakeSummarizer(failure=failure)
    )
    await server.start()
    try:
        async with connect(
            f"ws://127.0.0.1:{server.bound_port}/api/ws?token={'t' * 64}"
        ) as ws:
            await ws.recv()
            response = await rpc(
                ws,
                1,
                "becky.loops.summarize",
                {
                    "source_ref": SOURCE_REF,
                    "expected_revision": REVISION,
                    "force": False,
                },
            )
            assert response == {
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": -32000, "message": code},
            }
            assert "RAW TRANSCRIPT MUST NOT RETURN" not in json.dumps(response)
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_bridge_hides_unexpected_summary_error_details_from_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider_error = "PROVIDER_OUTPUT_SENTINEL"
    server = BeckyLoopsBridgeServer(
        config=config(),
        store=FakeStore(),
        summarizer=FakeSummarizer(failure=RuntimeError(provider_error)),
    )
    await server.start()
    try:
        async with connect(
            f"ws://127.0.0.1:{server.bound_port}/api/ws?token={'t' * 64}"
        ) as ws:
            await ws.recv()
            with caplog.at_level(logging.WARNING, logger="gateway.becky_loops"):
                response = await rpc(
                    ws,
                    1,
                    "becky.loops.summarize",
                    {
                        "source_ref": SOURCE_REF,
                        "expected_revision": REVISION,
                        "force": False,
                    },
                )
            assert response == {
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": -32600, "message": "protocol"},
            }
            assert provider_error not in caplog.text
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_bridge_rechecks_transcript_revision_before_summarizing() -> None:
    summarizer = FakeSummarizer()
    server = BeckyLoopsBridgeServer(
        config=config(), store=RacingStore(), summarizer=summarizer
    )
    await server.start()
    try:
        async with connect(
            f"ws://127.0.0.1:{server.bound_port}/api/ws?token={'t' * 64}"
        ) as ws:
            await ws.recv()
            response = await rpc(
                ws,
                1,
                "becky.loops.summarize",
                {
                    "source_ref": SOURCE_REF,
                    "expected_revision": REVISION,
                    "force": False,
                },
            )
            assert response["error"] == {
                "code": -32000,
                "message": "revision_conflict",
            }
            assert summarizer.calls == []
    finally:
        await server.stop()


def test_session_store_excludes_branch_delegate_and_tool_children() -> None:
    from gateway.becky_loops import SessionDBBeckyLoopsStore

    store = SessionDBBeckyLoopsStore(ProjectionDB())
    rows = store.list_topics("123456789")
    assert [row["title"] for row in rows] == ["Root"]


def test_session_store_excludes_telegram_general_topic() -> None:
    from gateway.becky_loops import SessionDBBeckyLoopsStore

    db = ProjectionDB()
    db.rows.append({
        **db.rows[0],
        "id": "general",
        "thread_id": "1",
        "title": "Cory Ng Test Message",
    })

    rows = SessionDBBeckyLoopsStore(db).list_topics("123456789")

    assert all(row["title"] != "Cory Ng Test Message" for row in rows)


def test_session_store_persists_shortcut_topic_exchange_idempotently() -> None:
    from gateway.becky_loops import SessionDBBeckyLoopsStore

    db = ShortcutProjectionDB()
    store = SessionDBBeckyLoopsStore(db)
    store._chat_id = "-1004476874933"

    session_id = store.record_shortcut_topic(
        title="Energy Audit Request",
        text="Please audit the last week's energy usage.",
        topic_id="44",
        message_id="104",
    )
    assert session_id is not None
    store.record_shortcut_topic(
        title="Energy Audit Request",
        text="Please audit the last week's energy usage.",
        topic_id="44",
        message_id="104",
    )
    store.record_shortcut_answer(
        session_id=session_id,
        text="I need an export first.",
        message_id="105",
    )

    assert db.sessions[session_id]["title"] == "Energy Audit Request"
    assert [message["role"] for message in db.messages[session_id]] == [
        "user",
        "assistant",
    ]


def test_session_store_routes_shortcut_exchange_to_gateway_topic_session() -> None:
    from gateway.becky_loops import SessionDBBeckyLoopsStore

    db = ShortcutProjectionDB()
    routing = GatewayShortcutSessionStore(db)
    store = SessionDBBeckyLoopsStore(db, session_store=routing)
    store._chat_id = "-1004476874933"

    session_id = store.record_shortcut_topic(
        title="Energy Audit Request",
        text="Can you audit my home energy usage?",
        topic_id="44",
        message_id="104",
    )
    store.record_shortcut_answer(
        session_id=session_id or "",
        text="Please provide last week's energy data.",
        message_id="105",
    )

    assert session_id == "gateway-session"
    assert routing.source.chat_type == "group"
    assert routing.source.thread_id == "44"
    assert [message["role"] for message in db.messages[session_id]] == [
        "user",
        "assistant",
    ]


def test_session_store_coalesces_restarted_sessions_for_one_topic() -> None:
    from gateway.becky_loops import SessionDBBeckyLoopsStore

    db = ProjectionDB()
    db.rows.insert(
        0,
        {**db.rows[0], "id": "newer", "title": "Newer", "last_active": 1_755_104_500.0},
    )
    store = SessionDBBeckyLoopsStore(db)
    rows = store.list_topics("123456789")
    assert [row["title"] for row in rows] == ["Newer"]


@pytest.mark.asyncio
async def test_bridge_close_and_reopen_fail_closed_without_topic_control() -> None:
    server = BeckyLoopsBridgeServer(
        config=config(), store=FakeStore(), summarizer=FakeSummarizer()
    )
    await server.start()
    try:
        async with connect(
            f"ws://127.0.0.1:{server.bound_port}/api/ws?token={'t' * 64}"
        ) as ws:
            await ws.recv()
            for request_id, method, params in (
                (
                    1,
                    "becky.loops.close",
                    {
                        "source_ref": SOURCE_REF,
                        "expected_revision": REVISION,
                        "idempotency_key": "8c9c8217-cc0f-463d-a430-173f1802edb2",
                    },
                ),
                (
                    2,
                    "becky.loops.reopen",
                    {
                        "source_ref": SOURCE_REF,
                        "idempotency_key": "8c9c8217-cc0f-463d-a430-173f1802edb2",
                        "context": {
                            "title": "A",
                            "summary": "B",
                            "decisions": [],
                            "unresolved_items": [],
                            "final_outcome": None,
                        },
                    },
                ),
            ):
                response = await rpc(ws, request_id, method, params)
                assert response["error"] == {
                    "code": -32000,
                    "message": "topic_control_unavailable",
                }
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_bridge_close_uses_connected_controller_and_replays_idempotently() -> (
    None
):
    controller = FakeTopicController()
    server = BeckyLoopsBridgeServer(
        config=control_config(),
        store=FakeStore(),
        summarizer=FakeSummarizer(),
        topic_controller=controller,
    )

    capabilities = await server._method("becky.loops.capabilities", {})
    assert capabilities["topic_control"] == "bot_api_private_topic"
    params = {
        "source_ref": SOURCE_REF,
        "expected_revision": REVISION,
        "idempotency_key": IDEMPOTENCY_KEY,
    }
    first = await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "becky.loops.close",
            "params": params,
        })
    )
    second = await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "becky.loops.close",
            "params": params,
        })
    )

    assert first["result"]["source_state"] == "closed"
    assert first["result"] == second["result"]
    assert controller.calls == [{"chat_id": "123456789", "thread_id": "20197"}]


@pytest.mark.asyncio
async def test_bridge_close_idempotency_results_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(becky_loops, "_MAX_CLOSE_RESULTS", 2)
    controller = FakeTopicController()
    server = BeckyLoopsBridgeServer(
        config=control_config(),
        store=CloseableStore(),
        summarizer=FakeSummarizer(),
        topic_controller=controller,
    )

    keys = [f"00000000-0000-4000-8000-{index:012d}" for index in range(3)]
    for key in keys:
        await server._close_topic(
            source_ref=SOURCE_REF,
            expected_revision=REVISION,
            idempotency_key=key,
        )

    assert len(server._close_results) == 2
    assert keys[0] not in server._close_results


@pytest.mark.asyncio
async def test_bridge_start_cancellation_closes_listener_after_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()
    close_calls = 0

    class Listener:
        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1

        async def wait_closed(self) -> None:
            closed.set()

    listener = Listener()

    async def delayed_serve(*args: Any, **kwargs: Any) -> Listener:
        del args, kwargs
        started.set()
        await release.wait()
        return listener

    monkeypatch.setattr(becky_loops, "serve", delayed_serve)
    server = BeckyLoopsBridgeServer(
        config=control_config(),
        store=FakeStore(),
        summarizer=FakeSummarizer(),
    )
    task = asyncio.create_task(server.start())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    release.set()
    await asyncio.wait_for(closed.wait(), timeout=1.0)
    assert close_calls == 1
    assert server._server is None


@pytest.mark.asyncio
async def test_bridge_close_reports_mtproto_control_method() -> None:
    controller = FakeTopicController()
    controller.method = "mtproto_private_topic"
    server = BeckyLoopsBridgeServer(
        config=BeckyLoopsConfig(
            enabled=True,
            chat_id="123456789",
            token="t" * 64,
            port=0,
            topic_control="mtproto_private_topic",
        ),
        store=FakeStore(),
        summarizer=FakeSummarizer(),
        topic_controller=controller,
    )

    capabilities = await server._method("becky.loops.capabilities", {})
    assert capabilities["topic_control"] == "mtproto_private_topic"

    response = await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "becky.loops.close",
            "params": {
                "source_ref": SOURCE_REF,
                "expected_revision": REVISION,
                "idempotency_key": IDEMPOTENCY_KEY,
            },
        })
    )

    assert response["result"]["control_method"] == "mtproto_private_topic"


@pytest.mark.asyncio
async def test_bridge_requires_explicit_controller_method_for_mtproto() -> None:
    controller = SimpleNamespace(is_connected=True, supports_close=True)
    server = BeckyLoopsBridgeServer(
        config=BeckyLoopsConfig(
            enabled=True,
            chat_id="123456789",
            token="t" * 64,
            port=0,
            topic_control="mtproto_private_topic",
        ),
        store=FakeStore(),
        summarizer=FakeSummarizer(),
        topic_controller=controller,
    )

    capabilities = await server._method("becky.loops.capabilities", {})

    assert capabilities["topic_control"] == "unavailable"


@pytest.mark.asyncio
async def test_bridge_close_rechecks_revision_and_maps_controller_failures() -> None:
    controller = FakeTopicController([
        becky_loops._TopicControlFailure("topic_control_unsupported")
    ])
    server = BeckyLoopsBridgeServer(
        config=control_config(),
        store=FakeStore(),
        summarizer=FakeSummarizer(),
        topic_controller=controller,
    )
    mismatch = await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "becky.loops.close",
            "params": {
                "source_ref": SOURCE_REF,
                "expected_revision": NEW_REVISION,
                "idempotency_key": IDEMPOTENCY_KEY,
            },
        })
    )
    assert mismatch["error"]["message"] == "revision_conflict"
    failed = await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "becky.loops.close",
            "params": {
                "source_ref": SOURCE_REF,
                "expected_revision": REVISION,
                "idempotency_key": IDEMPOTENCY_KEY,
            },
        })
    )
    assert failed["error"]["message"] == "topic_control_unsupported"


@pytest.mark.asyncio
async def test_bridge_reconciles_already_closed_topic_and_ends_session() -> None:
    store = CloseableStore()
    controller = FakeTopicController(
        [becky_loops._TopicControlFailure("topic_already_closed")]
    )
    server = BeckyLoopsBridgeServer(
        config=control_config(),
        store=store,
        summarizer=FakeSummarizer(),
        topic_controller=controller,
    )

    response = await server._dispatch(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "becky.loops.close",
            "params": {
                "source_ref": SOURCE_REF,
                "expected_revision": REVISION,
                "idempotency_key": IDEMPOTENCY_KEY,
            },
        })
    )

    assert response["result"]["source_state"] == "closed"
    assert store.ended_topics == [("123456789", "20197", "telegram_topic_closed")]


@pytest.mark.asyncio
async def test_telegram_topic_controller_calls_bot_api_and_fails_closed() -> None:
    class Bot:
        async def close_forum_topic(self, **kwargs: Any) -> bool:
            self.kwargs = kwargs
            return True

    adapter = SimpleNamespace(_bot=Bot(), is_connected=True)
    controller = becky_loops.TelegramTopicController(adapter)
    closed_at = await controller.close_topic(chat_id="8837347581", thread_id="3964")
    assert closed_at.tzinfo is not None
    assert adapter._bot.kwargs == {"chat_id": 8837347581, "message_thread_id": 3964}

    class BrokenBot:
        async def close_forum_topic(self, **kwargs: Any) -> bool:
            del kwargs
            raise RuntimeError("arbitrary provider detail")

    broken = becky_loops.TelegramTopicController(
        SimpleNamespace(_bot=BrokenBot(), is_connected=True)
    )
    with pytest.raises(becky_loops._TopicControlFailure) as error:
        await broken.close_topic(chat_id="8837347581", thread_id="3964")
    assert error.value.code == "topic_control_unavailable"

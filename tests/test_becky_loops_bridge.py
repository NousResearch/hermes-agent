import json
from datetime import UTC, datetime

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from gateway.becky_loops import BeckyLoopsBridgeServer, BeckyLoopsConfig


SOURCE_REF = "loop_" + "A" * 43
REVISION = "sha256:" + "a" * 64


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


class RacingStore(FakeStore):
    def revision_for_topic(self, row: dict, transcript: list[dict]) -> str:
        del row, transcript
        return "sha256:" + "b" * 64


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


def config(*, port: int = 0) -> BeckyLoopsConfig:
    return BeckyLoopsConfig(
        enabled=True,
        chat_id="123456789",
        token="t" * 64,
        port=port,
        topic_control="unavailable",
    )


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
    server = BeckyLoopsBridgeServer(config=config(), store=FakeStore())
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
                "schema_version": "1",
                "summary_schema_version": "1",
                "methods": ["list", "summarize", "close", "reopen"],
                "topic_control": "unavailable",
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
async def test_bridge_rejects_wrong_token_before_accepting_socket() -> None:
    server = BeckyLoopsBridgeServer(config=config(), store=FakeStore())
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
    server = BeckyLoopsBridgeServer(config=config(), store=FakeStore())
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
    server = BeckyLoopsBridgeServer(config=config(), store=FakeStore())
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


def test_public_text_uses_force_redaction_and_title_fallback() -> None:
    store = FakeStore()
    store.rows[0]["title"] = "123456789"
    server = BeckyLoopsBridgeServer(config=config(), store=store)
    index = server._public_index(store.rows[0])
    assert index["title"] == "Telegram loop"
    redacted = server._summary(
        store.rows[0],
        [{"role": "user", "content": "AWS key=AKIA12345678901234", "timestamp": 0}],
    )
    assert "AKIA12345678901234" not in json.dumps(redacted)


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
    server = BeckyLoopsBridgeServer(config=bridge_config, store=store)
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
async def test_bridge_summarize_is_bounded_and_revision_bound() -> None:
    server = BeckyLoopsBridgeServer(config=config(), store=FakeStore())
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
            assert len(summary["result"]["summary"]) <= 2000
            assert summary["result"]["decisions"]
            assert summary["result"]["waiting_on"] == "user"

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
async def test_bridge_rechecks_transcript_revision_before_summarizing() -> None:
    server = BeckyLoopsBridgeServer(config=config(), store=RacingStore())
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
    finally:
        await server.stop()


def test_session_store_excludes_branch_delegate_and_tool_children() -> None:
    from gateway.becky_loops import SessionDBBeckyLoopsStore

    store = SessionDBBeckyLoopsStore(ProjectionDB())
    rows = store.list_topics("123456789")
    assert [row["title"] for row in rows] == ["Root"]


@pytest.mark.asyncio
async def test_bridge_close_and_reopen_fail_closed_without_topic_control() -> None:
    server = BeckyLoopsBridgeServer(config=config(), store=FakeStore())
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

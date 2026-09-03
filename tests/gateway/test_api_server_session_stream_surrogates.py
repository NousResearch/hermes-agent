"""End-to-end regression: a lone UTF-16 surrogate must not kill the session
chat SSE stream.

``POST /api/sessions/{id}/chat/stream`` is the only SSE writer that serializes
with ``ensure_ascii=False``, so it is the only one where a lone surrogate
reaches the UTF-8 encoder raw.  The failure is silent and total: the response
has already sent HTTP 200, the writer loop aborts on the encode error, and the
client is left waiting for events that never arrive — no ``assistant.completed``,
no ``run.completed``, no ``done``.

Both ingress paths are covered, because they fail at different frames:

* the user's own text, echoed verbatim in ``run.started`` — reachable on frame
  #1 with no model involvement at all (a truncated emoji from a clipboard
  slice is enough); and
* model output, streamed through ``assistant.delta`` — the corruption class
  the surrogate chokepoints exist for.

These assert on the events the client actually receives, never on the shape of
the source.
"""

import json

from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB

# An unpaired high surrogate — what a clipboard slice through an emoji leaves
# behind.  Valid as a Python code point, unencodable as UTF-8.
LONE_SURROGATE = "\ud83d"
REPLACEMENT = "�"


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


@pytest.fixture
def adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = session_db
    return adapter


def _stream_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post(
        "/api/sessions/{session_id}/chat/stream",
        adapter._handle_session_chat_stream,
    )
    return app


def _parse_sse(body: str):
    """Parse an SSE body into ``[(event_name, payload)]`` in wire order."""
    events = []
    for block in body.split("\n\n"):
        name = None
        payload = None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: "):]
            elif line.startswith("data: "):
                payload = json.loads(line[len("data: "):])
        if name is not None:
            events.append((name, payload))
    return events


async def _post_turn(adapter, session_id: str, message: str) -> str:
    async with TestClient(TestServer(_stream_app(adapter))) as cli:
        resp = await cli.post(
            f"/api/sessions/{session_id}/chat/stream",
            json={"message": message},
        )
        assert resp.status == 200
        return await resp.text()


@pytest.mark.asyncio
async def test_stream_completes_when_user_message_carries_lone_surrogate(adapter, session_db):
    """The pasted-clipboard case: ``run.started`` echoes the user's text
    verbatim, so the stream dies on its very first frame — before the agent is
    ever consulted.
    """
    session_id = session_db.create_session("surrogate-paste", "api_server")
    pasted = f"summarize this {LONE_SURROGATE} clipboard slice"

    async def fake_run(**kwargs):
        return (
            {"final_response": "summary", "session_id": session_id, "messages": []},
            {"total_tokens": 3},
        )

    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        body = await _post_turn(adapter, session_id, pasted)

    events = _parse_sse(body)
    names = [name for name, _ in events]
    assert "done" in names, f"stream terminated early; events={names}"
    assert names[0] == "run.started", names
    assert "assistant.completed" in names, names
    assert "run.completed" in names, names
    assert names[-1] == "done", names

    run_started = next(payload for name, payload in events if name == "run.started")
    content = run_started["user_message"]["content"]
    # Delivered, repaired — not dropped and not silently truncated.
    assert LONE_SURROGATE not in content
    assert REPLACEMENT in content
    assert content.startswith("summarize this ")
    assert content.endswith(" clipboard slice")


@pytest.mark.asyncio
async def test_stream_completes_when_assistant_delta_carries_lone_surrogate(adapter, session_db):
    """The model-output case: a broken surrogate pair in a streamed delta must
    not take the rest of the turn down with it.
    """
    session_id = session_db.create_session("surrogate-delta", "api_server")

    async def fake_run(**kwargs):
        kwargs["stream_delta_callback"](f"here it is {LONE_SURROGATE}")
        kwargs["stream_delta_callback"](" and the rest of the answer")
        return (
            {"final_response": "here it is  and the rest of the answer", "session_id": session_id, "messages": []},
            {"total_tokens": 5},
        )

    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        body = await _post_turn(adapter, session_id, "plain ascii question")

    events = _parse_sse(body)
    names = [name for name, _ in events]
    assert "done" in names, f"stream terminated early; events={names}"
    assert names[-1] == "done", names
    assert "run.completed" in names, names

    deltas = [payload["delta"] for name, payload in events if name == "assistant.delta"]
    assert len(deltas) == 2, deltas
    assert LONE_SURROGATE not in deltas[0]
    assert REPLACEMENT in deltas[0]
    assert deltas[0].startswith("here it is ")
    # The delta *after* the bad one still arrives — the stream did not stall.
    assert deltas[1] == " and the rest of the answer"


@pytest.mark.asyncio
async def test_stream_leaves_surrogate_free_turns_byte_identical(adapter, session_db):
    """The repair must be reachable only from an encode failure: an ordinary
    non-ASCII turn keeps its raw UTF-8 bytes on the wire, U+FFFD absent.
    """
    session_id = session_db.create_session("surrogate-free", "api_server")

    async def fake_run(**kwargs):
        kwargs["stream_delta_callback"]("Münchner café 🏔")
        return (
            {"final_response": "Münchner café 🏔", "session_id": session_id, "messages": []},
            {"total_tokens": 4},
        )

    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        body = await _post_turn(adapter, session_id, "wie geht's — 🏔?")

    assert REPLACEMENT not in body
    # ensure_ascii=False means the code points ride the wire raw, unescaped.
    assert "Münchner café 🏔" in body

    events = _parse_sse(body)
    names = [name for name, _ in events]
    assert names[-1] == "done", names
    run_started = next(payload for name, payload in events if name == "run.started")
    assert run_started["user_message"]["content"] == "wie geht's — 🏔?"

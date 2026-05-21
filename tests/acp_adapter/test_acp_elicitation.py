import asyncio
import threading
from types import SimpleNamespace

import pytest

from acp_adapter.elicitation import (
    build_clarify_requested_schema,
    create_form_elicitation,
    extract_clarify_answer,
    make_elicitation_clarify_callback,
    supports_form_elicitation,
)


def test_supports_form_elicitation_with_object_form():
    caps = SimpleNamespace(elicitation=SimpleNamespace(form=SimpleNamespace()))
    assert supports_form_elicitation(caps) is True


def test_supports_form_elicitation_with_dict_form():
    caps = SimpleNamespace(elicitation={"form": {}})
    assert supports_form_elicitation(caps) is True


def test_supports_form_elicitation_with_empty_elicitation_object():
    # ACP RFD-style shorthand: empty elicitation object means form supported.
    caps = SimpleNamespace(elicitation={})
    assert supports_form_elicitation(caps) is True


def test_supports_form_elicitation_missing_is_false():
    assert supports_form_elicitation(SimpleNamespace()) is False
    assert supports_form_elicitation(None) is False


def test_supports_form_elicitation_url_only_is_false():
    caps = SimpleNamespace(elicitation={"url": {}})
    assert supports_form_elicitation(caps) is False


def test_open_ended_clarify_schema_has_required_answer_string():
    schema = build_clarify_requested_schema(question="What branch?", choices=None)
    assert schema["type"] == "object"
    assert schema["required"] == ["answer"]
    assert schema["properties"]["answer"]["type"] == "string"


def test_choice_clarify_schema_preserves_other_path():
    schema = build_clarify_requested_schema(
        question="Which approach?",
        choices=["Strict", "Fallback"],
    )
    props = schema["properties"]
    assert "answer" in props
    assert "other_answer" in props
    assert "Strict" in props["answer"].get("enum", [])
    assert "Fallback" in props["answer"].get("enum", [])
    assert "Other (type your answer)" in props["answer"].get("enum", [])


def test_extract_answer_from_accept_dict():
    response = {"action": "accept", "content": {"answer": "Strict"}}
    assert extract_clarify_answer(response) == "Strict"


def test_extract_other_answer_from_accept_dict():
    response = {
        "action": "accept",
        "content": {"answer": "Other (type your answer)", "other_answer": "Use raw JSON-RPC"},
    }
    assert extract_clarify_answer(response) == "Use raw JSON-RPC"


def test_extract_answer_from_accept_object():
    response = SimpleNamespace(action="accept", content={"answer": "Fallback"})
    assert extract_clarify_answer(response) == "Fallback"


def test_decline_cancel_or_missing_returns_standard_sentinel():
    assert extract_clarify_answer({"action": "decline"}) == "[user declined the clarification]"
    assert extract_clarify_answer({"action": "cancel"}) == "[user cancelled the clarification]"
    assert extract_clarify_answer(None) == "[clarify prompt could not be delivered]"


class TypedConn:
    def __init__(self):
        self.calls = []

    async def create_elicitation(self, **kwargs):
        self.calls.append(kwargs)
        return {"action": "accept", "content": {"answer": "typed"}}


class RawConn:
    def __init__(self):
        self.requests = []
        self._conn = self

    async def send_request(self, method, params):
        self.requests.append((method, params))
        return {"action": "accept", "content": {"answer": "raw"}}


@pytest.mark.asyncio
async def test_create_form_elicitation_prefers_typed_helper_if_available():
    conn = TypedConn()
    response = await create_form_elicitation(
        conn,
        session_id="s1",
        question="Q?",
        choices=None,
    )
    assert response["content"]["answer"] == "typed"
    assert conn.calls[0].get("session_id") == "s1" or conn.calls[0].get("sessionId") == "s1"


@pytest.mark.asyncio
async def test_create_form_elicitation_uses_raw_json_rpc_fallback():
    conn = RawConn()
    response = await create_form_elicitation(
        conn,
        session_id="s1",
        question="Q?",
        choices=["A", "B"],
    )
    assert response["content"]["answer"] == "raw"
    method, params = conn.requests[0]
    assert method == "elicitation/create"
    assert params["sessionId"] == "s1"
    assert params["mode"] == "form"
    assert params["message"] == "Q?"
    assert params["requestedSchema"]["required"] == ["answer"]


def test_elicitation_clarify_callback_schedules_on_loop():
    conn = RawConn()
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run_loop():
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    ready.wait(timeout=2)

    try:
        callback = make_elicitation_clarify_callback(conn, loop, "s1", timeout=2)
        assert callback("Q?", ["A", "B"]) == "raw"
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()

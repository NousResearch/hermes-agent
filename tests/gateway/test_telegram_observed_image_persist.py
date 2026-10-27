"""Observed-context turns with a native image persist the clean payload (#137627).

The observed Telegram wrapper is API-only. On the multimodal path the current user row is a
content-part list, and a plain-text persist override cannot replace a part-list content — so the
wrapped text part used to leak into the transcript, embedding the previous wrapped row twice per
image turn (doubling until compaction is blocked as ineffective). The clean override on this path
is the pre-wrap part list itself.
"""

from types import SimpleNamespace

from agent.image_routing import build_native_content_parts
from agent.session_persistence import SessionPersistenceMixin, _db_flush_row
# Module-level on purpose: loading gateway.run settles an interrupted-update probe whose
# module-level isfile must not run under the home IO guard (function-local import would).
from gateway.run import _CURRENT_ADDRESSED_MESSAGE_HEADER, _OBSERVED_GROUP_CONTEXT_HEADER
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext

_DATA_URL = "data:image/png;base64,aGVsbG8="


class _CapturingAgent:
    def __init__(self):
        self.captured = []

    def run_conversation(self, message, **kwargs):
        self.captured.append((message, kwargs))
        return {"completed": True}


def _turn_worker(monkeypatch, tmp_path, *, native_images):
    # A real file so build_native_content_parts admits the path; encoding is patched out.
    photo = tmp_path / "photo.png"
    photo.write_bytes(b"png")
    monkeypatch.setattr("agent.image_routing._file_to_data_url", lambda path: _DATA_URL)
    runner = SimpleNamespace(
        _consume_pending_native_image_paths=lambda key: [str(photo)] if native_images else [],
        _delivery_adapter_for=lambda source: None,
    )
    ctx = TurnContext()
    ctx.message = "what do you see in this photo"
    ctx.session_key = "telegram:-100200300"
    ctx.session_id = "sess-1"
    ctx.source = SimpleNamespace(user_id="42", user_name="Ti", is_bot=False)
    return TurnRunner(runner, ctx)


def _persisting_agent(override):
    agent = object.__new__(SessionPersistenceMixin)
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = override
    agent._persist_user_message_timestamp = None
    agent._persist_user_message_platform_id = None
    return agent


def test_observed_image_turn_persists_clean_part_list(monkeypatch, tmp_path):
    worker = _turn_worker(monkeypatch, tmp_path, native_images=True)
    agent = _CapturingAgent()
    observed = "<alice> earlier chatter"

    worker._run_conversation_with_approval(agent, [], observed, None, None)

    api_message, kwargs = agent.captured[0]
    # The API copy still carries the wrapper: observed context stays model-facing.
    assert isinstance(api_message, list)
    assert api_message[0]["text"].startswith(_OBSERVED_GROUP_CONTEXT_HEADER)
    assert worker._ctx.message in api_message[0]["text"]
    assert any(p.get("type") == "image_url" for p in api_message)
    # The persist override is the pre-wrap part list: clean text plus the native image part.
    persist = kwargs["persist_user_message"]
    assert isinstance(persist, list)
    assert persist[0]["type"] == "text"
    assert _OBSERVED_GROUP_CONTEXT_HEADER not in persist[0]["text"]
    assert _CURRENT_ADDRESSED_MESSAGE_HEADER not in persist[0]["text"]
    assert persist[0]["text"] == build_native_content_parts(
        worker._ctx.message, [str(tmp_path / "photo.png")]
    )[0][0]["text"]
    assert any(p.get("type") == "image_url" for p in persist)
    # End to end down to the session-db row: no observed header ever reaches the transcript.
    row = _db_flush_row(_persisting_agent(persist), {"role": "user", "content": api_message}, True)
    assert _OBSERVED_GROUP_CONTEXT_HEADER not in str(row["content"])
    assert _CURRENT_ADDRESSED_MESSAGE_HEADER not in str(row["content"])


def test_observed_text_turn_still_persists_bare_string(monkeypatch, tmp_path):
    worker = _turn_worker(monkeypatch, tmp_path, native_images=False)
    agent = _CapturingAgent()

    worker._run_conversation_with_approval(agent, [], "some observed backlog", None, None)

    api_message, kwargs = agent.captured[0]
    assert api_message.startswith(_OBSERVED_GROUP_CONTEXT_HEADER)
    assert kwargs["persist_user_message"] is worker._ctx.message


def test_no_observed_context_sends_no_persist_override(monkeypatch, tmp_path):
    worker = _turn_worker(monkeypatch, tmp_path, native_images=True)
    agent = _CapturingAgent()

    worker._run_conversation_with_approval(agent, [], None, None, None)

    api_message, kwargs = agent.captured[0]
    assert isinstance(api_message, list)
    assert "persist_user_message" not in kwargs

"""The common turn prologue can use an exact stored user row as live input."""
import pytest

from agent.session_persistence import SessionPersistenceMixin
from agent.turn_context import DurableUserInput
from persistence.conversation.messages import NewMessage
from persistence.core.identity import SessionId
from persistence_assembly import open_persistence, profile_scope
from run_agent import AIAgent
from tests.agent.test_turn_context import _build, _FakeAgent
from tests.fixtures.storage_seed import seed_session


class _Agent(SessionPersistenceMixin, _FakeAgent):
    def __init__(self, store):
        super().__init__()
        self.persistence = self._session_db = store
        self._session_db_created = True



@pytest.fixture
def stored_user(monkeypatch):
    monkeypatch.setattr("agent.auxiliary_client.set_runtime_main", lambda *a, **kw: None)
    with open_persistence(profile_scope()) as store:
        agent = _Agent(store)
        sid = agent.session_id
        seed_session(store, sid, "cli", title="Already titled")
        store.messages.append(sid, NewMessage(role="user", content="same words"))
        store.messages.append(sid, NewMessage(role="assistant", content="previous reply"))
        row_id = store.messages.append(sid, NewMessage(
            role="user", content="same words", timestamp=123.5, message_uid="exact-input",
            display_metadata={"label": "stored"},
        ))
        yield agent, store, row_id


def test_exact_stored_user_is_current_once_without_reinserting(stored_user):
    agent, store, row_id = stored_user
    sid = agent.session_id
    before = store.messages.read_transcript(sid)
    prefix = [msg.as_openai_message() for msg in store.messages.read_conversation(sid)[:-1]]
    ctx = _build(agent, conversation_history=prefix, user_message="same words",
                 durable_user_input=DurableUserInput(sid, row_id, "exact-input"))

    current = ctx.messages[ctx.current_turn_user_idx]
    assert ctx.current_turn_user_idx == len(prefix) == 2
    assert [(msg["role"], msg["content"]) for msg in ctx.messages] == [
        ("user", "same words"), ("assistant", "previous reply"), ("user", "same words"),
    ]
    assert current["_row_id"] == row_id
    assert current["message_uid"] == "exact-input"
    assert current["_db_persisted"] is True
    assert current["timestamp"] == 123.5
    assert current["display_metadata"] == {"label": "stored"}

    assert store.messages.read_transcript(sid) == before

    # Later flushes still skip the selected row; genuinely new rows still write.
    assert agent._flush_messages_to_session_db([*ctx.messages, {"role": "assistant", "content": "new reply"}])
    after = store.messages.read_transcript(sid)
    assert after[:-1] == before
    assert after[-1].role == "assistant" and after[-1].content == "new reply"


def test_model_only_prefix_preserves_raw_text_and_updates_exact_sidecar(stored_user):
    agent, store, row_id = stored_user
    sid = agent.session_id
    later_id = store.messages.append(sid, NewMessage(role="user", content="same words"))
    prefix = [msg.as_openai_message() for msg in store.messages.read_conversation(sid)[:2]]
    ctx = _build(agent, conversation_history=prefix,
                 user_message="Handle this gently: same words", persist_user_message="same words",
                 durable_user_input=DurableUserInput(sid, row_id, "exact-input"))
    assert ctx.original_user_message == "same words"
    assert ctx.messages[ctx.current_turn_user_idx]["api_content"] == "Handle this gently: same words"
    first, _, selected, later = store.messages.read_transcript(sid)
    assert first.api_content is None
    assert later.id == later_id and later.api_content is None
    assert selected.id == row_id and selected.content == "same words"
    assert selected.api_content == "Handle this gently: same words"


def _wrong_uid(store, sid, row_id):
    return DurableUserInput(sid, row_id, "wrong-uid"), "same words"


def _missing_row(store, sid, row_id):
    return DurableUserInput(sid, row_id + 1, "exact-input"), "same words"


def _foreign_chat(store, sid, row_id):
    foreign_sid = SessionId("foreign")
    seed_session(store, foreign_sid, "cli")
    foreign_id = store.messages.append(foreign_sid, NewMessage(
        role="user", content="same words", message_uid="foreign-input",
    ))
    return DurableUserInput(foreign_sid, foreign_id, "foreign-input"), "same words"


def _assistant_row(store, sid, row_id):
    assistant = store.messages.read_transcript(sid)[1]
    assert assistant.message_uid
    return DurableUserInput(sid, assistant.id, assistant.message_uid), assistant.content


def _rewound_row(store, sid, row_id):
    store.messages.rewind_user_turn(sid, -1)
    return DurableUserInput(sid, row_id, "exact-input"), "same words"


def _different_words(store, sid, row_id):
    return DurableUserInput(sid, row_id, "exact-input"), "different words"


_MISMATCHES = {
    "uid": _wrong_uid, "missing": _missing_row, "foreign": _foreign_chat,
    "role": _assistant_row, "inactive": _rewound_row, "content": _different_words,
}


@pytest.mark.parametrize("mismatch", list(_MISMATCHES))
def test_invalid_durable_reference_does_not_fall_back_to_an_insert(stored_user, mismatch):
    agent, store, row_id = stored_user
    sid = agent.session_id
    identity, content = _MISMATCHES[mismatch](store, sid, row_id)

    before = store.messages.read_transcript(sid)
    with pytest.raises(ValueError, match="durable user input"):
        _build(agent, user_message=content, durable_user_input=identity)
    assert store.messages.read_transcript(sid) == before


def test_next_turn_does_not_inherit_durable_identity(stored_user):
    agent, store, row_id = stored_user
    sid = agent.session_id
    _build(agent, user_message="same words", durable_user_input=DurableUserInput(sid, row_id, "exact-input"))
    ctx = _build(agent, user_message="same words")
    rows = store.messages.read_transcript(sid)
    assert len(rows) == 4
    assert rows[-1].id != row_id and rows[-1].content == "same words"
    assert ctx.messages[ctx.current_turn_user_idx]["_row_id"] == rows[-1].id


def test_real_agent_runs_the_already_stored_user_row(stored_user, monkeypatch):
    fake, store, row_id = stored_user
    sid = fake.session_id
    agent = AIAgent(
        api_key="test-key", base_url="http://127.0.0.1:9/v1", provider="openai-compat", model="test-model",
        enabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True,
        save_trajectories=False, platform="cli", persistence=store, session_id=sid,
    )
    from openai.types.chat import ChatCompletion

    # Exercise the actual facade and persistence; only the external model is replaced.
    sent = []
    def stop_at_model(client, **kwargs):
        # The SDK merges these fields into the body after Hermes bypasses its bulk transform.
        messages = (kwargs.get('extra_body') or {}).get('messages', kwargs['messages'])
        sent.append([dict(message) for message in messages])
        return ChatCompletion.model_validate({
            'id': 'test', 'object': 'chat.completion', 'model': 'test-model', 'created': 0,
            'choices': [{'index': 0, 'finish_reason': 'stop',
                         'message': {'role': 'assistant', 'content': 'new reply'}}],
        })

    monkeypatch.setattr("openai.resources.chat.completions.Completions.create", stop_at_model)
    try:
        result = agent.run_conversation("same words", durable_user_input=DurableUserInput(sid, row_id, "exact-input"))
        assert result['completed'] and result['final_response'] == 'new reply'
        assert [[m['content'] for m in request if m['role'] == 'user']
                for request in sent] == [['same words', 'same words']]
        rows = store.messages.read_transcript(sid)
        assert [row.content for row in rows] == ["same words", "previous reply", "same words", "new reply"]
        assert rows[-2].id == row_id
    finally:
        agent.close()

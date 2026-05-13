from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.codex_responses_adapter import (
    _chat_messages_to_responses_input,
    _preflight_codex_input_items,
)
from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX
from hermes_state import SessionDB


CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"


def _compressor(**overrides):
    kwargs = dict(
        model="gpt-5-codex",
        provider="openai-codex",
        api_mode="codex_responses",
        base_url=CODEX_BASE_URL,
        api_key="codex-token",
        protect_first_n=1,
        protect_last_n=1,
        quiet_mode=True,
    )
    kwargs.update(overrides)
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        c = ContextCompressor(**kwargs)
    # Force a small protected tail so the fixture's middle turns are compacted.
    c.tail_token_budget = 1
    return c


def _messages():
    return [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "First compactable turn"},
        {"role": "assistant", "content": "Assistant compactable turn"},
        {"role": "user", "content": "Second compactable turn"},
        {"role": "assistant", "content": "Second assistant compactable turn"},
        {"role": "user", "content": "Current task stays in tail"},
    ]


class _FakeCompactResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_codex_native_compaction_posts_to_compact_endpoint_and_stores_encrypted_item():
    c = _compressor()
    compact_payload = {"output": [{"type": "compaction_summary", "encrypted_content": "enc_blob"}]}

    with patch("agent.context_compressor.httpx.post", return_value=_FakeCompactResponse(compact_payload)) as post, \
         patch("agent.context_compressor.call_llm") as call_llm:
        compressed = c.compress(_messages(), current_tokens=90000)

    call_llm.assert_not_called()
    assert post.call_count == 1
    url = post.call_args.args[0]
    kwargs = post.call_args.kwargs
    assert url == f"{CODEX_BASE_URL}/responses/compact"
    assert kwargs["headers"]["Authorization"] == "Bearer codex-token"
    assert kwargs["headers"]["x-codex-beta-features"] == "remote_compaction_v2"
    assert kwargs["json"]["model"] == "gpt-5-codex"
    assert kwargs["json"]["instructions"] == "You are Hermes."
    assert kwargs["json"]["tools"] == []
    assert "store" not in kwargs["json"]
    assert {"role": "user", "content": "First compactable turn"} in kwargs["json"]["input"]
    assert all(item.get("type") != "trigger" for item in kwargs["json"]["input"])

    compaction_messages = [m for m in compressed if m.get("codex_compaction_items")]
    assert len(compaction_messages) == 1
    assert compaction_messages[0]["role"] == "assistant"
    assert compaction_messages[0]["content"] == ""
    assert compaction_messages[0]["codex_compaction_items"] == [
        {"type": "compaction_summary", "encrypted_content": "enc_blob"}
    ]


def test_native_compaction_rejects_non_chatgpt_codex_like_base_url():
    c = _compressor(base_url="https://unit.test/codex")
    summary_response = MagicMock()
    summary_response.choices = [MagicMock()]
    summary_response.choices[0].message.content = "Textual fallback summary"

    with patch("agent.context_compressor.httpx.post") as post, \
         patch("agent.context_compressor.call_llm", return_value=summary_response) as call_llm:
        c.compress(_messages(), current_tokens=90000)

    post.assert_not_called()
    call_llm.assert_called_once()


def test_native_compaction_requires_https_codex_base_url():
    c = _compressor(base_url="http://chatgpt.com/backend-api/codex")
    summary_response = MagicMock()
    summary_response.choices = [MagicMock()]
    summary_response.choices[0].message.content = "Textual fallback summary"

    with patch("agent.context_compressor.httpx.post") as post, \
         patch("agent.context_compressor.call_llm", return_value=summary_response) as call_llm:
        c.compress(_messages(), current_tokens=90000)

    post.assert_not_called()
    call_llm.assert_called_once()


def test_native_compaction_after_text_summary_includes_prior_summary_in_compact_input():
    c = _compressor()
    prior_summary = f"{SUMMARY_PREFIX}\nOLD CRITICAL SUMMARY"
    messages = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "older turn already summarized"},
        {"role": "assistant", "content": prior_summary},
        {"role": "user", "content": "new middle 1"},
        {"role": "assistant", "content": "answer 1"},
        {"role": "user", "content": "new middle 2"},
        {"role": "assistant", "content": "answer 2"},
        {"role": "user", "content": "tail"},
    ]
    compact_payload = {"output": [{"type": "compaction_summary", "encrypted_content": "enc_blob"}]}

    with patch("agent.context_compressor.httpx.post", return_value=_FakeCompactResponse(compact_payload)) as post, \
         patch("agent.context_compressor.call_llm") as call_llm:
        compressed = c.compress(messages, current_tokens=90000)

    call_llm.assert_not_called()
    compact_input_text = "\n".join(str(item.get("content", "")) for item in post.call_args.kwargs["json"]["input"])
    assert "OLD CRITICAL SUMMARY" in compact_input_text
    assert any(m.get("codex_compaction_items") for m in compressed)
    assert not any(
        isinstance(m.get("content"), str) and "OLD CRITICAL SUMMARY" in m["content"]
        for m in compressed
    )


def test_codex_compaction_item_replayed_as_responses_input():
    messages = [
        {"role": "user", "content": "before"},
        {
            "role": "assistant",
            "content": "",
            "codex_compaction_items": [
                {"type": "compaction_summary", "encrypted_content": "enc_blob"}
            ],
        },
        {"role": "user", "content": "after"},
    ]

    items = _chat_messages_to_responses_input(messages)

    assert {"type": "compaction_summary", "encrypted_content": "enc_blob"} in items


def test_codex_preflight_accepts_compaction_items():
    normalized = _preflight_codex_input_items([
        {"role": "user", "content": "before"},
        {"type": "compaction_summary", "encrypted_content": "enc_blob"},
        {"role": "user", "content": "after"},
    ])

    assert normalized[1] == {"type": "compaction_summary", "encrypted_content": "enc_blob"}


def test_native_compaction_falls_back_to_textual_summary_on_endpoint_failure():
    c = _compressor()
    summary_response = MagicMock()
    summary_response.choices = [MagicMock()]
    summary_response.choices[0].message.content = "Textual fallback summary"

    with patch("agent.context_compressor.httpx.post", side_effect=RuntimeError("boom")), \
         patch("agent.context_compressor.call_llm", return_value=summary_response) as call_llm:
        compressed = c.compress(_messages(), current_tokens=90000)

    call_llm.assert_called_once()
    assert not any(m.get("codex_compaction_items") for m in compressed)
    assert any(
        isinstance(m.get("content"), str) and m["content"].startswith(SUMMARY_PREFIX)
        for m in compressed
    )


def test_fallback_summary_preserves_existing_codex_compaction_checkpoint():
    c = _compressor()
    messages = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "before old checkpoint"},
        {
            "role": "assistant",
            "content": "",
            "codex_compaction_items": [
                {"type": "compaction_summary", "encrypted_content": "old_enc_blob"}
            ],
        },
        {"role": "user", "content": "new middle turn"},
        {"role": "assistant", "content": "new middle answer"},
        {"role": "user", "content": "tail"},
    ]
    summary_response = MagicMock()
    summary_response.choices = [MagicMock()]
    summary_response.choices[0].message.content = "Textual fallback summary"

    with patch("agent.context_compressor.httpx.post", side_effect=RuntimeError("boom")), \
         patch("agent.context_compressor.call_llm", return_value=summary_response):
        compressed = c.compress(messages, current_tokens=90000)

    preserved = [m for m in compressed if m.get("codex_compaction_items")]
    assert preserved == [
        {
            "role": "assistant",
            "content": "",
            "codex_compaction_items": [
                {"type": "compaction_summary", "encrypted_content": "old_enc_blob"}
            ],
        }
    ]
    assert any(
        isinstance(m.get("content"), str) and m["content"].startswith(SUMMARY_PREFIX)
        for m in compressed
    )


def test_manual_focus_compaction_uses_textual_summary_not_native_endpoint():
    c = _compressor()
    summary_response = MagicMock()
    summary_response.choices = [MagicMock()]
    summary_response.choices[0].message.content = "Focused textual summary"

    with patch("agent.context_compressor.httpx.post") as post, \
         patch("agent.context_compressor.call_llm", return_value=summary_response) as call_llm:
        compressed = c.compress(_messages(), current_tokens=90000, focus_topic="billing")

    post.assert_not_called()
    call_llm.assert_called_once()
    assert not any(m.get("codex_compaction_items") for m in compressed)


def test_codex_compaction_items_survive_session_history_reload(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        session_id = "s1"
        db.create_session(session_id=session_id, source="test", model="gpt-5-codex")
        db.append_message(
            session_id,
            role="assistant",
            content="",
            codex_compaction_items=[{"type": "compaction_summary", "encrypted_content": "enc_blob"}],
        )

        restored = db.get_messages_as_conversation(session_id)
    finally:
        db.close()

    assert restored == [
        {
            "role": "assistant",
            "content": "",
            "codex_compaction_items": [
                {"type": "compaction_summary", "encrypted_content": "enc_blob"}
            ],
        }
    ]

import logging
from types import SimpleNamespace

from agent.codex_responses_adapter import _chat_messages_to_responses_input
from agent.conversation_compression import COMPACTION_STATUS, compress_context
from hermes_state import SessionDB


class FakeTransport:
    def preflight_kwargs(self, api_kwargs, *, allow_stream=False):
        from agent.codex_responses_adapter import _preflight_codex_api_kwargs

        return _preflight_codex_api_kwargs(api_kwargs, allow_stream=allow_stream)


class FakeResponses:
    def __init__(self):
        self.compact_calls = []

    def compact(self, **kwargs):
        self.compact_calls.append(kwargs)
        return SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="compaction_summary",
                    id="cmp_1",
                    encrypted_content="compact_opaque",
                    created_by="responses.compact",
                )
            ],
            usage=None,
        )


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


class FakeSessionDB:
    def __init__(self):
        self.archived = []
        self.system_prompts = []

    def archive_and_compact(self, session_id, messages):
        self.archived.append((session_id, messages))

    def update_system_prompt(self, session_id, system_prompt):
        self.system_prompts.append((session_id, system_prompt))


class DummyCodexResponsesAgent:
    def __init__(
        self,
        *,
        codex_native_compaction=True,
        session_db=None,
        session_id="hermes-session-1",
    ):
        self.api_mode = "codex_responses"
        self.provider = "openai-codex"
        self.codex_native_compaction_enabled = codex_native_compaction
        self.model = "gpt-5.5"
        self.base_url = "https://chatgpt.com/backend-api/codex"
        self.api_key = "stub"
        self.session_id = session_id
        self.platform = "cli"
        self._cached_system_prompt = "cached prompt"
        self._client = FakeClient()
        self.context_compressor = SimpleNamespace(
            protect_last_n=1,
            compression_count=0,
            last_compression_rough_tokens=0,
            last_prompt_tokens=123,
            last_completion_tokens=45,
            awaiting_real_usage_after_compression=False,
            update_from_response=lambda usage: None,
        )
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_api_calls = 0
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = None
        self.session_cost_source = None
        self._session_db = session_db or FakeSessionDB()
        self._last_flushed_db_idx = 3
        self._flushed_db_message_ids = {1, 2, 3}
        self.memory_commits = []
        self.statuses = []
        self.warnings = []
        self.events = []

    def _build_api_kwargs(self, messages):
        return {
            "model": self.model,
            "instructions": self._cached_system_prompt,
            "input": _chat_messages_to_responses_input(messages),
            "store": False,
            "prompt_cache_key": "cache-key",
        }

    def _get_transport(self):
        return FakeTransport()

    def _ensure_primary_openai_client(self, *, reason):
        return self._client

    def _try_refresh_codex_client_credentials(self, *, force=False):
        return False

    def _emit_status(self, message):
        self.statuses.append(message)

    def _emit_warning(self, message):
        self.warnings.append(message)

    def _build_system_prompt(self, system_message):
        return "built prompt"

    def commit_memory_session(self, messages):
        self.memory_commits.append(messages)

    def event_callback(self, name, payload):
        self.events.append((name, payload))


def test_codex_responses_compression_calls_responses_compact_and_keeps_tail():
    agent = DummyCodexResponsesAgent()
    messages = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "latest user"},
    ]

    returned, prompt = compress_context(
        agent,
        messages,
        "system",
        approx_tokens=100000,
        task_id="test",
    )

    assert prompt == "cached prompt"
    assert agent._client.responses.compact_calls == [
        {
            "model": "gpt-5.5",
            "input": [
                {"role": "user", "content": "old user"},
                {"role": "assistant", "content": "old assistant"},
            ],
            "instructions": "cached prompt",
            "prompt_cache_key": "cache-key",
        }
    ]
    assert returned == [
        {
            "role": "assistant",
            "content": "",
            "codex_compaction_items": [
                {
                    "type": "compaction_summary",
                    "encrypted_content": "compact_opaque",
                    "id": "cmp_1",
                    "created_by": "responses.compact",
                }
            ],
        },
        {"role": "user", "content": "latest user"},
    ]
    assert agent.context_compressor.compression_count == 1
    assert agent.context_compressor.last_compression_rough_tokens == 100000
    assert agent.context_compressor.last_prompt_tokens == -1
    assert agent.context_compressor.awaiting_real_usage_after_compression is True
    assert agent.session_api_calls == 1
    assert agent.statuses
    assert agent._session_db.archived == [("hermes-session-1", returned)]
    assert agent._session_db.system_prompts == [
        ("hermes-session-1", "cached prompt")
    ]
    assert agent._last_flushed_db_idx == 0
    assert agent._flushed_db_message_ids == set()
    assert agent.memory_commits == [messages]
    assert agent.events == [
        (
            "session:compress",
            {
                "platform": "cli",
                "session_id": "hermes-session-1",
                "old_session_id": "",
                "in_place": True,
                "compression_count": 1,
                "runtime": "codex_responses",
                "compaction_items": 1,
            },
        )
    ]


def test_codex_responses_compression_persists_compacted_state_db(
    tmp_path, caplog
):
    db = SessionDB(tmp_path / "state.db")
    session_id = "codex-responses-session"
    db.create_session(session_id=session_id, source="cli", model="gpt-5.5")
    messages = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "latest user"},
    ]
    for msg in messages:
        db.append_message(session_id, role=msg["role"], content=msg["content"])

    agent = DummyCodexResponsesAgent(
        session_db=db,
        session_id=session_id,
    )

    with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
        returned, prompt = compress_context(
            agent,
            messages,
            "system",
            approx_tokens=100000,
            task_id="test",
        )

    assert prompt == "cached prompt"
    assert returned == [
        {
            "role": "assistant",
            "content": "",
            "codex_compaction_items": [
                {
                    "type": "compaction_summary",
                    "encrypted_content": "compact_opaque",
                    "id": "cmp_1",
                    "created_by": "responses.compact",
                }
            ],
        },
        {"role": "user", "content": "latest user"},
    ]
    assert agent.statuses == [COMPACTION_STATUS]
    assert "codex responses compaction started" in caplog.text
    assert "codex responses compaction done" in caplog.text

    active = db.get_messages_as_conversation(session_id)
    assert len(active) == 2
    assert active[0]["role"] == "assistant"
    assert active[0]["content"] == ""
    assert active[0]["codex_compaction_items"] == [
        {
            "type": "compaction_summary",
            "encrypted_content": "compact_opaque",
            "id": "cmp_1",
            "created_by": "responses.compact",
        }
    ]
    assert active[1]["content"] == "latest user"

    all_rows = db.get_messages(session_id, include_inactive=True)
    assert [(row["active"], row["compacted"]) for row in all_rows[:3]] == [
        (0, 1),
        (0, 1),
        (0, 1),
    ]
    assert [(row["active"], row["compacted"]) for row in all_rows[3:]] == [
        (1, 0),
        (1, 0),
    ]

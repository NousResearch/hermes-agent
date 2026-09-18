"""Memory-provider sync is an egress channel: scrub turns before the fan-out.

``security.redact_secrets`` covers display/telemetry surfaces, but
``MemoryManager.sync_all`` → ``provider.sync_turn`` and ``on_session_end``
ship turn content to external memory providers, whose stores archive it
verbatim (on-disk session transcripts, extraction VLM payloads). The same
fail-closed scrub telemetry uses (``agent.redact.redact_for_egress``) is
applied here once in MemoryManager, so every provider (OpenViking, Mem0,
Honcho, Hindsight, ...) is covered without touching each backend.

See: agent.memory_manager.MemoryManager.sync_all / _redact_rows_for_provider.
"""

from agent import memory_manager
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from agent.redact import REDACTION_UNAVAILABLE

_SK_SECRET = "sk-ant-api03-1234567890abcdef1234567890abcdef"
_TURN_WITH_SECRET = f'Remember the deploy notes\napi_key = "{_SK_SECRET}"'
_BEARER_TURN = "Authorization: Bearer AQ.opaque46chartokenwithnovendorprefixshape"
_TOOL_ROW = {
    "role": "tool",
    "tool_call_id": "call_1",
    "content": f'ov.conf dump: {{"api_key": "{_SK_SECRET}"}}',
}


class _RecordingProvider(MemoryProvider):
    """Captures exactly what the sync/session-end fan-out delivered."""

    _name = "recording"

    def __init__(self):
        self.synced = []
        self.session_rows = None

    @property
    def name(self) -> str:
        return self._name

    def initialize(self, session_id: str = "", **kwargs) -> None:
        pass

    def is_available(self) -> bool:
        return True

    def system_prompt_block(self) -> str:
        return ""

    def sync_turn(
        self,
        user_content,
        assistant_content,
        *,
        session_id: str = "",
        messages=None,
        turn_author=None,
    ) -> None:
        self.synced.append((user_content, assistant_content, messages))

    def on_session_end(self, messages) -> None:
        self.session_rows = messages

    def get_tool_schemas(self):
        return []


def _manager_with_recorder():
    mgr = MemoryManager()
    provider = _RecordingProvider()
    mgr.add_provider(provider)
    return mgr, provider


class TestSyncAllRedactsEgress:
    def test_user_and_assistant_content_scrubbed(self):
        mgr, provider = _manager_with_recorder()
        mgr.sync_all(_TURN_WITH_SECRET, _BEARER_TURN)
        mgr.flush_pending(timeout=5.0)
        ((user, assistant, messages),) = provider.synced
        assert _SK_SECRET not in user
        assert "AQ.opaque46" not in assistant
        assert "Remember the deploy notes" in user

    def test_message_rows_scrubbed_without_mutating_caller_rows(self):
        mgr, provider = _manager_with_recorder()
        rows = [{"role": "system", "content": "sys"}, dict(_TOOL_ROW)]
        mgr.sync_all("question", "answer", messages=rows)
        mgr.flush_pending(timeout=5.0)
        ((_, _, forwarded),) = provider.synced
        assert _SK_SECRET not in forwarded[1]["content"]
        assert "ov.conf dump" in forwarded[1]["content"]
        assert _SK_SECRET in rows[1]["content"]
        assert rows[1]["tool_call_id"] == forwarded[1]["tool_call_id"]

    def test_non_string_content_rows_pass_through_untouched(self):
        mgr, provider = _manager_with_recorder()
        rows = [
            {"role": "assistant", "content": None},
            {"role": "user", "content": "plain"},
        ]
        mgr.sync_all("question", "answer", messages=rows)
        mgr.flush_pending(timeout=5.0)
        ((_, _, forwarded),) = provider.synced
        assert forwarded[0]["content"] is None
        assert forwarded[1]["content"] == "plain"

    def test_on_session_end_rows_scrubbed(self):
        mgr, provider = _manager_with_recorder()
        rows = [dict(_TOOL_ROW)]
        mgr.on_session_end(rows)
        assert _SK_SECRET not in provider.session_rows[0]["content"]
        assert _SK_SECRET in rows[0]["content"]

    def test_redactor_failure_fails_closed(self, monkeypatch):
        from agent import redact as redact_module

        def _boom(text, **kwargs):
            raise RuntimeError("redactor exploded")

        monkeypatch.setattr(redact_module, "redact_sensitive_text", _boom)
        mgr, provider = _manager_with_recorder()
        mgr.sync_all(_TURN_WITH_SECRET, "reply")
        mgr.flush_pending(timeout=5.0)
        ((user, assistant, _),) = provider.synced
        assert user == REDACTION_UNAVAILABLE
        assert assistant == REDACTION_UNAVAILABLE
        assert _SK_SECRET not in user and _SK_SECRET not in assistant

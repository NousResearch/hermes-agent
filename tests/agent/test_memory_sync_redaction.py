"""Memory-provider sync scrubs secrets before provider egress.

Regression for #115104: ``MemoryManager.sync_all`` forwarded turn content to
external memory providers without ``redact_for_egress``, so secrets present in
tool output were archived verbatim in the provider's store. The scrub covers
the turn strings AND the forwarded ``messages`` transcript slice (same leak,
same path), on defensive copies — the live transcript is never mutated.
"""

from agent.memory_manager import MemoryManager, _redact_message_for_egress
from agent.memory_provider import MemoryProvider


# Proven redacted by tests/agent/test_redact.py (asserts on absence, never markers).
_SHAPED_SECRET = "export OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012"
_OPAQUE_SECRET = "Authorization: Bearer opaque0123456789abcdef"


class _RecordingProvider(MemoryProvider):
    """Captures exactly what each sync_turn fan-out received."""

    _name = "recording"

    def __init__(self):
        self.synced = []

    @property
    def name(self) -> str:
        return self._name

    def initialize(self, session_id: str = "", **kwargs) -> None:
        pass

    def is_available(self) -> bool:
        return True

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query, *, session_id: str = "") -> str:
        return ""

    def queue_prefetch(self, query, *, session_id: str = "") -> None:
        pass

    def sync_turn(self, user_content, assistant_content, *, session_id: str = "",
                  messages=None, **kwargs) -> None:
        self.synced.append((user_content, assistant_content, messages))

    def get_tool_schemas(self):
        return []


def _manager_with_recorder():
    mgr = MemoryManager()
    provider = _RecordingProvider()
    mgr.add_provider(provider)
    return mgr, provider


class TestSyncAllRedactsTurnStrings:
    def test_shaped_secret_in_assistant_content_is_scrubbed(self):
        mgr, provider = _manager_with_recorder()
        mgr.sync_all("check the deploy", f"done, config was {_SHAPED_SECRET}")
        mgr.flush_pending(timeout=5.0)

        assert len(provider.synced) == 1
        user_content, assistant_content, _ = provider.synced[0]
        assert user_content == "check the deploy"
        assert "abc123def456" not in assistant_content

    def test_opaque_bearer_in_user_content_is_scrubbed(self):
        mgr, provider = _manager_with_recorder()
        mgr.sync_all(f"curl -H '{_OPAQUE_SECRET}' https://x.example", "ok")
        mgr.flush_pending(timeout=5.0)

        assert len(provider.synced) == 1
        user_content, _, _ = provider.synced[0]
        assert "opaque0123456789abcdef" not in user_content
        assert "https://x.example" in user_content


class TestSyncAllRedactsMessagesSlice:
    def test_tool_output_secret_in_messages_is_scrubbed(self):
        mgr, provider = _manager_with_recorder()
        messages = [
            {"role": "user", "content": "read the config"},
            {"role": "assistant", "content": "reading it now"},
            {"role": "tool", "content": f"ov.conf contents: {_SHAPED_SECRET}"},
        ]
        mgr.sync_all("read the config", "reading it now", messages=messages)
        mgr.flush_pending(timeout=5.0)

        assert len(provider.synced) == 1
        _, _, received = provider.synced[0]
        assert received is not None
        assert "abc123def456" not in received[2]["content"]
        # Live transcript untouched: the provider got copies, not aliases.
        assert "abc123def456" in messages[2]["content"]
        assert received[2] is not messages[2]

    def test_clean_messages_pass_through_unchanged(self):
        mgr, provider = _manager_with_recorder()
        messages = [
            {"role": "user", "content": "the deploy is green"},
            {"role": "assistant", "content": "great news"},
        ]
        mgr.sync_all("the deploy is green", "great news", messages=messages)
        mgr.flush_pending(timeout=5.0)

        _, _, received = provider.synced[0]
        assert [m["content"] for m in received] == ["the deploy is green", "great news"]

    def test_multimodal_text_parts_are_scrubbed(self):
        mgr, provider = _manager_with_recorder()
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": f"token dump: {_OPAQUE_SECRET}"},
                {"type": "image_url", "image_url": {"url": "data:..."}},
            ]},
        ]
        mgr.sync_all("token dump", "noted", messages=messages)
        mgr.flush_pending(timeout=5.0)

        _, _, received = provider.synced[0]
        assert "opaque0123456789abcdef" not in received[0]["content"][0]["text"]
        assert received[0]["content"][1] == {"type": "image_url", "image_url": {"url": "data:..."}}


class TestRedactMessageForEgress:
    def test_non_dict_and_empty_content_pass_through(self):
        assert _redact_message_for_egress(None) is None
        assert _redact_message_for_egress("plain") == "plain"
        message = {"role": "user", "content": ""}
        assert _redact_message_for_egress(message) is message

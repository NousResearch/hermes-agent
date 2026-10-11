"""Every MemoryManager fan-out scrubs secrets before a memory provider sees them (#115104).

Providers archive whatever they are handed, so a key echoed into tool output used to land verbatim in the
provider's store. Asserts on absence of the secret, never on a mask format.
"""
import json

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider

_SECRET = "sk-proj-abc123def456ghi789jkl012"
_LEAK = f"export OPENAI_API_KEY={_SECRET} and Authorization: Bearer opaque0123456789abcdef"
_IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


class _Recorder(MemoryProvider):
    pre_compress_checkpoint_api_version = 2

    def __init__(self):
        self.seen = []

    @property
    def name(self):
        return "recorder"

    def initialize(self, session_id="", **kw): pass
    def is_available(self): return True
    def system_prompt_block(self): return ""
    def get_tool_schemas(self): return [{"name": "rec_tool", "description": "d", "parameters": {"type": "object"}}]
    def prefetch(self, query, *, session_id=""): self.seen.append(query); return ""
    def queue_prefetch(self, query, *, session_id=""): self.seen.append(query)
    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None, **kw): self.seen.append((user_content, assistant_content, messages))
    def handle_tool_call(self, tool_name, args, **kw): self.seen.append(args); return "{}"
    def on_turn_start(self, turn_number, message, **kw): self.seen.append((message, kw))
    def on_session_end(self, messages): self.seen.append(messages)
    def on_pre_compress(self, messages, **kw): self.seen.append(messages); return ""
    def on_delegation(self, task, result, **kw): self.seen.append((task, result))
    def on_memory_write(self, action, target, content, metadata=None): self.seen.append((content, metadata))


def _transcript():
    return [{"role": "user", "content": "read the config"},
            {"role": "tool", "content": f"ov.conf: {_LEAK}"},
            {"role": "user", "content": [{"type": "text", "text": _LEAK}, _IMAGE]},
            {"role": "assistant", "tool_calls": [{"function": {"arguments": f'{{"cmd": "{_LEAK}"}}'}}]}]


_TRANSCRIPT_FANOUTS = {"sync_all", "on_session_end", "on_pre_compress"}
FANOUTS = {
    "sync_all": lambda m, t: m.sync_all(f"user {_LEAK}", f"assistant {_LEAK}", messages=t),
    "prefetch_all": lambda m, t: m.prefetch_all(f"recall {_LEAK}"),
    "queue_prefetch_all": lambda m, t: m.queue_prefetch_all(f"recall {_LEAK}"),
    "handle_tool_call": lambda m, t: m.handle_tool_call("rec_tool", {"q": _LEAK, "nested": [_LEAK]}),
    "on_turn_start": lambda m, t: m.on_turn_start(2, _LEAK, previous_message=_LEAK, author_name="Ada"),
    "on_session_end": lambda m, t: m.on_session_end(t),
    "on_pre_compress": lambda m, t: m.on_pre_compress(t, evidence_messages=t),
    "on_delegation": lambda m, t: m.on_delegation(f"task {_LEAK}", f"result {_LEAK}"),
    "on_memory_write": lambda m, t: m.on_memory_write("add", "memory", _LEAK, metadata={"old_text": _LEAK}),
}


def _manager():
    mgr, provider = MemoryManager(), _Recorder()
    mgr.add_provider(provider)
    return mgr, provider


@pytest.mark.parametrize("fanout", sorted(FANOUTS))
def test_every_provider_fanout_scrubs_secrets_without_touching_the_live_transcript(fanout, monkeypatch):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)
    (mgr, provider), transcript = _manager(), _transcript()
    FANOUTS[fanout](mgr, transcript)
    mgr.flush_pending(timeout=5.0)

    assert provider.seen, f"{fanout} never reached the provider"
    seen = repr(provider.seen)
    assert _SECRET not in seen and "opaque0123456789abcdef" not in seen
    if fanout in _TRANSCRIPT_FANOUTS:
        assert "read the config" in seen and repr(_IMAGE) in seen  # clean text and inline media pass through
    assert _SECRET in repr(transcript)  # providers got copies; the agent's own transcript is untouched


def test_scrub_holds_with_redact_secrets_off(monkeypatch):
    # Provider egress follows chat/cron egress: redact_secrets governs local logs, not what leaves the agent.
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    mgr, provider = _manager()
    mgr.sync_all("hi", _LEAK)
    mgr.flush_pending(timeout=5.0)
    assert _SECRET not in repr(provider.seen)


def test_key_detected_in_a_json_tool_result_is_also_masked_where_the_model_repeats_it_bare():
    # The #115104 incident shape: an opaque (no vendor prefix) key in an ov.conf dump inside a JSON tool result,
    # then echoed by the assistant in prose where no pattern alone could recognise it.
    key = "AQ.Ab8RN6Jx2kq9Zs0VwT4yLm3PbQe7HcUfGdA1nX5oIr"
    tool = json.dumps({"output": json.dumps({"embedding": {"api_key": key}, "model": "gemini"})})
    mgr, provider = _manager()
    mgr.sync_all("show my ov.conf", f"Your key is {key}.", messages=[{"role": "tool", "content": tool}])
    mgr.flush_pending(timeout=5.0)
    user, _assistant, messages = provider.seen[0]
    assert key not in repr(provider.seen)
    assert user == "show my ov.conf" and json.loads(json.loads(messages[0]["content"])["output"])["model"] == "gemini"

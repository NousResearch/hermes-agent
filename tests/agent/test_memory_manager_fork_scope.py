"""Host-side fork scope in ``MemoryManager`` (``agent_context="subagent"``).

Under ``compression.checkpoint_required`` the fork sites build their child with
``skip_memory=False`` so it reaches the checkpoint contract (``initialize()`` +
``on_pre_compress``). That also wakes the ordinary write/recall fan-outs, which must not
run for a fork: the old ``skip_memory=True`` guard existed to keep its harness prompt and
output out of the user's memory. The manager enforces it from the ``agent_context``
captured at ``initialize_all``, whether or not a provider honours the label; the
checkpoint path stays open.
"""

from agent.memory_manager import MemoryManager
from agent.memory_provider import PRE_COMPRESS_CHECKPOINT_API_VERSION, MemoryProvider


class _RecordingProvider(MemoryProvider):
    """Records every fan-out the manager routes to it; reads no ``agent_context`` itself."""

    pre_compress_checkpoint_api_version = PRE_COMPRESS_CHECKPOINT_API_VERSION

    def __init__(self):
        self.calls = []

    @property
    def name(self):
        return "recording"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        return None

    def get_tool_schemas(self):
        return []

    def on_turn_start(self, turn_number, message, **kwargs):
        self.calls.append("on_turn_start")

    def prefetch(self, query, *, session_id=""):
        self.calls.append("prefetch")
        return "recall"

    def queue_prefetch(self, query, *, session_id=""):
        self.calls.append("queue_prefetch")

    def sync_turn(self, user_content, assistant_content, *, session_id="", **kwargs):
        self.calls.append("sync_turn")

    def on_session_end(self, messages):
        self.calls.append("on_session_end")

    def on_delegation(self, task, result, *, child_session_id="", **kwargs):
        self.calls.append("on_delegation")

    def on_pre_compress(self, messages):
        self.calls.append("on_pre_compress")
        return "checkpoint-saved"


_WRITE_AND_RECALL = ["on_turn_start", "prefetch", "sync_turn", "queue_prefetch", "on_session_end", "on_delegation"]


def _manager_with(provider, *, agent_context):
    mgr = MemoryManager()
    mgr.add_provider(provider)
    mgr.initialize_all("sess-1", agent_context=agent_context)
    return mgr


def _drive_session(mgr):
    mgr.on_turn_start(1, "user text")
    recall = mgr.prefetch_all("user text")
    mgr.sync_all("user text", "assistant text")
    mgr.queue_prefetch_all("user text")
    mgr.flush_pending(timeout=5)
    mgr.on_session_end([{"role": "user", "content": "user text"}])
    mgr.on_delegation("task", "result", child_session_id="child")
    return recall


def test_subagent_context_suppresses_every_write_and_recall_path():
    provider = _RecordingProvider()
    recall = _drive_session(_manager_with(provider, agent_context="subagent"))

    assert provider.calls == []
    assert recall == ""


def test_primary_and_cron_contexts_still_write_and_recall():
    """The control, and the scope: cron keeps the provider-side contract it had before."""
    for context in ("primary", "cron"):
        provider = _RecordingProvider()
        recall = _drive_session(_manager_with(provider, agent_context=context))
        assert sorted(provider.calls) == sorted(_WRITE_AND_RECALL), context
        assert recall, context


def test_manager_never_initialized_keeps_the_historical_primary_writes():
    provider = _RecordingProvider()
    mgr = MemoryManager()
    mgr.add_provider(provider)

    _drive_session(mgr)

    assert "sync_turn" in provider.calls


def test_subagent_checkpoint_on_pre_compress_still_saves():
    """The fork scope must not swallow the checkpoint: a v2 provider is driven and
    ``require_checkpoint`` is satisfied (no RuntimeError)."""
    provider = _RecordingProvider()
    mgr = _manager_with(provider, agent_context="subagent")
    messages = [{"role": "user", "content": "evidence"}, {"role": "assistant", "content": "reply"}]

    summary = mgr.on_pre_compress(messages, require_checkpoint=True,
                                  checkpoint_api_version=PRE_COMPRESS_CHECKPOINT_API_VERSION)

    assert provider.calls == ["on_pre_compress"]
    assert "checkpoint-saved" in summary

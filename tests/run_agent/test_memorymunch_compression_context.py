"""Regression tests for MemoryMunch no-drift compression context."""
from pathlib import Path

from agent.conversation_compression import compress_context
from agent.memory_manager import MemoryManager


class FakeMemoryManager:
    def __init__(self):
        self.called = False
        self.switched = False

    def on_pre_compress(self, messages):
        self.called = True
        return '<memorymunch-guardrail boundary="compression">Do not drift after compaction.</memorymunch-guardrail>'

    def on_session_switch(self, *args, **kwargs):
        self.switched = True


class FakeCompressor:
    compression_count = 0
    last_prompt_tokens = 0
    last_completion_tokens = 0
    _last_summary_error = None
    _last_aux_model_failure_model = None
    _last_aux_model_failure_error = None

    def __init__(self):
        self.seen_messages = None
        self.called = False

    def compress(self, messages, current_tokens=None, focus_topic=None):
        self.called = True
        self.seen_messages = messages
        return [{"role": "system", "content": "compressed"}]


class FakeSessionDB:
    def get_session_title(self, sid):
        return "Title"

    def end_session(self, sid, reason):
        pass

    def create_session(self, **kwargs):
        pass

    def get_next_title_in_lineage(self, title):
        return title + " 2"

    def set_session_title(self, sid, title):
        pass

    def update_system_prompt(self, sid, prompt):
        self.prompt = prompt


class FakeTodo:
    def format_for_injection(self):
        return ""


class FakeAgent:
    session_id = "old-sess"
    model = "gpt-test"
    platform = "telegram"
    tools = []
    logs_dir = Path("/tmp")
    session_log_file = None
    _last_compression_summary_warning = None
    _last_aux_fallback_warning_key = None
    _session_db_created = True
    _session_init_model_config = {}
    _todo_store = FakeTodo()
    _session_db = FakeSessionDB()
    _memory_manager = FakeMemoryManager()
    context_compressor = FakeCompressor()

    def __init__(self):
        self._memory_manager = FakeMemoryManager()
        self.context_compressor = FakeCompressor()

    def _emit_status(self, msg):
        pass

    def _emit_warning(self, msg):
        pass

    def _vprint(self, *args, **kwargs):
        pass

    def _invalidate_system_prompt(self):
        pass

    def _build_system_prompt(self, system_message):
        return "sys"

    def commit_memory_session(self, messages):
        pass


def test_compression_injects_exact_trigger_query_and_memory_provider_context():
    agent = FakeAgent()
    compress_context(
        agent,
        [
            {"role": "user", "content": "older task"},
            {"role": "assistant", "content": "older answer"},
            {"role": "user", "content": "EXACT_TRIGGER_QUERY before compaction"},
        ],
        approx_tokens=100,
        system_message="sys",
    )

    assert agent._memory_manager.called is True
    assert agent.context_compressor.seen_messages is not None
    injected = "\n".join(str(m.get("content", "")) for m in agent.context_compressor.seen_messages)
    assert "Exact pre-compression user message" in injected
    assert "EXACT_TRIGGER_QUERY before compaction" in injected
    assert "Memory provider compression context" in injected
    assert "Do not drift after compaction" in injected
    assert "background continuity evidence, not as a new user request" in injected
    assert agent._memory_manager.switched is True


class SourceOfTruthMemoryManager(FakeMemoryManager):
    def build_source_of_truth_compaction(self, messages, **kwargs):
        self.called = True
        return [
            {
                "role": "system",
                "content": (
                    "MemoryMunch/Graphify source-of-truth compaction checkpoint. "
                    "Use ACTIVE_SESSION_LEDGER and GRAPHIFY_REPORT as background continuity evidence, not a new user request."
                ),
            },
            {"role": "user", "content": kwargs.get("last_user_message", "")},
        ]


def test_memorymunch_source_of_truth_compaction_bypasses_blocking_compressor():
    agent = FakeAgent()
    agent._memory_manager = SourceOfTruthMemoryManager()

    compressed, _ = compress_context(
        agent,
        [
            {"role": "user", "content": "older task"},
            {"role": "assistant", "content": "older answer"},
            {"role": "user", "content": "LIVE_QUERY keep moving while compaction would normally pause"},
        ],
        approx_tokens=100,
        system_message="sys",
    )

    assert agent._memory_manager.called is True
    assert agent.context_compressor.called is False
    joined = "\n".join(str(m.get("content", "")) for m in compressed)
    assert "MemoryMunch/Graphify source-of-truth compaction checkpoint" in joined
    assert "ACTIVE_SESSION_LEDGER" in joined
    assert "GRAPHIFY_REPORT" in joined
    assert "LIVE_QUERY keep moving" in joined
    assert agent._memory_manager.switched is True


class MalformedSourceOfTruthProvider(FakeMemoryManager):
    name = "malformed"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return []

    def build_source_of_truth_compaction(self, messages, **kwargs):
        self.called = True
        return [{"role": "tool", "content": "bad"}]


def test_malformed_source_of_truth_compaction_falls_back_to_normal_compressor():
    agent = FakeAgent()
    provider = MalformedSourceOfTruthProvider()
    manager = MemoryManager()
    manager.add_provider(provider)
    agent._memory_manager = manager

    compressed, _ = compress_context(
        agent,
        [
            {"role": "user", "content": "older task"},
            {"role": "assistant", "content": "older answer"},
            {"role": "user", "content": "current query"},
        ],
        approx_tokens=100,
        system_message="sys",
    )

    assert provider.called is True
    assert agent.context_compressor.called is True
    assert compressed == [{"role": "system", "content": "compressed"}]


def test_source_of_truth_compaction_clears_stale_compressor_warning_state_and_counts():
    agent = FakeAgent()
    agent._memory_manager = SourceOfTruthMemoryManager()
    agent.context_compressor._last_summary_error = "stale previous error"
    agent.context_compressor._last_aux_model_failure_model = "stale-model"
    agent.context_compressor._last_aux_model_failure_error = "stale-error"

    compress_context(
        agent,
        [
            {"role": "user", "content": "older task"},
            {"role": "assistant", "content": "older answer"},
            {"role": "user", "content": "current query"},
        ],
        approx_tokens=100,
        system_message="sys",
    )

    assert agent.context_compressor.called is False
    assert agent.context_compressor.compression_count == 1
    assert agent.context_compressor._last_summary_error is None
    assert agent.context_compressor._last_aux_model_failure_model is None
    assert agent.context_compressor._last_aux_model_failure_error is None
    assert agent._last_compression_summary_warning is None

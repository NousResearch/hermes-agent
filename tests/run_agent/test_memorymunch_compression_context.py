"""Regression tests for MemoryMunch no-drift compression context."""
from pathlib import Path

from agent.conversation_compression import compress_context


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

    def compress(self, messages, current_tokens=None, focus_topic=None):
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

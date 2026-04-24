from types import SimpleNamespace

from run_agent import AIAgent


class FakeMemoryManager:
    def on_pre_compress(self, messages):
        return "preserve Chorus handoff"


class FakeCompressor:
    compression_count = 0
    last_prompt_tokens = 0
    last_completion_tokens = 0

    def __init__(self):
        self.focus_topic = None

    def compress(self, messages, current_tokens=None, focus_topic=None):
        self.focus_topic = focus_topic
        return list(messages)


class FakeTodoStore:
    def format_for_injection(self):
        return ""


def test_memory_provider_pre_compress_context_is_threaded_into_focus():
    agent = object.__new__(AIAgent)
    compressor = FakeCompressor()
    agent._memory_manager = FakeMemoryManager()
    agent.context_compressor = compressor
    agent._todo_store = FakeTodoStore()
    agent._session_db = None
    agent.session_id = "test"
    agent.model = "test-model"
    agent.compression_enabled = True
    agent.quiet_mode = True
    agent.platform = "cli"
    agent.logs_dir = None
    agent.session_log_file = None
    agent._cached_system_prompt = ""
    agent._last_flushed_db_idx = 0
    agent._vprint = lambda *args, **kwargs: None
    agent._safe_print = lambda *args, **kwargs: None
    agent.flush_memories = lambda *args, **kwargs: None
    agent.commit_memory_session = lambda *args, **kwargs: None
    agent._invalidate_system_prompt = lambda: None
    agent._build_system_prompt = lambda system_message: system_message

    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    agent._compress_context(messages, "system", approx_tokens=10, focus_topic="manual focus")

    assert "manual focus" in compressor.focus_topic
    assert "preserve Chorus handoff" in compressor.focus_topic

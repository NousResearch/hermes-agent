from types import SimpleNamespace


class _TodoStore:
    def format_for_injection(self):
        return ""


class _FakeCompressor:
    def __init__(self):
        self._last_compress_aborted = False
        self._last_summary_error = None
        self._last_aux_model_failure_model = None
        self._last_aux_model_failure_error = None
        self.compression_count = 1
        self.called_with = None

    def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
        self.called_with = list(messages)
        return [
            {"role": "user", "content": "head"},
            {
                "role": "assistant",
                "content": "[CONTEXT COMPACTION — REFERENCE ONLY] summary body",
                "_compressed_summary": True,
            },
            {"role": "user", "content": "tail"},
        ]

    def on_session_start(self, *args, **kwargs):
        pass


def _agent(tmp_path, *, enabled=True):
    compressor = _FakeCompressor()
    return SimpleNamespace(
        session_id="session/core",
        model="test/model",
        platform="cli",
        compression_in_place=True,
        compression_enabled=True,
        controlled_context_rebuild_enabled=enabled,
        controlled_context_rebuild_packet_budget=5000,
        controlled_context_rebuild_checkpoint_budget=5000,
        context_compressor=compressor,
        _memory_manager=None,
        _session_db=None,
        _todo_store=_TodoStore(),
        _last_compaction_in_place=False,
        _cached_system_prompt=None,
        _compression_warning=None,
        _emit_status=lambda *a, **k: None,
        _emit_warning=lambda *a, **k: None,
        _invalidate_system_prompt=lambda: None,
        _build_system_prompt=lambda system_message: f"SYSTEM:{system_message}",
        _current_main_runtime=lambda: {},
        _compression_feasibility_checked=True,
        tools=[],
        log_prefix="",
    )


def test_compress_context_prefixes_summary_and_writes_checkpoint(tmp_path, monkeypatch):
    from agent.conversation_compression import compress_context
    from agent.controlled_context_rebuild import CONTROLLED_CONTEXT_REBUILD_HEADER
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    token = set_hermes_home_override(tmp_path)
    try:
        agent = _agent(tmp_path, enabled=True)
        messages = [
            {
                "role": "user",
                "content": "давай patch /home/niko/.hermes/hermes-agent/agent/conversation_compression.py",
            },
            {
                "role": "assistant",
                "content": "will run pytest tests/agent/test_controlled_context_rebuild.py -q",
            },
        ]

        compressed, system_prompt = compress_context(
            agent,
            messages,
            "sys",
            approx_tokens=100_000,
            task_id="t",
        )
    finally:
        reset_hermes_home_override(token)

    summary = compressed[1]["content"]
    assert summary.startswith(CONTROLLED_CONTEXT_REBUILD_HEADER)
    assert "## LLM Compaction Summary" in summary
    assert (
        "/home/niko/.hermes/hermes-agent/agent/conversation_compression.py" in summary
    )
    assert "pytest tests/agent/test_controlled_context_rebuild.py -q" in summary
    assert [m["role"] for m in compressed] == ["user", "assistant", "user"]
    assert system_prompt == "SYSTEM:sys"
    checkpoint = tmp_path / "context" / "sessions" / "session_core" / "checkpoint.md"
    assert checkpoint.exists()
    assert CONTROLLED_CONTEXT_REBUILD_HEADER in checkpoint.read_text(encoding="utf-8")


def test_compress_context_can_disable_controlled_rebuild(tmp_path):
    from agent.conversation_compression import compress_context
    from agent.controlled_context_rebuild import CONTROLLED_CONTEXT_REBUILD_HEADER

    agent = _agent(tmp_path, enabled=False)
    compressed, _system_prompt = compress_context(
        agent,
        [{"role": "user", "content": "patch /x.py"}],
        "sys",
        approx_tokens=100_000,
        task_id="t",
    )

    assert not compressed[1]["content"].startswith(CONTROLLED_CONTEXT_REBUILD_HEADER)

"""Tests for general-session context handoff checkpoints."""

from __future__ import annotations

import json
from types import SimpleNamespace


class _FakeTodoStore:
    def __init__(self, items=None):
        self._items = list(items or [])

    def read(self):
        return list(self._items)

    def format_for_injection(self):
        if not self._items:
            return None
        return "[Your active task list was preserved across context compression]\n" + "\n".join(
            f"- [ ] {item['id']}. {item['content']} ({item['status']})"
            for item in self._items
        )


def test_build_session_handoff_includes_latest_request_and_next_action():
    from agent.session_handoff import build_session_handoff

    handoff = build_session_handoff(
        session_id="sess-123",
        context_pct=0.91,
        messages=[
            {"role": "user", "content": "Implement session handoff"},
            {"role": "assistant", "content": "Added tests for the core builder."},
            {"role": "user", "content": "Finish with TDD and keep current request first."},
        ],
        todos=[
            {"id": "core", "content": "Implement builder", "status": "completed"},
            {"id": "cli", "content": "Wire /session-check", "status": "pending"},
        ],
        changed_files=["agent/session_handoff.py", "tests/agent/test_session_handoff.py"],
    )

    assert handoff["schema"] == "hermes.session_handoff.v1"
    assert handoff["session_id"] == "sess-123"
    assert handoff["context_pct"] == 0.91
    assert "Finish with TDD" in handoff["active_user_request"]
    assert any("/session-check" in item for item in handoff["next_actions"])
    assert "latest user request" in handoff["resume_prompt"].lower()
    assert handoff["handoff_id"].startswith("h_")


def test_handoff_redacts_secrets_from_nested_fields():
    from agent.session_handoff import build_session_handoff

    handoff = build_session_handoff(
        session_id="sess-secret",
        context_pct=0.8,
        messages=[
            {"role": "user", "content": "Use API key sk-proj-abcdefghijklmnop1234567890"},
        ],
        commands_run=[
            {"command": "curl https://example.test", "exit_code": 1},
        ],
        current_state=["Token sk-proj-abcdefghijklmnop1234567890 appeared in output"],
    )

    dumped = json.dumps(handoff, sort_keys=True)
    assert "sk-proj-abcdefghijklmnop1234567890" not in dumped
    assert "[REDACTED]" in dumped


def test_write_session_handoff_bundle_persists_bounded_receipts(tmp_path):
    from agent.session_handoff import build_session_handoff, write_session_handoff_bundle

    handoff = build_session_handoff(
        session_id="sess-write",
        context_pct=0.7,
        messages=[{"role": "user", "content": "Continue the repo work"}],
        todos=[{"id": "finish", "content": "Run verification", "status": "pending"}],
        max_chars=900,
    )
    bundle = write_session_handoff_bundle(handoff=handoff, root=tmp_path, max_chars=900)

    assert bundle.handoff_id == handoff["handoff_id"]
    assert bundle.json_path.exists()
    assert bundle.markdown_path.exists()
    assert tmp_path in bundle.json_path.parents

    persisted = json.loads(bundle.json_path.read_text(encoding="utf-8"))
    assert persisted["handoff_id"] == handoff["handoff_id"]
    assert "Continue from handoff" in persisted["resume_prompt"]
    assert "latest user request" in persisted["reference_warning"].lower()
    assert bundle.markdown_path.read_text(encoding="utf-8").startswith("# Session handoff ready:")


def test_short_notice_does_not_include_full_handoff_body():
    from agent.session_handoff import build_session_handoff, format_short_notice

    handoff = build_session_handoff(
        session_id="sess-short",
        context_pct=0.76,
        messages=[{"role": "user", "content": "Very long implementation details " * 200}],
        max_chars=1200,
    )
    notice = format_short_notice(handoff)

    assert handoff["handoff_id"] in notice
    assert "Context: 76%" in notice
    assert "Very long implementation details" not in notice
    assert len(notice) < 300


def test_compress_context_writes_pre_compression_handoff(tmp_path, monkeypatch):
    from agent.conversation_compression import compress_context

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    class FakeCompressor:
        context_length = 1000
        last_prompt_tokens = 850
        last_completion_tokens = 0
        awaiting_real_usage_after_compression = False
        compression_count = 0
        _last_compress_aborted = False
        _last_summary_error = None
        _last_aux_model_failure_model = None
        _last_aux_model_failure_error = None

        def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
            self.compression_count += 1
            return [{"role": "user", "content": "Compressed summary"}]

    agent = SimpleNamespace(
        session_id="compress-session",
        model="unit-test-model",
        platform="cli",
        tools=[],
        config={"context_handoff": {"enabled": True, "max_chars": 1200}},
        context_compressor=FakeCompressor(),
        _compression_feasibility_checked=True,
        _cached_system_prompt="system prompt",
        _memory_manager=None,
        _session_db=None,
        _todo_store=_FakeTodoStore([
            {"id": "verify", "content": "Run the final verification", "status": "pending"},
        ]),
        event_callback=None,
        log_prefix="",
    )
    agent._emit_status = lambda message: None
    agent._emit_warning = lambda message: None
    agent._vprint = lambda *args, **kwargs: None
    agent._invalidate_system_prompt = lambda: None
    agent._build_system_prompt = lambda system_message: "new system prompt"

    messages = [
        {"role": "user", "content": "Initial request"},
        {"role": "assistant", "content": "Did partial work"},
        {"role": "user", "content": "Continue after sk-proj-abcdefghijklmnopqrstuvwxyz1234567890"},
    ]

    compressed, new_prompt = compress_context(
        agent,
        messages,
        "system prompt",
        approx_tokens=850,
        task_id="test",
    )

    assert compressed[0]["content"] == "Compressed summary"
    assert new_prompt == "new system prompt"
    receipts = list((tmp_path / "session_handoffs" / "compress-session").glob("*.json"))
    assert len(receipts) == 1
    persisted = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert persisted["session_id"] == "compress-session"
    assert persisted["context_pct"] == 0.85
    assert "Run the final verification" in persisted["resume_prompt"]
    dumped = json.dumps(persisted, sort_keys=True)
    assert "sk-pro...7890" not in dumped


def test_context_pressure_checkpoint_writes_once_and_emits_short_notice(tmp_path):
    from agent.session_handoff import maybe_checkpoint_context_pressure

    notices = []
    agent = SimpleNamespace(
        session_id="pressure-session",
        config={
            "context_handoff": {
                "enabled": True,
                "threshold": 0.65,
                "critical_threshold": 0.80,
                "message_limit": 300,
                "max_chars": 1400,
            }
        },
        context_compressor=SimpleNamespace(context_length=1000),
        _todo_store=_FakeTodoStore([
            {"id": "next", "content": "Continue with context pressure hook", "status": "pending"},
        ]),
    )
    agent._emit_status = notices.append

    messages = [
        {"role": "user", "content": "Build the automatic context handoff trigger"},
        {"role": "assistant", "content": "A" * 5000},
        {"role": "user", "content": "Keep the notice tiny"},
    ]

    first = maybe_checkpoint_context_pressure(
        agent,
        messages,
        used_tokens=850,
        root=tmp_path,
        reason="pre_api_context_pressure",
    )
    second = maybe_checkpoint_context_pressure(
        agent,
        messages,
        used_tokens=860,
        root=tmp_path,
        reason="pre_api_context_pressure",
    )

    assert first is not None
    assert second is None
    receipts = list((tmp_path / "session_handoffs" / "pressure-session").glob("*.json"))
    assert len(receipts) == 1
    persisted = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert persisted["context_pct"] == 0.85
    assert "pre_api_context_pressure" in persisted["risks"][0]
    assert any("Session handoff ready:" in notice for notice in notices)
    assert all("A" * 100 not in notice for notice in notices)

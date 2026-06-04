from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path.home() / ".hermes" / "hermes-agent" / "environments" / "roleplay_persona_env" / "run_local_roleplay_batch.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("roleplay_batch", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_make_agent_marks_local_runtime_as_custom(monkeypatch):
    module = _load_module()
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(module, "AIAgent", FakeAgent)
    monkeypatch.delenv("HERMES_STREAM_STALE_TIMEOUT", raising=False)
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)

    module.make_agent("http://127.0.0.1:8000/v1", "local-key", "gemma-4-e4b-it-4bit")

    assert captured["provider"] == "custom"
    assert captured["api_mode"] == "chat_completions"
    assert captured["base_url"] == "http://127.0.0.1:8000/v1"
    assert captured["api_key"] == "local-key"
    assert captured["model"] == "gemma-4-e4b-it-4bit"
    assert module.os.environ["HERMES_STREAM_STALE_TIMEOUT"] == "150"
    assert module.os.environ["HERMES_API_CALL_STALE_TIMEOUT"] == "180"


def test_case_timeout_for_scenario_uses_tighter_guardrails():
    module = _load_module()

    assert module.case_timeout_for_scenario("persona_boot", 600) == 180
    assert module.case_timeout_for_scenario("persona_multiturn", 600) == 180
    assert module.case_timeout_for_scenario("persona_conflict", 600) == 180
    assert module.case_timeout_for_scenario("persona_long_context", 600) == 420
    assert module.case_timeout_for_scenario("persona_long_context", 300) == 300


def test_extract_final_reply_handles_none_final_response_without_crashing():
    module = _load_module()

    assert module.extract_final_reply([], None) == ""


def test_extract_final_reply_prefers_last_assistant_message_even_when_final_response_is_none():
    module = _load_module()

    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "  嗷呜  "},
    ]

    assert module.extract_final_reply(messages, None) == "嗷呜"

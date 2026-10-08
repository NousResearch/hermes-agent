"""Compaction marks through the real NeMo Relay ATOF plugin, and the freshness reset they drive.

Relay records an LLM start projected to the current user turn while its agent scope is not fresh. A
``compaction`` mark on the session's own stack makes the agent fresh again, so the next LLM start
records the full compacted history. An aborted attempt is only a ``compaction.attempt`` mark and must
not reset anything.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

SESSION_ID = "compaction-atof-session"
SECRET = "TOPSECRET_TRANSCRIPT_TEXT"
LLM_NAME = "openai.chat_completions"
ABORT = object()  # summary sentinel: fail the summary with abort_on_summary_failure set


class _TodoStore:
    def format_for_injection(self):
        return ""


class _Agent:
    def __init__(self, compressor):
        self.context_compressor = compressor
        self.session_id = SESSION_ID
        self.platform = "cli"
        self.model = "test/main-model"
        self.provider = "test-provider"
        self.tools = []
        self._compression_feasibility_checked = True
        self.compression_in_place = True
        self._memory_manager = None
        self._session_db = None
        self._todo_store = _TodoStore()
        self._cached_system_prompt = None

    def _emit_status(self, _message):
        pass

    def _emit_warning(self, _message):
        pass

    def _invalidate_system_prompt(self):
        self._cached_system_prompt = None

    def _build_system_prompt(self, system_message):
        return system_message

    def commit_memory_session(self, _messages):
        pass


def _transcript():
    filler = " ".join(["context"] * 200)
    msgs = [{"role": "system", "content": "system prompt"}]
    for idx in range(10):
        msgs.append({"role": "user", "content": f"user message {idx} {SECRET} {filler}"})
        msgs.append({"role": "assistant", "content": f"assistant reply {idx} {SECRET} {filler}"})
    return msgs


def _chat_response(_request):
    return {
        "id": "r", "object": "chat.completion", "created": 0, "model": "test/main-model",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
    }


def _llm(messages, request_id):
    from agent import relay_llm

    relay_llm.execute(
        {"model": "test/main-model", "messages": messages}, _chat_response, name=LLM_NAME,
        model_name="test/main-model", session_id=SESSION_ID,
        metadata={"api_mode": "chat_completions", "api_request_id": request_id},
    )


def _write_plugins_toml(tmp_path, atof_dir):
    config = tmp_path / "plugins.toml"
    config.write_text(
        f"""version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 4

[components.config.atof]
enabled = true

[[components.config.atof.sinks]]
type = "file"
output_directory = {json.dumps(str(atof_dir))}
filename = "events.jsonl"
mode = "overwrite"
""",
        encoding="utf-8",
    )
    return config


def _run_turn_with_compaction(tmp_path, monkeypatch, summary):
    """One Relay turn: two LLM calls, one compression attempt, one more LLM call. Returns ATOF events
    and the message count of the last request."""
    relay = pytest.importorskip("nemo_relay")
    if getattr(relay, "_native", None) is None:
        pytest.skip("NeMo Relay native binding is unavailable on this platform")

    from agent import relay_runtime
    from agent.context_compressor import ContextCompressor
    from agent.conversation_compression import compress_context

    hermes_home, atof_dir = tmp_path / "hermes-home", tmp_path / "atof"
    hermes_home.mkdir()
    atof_dir.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv(relay_runtime.RELAY_PLUGINS_CONFIG_ENV, str(_write_plugins_toml(tmp_path, atof_dir)))
    relay_runtime._reset_for_tests()

    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(
            model="test/main-model", provider="test-provider", threshold_percent=0.50, quiet_mode=True,
            config_context_length=100_000, abort_on_summary_failure=summary is ABORT,
        )
    compressor.tail_token_budget = 10
    agent = _Agent(compressor)
    transcript = _transcript()
    coordinator = relay_runtime.SESSION_COORDINATOR
    profile_key = relay_runtime.current_profile_key()
    try:
        lease = coordinator.acquire_conversation(profile_key=profile_key, session_id=SESSION_ID, platform="cli")
        turn = coordinator.begin_turn(lease, turn_id="turn-1", task_id="task-1")
        try:
            _llm(transcript[:-2], "r1")
            _llm(transcript, "r2")
            with patch.object(compressor, "_generate_summary", return_value=None if summary is ABORT else summary):
                compressed, _ = compress_context(agent, transcript, "system prompt", approx_tokens=80_000)
            last_request = compressed + [{"role": "user", "content": "next question"}]
            _llm(last_request, "r3")
        finally:
            coordinator.end_turn(turn, outcome="success")
            coordinator.finalize_conversation(profile_key=profile_key, session_id=SESSION_ID)
    finally:
        relay_runtime._reset_for_tests()

    events = [
        json.loads(line)
        for line in (atof_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return events, len(last_request)



def _llm_start_message_counts(events):
    counts = []
    for event in events:
        if event.get("name") == LLM_NAME and event.get("scope_category") == "start":
            data = event.get("data") or {}
            counts.append(len((data.get("content") or data).get("messages") or []))
    return counts


def _compaction_marks(events):
    return [e for e in events if e.get("kind") == "mark" and str(e.get("name", "")).startswith("compaction")]


@pytest.mark.parametrize(
    ("summary", "method"), [("SANITIZED SUMMARY", "llm_summary"), (None, "deterministic_fallback")],
)
def test_committed_compaction_mark_resets_freshness_for_the_next_llm_start(tmp_path, monkeypatch, summary, method):
    events, last_request_len = _run_turn_with_compaction(tmp_path, monkeypatch, summary)

    [mark] = _compaction_marks(events)
    assert mark["name"] == "compaction"
    assert mark["data_schema"] == {"name": "hermes.compaction", "version": "1"}
    assert (mark["data"]["kind"], mark["data"]["outcome"], mark["data"]["method"]) == ("summarize", "committed", method)
    [turn_start] = [e for e in events if e.get("name") == "hermes.turn" and e.get("scope_category") == "start"]
    assert mark["parent_uuid"] == turn_start["uuid"]
    raw = json.dumps(mark)
    assert SECRET not in raw and "SANITIZED SUMMARY" not in raw

    first, second, after = _llm_start_message_counts(events)
    assert second < len(_transcript())  # not fresh: projected to the current user turn
    assert after == last_request_len  # fresh again after the compaction mark: full history


def test_aborted_attempt_is_an_attempt_mark_and_leaves_freshness_alone(tmp_path, monkeypatch):
    events, last_request_len = _run_turn_with_compaction(tmp_path, monkeypatch, ABORT)

    [mark] = _compaction_marks(events)
    assert mark["name"] == "compaction.attempt"
    assert (mark["data"]["outcome"], mark["data"]["failure_class"]) == ("failed", "summary_generation_aborted")
    *_, after = _llm_start_message_counts(events)
    assert after < last_request_len

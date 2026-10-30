"""A closed agent's message bodies leave the token-estimate memo.

``_MSG_TOKENS_CACHE`` keys hold every string they fingerprint, so after ``AIAgent.close()`` a long-lived
``hermes serve`` kept each closed session's message bodies resident until 4096 newer messages evicted
them (~50 MB after 20 sessions x 10 turns of 256 KB replies on macOS).
"""

import threading
from typing import Any

from agent.model_metadata import _MSG_TOKENS_CACHE, estimate_messages_tokens_rough
from run_agent import AIAgent


def _held() -> set[str]:
    """Long strings the memo still holds."""
    return _held_in(list(_MSG_TOKENS_CACHE.items()))


def _held_in(value: Any) -> set[str]:
    found: set[str] = set()
    stack: list[Any] = [value]
    while stack:
        value = stack.pop()
        if isinstance(value, (list, tuple)):
            stack.extend(value)
        elif isinstance(value, str) and len(value) > 64:
            found.add(value)
    return found


def _closing_agent(messages: list) -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "memo-release"
    agent.client = None
    agent._active_children_lock = threading.Lock()
    agent._active_children = set()
    agent._end_session_on_close = False
    agent._session_messages = messages
    return agent


def test_close_evicts_the_conversations_memo_entries_and_keeps_others():
    reply = "closed-session reply " * 5000
    other = "another live session's reply " * 5000
    closed = [{"role": "user", "content": "hi there, a long enough opening question for the memo key?"},
              {"role": "assistant", "content": reply}]
    live = [{"role": "assistant", "content": other}]
    before = estimate_messages_tokens_rough(closed)
    estimate_messages_tokens_rough(live)
    assert reply in _held() and other in _held()

    _closing_agent(closed).close()

    assert reply not in _held()
    assert other in _held()
    # Eviction only drops a pure-function memo: the estimate is unchanged when recomputed.
    assert estimate_messages_tokens_rough(closed) == before


def test_compression_evicts_the_bodies_the_summary_replaced(tmp_path, monkeypatch):
    """close() only sees the compressed history, so the bodies compression dropped are released then."""
    import os
    from unittest.mock import MagicMock, patch

    from agent import conversation_compression
    from agent.conversation_compression import CompressionCommitFence
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("S", source="cli")
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                        platform="cli", quiet_mode=True, session_db=db, session_id="S",
                        skip_context_files=True, skip_memory=True)
    agent.context_compressor = MagicMock()
    dropped = "pre-compression tool output " * 5000
    kept = "kept tail reply " * 5000
    history = [{"role": "user", "content": "q" * 80}, {"role": "tool", "content": dropped},
               {"role": "assistant", "content": kept}]
    estimate_messages_tokens_rough(history)
    compressed = [{"role": "user", "content": "[summary] " + "s" * 80}, history[2]]
    monkeypatch.setattr(conversation_compression, "compress_context",
                        lambda *_a, **_kw: (compressed, "prompt"))
    assert dropped in _held()

    agent._compress_context(history, "sys", commit_fence=CompressionCommitFence())

    assert dropped not in _held()
    assert kept in _held()  # still live in the compressed history


def test_close_evicts_the_system_prompt_the_history_never_holds():
    system_prompt = "a long per-session system prompt body " * 2000
    estimate_messages_tokens_rough([{"role": "system", "content": system_prompt}])
    agent = _closing_agent([{"role": "user", "content": "hi"}])
    agent._cached_system_prompt = system_prompt
    assert system_prompt in _held()

    agent.close()

    assert system_prompt not in _held()


def test_compression_telemetry_does_not_hold_the_one_shot_summary_prompt():
    from agent.context_compressor_telemetry import CompressionTelemetryMixin

    recorder = CompressionTelemetryMixin()
    recorder._active_compression_telemetry = {"effective_aux_context": None}
    prompt = "summarize this serialized transcript " * 20000

    recorder._record_aux_compression_call(prompt_messages=[{"role": "user", "content": prompt}],
                                          max_tokens=None, duration_ms=1)

    assert recorder._active_compression_telemetry["aux_prompt_tokens"] > 0
    assert prompt not in _held()


def test_close_evicts_a_prompt_caching_split_system_prompt():
    from agent.prompt_caching import apply_anthropic_cache_control

    static = "the static, cacheable system prefix " * 2000
    full = static + "\nper-session suffix: today's date and the active model"
    request = apply_anthropic_cache_control([{"role": "system", "content": full}, {"role": "user", "content": "hi"}],
                                            static_system_prefix=static)
    assert isinstance(request[0]["content"], list)  # the real split shape the memo sees
    estimate_messages_tokens_rough(request)
    agent = _closing_agent([{"role": "user", "content": "hi"}])
    agent._cached_system_prompt = full
    agent._cached_system_prompt_static = static
    assert static in _held()

    agent.close()

    assert static not in _held()


def _turn_agent() -> AIAgent:
    from unittest.mock import MagicMock, patch

    with (patch("model_tools.get_tool_definitions", return_value=[]),
          patch("model_tools.check_toolset_requirements", return_value={}),
          patch("agent.process_bootstrap.OpenAI")):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.client = MagicMock()
    agent._disable_streaming = True
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _run_marked_turn(agent: AIAgent) -> None:
    from unittest.mock import patch

    from tests.agent.test_run_agent import _mock_response

    agent._cached_system_prompt = "a long cached per-session system prompt " * 500
    agent.ephemeral_system_prompt = "an ephemeral gateway channel prompt " * 200
    agent.prefill_messages = [{"role": "user", "content": "a prefilled example question " * 100},
                              {"role": "assistant", "content": "a prefilled example answer " * 100}]
    agent._interruptible_api_call = lambda _kw: _mock_response(content="the reply", finish_reason="stop")
    with (patch.object(agent, "_flush_messages_to_session_db"), patch.object(agent, "_persist_session"),
          patch.object(agent, "_save_trajectory"), patch.object(agent, "_cleanup_task_resources")):
        agent.run_conversation("a pasted log the user ended with a newline " * 100 + "\n")


def _request_only() -> list[str]:
    return [s for s in _held() if "ephemeral gateway" in s or "prefilled example" in s or "pasted log" in s]


def test_close_evicts_the_request_only_strings_a_real_turn_estimated():
    """The ephemeral system-prompt concat, prefill rows and the whitespace-stripped user row reach the memo
    only through the per-call request copy; no history or cached prompt holds them, so release by value
    alone left them until FIFO eviction."""
    agent = _turn_agent()
    _run_marked_turn(agent)
    assert len(_request_only()) >= 4  # concat, 2 prefill rows, stripped user row (+ the history's copy)

    agent.close()

    assert not _request_only()


def test_release_clients_evicts_the_turns_memo_entries():
    """Gateway cache eviction and background-review forks end in release_clients(), never close()."""
    agent = _turn_agent()
    _run_marked_turn(agent)
    agent._session_messages = []  # gateway eviction drops the history right after release_clients()
    assert _request_only()

    agent.release_clients()

    assert not _request_only()


def test_a_rebuilt_equal_string_reuses_its_memo_entry():
    """Per-request strings rebuilt with the same bytes (the ephemeral concat, a skill-split tail, a MoA
    tool-result fold) hit the existing entry instead of holding one more copy per request."""
    base = "a cached system prompt body " * 500
    first = estimate_messages_tokens_rough([{"role": "system", "content": base + "\n\nephemeral"}])
    for _ in range(5):
        assert estimate_messages_tokens_rough([{"role": "system", "content": base + "\n\nephemeral"}]) == first

    target = base + "\n\nephemeral"
    assert sum(1 for key, _entry in list(_MSG_TOKENS_CACHE.items()) if target in _held_in(key)) == 1

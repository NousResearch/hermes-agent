"""A closed agent's message bodies leave the token-estimate memo.

``_MSG_TOKENS_CACHE`` pins every string it fingerprints, so after ``AIAgent.close()`` a long-lived
``hermes serve`` kept each closed session's message bodies resident until 4096 newer messages evicted
them (~50 MB after 20 sessions x 10 turns of 256 KB replies on macOS).
"""

import threading

from agent.model_metadata import _MSG_TOKENS_CACHE, estimate_messages_tokens_rough
from run_agent import AIAgent


def _pinned_ids() -> set[int]:
    return {id(pin) for pins, _text, _images in _MSG_TOKENS_CACHE.values() for pin in pins}


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
    assert id(reply) in _pinned_ids() and id(other) in _pinned_ids()

    _closing_agent(closed).close()

    assert id(reply) not in _pinned_ids()
    assert id(other) in _pinned_ids()
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
    assert id(dropped) in _pinned_ids()

    agent._compress_context(history, "sys", commit_fence=CompressionCommitFence())

    assert id(dropped) not in _pinned_ids()
    assert id(kept) in _pinned_ids()  # still live in the compressed history


def test_close_evicts_the_system_prompt_the_history_never_holds():
    system_prompt = "a long per-session system prompt body " * 2000
    estimate_messages_tokens_rough([{"role": "system", "content": system_prompt}])
    agent = _closing_agent([{"role": "user", "content": "hi"}])
    agent._cached_system_prompt = system_prompt
    assert id(system_prompt) in _pinned_ids()

    agent.close()

    assert id(system_prompt) not in _pinned_ids()


def test_compression_telemetry_does_not_pin_the_one_shot_summary_prompt():
    from agent.context_compressor_telemetry import CompressionTelemetryMixin

    recorder = CompressionTelemetryMixin()
    recorder._active_compression_telemetry = {"effective_aux_context": None}
    prompt = "summarize this serialized transcript " * 20000

    recorder._record_aux_compression_call(prompt_messages=[{"role": "user", "content": prompt}],
                                          max_tokens=None, duration_ms=1)

    assert recorder._active_compression_telemetry["aux_prompt_tokens"] > 0
    assert id(prompt) not in _pinned_ids()


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
    assert id(static) in _pinned_ids()

    agent.close()

    assert id(static) not in _pinned_ids()

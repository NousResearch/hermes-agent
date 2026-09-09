"""Tests for the ``ci`` plugin context engine (plugins/context_engine/ci).

Covers the ANCOS Phase-2B validation matrix on top of the ``bounded`` guarantees:
hard-budget preservation with T3 durable memory and T4 selective historical-session
retrieval supplements, authority/priority semantics (Knowledge wins, T3 before T4,
both before the current request), dedup / relevance / clamp bounds, second-stage
(re-)estimation, single-shot retrieval (no amplification), fail-closed tier failure,
fail-open engine failure, and the "no historical need -> no search" default.

All T3/T4 sources are injected via seams (``_memory_store_loader`` /
``_session_search_fn``) so every scenario is deterministic and hermetic; one
integration test drives the real ``load_on_disk_store`` path with a temp memory dir.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List

import pytest

from agent.model_metadata import estimate_messages_tokens_rough
from plugins.context_engine import discover_context_engines, load_context_engine
from plugins.context_engine.bounded import _CLAMP_MARKER
from plugins.context_engine.ci import (
    ContextIntelligenceEngine,
    _AUTHORITY_NOTE,
    _T3_LABEL,
    _T4_LABEL,
    register,
)


def _msg(role: str, content: Any = "", **extra: Any) -> Dict[str, Any]:
    m: Dict[str, Any] = {"role": role, "content": content}
    m.update(extra)
    return m


def _tokenish_text(seed: int, words: int) -> str:
    return " ".join(f"tok_{seed}_{i}_word" for i in range(words))


def _tool_turn(call_id: str, result_text: str) -> List[Dict[str, Any]]:
    return [
        _msg(
            "assistant",
            content="calling a tool",
            tool_calls=[
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "web_search", "arguments": "{}"},
                }
            ],
        ),
        _msg("tool", content=result_text, tool_call_id=call_id, name="web_search"),
        _msg("tool", content="second result " + result_text[-2000:], tool_call_id=call_id, name="web_search"),
    ]


def _build_big_request(rounds: int = 12, words: int = 12000, system: str = "You are a helpful assistant."):
    msgs: List[Dict[str, Any]] = [_msg("system", content=system)]
    for r in range(rounds):
        msgs.append(_msg("user", content=_tokenish_text(r, words)))
        if r % 3 == 0:
            msgs += _tool_turn(f"call{r}", result_text=_tokenish_text(r + 100, words))
        msgs.append(_msg("assistant", content=_tokenish_text(r + 200, 120)))
    msgs.append(_msg("user", content="CURRENT_REQUEST_SENTINEL " + _tokenish_text(999, 400)))
    return msgs


def _fake_store(
    memory_entries: List[str] = (),
    user_entries: List[str] = (),
    *,
    memory_enabled: bool = True,
    user_profile_enabled: bool = True,
):
    from tools.memory_tool_store import ENTRY_DELIMITER, MemoryStore

    store = MemoryStore(
        memory_char_limit=2200,
        user_char_limit=1375,
        memory_enabled=memory_enabled,
        user_profile_enabled=user_profile_enabled,
    )
    if memory_entries:
        store._system_prompt_snapshot["memory"] = ENTRY_DELIMITER.join(memory_entries)
    if user_entries:
        store._system_prompt_snapshot["user"] = ENTRY_DELIMITER.join(user_entries)
    return store


def _empty_store() -> Any:
    return _fake_store()


def _discovery_payload(*contents: str) -> Dict[str, Any]:
    """A canned success payload whose first result carries ``contents`` as messages."""
    return {
        "success": True,
        "results": [
            {
                "session_id": "recall-session-1",
                "matched_role": "user",
                "match_message_id": 1,
                "messages": [
                    {"id": i, "role": "user", "content": content} for i, content in enumerate(contents, start=1)
                ],
            }
        ],
        "count": 1,
    }


def _recording_search(*responses: Any) -> tuple:
    calls: List[Dict[str, Any]] = []

    def _proxy(**kwargs: Any) -> Any:
        calls.append(kwargs)
        if not responses:
            return {"success": True, "results": []}
        return responses[min(len(responses), len(calls)) - 1]

    return _proxy, calls


def _engine(context_length: int = 262144, store: Any = None, **kwargs: Any) -> ContextIntelligenceEngine:
    e = ContextIntelligenceEngine(**kwargs)
    e.context_length = context_length
    if store is not None:
        e._memory_store_loader = lambda: store
    elif e._memory_store_loader is None:
        e._memory_store_loader = lambda: _fake_store()
    e._session_search_fn = e._session_search_fn or (lambda **k: {"success": True, "results": []})
    return e


def _join(selected: List[Dict[str, Any]]) -> str:
    return "\n".join(str(m.get("content")) for m in selected)


def _count_labels(selected: List[Dict[str, Any]], label: str) -> int:
    return sum(1 for m in selected if label in str(m.get("content")))


def _assert_pair_valid(selected: List[Dict[str, Any]]) -> None:
    created_ids = set()
    for m in selected:
        for tc in m.get("tool_calls") or []:
            if isinstance(tc, dict) and tc.get("id"):
                created_ids.add(tc["id"])
    for m in selected:
        if m.get("role") == "tool":
            assert m.get("tool_call_id") in created_ids, "dangling tool result retained"


def _assert_replay_order(selected: List[Dict[str, Any]], original: List[Dict[str, Any]]) -> None:
    positions = {id(m): i for i, m in enumerate(original)}
    last = -1
    for m in selected:
        pos = positions.get(id(m), last)
        assert pos >= last, "selected messages are not in original order"
        last = pos


# ---- identity / discovery -----------------------------------------------------------
def test_name_and_zero_arg_instantiation():
    e = ContextIntelligenceEngine()
    assert e.name == "ci"
    assert isinstance(e, ContextIntelligenceEngine)


def test_plugin_discovery_includes_ci():
    assert "ci" in {name for name, _, _ in discover_context_engines()}


def test_load_context_engine_by_name():
    e = load_context_engine("ci")
    assert e is not None and e.name == "ci"


def test_register_entrypoint_captures_engine():
    class Collector:
        def __init__(self):
            self.engine = None

        def register_context_engine(self, engine):
            self.engine = engine

    collector = Collector()
    register(collector)
    assert collector.engine is not None and collector.engine.name == "ci"


# ---- fail-open + budget plumbing ----------------------------------------------------
def test_select_context_none_when_unknown_budget():
    assert _engine(context_length=0).select_context([_msg("user", content="hi")]) is None


def test_select_context_none_when_disabled():
    assert _engine(enabled=False).select_context([_msg("user", content="hi")], budget_tokens=262144) is None


@pytest.mark.parametrize("bad", ["nope", [], None, 42])
def test_select_context_fail_open_on_invalid_input(bad):
    assert _engine().select_context(bad, budget_tokens=262144) is None


def test_short_path_returns_unchanged_list_with_empty_memory():
    req = [_msg("system", content="S"), _msg("user", content="short")]
    out = _engine().select_context(req, budget_tokens=262144)
    assert out is req


def test_intelligence_stage_disabled_returns_base_selection():
    req = _build_big_request()
    e = _engine(intelligence_enabled=False)
    out = e.select_context(req, budget_tokens=262144)
    assert out is not None
    assert estimate_messages_tokens_rough(out) <= e._hard_budget_for(262144)


# ---- preservation guarantees (inherited bounded invariants) -------------------------
def test_system_and_current_request_preserved_with_memory():
    store = _fake_store(memory_entries=["remembered fact about migrations"])
    e = _engine(store=store, context_length=262144)
    req = _build_big_request()
    out = e.select_context(req, budget_tokens=262144)
    assert out is not None
    assert out[0]["role"] == "system"
    assert "CURRENT_REQUEST_SENTINEL" in str(out[-1].get("content"))


def test_request_messages_not_mutated_with_supplements():
    req = [_msg("system", content="S"), _msg("user", content="CURRENT what did we do earlier about refunds")]
    snapshot = copy.deepcopy(req)
    store = _fake_store(memory_entries=["refund policy entry"])
    _engine(store=store).select_context(req, budget_tokens=262144)
    assert req == snapshot


def test_replay_order_is_subsequence_with_supplements():
    req = [_msg("system", content="S"), _msg("user", content="first")]
    req.append(_msg("assistant", content="reply"))
    req.append(_msg("user", content="CURRENT what did we say earlier about x-ray optics"))
    store = _fake_store(memory_entries=["x-ray optics notes"])
    out = _engine(store=store, context_length=65536).select_context(req, budget_tokens=65536)
    assert out is not None
    assert out[0]["role"] == "system"
    _assert_replay_order(out, req)
    assert out[-1] is req[-1]


def test_pair_validity_with_history_retrieval():
    req = _build_big_request()
    req[-1] = _msg("user", content="CURRENT what did we do about the refund earlier")
    e = _engine(
        context_length=65536,
        history_retrieval_enabled=True,
        _session_search_fn=lambda **k: _discovery_payload("the refund policy for repeat merchants"),
    )
    out = e.select_context(req, budget_tokens=65536)
    assert out is not None
    _assert_pair_valid(out)


# ---- T3: durable memory -------------------------------------------------------------
def test_t3_injects_bounded_memory_supplement():
    req = [_msg("system", content="S"), _msg("user", content="CURRENT small request")]
    store = _fake_store(memory_entries=["the coffee contract renews in May"], user_entries=["Lives in Jakarta"])
    e = _engine(store=store)
    out = e.select_context(req, budget_tokens=262144)
    assert out is not None
    joined = _join(out)
    assert _T3_LABEL in joined
    assert _AUTHORITY_NOTE in joined
    assert "the coffee contract renews in May" in joined
    assert "Lives in Jakarta" in joined
    assert out[-1] is req[-1]


def test_empty_memory_no_overhead():
    req = [_msg("system", content="S"), _msg("user", content="hello")]
    e = _engine(store=_fake_store())
    out = e.select_context(req, budget_tokens=262144)
    assert out is req
    assert _T3_LABEL not in _join(out)


def test_memory_target_disabled_by_config_flags_is_skipped():
    req = [_msg("system", content="S"), _msg("user", content="hello")]
    store = _fake_store(
        memory_entries=["secret notes"], memory_enabled=False, user_profile_enabled=False
    )
    out = _engine(store=store).select_context(req, budget_tokens=262144)
    assert out is req
    assert "secret notes" not in _join(out)


def test_memory_already_in_system_prompt_is_deduped():
    req = [_msg("system", content="MEMORY (your personal notes): tax rate is 10%."), _msg("user", content="hello")]
    store = _fake_store(memory_entries=["MEMORY (your personal notes)", "tax rate is 10%."])
    out = _engine(store=store).select_context(req, budget_tokens=262144)
    assert out is req
    assert _T3_LABEL not in _join(out)


def test_oversized_memory_is_bounded_and_clamped():
    req = [_msg("system", content="S"), _msg("user", content="CURRENT small")]
    store = _fake_store(memory_entries=["M" * 3000, "N" * 3000, "O" * 3000, "P" * 3000])
    e = _engine(store=store, context_length=8192, supplement_max_ratio=1.0)
    out = e.select_context(req, budget_tokens=8192)
    assert out is not None
    joined = _join(out)
    assert _T3_LABEL in joined
    assert _CLAMP_MARKER in joined
    assert "M" * 3000 not in joined
    assert estimate_messages_tokens_rough(out) <= e._hard_budget_for(8192)
    assert out[-1] is req[-1]


def test_memory_failure_fails_closed_to_bounded_selection(caplog):
    req = [_msg("system", content="S"), _msg("user", content="CURRENT keep going")]
    store_loader = lambda: (_ for _ in ()).throw(RuntimeError("memory unreadable"))
    e = _engine(context_length=65536)
    e._memory_store_loader = store_loader
    with caplog.at_level(logging.WARNING, logger="plugins.context_engine.ci"):
        out = e.select_context(req, budget_tokens=65536)
    assert out is not None
    assert _T3_LABEL not in _join(out)
    assert out[-1] is req[-1]
    assert estimate_messages_tokens_rough(out) <= e._hard_budget_for(65536)


# ---- T4: selective historical retrieval ---------------------------------------------
def test_t4_fires_on_recall_intent_and_labels_content():
    req = [_msg("system", content="S"), _msg("user", content="CURRENT what did we do earlier about the refund policy")]
    e = _engine(
        history_retrieval_enabled=True,
        _session_search_fn=lambda **k: _discovery_payload("the refund policy was updated for repeat merchants"),
    )
    out = e.select_context(req, budget_tokens=262144)
    assert out is not None
    joined = _join(out)
    assert _T4_LABEL in joined
    assert "[historical recall]" in joined
    assert _AUTHORITY_NOTE in joined
    assert "the refund policy was updated for repeat merchants" in joined
    assert out[-1] is req[-1]


def test_no_historical_need_so_no_search():
    req = [_msg("system", content="S"), _msg("user", content="CURRENT hello, how are you today")]
    fn, calls = _recording_search()
    e = _engine(history_retrieval_enabled=True, _session_search_fn=fn, context_length=262144)
    out = e.select_context(req, budget_tokens=262144)
    assert out is req
    assert calls == []


def test_history_retrieval_off_default_never_searches():
    req = [_msg("system", content="S"), _msg("user", content="CURRENT what did we do earlier about taxes")]
    fn, calls = _recording_search()
    e = _engine(history_retrieval_enabled=False, _session_search_fn=fn)
    out = e.select_context(req, budget_tokens=262144)
    assert out is req
    assert calls == []


def test_massive_historical_retrieval_is_bounded():
    req = [_msg("system", content="S"), _msg("user", content="CURRENT what did we do earlier about the refund policy")]
    big_r = "refund policy notes " + "R" * 50000
    big_c = "refund policy notes " + "C" * 50000
    e = _engine(
        context_length=65536,
        history_retrieval_enabled=True,
        _session_search_fn=lambda **k: _discovery_payload(big_r, big_c),
    )
    out = e.select_context(req, budget_tokens=65536)
    assert out is not None
    joined = _join(out)
    assert _T4_LABEL in joined
    assert _CLAMP_MARKER in joined
    assert "R" * 50000 not in joined
    assert "C" * 50000 not in joined
    assert _count_labels(out, _T4_LABEL) <= 12
    assert estimate_messages_tokens_rough(out) <= e._hard_budget_for(65536)


def test_irrelevant_history_excluded_by_relevance_floor():
    req = [_msg("system", content="S"), _msg("user", content="CURRENT what did we say earlier about the refund policy")]
    e = _engine(
        history_retrieval_enabled=True,
        history_retrieval_min_relevance=2,
        _session_search_fn=lambda **k: _discovery_payload(
            "the refund policy was updated for repeat merchants",
            "tok_1_alpha_word tok_1_beta_word tok_1_gamma_word",
            "tok_2_alpha_word tok_2_beta_word tok_2_gamma_word",
            "tok_3_alpha_word tok_3_beta_word tok_3_gamma_word",
        ),
    )
    out = e.select_context(req, budget_tokens=262144)
    assert out is not None
    assert _count_labels(out, _T4_LABEL) == 1
    assert "the refund policy was updated for repeat merchants" in _join(out)


def test_duplicate_history_deduped_and_not_amplified():
    req = [
        _msg("system", content="S"),
        _msg("user", content="old question about the q topic"),
        _msg("assistant", content="already in the window q old text"),
        _msg("user", content="CURRENT what did we do earlier about the q topic"),
    ]
    e = _engine(
        history_retrieval_enabled=True,
        _session_search_fn=lambda **k: {
            "success": True,
            "results": [
                {
                    "session_id": "s1",
                    "messages": [
                        {"id": 1, "role": "user", "content": "off-context unique earlier q topic facts about zeta"},
                        {"id": 2, "role": "user", "content": "already in the window q old text"},
                    ],
                },
                {
                    "session_id": "s1",
                    "messages": [{"id": 3, "role": "user", "content": "off-context unique earlier q topic facts about zeta"}],
                },
            ],
        },
    )
    out = e.select_context(req, budget_tokens=262144)
    assert out is not None
    joined = _join(out)
    assert _count_labels(out, _T4_LABEL) == 1
    assert joined.count("off-context unique earlier q topic facts about zeta") == 1
    assert _count_labels(out, _T3_LABEL) == 0


def test_session_search_failure_skips_history_and_proceeds(caplog):
    req = [_msg("system", content="S"), _msg("user", content="CURRENT what did we do earlier about refunds")]

    def _raise(**k):
        raise RuntimeError("fts unavailable")

    e = _engine(history_retrieval_enabled=True, _session_search_fn=_raise)
    with caplog.at_level(logging.WARNING, logger="plugins.context_engine.ci"):
        out = e.select_context(req, budget_tokens=262144)
    assert out is not None
    assert _T4_LABEL not in _join(out)
    assert out[-1] is req[-1]
    assert estimate_messages_tokens_rough(out) <= e._hard_budget_for(262144)


def test_session_search_non_success_payload_skipped():
    req = [_msg("system", content="S"), _msg("user", content="CURRENT what did we do earlier about refunds")]
    e = _engine(
        history_retrieval_enabled=True,
        _session_search_fn=lambda **k: {"success": False, "error": "search failed"},
    )
    out = e.select_context(req, budget_tokens=262144)
    assert out is not None
    assert _T4_LABEL not in _join(out)


def test_retrieval_amplification_blocked_at_one_search_per_request():
    req = [_msg("system", content="S"), _msg("user", content="CURRENT what did we do earlier about the refund policy")]
    fn, calls = _recording_search(_discovery_payload("refund policy notes", "refund history details"))
    e = _engine(history_retrieval_enabled=True, _session_search_fn=fn)
    out = e.select_context(req, budget_tokens=262144)
    assert out is not None
    assert len(calls) == 1
    assert calls[0]["limit"] == 3
    assert calls[0]["detail"] == "adaptive"
    # A subsequent plain request must not trigger another search.
    plain = [_msg("system", content="S"), _msg("user", content="CURRENT hello there")]
    e.select_context(plain, budget_tokens=262144)
    assert len(calls) == 1


def test_tool_payload_bloat_does_not_inflate_selection():
    req = [_msg("system", content="S")]
    for r in range(4):
        req.append(_msg("user", content=_tokenish_text(r, 8000)))
        req += _tool_turn(f"old{r}", result_text="B" * 30000)
    req.append(_msg("user", content="CURRENT what did we do earlier about the refund policy"))
    req += _tool_turn("pending1", result_text="P" * 30000)
    e = _engine(
        context_length=65536,
        history_retrieval_enabled=True,
        _session_search_fn=lambda **k: _discovery_payload(
            "refund policy notes " + "H" * 200000, "refund policy notes " + "J" * 200000
        ),
    )
    out = e.select_context(req, budget_tokens=65536)
    assert out is not None
    joined = _join(out)
    assert "P" * 30000 in joined, "pending tool result must never be clamped or trimmed"
    _assert_pair_valid(out)
    assert out[-1]["content"] in ("P" * 30000, "second result " + ("P" * 30000)[-2000:])
    assert estimate_messages_tokens_rough(out) <= e._hard_budget_for(65536)


# ---- priority + authority semantics -------------------------------------------------
def test_conflicting_memory_vs_knowledge_knowledge_wins_on_order_and_label():
    system = _msg("system", content="KNOWLEDGE: the tax rate is 10%.")
    current = _msg("user", content="CURRENT apply the regional tax rate please")
    req = [system, current]
    store = _fake_store(memory_entries=["The regional tax rate is 25%."])
    out = _engine(store=store).select_context(req, budget_tokens=262144)
    assert out is not None
    idx_sys = out.index(system)
    assert idx_sys == 0
    mem_idx = next(i for i, m in enumerate(out) if _T3_LABEL in str(m.get("content")))
    cur_idx = out.index(current)
    assert idx_sys < mem_idx < cur_idx
    assert mem_idx == cur_idx - 1
    assert _AUTHORITY_NOTE in str(out[mem_idx].get("content"))
    assert out[-1] is current


def test_t3_has_priority_over_t4_when_budget_tight():
    req = [
        _msg("system", content="S"),
        _msg("user", content="CURRENT what did we decide earlier about the refund policy please"),
    ]
    store = _fake_store(memory_entries=["refund decision from memory"])
    e = _engine(
        store=store,
        context_length=8192,
        supplement_max_ratio=0.08,
        history_retrieval_enabled=True,
        _session_search_fn=lambda **k: _discovery_payload(
            "refund policy details " + " word " * 2000
        ),
    )
    out = e.select_context(req, budget_tokens=8192)
    assert out is not None
    joined = _join(out)
    assert _T3_LABEL in joined
    assert _T4_LABEL not in joined


def test_t3_before_t4_before_current_when_room_for_both():
    req = [_msg("system", content="S"), _msg("user", content="CURRENT what did we do earlier about the refund policy")]
    store = _fake_store(memory_entries=["refund decision from memory"])
    e = _engine(
        store=store,
        history_retrieval_enabled=True,
        _session_search_fn=lambda **k: _discovery_payload("the refund policy was updated for merchants"),
    )
    out = e.select_context(req, budget_tokens=262144)
    assert out is not None
    t3_idx = next(i for i, m in enumerate(out) if _T3_LABEL in str(m.get("content")))
    t4_idx = next(i for i, m in enumerate(out) if _T4_LABEL in str(m.get("content")))
    cur_idx = out.index(req[-1])
    assert t3_idx < t4_idx < cur_idx
    terms = [x for x in (str(m.get("content")) for m in out) if "refund" in x]
    assert "refund decision from memory" in _join(out)
    assert "the refund policy was updated for merchants" in _join(out)


# ---- context engine failure fails open ----------------------------------------------
def test_context_engine_failure_fails_open():
    class _Exploding(ContextIntelligenceEngine):
        def _select(self, request_messages, hard_budget):  # type: ignore[override]
            raise RuntimeError("boom")

    e = _Exploding()
    e.context_length = 262144
    assert e.select_context([_msg("user", content="hi")], budget_tokens=262144) is None


# ---- budget re-estimation (second stage) --------------------------------------------
def test_budget_reestimation_keeps_final_within_canonical_budget():
    req = [_msg("system", content="S"), _msg("user", content="CURRENT summarize")]
    store = _fake_store(memory_entries=[("Z" * 2400) for _ in range(6)])
    e = _engine(store=store, context_length=8192, supplement_max_ratio=1.0)
    out = e.select_context(req, budget_tokens=8192)
    assert out is not None
    assert estimate_messages_tokens_rough(out) <= e._hard_budget_for(8192)
    assert out[-1] is req[-1]


def test_second_stage_serialized_reduction_drops_supplements(monkeypatch):
    """Estimated tokens differ from serialized prompt size: the second stage trims
    supplements until the serialized estimate fits, never disturbing the current task."""
    filler = "alpha project debt restructuring "
    big_a = "refund alpha project " + "A" * 200000
    big_b = "refund alpha project " + "B" * 200000
    big_c = "refund alpha project " + "C" * 200000
    req = [
        _msg("system", content="S"),
        _msg("user", content="CURRENT what did we discuss earlier about the alpha project" + filler),
    ]

    def stub(messages):  # artificially small canonical estimate for every list
        return sum(30 if str(m.get("content") or "").startswith(_AUTHORITY_NOTE) else 5 for m in messages)

    monkeypatch.setattr("plugins.context_engine.ci._current_request_estimator", lambda: stub)

    e = _engine(
        context_length=262144,
        memory_integration_enabled=False,
        history_retrieval_enabled=True,
        history_retrieval_max_chars_per_message=400000,
        _session_search_fn=lambda **k: _discovery_payload(big_a, big_b, big_c),
    )
    out = e.select_context(req, budget_tokens=262144)
    assert out is not None
    hard_budget = e._hard_budget_for(262144)
    from plugins.context_engine.ci import _serialized_estimate

    assert _serialized_estimate(out) <= hard_budget
    kept = _count_labels(out, _T4_LABEL)
    assert kept >= 1, "reduction must keep as many supplements as fit"
    assert kept <= 2, "reduction must drop the supplements that overflow the serialized budget"
    assert out[-1] is req[-1]


# ---- realistic loader path (real on-disk MemoryStore) -------------------------------
def test_t3_real_on_disk_store_loader(monkeypatch, tmp_path):
    from tools import memory_tool

    (tmp_path / "MEMORY.md").write_text("coffee contract renews in May\n§\nJakarta-based user", encoding="utf-8")
    monkeypatch.setattr(memory_tool, "get_memory_dir", lambda: tmp_path)
    monkeypatch.setattr(memory_tool, "get_builtin_memory_config", lambda *a, **k: {})
    monkeypatch.setattr(memory_tool, "get_builtin_memory_store_flags", lambda *a, **k: (True, True))

    req = [_msg("system", content="S"), _msg("user", content="CURRENT small request")]
    e = _engine()
    e._memory_store_loader = None  # exercise the real loader
    out = e.select_context(req, budget_tokens=262144)
    assert out is not None
    joined = _join(out)
    assert _T3_LABEL in joined
    assert "coffee contract renews in May" in joined
    assert "Jakarta-based user" in joined


def test_t3_real_loader_respects_disabled_memory_flags(monkeypatch, tmp_path):
    from tools import memory_tool

    (tmp_path / "MEMORY.md").write_text("coffee contract renews in May", encoding="utf-8")
    monkeypatch.setattr(memory_tool, "get_memory_dir", lambda: tmp_path)
    monkeypatch.setattr(memory_tool, "get_builtin_memory_config", lambda *a, **k: {})
    monkeypatch.setattr(memory_tool, "get_builtin_memory_store_flags", lambda *a, **k: (False, False))

    req = [_msg("system", content="S"), _msg("user", content="CURRENT small request")]
    e = _engine()
    e._memory_store_loader = None  # exercise the real loader
    out = e.select_context(req, budget_tokens=262144)
    # The real store reports both targets disabled, so T3 must be skipped entirely.
    assert out is req
    assert _T3_LABEL not in _join(out)


# ---- determinism / recall intent ----------------------------------------------------
def test_recall_intent_detection():
    from plugins.context_engine.ci import _RECALL_INTENT_RE

    assert _RECALL_INTENT_RE.search("what did we decide earlier?")
    assert _RECALL_INTENT_RE.search("remember the alpha server")
    assert not _RECALL_INTENT_RE.search("how are you")


def test_recall_query_or_syntax_and_content_words():
    from plugins.context_engine.ci import _recall_query

    q = _recall_query("What did we discuss about Refund Policy?!", 4)
    tokens = q.split(" OR ")
    assert tokens and all(tok.isalnum() for tok in tokens)
    assert "policy" in tokens


# ---- telemetry ----------------------------------------------------------------------
def test_telemetry_fields_no_content(caplog):
    req = [_msg("system", content="S"), _msg("user", content="CURRENT small request")]
    store = _fake_store(memory_entries=["some memory note"])
    with caplog.at_level(logging.INFO, logger="plugins.context_engine.ci"):
        _engine(store=store).select_context(req, budget_tokens=262144)
    records = [r for r in caplog.records if "ci context intelligence" in r.getMessage()]
    assert records
    for field in (
        "memory_messages",
        "history_messages",
        "supplement_messages",
        "dropped_supplements",
        "base_tokens_est",
        "selected_tokens_est",
        "hard_budget",
    ):
        assert f"'{field}'" in records[0].getMessage()
    assert "some memory note" not in records[0].getMessage()


def test_status_reports_intelligence_flags():
    e = _engine(history_retrieval_enabled=True)
    status = e.get_status()
    assert status.get("engine") == "ci"
    assert status.get("intelligence_enabled") is True
    assert status.get("memory_integration_enabled") is True
    assert status.get("history_retrieval_enabled") is True


# ---- host pipeline integration ------------------------------------------------------
def test_host_pipeline_integration_ci_engine():
    from types import SimpleNamespace

    from agent.conversation_loop import _apply_context_engine_selection

    engine = load_context_engine("ci")
    assert engine is not None
    engine.update_model("test/ci", 262144)
    engine._memory_store_loader = lambda: _fake_store()
    engine.history_retrieval_enabled = False
    agent = SimpleNamespace(context_compressor=engine, session_id="ci-int-test")

    api_messages = [
        _msg("system", content="S"),
        _msg("user", content="first"),
        _msg("assistant", content="reply"),
        _msg("user", content="CURRENT_REQUEST_SENTINEL analyze"),
    ]
    conversation = copy.deepcopy(api_messages)
    incoming = api_messages[-1]
    sel = _apply_context_engine_selection(agent, api_messages, conversation, incoming, logger=logging.getLogger("test"))
    assert sel is not None
    assert sel[-1]["content"] == api_messages[-1]["content"]
    _assert_pair_valid(sel)
    assert estimate_messages_tokens_rough(sel) <= engine._hard_budget_for(262144)
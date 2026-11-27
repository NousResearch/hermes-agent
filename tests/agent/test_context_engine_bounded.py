"""Tests for the ``bounded`` plugin context engine (plugins/context_engine/bounded).

Covers the ANCOS Phase-1 validation matrix: hard-budget selection, request
preservation (system / current request / newest pending tool chain), tool-pair
validity, replay-safe ordering, no mutation of persisted history or of the original
request list, deterministic behavior, fail-open semantics, and a 160k-token
regression where the compressor's own compaction threshold has NOT yet fired.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List

import pytest

from agent.model_metadata import estimate_messages_tokens_rough
from plugins.context_engine import discover_context_engines, load_context_engine
from plugins.context_engine.bounded import (
    BoundedContextEngine,
    _CLAMP_MARKER,
    _current_request_index,
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
        _msg(
            "tool",
            content="second result " + result_text[-2000:],
            tool_call_id=call_id,
            name="web_search",
        ),
    ]


def _build_big_request(
    rounds: int = 12, words: int = 12000, system: str = "You are a helpful assistant."
) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = [_msg("system", content=system)]
    for r in range(rounds):
        msgs.append(_msg("user", content=_tokenish_text(r, words)))
        if r % 3 == 0:
            msgs += _tool_turn(f"call{r}", result_text=_tokenish_text(r + 100, words))
        msgs.append(_msg("assistant", content=_tokenish_text(r + 200, 120)))
    msgs.append(
        _msg("user", content="CURRENT_REQUEST_SENTINEL " + _tokenish_text(999, 400))
    )
    return msgs


def _build_request_near(target_tokens: int, rounds: int = 12):
    """Deterministically build a request whose rough estimate lands near ``target_tokens``."""
    words = max(200, int(target_tokens / max(rounds, 1) / 5.6))
    req = None
    est = 0
    for _ in range(10):
        req = _build_big_request(rounds=rounds, words=words)
        est = estimate_messages_tokens_rough(req)
        if est < target_tokens * 0.90:
            words = max(words + 1, int(words * target_tokens / max(est, 1) * 1.15))
        elif est > target_tokens * 1.10:
            words = max(1, int(words * target_tokens / max(est, 1) * 0.92))
        else:
            break
    return req, est


def _engine(context_length: int = 262144, **kwargs: Any) -> BoundedContextEngine:
    e = BoundedContextEngine(**kwargs)
    e.context_length = context_length
    return e


def _assert_pair_valid(selected: List[Dict[str, Any]]) -> None:
    created_ids = set()
    for m in selected:
        for tc in m.get("tool_calls") or []:
            if isinstance(tc, dict) and tc.get("id"):
                created_ids.add(tc["id"])
    for m in selected:
        if m.get("role") == "tool":
            assert m.get("tool_call_id") in created_ids, "dangling tool result retained"


def _assert_replay_order(
    selected: List[Dict[str, Any]], original: List[Dict[str, Any]]
) -> None:
    positions = {id(m): i for i, m in enumerate(original)}
    last = -1
    for m in selected:
        pos = positions.get(id(m), last)
        assert pos >= last, "selected messages are not in original order"
        last = pos


# ---- identity / discovery ------------------------------------------------------------
def test_name_and_zero_arg_instantiation():
    e = BoundedContextEngine()
    assert e.name == "bounded"
    assert isinstance(e, BoundedContextEngine)


def test_plugin_discovery_includes_bounded():
    assert "bounded" in {name for name, _, _ in discover_context_engines()}


def test_load_context_engine_by_name():
    e = load_context_engine("bounded")
    assert e is not None and e.name == "bounded"


def test_register_entrypoint_captures_engine():
    class Collector:
        def __init__(self):
            self.engine = None

        def register_context_engine(self, engine):
            self.engine = engine

    from plugins.context_engine.bounded import register

    collector = Collector()
    register(collector)
    assert collector.engine is not None and collector.engine.name == "bounded"


# ---- fail-open + budget plumbing ----------------------------------------------------
def test_select_context_none_when_unknown_budget():
    e = _engine(context_length=0)
    assert e.select_context([_msg("user", content="hi")]) is None


def test_select_context_none_when_disabled():
    e = _engine(enabled=False)
    assert e.select_context([_msg("user", content="hi")], budget_tokens=262144) is None


@pytest.mark.parametrize("bad", ["nope", [], None, 42])
def test_select_context_fail_open_on_invalid_input(bad):
    e = _engine()
    assert e.select_context(bad, budget_tokens=262144) is None


def test_select_context_short_path_returns_unchanged_list():
    req = [_msg("system", content="S"), _msg("user", content="short")]
    e = _engine()
    out = e.select_context(req, budget_tokens=262144)
    assert out is req


def test_hard_budget_formula():
    e = _engine()
    assert e._hard_budget_for(262144) == int(262144 * 0.5) - min(
        int(262144 * 0.2), min(4096, 26214) + min(8192, 26214)
    )


# ---- preservation guarantees --------------------------------------------------------
def test_system_and_current_request_preserved():
    e = _engine()
    req = _build_big_request()
    out = e.select_context(req, budget_tokens=262144)
    assert out is not None
    assert out[0]["role"] == "system"
    assert "CURRENT_REQUEST_SENTINEL" in str(out[-1].get("content"))


def test_newest_pending_tool_chain_preserved():
    e = _engine()
    req = [_msg("system", content="S")]
    for r in range(6):
        req += _tool_turn(f"old{r}", result_text=_tokenish_text(r, 9000))
    req.append(_msg("user", content="CURRENT_REQUEST_SENTINEL live"))
    req += _tool_turn("live1", result_text="live result")
    out = e.select_context(req, budget_tokens=262144)
    assert out is not None
    joined = "\n".join(m.get("content", "") for m in out)
    assert "live1" in str([tc for m in out for tc in (m.get("tool_calls") or [])])
    assert "live result" in joined


def test_pair_validity_of_selection():
    req = _build_big_request()
    out = _engine().select_context(req, budget_tokens=262144)
    assert out is not None
    _assert_pair_valid(out)


def test_replay_order_is_subsequence_and_system_first():
    req = _build_big_request()
    out = _engine().select_context(req, budget_tokens=262144)
    assert out is not None
    assert out[0]["role"] == "system"
    _assert_replay_order(out, req)


# ---- boundedness ---------------------------------------------------------------------
def test_selection_fits_hard_budget_256k():
    e = _engine(context_length=262144)
    req = _build_big_request()
    out = e.select_context(req, budget_tokens=262144)
    assert out is not None
    assert estimate_messages_tokens_rough(out) <= e._hard_budget_for(262144)


def test_selection_fits_hard_budget_64k():
    e = _engine(context_length=65536)
    req, est_in = _build_request_near(60000, rounds=6)
    assert est_in > e._hard_budget_for(65536)
    out = e.select_context(req, budget_tokens=65536)
    assert out is not None
    assert estimate_messages_tokens_rough(out) <= e._hard_budget_for(65536)


def test_omits_excess_history():
    req = _build_big_request()
    out = _engine().select_context(req, budget_tokens=262144)
    assert out is not None
    assert len(out) < len(req)


def test_huge_early_message_does_not_fail_open():
    e = _engine(context_length=65536)
    req = [
        _msg("system", content="S"),
        _msg("user", content=_tokenish_text(1, 100000)),
        _msg("user", content="CURRENT_REQUEST_SENTINEL small request"),
    ]
    out = e.select_context(req, budget_tokens=65536)
    assert out is not None
    assert estimate_messages_tokens_rough(out) <= e._hard_budget_for(65536)
    joined = "\n".join(str(m.get("content")) for m in out)
    assert "CURRENT_REQUEST_SENTINEL" in joined


def test_never_returns_empty():
    for ctx in (65536, 262144, 524288):
        req = _build_big_request()
        out = _engine(context_length=ctx).select_context(req, budget_tokens=ctx)
        assert out is None or len(out) > 0


# ---- 160k regression: compressor threshold NOT yet triggered -------------------------
def test_160k_regression_compression_not_yet_triggered_but_selection_bounds():
    e = BoundedContextEngine()
    e.update_model("test/bounded", 262144)
    req, est_in = _build_request_near(160000)
    hard_budget = e._hard_budget_for(262144)
    assert 130000 <= est_in <= 190000, f"build calibration off: {est_in}"
    assert est_in > hard_budget, "transcript must exceed the selection budget"
    # Compressor's own threshold (~75% of a <512k window = 196608) has not fired on
    # this transcript, so nothing would shrink it before the request went out.
    assert e.should_compress(est_in) is False
    assert e.should_compress_info(est_in)[0] is False
    out = e.select_context(req, budget_tokens=262144)
    assert out is not None
    assert estimate_messages_tokens_rough(out) <= hard_budget


# ---- determinism / no mutation ------------------------------------------------------
def test_deterministic_selection():
    req = _build_big_request()
    e = _engine()
    a = e.select_context(req, budget_tokens=262144)
    b = e.select_context(req, budget_tokens=262144)
    assert a is not None and b is not None
    assert a == b


def test_request_messages_not_mutated():
    req = _build_big_request()
    snapshot = copy.deepcopy(req)
    _engine().select_context(req, budget_tokens=262144)
    assert req == snapshot


def test_conversation_messages_not_consulted_or_mutated():
    req = _build_big_request()
    conv = copy.deepcopy(req)
    snapshot = copy.deepcopy(conv)
    out = _engine().select_context(
        req, conversation_messages=conv, budget_tokens=262144
    )
    assert out is not None
    assert conv == snapshot


def test_clamped_tool_result_content_not_altered_in_original():
    req = [_msg("system", content="S"), _msg("user", content="first")]
    for r in range(3):
        req.append(
            _msg(
                "assistant",
                content="calling a tool",
                tool_calls=[
                    {
                        "id": f"c{r}",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": "{}"},
                    }
                ],
            )
        )
        req.append(
            _msg("tool", content="B" * 30000, tool_call_id=f"c{r}", name="web_search")
        )
    req.append(_msg("user", content="CURRENT_REQUEST_SENTINEL current"))
    snapshot = copy.deepcopy(req)
    e = _engine(context_length=65536)
    asserted_budget = e._hard_budget_for(65536)
    assert estimate_messages_tokens_rough(req) > asserted_budget
    out = e.select_context(req, budget_tokens=65536)
    assert out is not None
    joined = "\n".join(str(m.get("content")) for m in out)
    assert _CLAMP_MARKER in joined, (
        "oversized old tool result should be clamped in the selection"
    )
    assert estimate_messages_tokens_rough(out) <= asserted_budget
    assert req == snapshot, "original request list must never be modified"


# ---- lifecycle delegation -----------------------------------------------------------
def test_update_model_sets_context_and_pins_compressor():
    e = BoundedContextEngine()
    e.update_model("test/bounded", 262144)
    assert e.context_length == 262144
    assert e.threshold_tokens == int(262144 * e.threshold_percent)
    assert e._inner() is not None


def test_model_thresholds_attr_mirrored_to_inner_compressor():
    e = BoundedContextEngine()
    e.update_model("test/bounded", 262144)
    e.model_thresholds = {"test/bounded": 0.9}
    inner = e._inner()
    assert inner is not None
    assert inner.model_thresholds == {"test/bounded": 0.9}


def test_update_from_response_fallback_without_compressor():
    e = _engine()
    e.update_from_response({"prompt_tokens": 5, "completion_tokens": 3})
    assert e.last_total_tokens == 8


def test_delegated_lifecycle_methods():
    e = BoundedContextEngine()
    e.update_model("test/bounded", 262144)
    e.on_session_start("sess-1")
    assert e.should_compress(1234) in (True, False)
    msgs = [_msg("system", content="S"), _msg("user", content="hello")]
    compressed = e.compress(msgs, force=True)
    assert isinstance(compressed, list)
    assert isinstance(e.prune_tool_results_only(msgs)[0], list)
    status = e.get_status()
    assert status.get("engine") == "bounded"
    assert status.get("selection_enabled") is True
    e.on_session_reset()
    e.on_session_end("sess-1", msgs)


# ---- telemetry (never content) ------------------------------------------------------
def test_selection_telemetry_fields_no_content(caplog):
    req = _build_big_request()
    with caplog.at_level(logging.INFO, logger="plugins.context_engine.bounded"):
        _engine().select_context(req, budget_tokens=262144)
    records = [r for r in caplog.records if "bounded selection" in r.getMessage()]
    assert records
    for field in (
        "mode",
        "input_messages",
        "output_messages",
        "input_tokens_est",
        "selected_tokens_est",
        "budget_tokens",
        "hard_budget",
        "omitted_messages",
        "pruned_tool_results",
        "clamped_tool_results",
    ):
        assert f"'{field}'" in records[0].getMessage()
    assert "CURRENT_REQUEST_SENTINEL" not in records[0].getMessage()


# ---- helper: multimodal user message -----------------------------------------------
def _multimodal_user(text: str) -> Dict[str, Any]:
    return _msg(
        "user",
        content=[
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
        ],
    )


def _build_turn_history(
    current: Dict[str, Any], rounds: int = 8, words: int = 12000
) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = [
        _msg("system", content="You are a helpful assistant.")
    ]
    for r in range(rounds):
        msgs.append(_msg("user", content=_tokenish_text(r, words)))
        msgs.append(_msg("assistant", content=_tokenish_text(r + 200, 120)))
    msgs.append(current)
    return msgs


# ---- F1 regression: the trailing user request must never be dropped ----------------
def test_f1_current_request_index_prefers_trailing_list_content_user():
    req = [
        _msg("system", content="S"),
        _msg("user", content="A"),
        _multimodal_user("B multimodal"),
    ]
    assert _current_request_index(req) == 2


def test_f1_multiturn_text_current_request_remains_current():
    e = _engine(context_length=262144)
    current = _msg("user", content="CURRENT_REQUEST_SENTINEL final text request")
    req = _build_turn_history(current, rounds=8, words=12000)
    out = e.select_context(req, budget_tokens=262144)
    assert out is not None
    assert len(out) < len(req)
    assert out[-1] is current
    assert out[-1]["content"] == "CURRENT_REQUEST_SENTINEL final text request"


def test_f1_multiturn_multimodal_tail_remains_current_request():
    e = _engine(context_length=262144)
    current = _multimodal_user("CURRENT_REQUEST_SENTINEL analyze this screenshot")
    req = _build_turn_history(current, rounds=8, words=12000)
    out = e.select_context(req, budget_tokens=262144)
    assert out is not None
    assert len(out) < len(req)
    assert out[-1] is current
    assert "CURRENT_REQUEST_SENTINEL analyze this screenshot" in str(
        out[-1].get("content")
    )


def test_f1_single_multimodal_request_is_current():
    e = _engine(context_length=262144)
    req = [_msg("system", content="S"), _multimodal_user("a single image request")]
    out = e.select_context(req, budget_tokens=262144)
    assert out is req
    assert out[-1]["content"][0]["text"] == "a single image request"


def test_f1_text_moa_aggregated_message_remains_current():
    e = _engine(context_length=262144)
    current = _msg(
        "user",
        content=(
            "CURRENT_REQUEST_SENTINEL user prompt\n\n"
            "[MoA aggregated context from reference models]"
        ),
    )
    req = _build_turn_history(current, rounds=8, words=12000)
    out = e.select_context(req, budget_tokens=262144)
    assert out is not None
    assert out[-1] is current
    assert "CURRENT_REQUEST_SENTINEL user prompt" in out[-1]["content"]


def test_f1_multimodal_tail_not_replaced_by_previous_user():
    e = _engine(context_length=262144)
    req = _build_turn_history(
        _msg("user", content="previous plain request"), rounds=8, words=12000
    )
    current = _multimodal_user("CURRENT_REQUEST_SENTINEL actual trailing image request")
    req.append(current)
    out = e.select_context(req, budget_tokens=262144)
    assert out is not None
    assert out[-1] is current
    assert "CURRENT_REQUEST_SENTINEL actual trailing image request" in str(
        out[-1].get("content")
    )
    assert "previous plain request" not in out[-1]["content"]


def test_f1_current_request_present_in_final_selection():
    e = _engine(context_length=262144)
    current = _msg("user", content="CURRENT_REQUEST_SENTINEL visible final request")
    req = _build_turn_history(current, rounds=12, words=16000)
    out = e.select_context(req, budget_tokens=262144)
    assert out is not None
    assert len(out) > 1
    assert current in out, "current request message identity must survive selection"
    assert out[-1] is current


# ---- R2 regression: wrapper counter / preflight-usage mirroring --------------------
def test_r2_inner_usage_mirrors_to_wrapper_counters():
    e = BoundedContextEngine()
    e.update_model("test/bounded", 262144)
    e.update_from_response({
        "prompt_tokens": 12345,
        "completion_tokens": 200,
        "total_tokens": 12545,
    })
    inner = e._inner()
    assert inner is not None
    assert e.last_prompt_tokens == inner.last_prompt_tokens == 12345
    assert e.last_completion_tokens == inner.last_completion_tokens == 200
    assert e.last_total_tokens == inner.last_total_tokens == 12545


def test_r2_wrapper_total_derived_when_provider_omits_total():
    e = BoundedContextEngine()
    e.update_model("test/bounded", 262144)
    e.update_from_response({"prompt_tokens": 500, "completion_tokens": 50})
    assert e.last_prompt_tokens == 500
    assert e.last_total_tokens == 550


def test_r2_host_facing_preflight_sees_real_usage():
    e = BoundedContextEngine()
    e.update_model("test/bounded", 262144)
    e.update_from_response({
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "total_tokens": 1100,
    })
    # Host preflight reads the engine's own last_prompt_tokens directly; a fit result
    # below the threshold must not trigger a compression pass.
    assert e.last_prompt_tokens == 1000
    assert 1000 < e.threshold_tokens
    assert e.should_compress(e.last_prompt_tokens) is False


def test_r2_should_defer_preflight_forwards_to_inner():
    e = BoundedContextEngine()
    e.update_model("test/bounded", 262144)
    inner = e._inner()
    assert inner is not None
    rough = e.threshold_tokens * 2
    assert e.should_defer_preflight_to_real_usage(
        rough
    ) == inner.should_defer_preflight_to_real_usage(rough)


def test_r2_wrapper_defer_not_shadowed_by_base_false():
    e = BoundedContextEngine()
    e.update_model("test/bounded", 262144)
    inner = e._inner()
    assert inner is not None
    inner.awaiting_real_usage_after_compression = True
    assert inner.last_real_prompt_tokens == 0
    assert e.should_defer_preflight_to_real_usage(e.threshold_tokens + 1) is True


def test_r2_no_extra_pass_from_zero_wrapper_counter():
    e = BoundedContextEngine()
    e.update_model("test/bounded", 262144)
    e.update_from_response({"prompt_tokens": 800, "completion_tokens": 50})
    assert e.last_prompt_tokens == 800
    assert e.should_compress(e.last_prompt_tokens) is False


def test_r2_inner_compressor_semantics_preserved():
    e = BoundedContextEngine()
    e.update_model("test/bounded", 262144)
    inner = e._inner()
    assert inner is not None
    e.update_from_response({
        "prompt_tokens": 300,
        "completion_tokens": 20,
        "total_tokens": 320,
    })
    assert inner.last_prompt_tokens == 300
    assert inner.last_completion_tokens == 20
    assert inner.awaiting_real_usage_after_compression is False
    assert e.should_compress_info(300)[0] is False


# ---- backfill: multi-tool-call batches ---------------------------------------------
def test_backfill_multi_tool_call_batch_selection_pair_valid():
    e = _engine(context_length=65536)
    req = [_msg("system", content="S")]
    for r in range(4):
        req.append(_msg("user", content=_tokenish_text(r, 8000)))
        req.append(
            _msg(
                "assistant",
                content="calling two tools",
                tool_calls=[
                    {
                        "id": f"c{r}a",
                        "type": "function",
                        "function": {"name": "a", "arguments": "{}"},
                    },
                    {
                        "id": f"c{r}b",
                        "type": "function",
                        "function": {"name": "b", "arguments": "{}"},
                    },
                ],
            )
        )
        req.append(_msg("tool", content="B" * 20000, tool_call_id=f"c{r}a", name="a"))
        req.append(_msg("tool", content="C" * 20000, tool_call_id=f"c{r}b", name="b"))
    req.append(_msg("user", content="CURRENT_REQUEST_SENTINEL final"))
    out = e.select_context(req, budget_tokens=65536)
    assert out is not None
    _assert_pair_valid(out)
    _assert_replay_order(out, req)
    assert out[-1]["content"] == "CURRENT_REQUEST_SENTINEL final"
    assert estimate_messages_tokens_rough(out) <= e._hard_budget_for(65536)


# ---- backfill: system-only requests ------------------------------------------------
def test_backfill_system_only_request_unchanged():
    e = _engine()
    system = _msg("system", content="You are a helpful assistant.")
    out = e.select_context([system], budget_tokens=262144)
    assert out is not None
    assert out == [system]
    assert out[0]["role"] == "system"


def test_backfill_oversized_system_only_fails_open():
    e = _engine(context_length=65536)
    system = _msg("system", content=_tokenish_text(7, 60000))
    assert estimate_messages_tokens_rough([system]) > e._hard_budget_for(65536)
    assert e.select_context([system], budget_tokens=65536) is None


# ---- backfill: pending tool chain clamp protection ---------------------------------
def test_backfill_pending_tool_chain_never_clamped_or_trimmed():
    e = _engine(context_length=65536)
    req = [_msg("system", content="S")]
    for r in range(4):
        req.append(_msg("user", content=_tokenish_text(r, 8000)))
        req += _tool_turn(f"old{r}", result_text="B" * 30000)
    req.append(_msg("user", content="CURRENT_REQUEST_SENTINEL current"))
    req += _tool_turn("pending1", result_text="P" * 30000)
    out = e.select_context(req, budget_tokens=65536)
    assert out is not None
    joined = "\n".join(str(m.get("content")) for m in out)
    assert "P" * 30000 in joined, "pending tool result must never be clamped or trimmed"
    _assert_pair_valid(out)
    assert out[-1]["content"] in (
        "P" * 30000,
        "second result " + ("P" * 30000)[-2000:],
    )


# ---- backfill: host _apply_context_engine_selection integration --------------------
def test_backfill_host_pipeline_integration_bounded_engine():
    from types import SimpleNamespace

    from agent.conversation_loop import _apply_context_engine_selection

    engine = load_context_engine("bounded")
    assert engine is not None
    engine.update_model("test/bounded", 262144)
    agent = SimpleNamespace(context_compressor=engine, session_id="int-test")

    api_messages = _build_turn_history(
        _multimodal_user("CURRENT_REQUEST_SENTINEL analyze"), rounds=8, words=12000
    )
    conversation = copy.deepcopy(api_messages)
    incoming = api_messages[-1]
    sel = _apply_context_engine_selection(
        agent, api_messages, conversation, incoming, logger=logging.getLogger("test")
    )
    assert sel is not None
    assert sel is not api_messages, "real selection must yield a trimmed request"
    _assert_pair_valid(sel)
    assert sel[-1]["content"] == api_messages[-1]["content"]
    assert "CURRENT_REQUEST_SENTINEL analyze" in str(sel[-1].get("content"))

"""Tests for the provider-bound tool-result projection (agent/tool_result_projection.py).

The projection replaces stale, large, recoverable tool results with stubs **on the
outbound request only** — the canonical transcript, session resume and the UI keep every
byte, and the full result is persisted to ``cache/spillover`` before the stub replaces it.
These tests pin that contract (freshness, recoverability, stability, fail-closed) rather
than any particular token count.

Mirrors the construction/patching conventions of
``tests/agent/test_proactive_tool_result_pruning.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.tool_result_projection import (
    PROJECTION_MARKER,
    build_stub,
    is_candidate,
    is_projected_tool_result,
    project_stale_tool_results,
    protected_tail_start,
    resolve_policy,
    tool_call_index,
    trigger_tokens,
)
from tools.tool_result_storage import get_spillover_dir

WINDOW = 200_000
BIG_CHARS = 20_000


def _compressor(**kw):
    defaults = dict(
        model="test",
        quiet_mode=True,
        threshold_percent=0.50,
        protect_first_n=2,
        protect_last_n=4,
        # Explicit numbers keep the tests independent of the auto-derived trigger.
        tool_result_projection="auto",
        tool_result_projection_min_tokens=8_000,
        tool_result_projection_min_reclaim_tokens=4_000,
        tool_result_projection_tail_min_tokens=2_000,
        tool_result_projection_tail_max_tokens=2_000,
        tool_result_projection_tail_messages=2,
        tool_result_projection_tail_max_messages=6,
        tool_result_projection_min_result_chars=4_000,
    )
    defaults.update(kw)
    with patch("agent.context_compressor.get_model_context_length", return_value=WINDOW):
        return ContextCompressor(**defaults)


def _agent(cc=None, *, caching=False):
    agent = SimpleNamespace(_use_prompt_caching=caching)
    agent.context_compressor = cc if cc is not None else _compressor()
    return agent


def _assistant_call(cid, name="read_file", args='{"path":"src/foo.py"}'):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": args}}],
    }


def _tool_msg(cid, content):
    return {"role": "tool", "tool_call_id": cid, "content": content}


def _build(n_pairs, big_indices, big_chars=BIG_CHARS, small="ok", tail_small=0):
    """``system`` + *n_pairs* (assistant tool_call, tool result) pairs.

    Results whose pair index is in *big_indices* carry a distinct ``big_chars`` payload.
    The last *tail_small* pairs are small (so they cannot be projected anyway).
    """
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n_pairs):
        cid = f"call_{i}"
        msgs.append(_assistant_call(cid))
        if i in big_indices and i < n_pairs - tail_small:
            msgs.append(_tool_msg(cid, chr(65 + (i % 26)) * big_chars))
        else:
            msgs.append(_tool_msg(cid, small))
    return msgs


def _content(msgs, cid):
    return [m for m in msgs if m.get("role") == "tool" and m.get("tool_call_id") == cid][0]["content"]


# ── the core win: stale payload leaves the wire, freshness stays ─────────────


def test_projects_old_large_results_and_keeps_the_recent_tail():
    msgs = _build(16, big_indices={0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
    projected = project_stale_tool_results(_agent(), msgs)

    assert projected >= 4
    for cid in ("call_0", "call_1", "call_2", "call_3"):
        content = _content(msgs, cid)
        assert is_projected_tool_result(content)
        assert cid in content                      # the model can still tell which call it was
        assert "read_file" in content
        assert len(content) < 600                  # a stub, not the payload


def test_small_results_are_never_projected():
    msgs = _build(12, big_indices=set())
    assert project_stale_tool_results(_agent(), msgs) == 0
    assert all(not is_projected_tool_result(m["content"]) for m in msgs if m.get("role") == "tool")


def test_protects_the_verbatim_tail():
    """A big result inside the protected tail keeps its bytes: the working set stays live."""
    cc = _compressor(
        tool_result_projection_tail_min_tokens=6_000,
        tool_result_projection_tail_max_tokens=6_000,
        tool_result_projection_tail_messages=2,
    )
    msgs = _build(6, big_indices={0, 1, 4, 5})          # 4 and 5 are the newest pairs
    project_stale_tool_results(_agent(cc), msgs)

    assert is_projected_tool_result(_content(msgs, "call_0"))
    assert is_projected_tool_result(_content(msgs, "call_1"))
    assert _content(msgs, "call_4") == "E" * BIG_CHARS  # untouched
    assert _content(msgs, "call_5") == "F" * BIG_CHARS  # untouched


def test_errors_multimodal_and_optout_results_keep_their_bytes():
    cc = _compressor(tool_result_projection_min_tokens=2_000, tool_result_projection_min_reclaim_tokens=1_000)
    msgs = [{"role": "system", "content": "sys"}]
    payloads = {
        "call_err": json.dumps({"error": "ENOENT: no such file", "size": "x" * BIG_CHARS}),
        "call_false": json.dumps({"success": False, "detail": "y" * BIG_CHARS}),
        "call_text": "Error: command failed\n" + "z" * BIG_CHARS,
        "call_optout": json.dumps({"projection_safe": False, "data": "w" * BIG_CHARS}),
        "call_ok": "q" * BIG_CHARS,
    }
    for cid, content in payloads.items():
        msgs.append(_assistant_call(cid))
        msgs.append(_tool_msg(cid, content))
    # Fresh tail: the mixed rows above must fall outside it to be eligible at all.
    for i in range(6):
        msgs.append(_assistant_call(f"call_tail_{i}"))
        msgs.append(_tool_msg(f"call_tail_{i}", "ok"))

    assert project_stale_tool_results(_agent(cc), msgs) == 1

    for cid in ("call_err", "call_false", "call_text", "call_optout"):
        assert _content(msgs, cid) == payloads[cid], cid
    assert is_projected_tool_result(_content(msgs, "call_ok"))


def test_multimodal_tool_result_is_untouched():
    cc = _compressor()
    content = [{"type": "text", "text": "x" * BIG_CHARS}]
    msgs = [{"role": "system", "content": "sys"}, _assistant_call("call_m"), _tool_msg("call_m", content)]
    assert project_stale_tool_results(_agent(cc), msgs) == 0
    assert _content(msgs, "call_m") is content


# ── invariant 1 + 2: canonical untouched, bytes recoverable ──────────────────


def test_canonical_transcript_is_never_modified():
    """The projection only ever runs on the per-call copy the assembly builds for the wire."""
    from agent.conversation_loop import _clone_message_for_send

    canonical = _build(12, big_indices={0, 1, 2, 3})
    snapshot = json.dumps(canonical, sort_keys=True)
    api_messages = [_clone_message_for_send(m) for m in canonical]

    assert project_stale_tool_results(_agent(), api_messages) > 0

    assert json.dumps(canonical, sort_keys=True) == snapshot   # not a byte moved
    assert is_projected_tool_result(_content(canonical, "call_0")) is False
    assert is_projected_tool_result(_content(api_messages, "call_0")) is True


def test_roles_order_and_tool_call_ids_survive_the_projection():
    msgs = _build(12, big_indices={0, 1, 2, 3})
    before = [(m.get("role"), m.get("tool_call_id")) for m in msgs]

    project_stale_tool_results(_agent(), msgs)

    assert [(m.get("role"), m.get("tool_call_id")) for m in msgs] == before
    # The assistant row that issued the call is untouched, so the pair stays valid.
    assert msgs[1]["tool_calls"][0]["id"] == "call_0"


def test_full_output_is_recoverable_from_the_spillover_file():
    cc = _compressor()
    msgs = _build(12, big_indices={0, 1, 2, 3})
    original = _content(msgs, "call_1")

    project_stale_tool_results(_agent(cc), msgs)

    stub = _content(msgs, "call_1")
    path = next(line for line in stub.splitlines() if "spillover" in line).rsplit(": ", 1)[1]
    assert Path(path).is_file()
    assert Path(path).read_text(encoding="utf-8") == original
    assert str(get_spillover_dir()) in path


def test_failed_persist_keeps_the_result_fail_closed():
    """No recoverable home → no stub. A stub pointing at nothing is worse than the bytes."""
    msgs = _build(12, big_indices={0, 1, 2, 3})
    with patch("tools.tool_result_storage.store_spillover_content", return_value=None):
        projected = project_stale_tool_results(_agent(), msgs)

    assert projected == 0
    assert all(not is_projected_tool_result(m["content"]) for m in msgs if m.get("role") == "tool")


def test_rows_sharing_or_missing_a_call_id_recover_their_own_bytes():
    """Two rows with the same (or no) ``tool_call_id`` must not share one spillover file: the
    stub has to point at the bytes of ITS row, not at whichever row wrote first."""
    cc = _compressor(tool_result_projection_min_tokens=2_000, tool_result_projection_min_reclaim_tokens=1_000)
    rows = [("dup", "A" * BIG_CHARS), ("dup", "B" * BIG_CHARS), ("", "C" * BIG_CHARS), ("", "D" * BIG_CHARS)]
    msgs = [{"role": "system", "content": "sys"}]
    for idx, (cid, payload) in enumerate(rows):
        msgs.append(_assistant_call(cid or f"call_{idx}"))
        msgs.append(_tool_msg(cid, payload))
    for i in range(6):  # fresh tail
        msgs.append(_assistant_call(f"call_t{i}"))
        msgs.append(_tool_msg(f"call_t{i}", "ok"))

    assert project_stale_tool_results(_agent(cc), msgs) == 4

    tool_rows = [m for m in msgs if m.get("role") == "tool" and m.get("tool_call_id") in {"dup", ""}]
    assert len(tool_rows) == 4
    seen = {}
    for row, (_cid, payload) in zip(tool_rows, rows):
        stub = row["content"]
        assert is_projected_tool_result(stub)
        path = next(line for line in stub.splitlines() if "spillover" in line).rsplit(": ", 1)[1]
        seen[path] = payload
    assert len(seen) == 4, "each row needs its own file"
    for path, payload in seen.items():
        assert Path(path).read_text(encoding="utf-8") == payload


def test_already_persisted_results_reuse_their_file_instead_of_rewriting_it():
    """A ``<persisted-output>`` row is already recoverable: the stub must point at the SAME file."""
    cc = _compressor(tool_result_projection_min_tokens=2_000, tool_result_projection_min_reclaim_tokens=1_000)
    spill = get_spillover_dir() / "call_p.txt"
    spill.parent.mkdir(parents=True, exist_ok=True)
    spill.write_text("full body", encoding="utf-8")
    body = (
        "<persisted-output>\nThis tool result was too large.\n"
        f"Full output saved to: {spill}\nPreview (first 5000 chars):\n" + "p" * BIG_CHARS + "\n</persisted-output>"
    )
    msgs = [{"role": "system", "content": "sys"}, _assistant_call("call_p"), _tool_msg("call_p", body)]
    # Fresh tail so the persisted row is outside it.
    for i in range(6):
        msgs.append(_assistant_call(f"call_t{i}"))
        msgs.append(_tool_msg(f"call_t{i}", "ok"))

    assert project_stale_tool_results(_agent(cc), msgs) == 1
    assert str(spill) in _content(msgs, "call_p")
    assert spill.read_text(encoding="utf-8") == "full body"


# ── invariant 5: stability (the prompt-cache budget) ────────────────────────


def test_reprojection_is_byte_identical_across_turns():
    """The same row must project to the same bytes every turn, or every turn is a cache break."""
    cc = _compressor()
    agent = _agent(cc)
    msgs = _build(12, big_indices={0, 1, 2, 3})
    project_stale_tool_results(agent, msgs)
    first = _content(msgs, "call_0")

    # A new turn arrives: canonical history is rebuilt into a fresh API copy.
    grown = [dict(m) for m in msgs]
    grown.append({"role": "user", "content": "next"})
    grown.append(_assistant_call("call_new"))
    grown.append(_tool_msg("call_new", "small"))
    project_stale_tool_results(agent, grown)

    assert _content(grown, "call_0") == first


def test_projection_is_monotone_even_inside_a_grown_tail():
    """Once projected, a row stays projected: un-stubbing would rewrite a cached prefix."""
    cc = _compressor()
    agent = _agent(cc)
    msgs = _build(12, big_indices={0, 1, 2, 3})
    project_stale_tool_results(agent, msgs)
    stub = _content(msgs, "call_3")

    # The tail budget now covers call_3's pair; stickiness must win over the tail.
    cc.tool_result_projection_tail_min_tokens = 900_000
    grown = [dict(m) for m in msgs] + [{"role": "user", "content": "next"}]
    project_stale_tool_results(agent, grown)

    assert _content(grown, "call_3") == stub


def test_a_pass_needs_freshly_stale_bytes_not_just_a_longer_wire():
    """The budget is charged against rows that became stale since the last pass, so the
    frontier sliding forward cannot authorize a cache break on every request."""
    agent = _agent()
    canonical = _build(12, big_indices={0, 1, 2, 3, 4, 5})

    first = [dict(m) for m in canonical]
    assert project_stale_tool_results(agent, first) >= 3

    # The next request rebuilds its copy from the untouched canonical history, so the same
    # rows are stubbed again — byte-identically (stability, not a new cache break).
    second = [dict(m) for m in canonical]
    replayed = project_stale_tool_results(agent, second)
    assert replayed >= 3
    assert _content(second, "call_0") == _content(first, "call_0")

    # One newly stale result is below the trigger: no second rewrite of the wire.
    grown = [dict(m) for m in canonical] + [
        _assistant_call("call_new"), _tool_msg("call_new", "N" * BIG_CHARS)
    ]
    assert project_stale_tool_results(agent, grown) == replayed
    assert _content(grown, "call_new") == "N" * BIG_CHARS


# ── invariant 6: the economic gates ─────────────────────────────────────────


def test_below_trigger_does_nothing():
    cc = _compressor(tool_result_projection_min_tokens=10_000_000)
    msgs = _build(12, big_indices={0, 1, 2, 3})
    assert project_stale_tool_results(_agent(cc), msgs) == 0


def test_below_min_reclaim_does_nothing():
    cc = _compressor(tool_result_projection_min_reclaim_tokens=10_000_000)
    msgs = _build(12, big_indices={0, 1, 2, 3})
    assert project_stale_tool_results(_agent(cc), msgs) == 0


def test_caching_route_declines_when_the_reclaim_cannot_pay_for_the_break():
    """The rewrite re-prefills everything after the first stub; that must be covered."""
    cc = _compressor()
    msgs = _build(12, big_indices={0, 1, 2, 3})
    # A huge protected tail makes the invalidated region far larger than the reclaim.
    msgs.append({"role": "user", "content": "t" * 200_000})

    assert project_stale_tool_results(_agent(cc, caching=True), msgs) == 0
    # Same request on a route without prompt caching pays nothing for the rewrite.
    assert project_stale_tool_results(_agent(cc, caching=False), msgs) == 4


def test_cache_capable_routes_need_a_bigger_pile_of_stale_bytes():
    policy = resolve_policy(_agent(_compressor(tool_result_projection_min_tokens=0)))
    assert trigger_tokens(policy, WINDOW, True) > trigger_tokens(policy, WINDOW, False)
    assert trigger_tokens(policy, None, False) > 0


def test_kill_switch_disables_the_pass():
    cc = _compressor(tool_result_projection="off")
    msgs = _build(12, big_indices={0, 1, 2, 3})
    assert project_stale_tool_results(_agent(cc), msgs) == 0


def test_policy_falls_back_to_defaults_on_a_bare_agent():
    policy = resolve_policy(SimpleNamespace())
    assert policy.enabled is True
    assert policy.min_result_chars > 0
    assert policy.tail_messages > 0


# ── helpers ─────────────────────────────────────────────────────────────────


def test_stub_is_a_pure_function_of_the_row():
    first = build_stub(
        tool_name="read_file", tool_args='{"path":"a.py"}', content_len=10, line_count=2,
        digest="deadbeef", recovery_path="/tmp/s.txt", already_persisted=False,
    )
    second = build_stub(
        tool_name="read_file", tool_args='{"path":"a.py"}', content_len=10, line_count=2,
        digest="deadbeef", recovery_path="/tmp/s.txt", already_persisted=False,
    )
    assert first == second
    assert first.startswith(PROJECTION_MARKER)
    assert first == first.strip()          # the send-path whitespace pass cannot move it
    assert not first.endswith("\n")


def test_stub_is_not_mistaken_for_an_existing_summary_stub():
    from agent.context_compressor import _is_summary_stub

    stub = build_stub(
        tool_name="terminal", tool_args='{"command":"pytest -q"}', content_len=1234, line_count=9,
        digest="cafe", recovery_path="/tmp/s.txt", already_persisted=False,
    )
    assert _is_summary_stub(stub) is False


def test_protected_tail_start_is_bounded_on_both_sides():
    msgs = _build(6, big_indices={0, 1})

    # Count cap: tiny messages can never spend a huge token budget, so the cap is what stops
    # the walk — the whole history must not become "the tail".
    capped = resolve_policy(
        _agent(
            _compressor(
                tool_result_projection_tail_min_tokens=10_000_000,
                tool_result_projection_tail_max_tokens=0,
                tool_result_projection_tail_messages=2,
                tool_result_projection_tail_max_messages=4,
            )
        )
    )
    assert protected_tail_start(msgs, capped, WINDOW) == len(msgs) - 4

    # Token budget: the walk stops as soon as the budget is spent.
    by_tokens = resolve_policy(
        _agent(
            _compressor(
                tool_result_projection_tail_min_tokens=6_000,
                tool_result_projection_tail_max_tokens=6_000,
                tool_result_projection_tail_messages=1,
            )
        )
    )
    assert 0 < protected_tail_start(msgs, by_tokens, WINDOW) < len(msgs)


def test_tool_call_index_maps_ids_to_name_and_args():
    index = tool_call_index([_assistant_call("c1", name="terminal", args='{"command":"ls"}')])
    assert index["c1"] == ("terminal", '{"command":"ls"}')


def test_is_candidate_rejects_small_and_non_tool_rows():
    policy = resolve_policy(_agent())
    assert is_candidate({"role": "user", "content": "x" * BIG_CHARS}, policy) is False
    assert is_candidate(_tool_msg("c", "small"), policy) is False
    assert is_candidate(_tool_msg("c", "x" * BIG_CHARS), policy) is True


@pytest.mark.parametrize("mode", ["auto", "on", "true", "enabled"])
def test_accepted_enable_modes(mode):
    assert resolve_policy(_agent(_compressor(tool_result_projection=mode))).enabled is True


@pytest.mark.parametrize("mode", ["off", "false", "disabled", "none", "0"])
def test_accepted_disable_modes(mode):
    assert resolve_policy(_agent(_compressor(tool_result_projection=mode))).enabled is False


# ── wired into the real request assembly ─────────────────────────────────────


class _CapturingCompletions:
    """One canned completion that records the messages the API call actually carried."""

    def __init__(self):
        self.requests: list = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        message = SimpleNamespace(content="Done.", tool_calls=[], reasoning=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=None,
        )


def _run_real_turn(monkeypatch, history):
    """Drive one real ``run_conversation`` turn and return ``(agent, result, sent_messages)``."""
    from run_agent import AIAgent

    completions = _CapturingCompletions()
    monkeypatch.setattr(
        "agent.process_bootstrap.OpenAI",
        lambda **_kw: SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *a, **k: [])

    agent = AIAgent(
        model="test-model", api_key="test-key", base_url="http://localhost:8080/v1",
        platform="cli", max_iterations=3, quiet_mode=True, skip_memory=True,
    )
    agent._disable_streaming = True
    # Deterministic policy: what the request carries, not how the window is inferred.
    agent.context_compressor.tool_result_projection_min_tokens = 8_000
    agent.context_compressor.tool_result_projection_min_reclaim_tokens = 4_000
    agent.context_compressor.tool_result_projection_tail_min_tokens = 2_000
    agent.context_compressor.tool_result_projection_tail_max_tokens = 2_000
    agent.context_compressor.tool_result_projection_tail_messages = 2
    agent.context_compressor.tool_result_projection_tail_max_messages = 6

    result = agent.run_conversation("keep going", conversation_history=[dict(m) for m in history])
    assert completions.requests, "the turn made no API call"
    return agent, result, completions.requests[0]["messages"]


def _stale_history():
    """Old, large tool results followed by a fresh tail of small ones."""
    history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "start"}]
    for i in range(10):
        history.append(_assistant_call(f"call_{i}", args=json.dumps({"path": f"src/f{i}.py"})))
        history.append(_tool_msg(f"call_{i}", chr(65 + i) * BIG_CHARS))
    for i in range(6):
        history.append(_assistant_call(f"call_tail_{i}"))
        history.append(_tool_msg(f"call_tail_{i}", "ok"))
    return history


def test_request_carries_stubs_while_the_transcript_keeps_the_bytes(monkeypatch):
    """The end-to-end contract: the wire is compacted, the stored conversation is not."""
    history = _stale_history()
    before = json.dumps(history, sort_keys=True)

    _agent_, result, sent = _run_real_turn(monkeypatch, history)

    stale_on_wire = [
        m for m in sent
        if m.get("role") == "tool" and isinstance(m.get("content"), str)
        and is_projected_tool_result(m["content"])
    ]
    assert stale_on_wire, "no stale tool result was projected on the wire"
    assert "spillover" in stale_on_wire[0]["content"]

    # The stored conversation still holds every byte, and the model still sees the tail.
    assert json.dumps(history, sort_keys=True) == before
    kept = [m for m in sent if m.get("role") == "tool" and m.get("tool_call_id") == "call_tail_5"]
    assert kept and kept[0]["content"] == "ok"
    assert not [m for m in sent if m.get("role") == "tool" and m["content"] == "J" * BIG_CHARS]


def test_spillover_written_by_the_projection_holds_the_original_bytes(monkeypatch):
    history = _stale_history()

    _agent_, _result, sent = _run_real_turn(monkeypatch, history)

    stub = next(
        m["content"] for m in sent
        if m.get("role") == "tool" and is_projected_tool_result(m.get("content") or "")
    )
    path = next(line for line in stub.splitlines() if "spillover" in line).rsplit(": ", 1)[1]
    recovered = Path(path).read_text(encoding="utf-8")
    assert recovered == next(m["content"] for m in history if m.get("tool_call_id") == "call_0")


def test_config_section_reaches_the_compressor():
    """``compression.tool_result_projection*`` must land on the compressor the pass reads."""
    from agent.agent_init import _parse_compression_config

    agent = SimpleNamespace(model="test", context_length=None, api_mode="chat_completions")
    cfg = {
        "compression": {
            "tool_result_projection": "off",
            "tool_result_projection_min_tokens": 12_345,
            "tool_result_projection_min_result_chars": 9_000,
            "tool_result_projection_min_reclaim_tokens": 7_000,
            "tool_result_projection_tail_ratio": 0.05,
            "tool_result_projection_tail_min_tokens": 6_000,
            "tool_result_projection_tail_max_tokens": 9_000,
            "tool_result_projection_tail_messages": 3,
            "tool_result_projection_tail_max_messages": 12,
        }
    }
    with patch("agent.context_compressor.get_model_context_length", return_value=WINDOW):
        cs = _parse_compression_config(agent, cfg)

    assert cs.tool_result_projection == "off"
    assert cs.tool_result_projection_min_tokens == 12_345
    assert cs.tool_result_projection_min_result_chars == 9_000
    assert cs.tool_result_projection_min_reclaim_tokens == 7_000
    assert cs.tool_result_projection_tail_ratio == pytest.approx(0.05)
    assert cs.tool_result_projection_tail_min_tokens == 6_000
    assert cs.tool_result_projection_tail_max_tokens == 9_000
    assert cs.tool_result_projection_tail_messages == 3
    assert cs.tool_result_projection_tail_max_messages == 12


def test_config_section_defaults_to_auto_and_survives_junk():
    from agent.agent_init import _parse_compression_config

    agent = SimpleNamespace(model="test", context_length=None, api_mode="chat_completions")
    junk = {"compression": {
        "tool_result_projection": None,
        "tool_result_projection_min_tokens": "nonsense",
        "tool_result_projection_tail_ratio": True,
    }}
    with patch("agent.context_compressor.get_model_context_length", return_value=WINDOW):
        cs = _parse_compression_config(agent, junk)

    assert cs.tool_result_projection == "auto"
    assert cs.tool_result_projection_min_tokens == 0
    assert cs.tool_result_projection_tail_ratio == pytest.approx(0.025)

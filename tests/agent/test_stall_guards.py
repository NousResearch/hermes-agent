"""Runtime anti-stall guards (agent.stall_guards).

Two guards, both notice/re-prompt-only:

1. Identical-call loop breaker — ``ToolCallGuardrailController.observe_identical_call``
   appends a compact notice to the tool RESULT on the 3rd consecutive call
   with identical (tool, canonical args) AND an identical result. It never
   blocks execution, exempts legitimately-repeatable pollers, and resets on
   any change of args, tool, or result — and per turn.

2. Said-continue-but-stopped detector — ``trailing_continue_intent`` flags a
   short reply that ENDS on an announced next action, feeding the existing
   bounded intent-ack continuation path (no new recovery machinery).

These assert behavior contracts, not message snapshots.
"""

from agent.agent_runtime_helpers import trailing_continue_intent
from agent.tool_guardrails import (
    IDENTICAL_RESULT_STUB_MIN_CHARS,
    STALL_GUARD_IDENTICAL_CALL_THRESHOLD,
    STALL_GUARD_REPEATABLE_TOOLS,
    ToolCallGuardrailController,
    is_stall_guard_repeatable,
)


def _observe_n(controller, n, tool="web_search", args=None, result="same result"):
    notices = []
    for _ in range(n):
        notices.append(
            controller.observe_identical_call(tool, args or {"query": "x"}, result)
        )
    return notices


# ── identical-call loop breaker ────────────────────────────────────────────


def test_fires_on_third_consecutive_identical_call_and_result():
    c = ToolCallGuardrailController()
    notices = _observe_n(c, 3)
    assert notices[0] is None
    assert notices[1] is None
    assert notices[2] is not None
    assert "hermes note" in notices[2]
    assert "3rd" in notices[2]
    assert "web_search" in notices[2]


def test_keeps_firing_past_threshold():
    c = ToolCallGuardrailController()
    notices = _observe_n(c, 4)
    assert notices[3] is not None
    assert "4th" in notices[3]


def test_does_not_fire_when_arguments_differ():
    c = ToolCallGuardrailController()
    for i in range(5):
        notice = c.observe_identical_call(
            "web_search", {"query": f"q{i}"}, "same result"
        )
        assert notice is None


def test_does_not_fire_when_results_differ():
    c = ToolCallGuardrailController()
    for i in range(5):
        notice = c.observe_identical_call(
            "terminal", {"command": "poll-status"}, f"output {i}"
        )
        assert notice is None


def test_streak_resets_when_a_different_call_intervenes():
    c = ToolCallGuardrailController()
    assert _observe_n(c, 2)[-1] is None
    # Different tool breaks the consecutive streak.
    assert c.observe_identical_call("read_file", {"path": "/a"}, "data") is None
    # Two more of the original are a fresh streak of 2 — still no notice.
    assert all(n is None for n in _observe_n(c, 2))


def test_arg_canonicalization_ignores_key_order():
    c = ToolCallGuardrailController()
    r = "same"
    assert c.observe_identical_call("t", {"a": 1, "b": 2}, r) is None
    assert c.observe_identical_call("t", {"b": 2, "a": 1}, r) is None
    assert c.observe_identical_call("t", {"a": 1, "b": 2}, r) is not None


def test_allowlisted_pollers_never_fire():
    c = ToolCallGuardrailController()
    for tool in ("process", "vendor_get_result", "job_poll"):
        for _ in range(STALL_GUARD_IDENTICAL_CALL_THRESHOLD + 2):
            assert c.observe_identical_call(tool, {"id": "j1"}, "Generating") is None


def test_allowlist_membership_contract():
    # The module constant drives the exemption; suffix conventions extend it.
    for tool in STALL_GUARD_REPEATABLE_TOOLS:
        assert is_stall_guard_repeatable(tool)
    assert is_stall_guard_repeatable("acme_get_result")
    assert not is_stall_guard_repeatable("web_search")
    assert not is_stall_guard_repeatable("terminal")


def test_resets_per_turn():
    c = ToolCallGuardrailController()
    assert _observe_n(c, 2)[-1] is None
    c.reset_for_turn()
    # Streak restarted: two more identical calls still under threshold.
    assert all(n is None for n in _observe_n(c, 2))
    # Third after the reset fires.
    assert _observe_n(c, 1)[-1] is not None


def test_never_blocks_execution():
    # The guard is observational only: before_call still allows the call even
    # after the notice fires repeatedly.
    c = ToolCallGuardrailController()
    _observe_n(c, 6)
    decision = c.before_call("web_search", {"query": "x"})
    assert decision.allows_execution


# ── result-append integration (AIAgent._append_guardrail_observation) ─────


def _fake_agent(stall_guards=True):
    from types import SimpleNamespace

    from run_agent import AIAgent

    agent = SimpleNamespace(
        _tool_guardrails=ToolCallGuardrailController(),
        _stall_guards=stall_guards,
        _tool_guardrail_halt_decision=None,
    )
    agent._stall_guards_enabled = lambda: AIAgent._stall_guards_enabled(agent)
    agent._set_tool_guardrail_halt = (
        lambda decision: AIAgent._set_tool_guardrail_halt(agent, decision)
    )
    agent._append = (
        lambda name, args, result, failed=False, tool_call_id="": (
            AIAgent._append_guardrail_observation(
                agent, name, args, result, failed=failed, tool_call_id=tool_call_id
            )
        )
    )
    return agent


def test_notice_appended_to_third_identical_result():
    agent = _fake_agent()
    args = {"query": "hermes"}
    r1 = agent._append("web_search", args, "results")
    r2 = agent._append("web_search", args, "results")
    r3 = agent._append("web_search", args, "results")
    assert "hermes note" not in r1
    assert "hermes note" not in r2
    assert "hermes note" in r3
    assert r3.startswith("results")  # notice appended, result preserved


def test_config_gate_disables_notice():
    agent = _fake_agent(stall_guards=False)
    args = {"query": "hermes"}
    for _ in range(4):
        result = agent._append("web_search", args, "results")
    assert "hermes note" not in result


def test_notice_streak_keys_on_raw_result_not_annotated_result():
    # The idempotent-no-progress warning suffix (whose count changes per call)
    # must not defeat result-identity matching: web_search is not in the
    # idempotent set by default, but read_file is — its results gain a
    # changing "[Tool loop warning: ...count=N...]" suffix from after_call,
    # and the streak must still be recognized from the raw result.
    agent = _fake_agent()
    args = {"path": "/tmp/x"}
    outs = [agent._append("read_file", args, "contents") for _ in range(3)]
    assert "hermes note" in outs[2]


# ── result-reference stubbing (byte-identical duplicate results) ──────────


_BIG = "x" * IDENTICAL_RESULT_STUB_MIN_CHARS  # exactly at the stub threshold


def test_stub_on_second_identical_call_first_full():
    agent = _fake_agent()
    args = {"query": "hermes"}
    r1 = agent._append("web_search", args, _BIG, tool_call_id="call_1")
    r2 = agent._append("web_search", args, _BIG, tool_call_id="call_2")
    assert r1 == _BIG  # first occurrence always enters context whole
    assert r2 != _BIG
    assert "byte-identical" in r2
    assert "web_search" in r2
    assert "call_1" in r2  # references the FIRST occurrence in the streak
    assert len(r2) < len(_BIG)


def test_no_stub_when_fresh_result_differs():
    agent = _fake_agent()
    args = {"query": "hermes"}
    agent._append("web_search", args, _BIG, tool_call_id="call_1")
    changed = "y" + _BIG
    r2 = agent._append("web_search", args, changed, tool_call_id="call_2")
    assert r2 == changed  # changed result flows through whole


def test_changed_result_resets_streak_then_stub_references_new_first():
    agent = _fake_agent()
    args = {"id": "job"}
    agent._append("web_search", args, _BIG, tool_call_id="a")
    changed = _BIG + "done"
    r2 = agent._append("web_search", args, changed, tool_call_id="b")
    assert r2 == changed
    r3 = agent._append("web_search", args, changed, tool_call_id="c")
    assert "byte-identical" in r3
    assert "tool_call_id b" in r3  # new streak's first occurrence, not 'a'


def test_no_stub_below_min_chars():
    agent = _fake_agent()
    small = "x" * (IDENTICAL_RESULT_STUB_MIN_CHARS - 1)
    args = {"query": "hermes"}
    agent._append("web_search", args, small, tool_call_id="c1")
    r2 = agent._append("web_search", args, small, tool_call_id="c2")
    assert "byte-identical" not in r2
    assert r2.startswith(small)  # full payload kept (pre-existing warning suffix allowed)


def test_no_stub_for_error_results():
    agent = _fake_agent()
    err = "Error executing tool: " + _BIG
    args = {"command": "boom"}
    agent._append("terminal", args, err, failed=True, tool_call_id="c1")
    r2 = agent._append("terminal", args, err, failed=True, tool_call_id="c2")
    assert "byte-identical" not in r2
    assert r2.startswith(err)  # models must see fresh errors whole


def test_pollers_get_stub_but_never_loop_notice():
    agent = _fake_agent()
    args = {"id": "job1"}
    results = [
        agent._append("vendor_get_result", args, _BIG, tool_call_id=f"c{i}")
        for i in range(4)
    ]
    assert results[0] == _BIG
    for r in results[1:]:
        assert "byte-identical" in r  # stubbed: unchanged poll saves context
        assert "consecutive identical call" not in r  # notice stays exempt


def test_third_identical_call_gets_stub_plus_loop_notice():
    agent = _fake_agent()
    args = {"query": "hermes"}
    agent._append("web_search", args, _BIG, tool_call_id="c1")
    agent._append("web_search", args, _BIG, tool_call_id="c2")
    r3 = agent._append("web_search", args, _BIG, tool_call_id="c3")
    assert "byte-identical" in r3  # stub replaces the payload
    assert "3rd consecutive identical call" in r3  # notice appended after it
    assert r3.index("byte-identical") < r3.index("3rd consecutive")


def test_stub_carries_spillover_path_when_first_result_persisted():
    agent = _fake_agent()
    args = {"query": "big"}
    agent._tool_guardrails.record_persisted_result(
        "c1", "/home/u/.hermes/cache/spillover/c1.txt"
    )
    agent._append("web_search", args, _BIG, tool_call_id="c1")
    r2 = agent._append("web_search", args, _BIG, tool_call_id="c2")
    assert "/home/u/.hermes/cache/spillover/c1.txt" in r2


def test_stub_includes_args_summary_for_compression_safety():
    agent = _fake_agent()
    args = {"query": "hermes result stubbing", "limit": 5}
    agent._append("web_search", args, _BIG, tool_call_id="c1")
    r2 = agent._append("web_search", args, _BIG, tool_call_id="c2")
    # Canonical-args preview so the model knows WHAT the call was even if
    # the referenced message is later evicted by compression.
    assert "hermes result stubbing" in r2


def test_stub_args_summary_truncated_to_120_chars():
    c = ToolCallGuardrailController()
    args = {"query": "q" * 500}
    assert c.observe_call("web_search", args, _BIG, tool_call_id="c1").stub is None
    stub = c.observe_call("web_search", args, _BIG, tool_call_id="c2").stub
    assert stub is not None
    args_part = stub.split("Args: ", 1)[1]
    assert len(args_part) < 200  # ~120-char preview + ellipsis + closer


def test_config_off_disables_stub():
    agent = _fake_agent(stall_guards=False)
    args = {"query": "hermes"}
    agent._append("web_search", args, _BIG, tool_call_id="c1")
    r2 = agent._append("web_search", args, _BIG, tool_call_id="c2")
    assert "byte-identical" not in r2
    assert r2.startswith(_BIG)


def test_streak_reset_by_different_call_means_next_identical_is_full():
    agent = _fake_agent()
    args = {"query": "hermes"}
    agent._append("web_search", args, _BIG, tool_call_id="c1")
    agent._append("read_file", {"path": "/a"}, "other", tool_call_id="c2")
    r3 = agent._append("web_search", args, _BIG, tool_call_id="c3")
    assert "byte-identical" not in r3
    assert r3.startswith(_BIG)  # fresh streak — first occurrence full again


def test_multimodal_content_never_stubbed_and_breaks_streak():
    c = ToolCallGuardrailController()
    args = {"path": "/img.png"}
    assert c.observe_call("vision", args, _BIG, tool_call_id="c1").stub is None
    # Non-string (multimodal) results never form or extend a streak.
    obs = c.observe_call("vision", args, None, tool_call_id="c2")
    assert obs.stub is None
    assert c.observe_call("vision", args, _BIG, tool_call_id="c3").stub is None


def test_observe_identical_call_backcompat_notice_still_fires():
    c = ToolCallGuardrailController()
    for _ in range(STALL_GUARD_IDENTICAL_CALL_THRESHOLD - 1):
        assert c.observe_identical_call("web_search", {"q": 1}, "r") is None
    assert c.observe_identical_call("web_search", {"q": 1}, "r") is not None


def test_extract_persisted_path_round_trip():
    # The stub's spillover reference is parsed from the <persisted-output>
    # block that maybe_persist_tool_result builds — assert the round trip.
    from tools.tool_result_storage import (
        _build_persisted_message,
        extract_persisted_path,
    )

    block = _build_persisted_message("preview", True, 50_000, "/tmp/spill/x.txt")
    assert extract_persisted_path(block) == "/tmp/spill/x.txt"
    assert extract_persisted_path("plain result") is None


# ── said-continue-but-stopped detector ─────────────────────────────────────


def test_detects_trailing_let_me_now():
    assert trailing_continue_intent("Found the config file. Let me now update it.")


def test_detects_trailing_i_will_now():
    assert trailing_continue_intent("The tests pass. I will now push the branch.")
    assert trailing_continue_intent("Good. I'll now run the linter")


def test_detects_trailing_next_i():
    assert trailing_continue_intent("Done with step one. Next, I check the logs")
    assert trailing_continue_intent("Step one complete. Next: I run the tests")


def test_ignores_intent_followed_by_more_content():
    # Intent phrase mid-message with substantive content after it — the model
    # already continued; nothing dangling.
    assert not trailing_continue_intent(
        "I will now explain the tradeoffs. First, caching: the design keeps "
        "the prefix stable. Second, alternation: roles must strictly alternate."
    )


def test_ignores_long_substantive_replies():
    long_reply = ("Here is the full analysis. " * 30) + "Let me now summarize."
    assert not trailing_continue_intent(long_reply)


def test_ignores_plain_final_answers():
    assert not trailing_continue_intent("The answer is 42.")
    assert not trailing_continue_intent("All tests pass and the branch is pushed.")
    assert not trailing_continue_intent("")
    assert not trailing_continue_intent(None)


def test_ignores_conversational_future_offers():
    # "I will" without the immediate-action shape must not trip the guard.
    assert not trailing_continue_intent(
        "I can help with that tomorrow if you'd like."
    )
    assert not trailing_continue_intent(
        "If you want, I will happily review the PR once CI is green. Just say so!"
    )

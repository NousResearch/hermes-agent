"""Pure tool-call guardrail primitive tests."""

import json

from agent.tool_guardrails import (
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    ToolCallSignature,
    ToolGuardrailDecision,
    append_toolguard_guidance,
    canonical_tool_args,
    classify_tool_failure,
)


def test_tool_call_signature_hashes_canonical_nested_unicode_args_without_exposing_raw_args():
    args_a = {
        "z": [{"β": "☤", "a": 1}],
        "a": {"y": 2, "x": "secret-token-value"},
    }
    args_b = {
        "a": {"x": "secret-token-value", "y": 2},
        "z": [{"a": 1, "β": "☤"}],
    }

    assert canonical_tool_args(args_a) == canonical_tool_args(args_b)
    sig_a = ToolCallSignature.from_call("web_search", args_a)
    sig_b = ToolCallSignature.from_call("web_search", args_b)

    assert sig_a == sig_b
    assert len(sig_a.args_hash) == 64
    metadata = sig_a.to_metadata()
    assert metadata == {"tool_name": "web_search", "args_hash": sig_a.args_hash}
    assert "secret-token-value" not in json.dumps(metadata)
    assert "☤" not in json.dumps(metadata)


def test_default_config_is_soft_warning_only_with_hard_stop_disabled():
    cfg = ToolCallGuardrailConfig()

    assert cfg.warnings_enabled is True
    assert cfg.hard_stop_enabled is False
    assert cfg.exact_failure_warn_after == 2
    assert cfg.same_tool_failure_warn_after == 3
    assert cfg.no_progress_warn_after == 2
    assert cfg.exact_failure_block_after == 5
    assert cfg.same_tool_failure_halt_after == 8
    assert cfg.no_progress_block_after == 5


def test_config_parses_nested_warn_and_hard_stop_thresholds():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {
            "warnings_enabled": False,
            "hard_stop_enabled": True,
            "warn_after": {
                "exact_failure": 3,
                "same_tool_failure": 4,
                "idempotent_no_progress": 5,
                "repeated_result": 9,
            },
            "hard_stop_after": {
                "exact_failure": 6,
                "same_tool_failure": 7,
                "idempotent_no_progress": 8,
                "repeated_result": 10,
            },
            "repeated_result_min_chars": 64,
        }
    )

    assert cfg.warnings_enabled is False
    assert cfg.hard_stop_enabled is True
    assert cfg.exact_failure_warn_after == 3
    assert cfg.same_tool_failure_warn_after == 4
    assert cfg.no_progress_warn_after == 5
    assert cfg.exact_failure_block_after == 6
    assert cfg.same_tool_failure_halt_after == 7
    assert cfg.no_progress_block_after == 8
    assert cfg.repeated_result_warn_after == 9
    assert cfg.repeated_result_halt_after == 10
    assert cfg.repeated_result_min_chars == 64


def test_default_config_guardrail_block_matches_dataclass_defaults():
    from hermes_cli.config import DEFAULT_CONFIG

    cfg = ToolCallGuardrailConfig.from_mapping(DEFAULT_CONFIG["tool_loop_guardrails"])

    assert cfg == ToolCallGuardrailConfig()


def test_default_repeated_identical_failed_call_warns_without_blocking():
    controller = ToolCallGuardrailController()
    args = {"query": "same"}

    decisions = []
    for _ in range(5):
        assert controller.before_call("web_search", args).action == "allow"
        decisions.append(
            controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
        )

    assert decisions[0].action == "allow"
    assert [d.action for d in decisions[1:]] == ["warn", "warn", "warn", "warn"]
    assert {d.code for d in decisions[1:]} == {"repeated_exact_failure_warning"}
    assert controller.before_call("web_search", args).action == "allow"
    assert controller.halt_decision is None


def test_hard_stop_enabled_blocks_repeated_exact_failure_before_next_execution():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            exact_failure_warn_after=2,
            exact_failure_block_after=2,
            same_tool_failure_halt_after=99,
        )
    )
    args = {"query": "same"}

    assert controller.before_call("web_search", args).action == "allow"
    first = controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    assert first.action == "allow"

    assert controller.before_call("web_search", args).action == "allow"
    second = controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    assert second.action == "warn"
    assert second.code == "repeated_exact_failure_warning"

    blocked = controller.before_call("web_search", args)
    assert blocked.action == "block"
    assert blocked.code == "repeated_exact_failure_block"
    assert blocked.count == 2


def test_success_resets_exact_signature_failure_streak():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, exact_failure_block_after=2, same_tool_failure_halt_after=99)
    )
    args = {"query": "same"}

    controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    controller.after_call("web_search", args, '{"ok":true}', failed=False)

    assert controller.before_call("web_search", args).action == "allow"
    controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    assert controller.before_call("web_search", args).action == "allow"


def test_file_mutation_lint_error_result_is_not_a_tool_failure():
    write_result = json.dumps({
        "bytes_written": 12,
        "lint": {"status": "error", "output": "SyntaxError: invalid syntax"},
    })
    patch_result = json.dumps({
        "success": True,
        "diff": "--- a/tmp.py\n+++ b/tmp.py\n",
        "lsp_diagnostics": "<diagnostics>ERROR [1:1] type mismatch</diagnostics>",
    })

    assert classify_tool_failure("write_file", write_result) == (False, "")
    assert classify_tool_failure("patch", patch_result) == (False, "")


def test_same_tool_varying_args_warns_by_default_without_halting():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(same_tool_failure_warn_after=2, same_tool_failure_halt_after=3)
    )

    first = controller.after_call("terminal", {"command": "cmd-1"}, '{"exit_code":1}', failed=True)
    second = controller.after_call("terminal", {"command": "cmd-2"}, '{"exit_code":1}', failed=True)
    third = controller.after_call("terminal", {"command": "cmd-3"}, '{"exit_code":1}', failed=True)
    fourth = controller.after_call("terminal", {"command": "cmd-4"}, '{"exit_code":1}', failed=True)

    assert first.action == "allow"
    assert [second.action, third.action, fourth.action] == ["warn", "warn", "warn"]
    assert {second.code, third.code, fourth.code} == {"same_tool_failure_warning"}
    assert "Do not switch to text-only replies" in second.message
    assert "keep using tools" in second.message
    assert "diagnose before retrying" in second.message
    assert "different tool" in second.message
    assert controller.halt_decision is None


def test_hard_stop_enabled_halts_same_tool_varying_args_failure_streak():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            exact_failure_block_after=99,
            same_tool_failure_warn_after=2,
            same_tool_failure_halt_after=3,
        )
    )

    first = controller.after_call("terminal", {"command": "cmd-1"}, '{"exit_code":1}', failed=True)
    assert first.action == "allow"
    second = controller.after_call("terminal", {"command": "cmd-2"}, '{"exit_code":1}', failed=True)
    assert second.action == "warn"
    assert second.code == "same_tool_failure_warning"
    third = controller.after_call("terminal", {"command": "cmd-3"}, '{"exit_code":1}', failed=True)
    assert third.action == "halt"
    assert third.code == "same_tool_failure_halt"
    assert third.count == 3


def test_idempotent_no_progress_repeated_result_warns_without_blocking_by_default():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(no_progress_warn_after=2, no_progress_block_after=2)
    )
    args = {"path": "/tmp/same.txt"}
    result = "same file contents"

    for _ in range(4):
        assert controller.before_call("read_file", args).action == "allow"
        decision = controller.after_call("read_file", args, result, failed=False)

    assert decision.action == "warn"
    assert decision.code == "idempotent_no_progress_warning"
    assert controller.before_call("read_file", args).action == "allow"
    assert controller.halt_decision is None


def test_hard_stop_enabled_blocks_idempotent_no_progress_future_repeat():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            no_progress_warn_after=2,
            no_progress_block_after=2,
        )
    )
    args = {"path": "/tmp/same.txt"}
    result = "same file contents"

    assert controller.before_call("read_file", args).action == "allow"
    assert controller.after_call("read_file", args, result, failed=False).action == "allow"
    assert controller.before_call("read_file", args).action == "allow"
    warn = controller.after_call("read_file", args, result, failed=False)
    assert warn.action == "warn"
    assert warn.code == "idempotent_no_progress_warning"

    blocked = controller.before_call("read_file", args)
    assert blocked.action == "block"
    assert blocked.code == "idempotent_no_progress_block"


def test_mutating_or_unknown_tools_are_not_blocked_for_repeated_identical_success_output_by_default():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(no_progress_warn_after=2, no_progress_block_after=2)
    )

    for _ in range(3):
        assert controller.before_call("write_file", {"path": "/tmp/x", "content": "x"}).action == "allow"
        assert controller.after_call("write_file", {"path": "/tmp/x", "content": "x"}, "ok", failed=False).action == "allow"
        assert controller.before_call("custom_tool", {"x": 1}).action == "allow"
        assert controller.after_call("custom_tool", {"x": 1}, "ok", failed=False).action == "allow"


def test_reset_for_turn_clears_bounded_guardrail_state():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, exact_failure_block_after=2, no_progress_block_after=2)
    )
    controller.after_call("web_search", {"query": "same"}, '{"error":"boom"}', failed=True)
    controller.after_call("web_search", {"query": "same"}, '{"error":"boom"}', failed=True)
    controller.after_call("read_file", {"path": "/tmp/x"}, "same", failed=False)
    controller.after_call("read_file", {"path": "/tmp/x"}, "same", failed=False)

    assert controller.before_call("web_search", {"query": "same"}).action == "block"
    assert controller.before_call("read_file", {"path": "/tmp/x"}).action == "block"

    controller.reset_for_turn()

    assert controller.before_call("web_search", {"query": "same"}).action == "allow"
    assert controller.before_call("read_file", {"path": "/tmp/x"}).action == "allow"


def test_repeated_identical_result_halts_successful_varying_arg_loop():
    """The real-world loop: execute_code 'succeeds' with different args every
    call but returns the same blocked/404 body. Failure- and signature-keyed
    counters miss it; result-repetition detection must catch it."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            repeated_result_warn_after=3,
            repeated_result_halt_after=5,
        )
    )
    blocked_body = (
        '{"status": "success", "output": "=== SOME REMOTE API RESPONSE ==='
        + "<!DOCTYPE html>" + "x" * 300 + '"}'
    )

    decisions = []
    for i in range(5):
        # Different arguments every call, never classified as a failure.
        decisions.append(
            controller.after_call(
                "execute_code", {"code": f"fetch_attempt_{i}()"}, blocked_body, failed=False
            )
        )

    assert decisions[2].action == "warn"
    assert decisions[2].code == "repeated_result_warning"
    assert decisions[4].action == "halt"
    assert decisions[4].code == "repeated_result_halt"
    assert controller.halt_decision is decisions[4]


def test_repeated_result_ignores_short_and_distinct_outputs():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, repeated_result_halt_after=3)
    )
    # Short identical results must not trip the guard.
    for _ in range(5):
        d = controller.after_call("execute_code", {"code": "x"}, '{"ok":true}', failed=False)
        assert d.action == "allow"
    # Long but genuinely distinct results = real progress, never halts.
    for i in range(5):
        body = '{"status":"success","output":"' + f"unique-{i}-" + "y" * 300 + '"}'
        d = controller.after_call("execute_code", {"code": f"c{i}"}, body, failed=False)
        assert d.action == "allow"
    assert controller.halt_decision is None


def _multimodal_vision_result(image_b64: str, question: str = "Describe everything visible.") -> dict:
    """Shape returned by a native vision-analysis tool result."""
    return {
        "_multimodal": True,
        "content": [
            {
                "type": "text",
                "text": (
                    "Image loaded into your context — you can see it natively now. "
                    "Use your built-in vision to answer the user."
                    f"\n\nQuestion: {question}" + " " * 120
                ),
            },
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ],
        "text_summary": "Image attached natively for the main model.",
    }


def test_repeated_multimodal_result_same_image_trips_guard():
    """Multimodal results embed base64 image payloads, so a naive str(result)
    hash is unique per image and identical *text* parts (the placeholder
    caption) can never be seen as repetition. Re-loading the *same* image
    repeatedly must still count as repetition."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            repeated_result_warn_after=3,
            repeated_result_halt_after=5,
        )
    )
    same = _multimodal_vision_result("A" * 4000)
    decisions = [
        controller.after_call("vision_analyze", {"image_url": f"/tmp/shot_{i}.png"}, same, failed=False)
        for i in range(5)
    ]
    assert decisions[2].action == "warn"
    assert decisions[2].code == "repeated_result_warning"
    assert decisions[4].action == "halt"
    assert decisions[4].code == "repeated_result_halt"


def test_repeated_multimodal_result_distinct_images_is_progress():
    """Distinct screenshots with an identical text part (scrolling through a long
    page) are legitimate progress and must NOT halt, even past the threshold."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, repeated_result_halt_after=5)
    )
    for i in range(8):
        result = _multimodal_vision_result(f"IMG{i}" * 1000)
        d = controller.after_call(
            "vision_analyze", {"image_url": f"/tmp/shot_{i}.png"}, result, failed=False
        )
        assert d.action == "allow"
    assert controller.halt_decision is None


def test_repeated_result_interleaved_sequences_never_warn():
    """A/B/A/C/A must stay silent: the same result recurring non-consecutively
    is not a loop, and the warn/halt messages promise 'the last N calls', so
    only a consecutive streak may count."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, repeated_result_warn_after=3)
    )
    bodies = {
        key: '{"status":"success","output":"' + key * 240 + '"}' for key in "ABC"
    }
    for key in "ABACA":
        d = controller.after_call("execute_code", {"code": key}, bodies[key], failed=False)
        assert d.action == "allow"
    assert controller.halt_decision is None


def test_repeated_result_streak_restarts_after_interruption():
    """A different result breaks the streak; a fresh run of identical results
    must reach the threshold on its own before warning, and the reported count
    reflects only the consecutive run."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(repeated_result_warn_after=3)
    )
    body_a = '{"status":"success","output":"' + "A" * 240 + '"}'
    body_b = '{"status":"success","output":"' + "B" * 240 + '"}'

    decisions = [
        controller.after_call("execute_code", {"code": f"c{i}"}, body, failed=False)
        for i, body in enumerate([body_a, body_a, body_b, body_a, body_a, body_a])
    ]

    assert [d.action for d in decisions[:5]] == ["allow"] * 5
    assert decisions[5].action == "warn"
    assert decisions[5].code == "repeated_result_warning"
    assert decisions[5].count == 3


def test_empty_success_envelope_loop_from_read_tool_trips_guard():
    """The production case from issue #60084's report thread: session_search
    returning {"success": true, "results": [], "count": 0} for 16 different
    hallucinated queries. Too short for repeated_result_min_chars, but an
    identical no-content success from a non-mutating tool is still a loop."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            repeated_result_warn_after=3,
            repeated_result_halt_after=5,
        )
    )
    envelope = '{"success": true, "results": [], "count": 0}'
    decisions = [
        controller.after_call("session_search", {"query": f"q{i}"}, envelope, failed=False)
        for i in range(5)
    ]
    assert decisions[2].action == "warn"
    assert decisions[2].code == "repeated_result_warning"
    assert decisions[4].action == "halt"
    assert decisions[4].code == "repeated_result_halt"


def test_empty_success_envelope_from_mutating_tool_is_ignored():
    """Mutating tools legitimately return the same terse success envelope on
    every call — each call did its work, so no loop may be inferred."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, repeated_result_warn_after=2)
    )
    envelope = '{"success": true, "results": [], "count": 0}'
    for i in range(6):
        d = controller.after_call("todo", {"add": f"item-{i}"}, envelope, failed=False)
        assert d.action == "allow"
    assert controller.halt_decision is None


def test_bare_success_ack_is_not_an_empty_envelope():
    """{"success": true} with no emptiness marker is an acknowledgment, not a
    no-content result; repeated acks from varying calls must stay silent."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, repeated_result_warn_after=2)
    )
    for i in range(6):
        d = controller.after_call(
            "web_extract", {"url": f"https://x/{i}"}, '{"success": true}', failed=False
        )
        assert d.action == "allow"
    assert controller.halt_decision is None


def test_append_toolguard_guidance_keeps_multimodal_dict_shape():
    """Guidance on a multimodal result must not string-concatenate the dict:
    it lands as a trailing text part and in text_summary, and the image parts
    survive untouched."""
    decision = ToolGuardrailDecision(
        action="warn",
        code="repeated_result_warning",
        message="Same result repeated.",
        tool_name="vision_analyze",
        count=3,
    )
    original = _multimodal_vision_result("A" * 64)

    updated = append_toolguard_guidance(original, decision)

    assert updated["_multimodal"] is True
    assert updated["content"][-1]["type"] == "text"
    assert "repeated_result_warning" in updated["content"][-1]["text"]
    assert "Tool loop warning" in updated["content"][-1]["text"]
    assert any(part.get("type") == "image_url" for part in updated["content"])
    assert "repeated_result_warning" in updated["text_summary"]
    # The caller's dict is not mutated in place.
    assert "repeated_result_warning" not in str(original)

"""Tests for edge-mode working memory scratchpad helpers."""

from unittest.mock import MagicMock

import pytest

from agent.context_compressor import SUMMARY_PREFIX
from agent.edge_scratchpad_delta import accumulate_scratchpad_delta
from agent.edge_working_memory import (
    EdgeCompressGuardError,
    absorb_assistant_scratchpad_update,
    begin_edge_turn_injection,
    append_compaction_delta_to_scratchpad,
    default_scratchpad,
    edge_scratchpad_for_injection,
    effective_compression_trigger_tokens,
    extract_compression_summary_text,
    format_edge_working_memory_injection,
    merge_focus_topic_with_scratchpad,
    maybe_edge_flush_mid_turn,
    validate_edge_compress_guard,
)


def test_default_scratchpad_layout():
    sp = default_scratchpad("Ship the feature")
    assert "### CRIT" in sp
    assert "**Goal:** Ship the feature" in sp
    assert "### KNOW" in sp
    assert "**Facts:**" in sp
    assert "### NEXT" in sp
    assert "[ ]" in sp


def test_merge_focus_with_scratchpad():
    m = merge_focus_topic_with_scratchpad("auth bug", "### CRIT\n- **Goal:** x")
    assert "auth bug" in m
    assert "Goal" in m


def test_merge_focus_scratchpad_only():
    m = merge_focus_topic_with_scratchpad(None, "**Goal:** z")
    assert m is not None
    assert "Goal" in m


def test_extract_compression_summary():
    body = "alpha discussed beta"
    msgs = [
        {"role": "user", "content": f"{SUMMARY_PREFIX}\n{body}\n\n--- END OF CONTEXT SUMMARY — x"},
    ]
    assert extract_compression_summary_text(msgs).startswith("alpha discussed")


def test_append_compaction_delta_inserts_under_facts():
    base = default_scratchpad("goal")
    assert "**Facts:**" in base
    out = append_compaction_delta_to_scratchpad(base, "found the root cause")
    assert "[cmpct]" in out
    assert "root cause" in out


def test_absorb_assistant_scratchpad_update():
    class A:
        edge_mode = True
        _edge_scratchpad = ""

    a = A()
    absorb_assistant_scratchpad_update(
        a,
        "Hello\n\n### CRIT\n- **Goal:** G\n",
    )
    assert a._edge_scratchpad.startswith("### CRIT")
    assert "**Goal:** G" in a._edge_scratchpad


def test_turn_injection_freeze_ignores_mid_turn_mutations():
    """API injection must stay byte-stable across mid-turn scratchpad updates."""

    class A:
        edge_mode = True
        _edge_scratchpad = default_scratchpad("original goal")

    a = A()
    begin_edge_turn_injection(a)
    first = format_edge_working_memory_injection(edge_scratchpad_for_injection(a))
    absorb_assistant_scratchpad_update(
        a,
        "### CRIT\n- **Goal:** mutated mid-turn\n",
    )
    assert "**Goal:** mutated mid-turn" in a._edge_scratchpad
    second = format_edge_working_memory_injection(edge_scratchpad_for_injection(a))
    assert first == second
    assert "original goal" in first
    assert "mutated mid-turn" not in second


def test_absorb_trims_after_crlf_markdown_separator():
    """CRLF from local SLMs must not bypass stop-sequence detection."""
    class A:
        edge_mode = True
        _edge_scratchpad = ""

    a = A()
    absorb_assistant_scratchpad_update(
        a,
        "intro\r\n### CRIT\r\n- **Goal:** G\r\n\r\n---\r\nIgnore this narrative tail.",
    )
    assert "### CRIT" in a._edge_scratchpad
    assert "Ignore this narrative tail" not in a._edge_scratchpad


def test_effective_compression_trigger_tokens_scaled():
    class C:
        threshold_tokens = 10000
        _compression_threshold_scale = 0.8

    assert effective_compression_trigger_tokens(C()) == 8000


def test_should_compress_honors_threshold_scale():
    from agent.context_compressor import ContextCompressor

    cc = ContextCompressor(
        model="gpt-4",
        threshold_percent=0.5,
        protect_first_n=0,
        protect_last_n=2,
        summary_target_ratio=0.2,
        base_url="",
        api_key="",
        config_context_length=100_000,
        provider="openai",
        api_mode="chat_completions",
    )
    cc.threshold_tokens = 10_000
    cc._compression_threshold_scale = 0.8
    cc._ineffective_compression_count = 0
    cc.last_prompt_tokens = 7999
    assert cc.should_compress() is False
    cc.last_prompt_tokens = 8000
    assert cc.should_compress() is True


def test_validate_edge_compress_guard_ok():
    class A:
        edge_mode = True
        _edge_scratchpad = default_scratchpad("Fix the bug in auth")

    validate_edge_compress_guard(A(), [{"role": "user", "content": "Fix the bug in auth please"}])


def test_validate_edge_compress_guard_raises_when_goal_missing():
    class A:
        edge_mode = True
        _edge_scratchpad = "### CRIT\n- **Phase:** x\n"

    with pytest.raises(EdgeCompressGuardError):
        validate_edge_compress_guard(A(), [{"role": "user", "content": "hello"}])


def test_validate_edge_compress_guard_raises_when_user_diverges():
    class A:
        edge_mode = True
        _edge_scratchpad = default_scratchpad("totally different topic xyz")

    with pytest.raises(EdgeCompressGuardError):
        validate_edge_compress_guard(
            A(), [{"role": "user", "content": "unrelated request about bananas"}]
        )


def test_validate_edge_compress_guard_accepts_paraphrase_via_word_overlap():
    class A:
        edge_mode = True
        _edge_scratchpad = default_scratchpad(
            "Ship the hardened authentication fix before production"
        )

    validate_edge_compress_guard(
        A(),
        [
            {
                "role": "user",
                "content": "Please ship the hardened authentication fix to production soon.",
            }
        ],
    )


def test_maybe_edge_flush_invokes_compress_when_round_cap():
    class A:
        edge_mode = True
        _edge_flush_assistant_rounds = 2
        _edge_flush_token_soft_limit = 0
        tools = None

    a = A()
    a._compress_context = MagicMock(
        return_value=([{"role": "system", "content": "x"}], "sys2")
    )
    msgs = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a1"},
        {"role": "assistant", "content": "a2"},
    ]
    out_m, out_s = maybe_edge_flush_mid_turn(
        a, msgs, "base", 0, "tid", "active-sys",
    )
    assert out_s == "sys2"
    a._compress_context.assert_called_once()


def test_accumulate_scratchpad_delta_adds_facts_dedupes_path():
    base = default_scratchpad("g")
    out = accumulate_scratchpad_delta(
        base,
        add_facts=["- see `src/a.py` for hook", "- dup `src/a.py` second"],
        dedupe_paths=True,
    )
    assert "src/a.py" in out
    assert out.count("src/a.py") == 1

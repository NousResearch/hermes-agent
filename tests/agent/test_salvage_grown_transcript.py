"""Mechanical salvage for compression candidates that would grow."""

from unittest.mock import MagicMock

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    _SUMMARY_END_MARKER,
    salvage_grown_transcript,
)
from agent.model_metadata import estimate_messages_tokens_rough
from tools.todo_tool import TODO_INJECTION_HEADER


def test_salvage_stubs_old_tools_and_keeps_todo_when_stubbing_suffices():
    """Tool stubbing alone gets under budget → the todo snapshot survives.

    The snapshot is the only in-transcript todo re-injection at the boundary
    (and may carry the pruned-skill reload notice), so it is last-resort only.
    """
    original = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "ok"},
        {"role": "tool", "tool_call_id": "a", "content": "A" * 4000},
        {"role": "tool", "tool_call_id": "b", "content": "B" * 4000},
        {"role": "tool", "tool_call_id": "c", "content": "keep-latest"},
    ]
    grown = original + [
        {
            "role": "user",
            "content": "Current todos:\n- [ ] x",
            "_todo_snapshot_synthetic": True,
        }
    ]
    assert estimate_messages_tokens_rough(grown) > estimate_messages_tokens_rough(original)

    out = salvage_grown_transcript(original, grown)

    assert out is not None
    assert estimate_messages_tokens_rough(out) < estimate_messages_tokens_rough(original)
    assert any(m.get("_todo_snapshot_synthetic") for m in out)
    tools = [m["content"] for m in out if m.get("role") == "tool"]
    assert tools[-1] == "keep-latest"
    assert any("cleared to save context space" in t for t in tools)


def test_salvage_drops_todo_only_as_last_resort():
    """When cheaper ops cannot get under budget, the snapshot is dropped."""
    original = [
        {"role": "user", "content": "please do the thing " + ("o" * 600)},
        {"role": "assistant", "content": "ok"},
    ]
    grown = [
        {"role": "user", "content": "summary of the ask"},
        {"role": "assistant", "content": "ok"},
        {
            "role": "user",
            "content": f"{TODO_INJECTION_HEADER}\n- [ ] " + ("t" * 800),
            "_todo_snapshot_synthetic": True,
        },
    ]
    assert estimate_messages_tokens_rough(grown) > estimate_messages_tokens_rough(original)

    out = salvage_grown_transcript(original, grown)

    assert out is not None
    assert estimate_messages_tokens_rough(out) < estimate_messages_tokens_rough(original)
    assert not any(m.get("_todo_snapshot_synthetic") for m in out)


def test_salvage_last_resort_preserves_pruned_skill_reload_notice():
    """7a16840add couples the reload notice into the snapshot — it survives."""
    from agent.conversation_compression import _PRUNED_SKILL_RELOAD_NOTICE_HEADER

    notice = (
        f"{_PRUNED_SKILL_RELOAD_NOTICE_HEADER}\n"
        "Reload with skill_view(name='example-skill') before acting."
    )
    original = [
        {"role": "user", "content": "please do the thing " + ("o" * 3000)},
        {"role": "assistant", "content": "ok"},
    ]
    grown = [
        {"role": "user", "content": "summary of the ask"},
        {"role": "assistant", "content": "ok"},
        {
            "role": "user",
            "content": f"{TODO_INJECTION_HEADER}\n- [ ] " + ("t" * 4000) + f"\n\n{notice}",
            "_todo_snapshot_synthetic": True,
        },
    ]
    assert estimate_messages_tokens_rough(grown) > estimate_messages_tokens_rough(original)

    out = salvage_grown_transcript(original, grown)

    assert out is not None
    assert estimate_messages_tokens_rough(out) < estimate_messages_tokens_rough(original)
    snapshot_rows = [m for m in out if m.get("_todo_snapshot_synthetic")]
    assert len(snapshot_rows) == 1
    assert snapshot_rows[0]["content"].startswith(_PRUNED_SKILL_RELOAD_NOTICE_HEADER)
    assert TODO_INJECTION_HEADER not in snapshot_rows[0]["content"]


def test_salvage_preserves_recovery_nudge_around_folded_todo():
    """Last-resort TODO removal must retain other content sharing its user row."""
    from agent.context_compressor import _skill_pruned_marker
    from agent.conversation_compression import (
        _PRUNED_SKILL_RELOAD_NOTICE_HEADER,
        _durable_compaction_messages,
        _fold_todo_snapshot,
    )
    from agent.conversation_loop import _EMPTY_TOOL_RESPONSE_NUDGE

    original = [
        {"role": "user", "content": "please continue " + ("o" * 3000)},
        {"role": "assistant", "content": "ok"},
    ]
    candidate = [
        {"role": "user", "content": f"summary of the ask\n{_skill_pruned_marker('example-skill')}"},
        {
            "role": "assistant",
            "content": "(empty)",
            "_empty_recovery_synthetic": True,
        },
        {
            "role": "user",
            "content": _EMPTY_TOOL_RESPONSE_NUDGE,
            "_empty_recovery_synthetic": True,
        },
    ]
    agent = MagicMock()
    agent._todo_store.has_items.return_value = True
    agent._todo_store.format_for_injection.return_value = (
        f"{TODO_INJECTION_HEADER}\n- [ ] " + ("t" * 4000)
    )
    _fold_todo_snapshot(agent, candidate)
    assert estimate_messages_tokens_rough(candidate) > estimate_messages_tokens_rough(original)

    out = salvage_grown_transcript(original, candidate)

    assert out is not None
    recovery_rows = [m for m in out if _EMPTY_TOOL_RESPONSE_NUDGE in str(m.get("content"))]
    assert len(recovery_rows) == 1
    assert TODO_INJECTION_HEADER not in str(recovery_rows[0]["content"])
    assert _PRUNED_SKILL_RELOAD_NOTICE_HEADER in str(recovery_rows[0]["content"])

    durable = _durable_compaction_messages(out)
    assert _EMPTY_TOOL_RESPONSE_NUDGE not in str(durable)
    assert not any(
        previous.get("role") == current.get("role")
        for previous, current in zip(durable, durable[1:])
    )


def test_salvage_returns_none_when_nothing_can_shrink():
    original = [{"role": "user", "content": "tiny"}]
    huge = [{"role": "user", "content": "X" * 200_000}]

    assert salvage_grown_transcript(original, huge) is None


def test_salvage_caps_oversized_summary():
    original = [
        {"role": "user", "content": "ask " + ("o" * 12_000)},
        {"role": "assistant", "content": "short reply"},
    ]
    grown = [
        {
            "role": "user",
            "content": (
                "[CONTEXT COMPACTION] "
                + ("S" * 20_000)
                + "\n\n"
                + _SUMMARY_END_MARKER
            ),
            COMPRESSED_SUMMARY_METADATA_KEY: True,
        },
        {"role": "assistant", "content": "short reply"},
    ]
    assert estimate_messages_tokens_rough(grown) > estimate_messages_tokens_rough(original)

    out = salvage_grown_transcript(original, grown)

    assert out is not None
    assert len(out[0]["content"]) < 12_000
    assert "truncated so compaction can shrink" in out[0]["content"]
    assert out[0]["content"].endswith(_SUMMARY_END_MARKER)


def test_salvage_never_truncates_merged_summary_with_live_user_tail():
    original = [{"role": "user", "content": "O" * 12_000}]
    merged = [
        {
            "role": "user",
            "content": (
                "[CONTEXT COMPACTION] "
                + ("S" * 20_000)
                + "\n\n"
                + _SUMMARY_END_MARKER
                + "\n\nLIVE USER REQUEST"
            ),
            COMPRESSED_SUMMARY_METADATA_KEY: True,
        }
    ]

    assert salvage_grown_transcript(original, merged) is None
    assert merged[0]["content"].endswith("LIVE USER REQUEST")


def test_salvage_does_not_cap_plain_user_text_quoting_summary_marker():
    original = [{"role": "user", "content": "O" * 20_000}]
    quoted = "ordinary user text " + ("Q" * 12_000) + "\n\n" + _SUMMARY_END_MARKER
    candidate = [{"role": "user", "content": quoted}]

    out = salvage_grown_transcript(original, candidate)

    assert out is not None
    assert out[0]["content"] == quoted
    assert "truncated so compaction can shrink" not in out[0]["content"]


def test_salvage_never_caps_unmarked_summary_shaped_live_user_text():
    original = [{"role": "user", "content": "O" * 20_000}]
    live_user_text = (
        "[CONTEXT COMPACTION] "
        + ("U" * 12_000)
        + "\n\n"
        + _SUMMARY_END_MARKER
    )
    candidate = [{"role": "user", "content": live_user_text}]

    out = salvage_grown_transcript(original, candidate)

    assert out is not None
    assert out[0]["content"] == live_user_text
    assert "truncated so compaction can shrink" not in out[0]["content"]

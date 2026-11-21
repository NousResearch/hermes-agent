"""Todo snapshot compaction preserves the newest real-human instruction."""

from types import SimpleNamespace

from tools.todo_tool import TODO_INJECTION_HEADER


class _TodoStore:
    def __init__(self, snapshot: str, *, has_items: bool = False) -> None:
        self.snapshot = snapshot
        self._has_items = has_items

    def format_for_injection(self) -> str:
        return self.snapshot

    def has_items(self) -> bool:
        return self._has_items


def _agent(snapshot: str, *, has_items: bool = False) -> SimpleNamespace:
    from agent.agent_runtime_helpers import repair_message_sequence

    return SimpleNamespace(
        _todo_store=_TodoStore(snapshot, has_items=has_items),
        _repair_message_sequence=lambda messages: repair_message_sequence(None, messages),
    )


def test_fold_removes_stale_snapshot_after_newer_human_correction() -> None:
    """A protected stale row must not become the latest user instruction."""
    from agent.conversation_compression import _fold_todo_snapshot

    correction = "Stop the old work and fix reply targeting."
    stale = f"{TODO_INJECTION_HEADER}\n- [>] old-task. Continue old work"
    fresh = f"{TODO_INJECTION_HEADER}\n- [>] new-task. Fix reply targeting"
    compressed = [
        {"role": "user", "content": "[CONTEXT COMPACTION — REFERENCE ONLY] summary"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": correction},
        {"role": "assistant", "content": "I am investigating."},
        {"role": "tool", "content": "diagnostic output"},
        {"role": "user", "content": stale, "_todo_snapshot_synthetic": True},
    ]

    # This is the normal production path: the store is authoritative and has
    # a fresh snapshot. The fold must not erase evidence of stale scaffolding
    # before choosing the newest real-human anchor.
    _fold_todo_snapshot(_agent(fresh, has_items=True), compressed)

    correction_row = next(
        message for message in compressed
        if str(message.get("content") or "").startswith(correction)
    )
    assert fresh in correction_row["content"]
    assert stale not in "\n".join(str(message.get("content") or "") for message in compressed)
    assert compressed[-1]["role"] == "tool"
    assert not any(message.get("_todo_snapshot_synthetic") for message in compressed)


def test_stale_structured_snapshot_rejoins_newest_human_correction() -> None:
    """Structured stale todo content cannot hide the newest real user turn."""
    from agent.conversation_compression import _inject_todo_snapshot_without_reordering_human_intent

    correction = [
        {"type": "text", "text": "Use the customer's corrected recipient."},
        {"type": "input_audio", "input_audio": {"data": "preserve-me"}},
    ]
    stale = f"{TODO_INJECTION_HEADER}\n- [>] old recipient"
    fresh = f"{TODO_INJECTION_HEADER}\n- [>] corrected recipient"
    compressed = [
        {"role": "user", "content": correction},
        {"role": "assistant", "content": "Acknowledged."},
        {
            "role": "user",
            "content": [{"type": "text", "text": stale}],
            "_todo_snapshot_synthetic": True,
        },
    ]

    _inject_todo_snapshot_without_reordering_human_intent(compressed, fresh)

    assert compressed[0]["content"][:2] == correction
    assert any(part.get("text") == fresh for part in compressed[0]["content"] if isinstance(part, dict))
    assert not any(stale in str(message.get("content") or "") for message in compressed)
    assert not any(message.get("_todo_snapshot_synthetic") for message in compressed)


def test_stale_snapshot_before_synthetic_assistant_tail_uses_human_correction() -> None:
    """A synthetic assistant tail is not a real-user anchor after stale cleanup."""
    from agent.conversation_compression import _inject_todo_snapshot_without_reordering_human_intent

    correction = "Send only to the corrected contact."
    stale = f"{TODO_INJECTION_HEADER}\n- [>] old recipient"
    fresh = f"{TODO_INJECTION_HEADER}\n- [>] corrected recipient"
    compressed = [
        {"role": "user", "content": correction},
        {"role": "user", "content": stale, "_todo_snapshot_synthetic": True},
        {"role": "assistant", "content": "[synthetic continuation]", "_synthetic": True},
    ]

    _inject_todo_snapshot_without_reordering_human_intent(compressed, fresh)

    assert fresh in compressed[0]["content"]
    assert compressed[-1]["role"] == "assistant"
    assert not any(message.get("_todo_snapshot_synthetic") for message in compressed)


def test_fold_repairs_internal_stale_snapshot_boundary_before_appending_fresh_one() -> None:
    """Removing an internal standalone snapshot must not leave assistant, assistant."""
    from agent.conversation_compression import _fold_todo_snapshot

    stale = f"{TODO_INJECTION_HEADER}\n- [>] old task"
    fresh = f"{TODO_INJECTION_HEADER}\n- [>] current task"
    compressed = [
        {"role": "assistant", "content": "summary"},
        {"role": "user", "content": stale, "_todo_snapshot_synthetic": True},
        {"role": "assistant", "content": "tail"},
    ]

    _fold_todo_snapshot(_agent(fresh, has_items=True), compressed)

    assert [message["role"] for message in compressed] == ["assistant", "user"]
    assert compressed[-1]["content"] == fresh


def test_fold_repairs_every_repeated_stale_snapshot_boundary() -> None:
    """Cleanup of multiple stale snapshots repairs every newly-adjacent assistant pair."""
    from agent.conversation_compression import _fold_todo_snapshot

    stale = f"{TODO_INJECTION_HEADER}\n- [>] old task"
    fresh = f"{TODO_INJECTION_HEADER}\n- [>] current task"
    compressed = [
        {"role": "assistant", "content": "summary"},
        {"role": "user", "content": stale, "_todo_snapshot_synthetic": True},
        {"role": "assistant", "content": "middle"},
        {"role": "user", "content": stale, "_todo_snapshot_synthetic": True},
        {"role": "assistant", "content": "tail"},
    ]

    _fold_todo_snapshot(_agent(fresh, has_items=True), compressed)

    assert [message["role"] for message in compressed] == ["assistant", "user"]
    assert compressed[-1]["content"] == fresh


def test_stale_snapshot_strip_preserves_image_only_content() -> None:
    """A stale text block must not discard an accompanying image input."""
    from agent.conversation_compression import _inject_todo_snapshot_without_reordering_human_intent

    stale = f"{TODO_INJECTION_HEADER}\n- [>] old task"
    fresh = f"{TODO_INJECTION_HEADER}\n- [>] current task"
    image = {"type": "input_image", "image_url": "data:image/png;base64,keep-me"}
    compressed = [{"role": "user", "content": [image, {"type": "text", "text": stale}], "_todo_snapshot_synthetic": True}]

    _inject_todo_snapshot_without_reordering_human_intent(compressed, fresh)

    assert len(compressed) == 1
    assert [message["role"] for message in compressed] == ["user"]
    assert compressed[0]["content"][0] == image
    assert any(part.get("text") == fresh for part in compressed[0]["content"] if isinstance(part, dict))
    assert compressed[0]["_todo_snapshot_synthetic"] is True


def test_stale_snapshot_strip_preserves_audio_only_content() -> None:
    """A stale text block must not discard an accompanying audio input."""
    from agent.conversation_compression import _inject_todo_snapshot_without_reordering_human_intent

    stale = f"{TODO_INJECTION_HEADER}\n- [>] old task"
    fresh = f"{TODO_INJECTION_HEADER}\n- [>] current task"
    audio = {"type": "input_audio", "input_audio": {"data": "keep-me"}}
    compressed = [{"role": "user", "content": [audio, {"type": "text", "text": stale}], "_todo_snapshot_synthetic": True}]

    _inject_todo_snapshot_without_reordering_human_intent(compressed, fresh)

    assert len(compressed) == 1
    assert [message["role"] for message in compressed] == ["user"]
    assert compressed[0]["content"][0] == audio
    assert any(part.get("text") == fresh for part in compressed[0]["content"] if isinstance(part, dict))
    assert compressed[0]["_todo_snapshot_synthetic"] is True


def test_no_stale_snapshot_keeps_release_tail_and_zero_human_behavior() -> None:
    """Without stale state, only a trailing real user absorbs the snapshot."""
    from agent.conversation_compression import _inject_todo_snapshot_without_reordering_human_intent

    fresh = f"{TODO_INJECTION_HEADER}\n- [>] current task"
    trailing_human = [{"role": "assistant", "content": "answer"}, {"role": "user", "content": "new request"}]
    assistant_tail = [{"role": "user", "content": "older request"}, {"role": "assistant", "content": "answer"}]
    zero_human = [{"role": "assistant", "content": "summary"}]

    _inject_todo_snapshot_without_reordering_human_intent(trailing_human, fresh)
    _inject_todo_snapshot_without_reordering_human_intent(assistant_tail, fresh)
    _inject_todo_snapshot_without_reordering_human_intent(zero_human, fresh)

    assert fresh in trailing_human[-1]["content"]
    for messages in (assistant_tail, zero_human):
        assert messages[-1] == {"role": "user", "content": fresh, "_todo_snapshot_synthetic": True}

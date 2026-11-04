"""Todo snapshot authority and explicit-state folding at a compression boundary."""

import contextlib
from collections.abc import Callable
from typing import Any


def compaction_todo_snapshot(store: Any, *, fallback: Any = None, authoritative: bool = False) -> tuple[str | None, bool]:
    """Return the live rendering or an opaque fallback, plus whether it supersedes older snapshots."""
    snapshot = store.format_for_injection()
    fallback = fallback if isinstance(fallback, str) else None
    # Empty structured clears and opaque carry-forward blocks supersede old snapshots too.
    authoritative = fallback is not None or authoritative
    has_items = getattr(store, "has_items", None)
    with contextlib.suppress(Exception):
        authoritative = authoritative or (bool(has_items()) if callable(has_items) else False)
    return snapshot or fallback, authoritative


def fold_todo_snapshot(
    compressed: list, todo_snapshot: str | None, *, authoritative: bool,
    repair_sequence: Callable[[list], Any],
) -> None:
    """Fold an explicit snapshot in place, repairing exposed seams with the caller's replay policy."""
    from agent.conversation_compression import (
        _is_real_user_message, _message_text, _pruned_skill_reload_notice,
        _replace_message_content, _strip_stale_todo_snapshot, _todo_snapshot_is_only_content,
    )
    if authoritative:
        for _todo_idx in range(len(compressed) - 1, -1, -1):
            _todo_message = compressed[_todo_idx]
            if not isinstance(_todo_message, dict) or _todo_message.get("role") != "user":
                continue
            _todo_content = _todo_message.get("content")
            _todo_stripped = _strip_stale_todo_snapshot(_todo_content)
            if _todo_stripped == _todo_content:
                continue
            if _todo_message.get("_todo_snapshot_synthetic") and _todo_snapshot_is_only_content(
                _todo_content, _todo_stripped
            ):
                compressed.pop(_todo_idx)
                if _todo_idx < len(compressed):
                    # A standalone snapshot can drift from the tail; deleting it may expose two
                    # assistant rows, so use the normal replay repair to keep metadata consistent.
                    repair_sequence(compressed)
            else:
                _replace_message_content(_todo_message, _todo_stripped)
                # No longer todo-only scaffolding; other synthetic flags stay authoritative and
                # _is_real_user_message() recomputes provenance from content + flags.
                _todo_message.pop("_todo_snapshot_synthetic", None)
            break
    if todo_snapshot:
        # If this boundary pruned skill bodies, the policy behind the todos is gone:
        # add a reload notice after TODO_INJECTION_HEADER so both strip together.
        # Retention parity (#84718): the snapshot below re-injects the imperative verbatim. If this same
        # boundary pruned skill bodies to [SKILL_PRUNED: ...] markers, the policy that governed those tasks
        # is gone — couple a reload instruction to the snapshot so the imperative never crosses the boundary
        # alone.
        _reload_notice = _pruned_skill_reload_notice(compressed)
        if _reload_notice:
            todo_snapshot = f"{todo_snapshot}\n\n{_reload_notice}"
        # Fold the snapshot into a trailing REAL user msg (no synthetic user/user pair);
        # strip old snapshots first. Scaffolding tails must not absorb it (provenance).
        # Any snapshot merged at an earlier boundary is stripped first so repeated compactions refresh
        # rather than accumulate todo state (#26981). Scaffolding tails (continuation marker, summary
        # handoff, a bare stale snapshot row) must never absorb the snapshot: merging would upgrade them to
        # "real user" evidence and break zero-user provenance (#69292), so those keep the flagged standalone
        # append and the real-user preservation pass continues to see todo scaffolding, not human intent.
        from agent.context_compressor import _append_text_to_content
        merged = False
        _tail = compressed[-1] if compressed and isinstance(compressed[-1], dict) else None
        if _tail is not None and _tail.get("role") == "user":
            _stripped = _strip_stale_todo_snapshot(_tail.get("content"))
            _probe = {key: value for key, value in _tail.items() if key != "content"}
            _probe["content"] = _stripped
            if _is_real_user_message(_probe):
                _snapshot_text = f"\n\n{todo_snapshot}" if isinstance(_stripped, str) and _stripped else todo_snapshot
                _replace_message_content(_tail, _append_text_to_content(_stripped, _snapshot_text))
                merged = True
            elif (
                _stripped != _tail.get("content") and not _message_text({"role": "user", "content": _stripped}).strip()
            ):
                # The tail was nothing but an earlier snapshot row —
                # refresh it in place instead of stacking a duplicate.
                _replace_message_content(_tail, todo_snapshot)
                _tail["_todo_snapshot_synthetic"] = True
                merged = True
        if not merged:
            compressed.append({"role": "user", "content": todo_snapshot, "_todo_snapshot_synthetic": True})

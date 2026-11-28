"""The pruned-skill reload notice: emission at the compression boundary and stale-notice strip.

Split out of :mod:`agent.conversation_compression` (FILE_LINES ratchet): the notice pair is a
self-contained handoff concern. The origin module keeps ``_fold_todo_snapshot`` and calls these
through the sibling; origin symbols are read through ``_cc()`` at call time so tests that patch
``agent.conversation_compression.X`` stay effective.
"""

from __future__ import annotations

from typing import Any

from agent.conversation_compression_scaffolding import _message_text


def _cc():
    """The origin module, resolved lazily: avoids the import cycle and keeps tests that patch
    ``agent.conversation_compression.X`` effective (attributes are read at call time)."""
    from agent import conversation_compression
    return conversation_compression


_PRUNED_SKILL_RELOAD_NOTICE_HEADER = "[Skills pruned during compression — reload before acting on these tasks]"


def _strip_stale_reload_notice(agent: Any, compressed: list) -> None:
    """Drop the standalone pruned-skill reload notice block from a previous boundary (in place).
    Content-targeted and unconditional (todo-strip parity): the replay user-merge pass welds a
    flagged notice into a real-user tail and keeps prev's keys, so neither the synthetic flag nor
    real-user exclusion can gate detection — strip by content; the cut preserves the human prefix
    (#26981 parity, sys-e18z77). Rows carrying TODO_INJECTION_HEADER are owned by the todo strip:
    cutting the coupled notice there would regress the #84718 coupling when the store is not
    authoritative."""
    from tools.todo_tool import TODO_INJECTION_HEADER
    for i in range(len(compressed) - 1, -1, -1):
        msg = compressed[i]
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        _text = _message_text(msg)
        if TODO_INJECTION_HEADER in _text:
            continue
        if not (msg.get("_pruned_skill_notice_synthetic") or _PRUNED_SKILL_RELOAD_NOTICE_HEADER in _text):
            continue
        content = msg.get("content")

        def _cut(text: str) -> str:
            idx = text.find(_PRUNED_SKILL_RELOAD_NOTICE_HEADER)
            return text[:idx].rstrip() if idx != -1 else text

        if isinstance(content, str):
            remainder = _cut(content)
        elif isinstance(content, list):
            remainder = [
                {**part, "text": _cut(str(part.get("text") or ""))} if isinstance(part, dict) and part.get("type") == "text" else part
                for part in content
            ]
            remainder = [p for p in remainder if isinstance(p, str) or (isinstance(p, dict) and (p.get("text") or p.get("type") != "text"))]
        else:
            remainder = content
        if _cc()._todo_snapshot_is_only_content(content, remainder):
            compressed.pop(i)
            if i < len(compressed):
                # Deleting can expose two assistant rows; use the normal replay repair (#26981 parity).
                agent._repair_message_sequence(compressed)
        else:
            _cc()._replace_message_content(msg, remainder)
            msg.pop("_pruned_skill_notice_synthetic", None)
        break


def _pruned_skill_reload_notice(compressed: list, *, with_todo: bool = True) -> str:
    """Reload notice for skills whose bodies were pruned, or ``""``.
    Scans ``[SKILL_PRUNED: ...]`` markers in the post-compression transcript; first-seen order, deduplicated,
    capped at ``_MAX_PRUNED_SKILL_MARKERS``. ``with_todo=False`` drops the task-list framing, which is false
    when no todo snapshot crosses the boundary (sys-e18z77)."""
    from agent.context_compressor_prune import _MAX_PRUNED_SKILL_MARKERS, _extract_pruned_skill_names
    names: list = []
    for message in compressed:
        if not isinstance(message, dict):
            continue
        for name in _extract_pruned_skill_names(_message_text(message)):
            if name not in names:
                names.append(name)
    del names[_MAX_PRUNED_SKILL_MARKERS:]
    if not names:
        return ""
    calls = "; ".join(f"skill_view(name='{name}')" for name in names)
    if with_todo:
        framing = (
            "The task list above crossed the compression boundary verbatim, but "
            "the skill instructions that governed it were pruned. Before "
            "executing any preserved task that depends on these skills, reload "
            f"them first: {calls}. After reloading, re-check that each pending "
            "task is still justified — findings recorded before the boundary may have invalidated it."
        )
    else:
        framing = (
            "Your conversation crossed the compression boundary, but the skill "
            "instructions that governed the work in progress were pruned. Before "
            "continuing work that depends on these skills, reload "
            f"them first: {calls}. After reloading, re-check that the work in "
            "progress is still justified — findings recorded before the boundary may have invalidated it."
        )
    return f"{_PRUNED_SKILL_RELOAD_NOTICE_HEADER}\n{framing}"

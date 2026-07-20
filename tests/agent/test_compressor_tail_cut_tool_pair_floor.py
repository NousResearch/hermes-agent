"""Regression coverage: the minimum-progress floor must not split a tool group.

``_find_tail_cut_by_tokens`` aligns ``cut_idx`` away from tool-call/result
boundaries (``_align_boundary_backward``, #1976) and both tail anchors
re-align after moving it. The final statement then raised the result to
``head_end + 1`` to guarantee compression always claims at least one message
— otherwise the caller's ``compress_start >= compress_end`` guard turns the
pass into a no-op that re-runs forever.

That raise discarded the alignment. When the floor landed *inside* a tool
group, the parent ``assistant(tool_calls)`` fell in the summarised region
while its ``tool`` results started the tail; ``_sanitize_tool_pairs`` then
dropped those orphans outright, so the tool output was neither summarised nor
kept — it vanished. That is precisely the silent loss
``_align_boundary_backward``'s docstring says the alignment exists to prevent.

The floor now re-aligns FORWARD (never backward, which would hand back the
message the floor just claimed), so a raised cut skips to the end of the
group and the call/result pair is summarised together.
"""

from __future__ import annotations

import itertools

import pytest


@pytest.fixture()
def compressor():
    from agent.context_compressor import ContextCompressor

    return ContextCompressor(model="main-model", quiet_mode=True)


def _tool_group(call_id: str, results: int = 1, payload: str = "r" * 60):
    msgs = [{
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": "f", "arguments": "{}"},
        }],
    }]
    for _ in range(results):
        msgs.append({"role": "tool", "tool_call_id": call_id, "content": payload})
    return msgs


def _cut(compressor, messages):
    head_end = compressor._protect_head_size(messages)
    start = compressor._align_boundary_forward(messages, head_end)
    return start, compressor._find_tail_cut_by_tokens(messages, start)


def _pairing_violations(messages, start, end):
    """Indices kept (head + tail) must never reference a summarised partner."""
    n = len(messages)
    kept = set(range(0, min(start, n))) | set(range(min(end, n), n))
    parent = {}
    for i, m in enumerate(messages):
        for tc in m.get("tool_calls") or []:
            parent[tc["id"]] = i

    problems = []
    for i in sorted(kept):
        msg = messages[i]
        for tc in msg.get("tool_calls") or []:
            for j, other in enumerate(messages):
                if (
                    other.get("role") == "tool"
                    and other.get("tool_call_id") == tc["id"]
                    and j not in kept
                ):
                    problems.append(
                        f"assistant(tool_calls) kept at {i}, result {j} summarised"
                    )
        if msg.get("role") == "tool":
            p = parent.get(msg.get("tool_call_id"))
            if p is not None and p not in kept:
                problems.append(f"tool kept at {i}, parent assistant {p} summarised")
    return problems


class TestFloorDoesNotSplitToolGroups:
    def test_back_to_back_tool_calls_keep_their_results(self, compressor):
        """Two consecutive tool calls — the ordinary agent shape.

        The floor used to land on the second group's ``tool`` result, leaving
        it orphaned in the tail while its parent was summarised away.
        """
        messages = [{"role": "system", "content": "sys"}]
        messages += _tool_group("call_1", results=2)
        messages += _tool_group("call_2", results=1, payload="IMPORTANT RESULT")

        start, end = _cut(compressor, messages)

        assert not _pairing_violations(messages, start, end)
        # The orphan is gone because the whole group moved into the summary.
        summarised = messages[start:end]
        assert any(
            m.get("role") == "tool" and "IMPORTANT RESULT" in str(m.get("content"))
            for m in summarised
        ), "the tool result must be summarised with its parent, not dropped"

    def test_tool_result_is_not_silently_dropped_by_sanitize(self, compressor):
        """End-to-end: the surviving transcript keeps a coherent pairing."""
        messages = [{"role": "system", "content": "sys"}]
        messages += _tool_group("call_1", results=2)
        messages += _tool_group("call_2", results=1, payload="IMPORTANT RESULT")

        start, end = _cut(compressor, messages)
        survivors = (
            messages[:start]
            + [{"role": "user", "content": "[CONTEXT COMPACTION — REFERENCE ONLY] ..."}]
            + messages[end:]
        )
        cleaned = compressor._sanitize_tool_pairs(survivors)

        assert len(cleaned) == len(survivors), (
            "_sanitize_tool_pairs had to strip an orphan — the cut split a group"
        )

    def test_floor_lands_on_tool_result_after_protected_head(self, compressor):
        """The floor path itself: head_end + 1 points straight at a result."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u" * 60},
            {"role": "user", "content": "u" * 60},
            {"role": "user", "content": "u" * 60},
        ]
        messages += _tool_group("call_1", results=1)

        start, end = _cut(compressor, messages)

        assert not _pairing_violations(messages, start, end)
        assert messages[end - 1].get("role") != "assistant" or not messages[
            end - 1
        ].get("tool_calls"), "cut must not sit between a tool_call and its result"

    def test_cut_still_makes_progress(self, compressor):
        """The floor's purpose survives: compression always claims a message."""
        messages = [{"role": "system", "content": "sys"}]
        messages += _tool_group("call_1", results=1)
        messages += _tool_group("call_2", results=1)

        start, end = _cut(compressor, messages)

        assert end > start, "compression must not become a no-op"


class TestToolPairingInvariantAcrossShapes:
    def test_no_well_formed_transcript_is_split_mid_group(self, compressor):
        """Property sweep over every well-formed block layout up to length 6.

        Blocks: user, assistant-text, tool group with one result, tool group
        with two results. On the unfixed floor ~26% of these split a pair.
        """
        blocks = ["U", "A", "T1", "T2"]
        offenders = []
        exercised = 0

        for n in range(2, 7):
            for layout in itertools.product(blocks, repeat=n):
                messages = [{"role": "system", "content": "sys"}]
                for i, b in enumerate(layout):
                    if b == "U":
                        messages.append({"role": "user", "content": "u" * 60})
                    elif b == "A":
                        messages.append({"role": "assistant", "content": "a" * 60})
                    else:
                        messages += _tool_group(
                            f"call_{i}", results=1 if b == "T1" else 2
                        )

                start, end = _cut(compressor, messages)
                if end <= start:
                    continue
                exercised += 1
                if _pairing_violations(messages, start, end):
                    offenders.append("-".join(layout))

        assert exercised > 1000, "sweep did not exercise a meaningful sample"
        assert not offenders, (
            f"{len(offenders)} layouts split a tool group, e.g. {offenders[:5]}"
        )

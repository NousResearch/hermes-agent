"""Regression tests for #32114 / P-0057 (compression side).

A skill_view body demoted by a *proactive* prune (no compaction boundary) leaves
only a ``[SKILL_PRUNED:]`` marker in the transcript while the repeat-view dedup
cache still serves "unchanged" stubs — the reload deadlock. The demotion must
mark the skill as ghosted in ``tools.skills_tool_dedup`` so the next skill_view
call self-heals with a full disk re-read.
"""

from unittest.mock import patch

from agent.context_compressor import (
    SKILL_PRUNED_MARKER_PREFIX,
    ContextCompressor,
    _skill_pruned_marker,
)
from tools.skills_tool_dedup import (
    _is_ghosted_skill_view,
    reset_skill_view_dedup,
)


def _make_compressor(**overrides):
    kwargs = dict(
        model="test/model",
        quiet_mode=True,
        protect_first_n=1,
        protect_last_n=2,
    )
    kwargs.update(overrides)
    with patch(
        "agent.context_compressor.get_model_context_length", return_value=100000
    ):
        return ContextCompressor(**kwargs)


def _filler(n, start=0):
    out = []
    for i in range(n):
        role = "user" if (start + i) % 2 == 0 else "assistant"
        out.append({"role": role, "content": f"filler {start + i} " + "y" * 400})
    return out


def _skill_view_pair(call_id, skill_name, size=6000):
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "skill_view",
                    "arguments": f'{{"name":"{skill_name}"}}',
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": f"# {skill_name} instructions\n" + "x" * size,
        },
    ]


class TestProactivePruneMarksGhosts:
    def test_old_skill_view_demotion_marks_ghost(self):
        c = _make_compressor()
        # Skill sits well behind the tail window -> Phase-1 demotes it.
        msgs = (
            _skill_view_pair("call_s", "stale-skill")
            + _filler(20)
        )
        reset_skill_view_dedup()
        result, pruned = c._prune_old_tool_results(msgs, protect_tail_count=4)
        assert pruned >= 1
        skill_row = result[1]
        assert _skill_pruned_marker("stale-skill") in skill_row["content"]
        # The transcript now holds only the marker; the dedup layer must know.
        assert _is_ghosted_skill_view("stale-skill") is True

    def test_pressure_pass_demotion_marks_ghost(self):
        c = _make_compressor()
        msgs = (
            _filler(2)
            + _skill_view_pair("call_s", "fresh-skill", size=60000)
            + [{"role": "user", "content": "active ask"}]
        )
        reset_skill_view_dedup()
        result, pruned = c._prune_old_tool_results(
            msgs, protect_tail_count=4, protect_tail_tokens=100
        )
        assert pruned >= 1
        assert SKILL_PRUNED_MARKER_PREFIX in result[3]["content"]
        assert _is_ghosted_skill_view("fresh-skill") is True

    def test_non_skill_demotion_does_not_mark_ghost(self):
        c = _make_compressor()
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_r",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"x.py"}'},
                }],
            },
            {"role": "tool", "tool_call_id": "call_r", "content": "z" * 6000},
        ] + _filler(20)
        reset_skill_view_dedup()
        _, pruned = c._prune_old_tool_results(msgs, protect_tail_count=4)
        assert pruned >= 1
        assert _is_ghosted_skill_view("x.py") is False


class TestLeanTailMarksGhosts:
    def test_stale_skill_view_row_demoted_and_ghosted(self):
        c = _make_compressor()
        c.tail_mode = "lean"
        # 8 tool rounds; the keep-window is 6, so the earliest round (the
        # skill_view pair) is stale and gets demoted to a recovery stub.
        msgs = []
        for r in range(8):
            if r == 0:
                msgs += _skill_view_pair("call_s", "lean-skill")
            else:
                msgs += [
                    {"role": "user", "content": f"round {r} " + "y" * 400},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": f"call_{r}",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"x.py"}'},
                        }],
                    },
                    {"role": "tool", "tool_call_id": f"call_{r}", "content": "z" * 2000},
                ]
        reset_skill_view_dedup()
        out = c._demote_stale_tail_tools(msgs, tail_start=0)
        demoted_row = out[1]
        # Result rows don't carry tool_name, so the stub is generically named;
        # the demotion itself is what must have happened here.
        assert "output demoted at compaction" in demoted_row["content"]
        assert "6,026 chars" in demoted_row["content"]
        assert _is_ghosted_skill_view("lean-skill") is True

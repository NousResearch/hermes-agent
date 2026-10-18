"""Superseded-path eviction — pass 1.5 of ``_prune_old_tool_results``.

Measured evidence (design doc P2): 48.1% of (session, path) pairs are touched more
than once — 829 redundant ops, worst cases a file read 83x / 54x / 23x in one
session. Every re-touch leaves another near-copy in the transcript while the OLDER
copy is the stale one, which is also the mechanism behind editing from a stale
version.

The pass is deterministic (no LLM call) and moves by back-reference like pass 1
(byte-identical dedupe): the newest result for a touched path stays verbatim, older
ones point at it. It is tail-agnostic, spares the unread pending round (#61932), and
leaves any result whose path it cannot extract alone. These tests pin the behaviour,
not the exact wording of the marker.

Cases: a path touched twice, a path touched once, two paths that must not cross-talk,
newest-wins when touches interleave and the newest body is the shortest, and results
whose path cannot be extracted.
"""

from __future__ import annotations

import json
from typing import Any

from agent.context_compressor import ContextCompressor, _SUPERSEDED_TOOL_RESULT_PREFIX


def _compressor() -> ContextCompressor:
    c = ContextCompressor.__new__(ContextCompressor)
    c.quiet_mode = True
    # The pending-round spare measures the unread round against the real input window.
    c.context_length = 200_000
    return c


def _round(call_id: str, tool: str, args, content: str) -> list[dict]:
    """One assistant(tool_calls) row plus its tool result. ``args`` may be a raw string
    (to model corrupt / unparseable argument JSON) or a dict."""
    arguments = args if isinstance(args, str) else json.dumps(args)
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool, "arguments": arguments},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": content},
    ]


def _body(tag: str, repeats: int = 60) -> str:
    return f"{tag}\n" + (tag + " ") * repeats


def _tool_rows(messages: list[dict]) -> dict[str, Any]:
    return {m["tool_call_id"]: m.get("content") for m in messages if m.get("role") == "tool"}


class TestSupersededPathEviction:
    def test_older_result_for_a_retouched_path_becomes_a_back_reference(self):
        old_body, new_body = _body("OLD"), _body("NEW")
        msgs = [{"role": "user", "content": "fix target.py"}]
        msgs += _round("c1", "read_file", {"path": "src/target.py"}, old_body)
        msgs += [{"role": "assistant", "content": "stale plan"}]
        msgs += _round("c2", "read_file", {"path": "src/target.py"}, new_body)
        msgs.append({"role": "user", "content": "apply the fix"})

        out, pruned = _compressor()._prune_old_tool_results(msgs, protect_tail_count=len(msgs))

        assert pruned == 1
        # Content is rewritten in place: no row is added, dropped, or re-ordered (alternation).
        assert len(out) == len(msgs)
        assert [m["role"] for m in out] == [m["role"] for m in msgs]
        older, newest = out[2], out[5]
        assert newest["content"] == new_body
        assert older["content"].startswith(_SUPERSEDED_TOOL_RESULT_PREFIX)
        # The back-reference names the path so the model knows what to re-read.
        assert "src/target.py" in older["content"]
        assert old_body not in older["content"]

    def test_a_path_touched_once_is_left_verbatim(self):
        body = _body("SOLO")
        msgs = [{"role": "user", "content": "read it"}]
        msgs += _round("c1", "read_file", {"path": "src/solo.py"}, body)
        msgs.append({"role": "assistant", "content": "done"})

        out, pruned = _compressor()._prune_old_tool_results(msgs, protect_tail_count=len(msgs))

        assert pruned == 0
        assert out[2]["content"] == body

    def test_two_paths_are_keyed_separately(self):
        a_old, a_new, b_body = _body("AOLD"), _body("ANEW"), _body("BBODY")
        msgs = [{"role": "user", "content": "two files"}]
        msgs += _round("c1", "read_file", {"path": "src/a.py"}, a_old)
        msgs += _round("c2", "read_file", {"path": "src/b.py"}, b_body)
        msgs += _round("c3", "read_file", {"path": "src/a.py"}, a_new)
        msgs.append({"role": "assistant", "content": "ok"})

        out, pruned = _compressor()._prune_old_tool_results(msgs, protect_tail_count=len(msgs))

        assert pruned == 1
        rows = _tool_rows(out)
        assert rows["c1"].startswith(_SUPERSEDED_TOOL_RESULT_PREFIX)
        assert "src/a.py" in rows["c1"]
        assert "src/b.py" not in rows["c1"]
        assert rows["c2"] == b_body  # b.py was touched once: never a candidate
        assert rows["c3"] == a_new

    def test_newest_touch_wins_when_touches_interleave(self):
        """The survivor is the highest-index touch per path — not the first, and not the
        longest: the newest bodies here are the shortest ones."""
        short = {"A3": _body("A3", repeats=2), "B2": _body("B2", repeats=2)}
        msgs = [{"role": "user", "content": "walk the tree"}]
        for call_id, path, tag in [
            ("c1", "src/a.py", "A1"),
            ("c2", "src/b.py", "B1"),
            ("c3", "src/a.py", "A2"),
            ("c4", "src/b.py", "B2"),
            ("c5", "src/a.py", "A3"),
        ]:
            msgs += _round(call_id, "read_file", {"path": path}, short.get(tag, _body(tag)))
        msgs.append({"role": "assistant", "content": "found it"})

        out, pruned = _compressor()._prune_old_tool_results(msgs, protect_tail_count=len(msgs))

        assert pruned == 3  # a.py: 2 older copies, b.py: 1
        rows = _tool_rows(out)
        assert rows["c1"].startswith(_SUPERSEDED_TOOL_RESULT_PREFIX)
        assert "src/a.py" in rows["c1"]
        assert rows["c3"].startswith(_SUPERSEDED_TOOL_RESULT_PREFIX)
        assert rows["c2"].startswith(_SUPERSEDED_TOOL_RESULT_PREFIX)
        assert "src/b.py" in rows["c2"]
        assert rows["c4"] == short["B2"]
        assert rows["c5"] == short["A3"]

    def test_tools_without_an_extractable_path_are_left_alone(self):
        """No path to key on -> no eviction, no crash: a missing key, corrupt argument JSON,
        a non-string path, a tool this pass does not key on, and an orphan tool result."""
        bodies = {
            "c1": _body("MISSING"),
            "c2": _body("CORRUPT"),
            "c3": _body("NONSTRING"),
            "c4": _body("TERMINAL"),
            "orphan": _body("ORPHAN"),
        }
        msgs = [{"role": "user", "content": "mixed batch"}]
        msgs += _round("c1", "read_file", {"offset": 5}, bodies["c1"])
        msgs += _round("c2", "patch", "{not json", bodies["c2"])
        msgs += _round("c3", "read_file", {"path": 7}, bodies["c3"])
        msgs += _round("c4", "terminal", {"command": "ls -la src"}, bodies["c4"])
        msgs += [{"role": "tool", "tool_call_id": "orphan", "content": bodies["orphan"]}]
        msgs.append({"role": "assistant", "content": "ok"})

        out, pruned = _compressor()._prune_old_tool_results(msgs, protect_tail_count=len(msgs))

        assert pruned == 0
        rows = _tool_rows(out)
        for call_id, body in bodies.items():
            assert rows[call_id] == body

    def test_a_second_run_does_not_re_supersede(self):
        msgs = [{"role": "user", "content": "fix target.py"}]
        msgs += _round("c1", "read_file", {"path": "src/target.py"}, _body("OLD"))
        msgs += _round("c2", "read_file", {"path": "src/target.py"}, _body("NEW"))
        msgs.append({"role": "assistant", "content": "ok"})

        c = _compressor()
        out, pruned = c._prune_old_tool_results(msgs, protect_tail_count=len(msgs))
        assert pruned == 1
        again, pruned_again = c._prune_old_tool_results(out, protect_tail_count=len(out))
        assert pruned_again == 0
        assert again == out

    def test_a_result_shorter_than_the_reference_is_left_alone(self):
        """The pass only moves bytes out: replacing a body shorter than the back-reference
        itself would grow the request."""
        stub = "tiny body"
        msgs = [{"role": "user", "content": "look twice"}]
        msgs += _round("c1", "read_file", {"path": "src/tiny.py"}, stub)
        msgs.append({"role": "assistant", "content": "again"})
        msgs += _round("c2", "read_file", {"path": "src/tiny.py"}, _body("FULL"))

        out, pruned = _compressor()._prune_old_tool_results(msgs, protect_tail_count=len(msgs))

        assert pruned == 0
        rows = _tool_rows(out)
        assert rows["c1"] == stub
        assert rows["c2"] == _body("FULL")

    def test_unread_pending_round_is_spared(self):
        """Two calls to the same path in one batch: the batch's older copy is output the
        model asked for and has not read yet (#61932) — only pre-batch copies go."""
        third, fourth = _body("THIRD"), _body("FOURTH")
        msgs: list[dict[str, Any]] = [{"role": "user", "content": "compare target.py twice"}]
        msgs += _round("c1", "read_file", {"path": "src/t.py"}, _body("FIRST"))
        msgs += _round("c2", "read_file", {"path": "src/t.py"}, _body("SECOND"))
        msgs.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "read_file", "arguments": json.dumps({"path": "src/t.py"})},
                    }
                    for call_id in ("c3", "c4")
                ],
            }
        )
        msgs.append({"role": "tool", "tool_call_id": "c3", "content": third})
        msgs.append({"role": "tool", "tool_call_id": "c4", "content": fourth})

        out, pruned = _compressor()._prune_old_tool_results(msgs, protect_tail_count=len(msgs))

        assert pruned == 2
        rows = _tool_rows(out)
        assert rows["c1"].startswith(_SUPERSEDED_TOOL_RESULT_PREFIX)
        assert rows["c2"].startswith(_SUPERSEDED_TOOL_RESULT_PREFIX)
        assert rows["c3"] == third  # spared: the unread pending batch
        assert rows["c4"] == fourth  # the newest touch survives

    def test_the_pass_makes_no_llm_call(self, monkeypatch):
        def _boom(*_args, **_kwargs):
            raise AssertionError("superseded-path eviction must stay deterministic (no LLM call)")

        monkeypatch.setattr("agent.context_compressor.call_llm", _boom)
        msgs = [{"role": "user", "content": "twice"}]
        msgs += _round("c1", "read_file", {"path": "src/x.py"}, _body("X1"))
        msgs += _round("c2", "read_file", {"path": "src/x.py"}, _body("X2"))
        msgs.append({"role": "assistant", "content": "ok"})

        out, pruned = _compressor()._prune_old_tool_results(msgs, protect_tail_count=len(msgs))

        assert pruned == 1
        assert _tool_rows(out)["c2"] == _body("X2")

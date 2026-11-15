"""Config-driven ``prune_exempt_tools`` tests (#149956).

Some tool results are instructions, not data: once compaction demotes them
to a one-line summary the model keeps acting as though the instructions are
still in context. ``context.prune_exempt_tools`` lets a deployment name the
tools whose results must survive demotion verbatim, across both the ordinary
prune pass and the pressure pass.

The tests assert three things:

- a named tool keeps its full body where an unnamed tool is demoted;
- the config parser normalizes the list (case, whitespace, non-strings) and
  degrades to an empty set rather than raising on a malformed value;
- an empty exemption list preserves the pre-change behaviour exactly.
"""

from unittest.mock import patch

from agent.context_compressor import ContextCompressor

BIG = 9000

def _make_compressor(**overrides):
    kwargs = dict(
        model="test/model",
        quiet_mode=True,
        protect_first_n=1,
        protect_last_n=1,
    )
    kwargs.update(overrides)
    with patch(
        "agent.context_compressor.get_model_context_length", return_value=100000
    ):
        return ContextCompressor(**kwargs)

def _tool_pair(call_id, tool_name, size=BIG):
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": "{}"},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": f"# {tool_name} body\n" + "x" * size,
        },
    ]

def _transcript():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    msgs += _tool_pair("c1", "fetch_docs")
    msgs += _tool_pair("c2", "shell")
    msgs += [{"role": "user", "content": "tail"}]
    return msgs

def _demoted(body):
    """A demoted result is a short one-line stand-in, not the original body."""
    return len(body) < 200 and "chars" in body

def _body(messages, call_id):
    for m in messages:
        if m.get("role") == "tool" and m.get("tool_call_id") == call_id:
            return m.get("content", "")
    raise AssertionError(f"no tool result for {call_id}")

class TestExemptToolSurvivesPruning:
    def test_named_tool_is_spared_and_unnamed_tool_is_demoted(self):
        c = _make_compressor(prune_exempt_tools={"fetch_docs"})
        out, pruned = c._prune_old_tool_results(
            _transcript(), protect_tail_count=1,
            prune_exempt_tools=c.prune_exempt_tools,
        )
        assert pruned >= 1
        assert len(_body(out, "c1")) > BIG
        assert _demoted(_body(out, "c2"))

    def test_no_exemptions_demotes_both(self):
        c = _make_compressor()
        out, pruned = c._prune_old_tool_results(
            _transcript(), protect_tail_count=1,
            prune_exempt_tools=c.prune_exempt_tools,
        )
        assert pruned >= 2
        assert _demoted(_body(out, "c1"))
        assert _demoted(_body(out, "c2"))

class TestExemptToolNormalization:
    def test_case_and_whitespace_are_normalized(self):
        c = _make_compressor(prune_exempt_tools=["  Fetch_Docs ", "SHELL"])
        assert c.prune_exempt_tools == {"fetch_docs", "shell"}

    def test_non_strings_and_blanks_are_dropped(self):
        c = _make_compressor(prune_exempt_tools=["fetch_docs", "", "   ", 7, None])
        assert c.prune_exempt_tools == {"fetch_docs"}

    def test_malformed_value_degrades_to_empty_set(self):
        c = _make_compressor(prune_exempt_tools="fetch_docs")
        assert c.prune_exempt_tools == set()

    def test_absent_value_is_an_empty_set(self):
        assert _make_compressor().prune_exempt_tools == set()

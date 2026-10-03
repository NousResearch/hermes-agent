"""Tests for _repair_tool_call_arguments — malformed JSON repair pipeline."""

import json

import pytest

from agent.message_sanitization import _repair_tool_call_arguments, _salvage_truncated_tool_args
class TestRepairToolCallArguments:
    """Verify each repair stage in the pipeline."""

    # -- Stage 1: empty / whitespace-only --

    def test_empty_string_returns_empty_object(self):
        assert _repair_tool_call_arguments("", "t") == "{}"



    # -- Stage 2: Python None literal --



    # -- Stage 3: trailing comma repair --


    def test_trailing_comma_in_array(self):
        result = _repair_tool_call_arguments('{"a": [1, 2,]}', "t")
        parsed = json.loads(result)
        assert parsed == {"a": [1, 2]}


    # -- Stage 4: unclosed brackets --



    # -- Stage 5: excess closing delimiters --



    # -- Stage 6: last resort --


    def test_unrepairable_partial_returns_empty_object(self):
        # Truncated in the middle of a string key — bracket closing won't help
        assert _repair_tool_call_arguments('{"truncated": "val', "t") == "{}"

    def test_unrepairable_garbage_returns_empty_object(self):
        # No JSON structure to reconstruct: brackets/closing quotes cannot help.
        assert _repair_tool_call_arguments("garbage no json", "t") == "{}"

    def test_braces_inside_string_values_do_not_skew_the_balance(self):
        # A "}" inside a value must not be counted as closing the object: naive counting
        # sees 2 "}"-worth of closes for 1 "{" and drops a repairable call to "{}".
        result = _repair_tool_call_arguments('{"code": "}", "x": 1', "t")
        assert json.loads(result) == {"code": "}", "x": 1}

    def test_truncated_nested_array_closes_in_stack_order(self):
        # {"items": [{"n": 1}, {"n": 2 needs "}]} appended (stack order), not "}}" —
        # count-based appending grouped all braces before all brackets and never parsed.
        result = _repair_tool_call_arguments('{"items": [{"n": 1}, {"n": 2', "t")
        assert json.loads(result) == {"items": [{"n": 1}, {"n": 2}]}

    # -- Balanced but misnested: the "]" of an array of objects dropped, a "}" closing in
    # its place (#115061, deepseek-v4-flash via a portal). Counts balance, so nothing can be
    # appended; the missing closer has to be inserted BEFORE the misplaced one. --

    @pytest.mark.parametrize("raw, expected", [
        ('{"a": [{"b": 1}, {"c": 2}}]}', {"a": [{"b": 1}, {"c": 2}]}),
        ('{"edits": [{"path": "a.py", "mode": "w"}, {"path": "b.py", "mode": "w"}}',
         {"edits": [{"path": "a.py", "mode": "w"}, {"path": "b.py", "mode": "w"}]}),
        ('{"tool": "edit", "args": {"items": [{"k": 1}, {"k": 2}}}}',
         {"tool": "edit", "args": {"items": [{"k": 1}, {"k": 2}]}}),
        ('{"calls": [{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {"y": 2}}}',
         {"calls": [{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {"y": 2}}]}),
        ('{"a": [1, 2}', {"a": [1, 2]}),
    ])
    def test_misnested_closer_is_inserted_before_the_misplaced_one(self, raw, expected):
        assert json.loads(_repair_tool_call_arguments(raw, "t")) == expected

    # -- Valid JSON passthrough (this path is via except, but still works) --


    # -- Combined repairs --



    # -- Stage 0: strict=False (literal control chars in strings) --
    # llama.cpp backends sometimes emit literal tabs/newlines inside JSON
    # string values. strict=False accepts these; we re-serialise to the
    # canonical wire form (#12068).




    # -- Stage 4: control-char escape fallback --

class TestSalvageTruncatedToolArgs:
    """Send-path prefix salvage for tool-call args that died mid-stream.

    Contract (behavior, not snapshot): the result is parseable JSON when not
    None; it keeps exactly the complete members the model actually streamed;
    and it never invents content — a cut mid-string keeps the string's prefix,
    never a completion of it.
    """

    def test_mid_string_cut_keeps_the_streamed_prefix(self):
        raw = '{"path": "/tmp/a.py", "content": "def main():\n    print('
        salvaged = json.loads(_salvage_truncated_tool_args(raw))
        assert salvaged["path"] == "/tmp/a.py"
        assert salvaged["content"] == "def main():\n    print("
        # nothing after the cut was invented
        assert salvaged["content"].endswith("print(")

    def test_long_real_world_content_survives(self):
        # The incident shape: an 8KB write_file cut ~halfway through content.
        real = open("README.md", encoding="utf-8").read()[:8000]
        full = '{"content": ' + json.dumps(real) + ', "path": "/tmp/x.py"}'
        raw = full[: len(full) // 2]
        salvaged = json.loads(_salvage_truncated_tool_args(raw))
        # the streamed half is at least a 4000-char prefix of `real`, and it
        # is exactly a prefix (no completion, no reordering)
        assert len(salvaged["content"]) >= len(raw) // 4
        assert real.startswith(salvaged["content"])
        assert "path" not in salvaged  # it had not streamed yet

    def test_raw_control_chars_inside_dangling_string_are_escaped(self):
        # local-model shape: literal newlines/tabs in the unclosed tail
        raw = '{"content": "line1\tline2\nline3'
        salvaged = json.loads(_salvage_truncated_tool_args(raw))
        assert salvaged["content"] == "line1\tline2\nline3"

    def test_dangling_backslash_does_not_swallow_the_closing_quote(self):
        # A lone trailing backslash in a DROPPED stream is an escape whose
        # escaped char never arrived. Dropping it (not keeping a literal
        # backslash) is the conservative, parseable choice — and it must not
        # escape the closing quote we add.
        raw = '{"content": "ends with backslash\\'
        salvaged = json.loads(_salvage_truncated_tool_args(raw))
        assert salvaged["content"] == "ends with backslash"

    def test_cut_inside_a_key_keeps_complete_members_before_it(self):
        raw = '{"content": "abc", "pa'  # second key truncated
        salvaged = json.loads(_salvage_truncated_tool_args(raw))
        assert salvaged == {"content": "abc"}

    def test_nested_array_truncation_is_not_safely_salvageable(self):
        # The cut is inside a nested container's member, after a comma: closing it
        # would invent a value the model never streamed (", "mo" -> trailing member).
        # Salvage must REFUSE here (None) so the existing safe {} / retry path
        # handles it — never ship a half-invented nested object.
        raw = '{"edits": [{"path": "a", "mode": "w"}, {"path": "b", "mo'
        assert _salvage_truncated_tool_args(raw) is None

    def test_unsalvageable_inputs_return_none(self):
        assert _salvage_truncated_tool_args('{"conte') is None   # no member boundary
        assert _salvage_truncated_tool_args('garbage') is None   # not an object
        assert _salvage_truncated_tool_args('[1, 2,') is None    # array
        assert _salvage_truncated_tool_args('') is None
        assert _salvage_truncated_tool_args('{"a":') is None     # key complete, no value
        assert _salvage_truncated_tool_args('  {"a": "b"}  ') == '{"a": "b"}'

    def test_delimiters_inside_strings_do_not_count_as_boundaries(self):
        raw = '{"content": "x, y, z", "path"'  # commas inside content, cut in key
        salvaged = json.loads(_salvage_truncated_tool_args(raw))
        assert salvaged == {"content": "x, y, z"}

    def test_salvage_is_stable_across_repeated_sends(self):
        """The send copy converges: canonicalizing the salvaged JSON again is
        a no-op (it is already valid), so repeated sends cannot drift."""
        raw = '{"content": "aaa\nbbb\nccc", "path": "/t"'
        once = _salvage_truncated_tool_args(raw)
        assert once is not None
        canonical = json.dumps(json.loads(once), separators=(",", ":"), sort_keys=True)
        assert json.loads(canonical) == json.loads(once)
        # and a re-salvage of the (now valid) JSON keeps the same object
        assert json.loads(_salvage_truncated_tool_args(once)) == json.loads(once)



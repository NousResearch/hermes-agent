"""Regression tests for issue #6841 — repeated tool-name and duplicated
JSON-argument repair.

The agent loop in ``run_agent.py`` receives malformed tool-call payloads
from certain model/provider combinations. Two corruption modes appear:

1. **Repeated tool names** — the tool name is duplicated back-to-back, e.g.
   ``skills_listskills_list``. The current ``_repair_tool_call`` pipeline
   rejects these as "Unknown tool" because the repeated form is not in
   ``valid_tool_names``.

2. **Duplicated adjacent JSON arguments** — multiple identical JSON object
   payloads are concatenated without a separator, e.g.
   ``{"category":"research"}{"category":"research"}``. ``json.loads`` raises
   ``Extra data`` and the call is rejected.

Both are real, observed patterns (per the issue body) and break the
generic tool-call path on otherwise-valid models.

These tests guard the conservative repair pipeline:
- Repeated tool names are collapsed only when the result is a valid tool.
- Mismatched duplicated JSON payloads are still rejected (no silent merge).
- Single, non-duplicated inputs are untouched.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from run_agent import AIAgent


VALID = {
    "skills_list",
    "session_search",
    "skill_view",
    "web_search",
    "execute_code",
    "read_file",
    "write_file",
    "terminal",
    "patch",
    "browser_click",
    "todo",
}


@pytest.fixture
def repair():
    """Bind ``_repair_tool_call`` to a stub agent with our VALID set."""
    stub = SimpleNamespace(valid_tool_names=VALID)
    return AIAgent._repair_tool_call.__get__(stub, AIAgent)


class TestRepeatedToolNameRepair:
    """L1 from LAYERS.md — repeated tool names collapse to a valid tool."""

    @pytest.mark.parametrize(
        "repeated,expected",
        [
            ("skills_listskills_list", "skills_list"),
            ("session_searchsession_search", "session_search"),
            ("skill_viewskill_view", "skill_view"),
            ("web_searchweb_search", "web_search"),
            ("execute_codeexecute_code", "execute_code"),
        ],
    )
    def test_collapse_double_repeated_name(self, repair, repeated, expected):
        """Issue body example: exact double-repeat of a valid tool name."""
        assert repair(repeated) == expected

    def test_collapse_triple_repeated_name(self, repair):
        """Issue body example: three back-to-back identical JSON
        payloads. Tool-name analog: ``foobar`` repeated three times
        cannot be collapsed (the result is ``foo``, not in VALID). But
        ``write_file`` repeated 1.5x is not valid either; the conservative
        pipeline must NOT guess. Verify that ``foo`` is the only safe
        output for an odd-length string and the fuzzy match fallback
        runs. Since ``foo`` is not in VALID, expect ``None``."""
        # 'foofoofoo' has len 9, not cleanly divisible; pipeline should
        # not silently invent a tool. Falls through to fuzzy match —
        # 'foofoofoo' has zero close matches at 0.7 cutoff.
        assert repair("foofoofoo") is None

    def test_collapse_repeated_does_not_match_unrelated_tool(self, repair):
        """Half of ``abcdefabcdef`` is ``abcdef`` — not in VALID. Pipeline
        must not claim a fuzzy match for an obviously unrelated string."""
        assert repair("abcdefabcdef") is None

    def test_collapse_preserves_existing_lowercase_repair(self, repair):
        """If the input has case issues AND a repeat, lowercase first,
        then collapse. ``SKILLS_LISTSKILLS_LIST`` -> ``skills_list``."""
        assert repair("SKILLS_LISTSKILLS_LIST") == "skills_list"

    def test_repeat_does_not_break_existing_dash_to_underscore(self, repair):
        """Existing dash-to-underscore path must keep working alongside
        the new collapse step."""
        assert repair("write-file") == "write_file"


class TestDuplicatedJsonArgsRepair:
    """L2 from LAYERS.md — duplicated adjacent JSON payloads collapse."""

    def test_exact_double_payload_collapses(self):
        from run_agent import _repair_tool_call_arguments
        result = _repair_tool_call_arguments(
            '{"category":"research"}{"category":"research"}', "t"
        )
        assert json.loads(result) == {"category": "research"}

    def test_exact_triple_payload_collapses(self):
        from run_agent import _repair_tool_call_arguments
        result = _repair_tool_call_arguments(
            '{"a":1}{"a":1}{"a":1}', "t"
        )
        assert json.loads(result) == {"a": 1}

    def test_nested_payload_double_collapses(self):
        """Real-world case from issue body: nested object with
        string-typed fields and an empty string value, repeated three
        times back-to-back."""
        from run_agent import _repair_tool_call_arguments
        raw = (
            '{"name":"reddit-saas-idea-miner","file_path":""}'
            '{"name":"reddit-saas-idea-miner","file_path":""}'
            '{"name":"reddit-saas-idea-miner","file_path":""}'
        )
        result = _repair_tool_call_arguments(raw, "t")
        assert json.loads(result) == {
            "name": "reddit-saas-idea-miner",
            "file_path": "",
        }

    def test_mismatched_payloads_not_collapsed(self):
        """Different keys/values back-to-back must not be silently
        merged. The conservative pipeline's job here is to *not
        invent* a combined object. The actual returned value may be:

        - the first object (if the existing repair stages leave it
          alone), or
        - ``{}`` (the last-resort fallback), or
        - any other single-object form the existing stages produce.

        What it MUST NOT do is silently invent ``{"a": 1, "b": 2}``
        or any other merged shape — that would corrupt the call."""
        from run_agent import _repair_tool_call_arguments
        result = _repair_tool_call_arguments('{"a":1}{"b":2}', "t")
        parsed = json.loads(result)
        assert parsed != {"a": 1, "b": 2}, (
            f"Mismatched payloads were silently merged: {parsed!r}"
        )
        # The result must always be a valid JSON object (the existing
        # contract — every repair result is a string that round-trips
        # through json.loads).
        assert isinstance(parsed, dict)

    def test_semantically_equal_but_not_byte_identical_payloads_not_collapsed(self):
        """Edge E2: equality of parsed dicts is not enough. The issue
        asks for byte-identical repeated payloads, so differently serialized
        objects must fall through to the existing safe-repair path."""
        from agent.message_sanitization import _collapse_repeated_json_objects

        raw = '{"a":1,"b":2}{"b":2,"a":1}'
        assert _collapse_repeated_json_objects(raw) is None

    def test_single_object_unchanged(self):
        """Single, valid JSON object must not be touched."""
        from run_agent import _repair_tool_call_arguments
        result = _repair_tool_call_arguments('{"a": 1}', "t")
        assert json.loads(result) == {"a": 1}

    def test_empty_string_still_returns_empty_object(self):
        """Edge E1: pre-existing contract must hold."""
        from run_agent import _repair_tool_call_arguments
        assert _repair_tool_call_arguments("", "t") == "{}"

    def test_none_type_still_returns_empty_object(self):
        """Edge E1: pre-existing contract must hold."""
        from run_agent import _repair_tool_call_arguments
        assert _repair_tool_call_arguments(None, "t") == "{}"

    def test_non_object_payloads_not_collapsed(self):
        """Number/array/bool payloads must not be subject to the
        same collapse. ``[1,2][1,2]`` is a corrupted array; the
        conservative pipeline should leave it for the existing
        trailing-comma / unclosed-bracket repair stages to handle
        (or fail safely) — but it MUST NOT invent a merged array."""
        from run_agent import _repair_tool_call_arguments
        result = _repair_tool_call_arguments("[1,2][1,2]", "t")
        # Must parse to a valid JSON value of some kind.
        parsed = json.loads(result)
        # Acceptable: first array [1,2], or any repaired form. Must
        # not equal a silently-merged [1,2,1,2].
        assert parsed != [1, 2, 1, 2], (
            f"Array payloads were silently concatenated: {parsed!r}"
        )


class TestRepairProducesValidJson:
    """Every repair result MUST round-trip through ``json.loads`` —
    either before (no fix) or after (fix applied)."""

    @pytest.mark.parametrize(
        "raw",
        [
            '{"a":1}{"a":1}',
            '{"a":1}{"a":1}{"a":1}',
            '{"name":"x","file_path":""}{"name":"x","file_path":""}',
            '{"x":1}{"y":2}',  # mismatched — must not silently merge
        ],
    )
    def test_repair_output_is_valid_json(self, raw):
        from run_agent import _repair_tool_call_arguments
        result = _repair_tool_call_arguments(raw, "t")
        # The result must always be parseable JSON.
        json.loads(result)

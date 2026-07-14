"""Tests for AIAgent._repair_tool_call — tool-name normalization.

Regression guard for #14784: Claude-style models sometimes emit
class-like tool-call names (``TodoTool_tool``, ``Patch_tool``,
``BrowserClick_tool``, ``PatchTool``). Before the fix they returned
"Unknown tool" even though the target tool was registered under a
snake_case name. The repair routine now normalizes CamelCase,
strips trailing ``_tool`` / ``-tool`` / ``tool`` suffixes (up to
twice to handle double-tacked suffixes like ``TodoTool_tool``), and
falls back to fuzzy match.

BUG-8 regression guard: the namespace-prefix guard prevents a shared
``mcp__knowledge__kb_`` prefix from inflating the SequenceMatcher ratio
enough to allow silent read-to-write repairs (e.g. ``kb_search`` ->
``kb_add``).  The fix is a conservative acceptance gate: the fuzzy
candidate is accepted ONLY when its operation-suffix similarity ratio
(computed after stripping the shared namespace prefix) is >= 0.7.  This
blocks cross-operation remaps while allowing legitimate operation-level
typos.  For same-namespace candidate pools the op-suffix and whole-name
orderings coincide, so n=5 does NOT select a different winner than n=1
would — the value is entirely in the gate that rejects diverging ops.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.mcp_tool import mcp_prefixed_tool_name


VALID = {
    "todo",
    "patch",
    "browser_click",
    "browser_navigate",
    "web_search",
    "read_file",
    "write_file",
    "terminal",
    "execute_code",
    "session_search",
}

# Extended VALID set used by TestNamespacePrefixGuard.  Uses the
# production double-underscore MCP naming (``mcp__<server>__<tool>``)
# produced by ``mcp_prefixed_tool_name`` so all guard tests exercise the
# real registration format rather than synthetic single-underscore names.
# Includes sibling operations so both the blocking and allow paths are
# covered.
_MCP_NS_SEARCH = mcp_prefixed_tool_name("knowledge", "kb_search")   # mcp__knowledge__kb_search
_MCP_NS_GET = mcp_prefixed_tool_name("knowledge", "kb_get")          # mcp__knowledge__kb_get
_MCP_NS_ADD = mcp_prefixed_tool_name("knowledge", "kb_add")          # mcp__knowledge__kb_add

VALID_NS = VALID | {_MCP_NS_SEARCH, _MCP_NS_GET, _MCP_NS_ADD}

# Production double-underscore MCP tool names built via the same helper
# Hermes uses at registration time (``mcp__<server>__<tool>``).  These
# cover the format produced by ``mcp_prefixed_tool_name`` in
# ``tools/mcp_tool.py``.
_MCP_KB_SEARCH = mcp_prefixed_tool_name("knowledge", "kb_search")  # mcp__knowledge__kb_search
_MCP_KB_ADD = mcp_prefixed_tool_name("knowledge", "kb_add")        # mcp__knowledge__kb_add
_MCP_KB_GET = mcp_prefixed_tool_name("knowledge", "kb_get")        # mcp__knowledge__kb_get

# Valid tool set containing the production double-underscore MCP tools alongside
# the existing non-namespaced tools so both code paths are exercised.
VALID_MCP_DD = VALID | {_MCP_KB_SEARCH, _MCP_KB_ADD, _MCP_KB_GET}


@pytest.fixture
def repair():
    """Return a bound _repair_tool_call built on a minimal shell agent.

    We avoid constructing a real AIAgent (which pulls in credential
    resolution, session DB, etc.) because the repair routine only
    reads self.valid_tool_names. A SimpleNamespace stub is enough to
    bind the unbound function.
    """
    from run_agent import AIAgent
    stub = SimpleNamespace(valid_tool_names=VALID)
    return AIAgent._repair_tool_call.__get__(stub, AIAgent)


class TestExistingBehaviorStillWorks:
    """Pre-existing repairs must keep working (no regressions)."""

    def test_lowercase_already_matches(self, repair):
        assert repair("browser_click") == "browser_click"

    def test_uppercase_simple(self, repair):
        assert repair("TERMINAL") == "terminal"

    def test_dash_to_underscore(self, repair):
        assert repair("web-search") == "web_search"

    def test_space_to_underscore(self, repair):
        assert repair("write file") == "write_file"

    def test_fuzzy_near_miss(self, repair):
        # One-character typo — fuzzy match at 0.7 cutoff
        assert repair("terminall") == "terminal"

    def test_unknown_returns_none(self, repair):
        assert repair("xyz_no_such_tool") is None


class TestClassLikeEmissions:
    """Regression coverage for #14784 — CamelCase + _tool suffix variants."""

    def test_camel_case_no_suffix(self, repair):
        assert repair("BrowserClick") == "browser_click"

    def test_camel_case_with_underscore_tool_suffix(self, repair):
        assert repair("BrowserClick_tool") == "browser_click"

    def test_camel_case_with_Tool_class_suffix(self, repair):
        assert repair("PatchTool") == "patch"

    def test_double_tacked_class_and_snake_suffix(self, repair):
        # Hardest case from the report: TodoTool_tool — strip both
        # '_tool' (trailing) and 'Tool' (CamelCase embedded) to reach 'todo'.
        assert repair("TodoTool_tool") == "todo"

    def test_simple_name_with_tool_suffix(self, repair):
        assert repair("Patch_tool") == "patch"

    def test_simple_name_with_dash_tool_suffix(self, repair):
        assert repair("patch-tool") == "patch"

    def test_camel_case_preserves_multi_word_match(self, repair):
        assert repair("ReadFile_tool") == "read_file"
        assert repair("WriteFileTool") == "write_file"

    def test_mixed_separators_and_suffix(self, repair):
        assert repair("write-file_Tool") == "write_file"


class TestEdgeCases:
    """Edge inputs that must not crash or produce surprising results."""

    def test_empty_string(self, repair):
        assert repair("") is None

    def test_only_tool_suffix(self, repair):
        # '_tool' by itself is not a valid tool name — must not match
        # anything plausible.
        assert repair("_tool") is None

    def test_none_passed_as_name(self, repair):
        # Defensive: real callers always pass str, but guard against
        # a bug upstream that sends None.
        assert repair(None) is None

    def test_very_long_name_does_not_match_by_accident(self, repair):
        # Fuzzy match should not claim a tool for something obviously unrelated.
        assert repair("ThisIsNotRemotelyARealToolName_tool") is None


class TestVolcEngineXmlPollution:
    """Regression coverage for #33007 — VolcEngine ``api/plan`` endpoint
    leaks raw XML attribute fragments into ``tool_use.name``.

    Observed in production with the ``anthropic_messages`` API mode:

        terminal" parameter="command" string="true
        execute_code" parameter="code" string="true
        session_search" parameter="session_id" string="true

    The fix trims at the first ``"``/``'``/``<``/``>`` so the rest of
    the repair pipeline can resolve the cleaned name to a real tool.
    """

    def test_terminal_with_xml_attribute_pollution(self, repair):
        # Exact pattern from the bug report (terminal call).
        polluted = 'terminal" parameter="command" string="true'
        assert repair(polluted) == "terminal"

    def test_execute_code_with_xml_attribute_pollution(self, repair):
        polluted = 'execute_code" parameter="code" string="true'
        assert repair(polluted) == "execute_code"

    def test_session_search_with_xml_attribute_pollution(self, repair):
        polluted = 'session_search" parameter="session_id" string="true'
        assert repair(polluted) == "session_search"

    def test_camel_case_tool_with_xml_pollution(self, repair):
        # If the polluted prefix is CamelCase / suffixed, the rest of
        # the pipeline (CamelCase -> snake_case, _tool strip) still runs.
        polluted = 'BrowserClick_tool" parameter="selector" string="true'
        assert repair(polluted) == "browser_click"

    def test_tool_name_with_trailing_quote_only(self, repair):
        # Minimal leak — just a stray trailing quote, no full attribute.
        assert repair('terminal"') == "terminal"

    def test_tool_name_with_angle_bracket_pollution(self, repair):
        # Defensive — same root cause, raw '<' bleeding through.
        assert repair("terminal<parameter=command") == "terminal"

    def test_tool_name_with_single_quote_pollution(self, repair):
        # Defensive — same root cause, single-quoted attribute style.
        assert repair("terminal' parameter='command' string='true") == "terminal"

    def test_clean_tool_name_unaffected_by_sanitizer(self, repair):
        # Pure passthrough — no XML/quote chars, no change.
        assert repair("execute_code") == "execute_code"
        assert repair("session_search") == "session_search"

    def test_space_separated_name_still_normalizes(self, repair):
        # Critical: the XML strip must NOT consume whitespace, or the
        # legitimate ``"write file" -> write_file`` repair path breaks.
        assert repair("write file") == "write_file"

    def test_pollution_with_unknown_tool_root_still_fails(self, repair):
        # Sanitizer must not mask invalid tool names by laundering them
        # through the cleaner.
        polluted = 'no_such_tool" parameter="x" string="true'
        assert repair(polluted) is None

    def test_leading_quote_falls_through_to_fuzzy_match(self, repair):
        # Sanitizer only trims when the XML char is at idx > 0 — a
        # name that *starts* with a quote is left untouched so the
        # rest of the pipeline (fuzzy match at 0.7 cutoff) can still
        # recover the obvious target.
        assert repair('"terminal"') == "terminal"


@pytest.fixture
def repair_ns():
    """Bound _repair_tool_call using VALID_NS — the extended tool set.

    Why: The base ``repair`` fixture uses VALID which lacks the
    ``mcp__knowledge__kb_*`` sibling operations needed to exercise both
    the blocking and allow paths of the namespace-prefix guard.
    What: Returns a bound method identical in structure to ``repair`` but
    backed by VALID_NS, which contains production double-underscore MCP
    names alongside the base non-namespaced tools.
    Test: Instantiate and call with a known-blocked pair; assert None is
    returned to confirm the fixture is wired correctly.
    """
    from run_agent import AIAgent
    stub = SimpleNamespace(valid_tool_names=VALID_NS)
    return AIAgent._repair_tool_call.__get__(stub, AIAgent)


class TestNamespacePrefixGuard:
    """BUG-8 regression: namespace-prefix guard prevents read->write repairs.

    All tool names here use the production double-underscore format
    (``mcp__<server>__<tool>``) via ``mcp_prefixed_tool_name``.

    The acceptance gate strips the shared leading ``_``-segment prefix from
    both the emitted name and any fuzzy-match candidate, then requires the
    operation suffixes to score >= 0.7.  This blocks ``mcp__knowledge__kb_search``
    from silently repairing to ``mcp__knowledge__kb_add`` just because the
    shared ``mcp__knowledge__`` prefix pushes the whole-name SequenceMatcher
    ratio above the 0.7 cutoff.

    For same-namespace candidate pools, whole-name and op-suffix orderings
    coincide.  The guard's value is entirely in the acceptance gate that
    blocks diverging operations.
    """

    def test_direct_match_fast_path(self, repair_ns):
        # Exact name in VALID_NS — fast-path returns before fuzzy even runs.
        # Confirms the guard does not break the trivial case.
        assert repair_ns(_MCP_NS_SEARCH) == _MCP_NS_SEARCH

    def test_op_typo_repaired_to_search(self, repair_ns):
        # Pathological transposition ``kb_saerch`` in the op portion.
        # Op suffixes: ``kb_saerch`` vs ``kb_search`` — ratio ~0.88 >= 0.7.
        # The gate must allow this repair.
        emitted = mcp_prefixed_tool_name("knowledge", "kb_saerch")
        assert repair_ns(emitted) == _MCP_NS_SEARCH

    def test_direct_match_get_fast_path(self, repair_ns):
        # ``mcp__knowledge__kb_get`` is in VALID_NS — fast-path, no fuzzy.
        # Confirms it does not accidentally map to kb_add or kb_search.
        assert repair_ns(_MCP_NS_GET) == _MCP_NS_GET

    def test_guard_blocks_cross_op_fuzzy_match(self, repair_ns):
        # ``mcp__knowledge__kb_aed``: op suffix ``kb_aed`` vs ``kb_add``
        # ratio is ``aed`` vs ``add`` ~0.67 < 0.7, so the gate blocks it.
        # Without the gate, the shared prefix would push the whole-name
        # ratio above 0.7 and fuzzy would silently return kb_add.
        # This test FAILS against the old unconditional ``return matches[0]``
        # because that code returns the closest whole-name match (kb_add)
        # rather than None — making this the authoritative behavior-changing
        # regression test.
        emitted = mcp_prefixed_tool_name("knowledge", "kb_aed")
        result = repair_ns(emitted)
        assert result is None

    def test_legitimate_typo_same_op_allowed(self, repair_ns):
        # One-character truncation ``kb_searc`` of ``kb_search``.
        # Op suffixes: ``kb_searc`` vs ``kb_search`` — ratio ~0.93 >= 0.7.
        # The gate must allow this repair.
        emitted = mcp_prefixed_tool_name("knowledge", "kb_searc")
        assert repair_ns(emitted) == _MCP_NS_SEARCH

    def test_non_namespaced_typo_still_works(self, repair_ns):
        # ``terminall`` has no shared namespace prefix with any candidate.
        # ``_repair_op_suffix`` returns the full name unchanged, so the
        # gate degrades to a plain whole-name ratio check — identical to
        # old fuzzy behaviour.
        assert repair_ns("terminall") == "terminal"

    def test_mcp_namespaced_op_typo_allowed(self, repair_ns):
        # Transposition ``kb_serach`` in the op portion of the full MCP name.
        # Shared prefix: ``mcp__knowledge__``; op suffix: ``kb_serach``
        # vs ``kb_search`` — ratio ~0.91 >= 0.7, so the gate allows it.
        emitted = mcp_prefixed_tool_name("knowledge", "kb_serach")
        assert repair_ns(emitted) == _MCP_NS_SEARCH

    def test_mcp_namespaced_cross_op_blocked(self, repair_ns):
        # ``mcp__knowledge__kb_aet``: op suffix ``kb_aet`` vs ``kb_add``
        # is ``aet`` vs ``add`` ~0.67 < 0.7, so the gate blocks it.
        # Like test_guard_blocks_cross_op_fuzzy_match, this FAILS against
        # the old unconditional ``return matches[0]`` code.
        emitted = mcp_prefixed_tool_name("knowledge", "kb_aet")
        result = repair_ns(emitted)
        assert result is None


@pytest.fixture
def repair_mcp_dd():
    """Bound _repair_tool_call backed by VALID_MCP_DD (double-underscore MCP tools).

    Why: The existing ``repair_ns`` fixture uses the single-underscore form
    ``mcp_knowledge_kb_search``.  Hermes registers native MCP tools as
    ``mcp__<server>__<tool>`` (double-underscore, via ``mcp_prefixed_tool_name``
    in ``tools/mcp_tool.py``).  This fixture exercises that production format.
    What: Returns a bound method identical to ``repair`` but backed by
    VALID_MCP_DD, which includes ``mcp__knowledge__kb_search``,
    ``mcp__knowledge__kb_add``, and ``mcp__knowledge__kb_get``.
    Test: Call with a known-blocked pair and assert None is returned.
    """
    from run_agent import AIAgent
    stub = SimpleNamespace(valid_tool_names=VALID_MCP_DD)
    return AIAgent._repair_tool_call.__get__(stub, AIAgent)


class TestMcpDoubleUnderscoreNaming:
    """BUG-8 + naming-format coverage for Hermes-native MCP tool names.

    Hermes registers MCP tools as ``mcp__<server>__<tool>`` (double-underscore
    delimiter, produced by ``mcp_prefixed_tool_name``).  The ``_repair_op_suffix``
    helper splits on ``"_"`` and compares segment-by-segment, so the empty
    strings produced by ``__`` (e.g. ``"mcp__svc__op".split("_")`` yields
    ``["mcp", "", "svc", "", "op"]``) participate in the shared-prefix walk as
    equal segments.  These tests verify that:

    1. The double-underscore prefix is stripped correctly so only the operation
       token (e.g. ``"kb_search"``) is compared.
    2. A typo in the operation is allowed (ALLOW case).
    3. A diverging operation is blocked even when the shared prefix is long (BLOCK
       case — this is the behavior-changing test that FAILS against old code).
    4. The correct operation is selected from the candidate list with the op-suffix
       gate applied (selection case — note: same-namespace orderings coincide for
       whole-name and op-suffix; see test docstring for details).
    5. Non-namespaced tools in the same valid set are unaffected.
    """

    def test_double_underscore_op_typo_allowed(self, repair_mcp_dd):
        """ALLOW: typo of ``kb_search`` repairs to ``mcp__knowledge__kb_search``.

        Why: Demonstrates that the double-underscore prefix is stripped
        correctly and the operation suffix ``kb_serch`` is compared to
        ``kb_search`` (ratio ~0.91 >= 0.7), allowing the repair.
        What: Both ``mcp__knowledge__kb_search`` and ``mcp__knowledge__kb_add``
        are valid; the emitted name's op suffix must unambiguously resolve to
        the search operation.
        Test: Assert repair returns ``mcp__knowledge__kb_search``.
        """
        emitted = mcp_prefixed_tool_name("knowledge", "kb_serch")
        assert repair_mcp_dd(emitted) == _MCP_KB_SEARCH

    def test_double_underscore_cross_op_blocked(self, repair_mcp_dd):
        """BLOCK: diverging op suffix is rejected by the acceptance gate.

        Why: Without the op-suffix acceptance gate (i.e. the old code that
        unconditionally returned ``matches[0]``), the shared ``mcp__knowledge__``
        prefix would push ``mcp__knowledge__kb_aet`` close enough to
        ``mcp__knowledge__kb_add`` in whole-name similarity that fuzzy match
        returns ``kb_add`` — silently remapping a read operation to a write.
        With the gate, op suffix ``kb_aet`` vs ``kb_add`` scores ``aet``/``add``
        ~0.67 < 0.7 and the repair is blocked (returns None).

        This is the BEHAVIOR-CHANGING regression test: it FAILS against the old
        unconditional ``return matches[0]`` code (which returns ``kb_add``) and
        PASSES with the acceptance gate.  It is the authoritative guard for BUG-8.
        Test: Assert repair returns None.
        """
        emitted = mcp_prefixed_tool_name("knowledge", "kb_aet")
        result = repair_mcp_dd(emitted)
        assert result is None

    def test_double_underscore_op_typo_selects_correct_operation(self, repair_mcp_dd):
        """ALLOW: transposition typo in op suffix selects the correct operation.

        Why: Verifies that the op-suffix scoring selects ``kb_search`` (the
        correct operation) and that the acceptance gate passes it through
        (ratio ~0.91 >= 0.7).

        Context on same-namespace ordering: for tools sharing an identical long
        prefix (``mcp__knowledge__``), whole-name similarity and op-suffix
        similarity rank candidates in the SAME order — the prefix term
        contributes equally to both ratios.  This means the n=5 op-suffix
        selection does NOT produce a different winner than n=1 whole-name
        matching would for same-namespace pools.  This test verifies that the
        correct operation is chosen and accepted; it does NOT demonstrate a
        whole-name/op-suffix ranking inversion (which is mathematically
        impossible for same-prefix pairs and would require cross-namespace
        scenarios).  The behavior-changing regression test is
        ``test_double_underscore_cross_op_blocked``, which exercises the
        acceptance gate that the old code lacked entirely.

        Test: Assert repair returns ``mcp__knowledge__kb_search``.
        """
        emitted = mcp_prefixed_tool_name("knowledge", "kb_serach")
        result = repair_mcp_dd(emitted)
        assert result == _MCP_KB_SEARCH

    def test_double_underscore_direct_match_fast_path(self, repair_mcp_dd):
        """Direct match: exact ``mcp__knowledge__kb_search`` skips fuzzy path.

        Why: Verifies the fast-path returns before the fuzzy/ranking step.
        What: Emitting the exact registered name should return immediately.
        Test: Assert repair returns ``mcp__knowledge__kb_search``.
        """
        assert repair_mcp_dd(_MCP_KB_SEARCH) == _MCP_KB_SEARCH

    def test_double_underscore_non_namespaced_unaffected(self, repair_mcp_dd):
        """Non-namespaced tools in VALID_MCP_DD must still repair correctly.

        Why: Ensures the ranking change does not regress non-MCP tools that
        share the same valid-tool set.
        What: ``terminall`` has no shared namespace prefix with any candidate;
        the op-suffix falls back to the full name, preserving ordinary fuzzy
        behaviour.
        Test: Assert repair returns ``terminal``.
        """
        assert repair_mcp_dd("terminall") == "terminal"


class TestRepairOpSuffixHelper:
    """Unit tests for ``_repair_op_suffix`` — the module-level suffix-stripping helper.

    Why: ``_repair_op_suffix`` is extracted from the nested function so it can
    be unit-tested independently and is allocated once rather than on every
    ``repair_tool_call`` invocation.  These tests verify isolation of the
    operation token across both single- and double-underscore MCP prefixes, and
    the no-shared-prefix fallback.
    """

    def setup_method(self):
        from agent.agent_runtime_helpers import _repair_op_suffix
        self.fn = _repair_op_suffix

    def test_double_underscore_strips_namespace_and_shared_op_prefix(self):
        """Strip shared prefix from ``mcp__knowledge__kb_search`` vs ``mcp__knowledge__kb_add``.

        Why: Core case — long shared prefix including the shared ``kb`` segment
        must be fully consumed, leaving only the distinguishing token.
        What: ``"mcp__knowledge__kb_search".split("_")`` yields
        ``["mcp", "", "knowledge", "", "kb", "search"]`` and
        ``"mcp__knowledge__kb_add".split("_")`` yields
        ``["mcp", "", "knowledge", "", "kb", "add"]``.  The shared leading
        segments are ``["mcp", "", "knowledge", "", "kb"]``, so the remainder
        of ``name`` is ``["search"]`` -> ``"search"``.
        Test: Direct call; assert return value equals ``"search"``.
        """
        result = self.fn(
            mcp_prefixed_tool_name("knowledge", "kb_search"),
            mcp_prefixed_tool_name("knowledge", "kb_add"),
        )
        assert result == "search"

    def test_no_shared_prefix_returns_full_name(self):
        """No shared leading segments — return the full name unchanged.

        Why: Non-namespaced tools have no common prefix; the helper must
        fall back to the full name so the caller's ratio check uses the
        complete string (preserving old fuzzy behaviour).
        What: ``"terminal"`` and ``"read_file"`` share no ``_``-split
        segments; ``"terminal"`` is returned as-is.
        Test: Direct call; assert return value equals ``"terminal"``.
        """
        result = self.fn("terminal", "read_file")
        assert result == "terminal"

    def test_single_shared_prefix_segment(self):
        """Strips a single shared leading segment (e.g. ``kb_``).

        Why: Covers the non-MCP (no double-underscore) namespaced case where
        two tools share a short prefix like ``kb_``.
        What: ``"kb_search"`` and ``"kb_add"`` share segment ``"kb"`` (after
        splitting on ``"_"``); the helper returns ``"search"``.
        Test: Direct call; assert return value equals ``"search"``.
        """
        result = self.fn("kb_search", "kb_add")
        assert result == "search"

    def test_name_is_prefix_of_other_returns_full_name(self):
        """Fallback when ``name`` is exhausted before ``other``.

        Why: Guards against returning an empty string when ``name`` is a
        strict segment-prefix of ``other``; the fallback preserves ``name``.
        What: ``"mcp"`` (1 segment) vs ``"mcp__svc__op"`` (5 segments) —
        the walk exhausts ``a`` at i=1, leaving ``"_".join([])`` == ``""``,
        so the ``or name`` fallback kicks in.
        Test: Direct call; assert return value equals ``"mcp"``.
        """
        result = self.fn("mcp", mcp_prefixed_tool_name("svc", "op"))
        assert result == "mcp"

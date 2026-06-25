"""Tests for AIAgent._repair_tool_call — tool-name normalization.

Regression guard for #14784: Claude-style models sometimes emit
class-like tool-call names (``TodoTool_tool``, ``Patch_tool``,
``BrowserClick_tool``, ``PatchTool``). Before the fix they returned
"Unknown tool" even though the target tool was registered under a
snake_case name. The repair routine now normalizes CamelCase,
strips trailing ``_tool`` / ``-tool`` / ``tool`` suffixes (up to
twice to handle double-tacked suffixes like ``TodoTool_tool``), and
falls back to fuzzy match.

BUG-8 regression guard: the namespace-prefix guard (step 6) prevents
a shared ``kb_`` / ``mcp_knowledge_kb_`` prefix from inflating the
SequenceMatcher ratio enough to allow silent read-to-write repairs
(e.g. ``kb_search`` -> ``kb_add``).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


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

# Extended VALID set used by TestNamespacePrefixGuard.  Includes
# sibling operations under the same ``kb_`` and ``mcp_knowledge_kb_``
# prefixes so the guard's cross-op blocking and same-op allowance can
# both be exercised.
VALID_NS = VALID | {
    "kb_search",
    "kb_get",
    "kb_add",
    "kb_update",
    "mcp_knowledge_kb_search",
    "mcp_knowledge_kb_get",
    "mcp_knowledge_kb_add",
}


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


# MCP-prefixed tool names registered under the canonical ``mcp_<server>_<tool>``
# scheme. Qwen3-class tool-callers token-split and corrupt the ``mcp_`` prefix.
MCP_VALID = {
    "mcp_knowledge_kb_get",
    "mcp_knowledge_kb_search",
    "mcp_proxmox_get_nodes",
}


@pytest.fixture
def mcp_repair():
    """Bind _repair_tool_call to a stub exposing MCP-prefixed tool names."""
    from run_agent import AIAgent

    stub = SimpleNamespace(valid_tool_names=MCP_VALID)
    return AIAgent._repair_tool_call.__get__(stub, AIAgent)


class TestNormaliseMcpToolPrefix:
    """Unit coverage for the prefix-normalisation helper in isolation.

    Qwen3-235B token-split emits ``mmcp_knowledge_kb_get`` (extra leading
    'm') or ``mcp_mcp_knowledge_kb_get`` (doubled prefix). The helper must
    collapse those to a single ``mcp_`` prefix and leave everything else
    untouched.
    """

    def test_strips_spurious_leading_m(self):
        from agent.agent_runtime_helpers import normalise_mcp_tool_prefix

        assert (
            normalise_mcp_tool_prefix("mmcp_knowledge_kb_get") == "mcp_knowledge_kb_get"
        )

    def test_collapses_doubled_prefix(self):
        from agent.agent_runtime_helpers import normalise_mcp_tool_prefix

        assert (
            normalise_mcp_tool_prefix("mcp_mcp_knowledge_kb_get")
            == "mcp_knowledge_kb_get"
        )

    def test_collapses_triple_nested_prefix(self):
        from agent.agent_runtime_helpers import normalise_mcp_tool_prefix

        # Three layers must collapse all the way to a single ``mcp_``.
        assert (
            normalise_mcp_tool_prefix("mcp_mcp_mcp_knowledge_kb_get")
            == "mcp_knowledge_kb_get"
        )

    def test_collapses_quadruple_nested_prefix(self):
        from agent.agent_runtime_helpers import normalise_mcp_tool_prefix

        # Arbitrary depth normalises to a single ``mcp_`` prefix.
        assert (
            normalise_mcp_tool_prefix("mcp_mcp_mcp_mcp_knowledge_kb_get")
            == "mcp_knowledge_kb_get"
        )

    def test_strips_leading_junk_before_mcp(self):
        from agent.agent_runtime_helpers import normalise_mcp_tool_prefix

        assert (
            normalise_mcp_tool_prefix("xmcp_knowledge_kb_get") == "mcp_knowledge_kb_get"
        )
        assert (
            normalise_mcp_tool_prefix("abmcp_knowledge_kb_get")
            == "mcp_knowledge_kb_get"
        )

    def test_legit_name_passes_through_unchanged(self):
        from agent.agent_runtime_helpers import normalise_mcp_tool_prefix

        assert (
            normalise_mcp_tool_prefix("mcp_knowledge_kb_get") == "mcp_knowledge_kb_get"
        )

    def test_non_mcp_name_passes_through_unchanged(self):
        from agent.agent_runtime_helpers import normalise_mcp_tool_prefix

        # No ``mcp_`` anywhere near the front — must not be touched.
        assert normalise_mcp_tool_prefix("kb_search") == "kb_search"
        assert normalise_mcp_tool_prefix("4pkoPc7Uh") == "4pkoPc7Uh"

    def test_junk_too_far_from_front_not_normalised(self):
        from agent.agent_runtime_helpers import normalise_mcp_tool_prefix

        # ``mcp_`` more than 5 chars in is not treated as a corrupted prefix.
        assert normalise_mcp_tool_prefix("toolongprefix_mcp_x") == "toolongprefix_mcp_x"

    def test_empty_string_passes_through(self):
        from agent.agent_runtime_helpers import normalise_mcp_tool_prefix

        assert normalise_mcp_tool_prefix("") == ""


class TestMcpPrefixRepairFastPath:
    """End-to-end: corrupted MCP names repair to the canonical name via the
    direct-match fast-path; legit and truly-garbled names behave correctly.
    """

    def test_mmcp_repairs_to_canonical(self, mcp_repair):
        assert mcp_repair("mmcp_knowledge_kb_get") == "mcp_knowledge_kb_get"

    def test_doubled_prefix_repairs_to_canonical(self, mcp_repair):
        assert mcp_repair("mcp_mcp_knowledge_kb_get") == "mcp_knowledge_kb_get"

    def test_leading_junk_repairs_to_canonical(self, mcp_repair):
        assert mcp_repair("xmcp_knowledge_kb_get") == "mcp_knowledge_kb_get"

    def test_legit_mcp_name_unchanged(self, mcp_repair):
        assert mcp_repair("mcp_knowledge_kb_get") == "mcp_knowledge_kb_get"

    def test_bare_name_without_prefix_not_falsely_normalised(self, mcp_repair):
        # ``kb_search`` has no ``mcp_`` prefix — normalisation is a no-op, and
        # it must NOT be silently rewritten to mcp_knowledge_kb_search by
        # the prefix fast-path. (Fuzzy match against an mcp_-prefixed name is
        # well below the 0.7 cutoff, so this returns None.)
        assert mcp_repair("kb_search") is None

    def test_truly_garbled_token_returns_none(self, mcp_repair):
        # Random token from a token-split must still fall through, not match.
        assert mcp_repair("4pkoPc7Uh") is None


@pytest.fixture
def repair_ns():
    """Bound _repair_tool_call using VALID_NS — the extended tool set.

    Why: The base ``repair`` fixture uses VALID which lacks ``kb_*`` and
    ``mcp_knowledge_kb_*`` names.  The namespace-prefix guard tests need
    sibling operations under the same prefix to exercise both the
    blocking and the allow paths.
    What: Returns a bound method identical in structure to ``repair`` but
    backed by VALID_NS.
    Test: Instantiate and call with a known-blocked pair; assert None is
    returned to confirm the fixture is wired correctly.
    """
    from run_agent import AIAgent
    stub = SimpleNamespace(valid_tool_names=VALID_NS)
    return AIAgent._repair_tool_call.__get__(stub, AIAgent)


class TestNamespacePrefixGuard:
    """BUG-8 regression: namespace-prefix guard prevents read->write repairs.

    The guard strips the shared leading ``_``-segment prefix from both the
    emitted name and any fuzzy-match candidate, then requires the operation
    suffixes to also score >= 0.7.  This stops ``kb_search`` from silently
    repairing to ``kb_add`` just because the shared ``kb_`` prefix pushes
    the full-name SequenceMatcher ratio above the cutoff.
    """

    def test_kb_search_does_not_repair_to_kb_add(self, repair_ns):
        # ``kb_search`` is in VALID_NS — direct match should return it.
        # If emitted exactly, the fast-path returns before fuzzy even runs.
        # The guard's job is to block the fuzzy path when the op suffixes
        # diverge too much; verify it is not broken by a direct match.
        assert repair_ns("kb_search") == "kb_search"

    def test_kb_search_typo_does_not_repair_to_kb_add(self, repair_ns):
        # ``kb_serach`` is not in VALID_NS.  The fuzzy match would find
        # ``kb_search`` (ratio ~0.89) AND ``kb_add`` as candidates.
        # With n=1 the closest match is ``kb_search``; however, without
        # the guard a slightly different typo could land on ``kb_add``.
        # We test a pathological typo ``kb_saerch`` that the guard must
        # handle correctly — the op suffixes ``saerch`` vs ``search``
        # score well (>= 0.7) so the repair IS allowed.
        assert repair_ns("kb_saerch") == "kb_search"

    def test_kb_get_does_not_repair_to_kb_add(self, repair_ns):
        # ``kb_get`` is in VALID_NS — direct match, no fuzzy needed.
        # Confirm it does not accidentally map to ``kb_add``.
        assert repair_ns("kb_get") == "kb_get"

    def test_guard_blocks_cross_op_fuzzy_match(self, repair_ns):
        # Construct a name that would score above 0.7 against ``kb_add``
        # purely because of the shared ``kb_`` prefix but whose op suffix
        # diverges from ``add``.  ``kb_aed`` shares prefix ``kb_`` with
        # all kb_* tools; its op suffix ``aed`` is close to ``add``
        # (ratio ~0.67 < 0.7) so the guard should block it.
        # (If the guard is absent, fuzzy would return ``kb_add``.)
        result = repair_ns("kb_aed")
        # The guard may block the repair entirely (None) or allow a
        # sufficiently close op match.  ``aed`` vs ``add`` = 2/3 ~0.67,
        # which is below 0.7, so the guard must return None.
        assert result is None

    def test_legitimate_typo_same_op_allowed(self, repair_ns):
        # ``kb_searc`` is a one-character truncation of ``kb_search``.
        # Op suffixes: ``searc`` vs ``search`` — SequenceMatcher ~0.91.
        # The guard must allow this repair.
        assert repair_ns("kb_searc") == "kb_search"

    def test_non_namespaced_typo_still_works(self, repair_ns):
        # ``terminall`` has no shared namespace prefix with any candidate.
        # The guard is a no-op when there is no shared prefix (the op
        # suffix falls back to the full name), so the original fuzzy
        # logic should still return ``terminal``.
        assert repair_ns("terminall") == "terminal"

    def test_mcp_namespaced_typo_in_op_suffix_allowed(self, repair_ns):
        # ``mcp_knowledge_kb_serach`` is a typo in the op portion.
        # Shared prefix: ``mcp_knowledge_kb_``; op suffix: ``serach``
        # vs ``search`` — ratio ~0.91 >= 0.7, so the guard allows it.
        assert repair_ns("mcp_knowledge_kb_serach") == "mcp_knowledge_kb_search"

    def test_mcp_namespaced_cross_op_blocked(self, repair_ns):
        # ``mcp_knowledge_kb_get`` is in VALID_NS — direct match first.
        # For the guard logic, test a variant that fuzzy-matches
        # ``mcp_knowledge_kb_add`` but should be blocked: ``mcp_knowledge_kb_aet``
        # has op suffix ``aet`` vs ``add`` (ratio = 2/3 ~0.67 < 0.7).
        result = repair_ns("mcp_knowledge_kb_aet")
        assert result is None

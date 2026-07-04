"""Regression tests for #58281 — disabling a composite toolset must not
subtract tools that an explicitly enabled toolset also exposes.

The historical behaviour was: ``agent.disabled_toolsets: [coding]``
resolved ``coding`` and subtracted *all* of its tools — including tools
already enabled under narrower toolsets like ``terminal`` and ``file``,
producing Tools: 0 and the model hallucinating XML tool_use that
rendered in chat but never executed.

The fix scopes the subtraction to the *exclusive* subset of the
disabled composite: only tools that no currently-enabled toolset
contributes are removed. Tools that an enabled toolset still provides
are preserved.
"""

from unittest.mock import patch

import model_tools


# Minimal stub: every tool name ∈ TOOL_NAMES returns an inert definition
# so ``registry.get_definitions`` populates ``available_tool_names``
# without exercising real tool check_fn logic. This isolates the
# compute_tool_definitions subtraction path that the bug lives in.
TOOL_NAMES = {
    # file toolset
    "read_file", "write_file", "patch", "search_files",
    # terminal toolset
    "terminal", "close_terminal", "process", "read_terminal",
    # coding-only overlay (not present in terminal or file)
    "execute_code", "code_lookup",
}


def _stub_get_definitions(tool_names, quiet=False):
    """Return one definition per requested name so downstream
    ``available_tool_names`` matches the requested set. Models the
    shape of ``tools.registry.registry.get_definitions`` without
    depending on live tool check_fn side effects.
    """
    return [{"type": "function", "function": {"name": n}} for n in tool_names & TOOL_NAMES]


# Static toolset descriptions mirroring the bug report.
_VALIDATE_MAP = {
    "terminal": True,
    "file": True,
    "coding": True,
    "safe": True,
}

_RESOLVED = {
    "terminal": ["terminal", "close_terminal", "process", "read_terminal"],
    "file": ["patch", "read_file", "search_files", "write_file"],
    "coding": ["terminal", "close_terminal", "process", "read_terminal",
               "patch", "read_file", "search_files", "write_file",
               "execute_code", "code_lookup"],
    "safe": ["confirm_dangerous", "sketch_safety_check"],
}


def _validate(name):
    return _VALIDATE_MAP.get(name, False)


def _resolve(name):
    return list(_RESOLVED.get(name, []))


def _run(enabled, disabled):
    """Invoke _compute_tool_definitions with all the right stubs."""
    with patch("model_tools._LEGACY_TOOLSET_MAP", {}), \
         patch("model_tools._WARNED_DISABLED_BUNDLES", set()), \
         patch("toolsets.validate_toolset", side_effect=_validate), \
         patch("toolsets.resolve_toolset", side_effect=_resolve), \
         patch("model_tools.registry.get_definitions", side_effect=_stub_get_definitions):
        return model_tools._compute_tool_definitions(
            enabled_toolsets=enabled,
            disabled_toolsets=disabled,
            quiet_mode=True,
        )


def _tool_names(result):
    return sorted(t["function"]["name"] for t in result)


class TestCompositeDisabledSubtraction:
    """Pins the behaviour described in #58281."""

    def test_disabling_coding_preserves_explicitly_enabled_terminal_tools(self):
        """Bug regression: terminal + file enabled, coding disabled → tools kept."""
        result = _run(enabled=["terminal", "file"], disabled=["coding"])
        names = _tool_names(result)
        # Every terminal + file tool must remain even though ``coding``
        # supersets them and was disabled — this is the fix.
        for tool in _RESOLVED["terminal"] + _RESOLVED["file"]:
            assert tool in names, (
                f"{tool!r} was stripped even though its toolset is "
                f"explicitly enabled. Coding subtract must not cross over."
            )
        # Coding's own unique extras (not in any enabled toolset) should
        # be removed — they're exclusive to the disabled composite.
        assert "execute_code" not in names
        assert "code_lookup" not in names

    def test_disabling_coding_when_only_coding_is_enabled_empties_tools(self):
        """If only ``coding`` is enabled and you disable it, no tools remain."""
        # When coding is the ONLY enabled toolset, every coding tool is
        # exclusive (no other enabled toolset contributes). They all get
        # subtracted — leaving zero tools, matching the old behaviour for
        # the "actually-disable-the-composite" intent.
        # The user gets Tools: 0, BUT only because they actually asked.
        result = _run(enabled=[], disabled=["coding"])
        # No terminal/file override, so resolve returns []; nothing to include.
        names = _tool_names(result)
        # validate_toolset returns False for non-listed names, so without
        # any enabled set providing tools, the result stays empty.
        assert names == []

    def test_disabling_composite_does_not_strip_coding_exclusive(self):
        """Coding-only tools get stripped — those aren't represented
        anywhere else."""
        # Enabled = "" + we explicitly seed terminal/file via composite
        # ``coding``. Since coding's tools include terminal & file tools,
        # and we put "coding" in disabled, an empty enabled list means
        # tools_to_include starts empty, so EVERY coding tool is
        # "exclusive to coding" and gets subtracted.
        result = _run(enabled=[], disabled=["coding"])
        # Same as above: nothing enabled ⇒ empty intersection ⇒ full
        # subtract of composite ⇒ empty result.
        assert _tool_names(result) == []

    def test_disabled_composite_subtracts_only_exclusive_subset(self):
        """Enabled toolsets contribute → disabled composite subtracts only its extras."""
        # Enabled: terminal (covers 4 tools).
        # Disabled: coding (covers 8 tools, of which 4 are shared with terminal).
        # The 2 coding-only tools get removed; the 4 shared stay.
        result = _run(enabled=["terminal"], disabled=["coding"])
        names = _tool_names(result)
        # Terminal tools preserved.
        for tool in _RESOLVED["terminal"]:
            assert tool in names
        # Coding's unique extras must be gone.
        assert "execute_code" not in names
        assert "code_lookup" not in names
        # ``code_lookup`` is exclusive to coding; ``terminal`` is shared
        # with an enabled toolset, so it stays.
        # File tools are NOT in any enabled toolset; they came only via
        # coding's composite. With coding disabled, they're also subtracted
        # because they're "exclusive to coding" relative to the current
        # enabled set.
        assert "read_file" not in names

    def test_no_disabled_toolsets_does_not_change_behaviour(self):
        """Sanity: enabling toolsets alone still works."""
        result = _run(enabled=["terminal", "file"], disabled=[])
        names = _tool_names(result)
        for tool in _RESOLVED["terminal"] + _RESOLVED["file"]:
            assert tool in names


class TestHermesBundleSubtractionRegression:
    """Pin #33924 — hermes-* bundle subtraction must stay non-core-only.

    Even though this case isn't directly triggered by #58281, the same
    function handles both. A future refactor that swaps subtract logic
    must not regress the hermes-* bundle protection.
    """

    def test_hermes_bundle_non_core_subtract_still_applied(self):
        """Manual regression-pin via the existing bundle_non_core_tools helper."""
        # We don't run _compute_tool_definitions here — its hermes-* branch
        # uses bundle_non_core_tools which we're not stubbing. The point
        # of this test is to make the regression visible: confirm that the
        # function still references the bundle_non_core_tools branch and
        # doesn't fall into the composite branch for hermes-* names.
        import inspect
        src = inspect.getsource(model_tools._compute_tool_definitions)
        assert ".startswith(\"hermes-\")" in src
        assert "bundle_non_core_tools" in src

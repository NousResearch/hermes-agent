"""Regression test for the read_file schema cross-tool reference strip.

The static ``read_file`` schema in ``tools/file_tools.py`` ends with::

    NOTE: Cannot read images or binary files — use vision_analyze for images.

When ``vision_analyze`` is not available (vision toolset disabled, multimodal
backend not configured) the model sees this hint in its system prompt and
hallucinates calls to a tool that isn't registered. The same shape of bug
was already fixed for ``browser_navigate``'s ``web_search`` / ``web_extract``
reference in ``model_tools.get_tool_definitions``; see the "DO NOT hardcode
cross-tool references in schema descriptions" pitfall in AGENTS.md.

These tests pin the dynamic strip:

- when vision_analyze is NOT in the resolved toolset, the trailing
  "— use vision_analyze for images" is stripped from the description;
- when vision_analyze IS in the resolved toolset, the reference stays so the
  model knows the proper fallback for image files.
"""
from __future__ import annotations

import pytest

import model_tools


@pytest.fixture(autouse=True)
def _clear_cache():
    """Each test starts with an empty quiet_mode cache so toolset changes
    aren't masked by a memoized definition list from a prior test."""
    model_tools._tool_defs_cache.clear()
    yield
    model_tools._tool_defs_cache.clear()


def _find_read_file_description(tool_defs):
    for td in tool_defs:
        fn = td.get("function", {})
        if fn.get("name") == "read_file":
            return fn.get("description", "")
    return None


class TestReadFileSchemaVisionStrip:
    def test_vision_reference_stripped_when_vision_analyze_unavailable(self):
        """File toolset enabled, vision toolset NOT — strip the dangling
        cross-tool suggestion so the model doesn't try a tool it lacks."""
        tools = model_tools.get_tool_definitions(
            enabled_toolsets=["file"], quiet_mode=True
        )
        desc = _find_read_file_description(tools)
        assert desc is not None, "read_file should be present when 'file' toolset is enabled"
        assert "vision_analyze" not in desc, (
            "read_file description must not mention vision_analyze when that "
            "tool isn't available — otherwise the model hallucinates calls "
            "to a non-existent tool. See AGENTS.md 'DO NOT hardcode "
            "cross-tool references in schema descriptions'."
        )
        # The remaining trailing NOTE should still warn about images,
        # just without the dangling tool suggestion.
        assert "Cannot read images or binary files" in desc

    def test_vision_reference_kept_when_vision_analyze_available(self):
        """When the vision toolset IS enabled alongside file, the
        cross-tool suggestion is useful and should stay in the description."""
        tools = model_tools.get_tool_definitions(
            enabled_toolsets=["file", "vision"], quiet_mode=True
        )
        names = {t.get("function", {}).get("name") for t in tools}
        if "vision_analyze" not in names:
            # Vision toolset is present but the tool's check_fn rejected
            # it in this environment (e.g. no vision-capable model
            # configured under hermetic CI). This test's premise — that
            # vision_analyze actually resolves — doesn't hold, so skip
            # instead of asserting on a runtime condition we can't
            # control from a unit test.
            pytest.skip("vision_analyze not registered in this environment")
        desc = _find_read_file_description(tools)
        assert desc is not None
        assert "vision_analyze" in desc, (
            "Cross-tool suggestion should remain when the referenced tool "
            "is actually available — it's useful guidance, not noise."
        )

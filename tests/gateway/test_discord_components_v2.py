"""Components V2 planning for Discord agent replies."""

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

# tests/gateway/conftest.py installs a mock `discord` at import time and this file must not
# disturb it. Do NOT call _ensure_discord_mock() here: it mints a *fresh* MagicMock, and the
# new object breaks isinstance identity (discord.Thread, discord.ForumChannel) for every
# sibling file that captured the old one at collection.
#
# The load below is by path rather than through `plugins.platforms.discord`, whose __init__
# imports the adapter. The renderer has no import-time discord dependency; build_layout_view
# imports lazily at call time and resolves to whatever conftest installed.


def _load_components_v2():
    """Load the renderer straight from its file, bypassing the package __init__."""
    path = _ROOT / "plugins" / "platforms" / "discord" / "components_v2.py"
    spec = importlib.util.spec_from_file_location("_hermes_components_v2_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c2 = _load_components_v2()


class TestPlanSegments:
    def test_empty_content_falls_back(self):
        assert c2.plan_segments("") is None
        assert c2.plan_segments("   \n  ") is None

    def test_simple_prose_is_one_segment(self):
        segs = c2.plan_segments("hello world")
        assert segs == [("prose", "hello world")]

    def test_over_total_text_budget_falls_back(self):
        """The 4000-char limit is the SUM across the tree, and is smaller than 8x2000."""
        assert c2.plan_segments("x" * 5000) is None

    def test_just_under_budget_is_rendered(self):
        assert c2.plan_segments("x" * 3000) is not None

    def test_code_fence_kept_intact(self):
        content = "before\n\n```python\nprint(1)\nprint(2)\n```\n\nafter"
        segs = c2.plan_segments(content)
        kinds = [k for k, _ in segs]
        assert "code" in kinds
        code = [c for k, c in segs if k == "code"][0]
        assert code.startswith("```python")
        assert code.rstrip().endswith("```")
        assert "print(1)" in code and "print(2)" in code

    def test_code_block_is_never_split_across_segments(self):
        content = "```\n" + ("line\n" * 200) + "```"
        segs = c2.plan_segments(content)
        if segs is not None:
            code_segs = [c for k, c in segs if k == "code"]
            assert len(code_segs) == 1

    def test_oversized_single_code_block_falls_back(self):
        """A fence too big to split is also past the whole-tree budget, so v2 declines."""
        assert c2.plan_segments("```\n" + "x" * 4500 + "\n```") is None

    def test_prose_under_the_component_limit_stays_one_segment(self):
        """Packing into as few components as possible preserves the 40-component budget;
        visual structure comes from fenced blocks, not from splitting every paragraph."""
        content = "\n\n".join("para " + str(i) + " " + "y" * 900 for i in range(4))
        segs = c2.plan_segments(content)
        assert segs is not None
        assert len(segs) == 1
        assert len(segs[0][1]) <= c2.MAX_TEXT_DISPLAY

    def test_long_prose_without_fences_falls_back_to_chunked_text(self):
        """No prose length-splitting: past the whole-tree budget the legacy chunker wins."""
        assert c2.plan_segments("\n\n".join("z" * 900 for _ in range(10))) is None

    def test_no_segment_exceeds_single_component_limit(self):
        segs = c2.plan_segments("z" * 3900)
        assert segs is not None
        for _kind, chunk in segs:
            assert len(chunk) <= c2.MAX_TEXT_DISPLAY

    def test_component_count_budget_rejects_too_many_fences(self):
        """Many small fenced blocks blow the 40-component cap long before the text cap."""
        content = "\n\n".join("```\nx\n```" for _ in range(30))
        assert c2.plan_segments(content) is None

    def test_component_count_stays_under_budget_when_accepted(self):
        content = "\n\n".join("```\nx\n```" for _ in range(5))
        segs = c2.plan_segments(content)
        assert segs is not None
        separators = max(0, len(segs) - 1)
        assert len(segs) + separators + 1 <= c2.MAX_TOTAL_COMPONENTS

    def test_total_rendered_text_within_budget(self):
        segs = c2.plan_segments("a" * 1000 + "\n\n" + "b" * 1000)
        assert segs is not None
        assert sum(len(c) for _k, c in segs) <= c2.MAX_DISPLAYABLE_TEXT

    def test_mixed_prose_and_multiple_fences(self):
        content = "intro\n\n```js\nlet a=1;\n```\n\nmid\n\n```sh\nls -la\n```\n\nend"
        segs = c2.plan_segments(content)
        assert segs is not None
        assert len([k for k, _ in segs if k == "code"]) == 2
        assert len([k for k, _ in segs if k == "prose"]) == 3

    def test_unclosed_fence_does_not_hang_or_drop_text(self):
        content = "text\n\n```python\nprint('no close')"
        segs = c2.plan_segments(content)
        assert segs is not None
        assert any("no close" in c for _k, c in segs)

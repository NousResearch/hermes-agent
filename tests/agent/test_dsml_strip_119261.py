"""DeepSeek DSML native-markup leak fail-soft (#119261).

DeepSeek V4 Pro via OpenRouter randomly emits its native DSML tool-call
serialization as visible text (fullwidth pipes U+FF5C)::

    <｜DSML｜tool_calls> <｜DSML｜invoke name="terminal"> ... </｜DSML｜invoke> </｜DSML｜tool_calls>

Neither ``strip_think_blocks`` (shared; everything routes through it) nor
its ``cli._strip_reasoning_tags`` mirror knew these tags, so the raw markup
reached Desktop and the turn died there. Both strippers must drop DSML
blocks/orphans/cut-tails so the existing empty-response recovery retries
the turn instead of displaying garbage.
"""

from agent.agent_runtime_helpers import strip_think_blocks
from cli import _strip_reasoning_tags

# Exact shape from the issue report (fullwidth VERTICAL BAR U+FF5C).
_ISSUE_SAMPLE = (
    "<\uff5cDSML\uff5ctool_calls> "
    "<\uff5cDSML\uff5cinvoke name=\"terminal\"> "
    "<\uff5cDSML\uff5cparameter name=\"background\" string=\"false\">true</\uff5cDSML\uff5cparameter> "
    "<\uff5cDSML\uff5cparameter name=\"command\" string=\"true\">"
    "npx five-server . --port 23456 --open=false</\uff5cDSML\uff5cparameter> "
    "</\uff5cDSML\uff5cinvoke> "
    "</\uff5cDSML\uff5ctool_calls>"
)


def _both(text: str) -> tuple[str, str]:
    return _strip_reasoning_tags(text), strip_think_blocks(None, text)


class TestDsmlLeakStripped:
    def test_issue_sample_fully_stripped(self):
        """Pure-DSML content strips to nothing -> empty-recovery retries, no death."""
        for out in _both(_ISSUE_SAMPLE):
            assert "DSML" not in out
            assert out.strip() == ""

    def test_inline_block_prose_survives(self):
        for out in _both("Answer <\uff5cDSML\uff5cparameter name=\"x\">1</\uff5cDSML\uff5cparameter> tail"):
            assert "Answer" in out and out.rstrip().endswith("tail")
            assert "DSML" not in out

    def test_orphan_closers_stripped(self):
        for out in _both("done</\uff5cDSML\uff5cinvoke> more</\uff5cDSML\uff5ctool_calls>"):
            assert "DSML" not in out
            assert "done" in out and "more" in out

    def test_unterminated_opener_cut_to_prefix(self):
        """Stream cut mid-serialization (#101899 analog): drop from opener to end."""
        for out in _both("Waiting.\n<\uff5cDSML\uff5cinvoke name=\"terminal\">partial"):
            assert out.strip() == "Waiting."

    def test_ascii_pipe_variant_stripped(self):
        for out in _both('<|DSML|invoke name="terminal">x</|DSML|invoke>ok'):
            assert "DSML" not in out
            assert out.rstrip().endswith("ok")

    def test_plain_prose_untouched(self):
        text = "DSML is a markup language. Use | pipes | freely."
        assert _strip_reasoning_tags(text) == text
        assert strip_think_blocks(None, text) == text

"""DeepSeek DSML native-markup leak fail-soft (#119261, #54283).

DeepSeek emits its native DSML tool-call serialization as visible text
(fullwidth pipes U+FF5C) via OpenRouter and other OpenAI-compatible hosts, and
Bedrock's deepseek.v3.2 leaks a cut ``<｜DSML｜function_calls`` opener beside a
valid toolUse block. Both strippers (storage ``strip_think_blocks`` and the CLI
display mirror) must drop blocks, orphan closers and cut tails so the existing
empty-response recovery retries the turn instead of displaying garbage, and so
replayed history stops teaching the model to write calls as text.
"""

import pytest

from agent.agent_runtime_helpers import strip_think_blocks
from cli import _strip_reasoning_tags

P = "\uff5c"  # fullwidth VERTICAL BAR

# Exact shape from the #119261 report.
_ISSUE_SAMPLE = (
    f"<{P}DSML{P}tool_calls> <{P}DSML{P}invoke name=\"terminal\"> "
    f"<{P}DSML{P}parameter name=\"background\" string=\"false\">true</{P}DSML{P}parameter> "
    f"<{P}DSML{P}parameter name=\"command\" string=\"true\">npx five-server . --port 23456</{P}DSML{P}parameter> "
    f"</{P}DSML{P}invoke> </{P}DSML{P}tool_calls>"
)

_STRIPPERS = (_strip_reasoning_tags, lambda text: strip_think_blocks(None, text))


@pytest.mark.parametrize(
    "text, expected",
    [
        (_ISSUE_SAMPLE, ""),
        (f"Answer <{P}DSML{P}parameter name=\"x\">1</{P}DSML{P}parameter> tail", "Answer  tail"),
        (f"done</{P}DSML{P}invoke> more</{P}DSML{P}tool_calls>", "done more"),
        # Stream cut mid-serialization (#101899 analog): drop from the opener on.
        (f"Waiting.\n<{P}DSML{P}invoke name=\"terminal\">partial", "Waiting."),
        # Bedrock deepseek.v3.2: the opener itself is cut before its ``>``.
        (f"I'll check the repo.\n\n<{P}DSML{P}function_calls", "I'll check the repo."),
        ('<|DSML|invoke name="terminal">x</|DSML|invoke>ok', "ok"),
    ],
)
def test_dsml_leak_stripped_on_both_strippers(text, expected):
    # Whitespace-insensitive: the strippers trim/fuse around removed markup the
    # same way the ASCII tool-call closers already do; the invariant is that the
    # markup is gone and the prose survives in order.
    for strip in _STRIPPERS:
        assert "".join(strip(text).split()) == "".join(expected.split())


def test_plain_prose_mentioning_dsml_untouched():
    text = "DSML is a markup language. Use | pipes | freely."
    for strip in _STRIPPERS:
        assert strip(text) == text

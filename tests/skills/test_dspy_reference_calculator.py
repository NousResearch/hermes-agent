"""The DSPy reference ReAct ``calculate`` tools must not eval model-generated text.

Readers copy these snippets into real agents, so the documented tool is executed
here (without DSPy) and driven with arithmetic plus harmless non-arithmetic probes.
"""
from pathlib import Path

import pytest

REFS = Path(__file__).resolve().parents[2] / "optional-skills/mlops/research/dspy/references"


def _calculate(doc):
    text = (REFS / doc).read_text(encoding="utf-8")
    block = next(chunk for chunk in text.split("```python\n")[1:]
                 if "def calculate(" in chunk.split("\n```")[0]).split("\n```")[0]
    tools = "\n".join(
        line for line in block[:block.index("# Create ")].splitlines()
        if not line.startswith("from dspy")
    )
    namespace = {}
    exec(compile(tools, str(REFS / doc), "exec"), namespace)
    return namespace["calculate"]


def _rejected(calculate, expression):
    try:
        result = calculate(expression)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return True
    return isinstance(result, str) and result.startswith("Invalid expression")


@pytest.mark.parametrize("doc", ["examples.md", "modules.md"])
def test_calculator_still_does_arithmetic(doc):
    calculate = _calculate(doc)
    assert float(calculate("67000000 / 10")) == 6700000.0
    assert float(calculate("1905 - 1879")) == 26
    assert float(calculate("(2 + 3) * 4 - 2 ** 3")) == 12


@pytest.mark.parametrize("doc", ["examples.md", "modules.md"])
@pytest.mark.parametrize("probe", [
    "(1)." + "__class__." + "__name__",   # attribute access
    "[0] " + "* 3",                        # container construction
    "len" + "('abc')",                     # builtin call
    "10 ** " + "1000000",                  # unbounded work
])
def test_calculator_rejects_non_arithmetic(doc, probe):
    assert _rejected(_calculate(doc), probe)

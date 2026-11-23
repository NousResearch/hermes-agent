"""The DSPy reference ReAct ``calculate`` tools must not eval model-generated text.

Readers copy these snippets into real agents, so the documented tool is executed
here (without DSPy) and driven with arithmetic plus harmless non-arithmetic probes.
"""
import ast
from pathlib import Path

import pytest

REFS = Path(__file__).resolve().parents[2] / "optional-skills/mlops/research/dspy/references"


def _calculate(doc):
    text = (REFS / doc).read_text(encoding="utf-8")
    block = next(chunk for chunk in text.split("```python\n")[1:]
                 if "def calculate(" in chunk.split("\n```")[0]).split("\n```")[0]
    # Keep only the tool definitions: stdlib imports, module-level ``_CONSTANT``
    # assignments and functions. Selecting by statement type (not by a prose
    # comment) keeps the test independent of the surrounding DSPy agent code.
    tools = ast.Module(body=[
        node for node in ast.parse(block).body
        if isinstance(node, ast.FunctionDef)
        or (isinstance(node, ast.Import)
            and not any(a.name.split(".")[0] == "dspy" for a in node.names))
        or (isinstance(node, ast.Assign)
            and all(isinstance(t, ast.Name) and t.id.startswith("_") for t in node.targets))
    ], type_ignores=[])
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
    assert float(calculate(" + ".join(["1"] * 60))) == 60  # flat sums nest one level per term


@pytest.mark.parametrize("doc", ["examples.md", "modules.md"])
@pytest.mark.parametrize("probe", [
    "(1)." + "__class__." + "__name__",   # attribute access
    "[0] " + "* 3",                        # container construction
    "len" + "('abc')",                     # builtin call
    "10 ** " + "1000000",                  # unbounded work
    "1 ** " + "1000000000",                # exponent budget, checked before computing
    "10 ** 99 " + "* 10 ** 99",            # magnitude budget
    "-" * 300 + "1",                       # length budget (also bounds recursion)
])
def test_calculator_rejects_non_arithmetic(doc, probe):
    assert _rejected(_calculate(doc), probe)

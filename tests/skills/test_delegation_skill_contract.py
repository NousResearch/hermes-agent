"""Regression for #156425: copied spike tasks match the active public schema.

Read the shipped document as example data, not implementation source. This
checks default, independently delivered and joined definitions without children
or model calls; it does not prove that an agent follows the surrounding prose.
"""

import ast
from copy import deepcopy
from pathlib import Path
import re

from jsonschema import validate
import pytest

from tools import delegate_tool_config
from tools.delegate_tool import oneshot_delegate_function
from tools.registry import registry

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("independent,joined", [(False, False), (True, False), (True, True)])
def test_spike_examples_match_delivery_schema(monkeypatch, independent, joined):
    monkeypatch.setattr(
        delegate_tool_config, "_load_config", lambda: {"independent_completions": independent}
    )
    definition = registry.get_definitions({"delegate_task"})[0]["function"]
    if joined:
        definition = oneshot_delegate_function(definition)
    schema = deepcopy(definition["parameters"])
    task_schema = schema["properties"]["tasks"]["items"]
    # The runtime may tolerate unknown keys; copyable docs must not advertise them.
    task_schema["additionalProperties"] = False
    needs_group = "group" in task_schema.get("required", [])
    document = (ROOT / "skills/software-development/spike/SKILL.md").read_text(encoding="utf-8")
    examples = re.findall(r"^```[^\n]*\n(.*?)^```", document, re.MULTILINE | re.DOTALL)
    matched = []
    for example in examples:
        if "delegate_task(" not in example:
            continue
        for node in ast.walk(ast.parse(example)):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "delegate_task":
                continue
            payload = {arg.arg: ast.literal_eval(arg.value) for arg in node.keywords}
            tasks = payload["tasks"]
            if any("group" in task for task in tasks) != needs_group:
                continue
            validate(payload, schema)
            if needs_group:
                assert len({task["group"] for task in tasks}) == 1, "Compare both verdicts together"
            matched.append(tasks)
    assert matched, "No copyable example for this advertised delivery schema"

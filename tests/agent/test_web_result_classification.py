"""Successful web envelopes must not feed the repeated-failure guardrail."""
import json

import pytest

from agent.display import _detect_tool_failure
from agent.tool_guardrails import ToolCallGuardrailController, classify_tool_failure
from tools.web_tools_truncate import _trim_results


@pytest.mark.parametrize("classifier", [_detect_tool_failure, classify_tool_failure])
@pytest.mark.parametrize("tool_name,result,failed", [
    ("web_extract", {"results": _trim_results([{"content": "Example Domain"}])}, False),
    ("web_extract", {"results": _trim_results([{"content": "a" * 600}])}, False),
    ("web_extract", {"results": [{"content": "docs", "error": None}, {"error": "Timeout"}]}, True),
    ("web_extract", {"results": [{"content": "a" * 600, "error": "Timeout"}]}, True),
    ("web_search", {"success": True, "data": {"web": []}, "error": None}, False),
    ("web_search", {"success": False, "error": None}, True),
    ("web_search", {"success": False, "error": "Provider unavailable"}, True),
    ("web_extract", {"error": "Repeated query refused", "guardrail_refusal": True}, False),
])
def test_web_classifiers_use_status_values(classifier, tool_name, result, failed):
    actual, suffix = classifier(tool_name, json.dumps(result))
    assert actual is failed
    if classifier is _detect_tool_failure and result.get("error") == "Provider unavailable":
        assert "Provider unavailable" in suffix


@pytest.mark.parametrize("explicit_failure", [True, False])
def test_successful_extract_does_not_accumulate_failure_warnings(explicit_failure):
    guard = ToolCallGuardrailController()
    result = json.dumps({"results": _trim_results([{"content": "Example Domain"}])})
    for _ in range(3):
        kwargs = {"failed": _detect_tool_failure("web_extract", result)[0]} if explicit_failure else {}
        decision = guard.after_call("web_extract", {"urls": ["https://example.com"]}, result, **kwargs)
        assert decision.code not in {"repeated_exact_failure_warning", "same_tool_failure_warning"}

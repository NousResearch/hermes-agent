"""Tool-level failures must not be inferred from successful diagnostic data.

Contract authored before the classification fix. Exercise both live display and
fallback guardrail seams, structured/malformed inputs, and genuine failures.
"""

import json

import pytest

from agent.display import _detect_tool_failure, get_cute_tool_message
from agent.tool_guardrails import ToolCallGuardrailController, classify_tool_failure


CASES = [
    pytest.param({"ok": True, "error": None, "returncode": 0}, False, id="null-error-success"),
    pytest.param({"success": True, "error": "", "message": "complete"}, False, id="empty-error-with-message"),
    pytest.param({"ok": True, "error": None, "message": "complete"}, False, id="null-error-with-message"),
    pytest.param({"ok": True, "data": {"by_level": {"error": 1, "warning": 1}}, "error": None}, False, id="health-domain-errors"),
    pytest.param({"data": {"error": "an application health finding"}}, False, id="nested-domain-error"),
    pytest.param({"success": True, "data": {"failed": True}}, False, id="nested-domain-failed"),
    pytest.param({"failed": False, "error": False}, False, id="false-flags"),
    pytest.param({"error": 0, "data": "error and failed are words"}, False, id="zero-error"),
    pytest.param({"error": [], "status": []}, False, id="empty-container-error"),
    pytest.param({"error": {}, "status": {}}, False, id="empty-object-error"),
    pytest.param({"ok": False, "error": None}, True, id="explicit-ok-false"),
    pytest.param({"success": False}, True, id="explicit-success-false"),
    pytest.param({"failed": True}, True, id="explicit-failed-true"),
    pytest.param({"isError": True}, True, id="mcp-isError"),
    pytest.param({"status": "failed"}, True, id="failed-status"),
    pytest.param({"status": "error"}, True, id="error-status"),
    pytest.param({"returncode": 1, "error": None}, True, id="nonzero-returncode"),
    pytest.param({"exit_code": -1, "error": None}, True, id="nonzero-exit"),
    pytest.param({"error": "connection refused"}, True, id="real-error"),
    pytest.param({"ok": True, "error": "contradictory failure"}, True, id="contradictory-error-fails-closed"),
    pytest.param({"error": {"code": "denied"}}, True, id="structured-error"),
    pytest.param({"padding": "x" * 700, "ok": False}, True, id="failure-after-prefix-limit"),
    pytest.param({"padding": "x" * 700, "error": None, "ok": True}, False, id="long-success"),
    pytest.param({"_multimodal": True, "content": [{"type": "text", "text": '"error"'}]}, False, id="multimodal-domain-text"),
    pytest.param({"error": "BLOCKED", "guardrail_refusal": True}, False, id="harness-refusal-exempt"),
    pytest.param({"error": "BLOCKED", "guardrail_refusal": "true"}, True, id="nonboolean-refusal-not-exempt"),
]


@pytest.mark.parametrize("payload,expected", CASES)
@pytest.mark.parametrize("encoded", [False, True], ids=["dict", "json"])
@pytest.mark.parametrize("tool_name", ["g2_health", "memory"])
@pytest.mark.parametrize("classifier", [_detect_tool_failure, classify_tool_failure], ids=["display", "guardrail"])
def test_tool_level_semantics_not_payload_words(payload, expected, encoded, tool_name, classifier):
    result = json.dumps(payload) if encoded else payload
    failed, suffix = classifier(tool_name, result)
    assert failed is expected
    assert bool(suffix) is expected


@pytest.mark.parametrize("result,expected", [
    (None, False),
    ("", False),
    ("ordinary output", False),
    ('["error", "failed"]', False),
    ('"error"', False),
    ('{"error": "truncated', True),
    ("Error executing tool 'g2_health': transport failed", True),
])
@pytest.mark.parametrize("classifier", [_detect_tool_failure, classify_tool_failure])
def test_non_object_and_unparseable_results(result, expected, classifier):
    assert classifier("g2_health", result)[0] is expected


@pytest.mark.parametrize("explicit", [False, True], ids=["fallback", "live-seam"])
def test_distinct_health_collections_do_not_build_false_failure_streak(explicit):
    controller = ToolCallGuardrailController()
    for service in ("lidarr", "radarr", "sonarr", "prowlarr"):
        args = {"service": service}
        result = json.dumps({"ok": True, "data": {"service": service, "by_level": {"error": 1}}, "error": None})
        failed = _detect_tool_failure("g2_health", result)[0] if explicit else None
        decision = controller.after_call("g2_health", args, result, failed=failed)
        assert decision.action == "allow"
        line = get_cute_tool_message("g2_health", args, 0.2, result=result)
        assert "[error]" not in line
    # Genuine failures still accumulate and produce the existing warning.
    for index in range(3):
        decision = controller.after_call("g2_health", {"attempt": index}, '{"ok":false,"error":"unreachable"}')
    assert decision.code == "same_tool_failure_warning"
